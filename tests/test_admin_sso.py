"""The Tabs handoff grants admin access only to explicitly allowed accounts."""

import base64
import hashlib
import hmac
import json
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from app_factory import create_test_app

from tracking import SnapshotStore

SECRET = "shared-test-secret-" * 4


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("web.middleware._rate_windows", {})
    monkeypatch.setenv("PLANNER_ADMIN_SSO_ENABLED", "true")
    monkeypatch.setenv("PLANNER_ADMIN_SSO_USERS", "reader")
    monkeypatch.setenv("SSO_SHARED_SECRET", SECRET)
    monkeypatch.delenv("PLANNER_ADMIN_SSO_URL", raising=False)
    return create_test_app(
        SNAPSHOT_STORE=SnapshotStore(tmp_path / "analytics.sqlite3")
    ).test_client()


def handoff(payload=None):
    if payload is None:
        payload = {"purpose": "gpumodels-sso", "user": "reader", "exp": int(time.time()) + 60}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = hmac.new(SECRET.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return encoded + "." + signature


def begin(client, next_path="/admin"):
    response = client.get("/admin/login", query_string={"next": next_path})
    assert response.status_code == 302
    provider = urlsplit(response.location)
    assert (provider.scheme, provider.netloc, provider.path) == (
        "https",
        "tabs.gskrd.me",
        "/sso/gpumodels",
    )
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    return parse_qs(provider.query)["state"][0]


def complete(client, **query):
    return client.get("/auth/callback", query_string=query)


def test_planner_stays_public_and_admin_requires_shared_login(client):
    assert client.get("/").status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/admin/data.json").status_code == 401
    response = client.get("/admin/people")
    assert response.status_code == 302
    assert urlsplit(response.location).path == "/admin/login"
    assert parse_qs(urlsplit(response.location).query)["next"] == ["/admin/people"]


def test_handoff_creates_independent_session_and_preserves_destination(client):
    state = begin(client, "/admin/people?days=7")
    code = handoff()
    response = complete(client, state=state, code=code, next="https://evil.test")
    assert response.status_code == 302
    assert response.location == "/admin/people?days=7"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert client.get("/admin/data.json").status_code == 200
    assert complete(client, state=state, code=code).status_code == 401


@pytest.mark.parametrize("state", [None, "wrong-state", "état"])
def test_handoff_rejects_unbound_browser_state(client, state):
    begin(client)
    query = {"code": handoff()}
    if state is not None:
        query["state"] = state
    assert complete(client, **query).status_code == 401
    assert client.get("/admin/data.json").status_code == 401


def test_handoff_requires_login_initiation_in_same_browser(client):
    assert complete(client, state="s" * 43, code=handoff()).status_code == 401


@pytest.mark.parametrize(
    "changes",
    [
        {"purpose": "contacts-sso"},
        {"exp": 0},
        {"exp": True},
        {"exp": "future"},
        {"exp": int(time.time()) + 3600},
        {"user": []},
        {"user": ""},
    ],
)
def test_handoff_rejects_wrong_audience_expiry_or_identity(client, changes):
    state = begin(client)
    payload = {"purpose": "gpumodels-sso", "user": "reader", "exp": int(time.time()) + 60}
    payload.update(changes)
    assert complete(client, state=state, code=handoff(payload)).status_code == 401
    assert client.get("/admin/data.json").status_code == 401


@pytest.mark.parametrize("code", ["", "invalid", "é.signature", "a." + "0" * 64, "x" * 4097])
def test_malformed_codes_do_not_crash(client, code):
    state = begin(client)
    assert complete(client, state=state, code=code).status_code == 401


def test_non_object_payload_is_rejected(client):
    state = begin(client)
    assert complete(client, state=state, code=handoff([])).status_code == 401


def test_valid_signature_required(client):
    state = begin(client)
    code = handoff().split(".")[0] + "." + "0" * 64
    assert complete(client, state=state, code=code).status_code == 401


def test_valid_shared_account_needs_explicit_admin_permission(client):
    state = begin(client)
    code = handoff({"purpose": "gpumodels-sso", "user": "other", "exp": int(time.time()) + 60})
    assert complete(client, state=state, code=code).status_code == 403
    assert client.get("/admin/data.json").status_code == 401


@pytest.mark.parametrize(
    "setting,value",
    [
        ("SSO_SHARED_SECRET", ""),
        ("PLANNER_ADMIN_SSO_USERS", ""),
        ("PLANNER_ADMIN_SSO_URL", "http://tabs.gskrd.me/sso/gpumodels"),
    ],
)
def test_incomplete_sso_configuration_fails_closed(client, monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    assert client.get("/admin").status_code == 503
    assert client.get("/admin/login").status_code == 503
    assert complete(client, code=handoff()).status_code == 503


def test_sso_disables_legacy_password_and_existing_password_sessions(client, monkeypatch):
    monkeypatch.setenv("PLANNER_ADMIN_PASSWORD", "legacy")
    monkeypatch.setenv("PLANNER_ADMIN_SSO_ENABLED", "false")
    assert client.post("/admin/login", data={"password": "legacy"}).status_code == 302
    monkeypatch.setenv("PLANNER_ADMIN_SSO_ENABLED", "true")
    assert client.get("/admin/data.json").status_code == 401
    assert client.post("/admin/login", data={"password": "legacy"}).status_code == 405


@pytest.mark.parametrize(
    "next_path",
    [
        "https://evil.test",
        "//evil.test",
        "/\\evil.test",
        "/admin\r\nLocation: evil",
        "/admin-other",
    ],
)
def test_post_login_destination_stays_inside_admin(client, next_path):
    state = begin(client, next_path)
    assert complete(client, state=state, code=handoff()).location == "/admin"


def test_pending_login_expires(client):
    state = begin(client)
    with client.session_transaction() as session:
        session["planner_admin_sso"] = {"state": state, "expires": 0, "next": "/admin"}
    assert complete(client, state=state, code=handoff()).status_code == 401


def test_session_expires_and_allowlist_changes_revoke_access(client, monkeypatch):
    state = begin(client)
    assert complete(client, state=state, code=handoff()).status_code == 302
    monkeypatch.setenv("PLANNER_ADMIN_SSO_USERS", "different-reader")
    assert client.get("/admin/data.json").status_code == 401
    monkeypatch.setenv("PLANNER_ADMIN_SSO_USERS", "reader")
    with client.session_transaction() as session:
        session["planner_admin_expires"] = 0
    assert client.get("/admin/data.json").status_code == 401


def test_logout_clears_local_access_and_visits_shared_logout(client):
    state = begin(client)
    assert complete(client, state=state, code=handoff()).status_code == 302
    response = client.post("/admin/logout")
    assert response.location == "https://tabs.gskrd.me/logout"
    assert client.get("/admin/data.json").status_code == 401
