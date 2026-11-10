"""Administrative routes: visitor insight pages, the snapshot browser, login."""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections import defaultdict
from functools import wraps
from urllib.parse import urlencode, urlsplit

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from data import (
    GPUS,
    MODELS,
)
from web.analytics import build_admin_payload, mark_internal_if_admin
from web.config import (
    ADMIN_PAGE_SIZE,
    ADMIN_SESSION_KEY,
    _env_bool,
)
from web.helpers import (
    _bounded_int,
)

admin_bp = Blueprint("admin", __name__)

# (endpoint, template, nav label, page title, subtitle)
INSIGHT_PAGES = (
    (
        "admin",
        "overview",
        "Overview",
        "Visitors",
        "Who uses the calculator, and what they plan with it",
    ),
    (
        "audience",
        "audience",
        "Audience & geo",
        "Audience & geography",
        "Where people are, and how they found the calculator",
    ),
    (
        "personas",
        "personas",
        "Personas",
        "Personas",
        "Distinct kinds of people, derived from what they do",
    ),
    ("people", "people", "People", "People", "One profile per visitor, each visit read as a story"),
    (
        "journeys",
        "journeys",
        "Journeys & friction",
        "Journeys & friction",
        "Where people get stuck, and what they do about it",
    ),
    (
        "demand",
        "demand",
        "Demand & gaps",
        "Demand & gaps",
        "What people want to run, and what the catalog is missing",
    ),
)


def _admin_password() -> str | None:
    return os.environ.get("PLANNER_ADMIN_PASSWORD")


def _admin_sso_enabled() -> bool:
    return _env_bool("PLANNER_ADMIN_SSO_ENABLED", False)


def _admin_sso_users() -> set[str]:
    return {
        user.strip()
        for user in os.environ.get("PLANNER_ADMIN_SSO_USERS", "").split(",")
        if user.strip()
    }


def _admin_sso_url() -> str:
    return os.environ.get("PLANNER_ADMIN_SSO_URL", "https://tabs.gskrd.me/sso/gpumodels")


def _admin_configuration_error() -> str | None:
    if _admin_sso_enabled():
        if len(os.environ.get("SSO_SHARED_SECRET", "")) < 32 or not _admin_sso_users():
            return (
                "Admin SSO requires SSO_SHARED_SECRET (32+ characters) and PLANNER_ADMIN_SSO_USERS."
            )
        provider = urlsplit(_admin_sso_url())
        if (
            provider.scheme != "https"
            or not provider.hostname
            or provider.username
            or provider.password
            or provider.query
            or provider.fragment
        ):
            return "PLANNER_ADMIN_SSO_URL must be an HTTPS sign-in URL."
    elif not _admin_password():
        return "Set PLANNER_ADMIN_PASSWORD to enable /admin."
    if not current_app.config.get("PLANNER_SECRET_CONFIGURED", False):
        return "Set PLANNER_SECRET_KEY to enable /admin securely."
    return None


def _admin_session_token() -> str:
    if _admin_sso_enabled():
        identity = json.dumps(
            [
                "planner-admin-sso",
                os.environ.get("SSO_SHARED_SECRET", ""),
                sorted(_admin_sso_users()),
            ]
        )
    else:
        identity = "planner-admin:" + (_admin_password() or "")
    return hmac.new(
        str(current_app.secret_key).encode(),
        identity.encode(),
        "sha256",
    ).hexdigest()


def _is_admin_authenticated() -> bool:
    if _admin_sso_enabled():
        expires = session.get("planner_admin_expires")
        if (
            session.get("planner_admin_user") not in _admin_sso_users()
            or not isinstance(expires, int)
            or expires <= time.time()
        ):
            return False
    token = session.get(ADMIN_SESSION_KEY)
    return isinstance(token, str) and hmac.compare_digest(
        token.encode(), _admin_session_token().encode()
    )


def _safe_admin_path(value: str) -> str:
    if (
        len(value) > 4096
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or not (value == "/admin" or value.startswith(("/admin/", "/admin?")))
    ):
        return "/admin"
    return value


def _sso_token_user(token: str) -> str | None:
    """Validate the short-lived, audience-scoped Tabs handoff."""
    if len(token) > 4096:
        return None
    try:
        encoded, signature = token.split(".")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", encoded) or not re.fullmatch(
            r"[0-9a-f]{64}", signature
        ):
            return None
        expected = hmac.new(
            os.environ.get("SSO_SHARED_SECRET", "").encode(), encoded.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if not isinstance(payload, dict):
            return None
        expires = payload.get("exp")
        now = time.time()
        if (
            payload.get("purpose") != "gpumodels-sso"
            or type(expires) is not int
            or not now < expires <= now + 60
        ):
            return None
        user = payload.get("user")
        return user if isinstance(user, str) and user else None
    except (ValueError, UnicodeError):
        return None


@admin_bp.after_request
def _protect_admin_responses(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        configuration_error = _admin_configuration_error()
        if configuration_error:
            return configuration_error, 503
        if not _is_admin_authenticated():
            if request.path.endswith(".json"):
                return jsonify({"error": "Admin login required."}), 401
            if _admin_sso_enabled():
                return redirect(url_for("admin.admin_login", next=request.full_path.rstrip("?")))
            return render_template("admin_login.html", error=None)
        return view(*args, **kwargs)

    return wrapped


def _render_insight(endpoint: str):
    _, template, _, title, subtitle = next(p for p in INSIGHT_PAGES if p[0] == endpoint)
    return render_template(
        f"admin/{template}.html",
        pages=INSIGHT_PAGES,
        page=endpoint,
        title=title,
        subtitle=subtitle,
        tracking_enabled=current_app.config.get("TRACKING_ENABLED", False),
    )


@admin_bp.route("/admin", methods=["GET"])
@admin_required
def admin():
    return _render_insight("admin")


@admin_bp.route("/admin/audience", methods=["GET"])
@admin_required
def audience():
    return _render_insight("audience")


@admin_bp.route("/admin/personas", methods=["GET"])
@admin_required
def personas():
    return _render_insight("personas")


@admin_bp.route("/admin/people", methods=["GET"])
@admin_required
def people():
    return _render_insight("people")


@admin_bp.route("/admin/journeys", methods=["GET"])
@admin_required
def journeys():
    return _render_insight("journeys")


@admin_bp.route("/admin/demand", methods=["GET"])
@admin_required
def demand():
    return _render_insight("demand")


@admin_bp.route("/admin/data.json", methods=["GET"])
@admin_required
def data():
    days = _bounded_int(request.args.get("days", 90), name="days", lo=1, hi=366)
    store = current_app.extensions["snapshot_store"]
    response = jsonify(build_admin_payload(store, days=days))
    response.headers["Cache-Control"] = "no-store"
    return response


@admin_bp.route("/admin/snapshots", methods=["GET"])
@admin_required
def snapshots():
    page = _bounded_int(request.args.get("page", 1), name="page", lo=1, hi=1_000_000)
    page_size = min(
        _bounded_int(request.args.get("per_page", ADMIN_PAGE_SIZE), name="per_page", lo=1, hi=500),
        ADMIN_PAGE_SIZE,
    )
    store = current_app.extensions["snapshot_store"]
    total_snapshots = store.count_snapshots()
    all_snapshots = store.list_snapshots(limit=page_size, offset=(page - 1) * page_size)
    visitor_map = defaultdict(list)
    for s in all_snapshots:
        visitor_map[s["visitor_id"]].append(s)
    visitors = sorted(
        [{"visitor_id": vid, "snapshots": snaps} for vid, snaps in visitor_map.items()],
        key=lambda v: v["snapshots"][0]["last_seen"],
        reverse=True,
    )
    return render_template(
        "admin.html",
        visitors=visitors,
        total_snapshots=total_snapshots,
        page=page,
        page_size=page_size,
        GPUS=GPUS,
        MODELS=MODELS,
    )


@admin_bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    password = _admin_password()
    configuration_error = _admin_configuration_error()
    if configuration_error:
        return configuration_error, 503
    if _admin_sso_enabled():
        if request.method == "POST":
            return "Password login is disabled. Use shared sign-in.", 405
        if _is_admin_authenticated():
            return redirect(url_for("admin.admin"))
        state = secrets.token_urlsafe(32)
        session["planner_admin_sso"] = {
            "state": state,
            "expires": int(time.time()) + 300,
            "next": _safe_admin_path(request.args.get("next", "/admin")),
        }
        return redirect(_admin_sso_url() + "?" + urlencode({"state": state, "next": "/admin"}))
    if request.method == "GET":
        return render_template("admin_login.html", error=None)
    if not hmac.compare_digest(
        request.form.get("password", "").encode("utf-8"), (password or "").encode("utf-8")
    ):
        return render_template("admin_login.html", error="Invalid password."), 401

    session[ADMIN_SESSION_KEY] = _admin_session_token()
    # Exclude this browser from visitor analytics from now on.
    mark_internal_if_admin()
    return redirect(url_for("admin.admin"))


@admin_bp.route("/auth/callback", methods=["GET"])
def admin_sso_callback():
    if not _admin_sso_enabled():
        return "Shared sign-in is disabled.", 404
    configuration_error = _admin_configuration_error()
    if configuration_error:
        return configuration_error, 503
    pending = session.pop("planner_admin_sso", None)
    state = request.args.get("state", "")
    if (
        not isinstance(pending, dict)
        or pending.get("expires", 0) <= time.time()
        or not hmac.compare_digest(state.encode(), str(pending.get("state", "")).encode())
    ):
        return "Invalid or expired sign-in. Start again from /admin.", 401
    user = _sso_token_user(request.args.get("code", ""))
    if not user:
        return "Invalid or expired sign-in. Start again from /admin.", 401
    if user not in _admin_sso_users():
        return "This account does not have planner admin access.", 403
    session[ADMIN_SESSION_KEY] = _admin_session_token()
    session["planner_admin_user"] = user
    session["planner_admin_expires"] = int(time.time()) + 8 * 60 * 60
    mark_internal_if_admin()
    return redirect(_safe_admin_path(pending.get("next", "/admin")))


@admin_bp.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop(ADMIN_SESSION_KEY, None)
    for key in ("planner_admin_user", "planner_admin_expires", "planner_admin_sso"):
        session.pop(key, None)
    if _admin_sso_enabled():
        provider = urlsplit(_admin_sso_url())
        return redirect(f"{provider.scheme}://{provider.netloc}/logout")
    return redirect(url_for("admin.admin"))
