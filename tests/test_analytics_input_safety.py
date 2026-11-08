"""Rejected API input and historical events must remain data in admin views."""

import copy
import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import pytest
from app_factory import create_test_app

from tracking import SnapshotStore
from web.admin import _admin_session_token
from web.agent_contract import EXAMPLE_REQUEST
from web.config import ADMIN_SESSION_KEY

MARKUP = '<img src=x onerror="window.analyticsInjection=true">'


@pytest.fixture
def analytics_client(tmp_path):
    store = SnapshotStore(tmp_path / "analytics.sqlite3")
    app = create_test_app(SNAPSHOT_STORE=store, TRACKING_ENABLED=True, REQUEST_RATE_LIMIT=100_000)
    with patch.dict("os.environ", {"PLANNER_ADMIN_PASSWORD": "test-admin"}):
        yield app.test_client(), store


def admin_events(client):
    with client.session_transaction() as session:
        with client.application.app_context():
            session[ADMIN_SESSION_KEY] = _admin_session_token()
    response = client.get("/admin/data.json")
    assert response.status_code == 200
    return [
        event
        for visitor in response.get_json()["visitors"]
        for session in visitor["sessions"]
        for event in session["events"]
        if event["type"] == "api_call"
    ]


@pytest.mark.parametrize("count", [MARKUP, "8", True, [], {}, -1, 0, 1.5, None, float("inf")])
def test_rejected_api_count_is_not_persisted_as_a_quantity(analytics_client, count):
    client, store = analytics_client
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    payload["scenario"]["panel_a"]["gpus"][0]["count"] = count
    response = client.post("/api/v1/calculate", json=payload)
    assert response.status_code == 400
    (event,) = store.list_analytics(since_ms=0)["events"]
    assert event["detail"]["x"]["gpus"][0][1] is None
    (event,) = admin_events(client)
    assert event["qty"] is None
    assert event["status"] == 400
    # The validation error can retain the input as text; rendering must escape it.


@pytest.mark.parametrize("count", [8, 8.0])
def test_valid_api_count_remains_an_integer(analytics_client, count):
    client, store = analytics_client
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    payload.pop("include", None)
    payload["scenario"]["panel_a"]["gpus"][0]["count"] = count
    assert client.post("/api/v1/calculate", json=payload).status_code == 200
    (event,) = store.list_analytics(since_ms=0)["events"]
    quantity = event["detail"]["x"]["gpus"][0][1]
    assert type(quantity) is int
    assert quantity == 8
    assert admin_events(client)[0]["qty"] == 8


@pytest.mark.parametrize("rows", [7, {}, "invalid", [None, 7]])
def test_malformed_api_collections_still_record_the_rejection(analytics_client, rows):
    client, store = analytics_client
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    payload["scenario"]["panel_a"].update(gpus=rows, models=rows)
    assert client.post("/api/v1/calculate", json=payload).status_code == 400
    (event,) = store.list_analytics(since_ms=0)["events"]
    assert event["detail"]["x"]["gpus"] == []
    assert event["detail"]["x"]["models"] == []
    assert admin_events(client)[0]["status"] == 400


def test_historical_raw_quantity_is_normalized_without_rewriting_history(analytics_client):
    client, store = analytics_client
    store.record_event(
        visitor_id="historical-agent",
        tab_id="api",
        event_type="api_call",
        detail={
            "x": {
                "endpoint": "POST /api/v1/calculate",
                "status": 400,
                "gpus": [["H100", MARKUP]],
                "models": [["l8", "bf16"]],
            }
        },
    )
    assert admin_events(client)[0]["qty"] is None
    assert store.list_analytics(since_ms=0)["events"][0]["detail"]["x"]["gpus"][0][1] == MARKUP


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is needed to execute the UI formatter"
)
def test_admin_event_formatter_escapes_untrusted_numeric_fields():
    root = Path(__file__).resolve().parents[1]
    page = (root / "templates/admin/people.html").read_text()
    formatter = page[page.index("  function stepText(e)") : page.index("  function intent(v)")]
    runtime = (root / "static/admin/admin.js").read_text()
    escape = next(line for line in runtime.splitlines() if "const esc =" in line)
    script = "\n".join(
        [
            escape,
            "const shortGpu = gpuName = shortModel = modelName = precLabel = String;",
            "const fmt = Number;",
            formatter,
            "const marker = " + json.dumps(MARKUP) + ";",
            "const events = ['api_call', 'gpu_add', 'gpu_qty', 'compare_gpu', 'gpu_cost', "
            "'model_count', 'result'].map(type => ({type, qty: marker, count: marker, "
            "usd_hr: marker, gpu: 'H100', model: 'l8', status: 400, fits: false, error: marker}));",
            "console.log(JSON.stringify(events.map(e => stepText(e)[0])));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True, timeout=10
    )

    class MarkupCheck(HTMLParser):
        def handle_starttag(self, tag, attrs):
            assert tag in {"span", "b"}
            assert all(name == "class" for name, _ in attrs)

    for rendered in json.loads(result.stdout):
        assert "&lt;img" in rendered
        MarkupCheck().feed(rendered)
