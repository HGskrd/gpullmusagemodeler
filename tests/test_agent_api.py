"""Public agent workflow, isolation and transport/validation regression coverage."""

import copy
import json
import re
from unittest.mock import Mock, patch

import pytest
from app_factory import create_test_app
from jsonschema import Draft202012Validator

import web.session_store as session_store
from data import GPUS, MODELS
from engine.economics import compute_revenue_projection
from planner_service import deserialize_scenario
from presentation.reports import format_projection_report
from web.agent_contract import CALCULATE_REQUEST, EXAMPLE_REQUEST
from web.config import BASE_DIR


@pytest.fixture
def client():
    app = create_test_app(REQUEST_RATE_LIMIT=100_000)
    return app.test_client()


def test_agent_can_discover_and_execute_example_without_cookies(client):
    llms = client.get("/llms.txt")
    assert llms.status_code == 200
    assert llms.mimetype == "text/plain"
    assert llms.text.startswith("# GPU/LLM Usage Modeler\n")
    for path in re.findall(r"\]\((/[^)]+)\)", llms.text):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "Set-Cookie" not in response.headers

    contract = client.get("/api/v1/openapi.json").get_json()
    schema = contract["paths"]["/api/v1/calculate"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    Draft202012Validator.check_schema(schema)
    payload = client.get("/api/v1/example").get_json()
    Draft202012Validator(schema).validate(payload)
    response = client.post("/api/v1/calculate", json=payload)
    assert response.status_code == 200
    assert "Set-Cookie" not in response.headers
    assert response.headers["Cache-Control"] == "no-store"
    assert "</llms.txt>" in response.headers["Link"]
    result = response.get_json()
    state_a, state_b = deserialize_scenario(payload["scenario"])
    expected = compute_revenue_projection(state_a, include_recommendations=False)
    projection = result["results"]["panel_a"]["projection"]
    assert projection["ready"]
    for key in ("cost_day", "margin_day", "total_cap_tokens_day", "token_coverage", "fates"):
        assert projection[key] == expected[key]
    assert projection["cost_day"] == GPUS["H100"].default_tco_per_gpu_hour * 24
    assert result["report"] == format_projection_report(state_a, state_b)
    assert result["results"]["panel_b"] is None
    assert result["results"]["panel_a"]["models"][0]["runnable"]
    assert "charts" not in result
    assert result["warnings"]
    Draft202012Validator(CALCULATE_REQUEST).validate({"scenario": result["scenario"]})
    replay = client.post("/api/v1/calculate", json={"scenario": result["scenario"]}).get_json()
    assert replay["results"]["panel_a"]["projection"]["cost_day"] == projection["cost_day"]


def test_catalog_keys_match_calculator_and_documented_defaults(client):
    result = client.get("/api/v1/catalog").get_json()
    assert {row["key"] for row in result["models"]} == {
        k for k, m in MODELS.items() if not m.hidden
    }
    assert {row["key"] for row in result["gpus"]} == set(GPUS)
    assert {row["kind"] for row in result["models"]} == {"llm", "embedding", "asr"}
    assert result["defaults"]["gpus"] == result["defaults"]["models"] == []
    assert result["distributions"]["input_token_buckets"]
    assert result["limits"]["request_bytes"] == client.application.config["MAX_CONTENT_LENGTH"]
    Draft202012Validator(CALCULATE_REQUEST).validate(
        {"scenario": {"type": "gpullm-scenario", "version": 1, "panel_a": result["defaults"]}}
    )


def test_current_browser_export_is_valid_api_input(client):
    export = client.get("/scenario/export").get_json()
    Draft202012Validator(CALCULATE_REQUEST).validate({"scenario": export})
    response = client.post("/api/v1/calculate", json={"scenario": export})
    assert response.status_code == 200
    assert response.get_json()["results"]["panel_b"] is not None


def test_markdown_request_example_is_executable(client):
    guide = (BASE_DIR / "docs" / "api.md").read_text()
    example = json.loads(re.search(r"```json\n(.*?)\n```", guide, re.S).group(1))
    response = client.post("/api/v1/calculate", json=example)
    assert response.status_code == 200
    result = response.get_json()
    assert result["charts"]["datasets"]
    assert result["report"]
    assert result["charts"]["mode"] == result["scenario"]["panel_a"]["mode"]


def test_comparison_calculations_are_independent(client):
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    payload["include"] = []
    payload["scenario"]["panel_b"] = copy.deepcopy(payload["scenario"]["panel_a"])
    payload["scenario"]["panel_a"]["gpus"][0]["cost_per_gpu_hour"] = 1
    payload["scenario"]["panel_b"]["gpus"][0]["cost_per_gpu_hour"] = 3
    result = client.post("/api/v1/calculate", json=payload).get_json()
    assert result["results"]["panel_a"]["projection"]["cost_day"] == 24
    assert result["results"]["panel_b"]["projection"]["cost_day"] == 72
    assert "report" not in result
    assert "charts" not in result


def test_api_never_reads_browser_state_or_persists_snapshots(client):
    client.get("/")
    before = copy.deepcopy(session_store._states)
    before_b = copy.deepcopy(session_store._compare_states)
    store = Mock()
    client.application.config["TRACKING_ENABLED"] = True
    client.application.extensions["snapshot_store"] = store
    with patch("web.middleware.allow_visitor_scope", side_effect=AssertionError("browser scope")):
        for _ in range(2):
            response = client.post("/api/v1/calculate", json=EXAMPLE_REQUEST)
            assert response.status_code == 200
            assert "Set-Cookie" not in response.headers
    assert session_store._states == before
    assert session_store._compare_states == before_b
    # Only cookie-free analytics: no snapshots, no browser visitor, api_call events.
    assert not [c for c in store.mock_calls if c[0] == "record_snapshot"]
    events = [c.kwargs for c in store.mock_calls if c[0] == "record_event"]
    assert [e["event_type"] for e in events] == ["api_call", "api_call"]
    assert all(e["visitor_id"].startswith("agent-") for e in events)
    assert {c.args[0] for c in store.mock_calls if c[0] == "touch_visitor"} <= {
        e["visitor_id"] for e in events
    }


@pytest.mark.parametrize(
    "payload", [None, [], {}, {"scenario": {}}, {"scenario": [], "include": []}]
)
def test_invalid_top_level_payload_is_json_400(client, payload):
    response = client.post(
        "/api/v1/calculate", data=json.dumps(payload), content_type="application/json"
    )
    assert response.status_code == 400
    assert response.get_json()["error"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unexpected",), True),
        (("include",), ["nonsense"]),
        (("scenario", "version"), True),
        (("scenario", "panel_a", "models", 0, "model_key"), []),
        (("scenario", "panel_a", "models", 0, "gpu_uid"), 999),
        (("scenario", "panel_a", "models", 0, "gpu_count"), -1),
        (("scenario", "panel_a", "models", 0, "prec"), "unknown"),
        (("scenario", "panel_a", "models", 0, "spec_method"), "not-supported"),
        (("scenario", "panel_a", "gpus", 0, "gpu_type"), {}),
        (("scenario", "panel_a", "gpus", 0, "count"), True),
        (("scenario", "panel_a", "gpus", 0, "count"), 1.5),
        (("scenario", "panel_a", "gpus", 0, "count"), float("inf")),
        (("scenario", "panel_a", "mu"), float("nan")),
        (("scenario", "panel_a", "mu"), 1.01),
        (("scenario", "panel_a", "mu"), "0.9"),
        (("scenario", "panel_a", "in_dist"), [1]),
        (("scenario", "panel_a", "projects", 0, "requires"), {}),
        (("scenario", "panel_a", "projects", 0, "kind_key"), "unknown-kind"),
        (("scenario", "panel_a", "projects", 0, "quality_weights"), {"unknown": 1}),
        (("scenario", "panel_a", "projects"), [{}] * 257),
        (("scenario", "panel_b"), "bad"),
    ],
)
def test_invalid_nested_values_are_rejected_before_calculation(client, path, value):
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    response = client.post("/api/v1/calculate", json=payload)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_duplicate_pool_ids_are_rejected(client):
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    payload["scenario"]["panel_a"]["gpus"] *= 2
    response = client.post("/api/v1/calculate", json=payload)
    assert response.status_code == 400
    assert "unique" in response.get_json()["error"]


def test_empty_and_unrunnable_scenarios_return_strict_json(client):
    empty = {"scenario": {"type": "gpullm-scenario", "version": 1, "panel_a": {}}}
    response = client.post("/api/v1/calculate", json=empty)
    assert response.status_code == 200
    assert not response.get_json()["results"]["panel_a"]["projection"]["ready"]
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    payload["scenario"]["panel_a"]["profiled_non_kv_gb"] = 4096
    response = client.post("/api/v1/calculate", json=payload)
    assert response.status_code == 200
    result = json.loads(response.text, parse_constant=lambda value: pytest.fail(value))
    assert not result["results"]["panel_a"]["models"][0]["runnable"]
    assert result["results"]["panel_a"]["models"][0]["decode_memory"] is None


@pytest.mark.parametrize(
    ("method", "path", "kwargs", "status"),
    [
        ("post", "/api/v1/calculate", {"data": "{", "content_type": "application/json"}, 400),
        ("post", "/api/v1/calculate", {"data": "{}"}, 415),
        ("get", "/api/v1/calculate", {}, 405),
        ("get", "/api/v1/missing", {}, 404),
        (
            "post",
            "/api/v1/calculate",
            {"data": "[" * 2000, "content_type": "application/json"},
            400,
        ),
    ],
)
def test_transport_errors_remain_json(client, method, path, kwargs, status):
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == status
    assert response.get_json()["error"]
    if status == 405:
        assert "POST" in response.headers["Allow"]


def test_oversized_and_rate_limited_requests_are_json(client):
    client.application.config["MAX_CONTENT_LENGTH"] = 32
    response = client.post("/api/v1/calculate", json=EXAMPLE_REQUEST)
    assert response.status_code == 413
    assert response.get_json()["error"]
    client.application.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    client.application.config["REQUEST_RATE_LIMIT"] = 1
    with patch("web.middleware._rate_windows", {}):
        assert client.post("/api/v1/calculate", json=EXAMPLE_REQUEST).status_code == 200
        response = client.post("/api/v1/calculate", json=EXAMPLE_REQUEST)
        assert response.status_code == 429
        assert response.get_json()["error"]
        assert int(response.headers["Retry-After"]) > 0


def test_unexpected_error_hides_internal_details(client):
    with patch("web.agent_api.deserialize_scenario", side_effect=RuntimeError("private-detail")):
        response = client.post("/api/v1/calculate", json=EXAMPLE_REQUEST)
    assert response.status_code == 500
    assert response.get_json() == {"error": "Unexpected server error."}
