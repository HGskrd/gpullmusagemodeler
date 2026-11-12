"""Regression coverage for persistent edits, bounded GPU variants and machine comparisons."""

import json
import uuid
from dataclasses import replace
from unittest.mock import patch

import pytest
from app_factory import create_test_app

import state as state_module
import web.planner as planner_routes
import web.session_store as session_store
from data import GPUS
from scenarios import deserialize_scenario, serialize_scenario
from state import PlannerState, pool_gpu


@pytest.fixture
def planner():
    app = create_test_app()
    client = app.test_client()
    tab = str(uuid.uuid4())
    headers = {"X-Tab-ID": tab}
    client.get("/", headers=headers)
    scope = next(key for key in session_store._states if key.endswith(f":{tab}"))
    return client, headers, session_store.get_state(scope)


@pytest.mark.parametrize("field", ["name", "capability"])
def test_project_edits_survive_the_next_request_and_export(planner, field):
    client, headers, state = planner
    project = state.projects[0]
    assert project.kind_key != "custom"
    data = {"uid": project.uid, "panel": "A", "field": field}
    if field == "name":
        data["value"] = "My workload"
    else:
        # Flip a real requirement, regardless of the preset's current baseline.
        data["cap"] = "tools"
        data["value"] = "off" if "tools" in project.requires else "on"
    response = client.post("/project/set", headers=headers, data=data)
    assert response.status_code == 200
    assert project.kind_key == "custom"
    assert client.get("/", headers=headers).status_code == 200
    restored, _ = deserialize_scenario(serialize_scenario(state, None))
    saved = restored.projects[0]
    if field == "name":
        assert saved.name == project.name == "My workload"
    else:
        assert ("tools" in saved.requires) == (data["value"] == "on")


def test_pool_gpu_lru_is_bounded_and_refreshes_replaced_catalog_entries(monkeypatch):
    cache = state_module._POOL_GPU_CACHE
    cache.clear()
    monkeypatch.setattr(state_module, "_POOL_GPU_CACHE_MAXSIZE", 2)
    try:
        first = pool_gpu("H100", 2, 100)
        pool_gpu("H100", 4, 100)
        assert pool_gpu("H100", 2, 100) is first  # refresh recency
        pool_gpu("H100", 8, 100)
        assert len(cache) == 2
        assert {key[1] for key in cache} == {2, 8}
        monkeypatch.setitem(GPUS, "H100", replace(GPUS["H100"], mem=123e9))
        assert pool_gpu("H100", 2, 100).mem == 123e9
        assert pool_gpu("H100", 2, 100) is not first
        # Canonical effective defaults share an entry; capped speeds do too.
        assert pool_gpu("H100", 0, 100) is pool_gpu("H100", GPUS["H100"].node_size, 100)
        assert pool_gpu("H100", 2, 1e6) is pool_gpu("H100", 2, 2e6)
    finally:
        cache.clear()


def test_updates_embed_comparison_and_fragment_uses_current_state(planner):
    client, headers, state = planner
    pool = state.gpus[0]
    with patch.object(
        planner_routes, "machine_options", wraps=planner_routes.machine_options
    ) as calc:
        response = client.post("/gpu/qty", headers=headers, data={"uid": pool.uid, "delta": 1})
        assert response.status_code == 200
        assert calc.call_count == 0
        assert b"data-pm-url=" not in response.data
        assert b'<table class="pm-grid">' in response.data
        url = f"/gpu/machine-options?panel=A&uid={pool.uid}"
        fragment = client.get(url, headers=headers)
        assert fragment.status_code == 200
        assert b'<table class="pm-grid">' in fragment.data
        assert fragment.headers["Cache-Control"] == "no-store"
        assert calc.call_count == 1
        assert calc.call_args.args == (state, pool)
        client.post(
            "/gpu/machines",
            headers=headers,
            data={"uid": pool.uid, "gpus_per_machine": 2, "network_gbps": 100},
        )
        assert calc.call_count == 1
        updated = client.get(url, headers=headers)
        assert updated.status_code == 200
        assert updated.data != fragment.data
        assert calc.call_count == 2


def test_lazy_comparison_does_not_substitute_a_missing_panel_or_pool(planner):
    client, headers, state = planner
    for url in ("/gpu/machine-options?uid=9999999", "/gpu/machine-options?panel=B&uid=9999999"):
        assert client.get(url, headers=headers).status_code == 404
    assert client.get("/gpu/machine-options?uid=invalid", headers=headers).status_code == 400


def test_legacy_inactive_settings_import_but_are_not_reexported():
    legacy = {
        "projection_demand_level": 0.7,
        "projection_night_batching": True,
        "projection_night_discount": 0.4,
        "projection_batch_eligible": 0.5,
        "projection_elasticity": 3,
    }
    original = serialize_scenario(PlannerState(), None)
    original["panel_a"].update(legacy)
    state, _ = deserialize_scenario(original)
    exported = serialize_scenario(state, None)
    assert set(legacy).isdisjoint(exported["panel_a"])
    assert exported == serialize_scenario(deserialize_scenario(exported)[0], None)


def test_use_case_export_import_controls_roundtrip_current_projects(planner):
    client, headers, state = planner
    html = client.get("/", headers=headers).get_data(as_text=True)
    assert 'aria-label="Export use cases A"' in html
    assert 'aria-label="Import use cases A"' in html
    payload = client.get("/project/export?panel=A", headers=headers).get_json()
    before = [(p.name, p.tokens_day, p.kind_key) for p in state.projects]
    response = client.post(
        "/project/import",
        headers=headers,
        data={"panel": "A", "json": json.dumps(payload)},
    )
    assert response.status_code == 200
    assert [(p.name, p.tokens_day, p.kind_key) for p in state.projects] == before
