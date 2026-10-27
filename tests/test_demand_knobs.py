"""Company size, tech-savviness, per-use-case volume locks, and the cloud switch."""

from __future__ import annotations

import math

from engine.economics import compute_revenue_projection
from planner_service import create_default_scenario
from state import (
    COMPANY_SIZE_REFERENCE,
    demand_multiplier,
    effective_scale_value,
    set_cloud_enabled,
    set_company_size,
    set_project_scale_locked,
    set_tech_savviness,
)


def _baseline(state) -> float:
    return compute_revenue_projection(state, include_recommendations=False)["baseline_tokens_day"]


def test_reference_company_leaves_demand_unchanged():
    state, _ = create_default_scenario()
    assert state.company_size == COMPANY_SIZE_REFERENCE
    assert demand_multiplier(state) == 1.0
    raw = sum(p.tokens_day for p in state.projects)
    assert math.isclose(_baseline(state), raw, rel_tol=1e-9)


def test_company_size_and_savviness_scale_unlocked_demand_linearly():
    state, _ = create_default_scenario()
    base = _baseline(state)
    set_company_size(state, 250)
    assert math.isclose(_baseline(state), base * 2.5, rel_tol=1e-9)
    set_tech_savviness(state, "high")
    assert math.isclose(demand_multiplier(state), 2.5 * 1.6, rel_tol=1e-9)
    assert math.isclose(_baseline(state), base * 4.0, rel_tol=1e-9)


def test_locking_keeps_the_shown_volume_and_stops_scaling():
    state, _ = create_default_scenario()
    set_company_size(state, 300)
    project = state.projects[0]
    shown = effective_scale_value(state, project)
    set_project_scale_locked(state, project.uid, True)
    assert math.isclose(effective_scale_value(state, project), shown, rel_tol=1e-9)
    set_company_size(state, 600)
    assert math.isclose(effective_scale_value(state, project), shown, rel_tol=1e-9)
    set_project_scale_locked(state, project.uid, False)
    assert math.isclose(effective_scale_value(state, project), shown, rel_tol=1e-9)
    set_company_size(state, 1200)
    assert math.isclose(effective_scale_value(state, project), shown * 2, rel_tol=1e-9)


def test_cloud_off_moves_all_cloud_outflow_to_lost():
    state, _ = create_default_scenario()
    set_cloud_enabled(state, True)
    with_cloud = compute_revenue_projection(state, include_recommendations=False)
    assert with_cloud["fates"]["spilled_tokens"] + with_cloud["fates"]["leaked_tokens"] > 0
    set_cloud_enabled(state, False)
    without = compute_revenue_projection(state, include_recommendations=False)
    assert without["fates"]["spilled_tokens"] == 0
    assert without["fates"]["leaked_tokens"] == 0
    assert without["value_cloud_day"] == 0
    assert without["fates"]["destroyed_tokens"] > with_cloud["fates"]["destroyed_tokens"]


def test_scenario_roundtrip_keeps_knobs_and_locks():
    from planner_service import deserialize_scenario
    from scenarios import serialize_scenario

    state, _ = create_default_scenario()
    set_company_size(state, 4200)
    set_tech_savviness(state, "low")
    set_cloud_enabled(state, False)
    set_project_scale_locked(state, state.projects[1].uid, True)
    restored, _ = deserialize_scenario(serialize_scenario(state, None))
    assert restored.company_size == 4200
    assert restored.tech_savviness == "low"
    assert restored.cloud_enabled is False
    assert restored.projects[1].scale_locked is True
    assert restored.projects[0].scale_locked is False


def test_pricing_and_gateway_switches_set_a_and_b_together(tmp_path):
    import uuid

    from app_factory import create_test_app

    from tracking import SnapshotStore
    from web.middleware import VISITOR_COOKIE
    from web.session_store import get_compare_state, get_state

    app = create_test_app(SNAPSHOT_STORE=SnapshotStore(tmp_path / "snapshots.sqlite3"))
    client = app.test_client()
    visitor, tab = str(uuid.uuid4()), str(uuid.uuid4())
    client.set_cookie(VISITOR_COOKIE, visitor)
    headers = {"X-Tab-ID": tab}
    client.get("/", headers=headers)
    client.post("/compare/duplicate", headers=headers)
    scope = f"{visitor}:{tab}"

    response = client.post("/pricing/toggle", data={"value": "on"}, headers=headers)
    assert response.status_code == 200
    assert 'id="assumptionsBar"' in response.get_data(as_text=True)
    assert get_state(scope).price_driven_demand
    assert get_compare_state(scope).price_driven_demand
    client.post("/pricing/toggle", data={}, headers=headers)
    assert not get_state(scope).price_driven_demand
    assert not get_compare_state(scope).price_driven_demand

    client.post("/gateway", data={"value": "october_2026"}, headers=headers)
    assert get_state(scope).corpo_cloud == "october_2026"
    assert get_compare_state(scope).corpo_cloud == "october_2026"

    # A row's "When" choice survives the definition re-sync on the next request.
    project = next(p for p in get_state(scope).projects if p.batch_eligible)
    client.post(
        "/project/day-shape",
        data={"panel": "A", "uid": str(project.uid), "value": "flat"},
        headers=headers,
    )
    client.get("/", headers=headers)
    reloaded = next(p for p in get_state(scope).projects if p.uid == project.uid)
    assert not reloaded.batch_eligible
    assert reloaded.day_shape == "flat"


def test_use_case_day_shape_and_batch_choice():
    from state import set_project_day_shape

    state, _ = create_default_scenario()
    project = state.projects[0]
    set_project_day_shape(state, project.uid, "flat")
    assert not project.batch_eligible and project.day_shape == "flat"
    set_project_day_shape(state, project.uid, "batch")
    assert project.batch_eligible and project.day_shape == "flat"
    set_project_day_shape(state, project.uid, "not-a-shape")
    assert project.batch_eligible
