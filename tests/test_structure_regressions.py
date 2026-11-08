"""Reachable failure cases from the implicit-structure review."""

import copy
import math
import os
import uuid
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
from app_factory import create_test_app

import benchmark_research
import web.analytics as analytics
import web.econ as econ_routes
import web.session_store as registry
from calc import _pp_boundary_counts, gpu_supports_mxfp4, gpu_supports_nvfp4
from data import GPUS
from data.hardware import MAX_GPU_POOLS, MAX_GPUS_PER_POOL, normalize_gpu_count
from engine.economics import _marginal_gpu_recommendations, compute_revenue_projection
from planner_service import add_model, add_models, apply_gpu_expansion, set_model_gpu_count
from presentation.econ import _chart_payload, econ_payload
from presentation.placement_view import _machines
from scenarios import deserialize_scenario, serialize_scenario
from state import (
    PlannerState,
    Project,
    add_gpu,
    add_project,
    normalize_projects,
    set_model_key,
    set_model_strat,
    set_project_field,
    set_project_scale_value,
)
from tracking import SnapshotStore, _restore_panel_state, _slim_panel_state
from web.config import ADMIN_SESSION_KEY
from web.helpers import _load_import_json
from web.middleware import VISITOR_COOKIE


def state_with_model(gpu="H100", count=1, model="l8"):
    state = PlannerState()
    add_gpu(state, gpu, count)
    add_model(state, model)
    state.projects = [Project(9001, "Custom", 0.1, 1e8, 4.0)]
    return state


def scoped_client(app):
    client = app.test_client()
    visitor, tab = str(uuid.uuid4()), str(uuid.uuid4())
    client.set_cookie(VISITOR_COOKIE, visitor)
    return client, {"X-Tab-ID": tab}, f"{visitor}:{tab}"


def test_unknown_quote_never_becomes_free_in_a_priced_fleet():
    state = state_with_model()
    state.gpus[0].cost_per_gpu_hour = 0
    add_gpu(state, "H200", 1)
    result = compute_revenue_projection(state)
    assert not result["economics_complete"]
    assert result["known_cost_day"] > 0
    assert math.isnan(result["cost_day"])
    assert math.isnan(result["margin_day"])
    assert math.isnan(result["models"][0]["internal_pm"])
    assert result["recommendations"] == []


def test_expansion_forecast_prices_the_same_purchase_as_apply(monkeypatch):
    monkeypatch.setitem(
        GPUS,
        "TEST_SYSTEM6",
        replace(GPUS["H100"], key="TEST_SYSTEM6", min_count=6, count_multiple=6),
    )
    state = state_with_model("TEST_SYSTEM6", 6)
    assignment = state.models[0]
    set_model_gpu_count(state, assignment.uid, 6)
    recommendations = _marginal_gpu_recommendations(state, -1e6, 0, 0, 0)
    (rec,) = recommendations
    applied = copy.deepcopy(state)
    apply_gpu_expansion(applied, assignment.uid, rec["added_gpus"])
    result = compute_revenue_projection(applied, include_recommendations=False)
    assert rec["purchased_gpus"] == 6
    assert applied.gpus[0].count == 12
    assert applied.models[0].gpu_count == rec["new_gpu_count"] == 7
    assert rec["margin_gain_day"] == pytest.approx(result["margin_day"] + 1e6)
    assert result["cost_day"] == pytest.approx(12 * state.gpus[0].cost_per_gpu_hour * 24)


def test_native_fp4_capability_follows_catalog_record_not_key_allowlists():
    gpu = replace(GPUS["RTX5060TI_16"], key="renamed-native-device")
    assert gpu_supports_nvfp4(gpu)
    assert gpu_supports_mxfp4(gpu)
    assert not gpu_supports_nvfp4(GPUS["H100"])
    assert not gpu_supports_nvfp4(GPUS["MI350X"])


def test_machine_map_and_pipeline_boundaries_use_the_same_packing():
    gpu = replace(GPUS["H100"], node_size=6)
    machines = _machines(4, 3, 1, 6, 12)
    assert len(machines) == 3
    assert [sum(slot["rep"] is not None for slot in machine) for machine in machines] == [4] * 3
    assert _pp_boundary_counts(4, 3, gpu) == (0, 2)
    assert sum(slot["rep"] is not None for machine in machines for slot in machine) == 12


def test_decision_layout_change_keeps_both_phases_together():
    state = state_with_model(count=4, model="d1-3b")
    assignment = state.models[0]
    set_model_gpu_count(state, assignment.uid, 4)
    set_model_strat(state, assignment.uid, 1, 1, 4, "prefill")
    assert (assignment.tp, assignment.pp, assignment.dp) == (1, 1, 4)
    assert (assignment.prefill_tp, assignment.prefill_pp, assignment.prefill_dp) == (1, 1, 4)


def test_bulk_add_failure_does_not_publish_any_mutations():
    state = PlannerState()
    add_gpu(state, "T4", 1)
    before = serialize_scenario(state, None)
    with pytest.raises(ValueError):
        add_models(state, ["l8", "kimi-k3"])
    assert serialize_scenario(state, None) == before


def test_scale_bound_is_identical_after_constructor_setter_and_import():
    state = PlannerState(projects=[Project(1, "Custom", 0.1, 0, 4, scale_value=2e6)])
    project = state.projects[0]
    assert project.tokens_day == 1e12
    assert project.scale_value == 1e6
    set_project_scale_value(state, project.uid, 2e6)
    assert project.tokens_day == 1e12
    assert project.scale_value == 1e6
    imported, _ = deserialize_scenario(serialize_scenario(state, None))
    normalize_projects(imported)
    assert imported.projects[0].tokens_day == project.tokens_day
    assert imported.projects[0].scale_value == project.scale_value


def test_definition_edits_and_import_overrides_become_explicit_custom_projects():
    state = PlannerState()
    add_project(state, "coding")
    project = state.projects[0]
    set_project_field(state, project.uid, "wtp_per_m", 199)
    normalize_projects(state)
    assert project.kind_key == "custom"
    assert project.wtp_per_m == 199
    exported = serialize_scenario(state, None)
    exported["panel_a"]["projects"][0].update(kind_key="coding", difficulty=0.1, wtp_per_m=42)
    imported, _ = deserialize_scenario(exported)
    normalize_projects(imported)
    assert imported.projects[0].kind_key == "custom"
    assert imported.projects[0].difficulty == 0.1
    assert imported.projects[0].wtp_per_m == 42


def test_model_swap_and_import_normalize_speculation():
    state = state_with_model()
    assignment = state.models[0]
    assignment.spec_method, assignment.spec_k = "eagle3", 4
    set_model_key(state, assignment.uid, "q08", "bf16")
    assert (assignment.spec_method, assignment.spec_k) == ("off", 0)
    payload = serialize_scenario(state, None)
    row = payload["panel_a"]["models"][0]
    row.update(model_key="g31", spec_method="mtp", spec_k=5, artifact_id="")
    imported, _ = deserialize_scenario(payload)
    assert imported.models[0].spec_method == "mtp"
    assert imported.models[0].spec_k == 0


@pytest.mark.parametrize(
    "fragment",
    [
        '"requires":7',
        '"requires":[{}]',
        '"cloud_enabled":"false"',
        '"company_size":1e400',
        '"gpu_type":[]',
    ],
)
def test_browser_import_boundary_rejects_malformed_nested_types(fragment):
    with pytest.raises(ValueError):
        _load_import_json("{" + fragment + "}")


def test_limits_are_enforced_before_export_and_do_not_silently_clamp_imports(monkeypatch):
    state = PlannerState()
    with pytest.raises(ValueError):
        add_gpu(state, "H100", MAX_GPUS_PER_POOL + 1)
    assert not state.gpus
    for key in list(GPUS)[:MAX_GPU_POOLS]:
        add_gpu(state, key, 1)
    with pytest.raises(ValueError):
        add_gpu(state, list(GPUS)[MAX_GPU_POOLS], 1)
    payload = serialize_scenario(state, None)
    payload["panel_a"]["gpus"][0]["count"] = MAX_GPUS_PER_POOL + 1
    with pytest.raises(ValueError):
        deserialize_scenario(payload)
    monkeypatch.setitem(
        GPUS,
        "TEST_SYSTEM6",
        replace(GPUS["H100"], key="TEST_SYSTEM6", min_count=6, count_multiple=6),
    )
    with pytest.raises(ValueError):
        normalize_gpu_count("TEST_SYSTEM6", MAX_GPUS_PER_POOL)


def test_snapshot_roundtrip_keeps_deleted_defaults_absent():
    assert _restore_panel_state(_slim_panel_state({"use_case_defs": []})) == {"use_case_defs": []}
    state = PlannerState()
    add_project(state, "coding")
    definitions = state.use_case_defs[1:]
    restored = _restore_panel_state(_slim_panel_state({"use_case_defs": definitions}))
    assert {d["key"] for d in restored["use_case_defs"]} == {d["key"] for d in definitions}


def test_unknown_fit_stays_unknown():
    plan = {"g": [[1, "H100", 1, 1]], "m": [[2, "l8", "bf16", 1, 1, None, None]]}
    assert analytics._final(plan)["fit"] is None
    assert analytics._fit_result(plan)["fits"] is None


def test_analytics_retries_failed_persistence(tmp_path):
    store = Mock()
    store.touch_visitor.side_effect = [OSError("temporary"), None]
    store.mark_internal.side_effect = [OSError("temporary"), None]
    app = create_test_app(TRACKING_ENABLED=True, SNAPSHOT_STORE=store)
    visitor = str(uuid.uuid4())
    with app.test_request_context(headers={"Cookie": f"{VISITOR_COOKIE}={visitor}"}):
        from flask import session

        session[ADMIN_SESSION_KEY] = "authenticated-test"
        analytics.touch_visitor(visitor, {"device": "agent"})
        analytics.touch_visitor(visitor, {"device": "agent"})
        analytics.mark_internal_if_admin()
        analytics.mark_internal_if_admin()
    assert store.touch_visitor.call_count == 2
    assert store.mark_internal.call_count == 2


def test_admin_password_rotation_revokes_existing_session(tmp_path, monkeypatch):
    app = create_test_app(SNAPSHOT_STORE=SnapshotStore(tmp_path / "analytics.sqlite3"))
    client = app.test_client()
    monkeypatch.setenv("PLANNER_ADMIN_PASSWORD", "before")
    assert client.post("/admin/login", data={"password": "before"}).status_code == 302
    assert client.get("/admin/data.json").status_code == 200
    monkeypatch.setenv("PLANNER_ADMIN_PASSWORD", "after")
    assert client.get("/admin/data.json").status_code == 401
    assert client.post("/admin/login", data={"password": "after"}).status_code == 302
    assert client.get("/admin/data.json").status_code == 200


def test_chart_cache_does_not_reuse_a_replaced_comparison():
    app = create_test_app()
    client, headers, scope = scoped_client(app)
    registry.get_state(scope)
    registry.duplicate_compare_state(scope).mu = 0.8
    with patch("web.api._build_chart_payload", side_effect=lambda a, b: {"mu": b.mu}):
        first = client.get("/api/chart-data", headers=headers)
        registry.clear_compare_state(scope)
        registry.duplicate_compare_state(scope).mu = 0.7
        second = client.get("/api/chart-data", headers=headers)
    assert first.get_json() == {"mu": 0.8}
    assert second.get_json() == {"mu": 0.7}
    assert first.headers["ETag"] != second.headers["ETag"]


def test_recommendation_cache_keeps_assignment_identities_separate():
    app = create_test_app()
    with patch.object(econ_routes, "compute_swap_recs", return_value=[]) as compute:
        for _ in range(2):
            client, headers, _ = scoped_client(app)
            assert client.get("/econ/swaps", headers=headers).status_code == 200
    assert compute.call_count == 2


def test_active_scope_is_not_evicted_under_capacity_pressure(monkeypatch):
    scope = str(uuid.uuid4())
    state = registry.get_state(scope)
    registry.acquire_scope_lease(scope)
    try:
        monkeypatch.setattr(registry, "_STATE_MAX_SCOPES", 1)
        registry.get_state(str(uuid.uuid4()))
        state.mu = 0.6
        assert registry.get_state(scope) is state
        assert state.mu == 0.6
    finally:
        registry.release_scope_lease(scope)


def test_snapshot_failure_does_not_turn_a_successful_mutation_into_an_error(tmp_path):
    store = SnapshotStore(tmp_path / "analytics.sqlite3")
    app = create_test_app(TRACKING_ENABLED=True, SNAPSHOT_STORE=store)
    client, headers, scope = scoped_client(app)
    registry.reset_state(scope, blank=True)
    with patch.object(store, "record_snapshot", side_effect=OSError("disk unavailable")):
        response = client.post("/gpu/add", data={"gpu_type": "H100", "count": 1}, headers=headers)
    assert response.status_code == 200
    assert registry.get_state(scope).gpus[0].count == 1


def test_non_kv_control_includes_an_imported_reservation():
    app = create_test_app()
    client, headers, scope = scoped_client(app)
    registry.get_state(scope).profiled_non_kv_gb = 64
    response = client.get("/session/sync", headers=headers)
    assert b'max="64" value="64"' in response.data


def test_initial_placeholder_controls_are_inert_until_tab_sync():
    response = create_test_app().test_client().get("/")
    assert b'inert aria-busy="true"' in response.data


def test_allocation_chart_uses_assignment_ids_for_duplicate_model_names():
    state = state_with_model(count=2)
    # Import accepts repeated assignments; exercise the chart with split supply.
    duplicate = copy.deepcopy(state.models[0])
    duplicate.uid += 10000
    state.models.append(duplicate)
    result = econ_payload(state)["p"]
    result["projects"][0]["per_model_served"] = [
        {"am_uid": assignment.uid, "name": assignment.model.name, "tokens": 1}
        for assignment in state.models
    ]
    payload = _chart_payload(result, [])
    sources = [edge["source"] for edge in payload["alloc"]["links"]]
    assert len(sources) == len(set(sources)) == 2
    assert set(sources) <= {node["name"] for node in payload["alloc"]["nodes"]}


def test_mixed_workload_volume_is_grouped_and_has_no_combined_unit_price():
    state = state_with_model()
    state.projects.append(Project(9002, "Speech", 0.1, 86400, 1, workload_kind="asr"))
    result = compute_revenue_projection(state, include_recommendations=False)
    assert result["mixed_workload_units"]
    assert set(result["volumes_by_unit"]) == {"tokens", "audio seconds"}
    assert math.isnan(result["cost_per_m_served"])
    assert math.isnan(result["token_coverage"])
    client, headers, scope = scoped_client(create_test_app())
    registry.replace_scope_states(scope, state, None)
    response = client.get("/econ/brief", headers=headers)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'aria-label="Demand by unit"' in html
    assert "audio seconds/day" in html
    assert "mixed accounting work units/day" in html
    assert "NaN" not in html


def test_benchmark_publication_rolls_back_if_the_second_publish_fails(tmp_path, monkeypatch):
    output = tmp_path / "queue.jsonl"
    original = os.link
    calls = []

    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("manifest publication failed")
        original(source, destination)

    args = ["--hardware", "H100", "--model", "l8", "--queue", "all", "--output", str(output)]
    with monkeypatch.context() as context:
        context.setattr(benchmark_research.os, "link", fail_second)
        with pytest.raises(OSError):
            benchmark_research.main(args)
    assert not output.exists()
    assert not output.with_suffix(".jsonl.manifest.json").exists()
    assert benchmark_research.main(args) == 0


def test_benchmark_retry_finishes_an_interrupted_publication(tmp_path):
    output = tmp_path / "queue.jsonl"
    args = ["--hardware", "H100", "--model", "l8", "--queue", "all", "--output", str(output)]
    assert benchmark_research.main(args) == 0
    before = output.read_bytes()
    output.with_suffix(".jsonl.manifest.json").unlink()
    assert benchmark_research.main(args) == 0
    assert output.read_bytes() == before
    assert output.with_suffix(".jsonl.manifest.json").exists()


def test_picker_uses_project_domain_quality_instead_of_global_proxy():
    from data import MODELS
    from presentation.model_picker import panel_quality_floor, picker_row

    project = Project(
        99,
        "Repository coding agent",
        0.55,
        1.2e9,
        4.0,
        requires=frozenset({"tools", "ctx_128k"}),
        min_success_rate=0.85,
        quality_floor=0.70,
        quality_domain="coding",
        quality_weights={"coding": 0.70, "reasoning": 0.20, "long_context": 0.10},
    )
    state = PlannerState(projects=[project])
    assert picker_row(
        state, MODELS["laguna-s-2-1"], panel_quality_floor(state), True
    ).passes_quality


def test_context_rejection_keeps_its_reason_in_processing_chart():
    from presentation.charts import chart_processing_pareto

    state = state_with_model(model="tiny-aya-global")
    points = [point for series in chart_processing_pareto(state, [1]) for point in series["data"]]
    unavailable = [point for point in points if point["y"] is None]
    assert unavailable
    assert any(point["unavailable_reason"] == "context_limit" for point in unavailable)
