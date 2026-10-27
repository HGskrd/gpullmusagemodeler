"""Native modality routing, units, capacity and persistence regressions."""

import copy
import math

import pytest

from calc import compute_decision, compute_embedding, compute_realtime_max_users
from data import MODELS
from data.use_case_modalities import model_workload_kind
from engine.economics import _native_deployment_capacity, compute_revenue_projection
from placement import _model_serves_project, auto_select_models, retune_models
from presentation.reports import format_projection_report
from scenarios import deserialize_scenario, serialize_scenario
from state import (
    GpuPool,
    ModelAssignment,
    PlannerState,
    _add_project_from_preset,
    _sync_aggregate_distribution,
    set_use_case_def_field,
)

CASES = [
    ("meeting_transcription", "voxtral-realtime-mini-4b", "asr"),
    ("live_captions", "voxtral-realtime-mini-4b", "asr"),
    ("search_indexing", "bge-m3", "embedding"),
    ("search_queries", "bge-m3", "embedding"),
    ("ticket_routing", "intern-decision-0.8b", "decision"),
    ("content_moderation", "intern-decision-0.8b", "decision"),
]


def scenario(preset, model):
    s = PlannerState(
        gpus=[GpuPool(1, "H100", 1, cost_per_gpu_hour=1.0)],
        models=[ModelAssignment(2, model, 1, 1, 1, 1, "bf16")],
        projects=[],
    )
    p = _add_project_from_preset(s, preset)
    retune_models(s, preserve_existing=False)
    return s, p


@pytest.mark.parametrize("preset,model,kind", CASES)
def test_native_presets_route_only_to_their_modality_and_do_not_use_text_cloud(preset, model, kind):
    s, p = scenario(preset, model)
    row = compute_revenue_projection(s, include_recommendations=False)["projects"][0]
    assert row["served"] > 0
    assert row["cloud_blocked"]
    assert row["spilled"] == row["leaked"] == 0
    assert row["quality_status"] == "uncalibrated"
    assert row["workload_kind"] == kind
    assert math.isfinite(row["internal_cost_day"])
    assert row["served"] + row["destroyed"] == pytest.approx(row["tokens_day"])
    assert _model_serves_project(MODELS[model], p)
    assert not _model_serves_project(MODELS["qwen38-27b"], p)
    for _, other_model, other_kind in CASES:
        if other_kind != kind:
            assert not _model_serves_project(MODELS[other_model], p)
    s.models = [ModelAssignment(2, "qwen38-27b", 1, 1, 1, 1, "bf16")]
    wrong = compute_revenue_projection(s, include_recommendations=False)["projects"][0]
    assert wrong["served"] == 0
    assert wrong["destroyed"] == wrong["tokens_day"]
    assert "capacity only" in format_projection_report(scenario(preset, model)[0], None)


@pytest.mark.parametrize("preset,model,kind", CASES)
def test_native_round_trip_and_definition_edits(preset, model, kind):
    s, _ = scenario(preset, model)
    restored, _ = deserialize_scenario(serialize_scenario(s, None))
    assert restored.projects[0].workload_kind == kind
    assert restored.projects[0].tokens_day == s.projects[0].tokens_day
    set_use_case_def_field(restored, preset, "workload_kind", "llm")
    assert restored.projects[0].workload_kind == "llm"


@pytest.mark.parametrize("preset,model,kind", CASES[::2])
def test_native_capacity_has_correct_units_and_no_decode_or_retry_tax(preset, model, kind):
    s, p = scenario(preset, model)
    am, gpu = s.models[0], s.gpus[0].gpu
    seq = 1 if kind == "asr" else int(p.scale_kind["token_multiplier"])
    daily, rps = _native_deployment_capacity(s, am, gpu, {"in_len": seq})
    strat = (am.tp, am.pp, am.dp)
    if kind == "asr":
        users = compute_realtime_max_users(
            am.model, strat, gpu, s.mu, s.profiled_non_kv_gb, am.prec, s.decode_efficiency
        )
        assert daily == users * 86400
        assert p.tokens_day == p.scale_value * 3600
    else:
        compute = compute_embedding if kind == "embedding" else compute_decision
        single = compute(
            am.model, strat, 1, seq, gpu, s.mu, s.profiled_non_kv_gb, am.prec, s.prefill_efficiency
        )
        assert rps >= single.rps
        assert daily == pytest.approx(rps * 86400 * seq)
        assert _native_deployment_capacity(s, am, gpu, {"in_len": 1_000_000}) == (0, 0)
    before = compute_revenue_projection(s, include_recommendations=False)["projects"][0]["served"]
    p.prefix_hit_rate = 1.0
    p.out_pre = "Long doc"
    p.tokens_day = 1e12
    p.scale_value = p.tokens_day / p.scale_kind["token_multiplier"]
    p.batch_eligible = True
    row = compute_revenue_projection(s, include_recommendations=False)["projects"][0]
    assert row["served"] >= before
    assert row["served"] == pytest.approx(daily)


def test_native_stages_do_not_change_llm_shape_or_serve_generation():
    s = PlannerState(projects=[])
    p = _add_project_from_preset(s, "chatbot")
    before = copy.deepcopy((s.in_dist, s.out_dist, s.prefix_hit_rate))
    for preset, model, _ in CASES:
        _add_project_from_preset(s, preset)
        assert not _model_serves_project(MODELS[model], p)
    _sync_aggregate_distribution(s)
    assert (s.in_dist, s.out_dist, s.prefix_hit_rate) == before


def test_live_asr_requires_streaming_profile():
    offline = next(
        m.key for m in MODELS.values() if m.is_asr_model and not m.realtime_profile.streaming
    )
    s, p = scenario("live_captions", offline)
    assert not _model_serves_project(s.models[0].model, p)
    assert (
        compute_revenue_projection(s, include_recommendations=False)["projects"][0]["served"] == 0
    )


def test_two_native_projects_share_one_deployments_time():
    s, p = scenario("search_indexing", "bge-m3")
    p.tokens_day = 1e12
    p.scale_value = p.tokens_day / 512
    one = compute_revenue_projection(s, include_recommendations=False)["fates"]["served_tokens"]
    twin = copy.deepcopy(p)
    twin.uid += 100
    s.projects.append(twin)
    rows = compute_revenue_projection(s, include_recommendations=False)["projects"]
    assert sum(r["served"] for r in rows) == pytest.approx(one)
    assert sum(r["internal_cost_day"] for r in rows) <= 24.0 + 1e-6


@pytest.mark.parametrize("preset,_,kind", CASES[::2])
def test_auto_selection_covers_native_workload(preset, _, kind):
    s, p = scenario(preset, "qwen38-27b")
    auto_select_models(s, "lean")
    assert s.models
    assert any(model_workload_kind(am.model) == kind for am in s.models)
