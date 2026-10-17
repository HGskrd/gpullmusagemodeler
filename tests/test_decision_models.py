"""Decision workload units, source geometry and planner isolation."""

import copy
from unittest.mock import patch

import pytest
from app_factory import create_test_app

from calc import (
    EfficiencyParams,
    compute_data,
    compute_decision,
    compute_decode,
    compute_memory,
    model_gpu_flops,
)
from data import GPUS, MODELS, OPEN_MODEL_ARCHITECTURE_SOURCES
from data.groups import _model_kind, models_by_kind
from engine.economics import compute_revenue_projection
from placement import _model_serves_project, resolve_deployment, retune_models
from presentation.charts import (
    chart_decision_throughput,
    chart_processing_pareto,
    chart_user_pareto,
)
from presentation.model_cards import get_model_info
from scenarios import deserialize_scenario, serialize_scenario
from state import GpuPool, ModelAssignment, PlannerState, Project, set_model_spec
from web.agent_contract import EXAMPLE_REQUEST
from web.api import _build_chart_payload

EXPECTED = [
    ["d1-3b", 3123483888, 6246967776, 32768],
    ["d1-omni-600m", 587161089, 2348644356, 16384],
    ["pplx-decider-v1.1-27b", 26086635760, 52173271520, 262144],
    ["pplx-decider-flash-v1.2", 4539918336, 9079836672, 262144],
    ["clef", 27356728560, 54713457120, 16384],
    ["clef-flash", 9409813744, 18819627488, 16384],
    ["mapika-decider-2b", 1881825088, 3763650176, 32768],
    ["mapika-decider-4b", 4205751296, 8411502592, 32768],
    ["mapika-decider-35b-a3b", 34660610688, 69321221376, 32768],
    ["kev-27b", 25624600064, 51249200128, 65536],
    ["intern-decision-0.8b", 852985920, 1705977024, 8192],
    ["intern-decision-2b", 2213241664, 4426488512, 8192],
    ["intern-decision-4b", 4539265536, 9078538752, 8192],
]


def state_for(key="d1-3b", count=1):
    state = PlannerState(
        gpus=[GpuPool(1, "H200", count)],
        models=[ModelAssignment(2, key, 1, count, 1, 1, "bf16")],
        projects=[],
        mode="decision",
        task_il=2048,
    )
    retune_models(state, preserve_existing=False)
    return state


def estimate(model=None, bs=1, seq=2048, strat=(1, 1, 1), **kwargs):
    return compute_decision(
        model or MODELS["d1-3b"],
        strat,
        bs,
        seq,
        GPUS["H200"],
        0.9,
        2,
        "bf16",
        EfficiencyParams(),
        **kwargs,
    )


@pytest.mark.parametrize("key,params,weight_bytes,cap", EXPECTED)
def test_pinned_geometry_native_storage_and_separate_picker(key, params, weight_bytes, cap):
    model = MODELS[key]
    assert model.key == key
    assert model.total_params == params
    assert model.weight_bytes("bf16") == weight_bytes
    assert model.max_context_tokens == cap == model.decision_profile.max_sequence_length
    assert _model_kind(model) == "decision"
    assert not model.is_embedding_model and not model.is_asr_model
    assert not model.capabilities
    assert model.quality_confidence == 0
    assert model.available_spec_profiles == ()
    assert model.quantization_profile("bf16").source_kind == "exact"
    source = OPEN_MODEL_ARCHITECTURE_SOURCES[key]
    assert len(source.revision) == 40
    assert source.revision in model.decision_profile.source
    picker = models_by_kind()
    assert model in picker["decision"][model.cat]
    assert all(model not in group for group in picker["llm"].values())


def test_units_one_forward_and_multiple_questions_are_distinct_from_tokens():
    result = estimate(bs=4, questions_per_request=3, options_per_question=5)
    assert result is not None
    assert result.rps * result.service_time == pytest.approx(4)
    assert result.decisions_per_second == pytest.approx(3 * result.rps)
    assert result.input_tps == pytest.approx(2048 * result.rps)
    assert result.output_tokens == 0
    assert result.max_batch >= 4
    assert estimate(seq=8192, bs=4).rps < estimate(seq=2048, bs=4).rps


def test_data_parallel_replicas_scale_requests_without_changing_batch_latency():
    single = estimate(bs=4)
    dual = estimate(bs=8, strat=(1, 1, 2))
    assert dual.service_time == pytest.approx(single.service_time)
    assert dual.rps == pytest.approx(2 * single.rps)
    assert dual.max_batch == 2 * single.max_batch


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(seq=0),
        dict(bs=0),
        dict(seq=32769),
        dict(questions_per_request=0),
        dict(options_per_question=1),
        dict(strat=(0, 1, 1)),
    ],
)
def test_invalid_shapes_and_serving_caps_are_rejected_without_truncation(kwargs):
    assert estimate(**kwargs) is None


def test_native_fp32_residency_and_conservative_activation_memory_limit():
    model = MODELS["d1-omni-600m"]
    assert model.weight_bytes("bf16") == 587161089 * 4
    base = estimate(model, seq=16384)
    assert base is not None
    assert estimate(model, seq=16385) is None
    assert estimate(model, seq=16384, bs=base.max_batch + 1) is None
    mem = compute_memory(model, 1, 1, GPUS["H200"], 0.9, 2, "bf16", EfficiencyParams())
    activation_bytes = 4 * 16384 * 1024 * 2 + 16
    assert base.max_batch == int(mem.kv_budget / activation_bytes)


def test_noncausal_perplexity_attention_is_charged_without_a_decode_loop():
    model = MODELS["pplx-decider-v1.1-27b"]
    assert not model.decision_profile.causal_attention
    import calc

    original = calc._prefill_attention_work
    with patch("calc._prefill_attention_work", wraps=original) as work:
        assert estimate(model, seq=8192) is not None
    assert work.call_args.kwargs["causal"] is False
    with patch("calc._compute_decode_core", side_effect=AssertionError("decode forbidden")):
        assert estimate(model) is not None


def test_generation_paths_reject_decisions_and_do_not_supply_project_tokens():
    state = state_for()
    model = state.models[0].model
    eff = EfficiencyParams()
    assert (
        compute_decode(
            model, 1, 1, 1, 1, GPUS["H200"], 0.9, 2, "bf16", state.in_dist, state.out_dist, eff
        )
        is None
    )
    assert (
        compute_data(
            model, (1, 1, 1), (1, 1, 1), 1, 2048, 128, GPUS["H200"], 0.9, 2, "bf16", 0, eff, eff
        )
        is None
    )
    assert not _model_serves_project(model, Project(3, "custom", 0.5, 1000, 1))
    projection = compute_revenue_projection(state)
    assert not projection["has_supply"]
    assert projection["total_cap_tokens_day"] == 0
    assert projection["models"] == []
    assert chart_user_pareto(state, [1, 2], deployment=resolve_deployment(state)) == []
    processing = chart_processing_pareto(state, [1, 2])
    assert all(not any(p.get("y") for p in ds["data"]) for ds in processing)


def test_decision_chart_card_and_spec_guard_survive_scenario_round_trip():
    state = state_for("intern-decision-2b")
    set_model_spec(state, 2, "ngram", 4)
    assert state.models[0].spec_method == "off"
    restored, _ = deserialize_scenario(serialize_scenario(state, None))
    info = get_model_info(restored, restored.models[0])
    assert info["runnable"] and info["prefill_fits"] and not info["decode_fits"]
    assert info["decision"]["sample"].output_tokens == 0
    assert info["spec_options"] == ()
    payload = _build_chart_payload(restored, None)
    assert payload["mode"] == "decision"
    assert payload["datasets"] == chart_decision_throughput(restored)
    assert payload["datasets"][0]["data"][0]["y"] > 0
    restored.task_il = 16384
    assert not get_model_info(restored, restored.models[0])["runnable"]
    assert all(p["y"] is None for p in chart_decision_throughput(restored)[0]["data"])


@pytest.mark.parametrize("key,params,weight_bytes,cap", EXPECTED)
def test_each_decision_checkpoint_calculates_and_reports_through_public_api(
    key, params, weight_bytes, cap
):
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    panel = payload["scenario"]["panel_a"]
    panel.update(
        gpus=[{"uid": 1, "gpu_type": "H200", "count": 2}],
        models=[{"model_key": key, "gpu_uid": 1, "gpu_count": 2, "prec": "bf16"}],
        projects=[],
        mode="decision",
        task_il=2048,
    )
    app = create_test_app(REQUEST_RATE_LIMIT=100000)
    with app.test_client() as client:
        response = client.post("/api/v1/calculate", json=payload)
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data["results"]["panel_a"]["models"][0]["runnable"]
    assert MODELS[key].name in data["report"]
    assert "single forward, zero generated tokens" in data["report"]
    assert "Estimated" in data["report"]


def test_single_forward_roofline_matches_independent_causal_attention_work():
    model, gpu = MODELS["d1-3b"], GPUS["H200"]
    seq, batch = 2048, 4
    eff = EfficiencyParams(comp_eff=1, bw_eff=1, overhead=0, paged_oh=0)
    result = compute_decision(model, (1, 1, 1), batch, seq, gpu, 0.9, 2, "bf16", eff)
    # d1 has eight full GQA layers, width 32*64, and causal S(S+1)/2 pairs.
    ffn_flops = 2 * 3123483888 * batch * seq
    attention_flops = 4 * batch * 8 * (32 * 64) * (seq * (seq + 1) / 2)
    compute_seconds = (ffn_flops + attention_flops) / model_gpu_flops(gpu, model, "bf16")
    memory_seconds = 6246967776 / gpu.effective_bw
    probability_output_seconds = 4 * 4 * batch / gpu.effective_bw
    expected = max(compute_seconds, memory_seconds) + probability_output_seconds
    assert result.service_time == pytest.approx(expected)
    assert result.rps == pytest.approx(batch / expected)


def test_decision_memory_surface_labels_forward_work_buffer():
    from flask import render_template

    from calc import strategy_label

    state = state_for()
    app = create_test_app()
    with app.test_request_context():
        html = render_template(
            "partials/membar.html",
            state=state,
            MODELS=MODELS,
            GPUS=GPUS,
            strategy_label=strategy_label,
        )
    assert "Liquid d1 3B (F " in html
    assert "work buffer" in html and "Decision layout" in html
    assert "KV reserved" not in html and "Decode layout" not in html


def test_perplexity_flash_v12_geometry_joint_readout_and_long_context_envelope():
    model = MODELS["pplx-decider-flash-v1.2"]
    assert (model.layers, model.hidden_size, model.num_heads, model.kv_heads, model.head_dim) == (
        32,
        2560,
        16,
        4,
        256,
    )
    assert (model.attention_layer_count, model.linear_attention_layer_count) == (8, 24)
    assert (
        model.linear_attention_heads,
        model.linear_attention_k_heads,
        model.linear_attention_head_dim,
    ) == (32, 16, 128)
    assert not model.decision_profile.causal_attention
    assert model.weight_bytes("bf16") == 9078531072 + 255 * 2560 * 2
    assert model.total_params == 4539265536 + 255 * 2560
    assert "32k" in model.decision_profile.note
    assert estimate(model, seq=262145) is None
    result = estimate(model, seq=32768)
    assert result is not None and result.output_tokens == 0
    assert estimate(model, seq=2048).rps > result.rps
