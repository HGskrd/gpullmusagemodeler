from dataclasses import replace

import pytest

from calc import resolve_spec_runtime, spec_finite_output_tau, spec_prefix_survival
from data import MODELS
from data.model_class import SpeculativeProfile


def test_survival_rejects_invalid_curves():
    for curve in ((0.8, 0.9), (float("nan"),), (-0.1,), (1.1,)):
        with pytest.raises(ValueError):
            spec_prefix_survival(0.5, len(curve), curve)


def test_finite_output_uses_prefix_distribution():
    # Two-token output: half of rounds emit two, half need another round.
    assert spec_finite_output_tau(0.9, 1, 2, (0.5,)) == pytest.approx(2 / 1.5)
    assert spec_finite_output_tau(0.9, 3, 1, (0.8, 0.6, 0.4)) == 1


def test_parallel_draft_width_is_independent_of_verification():
    profile = SpeculativeProfile(
        "DSpine test fixture",
        "dspine",
        1e8,
        5,
        True,
        2,
        0.5,
        0.1,
        "test",
        "synthetic, not a checkpoint calibration",
        proposed_tokens=3,
        prefix_survival_by_k=((3, (0.9, 0.7, 0.4)),),
        verify_anchor_positions=1,
    )
    model = replace(MODELS["q08"], speculative_profiles=(profile,))
    result = resolve_spec_runtime(model, "dspine", 2, 0, "bf16")
    assert result is not None
    assert result.proposed_tokens == 3
    assert result.k == 2
    assert result.tau == pytest.approx(2.6)
    override = resolve_spec_runtime(model, "dspine", 2, 0.5, "bf16")
    assert override is not None
    assert override.tau == pytest.approx(1.75)
    with pytest.raises(ValueError):
        resolve_spec_runtime(model, "dspine", 4, 0, "bf16")


def test_dspine_checkpoint_cache_and_acceptance():
    from calc import draft_cache_bytes

    model = MODELS["qwen38-27b"]
    runtime = resolve_spec_runtime(model, "dspine", 15, 0, "bf16")
    assert runtime is not None
    assert runtime.tau == pytest.approx(4.48)
    assert runtime.draft_weight_bytes == 4_051_824_128
    assert draft_cache_bytes(runtime, 8192, 1, 1, 0) == 167_772_160
    assert draft_cache_bytes(runtime, 100000, 1, 1, 0) == 167_772_160
    assert draft_cache_bytes(runtime, 8192, 1, 16, 0) == 167_772_160 / 8


def test_auto_off_card_memory_and_scorecard_match_baseline():
    from calc import spec_optimization_for
    from presentation.model_cards import get_model_info, tech_scorecard
    from state import GpuPool, ModelAssignment, PlannerState

    state = PlannerState(
        gpus=[GpuPool(1, "H100", 1)],
        models=[
            ModelAssignment(2, "qwen38-27b", 1, 1, 1, 1, "bf16", spec_method="dspine", spec_k=0)
        ],
    )
    am = state.models[0]
    selection = spec_optimization_for(state, am, am.model)
    assert not selection.beneficial
    auto = get_model_info(state, am)
    auto_score = tech_scorecard(state, {"p": {"models": []}})
    assert auto["spec"]["active"] is False
    assert auto["spec"]["candidate_feasible"] is False
    assert auto["spec"]["speedup"] == 0
    assert auto["spec"]["probe_bs"] == selection.probe_concurrency
    am.spec_method = "off"
    baseline = get_model_info(state, am)
    baseline_score = tech_scorecard(state, {"p": {"models": []}})
    assert auto["decode_max_slots"] == baseline["decode_max_slots"]
    assert auto["decode_mem"] == baseline["decode_mem"]
    assert {k: v for k, v in auto_score[0].items() if k != "info"} == {
        k: v for k, v in baseline_score[0].items() if k != "info"
    }


def test_dspine_manual_card_and_scenario_round_trip():
    from presentation.model_cards import get_model_info
    from scenarios import deserialize_scenario, serialize_scenario
    from state import GpuPool, ModelAssignment, PlannerState

    state = PlannerState(
        gpus=[GpuPool(1, "H100", 1)],
        models=[
            ModelAssignment(2, "qwen38-27b", 1, 1, 1, 1, "bf16", spec_method="dspine", spec_k=15)
        ],
    )
    restored, _ = deserialize_scenario(serialize_scenario(state, None))
    am = restored.models[0]
    assert (am.spec_method, am.spec_k) == ("dspine", 15)
    manual = get_model_info(restored, am)
    spec = manual["spec"]
    assert spec["active"]
    assert spec["unmeasured_prior"]
    assert (spec["proposed_tokens"], spec["draft_passes"], spec["verify_positions"]) == (15, 1, 16)
    assert spec["accepted_tokens"] == pytest.approx(3.48)
    am.spec_method = "off"
    baseline = get_model_info(restored, am)
    assert manual["decode_mem"].weights - baseline["decode_mem"].weights == pytest.approx(
        4_051_824_128
    )


def test_memory_bar_includes_active_draft_weights():
    from app_factory import create_test_app
    from flask import render_template

    from calc import strategy_label
    from data import GPUS
    from presentation.model_cards import get_model_info
    from state import GpuPool, ModelAssignment, PlannerState

    state = PlannerState(
        gpus=[GpuPool(1, "H100", 1)],
        models=[
            ModelAssignment(2, "qwen38-27b", 1, 1, 1, 1, "bf16", spec_method="dspine", spec_k=15)
        ],
    )
    with create_test_app().test_request_context():

        def render():
            return render_template(
                "partials/membar.html",
                state=state,
                MODELS=MODELS,
                GPUS=GPUS,
                strategy_label=strategy_label,
                get_model_info=get_model_info,
            )

        assert "weights 59.6 GB" in render()
        state.models[0].spec_k = 0
        assert "weights 55.6 GB" in render()
