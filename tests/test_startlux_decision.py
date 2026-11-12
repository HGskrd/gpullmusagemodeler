"""Pinned StartLux residency, workload semantics and public planner coverage."""

import copy

import pytest
from app_factory import create_test_app

from calc import EfficiencyParams, compute_decision
from data import GPUS, MODELS, OPEN_MODEL_ARCHITECTURE_SOURCES
from data.groups import _model_kind, models_by_kind
from web.agent_contract import EXAMPLE_REQUEST

# Independently recorded physical tensor payloads; AutoThink indexes overstate bytes.
EXPECTED = [
    ("v1.1-0.8b-instant", 873438784, 1746882752),
    ("v1.1-2b-instant", 2274069824, 4548144832),
    ("v1.1-4b-instant", 4659865088, 9319737856),
    ("v1.1-9b-instant", 9653104368, 19306216416),
    ("v1.1-27b-instant", 27781427952, 55562855904),
    ("v1.1-4b-autothink", 4539265536, 9078531072),
    ("v1.1-9b-autothink", 9409813744, 18819627488),
    ("v1.1-27b-autothink", 27356728560, 54713457120),
    ("v1-0.8b", 873438784, 1746882752),
    ("v1-2b", 2274069824, 4548144832),
    ("v1-4b", 4659865088, 9319737856),
    ("v1-9b", 9653104368, 19306216416),
    ("v1-27b", 27781427952, 55562855904),
    ("v1-35b-a3b", 35951822704, 71903645408),
]


def estimate(model, *, seq=2048, questions=1, options=4):
    return compute_decision(
        model,
        (1, 1, 1),
        1,
        seq,
        GPUS["H200"],
        0.9,
        2,
        "bf16",
        EfficiencyParams(),
        questions_per_request=questions,
        options_per_question=options,
    )


@pytest.mark.parametrize("suffix,params,native_bytes", EXPECTED)
def test_startlux_pinned_catalog_and_public_report(suffix, params, native_bytes):
    key = f"startlux-decision-{suffix}"
    model = MODELS[key]
    instant = suffix.endswith("instant")
    profile = model.decision_profile
    assert model.key == key and model.total_params == params
    assert model.weight_bytes("bf16") == native_bytes
    assert (
        profile.max_sequence_length == model.max_context_tokens == (131072 if instant else 262144)
    )
    assert profile.causal_attention is not instant
    assert profile.forward_per_question and profile.max_options == 26
    assert model.attention_layer_count * 4 == model.layers
    assert model.linear_attention_layer_count * 4 == model.layers * 3
    assert _model_kind(model) == "decision"
    assert model in models_by_kind()["decision"]["StartLux"]
    assert not model.capabilities and model.quality_confidence == 0
    assert model.available_spec_profiles == ()
    source = OPEN_MODEL_ARCHITECTURE_SOURCES[key]
    assert len(source.revision) == 40 and source.revision in profile.source
    assert model.quantization_profile("bf16").source_revision == source.revision
    if "autothink" in suffix:
        assert "think_mode=never" in model.name and "think_mode=never" in profile.note
        assert "second readout latency are unmodeled" in profile.note
    if "35b-a3b" in suffix:
        assert model.is_moe and model.active_params == 3e9
        assert (model.moe_routed_experts, model.moe_active_experts) == (256, 8)
        assert "proxy" in profile.note
    assert estimate(model, seq=model.max_context_tokens + 1) is None
    assert estimate(model, options=27) is None
    assert estimate(model, options=26).output_tokens == 0

    payload = copy.deepcopy(EXAMPLE_REQUEST)
    payload["scenario"]["panel_a"].update(
        gpus=[{"uid": 1, "gpu_type": "H200", "count": 1}],
        models=[{"model_key": key, "gpu_uid": 1, "gpu_count": 1, "prec": "bf16"}],
        projects=[],
        mode="decision",
        task_il=2048,
    )
    with create_test_app(REQUEST_RATE_LIMIT=100000).test_client() as client:
        response = client.post("/api/v1/calculate", json=payload)
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data["results"]["panel_a"]["models"][0]["runnable"]
    assert model.name in data["report"]
    assert "single forward, zero generated tokens" in data["report"]
    assert "Estimated" in data["report"]


@pytest.mark.parametrize("suffix", ["v1.1-4b-instant", "v1.1-4b-autothink", "v1-4b"])
def test_startlux_repeats_complete_forward_per_question(suffix):
    model = MODELS[f"startlux-decision-{suffix}"]
    single = estimate(model)
    triple = estimate(model, questions=3)
    assert triple.rps == pytest.approx(single.rps / 3)
    assert triple.decisions_per_second == pytest.approx(single.decisions_per_second)
    assert triple.output_tokens == 0


def test_instant_attention_cost_and_mixed_native_precision_are_preserved():
    # v1 4B and v1.1 4B Instant have identical tensors/geometry, different masking.
    causal = MODELS["startlux-decision-v1-4b"]
    instant = MODELS["startlux-decision-v1.1-4b-instant"]
    assert instant.weight_bytes("bf16") == instant.total_params * 2 + 3840 * 2
    assert estimate(instant, seq=32768).service_time > estimate(causal, seq=32768).service_time
    assert (instant.layers, instant.hidden_size, instant.num_heads, instant.kv_heads) == (
        32,
        2560,
        16,
        4,
    )
    assert (instant.linear_attention_heads, instant.linear_attention_k_heads) == (32, 16)
