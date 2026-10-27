"""LFM3 release geometry, complete residency, cache units and planner routing."""

import copy
from dataclasses import replace

import pytest
from app_factory import create_test_app

from calc import (
    kv_cache_bytes_for_sequence,
    linear_attention_state_bytes,
    per_replica_kv_cache_bytes,
    per_tp_linear_attention_state_bytes,
)
from data import MODELS, OPEN_MODEL_ARCHITECTURE_SOURCES, configured_model
from data.groups import _model_kind, models_by_kind
from web.agent_contract import EXAMPLE_REQUEST

EXPECTED = [
    ("lfm3-1.7b", 1_728_801_536, 3_457_603_072, 262144, "llm"),
    ("lfm3-1.7b-base", 1_728_801_536, 3_457_603_072, 262144, "llm"),
    ("lfm3-vl-2b", 1_918_181_888, 3_836_363_776, 128000, "llm"),
    ("lfm3-audio-2b", 2_075_892_018, 4_151_786_596, 32768, "llm"),
    ("lfm3-omni-2b", 2_240_487_474, 4_480_977_508, 32768, "llm"),
    ("d1-2b", 1_918_181_888, 3_836_363_776, 128000, "decision"),
    ("d1-omni-2b", 2_166_928_384, 4_333_856_768, 128000, "decision"),
]


@pytest.mark.parametrize("key,params,weight_bytes,context,kind", EXPECTED)
def test_pinned_family_geometry_storage_context_and_picker(
    key, params, weight_bytes, context, kind
):
    m = MODELS[key]
    assert m.key == key and not m.hidden
    assert m.total_params == m.active_params == params
    assert m.prefill_active_params == params
    assert not m.is_moe
    assert (m.layers, m.hidden_size, m.num_heads, m.kv_heads, m.head_dim) == (24, 2048, 16, 4, 128)
    assert m.attention_layer_count == m.kv_layer_count == 11
    assert (m.local_attention_layers, m.local_attention_window) == (5, 256)
    assert m.linear_attention_layer_count == 0  # Short conv has no recurrent matrix.
    assert m.short_conv_state_elements == 186368
    assert m.weight_bytes("bf16") == weight_bytes
    assert m.max_context_tokens == context
    assert m.quality_confidence == 0
    assert _model_kind(m) == kind
    assert m in models_by_kind()[kind][m.cat]
    source = OPEN_MODEL_ARCHITECTURE_SOURCES[key]
    profile = m.quantization_profile("bf16")
    assert profile.source_revision == source.revision
    assert profile.captured_at == "2026-10-27"
    assert profile.source_kind == "exact"
    if kind == "decision":
        assert not m.capabilities
        assert m.decision_profile.max_sequence_length == context
        assert not m.decision_profile.causal_attention
        assert m.available_spec_profiles == ()


def test_modality_and_base_capabilities_are_not_inherited_from_post_training():
    assert MODELS["lfm3-1.7b-base"].capabilities == {"ctx_128k"}
    assert {"tools", "reasoning", "ctx_128k"} <= MODELS["lfm3-1.7b"].capabilities
    assert "images" in MODELS["lfm3-vl-2b"].capabilities
    assert "audio" in MODELS["lfm3-audio-2b"].capabilities
    assert {"images", "audio"} <= MODELS["lfm3-omni-2b"].capabilities
    assert "reasoning" not in MODELS["lfm3-omni-2b"].capabilities
    # Conversational audio models have text-token estimates, not invented ASR calibration.
    assert not MODELS["lfm3-audio-2b"].is_asr_model


def test_global_local_and_fixed_convolution_cache_are_counted_once():
    m = MODELS["lfm3-1.7b"]
    state_bytes = 13 * 2048 * 7 * 2
    assert linear_attention_state_bytes(m, "bf16") == state_bytes == 372736
    for seq in (128, 256, 8192, 262144):
        # K and V, four KV heads, 128 elements/head, two bytes/element.
        kv = (6 * seq + 5 * min(seq, 256)) * 2 * 4 * 128 * 2
        assert kv_cache_bytes_for_sequence(m, seq, "bf16") == kv
        assert per_replica_kv_cache_bytes(m, seq, "bf16", 1, 1) == kv + state_bytes
        assert per_replica_kv_cache_bytes(m, seq, "bf16", 2, 2) == (kv + state_bytes) / 4
    assert per_tp_linear_attention_state_bytes(m, "bf16", 2) == state_bytes / 2
    # Four KV heads replicate beyond TP=4; conv channels can continue to shard.
    assert per_tp_linear_attention_state_bytes(m, "bf16", 8) == state_bytes / 8
    assert per_tp_linear_attention_state_bytes(m, "bf16", 3) == state_bytes
    for precision in ("fp8", "nvfp4"):
        assert linear_attention_state_bytes(m, precision) == state_bytes
    independent = configured_model("lfm3-1.7b", "", "legacy", "fp8", "fp32")
    assert linear_attention_state_bytes(independent, "bf16") == state_bytes
    assert kv_cache_bytes_for_sequence(independent, 8192, "bf16") == (
        kv_cache_bytes_for_sequence(m, 8192, "bf16") / 2
    )
    # Other hybrid architectures still retain their existing recurrent state.
    hybrid = MODELS["rwkv7-g1d-01b"]
    augmented = replace(hybrid, short_conv_state_elements=100)
    assert linear_attention_state_bytes(augmented, "bf16") == (
        linear_attention_state_bytes(hybrid, "bf16") + 200
    )


def test_official_quantizations_retain_high_precision_islands_and_bf16_cache():
    m = MODELS["lfm3-1.7b"]
    assert m.weight_bytes("fp8") == 2_209_949_952
    assert m.weight_bytes("nvfp4") == 1_829_689_408
    for precision in ("fp8", "nvfp4"):
        profile = m.quantization_profile(precision)
        assert profile.source_repo == "LiquidAI/LFM3-1.7B-" + precision.upper()
        assert profile.source_kind == "exact"
        assert profile.compute_precision_shares["bf16"] > 0.25
        assert sum(profile.compute_precision_shares.values()) == pytest.approx(1)
        assert m.kv_cache_bytes_per_elem(precision) == 2
    # FP32 codec islands survive the native inventory, rather than rounding to 2 B/param.
    audio = MODELS["lfm3-audio-2b"]
    assert audio.weight_bytes("bf16") == audio.total_params * 2 + 1280 * 2


@pytest.mark.parametrize("key,params,weight_bytes,context,kind", EXPECTED)
def test_family_runs_through_planner_api_and_report(key, params, weight_bytes, context, kind):
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    panel = payload["scenario"]["panel_a"]
    panel["gpus"][0]["gpu_type"] = "H200"
    panel["models"][0].update(model_key=key, prec="bf16")
    panel["projects"] = []
    if kind == "decision":
        panel["mode"] = "decision"
    app = create_test_app()
    with app.test_client() as client:
        response = client.post("/api/v1/calculate", json=payload)
    assert response.status_code == 200
    result = response.get_json()
    assert result["results"]["panel_a"]["models"][0]["runnable"]
    assert MODELS[key].name in result["report"]
    if kind == "decision":
        assert "single forward, zero generated tokens" in result["report"]
