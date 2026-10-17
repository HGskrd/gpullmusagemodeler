"""Source-backed release geometry, packed storage and planner acceptance."""

import copy

import pytest
from app_factory import create_test_app

from calc import kv_cache_bytes_for_sequence, linear_attention_state_bytes, model_gpu_flops
from data import (
    CLOUD_MODELS,
    CORPO_CLOUD_PRESETS,
    GPUS,
    MODELS,
    OPEN_MODEL_ARCHITECTURE_SOURCES,
    PUBLISHED_ASR_WER,
)
from data.groups import _model_kind
from engine.economics import _dated_cloud_pricing
from placement import retune_models
from presentation.charts import chart_asr_quality
from state import GpuPool, ModelAssignment, PlannerState
from web.agent_contract import EXAMPLE_REQUEST

TEXT_KEYS = (
    "beam-1.0-501b-a23b",
    "mimo-v2.6-flash-rl",
    "mimo-v2.6-pro-rl",
    "mimo-v2.6-distill-qwen-9b",
    "ternary-bonsai-2-2b",
    "ternary-bonsai-2-4b",
    "ternary-bonsai-2-9b",
    "ternary-bonsai-2-27b",
    "lfm2.5-24b-a2b",
    "mellum2.1-12b-a2.5b-thinking",
    "hy-memory-30b-a3b",
    "nemotron35-super-vl",
    "nemotron35-bolt-74b",
    "muse-spark-open-816b-a42b",
)
EMBEDDING_KEYS = ("pplx-embed-v2-late-0.6b", "pplx-embed-v2-late-9b", "embeddinggemma-2")
ALL_KEYS = TEXT_KEYS + EMBEDDING_KEYS + ("jais-asr",)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_new_releases_have_pinned_sources_and_correct_picker_kind(key):
    model = MODELS[key]
    assert model.key == key
    assert not model.hidden
    assert _model_kind(model) == (
        "embedding" if key in EMBEDDING_KEYS else "asr" if key == "jais-asr" else "llm"
    )
    assert len(OPEN_MODEL_ARCHITECTURE_SOURCES[key].revision) == 40
    if key in TEXT_KEYS:
        assert model.quality_confidence == 0
        assert "uncalibrated" in model.architecture_note
        assert model.speculative_profiles == ()


def test_beam_cache_charges_only_nine_global_layers_at_long_context():
    model = MODELS["beam-1.0-501b-a23b"]
    sequence = 32768
    # Two K/V tensors, eight KV heads, 128 channels, two BF16 bytes.
    expected = (9 * sequence + 43 * 512) * 2 * 8 * 128 * 2
    assert kv_cache_bytes_for_sequence(model, sequence, "bf16") == expected
    assert model.layers == 52
    assert model.moe_routed_experts == 256


def test_mimo_flash_keeps_local_and_global_kv_geometry_distinct():
    model = MODELS["mimo-v2.6-flash-rl"]
    assert (model.global_kv_heads, model.local_kv_heads) == (4, 8)
    assert model.attention_value_head_size == 128
    # Pinned runtime cache stores 128d K/V per head; QK matmul is 192d.
    assert kv_cache_bytes_for_sequence(model, 16384, "mxfp4") == (
        9 * 16384 * 2 * 4 * 128 + 39 * 128 * 2 * 8 * 128
    )
    assert model.weight_bytes("mxfp4") == 172923364096
    assert MODELS["mimo-v2.6-pro-rl"].weight_bytes("mxfp4") == 566027990784


@pytest.mark.parametrize(
    "key,language_bytes,vision_bytes",
    [
        ("ternary-bonsai-2-2b", 429183456, 364664352),
        ("ternary-bonsai-2-4b", 949549920, 366894624),
        ("ternary-bonsai-2-9b", 1997486976, 624229664),
        ("ternary-bonsai-2-27b", 5946648928, 629246976),
    ],
)
def test_ternary_residency_includes_vision_and_does_not_claim_fp4_compute(
    key, language_bytes, vision_bytes
):
    model = MODELS[key]
    assert model.weight_bytes(model.native_precision_key) == language_bytes + vision_bytes
    assert model.quantization_profile(model.native_precision_key).compute_precision_shares == {
        "bf16": 1.0
    }
    assert model_gpu_flops(GPUS["H100"], model, model.native_precision_key) == GPUS["H100"].bf16
    assert model.quantization_profile(model.native_precision_key).container == "gguf"
    assert (
        model.quantization_profile(model.native_precision_key).format_scheme.encoding
        == "TERNARY PTQ1_0"
    )
    assert model.kv_cache_bytes_per_elem(model.native_precision_key) == 2
    assert linear_attention_state_bytes(model, model.native_precision_key) > 0


def test_embedding_serving_limits_and_pruned_hybrid_geometry():
    small = MODELS["pplx-embed-v2-late-0.6b"]
    assert (small.layers, small.attention_layers, small.linear_attention_layers) == (12, 6, 6)
    assert small.total_params == 594321600
    assert small.weight_bytes("bf16") == 594321600 * 4
    assert (small.embedding_profile.query_length, small.embedding_profile.document_length) == (
        1024,
        4096,
    )
    gemma = MODELS["embeddinggemma-2"]
    assert gemma.max_context_tokens == gemma.embedding_profile.max_sequence_length == 8192
    assert (gemma.local_attention_window, gemma.global_head_dim) == (512, 512)
    assert gemma.embedding_profile.output_dim == 768


def test_bolt_is_iq2_storage_with_bf16_compute_proxy():
    model = MODELS["nemotron35-bolt-74b"]
    assert model.weight_bytes(model.native_precision_key) == 24079678824
    assert (model.attention_layers, model.linear_attention_layers, model.kv_layers) == (8, 40, 8)
    assert "IQ2" in model.native_precision_display
    assert model.quantization_profile(model.native_precision_key).compute_precision_shares == {
        "bf16": 1.0
    }


def test_muse_gated_config_proxy_and_deepseek_refresh_are_explicit():
    muse = MODELS["muse-spark-open-816b-a42b"]
    assert "Gated checkpoint config" in muse.architecture_note
    assert muse.local_attention_layers == 0  # conservative upper bound, not an invented split
    assert muse.weight_bytes("bf16") == 821906490368 * 2 + 130039296 * 4
    deepseek = MODELS["deepseek-v4.1-flash"]
    assert "1017" in deepseek.name
    assert deepseek.weight_bytes(deepseek.native_precision_key) == pytest.approx(510286023000)
    assert "Flash-1017" in OPEN_MODEL_ARCHITECTURE_SOURCES[deepseek.key].repository
    assert deepseek.system_ram_weight_bytes > 200e9


def test_api_release_pricing_and_successor_rates_keep_individual_dates():
    expected = {
        "claude-opus-5.5": (4, 0.2, 20),
        "claude-sonnet-5.5": (2, 0.1, 10),
        "claude-haiku-5.5": (0.1, 0.01, 0.5),
        "mistral-large-4": (0.68, 0.07, 2.09),
    }
    for key, prices in expected.items():
        model = CLOUD_MODELS[key]
        assert tuple(model[f] for f in ("in_per_m", "cached_in_per_m", "out_per_m")) == prices
        assert model["pricing_captured_at"] == "2026-10-17"
        assert model["max_context_tokens"] == 1000000
    assert set(expected) <= set(CORPO_CLOUD_PRESETS["october_2026"]["models"])
    haiku = CLOUD_MODELS["claude-haiku-5.5"]
    assert haiku["long_context_threshold_tokens"] == 100000
    assert haiku["long_context_out_per_m"] == 2.5
    assert "long_context_threshold_inclusive" not in haiku  # exactly 100k stays base tier
    mistral, effective = _dated_cloud_pricing(
        CLOUD_MODELS["mistral-large-4"], {"pricing_as_of": "2026-10-20"}
    )
    assert effective == "2026-10-20"
    assert (mistral["in_per_m"], mistral["out_per_m"]) == (1.36, 4.18)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_each_added_model_executes_a_planner_report(key):
    app = create_test_app(REQUEST_RATE_LIMIT=100000)
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    panel = payload["scenario"]["panel_a"]
    panel["gpus"] = [{"uid": 1, "gpu_type": "H200", "count": 16}]
    panel["models"] = [
        {"model_key": key, "gpu_uid": 1, "gpu_count": 16, "prec": MODELS[key].native_precision_key}
    ]
    panel["projects"] = []
    with app.test_client() as client:
        response = client.post("/api/v1/calculate", json=payload)
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    diagnostic = result["results"]["panel_a"]["models"][0]
    assert diagnostic["model_key"] == key
    assert diagnostic["runnable"], diagnostic
    assert MODELS[key].name in result["report"]


def test_jais_quality_chart_uses_published_arabic_protocol_only():
    assert PUBLISHED_ASR_WER["jais-asr"] == {"ar_ifm_macro": 22.85}
    state = PlannerState(
        gpus=[GpuPool(1, "H200", 1)], models=[ModelAssignment(2, "jais-asr", 1, 1, 1, 1, "bf16")]
    )
    retune_models(state, preserve_existing=False)
    datasets = chart_asr_quality(state)
    assert len(datasets) == 1
    point = datasets[0]["data"][0]
    assert point["wer"] == 22.85
    assert point["max_users"] > 0
    assert "Arabic" in point["language"]
    assert "not comparable" in point["source"]
    assert not point["placeholder"]
