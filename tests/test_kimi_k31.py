"""K3.1 Pro release geometry, native artifact and sparse-cache regressions."""

import pytest

from calc import (
    _decode_attention_work,
    _sparse_indexer_work,
    attention_kv_read_bytes_for_sequence,
    kv_cache_bytes_for_sequence,
    linear_attention_state_bytes,
)
from data import AA_QUALITY_PLACEHOLDER, MODELS, OPEN_MODEL_ARCHITECTURE_SOURCES
from data.artifacts import artifact_profiles, configured_model, default_artifact_id
from data.groups import models_by_kind


def test_kimi_k31_is_a_distinct_visible_llm_with_release_geometry():
    model = MODELS["kimi-k3.1-pro"]
    assert model.key == "kimi-k3.1-pro"
    assert model in models_by_kind()["llm"]["Kimi"]
    assert model.total_params == 2_780_110_363_392
    assert model.active_params == 104e9
    assert (model.layers, model.hidden_size, model.num_heads, model.kv_heads) == (93, 7168, 96, 96)
    assert (model.linear_attention_layer_count, model.attention_layer_count) == (69, 24)
    assert (model.mla_kv_dim, model.mla_rope_dim) == (512, 64)
    assert (model.sparse_attention_top_k, model.sparse_indexer_layers) == (2048, 24)
    assert (model.sparse_indexer_heads, model.sparse_indexer_head_dim) == (32, 128)
    assert (model.moe_routed_experts, model.moe_active_experts, model.moe_shared_experts) == (
        896,
        16,
        2,
    )
    assert (model.moe_latent_dim, model.moe_intermediate_dim) == (3584, 3072)
    assert model.attention_residual_block_count == 8
    assert model.max_context_tokens == 1_048_576
    assert model.capabilities == {"tools", "images", "reasoning", "ctx_128k"}
    assert model.quality_confidence == 0
    assert model.key in AA_QUALITY_PLACEHOLDER
    assert "not a benchmark" in model.architecture_note
    assert MODELS["kimi-k3"].sparse_attention_top_k == 0


def test_kimi_k31_native_artifact_uses_its_own_pinned_checkpoint():
    model = MODELS["kimi-k3.1-pro"]
    profile = model.quantization_profile("mxfp4")
    assert profile is not None
    assert profile in artifact_profiles(model.key)
    assert profile.source_kind == "exact"
    assert profile.source_repo == "moonshotai/Kimi-K3.1-Pro"
    assert profile.source_revision == OPEN_MODEL_ARCHITECTURE_SOURCES[model.key].revision
    assert profile.captured_at == "2026-11-04"
    assert profile.total_weight_bytes == 1_561_217_377_280
    assert model.total_params * model.weight_bytes_per_param("mxfp4") == pytest.approx(
        profile.total_weight_bytes
    )
    active_experts = 92 * 16 * 3 * 3584 * 3072
    assert profile.compute_precision_shares["mxfp4"] == pytest.approx(
        active_experts / model.active_params
    )
    assert model.active_weight_bytes("mxfp4") == pytest.approx(
        active_experts * 0.53125 + (model.active_params - active_experts) * 2
    )


@pytest.mark.parametrize("precision,main_bytes", [("bf16", 1152), ("mxfp4", 576)])
@pytest.mark.parametrize("seq", [1024, 131_072, 1_048_576])
def test_kimi_k31_retains_full_kv_but_only_reads_selected_latents(precision, main_bytes, seq):
    model = MODELS["kimi-k3.1-pro"]
    k3 = MODELS["kimi-k3"]
    index_bytes = 24 * seq * 132
    assert kv_cache_bytes_for_sequence(model, seq, precision) == 24 * seq * main_bytes + index_bytes
    assert attention_kv_read_bytes_for_sequence(model, seq, precision) == (
        24 * min(seq, 2048) * main_bytes + index_bytes
    )
    assert kv_cache_bytes_for_sequence(model, seq, precision) == (
        kv_cache_bytes_for_sequence(k3, seq, precision) + index_bytes
    )
    assert linear_attention_state_bytes(model, precision) == linear_attention_state_bytes(
        k3, precision
    )
    assert _sparse_indexer_work(model, 1, seq, prefill=False) == 2 * 24 * 32 * 128 * seq
    if seq > 2048:
        assert attention_kv_read_bytes_for_sequence(model, seq, precision) < (
            attention_kv_read_bytes_for_sequence(k3, seq, precision)
        )
        assert _decode_attention_work(model, 1, seq, 1) < _decode_attention_work(k3, 1, seq, 1)


def test_kimi_k31_index_cache_stays_fp8_when_main_cache_changes():
    model = MODELS["kimi-k3.1-pro"]
    configured = configured_model(
        model.key, default_artifact_id(model, "mxfp4"), "auto", "bf16", "bf16"
    )
    assert kv_cache_bytes_for_sequence(configured, 4096, "mxfp4") == 24 * 4096 * (1152 + 132)


def test_kimi_k31_uses_the_new_preview_speculator_and_workload_specific_acceptance():
    profile = next(p for p in MODELS["kimi-k3.1-pro"].speculative_profiles if p.method == "dspark")
    assert "Kimi-K3.1-Pro-DSpark-preview" in profile.source
    assert "5e654fbe920278f228a89519b06b2680aa56480f" in profile.source
    assert profile.draft_params == 2_686_752_257
    assert profile.default_k == 7
    assert profile.parallel_draft
    assert "GSM8K" in profile.note
    assert "uncalibrated" in profile.note
