"""Pinned V4.1 geometry and numerical CED/CSA2 accounting invariants."""

import math

import pytest
from app_factory import create_test_app

from calc import (
    EfficiencyParams,
    _decode_attention_work,
    _prefill_attention_work,
    attention_kv_read_bytes_for_sequence,
    compute_memory,
    compute_prefill,
    kv_cache_bytes_for_sequence,
    peak_stage_weight_bytes,
    per_replica_token_kv_cache_bytes,
    prefill_parameter_tokens,
)
from data import GPUS, MODELS
from data.model_sources import OPEN_MODEL_ARCHITECTURE_SOURCES
from presentation.reports import format_projection_report
from state import GpuPool, ModelAssignment, PlannerState

MODEL = MODELS["deepseek-v4.1-flash"]


def test_release_geometry_storage_and_provenance():
    assert MODEL.key == "deepseek-v4.1-flash"
    assert MODEL.total_params == 748e9
    assert MODEL.conditional_memory_params == 196e9
    assert (MODEL.prefill_active_params, MODEL.active_params) == (8e9, 16e9)
    assert (MODEL.layers, MODEL.hidden_size, MODEL.num_heads) == (40, 5120, 64)
    assert MODEL.moe_routed_experts == 384
    assert MODEL.moe_active_experts == 6
    assert MODEL.max_context_tokens == 1048576
    assert {"images", "reasoning", "tools", "ctx_128k"} <= MODEL.capabilities
    assert not MODEL.is_asr_model and not MODEL.is_embedding_model
    assert MODEL.weight_bytes("mxfp4") == pytest.approx(510_286_023_000)
    assert MODEL.quality_confidence == 0.5
    assert not MODEL.speculative_profiles  # Storage is included; no unmeasured speedup.
    profile = MODEL.quantization_profile("mxfp4")
    assert profile is not None
    assert profile.source_revision == OPEN_MODEL_ARCHITECTURE_SOURCES[MODEL.key].revision
    assert profile.captured_at == "2026-09-10"
    routed = 40 * 6 * 3 * 5120 * 2304
    assert MODEL.active_weight_bytes("mxfp4") == pytest.approx(
        routed * 0.53125 + (16e9 - routed) * 2
    )
    # Lower-bit generic conversions retain the FP8 conditional memory tables.
    assert MODEL.weight_bytes("fp4") > MODEL.conditional_memory_params


@pytest.mark.parametrize("seq", [-1, 0, 1, 2, 127, 128, 129, 32768, 1048576])
@pytest.mark.parametrize("precision", ["bf16", "fp8", "mxfp4"])
def test_four_global_stores_and_bounded_local_cache(seq, precision):
    n = max(seq, 0)
    expected = (3 * math.floor(n / 2) + n) * (288 + 68) + 40 * min(n, 128) * 528
    assert kv_cache_bytes_for_sequence(MODEL, seq, precision) == expected


def test_global_slope_and_parallel_replication():
    base = kv_cache_bytes_for_sequence(MODEL, 1048576, "mxfp4")
    assert base == 935_936_000
    assert base - kv_cache_bytes_for_sequence(MODEL, 1047552, "mxfp4") == 1024 * 890
    assert per_replica_token_kv_cache_bytes(MODEL, 1048576, "mxfp4", 1, 8) == base
    # PP splits local layers but must not divide the shared global pool.
    assert per_replica_token_kv_cache_bytes(MODEL, 1048576, "mxfp4", 2, 8) == (
        890 * 1048576 + 20 * 128 * 528
    )


def test_attention_reads_charge_consumers_and_hierarchical_indexing():
    seq = 1048576
    index_rows = 3 * (seq // 2) + seq + 4 * 16384
    expected = 40 * 128 * 528 + 38 * 512 * 288 + index_rows * 68
    assert attention_kv_read_bytes_for_sequence(MODEL, seq, "mxfp4") == expected
    # Main attention stays bounded; only the first four indexers grow with context.
    delta = _decode_attention_work(MODEL, 1, seq, 1) - _decode_attention_work(MODEL, 1, seq // 2, 1)
    assert delta == 2 * 32 * 128 * 2.5 * (seq // 2)


def test_ced_short_prompts_replay_and_pipeline_bottleneck():
    assert prefill_parameter_tokens(MODEL, 0) == 0
    assert prefill_parameter_tokens(MODEL, 64) == 16e9 * 64
    assert prefill_parameter_tokens(MODEL, 32768) == 8e9 * (32768 + 128)
    # Encoder stage remains the bottleneck: halving total CED work for PP2 is wrong.
    assert prefill_parameter_tokens(MODEL, 32768, 2) == 8e9 * 32768
    assert prefill_parameter_tokens(MODEL, 32768, 4) == 4e9 * 32768
    assert _prefill_attention_work(MODEL, 1, 0, 1) == 0
    assert _prefill_attention_work(MODEL, 1, 32768, 2) > (
        _prefill_attention_work(MODEL, 1, 32768, 1) / 2
    )


def test_memory_and_prefill_use_actual_checkpoint_and_full_resident_context():
    eff = EfficiencyParams()
    gpu = GPUS["B200"]
    assert compute_memory(MODEL, 1, 1, gpu, 0.9, 2, "mxfp4", eff) is None
    assert compute_memory(MODEL, 8, 1, gpu, 0.9, 2, "mxfp4", eff) is not None
    result = compute_prefill(
        MODEL,
        8,
        1,
        1,
        1,
        128,
        gpu,
        0.9,
        2,
        "mxfp4",
        eff,
        kv_residency_seq_len=1048576,
    )
    assert result is not None and result.service_time > 0
    assert (
        compute_prefill(
            MODEL,
            8,
            1,
            1,
            1,
            128,
            gpu,
            0.9,
            2,
            "mxfp4",
            eff,
            kv_residency_seq_len=1048577,
        )
        is None
    )


def test_picker_and_report_expose_new_model_and_limits():
    client = create_test_app().test_client()
    response = client.get("/picker/model?panel=A&kind=llm")
    assert response.status_code == 200
    assert MODEL.name in response.get_data(as_text=True)
    state = PlannerState(
        gpus=[GpuPool(901, "B200", 8)],
        models=[ModelAssignment(902, MODEL.key, 901, 8, 8, 1, "mxfp4")],
    )
    report = format_projection_report(state, None)
    assert "510.286 GB" in report
    assert "8B active prefill / 16B decode" in report
    assert "GPU-resident Engram" in report
    assert "speedup disabled" in report


def test_engram_tables_are_not_uniformly_spread_across_pipeline_stages():
    total = MODEL.weight_bytes("mxfp4")
    tables = MODEL.conditional_memory_weight_bytes
    assert tables == 202_758_032_400
    assert peak_stage_weight_bytes(MODEL, "mxfp4", 1) == pytest.approx(total)
    assert peak_stage_weight_bytes(MODEL, "mxfp4", 2) == pytest.approx(
        (total - tables) / 2 + tables
    )
    assert peak_stage_weight_bytes(MODEL, "mxfp4", 8) == pytest.approx(
        (total - tables) / 8 + tables / 2
    )
    # PP2 on two 288GB GPUs looks feasible under even weight division but is not.
    gpu = GPUS["B200"]
    assert compute_memory(MODEL, 1, 2, gpu, 0.9, 2, "mxfp4", EfficiencyParams()) is None
