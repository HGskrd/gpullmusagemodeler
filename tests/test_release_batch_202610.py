"""Pinned release inventory and independent workload/accounting invariants."""

import copy
from dataclasses import replace

import pytest
from app_factory import create_test_app

from calc import (
    EfficiencyParams,
    _active_weight_bytes,
    attention_kv_read_bytes_for_sequence,
    compute_decision,
    compute_embedding,
    compute_realtime_capacity,
    kv_cache_bytes_for_sequence,
    per_replica_token_kv_cache_bytes,
    per_replica_token_kv_read_bytes,
    prefill_parameter_tokens,
    resolve_execution,
    speech_chunk_work,
)
from data import GPUS, MODELS, OPEN_MODEL_ARCHITECTURE_SOURCES, artifact_profiles
from data.groups import _model_kind, models_by_kind
from data.models_releases_202610 import RELEASE_EVIDENCE, RELEASE_MODELS
from placement import retune_models
from state import GpuPool, ModelAssignment, PlannerState
from web.agent_contract import EXAMPLE_REQUEST
from web.api import _build_chart_payload


@pytest.mark.parametrize("key", RELEASE_MODELS)
def test_released_models_have_pinned_sources_native_storage_and_picker(key):
    model = MODELS[key]
    source = OPEN_MODEL_ARCHITECTURE_SOURCES[key]
    assert model.key == key
    assert len(source.revision) == 40
    assert model.total_params >= model.active_params > 0
    assert (
        model.weight_bytes(model.native_precision_key) == RELEASE_EVIDENCE[key]["checkpoint_bytes"]
    )
    assert not model.hidden
    assert model in models_by_kind()[_model_kind(model)][model.cat]
    assert model.quality_confidence == 0
    assert "uncalibrated" in model.architecture_note
    if _model_kind(model) != "llm":
        assert model.available_spec_profiles == ()


def test_causal_embedding_attention_reduces_compute_without_changing_residency():
    causal = MODELS["qwen3-embedding-0.6b"]
    bidirectional = replace(
        causal,
        key="test-qwen3-bidirectional",
        embedding_profile=replace(causal.embedding_profile, causal_attention=False),
    )
    args = ((1, 1, 1), 4, 16384, GPUS["H200"], 0.9, 2, "bf16", EfficiencyParams())
    a = compute_embedding(causal, *args)
    b = compute_embedding(bidirectional, *args)
    assert a and b
    assert a.service_time < b.service_time
    assert a.max_batch == b.max_batch


def test_shared_kv_charges_unique_producers_but_every_attention_consumer_reads():
    model = MODELS["nemotron35-bolt-e4b"]
    seq = 32768
    # Four global producers: K and V, two heads, 512 channels, BF16.
    # Twenty local producers retain 512 positions with 256 channels.
    expected = 4 * seq * 2 * 2 * 512 * 2 + 20 * 512 * 2 * 2 * 256 * 2
    assert kv_cache_bytes_for_sequence(model, seq, "bf16") == expected
    reads = per_replica_token_kv_read_bytes(model, seq, "bf16", 1, 1)
    assert reads == 7 * seq * 2 * 2 * 512 * 2 + 35 * 512 * 2 * 2 * 256 * 2
    assert reads > expected
    assert attention_kv_read_bytes_for_sequence(model, seq, "bf16") == reads
    # Cross-stage consumers cannot silently halve a shared producer pool.
    assert per_replica_token_kv_cache_bytes(model, seq, "bf16", 2, 1) == expected


def test_dory_repeats_parameter_work_without_duplicating_stored_weights():
    model = MODELS["dory1-4b-2loop"]
    single = replace(model, parameter_passes=1)
    ratio = 1 + 1835560960 / 3975229440
    assert model.parameter_passes == pytest.approx(ratio)
    assert model.weight_bytes("fp8") == single.weight_bytes("fp8") == 4750663680
    assert prefill_parameter_tokens(model, 4096) == pytest.approx(
        ratio * prefill_parameter_tokens(single, 4096)
    )
    assert _active_weight_bytes(model, "fp8") == pytest.approx(
        ratio * _active_weight_bytes(single, "fp8")
    )
    assert (model.attention_layers, model.kv_layers, model.local_kv_layers) == (56, 56, 42)


def test_checkpoint_storage_and_reference_runtime_casts_are_independent():
    laya = MODELS["laya-multilingual"]
    assert laya.weight_bytes("bf16") == 643818002
    assert laya.gpu_weight_bytes("bf16") == 1287635992
    assert laya.active_weight_bytes("bf16") == 1287635992
    ctc = MODELS["meta-omniasr-ctc-300m-v2"]
    assert ctc.weight_bytes("bf16") == 1303931584
    assert ctc.gpu_weight_bytes("bf16") == 651965792
    assert MODELS["kev-4b"].quantization_profile("bf16").source_kind == "estimate"


def test_bolt_q4_is_full_gguf_storage_with_bf16_compute():
    model = MODELS["nemotron35-bolt-e4b"]
    artifact = next(p for p in artifact_profiles(model.key) if p.container == "gguf")
    assert artifact.total_weight_bytes == 6180446208
    assert artifact.compute_precision_shares == {"bf16": 1.0}
    configured = replace(model, selected_artifact_id=artifact.artifact_id, execution_backend="auto")
    execution = resolve_execution(GPUS["H200"], configured, artifact.precision_key)
    assert execution.path == "w4a16"
    assert "does not claim native MXFP4" in execution.note


def test_independent_question_forwards_multiply_work_without_generating_tokens():
    model = MODELS["d3-flash"]
    args = (model, (1, 1, 1), 4, 2048, GPUS["H200"], 0.9, 2, "bf16", EfficiencyParams())
    one = compute_decision(*args)
    three = compute_decision(*args, questions_per_request=3)
    assert one and three
    assert three.service_time == pytest.approx(3 * one.service_time)
    assert three.decisions_per_second == pytest.approx(one.decisions_per_second)
    assert three.output_tokens == 0
    assert compute_decision(*args, options_per_question=256) is None


def test_compact_decision_encoders_include_local_and_head_attention():
    for key in ("julia-1", "laya-multilingual"):
        model = MODELS[key]
        assert model.attention_layer_count == 24
        assert model.local_attention_layers == 14
        assert not model.decision_profile.causal_attention
        assert model.decision_profile.forward_per_question


def test_speech_encoder_decoder_and_cross_attention_are_accounted_separately():
    large = speech_chunk_work(MODELS["whisper-large-v3"], "bf16")
    turbo = speech_chunk_work(MODELS["whisper-large-v3-turbo"], "bf16")
    assert large.encoder_flops == turbo.encoder_flops > 0
    assert large.decode_flops > turbo.decode_flops > 0
    assert large.cross_attention_flops > turbo.cross_attention_flops > 0
    assert large.scratch_bytes > turbo.scratch_bytes > 0
    assert large.prefill_flops == turbo.prefill_flops == 0


def test_accumulated_audio_streaming_charges_replay_without_duplicating_weights():
    offline = MODELS["qwen3-asr-0.6b"]
    stream = MODELS["qwen3-asr-0.6b-streaming"]
    a = speech_chunk_work(offline, "bf16")
    b = speech_chunk_work(stream, "bf16")
    assert b.encoder_flops == 8 * a.encoder_flops
    assert b.prefill_flops == 8 * a.prefill_flops
    assert stream.gpu_weight_bytes("bf16") == offline.gpu_weight_bytes("bf16")
    # The streaming runtime also regenerates its five-token rollback every
    # two seconds, increasing transcript work and peak self-KV conservatively.
    assert stream.realtime_profile.tokens_per_second == 8.5
    assert b.decode_flops > a.decode_flops
    assert b.scratch_bytes > a.scratch_bytes


def test_speech_data_parallel_capacity_scales_and_ctc_has_no_decoder():
    model = MODELS["meta-omniasr-ctc-300m-v2"]
    work = speech_chunk_work(model, "bf16")
    assert work.encoder_flops > 0
    assert work.prefill_flops == work.decode_flops == work.cross_attention_flops == 0
    args = (GPUS["H200"], 0.9, 2, "bf16", EfficiencyParams())
    single = compute_realtime_capacity(model, (1, 1, 1), 4, *args)
    dual = compute_realtime_capacity(model, (1, 1, 2), 8, *args)
    assert single and dual
    assert dual.realtime_factor == single.realtime_factor
    assert dual.total_tps == 2 * single.total_tps
    assert dual.max_slots == 2 * single.max_slots


@pytest.mark.parametrize(
    "key,mode,unit",
    [
        ("harrier-oss-v1-270m", "embedquality", "docs/s"),
        ("whisper-large-v3-turbo", "asrquality", "concurrent streams"),
        ("meta-omniasr-ctc-300m-v2", "asrquality", "concurrent streams"),
        ("meta-omniasr-llm-7b-v2", "asrquality", "concurrent streams"),
    ],
)
def test_unrated_models_keep_capacity_visible_without_inventing_quality(key, mode, unit):
    state = PlannerState(
        gpus=[GpuPool(1, "H200", 1)],
        models=[ModelAssignment(2, key, 1, 1, 1, 1, "bf16")],
        projects=[],
        mode=mode,
    )
    retune_models(state, preserve_existing=False)
    payload = _build_chart_payload(state, None)
    assert payload["datasets"] == []
    row = payload["capacity_only"][0]
    assert row["capacity"] > 0
    assert row["unit"] == unit
    assert "quality" not in row and "wer" not in row


def test_new_speech_cards_and_reports_use_audio_working_memory_labels():
    from presentation.model_cards import get_model_info

    state = PlannerState(
        gpus=[GpuPool(1, "H200", 1)],
        models=[ModelAssignment(2, "whisper-large-v3-turbo", 1, 1, 1, 1, "bf16")],
        projects=[],
    )
    retune_models(state, preserve_existing=False)
    info = get_model_info(state, state.models[0])
    assert info["realtime"]["chunk_work"].cross_attention_flops > 0


def test_native_technical_scorecard_has_capacity_even_without_generation_demand():
    from presentation.model_cards import tech_scorecard

    state = PlannerState(
        gpus=[GpuPool(1, "H200", 3)],
        models=[
            ModelAssignment(i + 2, key, 1, 1, 1, 1, "bf16")
            for i, key in enumerate(("qwen3-embedding-0.6b", "d3-flash", "whisper-large-v3-turbo"))
        ],
        projects=[],
    )
    retune_models(state, preserve_existing=False)
    rows = tech_scorecard(state, {"p": {"models": []}})
    assert len(rows) == 3
    assert all(row["capacity"] > 0 for row in rows)
    assert [row["capacity_unit"] for row in rows] == [
        "input tokens",
        "input tokens",
        "audio seconds",
    ]
    assert [row["native_label"] for row in rows] == ["Embedding", "Decision", "ASR"]


@pytest.mark.parametrize("key", RELEASE_MODELS)
def test_each_release_runs_through_public_planner_report(key):
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
    assert diagnostic["runnable"], diagnostic
    assert MODELS[key].name in result["report"]
