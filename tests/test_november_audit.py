"""Source-backed numerical and serving-contract regressions for the November audit."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from calc import (
    EfficiencyParams,
    _decode_attention_work,
    attention_kv_read_bytes_for_sequence,
    compute_decision,
    compute_embedding,
    compute_memory,
    embedding_output_bytes_per_input,
    embedding_vectors_per_input,
    kv_cache_bytes_for_sequence,
    per_replica_token_kv_cache_bytes,
    per_replica_token_kv_read_bytes,
    resolve_execution,
    speech_chunk_work,
)
from data import CLOUD_MODELS, GPUS, MODEL_ARTIFACT_PROFILES, MODELS, PRECISION_SPECS
from data.artifacts import artifact_profiles, configured_model, default_artifact_id
from data.specs import _storage_bytes
from data.tinybox import TINYBOX_GPU_CONFIGS
from engine.economics import _cloud_price_per_m_in_preset


def configured(key, precision="bf16", **settings):
    return configured_model(
        key,
        default_artifact_id(MODELS[key], precision),
        settings.get("execution", "auto"),
        settings.get("kv", "bf16"),
        "auto",
        settings.get("runtime", "generic"),
        settings.get("version", ""),
    )


@pytest.mark.parametrize(
    "key,expected",
    [
        ("mimo-v2.6-flash-rl", 3_045_457_920),
        ("mimo-v2.6-pro-rl", 6_750_208_000),
    ],
)
def test_unequal_key_value_widths_charge_resident_bytes_and_value_work(key, expected):
    model = replace(MODELS[key], cache_precision="bf16")
    assert kv_cache_bytes_for_sequence(model, 131_072, "bf16") == expected
    assert (
        kv_cache_bytes_for_sequence(replace(model, cache_precision="fp8"), 131_072, "bf16")
        == expected / 2
    )
    # Each attended local row pays QK192 + V128, rather than 2*QK192.
    all_local = replace(
        model, key=key + "-local-work", attention_layers=1, local_attention_layers=1
    )
    assert (
        _decode_attention_work(all_local, 1, 128, 1)
        == 2 * model.local_attention_head_count * (192 + 128) * 128
    )


@pytest.mark.parametrize(
    "key,size,producers,local,ple,active",
    [
        ("ge2", 10_246_357_958, 15, 12, 4_697_620_480, 2_298_639_651),
        ("ge4", 15_992_314_836, 24, 20, 5_637_144_576, 4_699_496_746),
    ],
)
def test_gemma_residency_producers_and_lookup_are_independent_of_active_compute(
    key, size, producers, local, ple, active
):
    model = configured(key)
    assert model.weight_bytes("bf16") == size
    assert (model.kv_layer_count, model.local_kv_layers) == (producers, local)
    assert model.conditional_memory_weight_bytes == ple
    assert model.active_weight_bytes("bf16") == active * 2
    estimate = replace(model, selected_artifact_id="estimate:fp8")
    assert estimate.weight_bytes("fp8") == pytest.approx(
        (model.total_params - model.conditional_memory_params)
        * PRECISION_SPECS["fp8"].effective_weight_bytes_per_param
        + ple
    )
    assert estimate.active_weight_bytes("fp8") == pytest.approx(
        active * PRECISION_SPECS["fp8"].effective_weight_bytes_per_param
    )
    seq = 1024
    expected = (
        producers - local
    ) * seq * model.kv_heads * 512 * 4 + local * 512 * model.kv_heads * 256 * 4
    assert kv_cache_bytes_for_sequence(model, seq, "bf16") == expected
    assert attention_kv_read_bytes_for_sequence(model, seq, "bf16") > expected
    assert per_replica_token_kv_cache_bytes(model, seq, "bf16", 2, 1) == expected
    alias = MODELS["gemma-4-e2b-asr" if key == "ge2" else "gemma-4-e4b-asr"]
    assert alias.weight_bytes("bf16") == size


def test_gemma_e4b_does_not_fit_a_16gb_card_as_bf16():
    model = configured("ge4")
    assert (
        compute_memory(model, 1, 1, GPUS["RTX4060TI_16"], 0.9, 2, "bf16", EfficiencyParams())
        is None
    )
    assert (
        compute_memory(model, 1, 1, GPUS["RTX4090"], 0.9, 2, "bf16", EfficiencyParams()) is not None
    )
    assert MODELS["g12"].max_context_tokens == 262_144


@pytest.mark.parametrize(
    "key,size,draft",
    [
        ("q08", 1_705_977_024, 40_905_728),
        ("q2", 4_426_488_512, 121_656_320),
        ("q4", 9_078_538_752, 241_199_104),
        ("q9", 18_819_635_168, 486_581_248),
    ],
)
def test_qwen_native_subset_keeps_vision_and_optional_draft_separate(key, size, draft):
    model = configured(key)
    assert model.weight_bytes("bf16") == size
    assert model.active_weight_bytes("bf16") < size
    assert model.speculative_profiles[0].exact_weight_bytes == draft
    assert model.speculative_profiles[0].active_params <= model.speculative_profiles[0].draft_params


def test_mistral_shared_cache_consumers_and_complete_artifacts():
    model = configured("mistral-large-4", "fp8")
    assert model.weight_bytes("fp8") == 1_060_548_427_328
    assert model.weight_bytes("nvfp4") == 610_030_457_408
    assert model.speculative_profiles == ()
    assert kv_cache_bytes_for_sequence(model, 1000, "fp8") == 83_840_000
    assert per_replica_token_kv_cache_bytes(model, 1000, "fp8", 2, 8) == 83_840_000
    # All72 consumers gather latents, while44 indexers scan full context.
    expected = 72 * 1000 * 576 * 2 + 44 * 1000 * 256
    assert per_replica_token_kv_read_bytes(model, 1000, "fp8", 1, 8) == expected
    assert per_replica_token_kv_read_bytes(model, 1000, "fp8", 2, 8) == expected / 2


@pytest.mark.parametrize(
    "key,dim", [("pplx-embed-v2-context-0.6b", 1024), ("pplx-embed-v2-context-9b", 2048)]
)
def test_contextual_embeddings_emit_chunk_vectors_from_one_full_document(key, dim):
    model = configured(key)
    assert [embedding_vectors_per_input(model, n) for n in (512, 513, 1024)] == [1, 2, 2]
    assert embedding_output_bytes_per_input(model, 1024) == dim * 4 * 2
    # Chunk count changes output storage, not the full-document attention workload.
    result = compute_embedding(
        model, (1, 1, 1), 1, 1024, GPUS["H200"], 0.9, 4, "bf16", EfficiencyParams()
    )
    longer = compute_embedding(
        model, (1, 1, 1), 1, 2048, GPUS["H200"], 0.9, 4, "bf16", EfficiencyParams()
    )
    assert result and longer and longer.service_time > result.service_time
    assert (
        compute_embedding(
            model, (1, 1, 1), 1, 262_145, GPUS["H200"], 0.9, 4, "bf16", EfficiencyParams()
        )
        is None
    )


def test_ctc_stages_have_no_generated_tokens_or_duplicate_resident_head():
    model = configured("granite-speech-5.0-470m-turboctc")
    work = speech_chunk_work(model, "bf16")
    assert model.weight_bytes("bf16") == 946_249_856
    assert [stage[1] for stage in model.realtime_profile.audio_encoder_stages] == [50, 25, 12.5]
    assert work.prefill_flops == work.decode_flops == work.cross_attention_flops == 0
    assert kv_cache_bytes_for_sequence(model, 100_000, "bf16") == 0
    assert work.weight_read_bytes == 946_249_856
    shorter = replace(
        model, realtime_profile=replace(model.realtime_profile, audio_chunk_seconds=15)
    )
    assert speech_chunk_work(shorter, "bf16").encoder_flops == work.encoder_flops / 2


def test_new_modalities_reach_quality_or_explicit_capacity_only_surfaces():
    from placement import retune_models
    from presentation.capacity_only import capacity_without_quality
    from presentation.charts import chart_asr_quality
    from state import GpuPool, ModelAssignment, PlannerState

    context = "pplx-embed-v2-context-0.6b"
    state = PlannerState(
        gpus=[GpuPool(1, "H200", 1)], models=[ModelAssignment(2, context, 1, 1, 1, 1, "bf16")]
    )
    retune_models(state, preserve_existing=False)
    rows = capacity_without_quality(state, "embedquality")
    assert len(rows) == 1 and rows[0]["capacity"] > 0 and rows[0]["unit"] == "docs/s"
    assert "no compatible published quality" in rows[0]["note"]
    state.models = [ModelAssignment(3, "granite-speech-5.0-470m-turboctc", 1, 1, 1, 1, "bf16")]
    retune_models(state, preserve_existing=False)
    datasets = chart_asr_quality(state)
    assert len(datasets) == 1
    point = datasets[0]["data"][0]
    assert point["wer"] == 5.51 and point["max_users"] > 0
    assert "unofficial" in point["source"] and "Japanese CER" in point["source"]


def test_decider_v12_uses_one_joint_forward_and_includes_readout():
    model = configured("pplx-decider-v1.2")
    assert model.weight_bytes("bf16") == 52_173_271_520
    args = (model, (1, 1, 1), 1, 1024, GPUS["H200"], 0.9, 4, "bf16", EfficiencyParams())
    one = compute_decision(*args, questions_per_request=1)
    four = compute_decision(*args, questions_per_request=4)
    assert one and four
    assert four.rps == pytest.approx(one.rps, rel=1e-8)
    assert four.decisions_per_second == pytest.approx(one.decisions_per_second * 4, rel=1e-8)
    assert one.output_tokens == four.output_tokens == 0
    assert compute_decision(*args, options_per_question=256) is None


def cloud_quote(row, **profile):
    request = {"in_len": 1000, "out_len": 0, "tokens_per_request": 1000, **profile}
    with patch(
        "engine.economics.cloud_policy.effective_corpo_models", return_value=[("test", row)]
    ):
        return _cloud_price_per_m_in_preset(0, 0, 0, request, profile.get("hit", 0), "test")


def test_cloud_actual_output_and_input_bounds():
    row = {**CLOUD_MODELS["gpt-6-astra"], "token_efficiency": 0.125}
    assert cloud_quote(row, out_len=20_000)[0] is None
    assert cloud_quote(row, out_len=16_000)[0] is not None
    assert cloud_quote(CLOUD_MODELS["gpt-6.1-sol"], in_len=922_001)[0] is None


@pytest.mark.parametrize(
    "as_of,expected",
    [
        (None, 0.70),
        ("2026-10-19", 0.68),
        ("2026-10-20", 1.36),
        ("2026-10-28", 1.36),
        ("2026-10-29", 0.70),
    ],
)
def test_mistral_current_and_historical_prices(as_of, expected):
    info, _ = cloud_quote(CLOUD_MODELS["mistral-large-4"], pricing_as_of=as_of)
    assert info["effective_in_per_m"] == expected


def test_google_promotions_tiers_storage_and_implicit_cache_contract():
    carbon = CLOUD_MODELS["gemini-4-carbon"]
    for day, expected in (("2026-11-30", 2.0), ("2026-12-01", 4.0)):
        assert cloud_quote(carbon, pricing_as_of=day)[0]["effective_in_per_m"] == expected
    assert cloud_quote(carbon, cache_write_tokens=1)[0] is None
    assert cloud_quote(carbon, cache_storage_token_hours=1)[0] is None
    for key in ("gemini-pro", "gemini-flash-lite"):
        row = CLOUD_MODELS[key]
        standard = cloud_quote(row, in_len=0, cache_storage_token_hours=1e6)[1]
        batch = cloud_quote(
            row, in_len=0, cache_storage_token_hours=1e6, cloud_service_tier="batch"
        )[1]
        assert batch == standard
    pro = CLOUD_MODELS["gemini-pro"]
    assert cloud_quote(pro, hit=1)[1] == cloud_quote(pro, hit=1, cloud_service_tier="batch")[1]
    lite = CLOUD_MODELS["gemini-flash-lite"]
    assert cloud_quote(lite, hit=1, cloud_service_tier="batch")[1] == pytest.approx(
        cloud_quote(lite, hit=1)[1] / 2
    )


@pytest.mark.parametrize("ttl,expected", [("5m", 10.0), ("1h", 16.0)])
def test_cache_ttl_selects_published_write_rate(ttl, expected):
    info, _ = cloud_quote(
        CLOUD_MODELS["claude-fable-5.2"], cache_write_tokens=100, cache_write_ttl=ttl
    )
    assert info["effective_cache_write_per_m"] == expected
    assert (
        cloud_quote(
            CLOUD_MODELS["claude-fable-5.2"], cache_write_tokens=100, cache_write_ttl="12h"
        )[0]
        is None
    )
    assert (
        cloud_quote(
            CLOUD_MODELS["gpt-6.1-sol"],
            cache_write_tokens=100,
            cache_write_ttl="12h",
            in_len=272_001,
        )[0]["effective_cache_write_per_m"]
        == 10.0
    )


def test_deepseek_calendar_is_bounded_and_weekends_stay_off_peak():
    row = CLOUD_MODELS["deepseek-v4-flash"]
    assert (
        cloud_quote(row, pricing_date_utc="2026-02-23", pricing_hour_utc=2)[0]["pricing_period"]
        == "holiday_off_peak"
    )
    assert (
        cloud_quote(row, pricing_date_utc="2026-03-02", pricing_hour_utc=2)[0]["pricing_period"]
        == "peak"
    )
    assert (
        cloud_quote(row, pricing_date_utc="2026-02-14", pricing_hour_utc=2)[0]["pricing_period"]
        == "off_peak"
    )
    assert cloud_quote(row, pricing_date_utc="2027-02-01", pricing_hour_utc=2)[0][
        "holiday_calendar_unverified"
    ]


def test_exact_formats_reconcile_and_integer_artifacts_never_claim_fp4_compute():
    for (key, _), profile in MODEL_ARTIFACT_PROFILES.items():
        if profile.captured_at == "2026-11-11":
            assert _storage_bytes(profile.storage_format_counts) == profile.total_weight_bytes, (
                key,
                profile.source_repo,
            )
    profiles = artifact_profiles("nanbeige43-33b")
    assert {p.format_scheme.encoding for p in profiles} >= {"E4M3", "INT8", "INT4"}
    integer = configured("nanbeige43-33b", "nvfp4")
    assert resolve_execution(GPUS["RTX5090"], integer, "nvfp4").path == "w4a16"
    assert (
        resolve_execution(
            GPUS["RTX5090"], replace(integer, execution_backend="native"), "nvfp4"
        ).status
        == "unsupported"
    )
    mlx = configured("lfm3-vl-2b", "nvfp4", runtime="mlx", version="0.7.11")
    assert resolve_execution(GPUS["MAC_STUDIO_M4_MAX"], mlx, "nvfp4").path == "w4a16"
    assert resolve_execution(GPUS["RTX5090"], mlx, "nvfp4").status == "unsupported"
    assert (
        resolve_execution(
            GPUS["MAC_STUDIO_M4_MAX"], replace(mlx, runtime_version="0.7.10"), "nvfp4"
        ).status
        == "unsupported"
    )


def test_all_six_r9700_systems_use_the_published_210w_limit():
    six = [config for config in TINYBOX_GPU_CONFIGS.values() if config.gpu == "6XR9700"]
    assert len(six) == 24
    assert all(config.board_watts == 210 and GPUS[config.key].tdp_watts == 210 for config in six)


@pytest.mark.parametrize("encoding,precision", [("INT4", "nvfp4"), ("INT8", "fp8")])
def test_integer_artifact_labels_do_not_inherit_float_precision_slots(encoding, precision):
    from presentation.charts import _label
    from presentation.model_cards import short_artifact_label
    from state import ModelAssignment

    profile = next(
        p for p in artifact_profiles("nanbeige43-33b") if p.format_scheme.encoding == encoding
    )
    am = ModelAssignment(
        2,
        "nanbeige43-33b",
        1,
        1,
        1,
        1,
        precision,
        artifact_id=profile.artifact_id,
        execution_backend="auto",
    )
    model = am.model
    assert model.weight_precision_display(precision) == encoding
    assert short_artifact_label(model, {"key": profile.artifact_id}) == encoding
    label = _label(am, model)
    assert encoding in label and precision.upper() not in label
    assert GPUS["CrescentIsland"].bw == 1.5e12


@pytest.mark.parametrize(
    "key",
    [
        "mistral-large-4",
        "pplx-embed-v2-context-0.6b",
        "pplx-embed-v2-context-9b",
        "pplx-decider-v1.2",
        "granite-speech-5.0-470m-turboctc",
    ],
)
def test_new_catalog_entries_execute_and_roundtrip_a_public_planner_report(key):
    import copy

    from app_factory import create_test_app

    from web.agent_contract import EXAMPLE_REQUEST

    payload = copy.deepcopy(EXAMPLE_REQUEST)
    panel = payload["scenario"]["panel_a"]
    panel["gpus"] = [{"uid": 1, "gpu_type": "H200", "count": 16}]
    model = MODELS[key]
    panel["models"] = [
        {
            "model_key": key,
            "gpu_uid": 1,
            "gpu_count": 16,
            "prec": model.native_precision_key,
            "artifact_id": default_artifact_id(model, model.native_precision_key),
            "execution_backend": "auto",
            "kv_precision": "bf16",
            "recurrent_precision": "auto",
        }
    ]
    panel["projects"] = []
    with create_test_app(REQUEST_RATE_LIMIT=100_000).test_client() as client:
        response = client.post("/api/v1/calculate", json=payload)
        assert response.status_code == 200, response.get_json()
        result = response.get_json()
        diagnostic = result["results"]["panel_a"]["models"][0]
        assert diagnostic["runnable"], diagnostic
        assert model.name in result["report"]
        replay = client.post("/api/v1/calculate", json={**payload, "scenario": result["scenario"]})
        assert replay.status_code == 200
        assert replay.get_json()["report"] == result["report"]
