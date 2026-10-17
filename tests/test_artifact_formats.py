"""Behavioral coverage for artifact identity, scenario migration, and runtime math."""

import copy
import importlib
import json
from dataclasses import replace

import pytest
from app_factory import create_test_app

from calc import (
    EfficiencyParams,
    compute_memory,
    kv_cache_bytes_for_sequence,
    model_gpu_flops,
    operator_compute_seconds,
    per_tp_linear_attention_state_bytes,
    resolve_execution,
    valid_strategies,
)
from data import GPUS, MODEL_ARTIFACT_PROFILES, MODELS, artifact_options, default_artifact_id
from data.specs import _storage_bytes
from planner_service import deserialize_scenario
from presentation.model_cards import get_model_info
from scenarios import serialize_scenario
from state import GpuPool, ModelAssignment, PlannerState, set_model_artifact


def assignment(key="nemotron35-lightning", prec="nvfp4", **settings):
    return ModelAssignment(
        2,
        key,
        1,
        1,
        1,
        1,
        prec,
        execution_backend="auto",
        kv_precision="auto",
        recurrent_precision="auto",
        **settings,
    )


def legacy_payload():
    return {
        "type": "gpullm-scenario",
        "version": 1,
        "panel_a": {
            "gpus": [{"uid": 1, "gpu_type": "H100", "count": 1}],
            "models": [
                {
                    "model_key": "qwen38-27b",
                    "gpu_uid": 1,
                    "gpu_count": 1,
                    "prec": "fp8",
                    "spec_method": "off",
                }
            ],
            "projects": [],
        },
        "panel_b": None,
    }


def test_storage_rejects_unknown_dtype_even_with_total_override():
    assert _storage_bytes({"F16": 8, "U8": 9, "F8_E8M0": 4}) == 29
    profile = next(iter(MODEL_ARTIFACT_PROFILES.values()))
    with pytest.raises(ValueError, match="Unknown checkpoint storage dtype"):
        replace(profile, storage_format_counts={"UNKNOWN": 100})
    with pytest.raises(ValueError, match="sum to one"):
        replace(profile, compute_precision_shares={"fp8": 0.1})


def test_all_existing_artifacts_are_discoverable_including_larger_casts():
    for key in ("kimi-k3", "deepseek-v4-pro", "deepseek-v4-flash"):
        rows = artifact_options(MODELS[key])
        assert rows[0]["native"]
        assert any(r["precision"] == "nvfp4" and r["key"].startswith("hf:") for r in rows)
    for (key, artifact_id), profile in MODEL_ARTIFACT_PROFILES.items():
        assert artifact_id.endswith(profile.source_revision)
        assert len(profile.source_revision) == 40
        assert any(o["key"] == artifact_id for o in artifact_options(MODELS[key]))
        assert "estimated" in profile.compute_evidence


def test_profiles_registered_during_catalog_assembly_are_indexed(monkeypatch):
    from data import artifacts
    from data.model_class import MODEL_QUANTIZATION_PROFILES

    key = "qwen38-27b"
    original = MODELS[key].quantization_profile("fp8")
    late_profile = replace(original, source_revision="e" * 40)
    identity = (key, late_profile.artifact_id)
    monkeypatch.setitem(MODEL_QUANTIZATION_PROFILES, (key, "fp8"), late_profile)
    # Track index cleanup even though catalog assembly creates the index entry.
    monkeypatch.setitem(MODEL_ARTIFACT_PROFILES, identity, late_profile)
    del MODEL_ARTIFACT_PROFILES[identity]
    importlib.reload(artifacts)
    assert artifacts.artifact_precision(MODELS[key], late_profile.artifact_id) == "fp8"
    assert any(row["key"] == late_profile.artifact_id for row in artifact_options(MODELS[key]))


def test_multiple_same_precision_artifacts_keep_distinct_memory(monkeypatch):
    key = "qwen38-27b"
    original = MODELS[key].quantization_profile("fp8")
    alternate = replace(
        original,
        source_revision="f" * 40,
        total_weight_bytes_override=original.total_weight_bytes + 1e9,
    )
    monkeypatch.setitem(MODEL_ARTIFACT_PROFILES, (key, alternate.artifact_id), alternate)
    first = assignment(key, "fp8").model
    second = assignment(key, "fp8", artifact_id=alternate.artifact_id).model
    assert second.weight_bytes("fp8") - first.weight_bytes("fp8") == pytest.approx(1e9)
    assert first.key == second.key == key
    assert sum(o["precision"] == "fp8" for o in artifact_options(MODELS[key])) == 2
    rows = [o for o in artifact_options(MODELS[key]) if o["precision"] == "fp8"]
    assert len({o["label"] for o in rows}) == 2


def test_v1_migration_keeps_precision_topology_and_legacy_model():
    state, _ = deserialize_scenario(legacy_payload())
    am = state.models[0]
    assert am.prec == "fp8"
    assert am.execution_backend == am.kv_precision == am.recurrent_precision == "legacy"
    assert am.model is MODELS[am.model_key]
    assert (am.tp, am.pp, am.dp) == (1, 1, 1)
    exported = serialize_scenario(state, None)
    assert exported["version"] == 2
    replay, _ = deserialize_scenario(exported)
    assert replay.models[0].model is MODELS[am.model_key]
    assert replay.models[0].artifact_id == am.artifact_id


def test_v2_roundtrip_preserves_ab_artifact_runtime_and_cache_choices():
    state = PlannerState(
        gpus=[GpuPool(1, "H100", 1)],
        models=[assignment(serving_runtime="vllm", runtime_version="0.27.1")],
    )
    compare = copy.deepcopy(state)
    compare.models[0].execution_backend = "expanded"
    compare.models[0].kv_precision = "bf16"
    exported = serialize_scenario(state, compare)
    a, b = deserialize_scenario(exported)
    assert a.models[0].artifact_id == state.models[0].artifact_id
    assert a.models[0].serving_runtime == "vllm"
    assert a.models[0].runtime_version == "0.27.1"
    assert b.models[0].execution_backend == "expanded"
    assert b.models[0].kv_precision == "bf16"
    assert a.models[0].spec_method == "off"


@pytest.mark.parametrize(
    "change",
    [
        {"artifact_id": "hf:missing@bad"},
        {"artifact_id": default_artifact_id(MODELS["nemotron35-lightning"], "nvfp4")},
        {"execution_backend": "invented"},
        {"kv_precision": "int4"},
        {"serving_runtime": "invented"},
    ],
)
def test_import_rejects_invalid_or_mismatched_choices(change):
    payload = legacy_payload()
    payload["version"] = 2
    payload["panel_a"]["models"][0].update(change)
    with pytest.raises(ValueError):
        deserialize_scenario(payload)


def test_independent_kv_and_recurrent_formats_do_not_share_cached_results():
    am = assignment("qwen38-27b", "fp8")
    m = am.model
    fp8 = kv_cache_bytes_for_sequence(m, 4096, "fp8")
    state_fp32 = per_tp_linear_attention_state_bytes(m, "fp8", 1)
    am.kv_precision = "bf16"
    assert kv_cache_bytes_for_sequence(am.model, 4096, "fp8") == pytest.approx(fp8 * 2)
    assert per_tp_linear_attention_state_bytes(am.model, "fp8", 1) == state_fp32
    am.recurrent_precision = "bf16"
    state_bf16 = per_tp_linear_attention_state_bytes(am.model, "fp8", 1)
    assert state_bf16 < state_fp32 < state_bf16 * 2
    assert am.model.key == m.key


def test_fixed_architecture_cache_layout_ignores_generic_kv_selector():
    am = assignment("deepseek-v4.1-flash", "mxfp4")
    before = kv_cache_bytes_for_sequence(am.model, 8192, "mxfp4")
    am.kv_precision = "bf16"
    assert kv_cache_bytes_for_sequence(am.model, 8192, "mxfp4") == before


def test_weight_expansion_changes_gpu_residency_not_checkpoint_or_host_bytes():
    am = assignment("deepseek-v4.1-flash", "mxfp4")
    before = am.model.weight_memory("mxfp4")
    am.execution_backend = "expanded"
    after = am.model.weight_memory("mxfp4")
    assert after.checkpoint_bytes == before.checkpoint_bytes
    assert after.host_bytes == before.host_bytes == pytest.approx(202.758e9, rel=1e-5)
    assert after.gpu_bytes > before.gpu_bytes
    assert after.repacking_bytes == pytest.approx(after.gpu_bytes - before.gpu_bytes)
    assert after.loading_workspace_bytes is None


def test_native_unsupported_path_has_no_topology_or_capacity():
    am = assignment()
    am.execution_backend = "native"
    model = am.model
    gpu = GPUS["H100"]
    assert resolve_execution(gpu, model, am.prec).status == "unsupported"
    assert not valid_strategies(model, 1, gpu, 0.9, 4, am.prec)
    assert compute_memory(model, 1, 1, gpu, 0.9, 4, am.prec, EfficiencyParams()) is None


def test_runtime_recipe_and_operator_compute_do_not_claim_fp4_peak_for_w4a16():
    am = assignment(serving_runtime="vllm", runtime_version="0.27.1")
    gpu = GPUS["DGX_SPARK"]
    resolved = resolve_execution(gpu, am.model, am.prec)
    assert resolved.path == "w4a16"
    assert resolved.recipe_source.startswith("https://huggingface.co/nvidia/")
    assert model_gpu_flops(gpu, am.model, am.prec) <= gpu.fp8
    am.execution_backend = "native"
    native = am.model
    rate = model_gpu_flops(gpu, native, am.prec)
    expected = 1e12 / rate + 2e12 / gpu.bf16
    assert operator_compute_seconds(gpu, native, am.prec, 1e12, 2e12) == pytest.approx(expected)
    assert expected > 3e12 / rate


def test_active_traffic_keeps_kimi_geometry_and_phase_bound():
    am = assignment("kimi-k3", "mxfp4")
    assert am.model.active_weight_bytes(am.prec) == pytest.approx(136.8780941525496e9)
    am = assignment("deepseek-v4.1-flash", "mxfp4")
    assert am.model.compute_shares(am.prec, "prefill") == {"bf16": 1.0}
    assert am.model.active_weight_bytes(am.prec, "prefill") == pytest.approx(
        am.model.prefill_active_params * 2
    )


def test_artifact_mutation_invalidates_cached_card_and_keeps_failed_changes_atomic():
    am = ModelAssignment(2, "qwen38-27b", 1, 1, 1, 1, "fp8")
    state = PlannerState(gpus=[GpuPool(1, "H100", 1)], models=[am])
    before = get_model_info(state, am)
    revision = state.revision
    set_model_artifact(state, am.uid, am.artifact_id, kv="bf16")
    after = get_model_info(state, am)
    assert state.revision > revision
    assert before["kv_format"] != after["kv_format"]
    current = (am.artifact_id, am.prec, am.execution_backend)
    with pytest.raises(ValueError):
        set_model_artifact(state, am.uid, "hf:not-real")
    assert (am.artifact_id, am.prec, am.execution_backend) == current


def test_browser_routes_and_agent_api_roundtrip_artifact_settings():
    client = create_test_app(REQUEST_RATE_LIMIT=100_000).test_client()
    payload = legacy_payload()
    assert client.post("/scenario/import", data={"json": json.dumps(payload)}).status_code == 200
    before = client.get("/scenario/export").get_json()
    row = before["panel_a"]["models"][0]
    response = client.post(
        "/model/artifact",
        data={
            "panel": "A",
            "uid": row["uid"],
            "artifact_id": row["artifact_id"],
            "execution_backend": "auto",
            "kv_precision": "bf16",
            "recurrent_precision": "fp32",
            "serving_runtime": "vllm",
            "runtime_version": "0.27.1",
        },
    )
    assert response.status_code == 200
    assert "Artifact for Qwen" in response.text
    # A subsequent artifact-only picker request must preserve independent settings.
    assert (
        client.post(
            "/model/artifact",
            data={"panel": "A", "uid": row["uid"], "artifact_id": row["artifact_id"]},
        ).status_code
        == 200
    )
    exported = client.get("/scenario/export").get_json()
    replay = exported["panel_a"]["models"][0]
    assert replay["serving_runtime"] == "vllm" and replay["runtime_version"] == "0.27.1"
    assert replay["recurrent_precision"] == "fp32"
    assert exported["panel_a"]["models"][0]["kv_precision"] == "bf16"
    result = client.post("/api/v1/calculate", json={"scenario": exported, "include": ["report"]})
    assert result.status_code == 200
    model = result.get_json()["results"]["panel_a"]["models"][0]
    assert model["artifact_id"] == row["artifact_id"]
    assert model["execution"]["runtime"] == "vllm"
    assert "Artifact for" in result.get_json()["report"]
    invalid = client.post(
        "/model/artifact", data={"panel": "A", "uid": row["uid"], "artifact_id": "bad"}
    )
    assert invalid.status_code == 400


def test_v1_prefill_keeps_the_original_mixed_compute_assumption():
    model = MODELS["deepseek-v4.1-flash"]
    assert model.compute_shares("mxfp4", "prefill") == model.compute_shares("mxfp4", "decode")
    assert model_gpu_flops(GPUS["B300"], model, "mxfp4", "prefill") == model_gpu_flops(
        GPUS["B300"], model, "mxfp4", "decode"
    )
