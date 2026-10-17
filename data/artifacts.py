"""Artifact discovery and validated, backwards-compatible deployment choices."""

from dataclasses import replace
from functools import lru_cache

from .model_class import MODEL_ARTIFACT_PROFILES, MODEL_QUANTIZATION_PROFILES, Model
from .models import MODELS
from .specs import (
    CACHE_PRECISIONS,
    EXECUTION_BACKENDS,
    PRECISION_SPECS,
    PRECISIONS,
    RECURRENT_PRECISIONS,
    SERVING_RUNTIMES,
)

# Family modules can register checkpoint profiles while MODELS is assembled.
# Index those after catalog imports, preserving additional same-precision artifacts.
for (_model_key, _precision), _profile in MODEL_QUANTIZATION_PROFILES.items():
    MODEL_ARTIFACT_PROFILES.setdefault((_model_key, _profile.artifact_id), _profile)


def artifact_profiles(model_key: str):
    return tuple(p for (key, _), p in MODEL_ARTIFACT_PROFILES.items() if key == model_key)


def default_artifact_id(model: Model, prec: str) -> str:
    profile = model.quantization_profile(prec)
    if profile:
        return profile.artifact_id
    return f"native:{prec}" if prec == model.native_precision_key else f"estimate:{prec}"


def artifact_precision(model: Model, artifact_id: str) -> str:
    profile = MODEL_ARTIFACT_PROFILES.get((model.key, artifact_id))
    if profile:
        return profile.precision_key
    if artifact_id == f"native:{model.native_precision_key}":
        return model.native_precision_key
    if artifact_id.startswith("estimate:") and artifact_id[9:] in PRECISIONS:
        return artifact_id[9:]
    raise ValueError(
        "Unknown artifact for this model; select a catalog artifact or explicit estimate."
    )


def validate_settings(execution: str, kv: str, recurrent: str) -> None:
    if (
        execution not in EXECUTION_BACKENDS
        or kv not in CACHE_PRECISIONS
        or recurrent not in RECURRENT_PRECISIONS
    ):
        raise ValueError("Invalid execution or cache format setting.")


def validate_runtime(runtime: str, version: str) -> None:
    if runtime not in SERVING_RUNTIMES or len(version) > 64 or any(ord(c) < 32 for c in version):
        raise ValueError("Invalid serving runtime or version.")


def configured_model(
    model_key: str,
    artifact_id: str,
    execution: str,
    kv: str,
    recurrent: str,
    serving_runtime: str = "generic",
    runtime_version: str = "",
) -> Model:
    model = MODELS[model_key]
    validate_settings(execution, kv, recurrent)
    validate_runtime(serving_runtime, runtime_version)
    if artifact_id:
        artifact_precision(model, artifact_id)
    # Exact legacy behavior (including existing memory/throughput assumptions).
    if execution == kv == recurrent == "legacy" and artifact_id in {
        "",
        default_artifact_id(model, artifact_precision(model, artifact_id)) if artifact_id else "",
    }:
        return model
    return _configured_model(
        model_key,
        id(model),
        artifact_id,
        execution,
        kv,
        recurrent,
        serving_runtime,
        runtime_version,
    )


@lru_cache(maxsize=4096)
def _configured_model(
    model_key: str,
    catalog_identity: int,
    artifact_id: str,
    execution: str,
    kv: str,
    recurrent: str,
    serving_runtime: str,
    runtime_version: str,
) -> Model:
    # Catalog identity also prevents stale configured copies when tests or a
    # future catalog reload replace an immutable Model at the same key.
    model = MODELS[model_key]
    return replace(
        model,
        selected_artifact_id=artifact_id,
        execution_backend=execution,
        cache_precision=kv,
        recurrent_precision=recurrent,
        serving_runtime=serving_runtime,
        runtime_version=runtime_version,
    )


def artifact_options(model: Model) -> list[dict]:
    base = MODELS[model.key]
    native = default_artifact_id(base, base.native_precision_key)
    options: list[dict] = []
    profiles = artifact_profiles(base.key)
    native_profile = next((p for p in profiles if p.artifact_id == native), None)
    if native_profile is None:
        options.append(
            {
                "key": native,
                "precision": base.native_precision_key,
                "label": f"Native · {base.native_precision_display}",
                "description": base.native_precision_description,
                "checkpoint_bytes": base.weight_bytes(base.native_precision_key),
                "evidence": "catalog storage estimate",
                "native": True,
            }
        )
    for profile in profiles:
        is_native = profile.artifact_id == native
        options.append(
            {
                "key": profile.artifact_id,
                "precision": profile.precision_key,
                "label": f"Native · {base.native_precision_display}"
                if is_native
                else f"{profile.label} · artifact",
                "description": f"{profile.source_repo}@{profile.source_revision[:12]}. {profile.notes}",
                "checkpoint_bytes": profile.total_weight_bytes,
                "evidence": profile.storage_evidence,
                "evidence_details": {
                    "storage": profile.storage_evidence,
                    "active_traffic": profile.active_traffic_evidence,
                    "compute": profile.compute_evidence,
                    "runtime": profile.runtime_evidence,
                },
                "source": {
                    "repo": profile.source_repo,
                    "revision": profile.source_revision,
                    "captured_at": profile.captured_at,
                },
                "container": profile.container,
                "scheme": {
                    "encoding": profile.format_scheme.encoding,
                    "block_shape": profile.format_scheme.block_shape,
                    "scale_format": profile.format_scheme.scale_format,
                    "zero_point_format": profile.format_scheme.zero_point_format,
                },
                "quantized": profile.quantized,
                "retained": profile.retained,
                "native": is_native,
            }
        )
    for prec in PRECISIONS:
        if prec == base.native_precision_key or any(p.precision_key == prec for p in profiles):
            continue
        options.append(
            {
                "key": f"estimate:{prec}",
                "precision": prec,
                "label": f"{PRECISION_SPECS[prec].label} · estimated conversion",
                "description": "Generic conversion assumption; no released artifact is pinned.",
                "checkpoint_bytes": base.weight_bytes(prec),
                "evidence": "generic estimate",
                "native": False,
            }
        )
    selected = model.selected_artifact_id
    if selected.startswith("estimate:") and not any(row["key"] == selected for row in options):
        prec = artifact_precision(base, selected)
        options.append(
            {
                "key": selected,
                "precision": prec,
                "label": f"{PRECISION_SPECS[prec].label} · estimated conversion",
                "description": "Explicit generic conversion assumption; not a pinned artifact.",
                "checkpoint_bytes": model.weight_bytes(prec),
                "evidence": "generic estimate",
                "native": False,
            }
        )
    for profile in profiles:
        if sum(p.precision_key == profile.precision_key for p in profiles) > 1:
            row = next(o for o in options if o["key"] == profile.artifact_id)
            row["label"] += f" · {profile.source_repo} @{profile.source_revision[:7]}"
    return sorted(
        options,
        key=lambda row: (
            not row["native"],
            row["key"].startswith("estimate:"),
            -row["checkpoint_bytes"],
            row["key"],
        ),
    )
