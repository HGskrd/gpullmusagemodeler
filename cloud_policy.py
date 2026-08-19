"""Deployment-level corporate cloud policy.

Corporate deployments rarely expose the whole public cloud catalog: procurement
picks a subset of gateway models, often at negotiated rates that differ from the
published sticker prices. This module loads an optional JSON policy file (env
PLANNER_CLOUD_POLICY) that restricts the routable cloud catalog and overrides
per-model prices without touching the public catalog in data.py.

Policy file schema (all sections optional):

    {
      "allowed_models": ["gemini-pro", "claude-sonnet"],       // absent/null = whole catalog
      "price_overrides": {                                     // negotiated rates, $/1M tokens
        "gemini-pro": {"in_per_m": 1.0, "cached_in_per_m": 0.10, "out_per_m": 8.0}
      },
      "corpo_presets": {                                       // extra selectable gateways
        "acme": {"label": "ACME gateway", "models": ["gemini-pro", "claude-sonnet"]}
      }
    }

The file is loaded once at startup; invalid policies fail fast with a clear
error instead of silently mis-pricing routing decisions. Restart to apply
changes.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from data import CLOUD_MODELS, CORPO_CLOUD_PRESETS

POLICY_ENV_VAR = "PLANNER_CLOUD_POLICY"

_OVERRIDABLE_PRICE_FIELDS = frozenset({"in_per_m", "cached_in_per_m", "out_per_m"})


@dataclass(frozen=True)
class CloudPolicy:
    """Validated deployment policy. allowed_models None means no restriction."""

    allowed_models: frozenset[str] | None = None
    price_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    corpo_presets: dict[str, dict] = field(default_factory=dict)


_POLICY: CloudPolicy | None = None


def _validate_price(key: str, field_name: str, value) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"cloud policy: price override {key}.{field_name} must be a finite number >= 0, got {value!r}"
        )
    return float(value)


def validate_policy(raw: dict) -> CloudPolicy:
    """Validate a policy document, failing fast on anything that would mis-price routing."""
    if not isinstance(raw, dict):
        raise ValueError("cloud policy: top level must be a JSON object")
    unknown_sections = set(raw) - {"allowed_models", "price_overrides", "corpo_presets"}
    if unknown_sections:
        raise ValueError(f"cloud policy: unknown sections {sorted(unknown_sections)}")

    allowed = raw.get("allowed_models")
    if allowed is not None:
        if not isinstance(allowed, list) or not all(isinstance(k, str) for k in allowed):
            raise ValueError("cloud policy: allowed_models must be a list of model keys")
        unknown = sorted(set(allowed) - set(CLOUD_MODELS))
        if unknown:
            raise ValueError(
                f"cloud policy: allowed_models references unknown catalog keys {unknown}"
            )
        if not allowed:
            raise ValueError(
                "cloud policy: allowed_models is empty; omit it to allow the whole catalog"
            )
        allowed = frozenset(allowed)

    raw_overrides = raw.get("price_overrides")
    if raw_overrides is None:
        raw_overrides = {}
    if not isinstance(raw_overrides, dict):
        raise ValueError("cloud policy: price_overrides must be a JSON object")
    overrides: dict[str, dict[str, float]] = {}
    for key, fields in raw_overrides.items():
        if key not in CLOUD_MODELS:
            raise ValueError(f"cloud policy: price override for unknown catalog key {key!r}")
        if not isinstance(fields, dict):
            raise ValueError(f"cloud policy: price override {key!r} must be an object")
        unknown_fields = sorted(set(fields) - _OVERRIDABLE_PRICE_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"cloud policy: price override {key!r} has unknown fields {unknown_fields}; "
                f"allowed: {sorted(_OVERRIDABLE_PRICE_FIELDS)}"
            )
        if not fields:
            raise ValueError(f"cloud policy: price override {key!r} is empty")
        overrides[key] = {f: _validate_price(key, f, v) for f, v in fields.items()}
    if allowed is not None:
        for key in overrides:
            if key not in allowed:
                raise ValueError(f"cloud policy: price override {key!r} is not in allowed_models")

    raw_presets = raw.get("corpo_presets")
    if raw_presets is None:
        raw_presets = {}
    if not isinstance(raw_presets, dict):
        raise ValueError("cloud policy: corpo_presets must be a JSON object")
    presets: dict[str, dict] = {}
    for preset_key, preset in raw_presets.items():
        if not isinstance(preset_key, str) or not preset_key.strip():
            raise ValueError("cloud policy: corpo preset keys must be non-empty strings")
        if preset_key in CORPO_CLOUD_PRESETS:
            raise ValueError(
                f"cloud policy: corpo preset {preset_key!r} collides with a built-in preset"
            )
        if (
            not isinstance(preset, dict)
            or not isinstance(preset.get("label"), str)
            or not preset["label"].strip()
        ):
            raise ValueError(f"cloud policy: corpo preset {preset_key!r} needs a non-empty label")
        models = preset.get("models")
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(k, str) for k in models)
        ):
            raise ValueError(
                f"cloud policy: corpo preset {preset_key!r} needs a non-empty models list"
            )
        unknown = sorted(set(models) - set(CLOUD_MODELS))
        if unknown:
            raise ValueError(
                f"cloud policy: corpo preset {preset_key!r} references unknown catalog keys {unknown}"
            )
        if allowed is not None:
            blocked = sorted(set(models) - set(allowed))
            if blocked:
                raise ValueError(
                    f"cloud policy: corpo preset {preset_key!r} references models outside "
                    f"allowed_models {blocked}"
                )
        presets[preset_key] = {"label": preset["label"].strip(), "models": tuple(models)}

    return CloudPolicy(allowed_models=allowed, price_overrides=overrides, corpo_presets=presets)


def configure(policy: CloudPolicy | None) -> None:
    """Install the validated policy (None clears it). Called once at startup and by tests."""
    global _POLICY
    _POLICY = policy


def configure_from_env() -> CloudPolicy | None:
    """Load and install the policy named by PLANNER_CLOUD_POLICY. No-op when unset."""
    path = os.environ.get(POLICY_ENV_VAR, "").strip()
    if not path:
        configure(None)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError as exc:
        raise RuntimeError(
            f"{POLICY_ENV_VAR}={path}: cannot read cloud policy file: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{POLICY_ENV_VAR}={path}: invalid JSON: {exc}") from exc
    try:
        policy = validate_policy(raw)
    except ValueError as exc:
        raise RuntimeError(f"{POLICY_ENV_VAR}={path}: {exc}") from exc
    configure(policy)
    return policy


def policy_active() -> bool:
    return _POLICY is not None


def is_allowed(key: str) -> bool:
    if _POLICY is None or _POLICY.allowed_models is None:
        return key in CLOUD_MODELS
    return key in _POLICY.allowed_models


def effective_entry(key: str) -> dict | None:
    """Catalog entry with policy price overrides merged in; None when policy blocks the model.

    Overridden entries carry price_source='override' so UI and reports can mark them."""
    if not is_allowed(key):
        return None
    entry = dict(CLOUD_MODELS[key])
    if _POLICY is not None and key in _POLICY.price_overrides:
        entry.update(_POLICY.price_overrides[key])
        entry["price_source"] = "override"
    else:
        entry["price_source"] = "catalog"
    return entry


def corpo_presets() -> dict:
    """Built-in corpo gateway presets plus any policy-defined custom presets."""
    if _POLICY is None or not _POLICY.corpo_presets:
        return CORPO_CLOUD_PRESETS
    return {**CORPO_CLOUD_PRESETS, **_POLICY.corpo_presets}


def effective_catalog() -> dict[str, dict]:
    """All policy-allowed catalog entries with negotiated prices applied."""
    return {key: entry for key in CLOUD_MODELS if (entry := effective_entry(key)) is not None}


def effective_corpo_models(preset_name: str) -> list[tuple[str, dict]]:
    """(key, effective entry) pairs for a corpo preset, restricted by the policy allowlist."""
    presets = corpo_presets()
    preset = presets.get(preset_name) or presets.get("current")
    if preset is None:
        return []
    pairs = []
    for key in preset["models"]:
        entry = effective_entry(key)
        if entry is not None:
            pairs.append((key, entry))
    return pairs


def summary() -> dict:
    """Policy status for UI badges and the projection report."""
    if _POLICY is None:
        return {
            "active": False,
            "allowed_count": len(CLOUD_MODELS),
            "total_count": len(CLOUD_MODELS),
            "overridden": [],
            "custom_presets": [],
        }
    allowed = _POLICY.allowed_models
    return {
        "active": True,
        "allowed_count": len(allowed) if allowed is not None else len(CLOUD_MODELS),
        "total_count": len(CLOUD_MODELS),
        "overridden": sorted(_POLICY.price_overrides),
        "custom_presets": sorted(_POLICY.corpo_presets),
    }
