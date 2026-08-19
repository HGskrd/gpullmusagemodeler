"""JSON export/import formats: use-case library, project set, and A/B scenario."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict
from typing import Any, Optional

from data import (
    COUNTRIES,
    DIST_PRESETS,
    EMBEDDING_DOC_BUCKETS,
    EMBEDDING_DOC_PRESETS,
    GPUS,
    INPUT_BUCKETS,
    MODELS,
    OUTPUT_BUCKETS,
    normalize_gpu_count,
    normalize_precision,
    normalize_quality_domain,
    normalize_quality_weights,
)
from state import (
    ALLOWED_CAPABILITIES,
    DEFAULT_SCALE_KIND,
    GpuPool,
    ModelAssignment,
    PlannerState,
    Project,
    _bounded_project_value,
    _find_preset,
    _find_use_case_def,
    _next_uid,
    _normalize_scale_kind,
    _normalize_use_case_def,
    _payload_float,
    _payload_optional_float,
    _sync_aggregate_distribution,
    _sync_projects_from_use_case_defs,
    get_use_case_defs,
    normalize_auto_strategy,
    normalize_corpo_cloud,
    normalize_day_shape,
    normalize_plot_mode,
    scale_value_to_tokens,
    tokens_to_scale_value,
)


def serialize_use_case_defs(state: PlannerState) -> dict:
    return {
        "type": "gpullm-use-case-library",
        "version": 1,
        "use_cases": copy.deepcopy(get_use_case_defs(state)),
    }


def replace_use_case_defs(state: PlannerState, payload: Any) -> int:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("use_cases")
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("Use-case JSON must contain a use_cases array.")
    if len(items) > 256:
        raise ValueError("Use-case JSON may contain at most 256 definitions.")

    normalized = []
    seen = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Each use case must be a JSON object.")
        raw = (
            dict(item.get("definition", item))
            if isinstance(item.get("definition"), dict)
            else dict(item)
        )
        # Accept the earlier selected-instance export shape by folding scale into seed values.
        scale = item.get("scale") if isinstance(item.get("scale"), dict) else {}
        if "tokens_day" not in raw and "tokens_day" in scale:
            raw["tokens_day"] = scale["tokens_day"]
        if "scale_value" not in raw and "value" in scale:
            raw["scale_value"] = scale["value"]
        if "latent_jobs_day" not in raw and "latent_jobs_day" in scale:
            raw["latent_jobs_day"] = scale["latent_jobs_day"]
        raw["key"] = (
            item.get("kind_key")
            or item.get("key")
            or raw.get("key")
            or item.get("name")
            or f"use_case_{idx + 1}"
        )
        raw["name"] = item.get("name") or raw.get("name") or raw["key"]
        normalized_item = _normalize_use_case_def(raw, fallback_key=f"use_case_{idx + 1}")
        base_key = normalized_item["key"]
        if base_key in seen:
            i = 2
            while f"{base_key}_{i}" in seen:
                i += 1
            normalized_item["key"] = f"{base_key}_{i}"
        seen.add(normalized_item["key"])
        normalized.append(normalized_item)

    state.use_case_defs = normalized
    _sync_projects_from_use_case_defs(state)
    return len(normalized)


def _project_definition_payload(proj: Project) -> dict:
    return {
        "difficulty": float(proj.difficulty),
        "scale_kind": copy.deepcopy(getattr(proj, "scale_kind", DEFAULT_SCALE_KIND)),
        "wtp_per_m": float(proj.wtp_per_m),
        "requires": sorted(proj.requires),
        "min_success_rate": float(proj.min_success_rate),
        "quality_floor": float(getattr(proj, "quality_floor", 0.0)),
        "quality_domain": normalize_quality_domain(getattr(proj, "quality_domain", "general")),
        "quality_weights": normalize_quality_weights(
            getattr(proj, "quality_weights", None),
            getattr(proj, "quality_domain", "general"),
        ),
        "batch_eligible": bool(proj.batch_eligible),
        "prefix_hit_rate": float(getattr(proj, "prefix_hit_rate", 0.0)),
        "unlock_price_per_m": float(proj.unlock_price_per_m),
        "in_pre": proj.in_pre,
        "out_pre": proj.out_pre,
    }


def _project_scale_payload(proj: Project) -> dict:
    return {
        "value": float(
            getattr(
                proj,
                "scale_value",
                tokens_to_scale_value(proj.tokens_day, getattr(proj, "scale_kind", {})),
            )
        ),
        "tokens_day": float(proj.tokens_day),
        "latent_jobs_day": float(proj.latent_jobs_day),
    }


def serialize_project_set(state: PlannerState) -> dict:
    """JSON-save format for the demand side only.

    Each row keeps definition and scale separate so a saved file can contain many
    use-case kinds while each organization chooses its own deployment scale.
    """
    return {
        "type": "gpullm-use-case-set",
        "version": 1,
        "use_cases": [
            {
                "name": proj.name,
                "kind_key": getattr(proj, "kind_key", "custom"),
                "scale": _project_scale_payload(proj),
                "definition": _project_definition_payload(proj),
            }
            for proj in state.projects
        ],
    }


def _payload_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _project_from_payload(state: PlannerState, item: dict) -> Project:
    kind_key = str(item.get("kind_key") or item.get("kind") or "custom")
    preset = _find_use_case_def(state, kind_key) if kind_key != "custom" else None
    if preset is None and kind_key != "custom":
        preset = _find_preset(kind_key)
    base = preset or {
        "key": "custom",
        "name": "Custom use case",
        "difficulty": 0.3,
        "tokens_day": 500e6,
        "scale_value": 500.0,
        "scale_kind": copy.deepcopy(DEFAULT_SCALE_KIND),
        "wtp_per_m": 1.0,
        "requires": (),
        "min_success_rate": 0.85,
        "quality_floor": 0.0,
        "quality_domain": "general",
        "quality_weights": {"general": 1.0},
        "batch_eligible": False,
        "prefix_hit_rate": 0.0,
        "latent_jobs_day": 0.0,
        "unlock_price_per_m": 0.0,
        "in_pre": "Chat",
        "out_pre": "Chat",
    }
    definition = _payload_dict(item.get("definition")) or item
    scale = _payload_dict(item.get("scale")) or item
    requires_raw = definition.get("requires", base.get("requires", ()))
    if isinstance(requires_raw, str):
        requires_iter = (requires_raw,)
    else:
        requires_iter = requires_raw or ()
    requires = frozenset(c for c in requires_iter if c in ALLOWED_CAPABILITIES)
    scale_kind_source = definition if isinstance(definition.get("scale_kind"), dict) else base
    scale_kind = _normalize_scale_kind(scale_kind_source)
    scale_value = _payload_optional_float(scale, "value")
    if scale_value is None:
        scale_value = _payload_optional_float(definition, "scale_value")
    if scale_value is None:
        scale_value = tokens_to_scale_value(
            _payload_float(scale, "tokens_day", float(base.get("tokens_day", 500e6))),
            scale_kind,
        )

    proj = Project(
        uid=_next_uid(),
        name=str(item.get("name") or base["name"])[:60],
        difficulty=_bounded_project_value(
            "difficulty",
            _payload_float(definition, "difficulty", float(base["difficulty"])),
        ),
        tokens_day=_bounded_project_value(
            "tokens_day",
            scale_value_to_tokens(scale_value, scale_kind),
        ),
        wtp_per_m=_bounded_project_value(
            "wtp_per_m",
            _payload_float(definition, "wtp_per_m", float(base["wtp_per_m"])),
        ),
        scale_value=max(0.0, float(scale_value)),
        scale_kind=copy.deepcopy(scale_kind),
        kind_key=str(base["key"]) if preset else "custom",
        batch_eligible=bool(definition.get("batch_eligible", base.get("batch_eligible", False))),
        requires=requires,
        min_success_rate=_bounded_project_value(
            "min_success_rate",
            _payload_float(
                definition, "min_success_rate", float(base.get("min_success_rate", 0.85))
            ),
        ),
        quality_floor=_bounded_project_value(
            "quality_floor",
            _payload_float(definition, "quality_floor", float(base.get("quality_floor", 0.0))),
        ),
        quality_domain=normalize_quality_domain(
            definition.get("quality_domain", base.get("quality_domain", "general"))
        ),
        quality_weights=normalize_quality_weights(
            definition.get("quality_weights", base.get("quality_weights")),
            definition.get("quality_domain", base.get("quality_domain", "general")),
        ),
        prefix_hit_rate=min(
            max(
                _payload_float(
                    definition, "prefix_hit_rate", float(base.get("prefix_hit_rate", 0.0))
                ),
                0.0,
            ),
            1.0,
        ),
        latent_jobs_day=_bounded_project_value(
            "latent_jobs_day",
            _payload_float(scale, "latent_jobs_day", float(base.get("latent_jobs_day", 0.0))),
        ),
        unlock_price_per_m=_bounded_project_value(
            "unlock_price_per_m",
            _payload_float(
                definition, "unlock_price_per_m", float(base.get("unlock_price_per_m", 0.0))
            ),
        ),
        in_pre=str(definition.get("in_pre", base.get("in_pre", "Chat"))),
        out_pre=str(definition.get("out_pre", base.get("out_pre", "Chat"))),
    )
    return proj


def replace_project_set(state: PlannerState, payload: Any) -> int:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("use_cases")
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("Use-case JSON must contain a use_cases array.")
    if len(items) > 256:
        raise ValueError("Use-case JSON may contain at most 256 use cases.")

    projects = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each use case must be a JSON object.")
        projects.append(_project_from_payload(state, item))

    state.projects = projects
    _sync_aggregate_distribution(state)
    return len(projects)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_compatible(item) for item in value)
    return value


def serialize_scenario(state_a: PlannerState, state_b: Optional[PlannerState]) -> dict[str, Any]:
    """Return a versioned, complete A/B planner scenario."""
    return {
        "type": "gpullm-scenario",
        "version": 1,
        "panel_a": _json_compatible(asdict(state_a)),
        "panel_b": _json_compatible(asdict(state_b)) if state_b is not None else None,
    }


def _scenario_float(raw: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return min(max(value, lo), hi)


def _scenario_int(raw: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, lo), hi)


def _scenario_spec_method(model_key: str, raw: Any) -> str:
    method = str(raw or "off")
    if method == "off":
        return "off"
    model = MODELS.get(model_key)
    if model is None:
        return "off"
    if method == "ngram":
        # ngram needs no draft weights: available on any plain text model.
        if not model.is_realtime_only and not model.is_embedding_model:
            return "ngram"
        return "off"
    profiles = getattr(model, "speculative_profiles", ()) or ()
    if method in {getattr(p, "method", "") for p in profiles}:
        return method
    return "off"


def _scenario_dist(raw: Any, expected: int, fallback: list[int]) -> list[int]:
    if not isinstance(raw, list):
        return list(fallback)
    values = []
    for idx in range(expected):
        try:
            value = int(raw[idx]) if idx < len(raw) else 0
        except (TypeError, ValueError):
            value = 0
        values.append(min(max(value, 0), 1_000_000))
    return values if any(values) else list(fallback)


def deserialize_planner_state(payload: Any) -> PlannerState:
    """Validate and reconstruct one panel from a scenario payload."""
    if not isinstance(payload, dict):
        raise ValueError("Each scenario panel must be a JSON object.")
    state = PlannerState()

    use_case_defs = payload.get("use_case_defs")
    if isinstance(use_case_defs, list):
        if len(use_case_defs) > 256:
            raise ValueError("A scenario may contain at most 256 use-case definitions.")
        replace_use_case_defs(state, {"use_cases": use_case_defs})

    projects = payload.get("projects", [])
    if not isinstance(projects, list) or len(projects) > 256:
        raise ValueError("A scenario panel may contain at most 256 use cases.")
    replace_project_set(state, {"use_cases": projects})

    gpu_rows = payload.get("gpus", [])
    if not isinstance(gpu_rows, list) or len(gpu_rows) > 64:
        raise ValueError("A scenario panel may contain at most 64 GPU pools.")
    uid_map: dict[str, int] = {}
    state.gpus = []
    for row in gpu_rows:
        if not isinstance(row, dict) or row.get("gpu_type") not in GPUS:
            raise ValueError("Scenario contains an invalid GPU pool.")
        gpu_type = str(row["gpu_type"])
        count = normalize_gpu_count(gpu_type, _scenario_int(row, "count", 1, 1, 100_000))
        new_uid = _next_uid()
        old_uid = str(row.get("uid", new_uid))
        if old_uid in uid_map:
            raise ValueError("Scenario GPU pool identifiers must be unique.")
        uid_map[old_uid] = new_uid
        country = str(row.get("country", "FR"))
        if country not in COUNTRIES:
            country = "FR"
        state.gpus.append(
            GpuPool(
                new_uid,
                gpu_type,
                count,
                _scenario_float(
                    row,
                    "cost_per_gpu_hour",
                    GPUS[gpu_type].default_tco_per_gpu_hour,
                    0.0,
                    1_000_000.0,
                ),
                country,
            )
        )

    model_rows = payload.get("models", [])
    if not isinstance(model_rows, list) or len(model_rows) > 512:
        raise ValueError("A scenario panel may contain at most 512 model assignments.")
    state.models = []
    for row in model_rows:
        if not isinstance(row, dict) or row.get("model_key") not in MODELS:
            raise ValueError("Scenario contains an invalid model assignment.")
        gpu_uid = uid_map.get(str(row.get("gpu_uid")))
        pool = state.find_gpu(gpu_uid) if gpu_uid is not None else None
        if pool is None:
            raise ValueError("Every scenario model must reference a valid GPU pool.")
        available = max(0, pool.count - state.used_gpu_for_pool(pool.uid))
        gpu_count = min(_scenario_int(row, "gpu_count", 0, 0, pool.count), available)
        assignment = ModelAssignment(
            _next_uid(),
            str(row["model_key"]),
            pool.uid,
            gpu_count,
            _scenario_int(row, "tp", 1, 1, max(1, gpu_count)),
            _scenario_int(row, "dp", 1, 1, max(1, gpu_count)),
            normalize_precision(str(row.get("prec", "bf16"))),
            _scenario_int(row, "pp", 1, 1, max(1, gpu_count)),
            _scenario_int(row, "prefill_tp", 1, 1, max(1, gpu_count)),
            _scenario_int(row, "prefill_pp", 1, 1, max(1, gpu_count)),
            _scenario_int(row, "prefill_dp", 1, 1, max(1, gpu_count)),
            _scenario_spec_method(str(row["model_key"]), row.get("spec_method", "off")),
            _scenario_int(row, "spec_k", 0, 0, 32),
        )
        state.models.append(assignment)

    for key, default, lo, hi in (
        ("mu", 0.90, 0.01, 1.0),
        ("profiled_non_kv_gb", 4.0, 0.0, 4096.0),
        ("kv_slack", 0.02, 0.0, 1.0),
        ("moe_imbalance", 1.15, 0.1, 10.0),
        ("pd_interference", 0.0, 0.0, 1.0),
        ("spec_acceptance", 0.0, 0.0, 0.99),
        ("prefill_bw_eff", 0.80, 0.01, 1.0),
        ("prefill_comp_eff", 0.75, 0.01, 1.0),
        ("prefill_overhead", 0.08, 0.0, 1.0),
        ("prefill_paged_oh", 0.10, 0.0, 1.0),
        ("prefill_ar_overlap", 0.30, 0.0, 1.0),
        ("decode_bw_eff", 0.80, 0.01, 1.0),
        ("decode_comp_eff", 0.75, 0.01, 1.0),
        ("decode_overhead", 0.08, 0.0, 1.0),
        ("decode_paged_oh", 0.10, 0.0, 1.0),
        ("decode_ar_overlap", 0.30, 0.0, 1.0),
        ("projection_demand_level", 0.65, 0.05, 1.20),
        ("projection_night_discount", 0.30, 0.0, 0.80),
        ("projection_batch_eligible", 0.35, 0.0, 1.0),
        ("projection_elasticity", 2.0, 0.0, 4.0),
    ):
        setattr(state, key, _scenario_float(payload, key, default, lo, hi))
    state.decode_sched_budget = _scenario_int(payload, "decode_sched_budget", 16384, 1, 10_000_000)
    state.task_il = _scenario_int(payload, "task_il", 2048, 1, 10_000_000)
    state.task_ol = _scenario_int(payload, "task_ol", 32, 0, 10_000_000)
    state.in_dist = _scenario_dist(
        payload.get("in_dist"), len(INPUT_BUCKETS), list(DIST_PRESETS["Chat"]["in"])
    )
    state.out_dist = _scenario_dist(
        payload.get("out_dist"), len(OUTPUT_BUCKETS), list(DIST_PRESETS["Chat"]["out"])
    )
    state.embedding_doc_dist = _scenario_dist(
        payload.get("embedding_doc_dist"),
        len(EMBEDDING_DOC_BUCKETS),
        list(EMBEDDING_DOC_PRESETS["Doc"]),
    )
    state.in_pre = (
        str(payload.get("in_pre", "Chat")) if payload.get("in_pre") in DIST_PRESETS else ""
    )
    state.out_pre = (
        str(payload.get("out_pre", "Chat")) if payload.get("out_pre") in DIST_PRESETS else ""
    )
    state.embedding_doc_pre = (
        str(payload.get("embedding_doc_pre"))
        if payload.get("embedding_doc_pre") in EMBEDDING_DOC_PRESETS
        else ""
    )
    state.mode = normalize_plot_mode(payload.get("mode"))
    state.projection_day_shape = normalize_day_shape(payload.get("projection_day_shape"))
    state.corpo_cloud = normalize_corpo_cloud(payload.get("corpo_cloud"))
    state.projection_night_batching = bool(payload.get("projection_night_batching", False))
    state.auto_mode = bool(payload.get("auto_mode", False))
    state.auto_strategy = normalize_auto_strategy(payload.get("auto_strategy"))
    excluded = payload.get("auto_excluded", [])
    state.auto_excluded = (
        [str(key) for key in excluded if key in MODELS] if isinstance(excluded, list) else []
    )
    # Ignore the legacy user-controlled panel rate and derive the portfolio
    # value from the imported use-case priors.
    _sync_aggregate_distribution(state)
    return state


def deserialize_scenario(payload: Any) -> tuple[PlannerState, Optional[PlannerState]]:
    if not isinstance(payload, dict) or payload.get("type") != "gpullm-scenario":
        raise ValueError("Scenario JSON must have type 'gpullm-scenario'.")
    if payload.get("version") != 1:
        raise ValueError("Unsupported scenario version.")
    state_a = deserialize_planner_state(payload.get("panel_a"))
    panel_b = payload.get("panel_b")
    state_b = deserialize_planner_state(panel_b) if panel_b is not None else None
    return state_a, state_b
