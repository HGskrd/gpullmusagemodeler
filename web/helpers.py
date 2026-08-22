"""Request-scoped helpers shared by every planner blueprint.

Form coercion, planner-state lookup, the HTMX response envelope and the Jinja
context. Blueprints import from here; nothing here imports a blueprint, which
keeps ``app.py`` off the import path of the route modules.
"""

import json
import math
from functools import wraps

from flask import (
    current_app,
    make_response,
    render_template,
    request,
)

import cloud_policy
from calc import (
    avg_dist,
    dist_percentile,
    effective_prefill_length,
    normalize_dist,
    strategy_label,
)
from data import (
    BATCH_SIZES,
    CAPABILITY_LABELS,
    COUNTRIES,
    DAY_SHAPES,
    DIST_PRESETS,
    EMBEDDING_DOC_BUCKETS,
    EMBEDDING_DOC_PRESETS,
    GPUS,
    INPUT_BUCKETS,
    MODEL_CAPABILITIES,
    MODEL_DOMAIN_QUALITY_ANCHORS,
    MODELS,
    OUTPUT_BUCKETS,
    PRECISION_DESCRIPTIONS,
    PRECISION_LABELS,
    PRECISIONS,
    PROJECT_PRESETS,
    QUALITY_DOMAIN_LABELS,
    SCALE_MODELS,
    TASK_PRESETS,
    effective_quality,
    gpu_cards_by_vendor,
    gpus_by_vendor,
    models_by_category,
    quality_to_aa_intelligence,
    quality_weights_label,
    required_quality,
)
from presentation.econ import econ_payload
from presentation.model_cards import (
    get_model_info,
    get_model_infos,
)
from state import (
    AUTO_MODEL_STRATEGIES,
    VISIBLE_PLOT_MODES,
    PlannerState,
    _normalize_use_case_def,
    format_scale_value,
    project_scale_config,
    scale_decimals,
)
from use_case_evidence import (
    USE_CASE_DETAILS,
    USE_CASE_RESEARCH_CAPTURED_AT,
    USE_CASE_SOURCES,
)
from web.config import (
    MAX_IMPORT_BYTES,
)
from web.middleware import (
    _scope_id,
    _tab_id,
    _visitor_id,
)
from web.session_store import (
    get_compare_state,
    get_state,
)


def _finite_float(raw, *, name: str, lo: float, hi: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo:g} and {hi:g}.")
    return value


def _bounded_int(raw, *, name: str, lo: int, hi: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer.") from None
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}.")
    return value


def _form_int(field: str, default: int | None = None) -> int:
    """Read an integer form field.

    Raises ValueError (rendered as 400) rather than letting a bare int(None)
    surface as a 500 with the interpreter's message in the response body.
    """
    raw = request.form.get(field)
    if raw is None or raw == "":
        if default is not None:
            return default
        raise ValueError(f"{field} is required.")
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer.") from None


def _form_float(field: str, default: float | None = None) -> float:
    """Read a finite float form field, raising ValueError (400) when malformed."""
    raw = request.form.get(field)
    if raw is None or raw == "":
        if default is not None:
            return default
        raise ValueError(f"{field} is required.")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number.") from None
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    return value


def _load_import_json(raw: str) -> object:
    if len(raw.encode("utf-8")) > MAX_IMPORT_BYTES:
        raise ValueError(f"Imported JSON exceeds the {MAX_IMPORT_BYTES}-byte limit.")

    def reject_constant(value: str):
        raise ValueError(f"Non-finite number {value} is not valid scenario data.")

    return json.loads(raw, parse_constant=reject_constant)


def use_case_detail_for(definition: dict) -> dict:
    """Return evidence only while a definition still matches its built-in baseline."""
    key = str(definition.get("key", ""))
    baseline = next((item for item in PROJECT_PRESETS if item["key"] == key), None)
    if baseline is None or definition != _normalize_use_case_def(baseline):
        return {}
    return USE_CASE_DETAILS.get(key, {})


def _state(panel: str = "A") -> PlannerState | None:
    if panel == "B":
        return get_compare_state(_scope_id())
    return get_state(_scope_id())


def _request_state() -> PlannerState | None:
    return _state(request.values.get("panel", "A"))


def with_planner_state(view):
    """Pass the request's planner panel to the view, or re-render if it is gone.

    Replaces the preamble every mutation handler repeated verbatim::

        s = _request_state()
        if s is None:
            return _htmx_response()

    A missing panel means the scope was pruned or reset between the page load
    and this request; re-rendering the canonical pair is the existing recovery
    and is preserved exactly.
    """

    @wraps(view)
    def wrapped(*args, **kwargs):
        state = _request_state()
        if state is None:
            return _htmx_response()
        return view(state, *args, **kwargs)

    return wrapped


def _template_context() -> dict:
    return {
        "GPUS": GPUS,
        "MODELS": MODELS,
        "DIST_PRESETS": DIST_PRESETS,
        "EMBEDDING_DOC_BUCKETS": EMBEDDING_DOC_BUCKETS,
        "EMBEDDING_DOC_PRESETS": EMBEDDING_DOC_PRESETS,
        "TASK_PRESETS": TASK_PRESETS,
        "DAY_SHAPES": DAY_SHAPES,
        "CLOUD_MODELS": cloud_policy.effective_catalog(),
        "CORPO_CLOUD_PRESETS": cloud_policy.corpo_presets(),
        "CLOUD_POLICY": cloud_policy.summary(),
        "PROJECT_PRESETS": PROJECT_PRESETS,
        "MODEL_CAPABILITIES": MODEL_CAPABILITIES,
        "CAPABILITY_LABELS": CAPABILITY_LABELS,
        "QUALITY_DOMAIN_LABELS": QUALITY_DOMAIN_LABELS,
        "MODEL_DOMAIN_QUALITY_ANCHORS": MODEL_DOMAIN_QUALITY_ANCHORS,
        "quality_weights_label": quality_weights_label,
        "SCALE_MODELS": SCALE_MODELS,
        "COUNTRIES": COUNTRIES,
        "VISIBLE_PLOT_MODES": VISIBLE_PLOT_MODES,
        "INPUT_BUCKETS": INPUT_BUCKETS,
        "OUTPUT_BUCKETS": OUTPUT_BUCKETS,
        "BATCH_SIZES": BATCH_SIZES,
        "PRECISIONS": PRECISIONS,
        "PRECISION_LABELS": PRECISION_LABELS,
        "PRECISION_DESCRIPTIONS": PRECISION_DESCRIPTIONS,
        "AUTO_MODEL_STRATEGIES": AUTO_MODEL_STRATEGIES,
        "USE_CASE_RESEARCH_CAPTURED_AT": USE_CASE_RESEARCH_CAPTURED_AT,
        "USE_CASE_SOURCES": USE_CASE_SOURCES,
        "use_case_detail_for": use_case_detail_for,
        "models_by_category": models_by_category,
        "gpu_cards_by_vendor": gpu_cards_by_vendor,
        "gpus_by_vendor": gpus_by_vendor,
        "normalize_dist": normalize_dist,
        "avg_dist": avg_dist,
        "dist_percentile": dist_percentile,
        "get_model_info": get_model_info,
        "get_model_infos": get_model_infos,
        "effective_prefill_length": effective_prefill_length,
        "strategy_label": strategy_label,
        "required_quality": required_quality,
        "effective_quality": effective_quality,
        "quality_to_aa_intelligence": quality_to_aa_intelligence,
        "project_scale_config": project_scale_config,
        "format_scale_value": format_scale_value,
        "scale_decimals": scale_decimals,
        "math": math,
    }


def _planner_view_context(state_a: PlannerState, state_b: PlannerState | None) -> dict:
    """Build controller-owned view models before rendering planner templates."""
    return _template_context() | {
        "ECON_A": econ_payload(state_a),
        "ECON_B": econ_payload(state_b) if state_b is not None else None,
    }


def _record_snapshot(
    reason: str, state_a: PlannerState, state_b: PlannerState | None, path: str | None = None
):
    if not current_app.config["TRACKING_ENABLED"]:
        return
    current_app.extensions["snapshot_store"].record_snapshot(
        visitor_id=_visitor_id(),
        tab_id=_tab_id() or "default",
        reason=reason,
        path=path or request.path,
        state_a=state_a,
        state_b=state_b,
    )


def _htmx_response(state_a=None):
    # Always render the canonical A/B pair. Callers may pass the mutated panel,
    # which is panel B for compare-side edits.
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    resp = make_response(
        render_template("partials/htmx_response.html", A=sa, B=sb, **_planner_view_context(sa, sb))
    )
    resp.headers["HX-Trigger"] = "refreshChart"
    return resp


def _tracked_htmx_response(reason: str, state_a: PlannerState | None = None):
    # Always render and snapshot the canonical A/B pair. Callers may pass the
    # mutated panel, which is panel B for compare-side edits.
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    _record_snapshot(reason, sa, sb)
    resp = make_response(
        render_template("partials/htmx_response.html", A=sa, B=sb, **_planner_view_context(sa, sb))
    )
    resp.headers["HX-Trigger"] = "refreshChart"
    return resp
