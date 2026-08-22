"""Use-case library routes: the definition editor and its import/export."""

import json

from flask import (
    Blueprint,
    jsonify,
    make_response,
    render_template,
    request,
)

from data import (
    aa_intelligence_to_quality,
)
from placement import (
    retune_models,
)
from scenarios import (
    replace_use_case_defs,
    serialize_use_case_defs,
)
from state import (
    add_use_case_def,
    get_use_case_defs,
    remove_use_case_def,
    set_use_case_def_capability,
    set_use_case_def_field,
)
from use_case_evidence import (
    USE_CASE_DETAILS,
)
from web.helpers import (
    _load_import_json,
    _record_snapshot,
    _template_context,
)
from web.middleware import (
    _scope_id,
)
from web.session_store import (
    get_compare_state,
    get_state,
)

use_cases_bp = Blueprint("use_cases", __name__)


@use_cases_bp.route("/use-cases")
def use_cases():
    s = get_state(_scope_id())
    return render_template(
        "use_cases.html",
        state=s,
        USE_CASE_DEFS=get_use_case_defs(s),
        use_case_details=USE_CASE_DETAILS,
        **_template_context(),
    )


def _use_case_library_response(reason: str | None = None):
    s = get_state(_scope_id())
    if reason:
        _record_snapshot(reason, s, get_compare_state(_scope_id()), path="/use-cases")
    return render_template(
        "partials/use_case_library.html",
        state=s,
        USE_CASE_DEFS=get_use_case_defs(s),
        use_case_details=USE_CASE_DETAILS,
        **_template_context(),
    )


@use_cases_bp.route("/use-cases/library")
def use_cases_library():
    return _use_case_library_response()


@use_cases_bp.route("/use-cases/definition/add", methods=["POST"])
def use_case_definition_add():
    s = get_state(_scope_id())
    add_use_case_def(s)
    return _use_case_library_response("use_case_def_add")


@use_cases_bp.route("/use-cases/definition/remove", methods=["POST"])
def use_case_definition_remove():
    s = get_state(_scope_id())
    remove_use_case_def(s, request.form.get("key", ""))
    return _use_case_library_response("use_case_def_remove")


@use_cases_bp.route("/use-cases/definition/set", methods=["POST"])
def use_case_definition_set():
    s = get_state(_scope_id())
    key = request.form.get("key", "")
    field_name = request.form.get("field", "")
    raw_value = request.form.get("value", "")
    if field_name == "capability":
        set_use_case_def_capability(
            s, key, request.form.get("cap", ""), raw_value in ("on", "true", "1")
        )
    elif field_name == "batch_eligible":
        set_use_case_def_field(s, key, "batch_eligible", raw_value in ("on", "true", "1"))
    elif field_name == "tokens_day_m":
        set_use_case_def_field(s, key, "tokens_day", float(raw_value or 0.0) * 1e6)
    elif field_name == "latent_jobs_day_m":
        set_use_case_def_field(s, key, "latent_jobs_day", float(raw_value or 0.0) * 1e6)
    elif field_name == "wtp_per_m_cents":
        set_use_case_def_field(s, key, "wtp_per_m", float(raw_value or 0.0) / 100.0)
    elif field_name == "unlock_price_per_m_cents":
        set_use_case_def_field(s, key, "unlock_price_per_m", float(raw_value or 0.0) / 100.0)
    elif field_name == "min_success_rate_pct":
        set_use_case_def_field(s, key, "min_success_rate", float(raw_value or 0.0) / 100.0)
    elif field_name == "quality_floor_pct":
        set_use_case_def_field(s, key, "quality_floor", float(raw_value or 0.0) / 100.0)
    elif field_name.startswith("quality_weight_") and field_name.endswith("_pct"):
        domain = field_name.removeprefix("quality_weight_").removesuffix("_pct")
        set_use_case_def_field(s, key, f"quality_weight_{domain}", float(raw_value or 0.0) / 100.0)
    elif field_name == "difficulty_aa_index":
        set_use_case_def_field(
            s, key, "difficulty", aa_intelligence_to_quality(float(raw_value or 0.0))
        )
    elif field_name == "difficulty_elo":  # Legacy form payloads from pre-AA-index UI.
        set_use_case_def_field(
            s, key, "difficulty", max(0.0, min(1.0, float(raw_value or 0.0) / 3000.0))
        )
    elif field_name == "scale_value":
        set_use_case_def_field(s, key, "scale_value", float(raw_value or 0.0))
    elif field_name == "scale_token_multiplier":
        set_use_case_def_field(s, key, "scale_token_multiplier", float(raw_value or 0.0))
    elif field_name in {"scale_max", "scale_step"}:
        set_use_case_def_field(s, key, field_name, float(raw_value or 0.0))
    elif field_name in {
        "name",
        "scale_hint",
        "scale_model",
        "scale_label",
        "scale_unit",
        "scale_formula",
        "quality_domain",
        "in_pre",
        "out_pre",
    }:
        set_use_case_def_field(s, key, field_name, raw_value)
    elif field_name in {
        "tokens_day",
        "wtp_per_m",
        "difficulty",
        "min_success_rate",
        "quality_floor",
        "latent_jobs_day",
        "unlock_price_per_m",
    }:
        set_use_case_def_field(s, key, field_name, float(raw_value or 0.0))
    return _use_case_library_response("use_case_def_set")


@use_cases_bp.route("/use-cases/export")
def use_case_definition_export():
    s = get_state(_scope_id())
    body = json.dumps(serialize_use_case_defs(s), indent=2, sort_keys=True) + "\n"
    resp = make_response(body)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    resp.headers["Content-Disposition"] = 'attachment; filename="use-case-library.json"'
    return resp


@use_cases_bp.route("/use-cases/import", methods=["POST"])
def use_case_definition_import():
    s = get_state(_scope_id())
    raw = request.form.get("json", "")
    if not raw.strip():
        return jsonify({"error": "Choose a use-case JSON file first."}), 400
    replace_use_case_defs(s, _load_import_json(raw))
    retune_models(s, preserve_existing=False)
    return _use_case_library_response("use_case_def_import")
