"""Planner configuration routes.

GPU pools, model assignments, distributions, efficiency settings, task
presets and the projection levers."""

import json

from flask import (
    Blueprint,
    jsonify,
    make_response,
    render_template,
    request,
)

from calc import (
    avg_dist,
    resolve_spec_runtime,
    valid_strategies,
)
from data import (
    EMBEDDING_DOC_BUCKETS,
    EMBEDDING_DOC_PRESETS,
    GPUS,
    MODELS,
    PRECISIONS,
    TASK_PRESETS,
    aa_intelligence_to_quality,
)
from placement import (
    auto_select_models,
    retune_models,
)
from planner_service import (
    add_model,
    add_models,
    change_gpu_qty,
    create_default_scenario,
    set_model_gpu_count,
    set_model_gpu_pool,
    set_model_prec,
    set_model_spec,
)
from scenarios import (
    replace_project_set,
    serialize_project_set,
)
from state import (
    add_gpu,
    add_project,
    auto_exclude_model,
    auto_reallow_model,
    get_use_case_defs,
    normalize_auto_strategy,
    normalize_plot_mode,
    remove_gpu,
    remove_model,
    remove_project,
    set_dist_preset,
    set_dist_value,
    set_gpu_cost,
    set_model_strat,
    set_project_batch_eligible,
    set_project_capability,
    set_project_dist_preset,
    set_project_field,
    set_project_kind,
    set_project_name,
    set_project_scale_value,
    set_projection_choice,
    set_projection_pct,
    set_projection_toggle,
)
from web.config import (
    EFFICIENCY_SETTING_BOUNDS,
    INTEGER_SETTING_BOUNDS,
)
from web.helpers import (
    _bounded_int,
    _finite_float,
    _form_float,
    _form_int,
    _htmx_response,
    _load_import_json,
    _planner_view_context,
    _record_snapshot,
    _request_state,
    _tracked_htmx_response,
    with_planner_state,
)
from web.middleware import (
    _scope_id,
    _tab_id,
)
from web.session_store import (
    get_compare_state,
    get_state,
)

planner_bp = Blueprint("planner", __name__)


@planner_bp.route("/")
def index():
    tab_id = _tab_id(optional=True)
    if tab_id is None:
        default_a, default_b = create_default_scenario()
        return render_template(
            "index.html", A=default_a, B=default_b, **_planner_view_context(default_a, default_b)
        )

    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    _record_snapshot("open", sa, sb)
    return render_template("index.html", A=sa, B=sb, **_planner_view_context(sa, sb))


@planner_bp.route("/explainer")
def explainer():
    return render_template("explainer.html")


@planner_bp.route("/gpu/add", methods=["POST"])
@with_planner_state
def gpu_add(s):
    gpu_type = request.form.get("gpu_type")
    if gpu_type not in GPUS:
        return jsonify({"error": "Invalid GPU type"}), 400

    count = _form_int("count", 8)
    if count <= 0:
        return jsonify({"error": "Count must be positive"}), 400

    add_gpu(s, gpu_type, count)
    return _tracked_htmx_response("gpu_add", s)


@planner_bp.route("/gpu/qty", methods=["POST"])
@with_planner_state
def gpu_qty(s):
    uid = _form_int("uid")
    if "count" in request.form:
        gp = s.find_gpu(uid)
        if gp is None:
            return jsonify({"error": "GPU pool not found"}), 404
        count = max(0, int(_form_float("count", 0.0)))
        change_gpu_qty(s, uid, count - gp.count)
    else:
        delta = _form_int("delta")
        change_gpu_qty(s, uid, delta)
    # GPU count edits can fire repeatedly while a user types or steps the
    # control. Keep this hot path untracked to avoid a snapshot row for every
    # transient keystroke; stable changes are captured by subsequent actions.
    return _htmx_response(s)


@planner_bp.route("/gpu/remove", methods=["POST"])
@with_planner_state
def gpu_remove(s):
    uid = _form_int("uid")
    remove_gpu(s, uid)
    return _tracked_htmx_response("gpu_remove", s)


@planner_bp.route("/gpu/cost", methods=["POST"])
@with_planner_state
def gpu_cost(s):
    uid = _form_int("uid")
    cost = _form_float("value", 0.0)
    set_gpu_cost(s, uid, cost)
    # Like GPU quantity, cost edits are high-frequency numeric updates.
    return _htmx_response(s)


@planner_bp.route("/model/add", methods=["POST"])
@with_planner_state
def model_add(s):
    model_key = request.form.get("model_key")
    if model_key not in MODELS or MODELS[model_key].hidden:
        return jsonify({"error": "Invalid model key"}), 400

    add_model(s, model_key)
    return _htmx_response(s)


@planner_bp.route("/model/add-many", methods=["POST"])
@with_planner_state
def model_add_many(s):
    model_keys = [key for key in request.form.getlist("model_key") if key]
    if not model_keys and request.form.get("model_keys"):
        model_keys = [
            key.strip() for key in request.form.get("model_keys", "").split(",") if key.strip()
        ]
    if not model_keys:
        return jsonify({"error": "No model keys supplied"}), 400
    invalid = [key for key in model_keys if key not in MODELS or MODELS[key].hidden]
    if invalid:
        return jsonify({"error": "Invalid model key"}), 400

    add_models(s, model_keys)
    return _htmx_response(s)


@planner_bp.route("/model/auto", methods=["POST"])
@with_planner_state
def model_auto(s):
    strategy = normalize_auto_strategy(
        request.form.get("strategy")
        or request.form.get("auto_strategy")
        or getattr(s, "auto_strategy", None)
    )
    auto_select_models(s, strategy)
    return _tracked_htmx_response("model_auto", s)


@planner_bp.route("/model/auto-exclude", methods=["POST"])
@with_planner_state
def model_auto_exclude(s):
    uid = _form_int("uid")
    auto_exclude_model(s, uid)
    return _htmx_response(s)


@planner_bp.route("/model/auto-reallow", methods=["POST"])
@with_planner_state
def model_auto_reallow(s):
    model_key = request.form.get("key", "")
    auto_reallow_model(s, model_key)
    return _htmx_response(s)


@planner_bp.route("/model/remove", methods=["POST"])
@with_planner_state
def model_remove(s):
    uid = _form_int("uid")
    remove_model(s, uid)
    return _htmx_response(s)


@planner_bp.route("/model/prec", methods=["POST"])
@with_planner_state
def model_prec(s):
    uid = _form_int("uid")
    prec = request.form.get("prec")
    if prec not in PRECISIONS:
        return jsonify({"error": "Invalid precision"}), 400

    set_model_prec(s, uid, prec)
    return _htmx_response(s)


@planner_bp.route("/model/spec", methods=["POST"])
@with_planner_state
def model_spec(s):
    uid = _form_int("uid")
    method = request.form.get("method", "off")
    try:
        spec_k = _form_int("spec_k", 0)
    except (TypeError, ValueError):
        spec_k = 0
    am = s.find_model(uid)
    if am is not None and method != "off" and spec_k > 0:
        model = MODELS.get(am.model_key)
        profile = (
            next(
                (p for p in getattr(model, "available_spec_profiles", ()) if p.method == method),
                None,
            )
            if model is not None
            else None
        )
        supported_ks = tuple(getattr(profile, "supported_ks", ()) or ())
        if supported_ks and spec_k not in supported_ks:
            if method != am.spec_method:
                # The method selector submits the old method's k alongside
                # the new profile. Start the new profile in Auto instead of
                # rejecting an otherwise valid method change.
                spec_k = 0
            else:
                return jsonify(
                    {
                        "error": f"k={spec_k} is unsupported for {profile.label}; choose one of {supported_ks} or Auto"
                    }
                ), 400
    set_model_spec(s, uid, method, spec_k)
    return _htmx_response(s)


@planner_bp.route("/model/count", methods=["POST"])
@with_planner_state
def model_count(s):
    uid = _form_int("uid")
    count = _form_int("count")
    if count < 0:
        return jsonify({"error": "Count cannot be negative"}), 400

    set_model_gpu_count(s, uid, count)
    return _htmx_response(s)


@planner_bp.route("/model/strat", methods=["POST"])
@with_planner_state
def model_strat(s):
    uid = _form_int("uid")
    phase = request.form.get("phase", "decode")
    tp = _form_int("tp")
    pp = _form_int("pp", 1)
    dp = _form_int("dp")

    # Validate the strategy before setting
    am = s.find_model(uid)
    if am is None:
        return jsonify({"error": "Model not found"}), 404

    gp = s.find_gpu(am.gpu_uid)
    if gp is None:
        return jsonify({"error": "GPU not found"}), 404

    model = MODELS[am.model_key]
    spec = resolve_spec_runtime(model, am.spec_method, am.spec_k, s.spec_acceptance, am.prec)
    valid = valid_strategies(model, am.gpu_count, gp.gpu, s.mu, s.profiled_non_kv_gb, am.prec, spec)
    strategy = (tp, pp, dp)
    if strategy not in valid:
        return jsonify({"error": "Invalid strategy for this model/GPU combination"}), 400

    set_model_strat(s, uid, tp, pp, dp, phase)
    return _htmx_response(s)


@planner_bp.route("/model/gpu_pool", methods=["POST"])
@with_planner_state
def model_gpu_pool(s):
    uid = _form_int("uid")
    gpu_uid = _form_int("gpu_uid")
    set_model_gpu_pool(s, uid, gpu_uid)
    return _htmx_response(s)


@planner_bp.route("/dist/preset", methods=["POST"])
@with_planner_state
def dist_preset(s):
    kind = request.form.get("kind")
    preset = request.form.get("preset")
    set_dist_preset(s, kind, preset)
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("dist_preset", s)


@planner_bp.route("/dist/slide", methods=["POST"])
@with_planner_state
def dist_slide(s):
    kind = request.form.get("kind")
    if kind not in {"in", "out", "embedding_doc"}:
        return jsonify({"error": "Invalid distribution kind"}), 400
    index = _bounded_int(request.form.get("index"), name="index", lo=0, hi=256)
    value = _bounded_int(request.form.get("value"), name="value", lo=0, hi=1_000_000)
    set_dist_value(s, kind, index, value)
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("dist_slide", s)


@planner_bp.route("/settings/mu", methods=["POST"])
@with_planner_state
def settings_mu(s):
    value = _bounded_int(request.form.get("value"), name="gpu_mem_util", lo=50, hi=98)
    s.mu = value / 100
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("settings_mu", s)


@planner_bp.route("/settings/non-kv", methods=["POST"])
@with_planner_state
def settings_non_kv(s):
    value = _finite_float(
        request.form.get("value"), name="profiled non-KV memory", lo=0.0, hi=4096.0
    )
    s.profiled_non_kv_gb = value
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("settings_non_kv", s)


@planner_bp.route("/settings/eff", methods=["POST"])
@with_planner_state
def settings_eff(s):
    key = request.form.get("key")
    bounds = EFFICIENCY_SETTING_BOUNDS.get(key)
    if bounds is None:
        return jsonify({"error": "Invalid efficiency setting"}), 400
    lo, hi = bounds
    value = _finite_float(request.form.get("value"), name=key, lo=lo * 100, hi=hi * 100) / 100
    setattr(s, key, value)
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("settings_eff", s)


@planner_bp.route("/settings/int", methods=["POST"])
@with_planner_state
def settings_int(s):
    key = request.form.get("key")
    bounds = INTEGER_SETTING_BOUNDS.get(key)
    if bounds is None:
        return jsonify({"error": "Invalid integer setting"}), 400
    value = _bounded_int(request.form.get("value"), name=key, lo=bounds[0], hi=bounds[1])
    setattr(s, key, value)
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("settings_int", s)


@planner_bp.route("/projection/pct", methods=["POST"])
@with_planner_state
def projection_pct(s):
    key = request.form.get("key", "")
    value = _form_int("value") / 100
    set_projection_pct(s, key, value)
    return _tracked_htmx_response("projection_pct", s)


@planner_bp.route("/projection/choice", methods=["POST"])
@with_planner_state
def projection_choice(s):
    key = request.form.get("key", "")
    value = request.form.get("value", "")
    set_projection_choice(s, key, value)
    return _tracked_htmx_response("projection_choice", s)


@planner_bp.route("/projection/toggle", methods=["POST"])
@with_planner_state
def projection_toggle(s):
    key = request.form.get("key", "")
    # HTMX sends a form-encoded value only when the checkbox is checked.
    value = request.form.get("value") in ("on", "true", "1")
    set_projection_toggle(s, key, value)
    return _tracked_htmx_response("projection_toggle", s)


@planner_bp.route("/project/add", methods=["POST"])
@with_planner_state
def project_add(s):
    preset_key = request.form.get("preset") or None
    add_project(s, preset_key)
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("project_add", s)


@planner_bp.route("/project/add-all", methods=["POST"])
@with_planner_state
def project_add_all(s):
    for preset in get_use_case_defs(s):
        add_project(s, str(preset["key"]))
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("project_add_all", s)


@planner_bp.route("/project/remove", methods=["POST"])
@with_planner_state
def project_remove(s):
    uid = _form_int("uid", 0)
    remove_project(s, uid)
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("project_remove", s)


@planner_bp.route("/project/set", methods=["POST"])
@with_planner_state
def project_set(s):
    uid = _form_int("uid", 0)
    field_name = request.form.get("field", "")
    raw_value = request.form.get("value", "")
    if field_name == "kind":
        set_project_kind(s, uid, raw_value)
        retune_models(s, preserve_existing=False)
    elif field_name == "name":
        set_project_name(s, uid, raw_value)
    elif field_name == "batch_eligible":
        set_project_batch_eligible(s, uid, raw_value in ("on", "true", "1"))
    elif field_name == "tokens_day_m":
        # slider gives millions of tokens/day; persist in tokens/day
        set_project_field(s, uid, "tokens_day", float(raw_value or 0.0) * 1e6)
        retune_models(s, preserve_existing=False)
    elif field_name == "scale_value":
        set_project_scale_value(s, uid, float(raw_value or 0.0))
        retune_models(s, preserve_existing=False)
    elif field_name == "wtp_per_m_cents":
        # slider gives cents per M tokens; persist as $/M tokens
        set_project_field(s, uid, "wtp_per_m", float(raw_value or 0.0) / 100.0)
    elif field_name == "min_success_rate_pct":
        # slider gives whole-number percent; persist as 0..1 fraction
        set_project_field(s, uid, "min_success_rate", float(raw_value or 0.0) / 100.0)
    elif field_name == "quality_floor_pct":
        set_project_field(s, uid, "quality_floor", float(raw_value or 0.0) / 100.0)
    elif field_name == "difficulty_pct":
        # legacy: slider gave whole-number percent; persist as 0..1 fraction
        set_project_field(s, uid, "difficulty", float(raw_value or 0.0) / 100.0)
    elif field_name == "difficulty_aa_index":
        # UI uses the same published source scale as the model anchors.
        set_project_field(s, uid, "difficulty", aa_intelligence_to_quality(float(raw_value or 0.0)))
    elif field_name == "difficulty_elo":  # Legacy form payloads from pre-AA-index UI.
        elo = float(raw_value or 0.0)
        set_project_field(s, uid, "difficulty", max(0.0, min(1.0, elo / 3000.0)))
    elif field_name == "capability":
        # `cap` field carries the capability name; `value` is on/off
        cap_name = request.form.get("cap", "")
        set_project_capability(s, uid, cap_name, raw_value in ("on", "true", "1"))
    elif field_name == "latent_jobs_day_m":
        # slider gives millions of latent tokens/day; persist as tokens/day
        set_project_field(s, uid, "latent_jobs_day", float(raw_value or 0.0) * 1e6)
    elif field_name == "unlock_price_per_m_cents":
        # slider gives cents per M tokens; persist as $/M tokens
        set_project_field(s, uid, "unlock_price_per_m", float(raw_value or 0.0) / 100.0)
    elif field_name == "in_pre":
        set_project_dist_preset(s, uid, "in", raw_value)
        retune_models(s, preserve_existing=False)
    elif field_name == "out_pre":
        set_project_dist_preset(s, uid, "out", raw_value)
        retune_models(s, preserve_existing=False)
    elif field_name in (
        "tokens_day",
        "wtp_per_m",
        "difficulty",
        "min_success_rate",
        "quality_floor",
        "latent_jobs_day",
        "unlock_price_per_m",
    ):
        set_project_field(s, uid, field_name, float(raw_value or 0.0))
        if field_name == "tokens_day":
            retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("project_set", s)


@planner_bp.route("/project/export")
def project_export():
    s = _request_state()
    if s is None:
        return jsonify({"error": "No state found"}), 404
    payload = serialize_project_set(s)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    resp = make_response(body)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    resp.headers["Content-Disposition"] = 'attachment; filename="use-cases.json"'
    return resp


@planner_bp.route("/project/import", methods=["POST"])
@with_planner_state
def project_import(s):
    raw = request.form.get("json", "")
    if not raw.strip():
        return jsonify({"error": "Choose a use-case JSON file first."}), 400
    payload = _load_import_json(raw)
    replace_project_set(s, payload)
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("project_import", s)


@planner_bp.route("/mode", methods=["POST"])
def set_mode():
    mode = normalize_plot_mode(request.form.get("mode"))
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    if sa:
        sa.mode = mode
    if sb:
        sb.mode = mode
    return _tracked_htmx_response("mode", sa)


@planner_bp.route("/task/preset", methods=["POST"])
@with_planner_state
def task_preset(s):
    key = request.form.get("key")
    if getattr(s, "mode", "") in ("embedding", "embedquality"):
        preset = EMBEDDING_DOC_PRESETS.get(key)
        if preset:
            s.embedding_doc_dist = list(preset)
            s.embedding_doc_pre = key
            s.task_il = avg_dist(s.embedding_doc_dist, EMBEDDING_DOC_BUCKETS)
            s.task_ol = 0
            retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("task_preset", s)

    tp = TASK_PRESETS.get(key)
    if tp:
        s.task_il = tp["i"]
        s.task_ol = tp["o"]
        retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("task_preset", s)


@planner_bp.route("/task/length", methods=["POST"])
@with_planner_state
def task_length(s):
    length = 2 ** _form_int("value")
    kind = request.form.get("kind")
    if kind == "in":
        s.task_il = length
    else:
        s.task_ol = length
    retune_models(s, preserve_existing=False)
    return _tracked_htmx_response("task_length", s)
