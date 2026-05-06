"""Flask application for vLLM Multi-Model Planner."""

import math
import os
import uuid
from collections import defaultdict
from pathlib import Path

from flask import Flask, g, jsonify, make_response, redirect, render_template, request, session, url_for

from data import (
    GPUS,
    MODELS,
    DIST_PRESETS,
    TASK_PRESETS,
    INPUT_BUCKETS,
    OUTPUT_BUCKETS,
    BATCH_SIZES,
    DAY_SHAPES,
    CLOUD_MODELS,
    CORPO_CLOUD_PRESETS,
    PROJECT_PRESETS,
    MODEL_CAPABILITIES,
    CAPABILITY_LABELS,
    COUNTRIES,
    CARBON_INTENSITY_HOURLY,
    PRECISIONS,
    PRECISION_LABELS,
    PRECISION_DESCRIPTIONS,
    models_by_category,
    gpu_cards_by_vendor,
    gpus_by_vendor,
    required_quality,
    success_rate,
)
from calc import (
    avg_dist,
    chart_processing_pareto,
    chart_user_pareto,
    compute_revenue_projection,
    get_decode_bs,
    get_processing_pareto_bs,
    dist_percentile,
    effective_prefill_length,
    normalize_dist,
    strategy_label,
    valid_strategies,
)
from state import (
    PlannerState,
    VISIBLE_PLOT_MODES,
    add_gpu,
    add_model,
    add_project,
    change_gpu_qty,
    clear_compare_state,
    create_default_state,
    duplicate_compare_state,
    get_compare_state,
    get_model_info,
    get_model_infos,
    get_state,
    remove_gpu,
    remove_model,
    remove_project,
    retune_models,
    normalize_plot_mode,
    set_dist_preset,
    set_dist_value,
    set_gpu_cost,
    set_model_gpu_count,
    set_model_gpu_pool,
    set_model_prec,
    set_model_strat,
    set_prefix_hit_rate,
    set_project_batch_eligible,
    set_project_capability,
    set_project_dist_preset,
    set_project_field,
    set_project_name,
    set_projection_choice,
    set_projection_pct,
    set_projection_toggle,
)
from tracking import SnapshotStore


BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")


app = Flask(__name__)
app.secret_key = os.environ.get("PLANNER_SECRET_KEY", "vllm-planner-dev-key")

VISITOR_COOKIE = "planner_vid"
TAB_PARAM = "tab_id"
ADMIN_SESSION_KEY = "planner_admin_ok"
SNAPSHOT_STORE = SnapshotStore(BASE_DIR / "instance" / "planner_snapshots.json")


def _new_id() -> str:
    return str(uuid.uuid4())


def _visitor_id() -> str:
    visitor_id = getattr(g, "visitor_id", None)
    if visitor_id:
        return visitor_id

    visitor_id = request.cookies.get(VISITOR_COOKIE)
    if not visitor_id:
        visitor_id = _new_id()
    g.visitor_id = visitor_id
    return visitor_id


def _tab_id(optional: bool = False) -> str | None:
    tab_id = (
        request.headers.get("X-Tab-ID")
        or request.form.get(TAB_PARAM)
    )
    if tab_id:
        return tab_id
    return None if optional else "default"


def _scope_id() -> str:
    return f"{_visitor_id()}:{_tab_id()}"


@app.after_request
def _set_identity_cookie(response):
    visitor_id = getattr(g, "visitor_id", None)
    if visitor_id and request.cookies.get(VISITOR_COOKIE) != visitor_id:
        response.set_cookie(
            VISITOR_COOKIE,
            visitor_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="Lax",
            secure=request.is_secure,
        )
    return response


def _admin_password() -> str | None:
    return os.environ.get("PLANNER_ADMIN_PASSWORD")


def _is_admin_authenticated() -> bool:
    return bool(session.get(ADMIN_SESSION_KEY))


def _state(panel: str = "A") -> PlannerState | None:
    if panel == "B":
        return get_compare_state(_scope_id())
    return get_state(_scope_id())


def _request_state() -> PlannerState | None:
    return _state(request.values.get("panel", "A"))


def _template_context() -> dict:
    return {
        "GPUS": GPUS,
        "MODELS": MODELS,
        "DIST_PRESETS": DIST_PRESETS,
        "TASK_PRESETS": TASK_PRESETS,
        "DAY_SHAPES": DAY_SHAPES,
        "CLOUD_MODELS": CLOUD_MODELS,
        "CORPO_CLOUD_PRESETS": CORPO_CLOUD_PRESETS,
        "PROJECT_PRESETS": PROJECT_PRESETS,
        "MODEL_CAPABILITIES": MODEL_CAPABILITIES,
        "CAPABILITY_LABELS": CAPABILITY_LABELS,
        "COUNTRIES": COUNTRIES,
        "VISIBLE_PLOT_MODES": VISIBLE_PLOT_MODES,
        "INPUT_BUCKETS": INPUT_BUCKETS,
        "OUTPUT_BUCKETS": OUTPUT_BUCKETS,
        "BATCH_SIZES": BATCH_SIZES,
        "PRECISIONS": PRECISIONS,
        "PRECISION_LABELS": PRECISION_LABELS,
        "PRECISION_DESCRIPTIONS": PRECISION_DESCRIPTIONS,
        "models_by_category": models_by_category,
        "gpu_cards_by_vendor": gpu_cards_by_vendor,
        "gpus_by_vendor": gpus_by_vendor,
        "normalize_dist": normalize_dist,
        "avg_dist": avg_dist,
        "dist_percentile": dist_percentile,
        "get_model_info": get_model_info,
        "get_model_infos": get_model_infos,
        "compute_revenue_projection": compute_revenue_projection,
        "valid_strategies": valid_strategies,
        "effective_prefill_length": effective_prefill_length,
        "strategy_label": strategy_label,
        "required_quality": required_quality,
        "success_rate": success_rate,
        "math": math,
    }


def _record_snapshot(reason: str, state_a: PlannerState, state_b: PlannerState | None, path: str | None = None):
    SNAPSHOT_STORE.record_snapshot(
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
    resp = make_response(render_template("partials/htmx_response.html", A=sa, B=sb, **_template_context()))
    resp.headers["HX-Trigger"] = "refreshChart"
    return resp


def _tracked_htmx_response(reason: str, state_a: PlannerState | None = None):
    # Always render and snapshot the canonical A/B pair. Callers may pass the
    # mutated panel, which is panel B for compare-side edits.
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    _record_snapshot(reason, sa, sb)
    resp = make_response(render_template("partials/htmx_response.html", A=sa, B=sb, **_template_context()))
    resp.headers["HX-Trigger"] = "refreshChart"
    return resp


@app.route("/")
def index():
    tab_id = _tab_id(optional=True)
    if tab_id is None:
        return render_template("index.html", A=create_default_state(), B=None, **_template_context())

    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    _record_snapshot("open", sa, sb)
    return render_template("index.html", A=sa, B=sb, **_template_context())


@app.route("/explainer")
def explainer():
    return render_template("explainer.html")


@app.route("/session/sync")
def session_sync():
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    _record_snapshot("open", sa, sb, path="/")
    return _htmx_response(sa)


@app.route("/gpu/add", methods=["POST"])
def gpu_add():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        gpu_type = request.form.get("gpu_type")
        if gpu_type not in GPUS:
            return jsonify({"error": "Invalid GPU type"}), 400
        
        count = int(request.form.get("count", 8))
        if count <= 0:
            return jsonify({"error": "Count must be positive"}), 400
            
        add_gpu(s, gpu_type, count)
        return _tracked_htmx_response("gpu_add", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gpu/qty", methods=["POST"])
def gpu_qty():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        delta = int(request.form.get("delta"))
        change_gpu_qty(s, uid, delta)
        return _tracked_htmx_response("gpu_qty", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gpu/remove", methods=["POST"])
def gpu_remove():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        remove_gpu(s, uid)
        return _tracked_htmx_response("gpu_remove", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gpu/cost", methods=["POST"])
def gpu_cost():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        cost = float(request.form.get("value", 0))
        set_gpu_cost(s, uid, cost)
        return _tracked_htmx_response("gpu_cost", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/add", methods=["POST"])
def model_add():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        model_key = request.form.get("model_key")
        if model_key not in MODELS or MODELS[model_key].hidden:
            return jsonify({"error": "Invalid model key"}), 400
            
        add_model(s, model_key)
        return _tracked_htmx_response("model_add", s)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/remove", methods=["POST"])
def model_remove():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        remove_model(s, uid)
        return _tracked_htmx_response("model_remove", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/prec", methods=["POST"])
def model_prec():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        prec = request.form.get("prec")
        if prec not in PRECISIONS:
            return jsonify({"error": "Invalid precision"}), 400
            
        set_model_prec(s, uid, prec)
        return _tracked_htmx_response("model_prec", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/count", methods=["POST"])
def model_count():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        count = int(request.form.get("count"))
        if count < 0:
            return jsonify({"error": "Count cannot be negative"}), 400
            
        set_model_gpu_count(s, uid, count)
        return _tracked_htmx_response("model_count", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/strat", methods=["POST"])
def model_strat():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        phase = request.form.get("phase", "decode")
        tp = int(request.form.get("tp"))
        pp = int(request.form.get("pp", 1))
        dp = int(request.form.get("dp"))
        
        # Validate the strategy before setting
        am = s.find_model(uid)
        if am is None:
            return jsonify({"error": "Model not found"}), 404
            
        gp = s.find_gpu(am.gpu_uid)
        if gp is None:
            return jsonify({"error": "GPU not found"}), 404
            
        model = MODELS[am.model_key]
        valid = valid_strategies(model, am.gpu_count, gp.gpu, s.mu, s.profiled_non_kv_gb, am.prec)
        strategy = (tp, pp, dp)
        if strategy not in valid:
            return jsonify({"error": "Invalid strategy for this model/GPU combination"}), 400
            
        set_model_strat(s, uid, tp, pp, dp, phase)
        return _tracked_htmx_response("model_strat", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/gpu_pool", methods=["POST"])
def model_gpu_pool():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid"))
        gpu_uid = int(request.form.get("gpu_uid"))
        set_model_gpu_pool(s, uid, gpu_uid)
        return _tracked_htmx_response("model_gpu_pool", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dist/preset", methods=["POST"])
def dist_preset():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        kind = request.form.get("kind")
        preset = request.form.get("preset")
        set_dist_preset(s, kind, preset)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("dist_preset", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dist/slide", methods=["POST"])
def dist_slide():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        kind = request.form.get("kind")
        index = int(request.form.get("index"))
        value = int(request.form.get("value"))
        set_dist_value(s, kind, index, value)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("dist_slide", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/mu", methods=["POST"])
def settings_mu():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        value = int(request.form.get("value"))
        s.mu = value / 100
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_mu", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/non-kv", methods=["POST"])
def settings_non_kv():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        value = float(request.form.get("value"))
        s.profiled_non_kv_gb = max(0.0, value)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_non_kv", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/eff", methods=["POST"])
def settings_eff():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key")
        value = int(request.form.get("value")) / 100
        if hasattr(s, key):
            setattr(s, key, value)
            retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_eff", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/int", methods=["POST"])
def settings_int():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key")
        value = int(request.form.get("value"))
        if hasattr(s, key):
            setattr(s, key, value)
            retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_int", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings/prefix-hit", methods=["POST"])
def settings_prefix_hit():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        value = int(request.form.get("value"))
        set_prefix_hit_rate(s, value / 100)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("settings_prefix_hit", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/projection/pct", methods=["POST"])
def projection_pct():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key", "")
        value = int(request.form.get("value")) / 100
        set_projection_pct(s, key, value)
        return _tracked_htmx_response("projection_pct", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/projection/choice", methods=["POST"])
def projection_choice():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key", "")
        value = request.form.get("value", "")
        set_projection_choice(s, key, value)
        return _tracked_htmx_response("projection_choice", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/projection/toggle", methods=["POST"])
def projection_toggle():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key", "")
        # HTMX sends a form-encoded value only when the checkbox is checked.
        value = request.form.get("value") in ("on", "true", "1")
        set_projection_toggle(s, key, value)
        return _tracked_htmx_response("projection_toggle", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/add", methods=["POST"])
def project_add():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        preset_key = request.form.get("preset") or None
        add_project(s, preset_key)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("project_add", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/remove", methods=["POST"])
def project_remove():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid", "0"))
        remove_project(s, uid)
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("project_remove", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/project/set", methods=["POST"])
def project_set():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        uid = int(request.form.get("uid", "0"))
        field_name = request.form.get("field", "")
        raw_value = request.form.get("value", "")
        if field_name == "name":
            set_project_name(s, uid, raw_value)
        elif field_name == "batch_eligible":
            set_project_batch_eligible(s, uid, raw_value in ("on", "true", "1"))
        elif field_name == "tokens_day_m":
            # slider gives millions of tokens/day; persist in tokens/day
            set_project_field(s, uid, "tokens_day", float(raw_value or 0.0) * 1e6)
            retune_models(s, preserve_existing=False)
        elif field_name == "wtp_per_m_cents":
            # slider gives cents per M tokens; persist as $/M tokens
            set_project_field(s, uid, "wtp_per_m", float(raw_value or 0.0) / 100.0)
        elif field_name == "min_success_rate_pct":
            # slider gives whole-number percent; persist as 0..1 fraction
            set_project_field(s, uid, "min_success_rate", float(raw_value or 0.0) / 100.0)
        elif field_name == "difficulty_pct":
            # legacy: slider gave whole-number percent; persist as 0..1 fraction
            set_project_field(s, uid, "difficulty", float(raw_value or 0.0) / 100.0)
        elif field_name == "difficulty_elo":
            # slider/input gives ELO on a 0..3000 axis; persist as 0..1 fraction
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
        elif field_name in ("tokens_day", "wtp_per_m", "difficulty", "min_success_rate", "latent_jobs_day", "unlock_price_per_m"):
            set_project_field(s, uid, field_name, float(raw_value or 0.0))
            if field_name == "tokens_day":
                retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("project_set", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/mode", methods=["POST"])
def set_mode():
    try:
        mode = normalize_plot_mode(request.form.get("mode"))
        sa = get_state(_scope_id())
        sb = get_compare_state(_scope_id())
        if sa:
            sa.mode = mode
        if sb:
            sb.mode = mode
        return _tracked_htmx_response("mode", sa)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare/duplicate", methods=["POST"])
def compare_duplicate():
    try:
        duplicate_compare_state(_scope_id())
        return _tracked_htmx_response("compare_duplicate")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare/close", methods=["POST"])
def compare_close():
    try:
        clear_compare_state(_scope_id())
        return _tracked_htmx_response("compare_close")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/task/preset", methods=["POST"])
def task_preset():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        key = request.form.get("key")
        tp = TASK_PRESETS.get(key)
        if tp:
            s.task_il = tp["i"]
            s.task_ol = tp["o"]
            retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("task_preset", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/task/length", methods=["POST"])
def task_length():
    try:
        s = _request_state()
        if s is None:
            return _htmx_response()
        length = 2 ** int(request.form.get("value"))
        kind = request.form.get("kind")
        if kind == "in":
            s.task_il = length
        else:
            s.task_ol = length
        retune_models(s, preserve_existing=False)
        return _tracked_htmx_response("task_length", s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart-data")
def chart_data():
    try:
        sa = get_state(_scope_id())
        sb = get_compare_state(_scope_id())
        mode = sa.mode
        states = [sa] + ([sb] if sb else [])

        if mode == "processingpareto":
            batch_sizes = get_processing_pareto_bs(states)
            datasets = chart_processing_pareto(sa, batch_sizes)
            if sb:
                datasets += chart_processing_pareto(sb, batch_sizes, " (B)")
            return jsonify({"type": "line", "datasets": datasets, "mode": mode, "x_max": batch_sizes[-1]})

        batch_sizes = get_decode_bs(states)
        datasets = chart_user_pareto(sa, batch_sizes)
        if sb:
            datasets += chart_user_pareto(sb, batch_sizes, " (B)")
        return jsonify({"type": "line", "datasets": datasets, "mode": mode, "x_max": batch_sizes[-1]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/picker/gpu")
def picker_gpu():
    return render_template("partials/gpu_picker.html", panel=request.args.get("panel", "A"), gpu_cards_by_vendor=gpu_cards_by_vendor())


@app.route("/picker/model")
def picker_model():
    return render_template("partials/model_picker.html", panel=request.args.get("panel", "A"), models_by_category=models_by_category())


@app.route("/picker/project")
def picker_project():
    return render_template("partials/project_picker.html", panel=request.args.get("panel", "A"), PROJECT_PRESETS=PROJECT_PRESETS)


@app.route("/admin", methods=["GET"])
def admin():
    if not _admin_password():
        return (
            "Set PLANNER_ADMIN_PASSWORD to enable /admin.",
            503,
        )
    if not _is_admin_authenticated():
        return render_template("admin_login.html", error=None)
    all_snapshots = SNAPSHOT_STORE.list_snapshots()
    visitor_map = defaultdict(list)
    for s in all_snapshots:
        visitor_map[s["visitor_id"]].append(s)
    visitors = sorted(
        [{"visitor_id": vid, "snapshots": snaps} for vid, snaps in visitor_map.items()],
        key=lambda v: v["snapshots"][0]["last_seen"],
        reverse=True,
    )
    return render_template("admin.html", visitors=visitors, total_snapshots=len(all_snapshots), GPUS=GPUS, MODELS=MODELS)


@app.route("/admin/login", methods=["POST"])
def admin_login():
    password = _admin_password()
    if not password:
        return (
            "Set PLANNER_ADMIN_PASSWORD to enable /admin.",
            503,
        )
    if request.form.get("password") != password:
        return render_template("admin_login.html", error="Invalid password."), 401

    session[ADMIN_SESSION_KEY] = True
    return redirect(url_for("admin"))


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.pop(ADMIN_SESSION_KEY, None)
    return redirect(url_for("admin"))


@app.template_filter("fmt_num")
def fmt_num(n):
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(int(n))


@app.template_filter("fmt_money")
def fmt_money(value):
    value = float(value or 0.0)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{sign}${value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{sign}${value / 1e3:.1f}k"
    if value >= 100:
        return f"{sign}${value:,.0f}"
    if value >= 1:
        return f"{sign}${value:,.2f}"
    return f"{sign}${value:,.3f}"


@app.template_filter("log2int")
def log2int(n):
    return int(math.log2(n)) if n > 0 else 0


if __name__ == "__main__":
    app.run(debug=True, port=5014)
