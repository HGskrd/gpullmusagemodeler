"""JSON and picker routes: chart data, the text report, pickers and health."""

from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
)

from calc import (
    get_decode_bs,
    get_processing_pareto_bs,
    get_realtime_bs,
)
from data import (
    GPUS,
    MODEL_KINDS,
    models_by_kind,
)
from data.hardware_picker import picker_vendors
from placement import (
    resolve_deployment,
)
from presentation.capacity_only import capacity_without_quality
from presentation.charts import (
    DECISION_LENGTHS,
    chart_asr_quality,
    chart_decision_latency,
    chart_embedding_quality,
    chart_processing_pareto,
    chart_realtime_capacity,
    chart_user_pareto,
    decision_use_cases,
    embedding_quality_axis_range,
)
from presentation.model_cards import (
    get_model_infos,
)
from presentation.model_picker import panel_quality_floor, picker_rows
from presentation.reports import format_projection_report
from state import (
    PlannerState,
    get_use_case_defs,
)
from web.analytics import record_event
from web.cache import ScopedRevisionCache, combined_revision
from web.helpers import (
    _state,
    _template_context,
)
from web.middleware import (
    _scope_id,
)
from web.session_store import (
    get_compare_state,
    get_state,
)

api_bp = Blueprint("api", __name__)


def _annotate_chart_spec(state: PlannerState, datasets: list[dict]) -> list[dict]:
    """Attach backend-derived speculative assumptions for chart tooltips.

    Processing Pareto is an aggregate across deployed models, so its disclosure
    intentionally summarizes every enabled drafter. Model-specific chart points
    may provide more precise per-load fields, which the client prefers.
    """
    disclosures = []
    for info in get_model_infos(state):
        am = info["am"]
        spec = info.get("spec")
        if am.gpu_count <= 0 or spec is None:
            continue
        k_label = (
            f"Auto→{spec['k']}"
            if am.spec_k == 0 and spec["active"]
            else "Auto→off"
            if am.spec_k == 0
            else str(spec["k"])
        )
        disclosures.append(
            f"{info['model'].name}: {spec['profile'].label}, k {k_label}, "
            f"α {spec['alpha'] * 100:.0f}% ({spec['alpha_source']}), "
            f"{spec['speedup']:.2f}× @ {spec['probe_bs']} users"
        )
    disclosure = "; ".join(disclosures)
    if disclosure:
        for dataset in datasets:
            dataset["spec_disclosure"] = disclosure
    return datasets


def _build_chart_payload(sa: PlannerState, sb: PlannerState | None) -> dict:
    mode = sa.mode
    states = [sa] + ([sb] if sb else [])
    deployments = [resolve_deployment(state) for state in states]

    if mode == "processingpareto":
        batch_sizes = get_processing_pareto_bs(states)
        datasets = _annotate_chart_spec(sa, chart_processing_pareto(sa, batch_sizes))
        if sb:
            datasets += _annotate_chart_spec(sb, chart_processing_pareto(sb, batch_sizes, " (B)"))
        return {"type": "line", "datasets": datasets, "mode": mode, "x_max": batch_sizes[-1]}

    if mode == "decision":
        use_cases = decision_use_cases(sa)
        lengths = sorted(set(DECISION_LENGTHS) | {u["len"] for u in use_cases})
        datasets = chart_decision_latency(sa, lengths=lengths)
        if sb:
            datasets += chart_decision_latency(sb, " (B)", lengths)
        return {"type": "line", "datasets": datasets, "mode": mode, "use_cases": use_cases}

    if mode == "asrquality":
        datasets = chart_asr_quality(sa)
        capacity = capacity_without_quality(sa, mode)
        if sb:
            datasets += chart_asr_quality(sb, " (B)")
            capacity += capacity_without_quality(sb, mode, " (B)")
        return {
            "type": "scatter",
            "datasets": datasets,
            "mode": mode,
            **({"capacity_only": capacity} if capacity else {}),
        }

    if mode == "realtime":
        batch_sizes = get_realtime_bs(states, deployments=deployments)
        datasets = chart_realtime_capacity(sa, batch_sizes)
        if sb:
            datasets += chart_realtime_capacity(sb, batch_sizes, " (B)")
        return {"type": "line", "datasets": datasets, "mode": mode, "x_max": batch_sizes[-1]}

    if mode == "embedquality":
        datasets = chart_embedding_quality(sa)
        capacity = capacity_without_quality(sa, mode)
        if sb:
            datasets += chart_embedding_quality(sb, " (B)")
            capacity += capacity_without_quality(sb, mode, " (B)")
        return {
            "type": "scatter",
            "datasets": datasets,
            "mode": mode,
            **embedding_quality_axis_range(datasets),
            **({"capacity_only": capacity} if capacity else {}),
        }

    batch_sizes = get_decode_bs(states, deployments=deployments)
    datasets = _annotate_chart_spec(
        sa, chart_user_pareto(sa, batch_sizes, deployment=deployments[0])
    )
    if sb:
        datasets += _annotate_chart_spec(
            sb,
            chart_user_pareto(sb, batch_sizes, " (B)", deployment=deployments[1]),
        )
    return {"type": "line", "datasets": datasets, "mode": mode, "x_max": batch_sizes[-1]}


@api_bp.route("/api/chart-data")
def chart_data():
    scope_id = _scope_id()
    sa = get_state(scope_id)
    sb = get_compare_state(scope_id)
    cache_key = (combined_revision(sa, sb), "AB", sa.mode)
    cache: ScopedRevisionCache = current_app.extensions["chart_json_cache"]
    cached = cache.get(scope_id, cache_key)
    if cached is None:
        response = jsonify(_build_chart_payload(sa, sb))
        cached = cache.put(scope_id, cache_key, response.get_data())
    else:
        response = current_app.response_class(cached.body, mimetype="application/json")
    # A weak validator is correct for both identity and middleware-gzipped
    # representations of the same JSON payload.
    response.set_etag(cached.etag, weak=True)
    response.make_conditional(request)
    return response


@api_bp.route("/api/projection-report")
def projection_report():
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    record_event("export_report", sa, sb)
    return jsonify({"text": format_projection_report(sa, sb)})


@api_bp.route("/picker/gpu")
def picker_gpu():
    return render_template(
        "partials/gpu_picker.html",
        panel=request.args.get("panel", "A"),
        picker_vendors=picker_vendors(),
        GPUS=GPUS,
    )


@api_bp.route("/picker/model")
def picker_model():
    kind_groups = models_by_kind()
    valid_kind_keys = {kind_key for kind_key, _ in MODEL_KINDS}
    active_kind = request.args.get("kind")
    if active_kind not in valid_kind_keys or not kind_groups.get(active_kind):
        active_kind = next(
            (kind_key for kind_key, _ in MODEL_KINDS if kind_groups.get(kind_key)), None
        )
    panel = request.args.get("panel", "A")
    state = _state(panel) or get_state(_scope_id())
    # Fit/quality rows are computed only for the active tab (switching tabs
    # re-requests this route), which keeps the open well under 150 ms.
    rows = picker_rows(state, kind_groups.get(active_kind, {}), active_kind) if active_kind else {}
    return render_template(
        "partials/model_picker.html",
        panel=panel,
        models_by_kind=kind_groups,
        MODEL_KINDS=MODEL_KINDS,
        active_kind=active_kind,
        picker_rows=rows,
        has_gpus=bool(state.gpus),
        # With every GPU already assigned, "fits free GPUs" would dim every row
        # on open; start with the filter off in that case.
        has_free_gpus=any(state.free_gpu_for_pool(pool.uid) > 0 for pool in state.gpus),
        quality_floor=panel_quality_floor(state) if active_kind == "llm" else None,
    )


@api_bp.route("/picker/project")
def picker_project():
    panel = request.args.get("panel", "A")
    s = _state(panel) or get_state(_scope_id())
    context = _template_context()
    context["PROJECT_PRESETS"] = get_use_case_defs(s)
    return render_template("partials/project_picker.html", panel=panel, **context)


@api_bp.route("/healthz", methods=["GET"])
def healthz():
    storage_ok = (
        True
        if not current_app.config["TRACKING_ENABLED"]
        else current_app.extensions["snapshot_store"].healthcheck()
    )
    return jsonify({"status": "ok" if storage_ok else "degraded", "storage": storage_ok}), (
        200 if storage_ok else 503
    )
