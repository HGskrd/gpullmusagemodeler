"""Scenario and session routes: sync, reset, compare panels, import/export."""

import json

from flask import (
    Blueprint,
    current_app,
    g,
    jsonify,
    make_response,
    request,
)

from planner_service import (
    deserialize_scenario,
)
from scenarios import (
    serialize_scenario,
)
from state import (
    set_cloud_enabled,
    set_price_driven_demand,
    set_projection_choice,
    set_projection_toggle,
)
from web.analytics import record_event
from web.helpers import (
    _htmx_response,
    _load_import_json,
    _record_snapshot,
    _tracked_htmx_response,
)
from web.middleware import (
    VISITOR_COOKIE,
    _scope_id,
    _visitor_id,
)
from web.session_store import (
    clear_compare_state,
    delete_visitor_states,
    duplicate_compare_state,
    get_compare_state,
    get_scope_lock,
    get_state,
    replace_scope_states,
    reset_state,
)

scenarios_bp = Blueprint("scenarios", __name__)


@scenarios_bp.route("/session/sync")
def session_sync():
    sa = get_state(_scope_id())
    sb = get_compare_state(_scope_id())
    _record_snapshot("open", sa, sb, path="/")
    return _htmx_response(sa)


@scenarios_bp.route("/compare/duplicate", methods=["POST"])
def compare_duplicate():
    duplicate_compare_state(_scope_id())
    return _tracked_htmx_response("compare_duplicate")


@scenarios_bp.route("/compare/close", methods=["POST"])
def compare_close():
    clear_compare_state(_scope_id())
    # Equal demand only means something next to a second config.
    state = get_state(_scope_id())
    if state is not None and not state.projection_include_latent:
        set_projection_toggle(state, "projection_include_latent", True)
    return _tracked_htmx_response("compare_close")


@scenarios_bp.route("/compare/equal-demand", methods=["POST"])
def compare_equal_demand():
    """Score A and B against the same demand (baseline only) or each against its own.

    Latent demand activates with each deployment's internal price, so two configs
    over identical use cases can report different total demand. Equal demand turns
    latent activation off on both panels together.
    """
    equal = request.form.get("value") in ("on", "true", "1")
    for state in (get_state(_scope_id()), get_compare_state(_scope_id())):
        if state is not None:
            set_projection_toggle(state, "projection_include_latent", not equal)
    return _tracked_htmx_response("compare_equal_demand")


@scenarios_bp.route("/cloud/toggle", methods=["POST"])
def cloud_toggle():
    """App-wide: with cloud off, demand the GPUs cannot serve is lost, in A and B."""
    enabled = request.form.get("value") in ("on", "true", "1")
    for state in (get_state(_scope_id()), get_compare_state(_scope_id())):
        if state is not None:
            set_cloud_enabled(state, enabled)
    return _tracked_htmx_response("cloud_toggle")


@scenarios_bp.route("/pricing/toggle", methods=["POST"])
def pricing_toggle():
    """Scenario-wide: demand reacts to internal prices (chargeback) or is the workload entered."""
    enabled = request.form.get("value") in ("on", "true", "1")
    for state in (get_state(_scope_id()), get_compare_state(_scope_id())):
        if state is not None:
            set_price_driven_demand(state, enabled)
    return _tracked_htmx_response("pricing_toggle")


@scenarios_bp.route("/gateway", methods=["POST"])
def gateway_set():
    """Scenario-wide: which cloud models take overflow, for A and B."""
    value = request.form.get("value", "")
    for state in (get_state(_scope_id()), get_compare_state(_scope_id())):
        if state is not None:
            set_projection_choice(state, "corpo_cloud", value)
    return _tracked_htmx_response("gateway_set")


@scenarios_bp.route("/session/reset", methods=["POST"])
def session_reset():
    state = reset_state(_scope_id(), blank=True)
    return _tracked_htmx_response("session_reset", state)


@scenarios_bp.route("/session/data", methods=["DELETE", "POST"])
def session_data_delete():
    visitor_id = _visitor_id()
    visitor_lock = get_scope_lock(f"visitor:{visitor_id}")
    with visitor_lock:
        states_deleted = delete_visitor_states(visitor_id)
        snapshots_deleted = current_app.extensions["snapshot_store"].delete_visitor(visitor_id)
    g.suppress_identity_cookie = True
    response = jsonify(
        {
            "deleted": True,
            "snapshots_deleted": snapshots_deleted,
            "states_deleted": states_deleted,
        }
    )
    response.delete_cookie(
        VISITOR_COOKIE,
        httponly=True,
        samesite="Lax",
        secure=current_app.config["SESSION_COOKIE_SECURE"] or request.is_secure,
    )
    return response


@scenarios_bp.route("/scenario/export", methods=["GET"])
def scenario_export():
    state_a, state_b = get_state(_scope_id()), get_compare_state(_scope_id())
    record_event("scenario_export", state_a, state_b)
    payload = serialize_scenario(state_a, state_b)
    body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    response = make_response(body)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    response.headers["Content-Disposition"] = 'attachment; filename="gpu-llm-scenario.json"'
    return response


@scenarios_bp.route("/scenario/import", methods=["POST"])
def scenario_import():
    raw = request.form.get("json", "")
    if not raw.strip():
        return jsonify({"error": "Choose a scenario JSON file first."}), 400
    try:
        state_a, state_b = deserialize_scenario(_load_import_json(raw))
    except (json.JSONDecodeError, ValueError) as error:
        message = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        return jsonify({"error": f"Invalid scenario JSON: {message}"}), 400
    replace_scope_states(_scope_id(), state_a, state_b)
    return _tracked_htmx_response("scenario_import", state_a)
