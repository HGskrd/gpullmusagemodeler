"""Economics routes: the standalone /econ/ explorations and the swaps panel."""

from flask import Blueprint, abort, current_app, g, render_template, request

from engine.economics import (
    compute_revenue_projection,
)
from presentation.econ import compute_swap_recs, econ_payload
from web.cache import FingerprintCache, scenario_fingerprint
from web.session_store import (
    get_compare_state,
    get_state,
)

econ_bp = Blueprint("econ", __name__, url_prefix="/econ")

VARIANTS = {
    "flow": {
        "title": "Money & token flow",
        "desc": "Twin sankeys — one in dollars, one in tokens — showing where a day of demand ends up: captured on-prem, paid to cloud, or destroyed.",
    },
    "dashboard": {
        "title": "Executive dashboard",
        "desc": "KPI tiles, a value bridge from opportunity down to what was destroyed, and stacked demand outcomes.",
    },
    "brief": {
        "title": "Narrative brief",
        "desc": "The same numbers written out as an analyst memo — figures inline, diagnosis and actions first.",
    },
    "supply": {
        "title": "Fleet cockpit",
        "desc": "Deployed-model view: utilization, internal tariff sheet, capacity pressure, and the best model to swap in on existing GPUs.",
    },
}


def _scoped_state(panel: str = "A"):
    # _protect_and_lock_request_scope (app.py) populates g.planner_scope_id for
    # every scoped endpoint before this blueprint runs.
    scope_id = getattr(g, "planner_scope_id", None) or f"{g.visitor_id}:default"
    if panel == "B":
        return get_compare_state(scope_id)
    return get_state(scope_id)


def _context(variant: str | None = None) -> dict:
    state = _scoped_state()
    return {
        "variant": variant,
        "variants": VARIANTS,
        "state": state,
        "panel": "A",
        "ep": econ_payload(state),
    }


@econ_bp.get("/")
def index():
    return render_template("econ/index.html", **_context())


@econ_bp.get("/swaps")
def swaps():
    """Lazy model-swap recommendations for the dashboard/brief/fleet views.

    Kept out of the main HTMX response path: the search costs ~300ms and would
    otherwise tax every slider tick.
    """
    panel = request.args.get("panel", "A")
    state = _scoped_state("B" if panel == "B" else "A")
    if state is None:
        return ""
    cache: FingerprintCache = current_app.extensions["swap_recommendation_cache"]
    fingerprint = scenario_fingerprint(state)
    cached = cache.get(fingerprint)
    if cached is None:
        p = compute_revenue_projection(state)
        cached = cache.put(
            fingerprint,
            {"projection": p, "swap_recs": compute_swap_recs(state, p)},
        )
    p = cached["projection"]
    return render_template(
        "partials/econ/swaps.html",
        swap_recs=cached["swap_recs"],
        gpu_recs=p["recommendations"],
        view=request.args.get("view", "table"),
        panel=panel,
    )


@econ_bp.get("/<slug>")
def variant(slug: str):
    if slug not in VARIANTS:
        abort(404)
    return render_template(f"econ/{slug}.html", **_context(slug))
