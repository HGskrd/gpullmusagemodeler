"""Presentation layer for the demand & economics projection.

econ_payload() builds the shared view-model (projection + chart JSON) consumed
by both the standalone /econ/ variant pages and the main planner's economics
section (templates/partials/econ/*). The /econ/ pages remain as full-page
explorations; the same partials render inside the calculator via HTMX.
"""

import json

from flask import Blueprint, abort, g, render_template, request

import cloud_policy
from calc import (
    _marginal_model_swap_recommendations,
    compute_revenue_projection,
)
from data import quality_to_aa_intelligence
from state import get_compare_state, get_state

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


# JSON-safe field subset for chart payloads (full rows contain dataclasses).
_PROJECT_JSON_KEYS = (
    "name", "tokens_day", "served_pct", "spilled_pct", "leaked_pct", "destroyed_pct",
    "value_served", "value_spilled", "value_leaked", "value_destroyed",
    "wtp_per_m", "cloud_pm", "cloud_blocked", "cheapest_effective_pm",
)


def _chart_payload(p: dict, model_tokens: list[dict]) -> dict:
    """Sankey/bridge/stack payloads. Node colors are semantic keys resolved
    client-side from CSS variables, except per-model colors which are catalog
    values passed through as-is."""
    f = p["fates"]
    margin = p["margin_day"]

    m_nodes = [
        {"name": "Total opportunity", "depth": 0, "c": "total"},
        {"name": "Destroyed (shelved)", "depth": 1, "c": "destroyed"},
        {"name": "Paid to cloud", "depth": 1, "c": "spilled"},
        {"name": "Served on-prem", "depth": 1, "c": "served"},
    ]
    m_links = [
        {"source": "Total opportunity", "target": "Served on-prem", "value": p["value_served_day"]},
        {"source": "Total opportunity", "target": "Paid to cloud", "value": p["value_cloud_day"]},
        {"source": "Total opportunity", "target": "Destroyed (shelved)", "value": p["value_destroyed_day"]},
    ]
    if p["value_spilled_day"] > 0:
        m_nodes.append({"name": "Spilled (capacity)", "depth": 2, "c": "spilled"})
        m_links.append({"source": "Paid to cloud", "target": "Spilled (capacity)", "value": p["value_spilled_day"]})
    if p["value_leaked_day"] > 0:
        m_nodes.append({"name": "Leaked (fit/price)", "depth": 2, "c": "leaked"})
        m_links.append({"source": "Paid to cloud", "target": "Leaked (fit/price)", "value": p["value_leaked_day"]})
    if p["cost_day"] > 0 and p["value_served_day"] > 0:
        if margin > 0:
            m_nodes.append({"name": "Net margin", "depth": 2, "c": "margin"})
            m_links.append({"source": "Served on-prem", "target": "Net margin", "value": margin})
        m_nodes.append({"name": "Covers cluster cost", "depth": 2, "c": "cost"})
        m_links.append({
            "source": "Served on-prem", "target": "Covers cluster cost",
            "value": p["value_served_day"] - margin if margin > 0 else p["value_served_day"],
        })

    t_nodes = [
        {"name": "Active demand", "depth": 0, "c": "total"},
        {"name": "Served on-prem", "depth": 1, "c": "served"},
        {"name": "Spilled (capacity)", "depth": 1, "c": "spilled"},
        {"name": "Leaked (fit/price)", "depth": 1, "c": "leaked"},
        {"name": "Destroyed (shelved)", "depth": 1, "c": "destroyed"},
    ]
    t_links = [
        {"source": "Active demand", "target": "Served on-prem", "value": f["served_tokens"]},
        {"source": "Active demand", "target": "Spilled (capacity)", "value": f["spilled_tokens"]},
        {"source": "Active demand", "target": "Leaked (fit/price)", "value": f["leaked_tokens"]},
        {"source": "Active demand", "target": "Destroyed (shelved)", "value": f["destroyed_tokens"]},
    ]
    for mt in model_tokens:
        if mt["tokens"] > 0:
            t_nodes.append({"name": mt["name"], "depth": 2, "color": mt["color"]})
            t_links.append({"source": "Served on-prem", "target": mt["name"], "value": mt["tokens"]})

    a_nodes: list[dict] = []
    a_links: list[dict] = []
    for i, row in enumerate(p["projects"], start=1):
        if not row["any_served"]:
            continue
        uc = f"uc{i}::{row['name']}"
        a_nodes.append({"name": uc, "c": "uc"})
        for pm in row["per_model_served"]:
            a_links.append({"source": f"m::{pm['name']}", "target": uc, "value": pm["tokens"]})
    served_model_names = {link["source"] for link in a_links}
    for m in p["models"]:
        name = f"m::{m['name']}"
        if name in served_model_names:
            a_nodes.append({"name": name, "color": m["color"]})

    return {
        "money": {"nodes": m_nodes, "links": m_links},
        "tokens": {"nodes": t_nodes, "links": t_links},
        "bridge": {
            "money": {
                "total": p["value_opportunity_day"], "cloud": p["value_cloud_day"],
                "captured": p["value_served_day"], "destroyed": p["value_destroyed_day"],
                "totalName": "Opportunity",
            },
            "tokens": {
                "total": f["total_tokens"], "cloud": f["spilled_tokens"] + f["leaked_tokens"],
                "captured": f["served_tokens"], "destroyed": f["destroyed_tokens"],
                "totalName": "Active demand",
            },
        },
        "stack": [{k: r[k] for k in _PROJECT_JSON_KEYS} for r in p["projects"]],
        "alloc": {"nodes": a_nodes, "links": a_links},
    }


def compute_swap_recs(state, p: dict) -> list[dict]:
    """Best same-hardware model replacements (~300ms; call lazily, not per HTMX tick)."""
    if p["cost_day"] <= 0 or not p["has_supply"]:
        return []
    return _marginal_model_swap_recommendations(
        state, p["margin_day"], p["value_cloud_day"], p["value_destroyed_day"],
        p["fates"]["served_tokens"],
    )


def econ_payload(state) -> dict:
    """Shared view-model for the economics partials (variant pages + main section)."""
    # Expansion recommendations simulate an extra GPU for every deployed model.
    # They are intentionally fetched through /econ/swaps; computing them here
    # would put that search on every HTMX click even though the main economics
    # section does not render the recommendations.
    p = compute_revenue_projection(state, include_recommendations=False)
    p["value_opportunity_day"] = p["value_served_day"] + p["value_lost_day"]
    preset = cloud_policy.corpo_presets().get(p["corpo_cloud"]) or {}
    p["corpo_cloud_label"] = preset.get("label", p["corpo_cloud"])
    model_token_totals: dict[str, dict] = {}
    for row in p["projects"]:
        for pm in row["per_model_served"]:
            entry = model_token_totals.setdefault(
                pm["name"], {"name": pm["name"], "color": pm["color"], "tokens": 0.0}
            )
            entry["tokens"] += pm["tokens"]
    model_tokens = sorted(model_token_totals.values(), key=lambda d: -d["tokens"])
    charts = _chart_payload(p, model_tokens)
    return {
        "p": p,
        "f": p["fates"],
        "margin": p["margin_day"],
        "model_tokens": model_tokens,
        "aa": quality_to_aa_intelligence,
        # Safe inside <script type="application/json">: escape any '<' in names.
        "charts_json": json.dumps(charts).replace("<", "\\u003c"),
    }


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
    p = compute_revenue_projection(state)
    return render_template(
        "partials/econ/swaps.html",
        swap_recs=compute_swap_recs(state, p),
        gpu_recs=p["recommendations"],
        view=request.args.get("view", "table"),
        panel=panel,
    )


@econ_bp.get("/<slug>")
def variant(slug: str):
    if slug not in VARIANTS:
        abort(404)
    return render_template(f"econ/{slug}.html", **_context(slug))
