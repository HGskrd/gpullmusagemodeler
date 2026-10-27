"""View models for the demand and economics projection.

``econ_payload()`` builds the projection plus its chart JSON for both the
standalone ``/econ/`` pages and the economics section of the main planner.
Pure: it takes planner state and returns data, with no request context.
"""

import json

import cloud_policy
from data import quality_to_aa_intelligence
from engine.economics import (
    _marginal_model_swap_recommendations,
    compute_revenue_projection,
)

_PROJECT_JSON_KEYS = (
    "name",
    "tokens_day",
    "served_pct",
    "spilled_pct",
    "leaked_pct",
    "destroyed_pct",
    "value_served",
    "value_spilled",
    "value_leaked",
    "value_destroyed",
    "wtp_per_m",
    "cloud_pm",
    "cloud_blocked",
    "cheapest_effective_pm",
)


def _chart_payload(p: dict, model_tokens: list[dict]) -> dict:
    """Sankey/bridge/stack payloads. Node colors are semantic keys resolved
    client-side from CSS variables, except per-model colors which are catalog
    values passed through as-is."""
    f = p["fates"]
    margin = p["margin_day"]

    m_nodes = [
        {"name": "Total opportunity", "depth": 0, "c": "total"},
        {"name": "Lost", "depth": 1, "c": "destroyed"},
        {"name": "Cloud spend", "depth": 1, "c": "spilled"},
        {"name": "Served", "depth": 1, "c": "served"},
    ]
    m_links = [
        {"source": "Total opportunity", "target": "Served", "value": p["value_served_day"]},
        {"source": "Total opportunity", "target": "Cloud spend", "value": p["value_cloud_day"]},
        {
            "source": "Total opportunity",
            "target": "Lost",
            "value": p["value_destroyed_day"],
        },
    ]
    if p["value_spilled_day"] > 0:
        m_nodes.append({"name": "Sent to cloud", "depth": 2, "c": "spilled"})
        m_links.append(
            {
                "source": "Cloud spend",
                "target": "Sent to cloud",
                "value": p["value_spilled_day"],
            }
        )
    if p["value_leaked_day"] > 0:
        m_nodes.append({"name": "Wrong model", "depth": 2, "c": "leaked"})
        m_links.append(
            {
                "source": "Cloud spend",
                "target": "Wrong model",
                "value": p["value_leaked_day"],
            }
        )
    if p["cost_day"] > 0 and p["value_served_day"] > 0:
        if margin > 0:
            m_nodes.append({"name": "Net margin", "depth": 2, "c": "margin"})
            m_links.append({"source": "Served", "target": "Net margin", "value": margin})
        m_nodes.append({"name": "Covers cluster cost", "depth": 2, "c": "cost"})
        m_links.append(
            {
                "source": "Served",
                "target": "Covers cluster cost",
                "value": p["value_served_day"] - margin if margin > 0 else p["value_served_day"],
            }
        )

    # The sankey keeps input order within each column (layoutIterations 0), so
    # list children in the same order as their parents: no crossing bands.
    m_order = [
        "Total opportunity",
        "Served",
        "Cloud spend",
        "Lost",
        "Net margin",
        "Covers cluster cost",
        "Sent to cloud",
        "Wrong model",
    ]
    m_nodes.sort(key=lambda n: m_order.index(str(n["name"])))

    t_nodes = [
        {"name": "Active demand", "depth": 0, "c": "total"},
        {"name": "Served", "depth": 1, "c": "served"},
        {"name": "Sent to cloud", "depth": 1, "c": "spilled"},
        {"name": "Wrong model", "depth": 1, "c": "leaked"},
        {"name": "Lost", "depth": 1, "c": "destroyed"},
    ]
    t_links = [
        {"source": "Active demand", "target": "Served", "value": f["served_tokens"]},
        {"source": "Active demand", "target": "Sent to cloud", "value": f["spilled_tokens"]},
        {"source": "Active demand", "target": "Wrong model", "value": f["leaked_tokens"]},
        {
            "source": "Active demand",
            "target": "Lost",
            "value": f["destroyed_tokens"],
        },
    ]
    for mt in model_tokens:
        if mt["tokens"] > 0:
            t_nodes.append({"name": mt["name"], "depth": 2, "color": mt["color"]})
            t_links.append({"source": "Served", "target": mt["name"], "value": mt["tokens"]})

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
                "total": p["value_opportunity_day"],
                "cloud": p["value_cloud_day"],
                "captured": p["value_served_day"],
                "destroyed": p["value_destroyed_day"],
                "totalName": "Opportunity",
            },
            "tokens": {
                "total": f["total_tokens"],
                "cloud": f["spilled_tokens"] + f["leaked_tokens"],
                "captured": f["served_tokens"],
                "destroyed": f["destroyed_tokens"],
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
        state,
        p["margin_day"],
        p["value_cloud_day"],
        p["value_destroyed_day"],
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
