"""View model for compare mode: what differs between Config A and Config B.

Compare mode renders both full panels side by side; this summary sits above
them and answers "what did I change, and is the comparison fair?". Pure: it
reads two planner states plus their economics payloads and returns data.
"""

from __future__ import annotations

from typing import Any

from data import GPUS, PRECISION_LABELS

# Settings compared field-by-field, with the label shown when they differ.
_SETTING_LABELS: tuple[tuple[str, str], ...] = (
    ("mu", "Memory budget"),
    ("profiled_non_kv_gb", "Runtime reserve (GB)"),
    ("prefill_bw_eff", "Prefill bandwidth efficiency"),
    ("prefill_comp_eff", "Prefill compute efficiency"),
    ("prefill_overhead", "Prefill overhead"),
    ("prefill_paged_oh", "Prefill paged-KV overhead"),
    ("prefill_ar_overlap", "Prefill all-reduce overlap"),
    ("decode_bw_eff", "Decode bandwidth efficiency"),
    ("decode_comp_eff", "Decode compute efficiency"),
    ("decode_overhead", "Decode overhead"),
    ("decode_paged_oh", "Decode paged-KV overhead"),
    ("decode_ar_overlap", "Decode all-reduce overlap"),
    ("decode_sched_budget", "Scheduler budget"),
    ("kv_slack", "KV slack reserve"),
    ("moe_imbalance", "MoE imbalance"),
    ("pd_interference", "Prefill/decode interference"),
    ("spec_acceptance", "Speculative acceptance"),
    ("corpo_cloud", "Cloud gateway"),
    ("price_driven_demand", "Price-driven demand"),
)

# Total demand differing by more than this share means A and B are not being
# scored against the same work.
DEMAND_MISMATCH_SHARE = 0.05


def _pools(state) -> list[str]:
    return sorted(
        f"{pool.count}× {GPUS[pool.gpu_type].name} at ${pool.cost_per_gpu_hour:.2f}/GPU·hr"
        for pool in state.gpus
    )


def _models(state) -> list[str]:
    return sorted(
        f"{assignment.model.name} · {PRECISION_LABELS.get(assignment.prec, assignment.prec)}"
        f" · {assignment.gpu_count} GPU{'s' if assignment.gpu_count != 1 else ''}"
        for assignment in state.models
        if assignment.gpu_count > 0
    )


def _use_cases(state) -> list[tuple[str, float]]:
    return sorted(
        (project.name, round(float(project.scale_value or 0.0), 6)) for project in state.projects
    )


def _fmt_setting(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def compare_summary(state_a, state_b, econ_a: dict, econ_b: dict) -> dict[str, Any]:
    """Rows that differ, a list of identical areas, and demand-fairness facts."""
    diffs: list[dict[str, Any]] = []
    same: list[str] = []

    pools_a, pools_b = _pools(state_a), _pools(state_b)
    if pools_a != pools_b:
        diffs.append({"label": "Cluster", "a": pools_a, "b": pools_b})
    else:
        same.append("cluster (" + "; ".join(pools_a) + ")" if pools_a else "cluster")

    models_a, models_b = _models(state_a), _models(state_b)
    if models_a != models_b:
        only_a = [m for m in models_a if m not in models_b]
        only_b = [m for m in models_b if m not in models_a]
        diffs.append({"label": "Models", "a": only_a or ["same"], "b": only_b or ["same"]})
    else:
        same.append("models")

    uc_a, uc_b = _use_cases(state_a), _use_cases(state_b)
    if uc_a != uc_b:
        names_a = {name for name, _ in uc_a}
        names_b = {name for name, _ in uc_b}
        changed = [name for name, scale in uc_a if name in names_b and (name, scale) not in uc_b]
        diffs.append(
            {
                "label": "Use cases",
                "a": sorted(names_a - names_b) + [f"{n}: different scale" for n in changed],
                "b": sorted(names_b - names_a) or (["same set"] if changed else []),
            }
        )
    else:
        same.append(f"{len(uc_a)} use cases")

    setting_diffs = []
    for key, label in _SETTING_LABELS:
        va, vb = getattr(state_a, key, None), getattr(state_b, key, None)
        if va != vb:
            setting_diffs.append({"label": label, "a": [_fmt_setting(va)], "b": [_fmt_setting(vb)]})
    if setting_diffs:
        diffs.extend(setting_diffs)
    else:
        same.append("memory, engine tuning and demand settings")

    pa, pb = econ_a["p"], econ_b["p"]
    demand_a = econ_a["f"]["total_tokens"]
    demand_b = econ_b["f"]["total_tokens"]
    larger = max(demand_a, demand_b)
    mismatch = larger > 0 and abs(demand_a - demand_b) / larger > DEMAND_MISMATCH_SHARE
    equal_demand = not (
        getattr(state_a, "projection_include_latent", True)
        or getattr(state_b, "projection_include_latent", True)
    )
    return {
        "diffs": diffs,
        "same": same,
        "demand_a": demand_a,
        "demand_b": demand_b,
        "latent_a": pa["latent_active_tokens_day"],
        "latent_b": pb["latent_active_tokens_day"],
        "demand_mismatch": mismatch,
        "equal_demand": equal_demand,
    }
