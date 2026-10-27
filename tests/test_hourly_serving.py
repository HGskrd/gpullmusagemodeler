"""Hour-by-hour serving, batch work filling troughs, and sizing vs price-driven demand."""

from __future__ import annotations

import math

from data.use_cases import DAY_SHAPES
from engine.economics import compute_revenue_projection
from state import GpuPool, ModelAssignment, PlannerState, Project


def _state(*projects: Project) -> PlannerState:
    return PlannerState(
        gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
        models=[ModelAssignment(2, "qwen38-27b", 1, 1, 1, 1, "bf16")],
        projects=list(projects),
    )


def _project(uid: int, tokens_day: float, **kw) -> Project:
    kw.setdefault("difficulty", 0.25)
    kw.setdefault("wtp_per_m", 50.0)
    kw.setdefault("min_success_rate", 0.5)
    kw.setdefault("quality_floor", 0.0)
    return Project(uid, f"Use case {uid}", tokens_day=tokens_day, **kw)


def _full_load_tokens(project: Project) -> float:
    """Useful tokens/day the deployment serves for this request shape at 24 h full load:
    a flat demand far above capacity, measured by what gets served."""
    flood = _project(project.uid, 1e15, day_shape="flat", batch_eligible=project.batch_eligible)
    projection = compute_revenue_projection(_state(flood), include_recommendations=False)
    return next(r for r in projection["projects"] if r["project"].uid == project.uid)["served"]


def _served_share(projection: dict, uid: int) -> float:
    row = next(r for r in projection["projects"] if r["project"].uid == uid)
    return row["served"] / row["tokens_day"]


def _hourly_served_share(weights: list[float], demand_over_peak_cap: float) -> float:
    """Closed form: demand follows `weights`; capacity per hour is the peak hour's need
    at demand_over_peak_cap = 1. Returns served / demand."""
    peak = max(weights)
    served = sum(min(demand_over_peak_cap * w / peak, 1.0) for w in weights)
    demand = sum(demand_over_peak_cap * w / peak for w in weights)
    return served / demand


def test_saturated_interactive_demand_is_served_hour_by_hour():
    weights = DAY_SHAPES["workday"]["weights"]
    peak_over_mean = max(weights) / (sum(weights) / len(weights))
    probe = _project(1, 1.0)
    full_load = _full_load_tokens(probe)
    # 1.5x what the busiest hour can take: quiet hours still run at their own demand.
    demand = 1.5 * full_load / peak_over_mean
    projection = compute_revenue_projection(
        _state(_project(1, demand)), include_recommendations=False
    )
    expected = _hourly_served_share(weights, 1.5)
    assert math.isclose(_served_share(projection, 1), expected, rel_tol=1e-6)
    # The old daily-total model would have served only 1/1.5 of demand.
    assert _served_share(projection, 1) > 1 / 1.5 + 0.15


def test_batch_work_fills_the_hours_interactive_traffic_leaves_free():
    weights = DAY_SHAPES["workday"]["weights"]
    peak_over_mean = max(weights) / (sum(weights) / len(weights))
    full_load = _full_load_tokens(_project(1, 1.0))
    interactive = full_load / peak_over_mean  # exactly fills the busiest hour
    batch = full_load - interactive  # exactly the troughs left over
    projection = compute_revenue_projection(
        _state(_project(1, interactive), _project(2, batch * 0.99, batch_eligible=True)),
        include_recommendations=False,
    )
    assert math.isclose(_served_share(projection, 1), 1.0, rel_tol=1e-6)
    assert math.isclose(_served_share(projection, 2), 1.0, rel_tol=1e-6)
    assert projection["models"][0]["utilization"] > 0.98


def test_interactive_demand_is_served_before_batch_regardless_of_value():
    full_load = _full_load_tokens(_project(1, 1.0))
    projection = compute_revenue_projection(
        _state(
            _project(1, full_load, wtp_per_m=1.0),
            _project(2, full_load, wtp_per_m=500.0, batch_eligible=True),
        ),
        include_recommendations=False,
    )
    assert _served_share(projection, 1) > _served_share(projection, 2)


def test_sizing_mode_serves_the_entered_workload_whatever_its_price_ceiling():
    cheap = _project(
        1, 1_000_000, wtp_per_m=1e-6, latent_jobs_day=5_000_000, unlock_price_per_m=99.0
    )
    sizing = compute_revenue_projection(_state(cheap), include_recommendations=False)
    assert sizing["fates"]["total_tokens"] == 1_000_000
    assert math.isclose(_served_share(sizing, 1), 1.0, rel_tol=1e-9)

    priced = _state(cheap)
    priced.price_driven_demand = True
    chargeback = compute_revenue_projection(priced, include_recommendations=False)
    # Price-driven: the ceiling stops local serving and latent demand can appear.
    assert _served_share(chargeback, 1) < 1e-9
