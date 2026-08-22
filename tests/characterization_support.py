"""Stable inputs and serialization helpers for estimator characterization tests."""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from engine.economics import compute_revenue_projection
from placement import resolve_deployment, retune_models
from planner_service import deserialize_scenario
from presentation.charts import (
    chart_aggregate,
    chart_asr_quality,
    chart_data_processing,
    chart_decode,
    chart_embedding_quality,
    chart_embedding_throughput,
    chart_pareto,
    chart_processing_pareto,
    chart_realtime_capacity,
    chart_user_experience,
    chart_user_pareto,
)
from presentation.reports import format_projection_report
from state import GpuPool, ModelAssignment, PlannerState, Project

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).with_name("fixtures") / "characterization"


def _default_state() -> PlannerState:
    with (ROOT / "default_scenario.json").open(encoding="utf-8") as handle:
        state, _compare = deserialize_scenario(json.load(handle))
    gpu_uids = {}
    for index, pool in enumerate(state.gpus, start=1):
        gpu_uids[pool.uid] = 1_000 + index
        pool.uid = 1_000 + index
    for index, assignment in enumerate(state.models, start=1):
        assignment.uid = 2_000 + index
        assignment.gpu_uid = gpu_uids[assignment.gpu_uid]
    for index, project in enumerate(state.projects, start=1):
        project.uid = 3_000 + index
    return state


def projection_cases() -> dict[str, PlannerState]:
    """Default plus deliberately different demand/supply edge cases."""
    no_supply = PlannerState(
        projects=[
            Project(
                101,
                "No affordable route",
                difficulty=0.80,
                tokens_day=2_500_000,
                wtp_per_m=0.01,
                requires=frozenset({"reasoning"}),
                min_success_rate=0.95,
                quality_floor=0.90,
            )
        ]
    )

    constrained = PlannerState(
        gpus=[GpuPool(201, "H100", 2, cost_per_gpu_hour=1.25, country="FR")],
        models=[ModelAssignment(202, "q27", 201, 2, 2, 1, "bf16")],
        projects=[
            Project(
                203,
                "Interactive coding",
                difficulty=0.55,
                tokens_day=75_000_000,
                wtp_per_m=5.0,
                requires=frozenset({"tools"}),
                min_success_rate=0.75,
                quality_floor=0.60,
                prefix_hit_rate=0.45,
                in_pre="Code",
                out_pre="Code",
                quality_domain="coding",
                quality_weights={"coding": 0.8, "reasoning": 0.2},
            ),
            Project(
                204,
                "Nightly summaries",
                difficulty=0.25,
                tokens_day=250_000_000,
                wtp_per_m=1.5,
                batch_eligible=True,
                min_success_rate=0.70,
                quality_floor=0.45,
                latent_jobs_day=100_000_000,
                unlock_price_per_m=0.75,
                prefix_hit_rate=0.20,
                in_pre="Long doc",
                out_pre="Long doc",
                quality_domain="long_context",
            ),
        ],
        projection_night_batching=True,
    )
    retune_models(constrained, preserve_existing=False)

    mixed_fit = PlannerState(
        gpus=[GpuPool(301, "H100", 4, cost_per_gpu_hour=1.75, country="US")],
        models=[
            ModelAssignment(302, "q08", 301, 1, 1, 1, "fp8"),
            ModelAssignment(303, "q27", 301, 3, 1, 1, "bf16"),
        ],
        projects=[
            Project(
                304,
                "Bulk classification",
                difficulty=0.10,
                tokens_day=900_000_000,
                wtp_per_m=0.5,
                batch_eligible=True,
                min_success_rate=0.70,
                quality_floor=0.30,
                latent_jobs_day=400_000_000,
                unlock_price_per_m=0.15,
                in_pre="Classify",
                out_pre="Classify",
            ),
            Project(
                305,
                "Vision-only extraction",
                difficulty=0.35,
                tokens_day=50_000_000,
                wtp_per_m=4.0,
                requires=frozenset({"images"}),
                min_success_rate=0.80,
                quality_floor=0.55,
                quality_domain="vision",
            ),
        ],
        projection_day_shape="flat",
    )
    retune_models(mixed_fit, preserve_existing=False)

    return {
        "default_scenario": _default_state(),
        "no_supply": no_supply,
        "constrained": constrained,
        "mixed_fit": mixed_fit,
    }


def chart_state() -> PlannerState:
    """One state spanning text, embedding, and realtime/ASR chart families."""
    state = PlannerState(
        gpus=[GpuPool(401, "H100", 8, cost_per_gpu_hour=1.5, country="FR")],
        models=[
            ModelAssignment(402, "q08", 401, 1, 1, 1, "fp8"),
            ModelAssignment(403, "q27", 401, 2, 1, 1, "bf16"),
            ModelAssignment(404, "mxbai-embed-xsmall-v1", 401, 1, 1, 1, "bf16"),
            ModelAssignment(405, "denseon", 401, 2, 1, 1, "bf16"),
            ModelAssignment(406, "voxtral-realtime-mini-4b", 401, 1, 1, 1, "bf16"),
        ],
    )
    retune_models(state, preserve_existing=False)
    return state


def projection_outputs() -> dict[str, dict]:
    return {name: compute_revenue_projection(state) for name, state in projection_cases().items()}


def report_outputs() -> dict[str, str]:
    """Plain-text projection report for each projection case.

    The report is what users copy out of the planner, so its wording and its
    numbers are both part of the observable output.
    """
    cases = projection_cases()
    outputs = {name: format_projection_report(state, None) for name, state in cases.items()}
    # One A/B pair, since the compare panel takes a different formatting branch.
    names = sorted(cases)
    outputs["compare_default_vs_no_supply"] = format_projection_report(
        cases[names[0]], cases[names[1]]
    )
    return outputs


def chart_outputs() -> dict[str, list[dict]]:
    state = chart_state()
    deployment = resolve_deployment(state)
    batch_sizes = [1, 4, 16]
    return {
        "chart_decode": chart_decode(state, batch_sizes, deployment=deployment),
        "chart_pareto": chart_pareto(state, deployment=deployment),
        "chart_user_pareto": chart_user_pareto(state, batch_sizes, deployment=deployment),
        "chart_aggregate": chart_aggregate(state, batch_sizes, deployment=deployment),
        "chart_data_processing": chart_data_processing(state, batch_sizes),
        "chart_embedding_throughput": chart_embedding_throughput(state, batch_sizes),
        "chart_embedding_quality": chart_embedding_quality(state),
        "chart_processing_pareto": chart_processing_pareto(state, batch_sizes),
        "chart_user_experience": chart_user_experience(state),
        "chart_realtime_capacity": chart_realtime_capacity(state, batch_sizes),
        "chart_asr_quality": chart_asr_quality(state),
    }


def _canonical(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            **{field.name: _canonical(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        # libm and reduction order can move the final few binary digits across
        # Python versions and platforms. Goldens should catch estimator changes,
        # not differences far below the planner's observable precision.
        return float(format(value, ".12g"))
    return value


def canonical_json(value: Any) -> str:
    return (
        json.dumps(
            _canonical(value),
            allow_nan=True,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
