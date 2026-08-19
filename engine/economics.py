"""Internal-market projection and marginal deployment recommendations."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Optional, cast

import cloud_policy
from calc import (
    DATA_BATCH_SIZES,
    GPU_POWER_UTILIZATION,
    LATENT_UNLOCK_STEEPNESS,
    MARGINAL_RECOMMENDATION_LIMIT,
    NIGHT_HOURS,
    DeploymentPeakResult,
    _batch_axis_sweep,
    _iter_resolved_models,
    avg_dist,
    compute_data,
    compute_data_capacity,
    default_strategy,
    spec_runtime_for,
    valid_strategies,
)
from data import (
    CORPO_CLOUD_DEFAULT,
    DAY_SHAPES,
    DEFAULT_COUNTRY,
    DIST_PRESETS,
    GPU,
    INPUT_BUCKETS,
    MODELS,
    OUTPUT_BUCKETS,
    QUALITY_DOMAIN_LABELS,
    Model,
    carbon_intensity_avg,
    effective_quality,
    model_domain_anchor,
    model_profile_quality,
    model_profile_success_rate,
    normalize_quality_domain,
    normalize_quality_weights,
    quality_weights_label,
    required_quality,
    success_rate,
)


@dataclass(frozen=True)
class DemandFates:
    """Portfolio demand outcomes, with useful-token units made explicit."""

    total_tokens_day: float
    served_tokens_day: float
    spilled_tokens_day: float
    leaked_tokens_day: float
    destroyed_tokens_day: float

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DemandFates":
        return cls(
            total_tokens_day=row["total_tokens"],
            served_tokens_day=row["served_tokens"],
            spilled_tokens_day=row["spilled_tokens"],
            leaked_tokens_day=row["leaked_tokens"],
            destroyed_tokens_day=row["destroyed_tokens"],
        )

    def to_dict(self) -> dict[str, float]:
        total = self.total_tokens_day
        return {
            "total_tokens": total,
            "served_tokens": self.served_tokens_day,
            "spilled_tokens": self.spilled_tokens_day,
            "leaked_tokens": self.leaked_tokens_day,
            "destroyed_tokens": self.destroyed_tokens_day,
            "served_pct": (self.served_tokens_day / total * 100.0) if total > 0 else 0.0,
            "spilled_pct": (self.spilled_tokens_day / total * 100.0) if total > 0 else 0.0,
            "leaked_pct": (self.leaked_tokens_day / total * 100.0) if total > 0 else 0.0,
            "destroyed_pct": (self.destroyed_tokens_day / total * 100.0) if total > 0 else 0.0,
        }


@dataclass(frozen=True)
class ProjectOutcome:
    """Typed units for one project while retaining its compatibility payload."""

    name: str
    demand_tokens_day: float
    served_tokens_day: float
    spilled_tokens_day: float
    leaked_tokens_day: float
    destroyed_tokens_day: float
    internal_cost_usd_day: float
    value_served_usd_day: float
    margin_usd_day: float
    co2_gco2_day: float
    _payload: dict[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ProjectOutcome":
        return cls(
            name=row["name"],
            demand_tokens_day=row["tokens_day"],
            served_tokens_day=row["served"],
            spilled_tokens_day=row["spilled"],
            leaked_tokens_day=row["leaked"],
            destroyed_tokens_day=row["destroyed"],
            internal_cost_usd_day=row["internal_cost_day"],
            value_served_usd_day=row["value_served"],
            margin_usd_day=row["margin_day"],
            co2_gco2_day=row["co2_kg_day"] * 1000.0,
            _payload=row,
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


@dataclass(frozen=True)
class ModelUtilization:
    """Typed capacity, cost, and environmental result for one deployment."""

    name: str
    capacity_tokens_day: float
    served_tokens_day: float
    utilization_fraction: float
    internal_usd_per_million_tokens: float
    gpu_cost_usd_day: float
    co2_gco2_day: float
    _payload: dict[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ModelUtilization":
        return cls(
            name=row["name"],
            capacity_tokens_day=row["daily_tokens_cap"],
            served_tokens_day=row["served_tokens"],
            utilization_fraction=row["utilization"],
            internal_usd_per_million_tokens=row["internal_pm"],
            gpu_cost_usd_day=row["gpu_cost_day"],
            co2_gco2_day=row["co2_kg_day"] * 1000.0,
            _payload=row,
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


@dataclass(frozen=True)
class ProjectionResult:
    """Typed economics result with a lossless legacy-dict adapter."""

    projects: tuple[ProjectOutcome, ...]
    models: tuple[ModelUtilization, ...]
    fates: DemandFates
    value_served_usd_day: float
    value_cloud_usd_day: float
    value_destroyed_usd_day: float
    cost_usd_day: float
    margin_usd_day: float
    co2_gco2_day: float
    _payload: dict[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectionResult":
        return cls(
            projects=tuple(ProjectOutcome.from_dict(row) for row in payload["projects"]),
            models=tuple(ModelUtilization.from_dict(row) for row in payload["models"]),
            fates=DemandFates.from_dict(payload["fates"]),
            value_served_usd_day=payload["value_served_day"],
            value_cloud_usd_day=payload["value_cloud_day"],
            value_destroyed_usd_day=payload["value_destroyed_day"],
            cost_usd_day=payload["cost_day"],
            margin_usd_day=payload["margin_day"],
            co2_gco2_day=payload["co2_kg_day_total"] * 1000.0,
            _payload=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self._payload)
        payload["projects"] = [project.to_dict() for project in self.projects]
        payload["models"] = [model.to_dict() for model in self.models]
        payload["fates"] = self.fates.to_dict()
        return payload


def _model_kind_for_swap(model: Model) -> str:
    if getattr(model, "embedding_profile", None) is not None:
        return "embedding"
    if getattr(model, "realtime_profile", None) is not None:
        return "asr"
    return "llm"


def _project_workload_profile(project, fallback: dict) -> dict:
    """Average in/out lengths for one project's declared workload shape.

    The capacity model still uses the aggregate workload to estimate shared GPU supply, but
    routing economics need the project's own shape. Otherwise a short classification stream
    inherits the blended portfolio's long-output token tax and can look falsely priced out.
    """
    in_preset = DIST_PRESETS.get(getattr(project, "in_pre", "")) or DIST_PRESETS["Chat"]
    out_preset = DIST_PRESETS.get(getattr(project, "out_pre", "")) or DIST_PRESETS["Chat"]
    in_len: float = avg_dist(in_preset["in"], INPUT_BUCKETS)
    out_len: float = avg_dist(out_preset["out"], OUTPUT_BUCKETS)
    if in_len <= 0 or out_len <= 0:
        in_len = float(fallback["in_len"])
        out_len = float(fallback["out_len"])
    return {
        "in_len": in_len,
        "out_len": out_len,
        "tokens_per_request": max(1.0, in_len + out_len),
    }


def _swap_candidate_shortlist(current: Model, state, current_kind: str, per_slot_cap: int):
    current_quality, _, _ = _portfolio_domain_quality(current, state.projects)
    candidates = []
    for cand_key, cand in MODELS.items():
        if (
            cand_key == current.key
            or getattr(cand, "hidden", False)
            or _model_kind_for_swap(cand) != current_kind
        ):
            continue
        portfolio_quality, _, _ = _portfolio_domain_quality(cand, state.projects)
        candidates.append((cand_key, cand, portfolio_quality))

    cap = max(1, int(per_slot_cap))
    nearest_n = max(1, cap // 2)
    quality_n = max(1, cap // 4)
    efficient_n = max(1, cap - nearest_n - quality_n)
    selected: list[tuple[str, Model, float]] = []
    seen: set[str] = set()
    groups = (
        sorted(candidates, key=lambda row: (abs(row[2] - current_quality), row[0]))[:nearest_n],
        sorted(
            candidates,
            key=lambda row: (
                -_portfolio_domain_quality(row[1], state.projects)[2],
                -row[2],
                row[0],
            ),
        )[:quality_n],
        sorted(
            candidates,
            key=lambda row: (
                row[1].active_params / max(float(row[1].token_efficiency), 1e-6),
                -row[2],
                row[0],
            ),
        )[:efficient_n],
    )
    for group in groups:
        for row in group:
            if row[0] not in seen:
                seen.add(row[0])
                selected.append(row)
    if len(selected) < cap:
        for row in sorted(candidates, key=lambda item: (abs(item[2] - current_quality), item[0])):
            if row[0] not in seen:
                seen.add(row[0])
                selected.append(row)
            if len(selected) >= cap:
                break
    return selected[:cap]


def _workload_profile(state) -> dict:
    """Average in/out lengths from the planner's distributions — a single workload for all models."""
    in_len = avg_dist(state.in_dist, INPUT_BUCKETS)
    out_len = avg_dist(state.out_dist, OUTPUT_BUCKETS)
    return {
        "in_len": in_len,
        "out_len": out_len,
        "tokens_per_request": in_len + out_len,
    }


def _build_model_supply(state, profile, prefix_hit_rate, peak_factor_eff) -> list[dict]:
    """For each deployed model, compute peak RPS, sustainable tokens/day, and internal $/M."""
    tokens_per_req = max(1.0, profile["tokens_per_request"])
    pool_rate = {gp.uid: gp.cost_per_gpu_hour * 24.0 for gp in state.gpus}
    pool_country = {gp.uid: getattr(gp, "country", DEFAULT_COUNTRY) for gp in state.gpus}
    day_shape = (
        DAY_SHAPES.get(getattr(state, "projection_day_shape", "workday")) or DAY_SHAPES["workday"]
    )
    day_weights = cast(list[float], day_shape["weights"]) or [1.0] * 24
    night_weights = [1.0 if h in NIGHT_HOURS else 0.0 for h in range(24)]
    supply = []
    for am, gpu in _iter_resolved_models(state):
        if (
            getattr(am.model, "is_realtime_only", False)
            or getattr(am.model, "embedding_profile", None) is not None
        ):
            continue
        cap = compute_data_capacity(
            am.model,
            (am.prefill_tp, am.prefill_pp, am.prefill_dp),
            (am.tp, am.pp, am.dp),
            profile["in_len"],
            profile["out_len"],
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            prefix_hit_rate,
            state.prefill_efficiency,
            state.decode_efficiency,
            spec_runtime_for(state, am, am.model),
        )
        batch_sizes = _batch_axis_sweep([cap], DATA_BATCH_SIZES)
        best = _best_deployment_result_for_model(
            state, am, gpu, profile["in_len"], profile["out_len"], batch_sizes
        )
        peak_rps = best.rps if (best and best.rps > 0) else 0.0
        # Sustainable daily token capacity: honor peak-hour headroom so we don't promise
        # throughput the day-shape can't actually sustain without thrashing.
        daily_tokens_cap = (
            peak_rps * 3600.0 * 24.0 * tokens_per_req / peak_factor_eff if peak_rps > 0 else 0.0
        )
        gpu_cost_day = pool_rate.get(am.gpu_uid, 0.0) * am.gpu_count
        internal_pm = (gpu_cost_day * 1e6 / daily_tokens_cap) if daily_tokens_cap > 0 else math.inf
        country = pool_country.get(am.gpu_uid, DEFAULT_COUNTRY)
        grid_day = carbon_intensity_avg(country, day_weights)
        grid_night = carbon_intensity_avg(country, night_weights)
        supply.append(
            {
                "am": am,
                "am_uid": am.uid,
                "gpu": gpu,
                "gpu_uid": am.gpu_uid,
                "gpu_count": am.gpu_count,
                "model": am.model,
                "quality": float(am.model.quality),
                "effective_quality": effective_quality(am.model),
                "quality_confidence": min(
                    max(float(getattr(am.model, "quality_confidence", 1.0)), 0.0), 1.0
                ),
                "token_efficiency": max(float(am.model.token_efficiency), 1e-6),
                "peak_rps": peak_rps,
                "daily_tokens_cap": daily_tokens_cap,
                "remaining_cap": daily_tokens_cap,
                "remaining_fraction": 1.0,
                "used_fraction": 0.0,
                "served_tokens": 0.0,
                "gpu_cost_day": gpu_cost_day,
                "internal_pm": internal_pm,
                # These are accumulated with each project's own task shape while routing.
                "served_tasks": 0.0,
                "served_co2_g_day": 0.0,
                "served_co2_g_night": 0.0,
                "country": country,
                "grid_gco2_per_kwh_day": grid_day,
                "grid_gco2_per_kwh_night": grid_night,
                "runnable": peak_rps > 0,
            }
        )
    return supply


def _actual_token_multiplier(token_efficiency: float, in_len: float, out_len: float) -> float:
    """Actual GPU/cloud tokens consumed per useful workload token.

    Token efficiency is an output-token verbosity proxy. Prompts do not get longer just
    because a model thinks or writes more, so only the output side is scaled.
    """
    eff = max(float(token_efficiency), 1e-6)
    useful = max(float(in_len) + float(out_len), 1.0)
    actual = max(float(in_len), 0.0) + max(float(out_len), 0.0) / eff
    return max(actual / useful, 1e-9)


@dataclass(frozen=True)
class _ProjectionContext:
    state: Any
    profile: dict[str, Any]
    prefix_hit_rate: float
    corpo_cloud: str
    day_shape: dict[str, Any]
    peak_factor: float
    batch_share: float
    night_batching: bool
    peak_factor_eff: float
    supply: tuple[dict[str, Any], ...]
    projects: tuple[Any, ...]
    projects_sorted: tuple[Any, ...] = ()
    project_outcomes: tuple[ProjectOutcome, ...] = ()
    model_utilizations: tuple[ModelUtilization, ...] = ()
    fates: DemandFates | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def build_supply(state) -> _ProjectionContext:
    """Resolve workload shape, peak demand, and per-deployment daily supply."""
    profile = _workload_profile(state)
    prefix_hit_rate = min(max(state.prefix_hit_rate, 0.0), 1.0)
    corpo_cloud = getattr(state, "corpo_cloud", CORPO_CLOUD_DEFAULT)
    day_shape = DAY_SHAPES.get(state.projection_day_shape) or DAY_SHAPES["workday"]
    weights = cast(list[float], day_shape["weights"]) or [1.0]
    mean_w = sum(weights) / len(weights)
    peak_factor = (max(weights) / mean_w) if mean_w > 0 else 1.0

    projects = tuple(state.projects)
    total_demand = sum(max(0.0, p.tokens_day) for p in projects)
    batch_demand = sum(max(0.0, p.tokens_day) for p in projects if p.batch_eligible)
    batch_share = (batch_demand / total_demand) if total_demand > 0 else 0.0
    night_batching = bool(state.projection_night_batching)
    peak_factor_eff = (
        (1.0 - batch_share) * peak_factor + batch_share if night_batching else peak_factor
    )
    peak_factor_eff = max(peak_factor_eff, 1.0)
    supply = _build_model_supply(state, profile, prefix_hit_rate, peak_factor_eff)
    return _ProjectionContext(
        state=state,
        profile=profile,
        prefix_hit_rate=prefix_hit_rate,
        corpo_cloud=corpo_cloud,
        day_shape=day_shape,
        peak_factor=peak_factor,
        batch_share=batch_share,
        night_batching=night_batching,
        peak_factor_eff=peak_factor_eff,
        supply=tuple(supply),
        projects=projects,
    )


def classify_demand(context: _ProjectionContext) -> _ProjectionContext:
    """Order demand by economic value and compatibility scarcity."""
    projects_sorted = sorted(
        context.projects,
        key=lambda p: (
            -float(p.wtp_per_m),
            -required_quality(
                float(getattr(p, "difficulty", 0.5)),
                float(getattr(p, "min_success_rate", 0.85)),
                quality_floor=float(getattr(p, "quality_floor", 0.0)),
            ),
            -len(getattr(p, "requires", frozenset()) or frozenset()),
            p.uid,
        ),
    )
    return replace(context, projects_sorted=tuple(projects_sorted))


def allocate_capacity(context: _ProjectionContext) -> _ProjectionContext:
    """Route useful project demand through compatible affordable capacity."""
    state = context.state
    profile = context.profile
    corpo_cloud = context.corpo_cloud
    peak_factor = context.peak_factor
    night_batching = context.night_batching
    projects = context.projects
    projects_sorted = context.projects_sorted
    supply = [dict(model_supply) for model_supply in context.supply]
    routed: dict[int, dict] = {}
    for p in projects_sorted:
        difficulty = float(getattr(p, "difficulty", 0.5))
        slo = float(getattr(p, "min_success_rate", 0.85))
        quality_floor = float(getattr(p, "quality_floor", 0.0))
        quality_domain = normalize_quality_domain(getattr(p, "quality_domain", "general"))
        quality_weights = normalize_quality_weights(
            getattr(p, "quality_weights", None),
            quality_domain,
        )
        required_caps = frozenset(getattr(p, "requires", frozenset()) or frozenset())
        project_prefix_hit_rate = min(max(float(getattr(p, "prefix_hit_rate", 0.0)), 0.0), 1.0)
        project_profile = _project_workload_profile(p, profile)
        cloud_info, cloud_pm = _cloud_price_per_m_in_preset(
            difficulty,
            slo,
            quality_floor,
            project_profile,
            project_prefix_hit_rate,
            corpo_cloud,
            required_caps,
            quality_domain,
        )
        cloud_blocked = cloud_info is None
        cloud_details = cloud_info or {}
        wtp = float(p.wtp_per_m)
        total = max(0.0, float(p.tokens_day))

        # Candidate list with capability + success-rate gates. `useful tokens` = work the
        # project needs done; token efficiency affects generated/output tokens only, so the
        # actual GPU tokens burned per useful token depends on this workload's input/output mix.
        candidates: list[dict] = []
        runnable_seen = False
        capability_compatible_seen = False
        floor_compatible_seen = False
        slo_compatible_seen = False
        for me in supply:
            if not me["runnable"]:
                continue
            runnable_seen = True
            if not (required_caps <= me["model"].capabilities):
                continue
            capability_compatible_seen = True
            project_quality = model_profile_quality(me["model"], quality_weights, quality_domain)
            if project_quality + 1e-9 < quality_floor:
                continue
            floor_compatible_seen = True
            sr = model_profile_success_rate(
                me["model"], difficulty, quality_weights, quality_domain
            )
            if sr + 1e-9 < slo:
                continue
            slo_compatible_seen = True
            token_mult = _actual_token_multiplier(
                me["token_efficiency"],
                float(project_profile["in_len"]),
                float(project_profile["out_len"]),
            )
            retry_mult = 1.0 / max(sr, 1e-6)
            project_peak_factor = 1.0 if (night_batching and p.batch_eligible) else peak_factor
            shape_daily_cap, shape_peak_rps = _deployment_capacity_for_profile(
                state,
                me["am"],
                me["gpu"],
                project_profile,
                project_peak_factor,
                project_prefix_hit_rate,
            )
            if shape_daily_cap <= 0:
                continue
            shape_internal_pm = me["gpu_cost_day"] * 1e6 / shape_daily_cap
            project_tpt = tokens_per_task(
                me["model"],
                int(project_profile["in_len"]),
                int(project_profile["out_len"]),
            )
            project_tokens_per_sec = shape_peak_rps * project_tpt
            candidates.append(
                {
                    "me": me,
                    "success_rate": sr,
                    "retry_mult": retry_mult,
                    "token_mult": token_mult * retry_mult,
                    "shape_daily_cap": shape_daily_cap,
                    "shape_peak_rps": shape_peak_rps,
                    "shape_internal_pm": shape_internal_pm,
                    "effective_pm": shape_internal_pm * token_mult * retry_mult,
                    "tokens_per_task": project_tpt,
                    "co2_g_per_task_day": co2_g_per_task(
                        me["gpu"],
                        me["gpu_count"],
                        project_tpt,
                        project_tokens_per_sec,
                        me["grid_gco2_per_kwh_day"],
                    ),
                    "co2_g_per_task_night": co2_g_per_task(
                        me["gpu"],
                        me["gpu_count"],
                        project_tpt,
                        project_tokens_per_sec,
                        me["grid_gco2_per_kwh_night"],
                    ),
                }
            )
        candidates.sort(key=lambda c: c["effective_pm"])

        # Latent demand activates smoothly around the unlock price. This keeps the configured
        # unlock as the midpoint while avoiding discontinuous portfolio demand jumps.
        baseline_tokens = total
        latent_pool = max(0.0, float(getattr(p, "latent_jobs_day", 0.0)))
        unlock_price = float(getattr(p, "unlock_price_per_m", 0.0))
        cheapest_pm = candidates[0]["effective_pm"] if candidates else float("inf")
        latent_activation = (
            latent_activation_share(cheapest_pm, unlock_price) if candidates else 0.0
        )
        latent_active = latent_pool * latent_activation
        latent_unlocked = latent_active > 1.0
        total = baseline_tokens + latent_active

        served = 0.0  # useful tokens delivered (project-perspective)
        per_model_served: list[tuple[dict, float, float, int]] = []
        internal_cost = 0.0
        co2_g_day_project = 0.0
        # Internal price cap: never charge above WTP; if cloud is reachable, also cap at cloud
        # (otherwise the project would just buy from cloud instead of paying us more).
        price_cap = wtp if cloud_blocked else min(wtp, cloud_pm)
        has_affordable_candidate = any(c["effective_pm"] <= price_cap + 1e-9 for c in candidates)
        for c in candidates:
            me = c["me"]
            if me["remaining_fraction"] <= 0:
                continue
            if c["effective_pm"] > price_cap:
                continue
            useful_remaining = total - served
            if useful_remaining <= 0:
                break
            shape_remaining = me["remaining_fraction"] * c["shape_daily_cap"]
            useful_take = min(useful_remaining, shape_remaining / c["token_mult"])
            if useful_take <= 0:
                continue
            actual_take = useful_take * c["token_mult"]
            fraction_used = actual_take / c["shape_daily_cap"]
            me["remaining_fraction"] = max(0.0, me["remaining_fraction"] - fraction_used)
            me["used_fraction"] = min(1.0, me["used_fraction"] + fraction_used)
            me["remaining_cap"] = me["daily_tokens_cap"] * me["remaining_fraction"]
            me["served_tokens"] += actual_take
            per_model_served.append((me, useful_take, actual_take, c["success_rate"]))
            internal_cost += (actual_take / 1e6) * c["shape_internal_pm"]
            tpt_m = c["tokens_per_task"]
            if tpt_m > 0:
                attempt_tasks = actual_take / tpt_m
                co2_day = attempt_tasks * c["co2_g_per_task_day"]
                co2_night = attempt_tasks * c["co2_g_per_task_night"]
                co2_g_day_project += co2_day
                me["served_tasks"] += attempt_tasks
                me["served_co2_g_day"] += co2_day
                me["served_co2_g_night"] += co2_night
            served += useful_take

        unserved = max(0.0, total - served)
        spilled, leaked, destroyed = 0.0, 0.0, 0.0
        if unserved > 0:
            if cloud_blocked:
                # No model in the corpo catalog can serve this tier — there's no cloud to flee
                # to. The work is dropped regardless of WTP.
                destroyed = unserved
            else:
                # "Had a usable home" means a matching model exists below the project price cap.
                # If none was served, the reason can still be capacity exhaustion caused by
                # higher-priority workloads, so classify that as spill instead of wrong-model leak.
                if served > 0 or has_affordable_candidate:
                    spilled = unserved
                else:
                    leaked = unserved
                if cloud_pm > wtp:
                    destroyed = spilled + leaked
                    spilled, leaked = 0.0, 0.0

        # Value of internally served tokens reflects the cheapest substitute (cloud price);
        # when cloud is blocked there's no substitute, so use WTP as the realized value.
        value_basis = wtp if cloud_blocked else cloud_pm
        # useful_t is completed work; retry cost and capacity were already charged.
        value_served = sum(
            (useful_t / 1e6) * value_basis for _, useful_t, _, _sr in per_model_served
        )
        baseline_tokens_per_task = max(float(project_profile["tokens_per_request"]), 1.0)
        tasks_served_day = served / baseline_tokens_per_task
        co2_g_per_task_project = (
            (co2_g_day_project / tasks_served_day) if tasks_served_day > 0 else 0.0
        )
        routed[p.uid] = {
            "project": p,
            "name": p.name,
            "difficulty": difficulty,
            "tokens_day": total,
            "cloud_pm": 0.0 if cloud_blocked else cloud_pm,
            "cloud_label": "blocked — no compatible cloud"
            if cloud_blocked
            else cloud_details["label"],
            "cloud_vendor": "" if cloud_blocked else cloud_details["vendor"],
            "cloud_regions": () if cloud_blocked else cloud_details.get("regions", ()),
            "cloud_grid_gco2_per_kwh": 0.0
            if cloud_blocked
            else cloud_details.get("grid_gco2_per_kwh", 0.0),
            "cloud_price_source": ""
            if cloud_blocked
            else cloud_details.get("price_source", "catalog"),
            "cloud_blocked": cloud_blocked,
            "served": served,
            "spilled": spilled,
            "leaked": leaked,
            "destroyed": destroyed,
            "served_pct": (served / total * 100.0) if total > 0 else 0.0,
            "spilled_pct": (spilled / total * 100.0) if total > 0 else 0.0,
            "leaked_pct": (leaked / total * 100.0) if total > 0 else 0.0,
            "destroyed_pct": (destroyed / total * 100.0) if total > 0 else 0.0,
            "internal_cost_day": internal_cost,
            "quality_floor": quality_floor,
            "quality_domain": quality_domain,
            "quality_domain_label": QUALITY_DOMAIN_LABELS[quality_domain],
            "quality_weights": quality_weights,
            "quality_mix_label": quality_weights_label(quality_weights, quality_domain),
            "prefix_hit_rate": project_prefix_hit_rate,
            "value_served": value_served,
            "value_spilled": (spilled / 1e6) * value_basis,
            "value_leaked": (leaked / 1e6) * value_basis,
            "value_destroyed": (destroyed / 1e6) * value_basis,
            "margin_day": value_served - internal_cost,
            "tasks_served_day": tasks_served_day,
            "co2_kg_day": co2_g_day_project / 1000.0,
            "co2_g_per_task_avg": co2_g_per_task_project,
            "wtp_per_m": wtp,
            "requires": sorted(required_caps),
            "min_success_rate": slo,
            "has_compatible": bool(candidates),
            "cap_blocked_for_project": runnable_seen and not capability_compatible_seen,
            "quality_floor_blocked_for_project": (
                capability_compatible_seen and not floor_compatible_seen
            ),
            "slo_blocked_for_project": floor_compatible_seen and not slo_compatible_seen,
            "capacity_blocked_for_project": slo_compatible_seen and not candidates,
            # True when *any* of the actually-serving candidates isn't a near-perfect fit
            # (success_rate < ~1.0) — used by the UI to flag "served, but via a stretched model".
            "any_suboptimal": any(sr < 0.99 for *_, sr in per_model_served),
            "any_served": served > 0,
            "baseline_tokens_day": baseline_tokens,
            "latent_jobs_day": latent_pool,
            "unlock_price_per_m": unlock_price,
            "latent_unlocked": latent_unlocked,
            "latent_active_tokens": latent_active,
            "latent_activation_pct": latent_activation * 100.0,
            "cheapest_effective_pm": (0.0 if math.isinf(cheapest_pm) else cheapest_pm),
            # Diagnostic hint: cheapest is within ~1.5× of unlock price but not yet under it.
            "latent_close_to_unlock": (
                latent_pool > 0
                and unlock_price > 0
                and latent_activation < 0.50
                and bool(candidates)
                and cheapest_pm <= unlock_price * 1.5 + 1e-9
            ),
            "per_model_served": [
                {
                    "am_uid": me["am_uid"],
                    "name": me["model"].name,
                    "tokens": useful_t,
                    "actual_tokens": actual_t,
                    "success_rate": sr,
                    "effective_quality": model_profile_quality(
                        me["model"], quality_weights, quality_domain
                    ),
                    "quality_anchor": " · ".join(
                        (
                            anchor.benchmark
                            if (anchor := model_domain_anchor(me["model"], domain)) is not None
                            else f"{QUALITY_DOMAIN_LABELS[domain]} global fallback"
                        )
                        for domain in quality_weights
                    ),
                    "quality_components": [
                        {
                            "domain": domain,
                            "label": QUALITY_DOMAIN_LABELS[domain],
                            "weight": weight,
                            "effective_quality": effective_quality(me["model"], domain),
                            "benchmark": (
                                anchor.benchmark
                                if (anchor := model_domain_anchor(me["model"], domain)) is not None
                                else "Global quality fallback"
                            ),
                            "anchored": model_domain_anchor(me["model"], domain) is not None,
                        }
                        for domain, weight in quality_weights.items()
                    ],
                    "retry_mult": 1.0 / max(sr, 1e-6),
                    "color": me["model"].color,
                }
                for me, useful_t, actual_t, sr in per_model_served
            ],
        }
    project_rows = [routed[p.uid] for p in projects if p.uid in routed]
    return replace(
        context,
        supply=tuple(supply),
        project_outcomes=tuple(ProjectOutcome.from_dict(row) for row in project_rows),
    )


def price_outcomes(context: _ProjectionContext) -> _ProjectionContext:
    """Aggregate project fates, substitute value, cost, and margin."""
    project_rows = [outcome.to_dict() for outcome in context.project_outcomes]
    total_tokens = sum(row["tokens_day"] for row in project_rows)
    total_served = sum(row["served"] for row in project_rows)
    total_spilled = sum(row["spilled"] for row in project_rows)
    total_leaked = sum(row["leaked"] for row in project_rows)
    total_destroyed = sum(row["destroyed"] for row in project_rows)
    value_served = sum(row["value_served"] for row in project_rows)
    value_spilled = sum(row["value_spilled"] for row in project_rows)
    value_leaked = sum(row["value_leaked"] for row in project_rows)
    value_destroyed = sum(row["value_destroyed"] for row in project_rows)
    value_cloud = value_spilled + value_leaked
    value_lost = value_cloud + value_destroyed
    value_opportunity = value_served + value_lost
    cost_day = sum(pool.cost_per_gpu_hour * 24.0 * pool.count for pool in context.state.gpus)
    metrics = {
        "value_served": value_served,
        "value_spilled": value_spilled,
        "value_leaked": value_leaked,
        "value_destroyed": value_destroyed,
        "value_cloud": value_cloud,
        "value_lost": value_lost,
        "cost_day": cost_day,
        "cost_per_m_served": (cost_day * 1e6 / total_served) if total_served > 0 else 0.0,
        "margin_day": value_served - cost_day,
        "revenue_multiple": (value_served / cost_day) if cost_day > 0 else 0.0,
        "token_coverage": (total_served / total_tokens) if total_tokens > 0 else 0.0,
        "value_capture_rate": (value_served / value_opportunity if value_opportunity > 0 else 0.0),
        "baseline_tokens_total": sum(row["baseline_tokens_day"] for row in project_rows),
        "latent_active_tokens_total": sum(row["latent_active_tokens"] for row in project_rows),
    }
    fates = DemandFates(
        total_tokens_day=total_tokens,
        served_tokens_day=total_served,
        spilled_tokens_day=total_spilled,
        leaked_tokens_day=total_leaked,
        destroyed_tokens_day=total_destroyed,
    )
    return replace(context, fates=fates, metrics=metrics)


def calculate_environmental_impact(context: _ProjectionContext) -> _ProjectionContext:
    """Calculate routed-energy impact and per-deployment utilization rows."""
    supply = context.supply
    co2_numer_gco2 = sum(row["served_co2_g_day"] for row in supply)
    served_attempt_tasks = sum(row["served_tasks"] for row in supply)
    model_rows = []
    total_cap_tokens_day = sum(row["daily_tokens_cap"] for row in supply)
    total_actual_served_tokens_day = sum(row["served_tokens"] for row in supply)
    total_gpu_weight = sum(max(row["gpu_count"], 0) for row in supply)
    time_utilization = (
        sum(row.get("used_fraction", 0.0) * max(row["gpu_count"], 0) for row in supply)
        / total_gpu_weight
        if total_gpu_weight > 0
        else 0.0
    )
    for row in supply:
        cap = row["daily_tokens_cap"]
        util = row.get("used_fraction", 0.0)
        saturated = cap > 0 and row.get("remaining_fraction", 1.0) <= 0.01
        served_tasks = row["served_tasks"]
        tokens_per_task = row["served_tokens"] / served_tasks if served_tasks > 0 else 0.0
        co2_day_gco2 = row["served_co2_g_day"] / served_tasks if served_tasks > 0 else 0.0
        co2_night_gco2 = row["served_co2_g_night"] / served_tasks if served_tasks > 0 else 0.0
        model_rows.append(
            {
                "am_uid": row["am_uid"],
                "model": row["model"],
                "name": row["model"].name,
                "color": row["model"].color,
                "quality": row["quality"],
                "effective_quality": row["effective_quality"],
                "quality_confidence": row["quality_confidence"],
                "token_efficiency": row["token_efficiency"],
                "gpu_count": row["gpu_count"],
                "peak_rps": row["peak_rps"],
                "daily_tokens_cap": cap,
                "served_tokens": row["served_tokens"],
                "utilization": util,
                "internal_pm": 0.0 if math.isinf(row["internal_pm"]) else row["internal_pm"],
                "internal_input_pm": (
                    0.0 if math.isinf(row["internal_pm"]) else row["internal_pm"]
                ),
                "internal_output_pm": (
                    0.0
                    if math.isinf(row["internal_pm"])
                    else row["internal_pm"] / max(row["token_efficiency"], 1e-6)
                ),
                "gpu_cost_day": row["gpu_cost_day"],
                "tokens_per_task": tokens_per_task,
                "country": row.get("country", DEFAULT_COUNTRY),
                "grid_gco2_per_kwh_day": row.get("grid_gco2_per_kwh_day", 0.0),
                "grid_gco2_per_kwh_night": row.get("grid_gco2_per_kwh_night", 0.0),
                "co2_g_per_task_day": co2_day_gco2,
                "co2_g_per_task_night": co2_night_gco2,
                "co2_kg_day": row["served_co2_g_day"] / 1000.0,
                "saturated": saturated,
                "runnable": row["runnable"],
                "status": (
                    "NOT RUNNABLE"
                    if not row["runnable"]
                    else "SATURATED"
                    if saturated
                    else "IDLE"
                    if util < 0.05
                    else "OK"
                ),
            }
        )
    metrics = context.metrics | {
        "co2_kg_day_total": co2_numer_gco2 / 1000.0,
        "co2_g_per_task_avg": (
            co2_numer_gco2 / served_attempt_tasks if served_attempt_tasks > 0 else 0.0
        ),
        "total_cap_tokens_day": total_cap_tokens_day,
        "actual_served_tokens": total_actual_served_tokens_day,
        "utilization": time_utilization,
    }
    return replace(
        context,
        model_utilizations=tuple(ModelUtilization.from_dict(row) for row in model_rows),
        metrics=metrics,
    )


def summarize_projection(
    context: _ProjectionContext, include_recommendations: bool = True
) -> ProjectionResult:
    """Build the stable public projection result from the completed stages."""
    if context.fates is None:
        raise ValueError("Projection outcomes must be priced before summarization.")
    metrics = context.metrics
    recommendations = (
        _marginal_gpu_recommendations(
            context.state,
            metrics["margin_day"],
            metrics["value_cloud"],
            metrics["value_destroyed"],
            context.fates.served_tokens_day,
        )
        if include_recommendations
        else []
    )
    payload = {
        "ready": bool(context.supply) and bool(context.project_outcomes),
        "has_supply": bool(context.supply),
        "has_demand": bool(context.project_outcomes),
        "corpo_cloud": context.corpo_cloud,
        "day_shape_label": context.day_shape["label"],
        "day_shape_note": context.day_shape.get("note", ""),
        "peak_factor": context.peak_factor,
        "peak_factor_eff": context.peak_factor_eff,
        "batch_share": context.batch_share,
        "night_batching": context.night_batching,
        "projects": [outcome.to_dict() for outcome in context.project_outcomes],
        "models": [utilization.to_dict() for utilization in context.model_utilizations],
        "fates": context.fates.to_dict(),
        "value_served_day": metrics["value_served"],
        "value_spilled_day": metrics["value_spilled"],
        "value_leaked_day": metrics["value_leaked"],
        "value_destroyed_day": metrics["value_destroyed"],
        "value_cloud_day": metrics["value_cloud"],
        "value_lost_day": metrics["value_lost"],
        "avoidable_cloud_outflow_day": metrics["value_cloud"],
        "cost_day": metrics["cost_day"],
        "cost_per_m_served": metrics["cost_per_m_served"],
        "co2_kg_day_total": metrics["co2_kg_day_total"],
        "co2_g_per_task_avg": metrics["co2_g_per_task_avg"],
        "margin_day": metrics["margin_day"],
        "coverage": metrics["revenue_multiple"],
        "revenue_multiple": metrics["revenue_multiple"],
        "token_coverage": metrics["token_coverage"],
        "value_capture_rate": metrics["value_capture_rate"],
        "baseline_tokens_day": metrics["baseline_tokens_total"],
        "latent_active_tokens_day": metrics["latent_active_tokens_total"],
        "recommendations": recommendations,
        "total_gpus_used": sum(row["gpu_count"] for row in context.supply),
        "total_gpus": sum(pool.count for pool in context.state.gpus),
        "total_cap_tokens_day": metrics["total_cap_tokens_day"],
        "actual_served_tokens": metrics["actual_served_tokens"],
        "utilization": metrics["utilization"],
        "workload_in_len": context.profile["in_len"],
        "workload_out_len": context.profile["out_len"],
    }
    return ProjectionResult.from_dict(payload)


def compute_projection_result(state, include_recommendations: bool = True) -> ProjectionResult:
    """Run the pure projection stages and return the typed result."""
    supplied = build_supply(state)
    classified = classify_demand(supplied)
    allocated = allocate_capacity(classified)
    priced = price_outcomes(allocated)
    impacted = calculate_environmental_impact(priced)
    return summarize_projection(impacted, include_recommendations)


def compute_revenue_projection(state, include_recommendations: bool = True) -> dict:
    """Compatibility adapter returning the historical nested-dict payload."""
    return compute_projection_result(state, include_recommendations).to_dict()


def _marginal_model_swap_recommendations(
    state,
    base_margin: float,
    base_cloud: float,
    base_destroyed: float,
    base_served_tokens: float,
    per_slot_cap: int = 20,
) -> list[dict]:
    """Estimate the best same-hardware model replacements for current deployments.

    Complement to _marginal_gpu_recommendations: instead of adding a GPU to an
    existing assignment, swap the deployed model for another catalog model of the
    same kind on the same GPUs, retune topology, and rescore the projection.
    Candidates combine the nearest portfolio-domain quality, the strongest domain
    quality, and the smallest inference work size, then are capped per slot to bound
    runtime. Scoring mirrors the GPU expansion recommender.
    """
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for am, gpu in _iter_resolved_models(state):
        key = (am.uid, am.gpu_uid)
        if key in seen:
            continue
        seen.add(key)
        current = am.model
        current_kind = _model_kind_for_swap(current)
        pool = _swap_candidate_shortlist(current, state, current_kind, per_slot_cap)
        current_portfolio_quality, quality_mix, current_anchor_share = _portfolio_domain_quality(
            current, state.projects
        )
        evaluated = 0
        for cand_key, cand, candidate_portfolio_quality in pool:
            if evaluated >= per_slot_cap:
                break
            sim = copy.deepcopy(state)
            sim_am = next((m for m in sim.models if m.uid == am.uid), None)
            sim_gp = next((gp for gp in sim.gpus if gp.uid == am.gpu_uid), None)
            if sim_am is None or sim_gp is None:
                continue
            sim_am.model_key = cand_key
            # Keep the deployment's precision when the candidate supports it;
            # fall back to bf16 rather than discarding a viable swap outright.
            chosen_prec = None
            chosen_spec = None
            for prec in (sim_am.prec, "bf16"):
                sim_am.prec = prec
                spec = spec_runtime_for(sim, sim_am, sim_am.model)
                if valid_strategies(
                    sim_am.model,
                    sim_am.gpu_count,
                    sim_gp.gpu,
                    sim.mu,
                    sim.profiled_non_kv_gb,
                    prec,
                    spec,
                ):
                    chosen_prec = prec
                    chosen_spec = spec
                    break
            if chosen_prec is None:
                continue
            evaluated += 1
            strategy = default_strategy(
                sim_am.model,
                sim_am.gpu_count,
                sim_gp.gpu,
                sim.mu,
                sim.profiled_non_kv_gb,
                chosen_prec,
                chosen_spec,
            )
            sim_am.tp, sim_am.pp, sim_am.dp = strategy
            sim_am.prefill_tp, sim_am.prefill_pp, sim_am.prefill_dp = strategy

            projected = compute_revenue_projection(sim, include_recommendations=False)
            candidate_portfolio_quality, _, candidate_anchor_share = _portfolio_domain_quality(
                cand, state.projects
            )
            margin_gain = projected["margin_day"] - base_margin
            cloud_reduced = max(0.0, base_cloud - projected["value_cloud_day"])
            destroyed_reduced = max(0.0, base_destroyed - projected["value_destroyed_day"])
            score = margin_gain + 0.25 * cloud_reduced + 0.50 * destroyed_reduced
            if score <= 0 and margin_gain <= 0 and cloud_reduced <= 0 and destroyed_reduced <= 0:
                continue
            rows.append(
                {
                    "current_key": am.model_key,
                    "current_name": current.name,
                    "candidate_key": cand_key,
                    "candidate_name": cand.name,
                    "gpu_name": sim_gp.gpu.name,
                    "gpu_count": sim_am.gpu_count,
                    "current_quality": current_portfolio_quality,
                    "candidate_quality": candidate_portfolio_quality,
                    "current_global_quality": effective_quality(current),
                    "candidate_global_quality": effective_quality(cand),
                    "quality_mix": quality_mix,
                    "current_anchor_share": current_anchor_share,
                    "candidate_anchor_share": candidate_anchor_share,
                    "prec": chosen_prec,
                    "margin_gain_day": margin_gain,
                    "cloud_reduced_day": cloud_reduced,
                    "destroyed_reduced_day": destroyed_reduced,
                    "served_gain_tokens": max(
                        0.0, projected["fates"]["served_tokens"] - base_served_tokens
                    ),
                    "margin_before_day": base_margin,
                    "margin_after_day": projected["margin_day"],
                    "cloud_before_day": base_cloud,
                    "cloud_after_day": projected["value_cloud_day"],
                    "destroyed_before_day": base_destroyed,
                    "destroyed_after_day": projected["value_destroyed_day"],
                    "served_before_tokens": base_served_tokens,
                    "served_after_tokens": projected["fates"]["served_tokens"],
                    "score": score,
                }
            )

    rows.sort(key=lambda r: (-r["score"], -r["margin_gain_day"], r["candidate_name"]))
    return rows[:MARGINAL_RECOMMENDATION_LIMIT]


def _deployment_capacity_for_profile(
    state,
    am,
    gpu: GPU,
    profile: dict,
    peak_factor: float,
    prefix_hit_rate: Optional[float] = None,
) -> tuple[float, float]:
    """Return shape-specific daily token capacity and peak RPS.

    Capacity is recomputed for each workload shape. Routing consumes a shared
    normalized deployment-time fraction, so long and short requests no longer
    spend an interchangeable blended token budget.
    """
    in_len = int(profile["in_len"])
    out_len = int(profile["out_len"])
    prefix_rate = (
        state.prefix_hit_rate
        if prefix_hit_rate is None
        else min(max(float(prefix_hit_rate), 0.0), 1.0)
    )
    cap = compute_data_capacity(
        am.model,
        (am.prefill_tp, am.prefill_pp, am.prefill_dp),
        (am.tp, am.pp, am.dp),
        in_len,
        out_len,
        gpu,
        state.mu,
        state.profiled_non_kv_gb,
        am.prec,
        prefix_rate,
        state.prefill_efficiency,
        state.decode_efficiency,
        spec_runtime_for(state, am, am.model),
    )
    batch_sizes = _batch_axis_sweep([cap], DATA_BATCH_SIZES)
    best = _best_deployment_result_for_model(
        state,
        am,
        gpu,
        in_len,
        out_len,
        batch_sizes,
        prefix_rate,
    )
    peak_rps = best.rps if best and best.rps > 0 else 0.0
    tokens_per_request = max(float(profile["tokens_per_request"]), 1.0)
    daily_tokens = (
        peak_rps * 86400.0 * tokens_per_request / max(float(peak_factor), 1.0)
        if peak_rps > 0
        else 0.0
    )
    return daily_tokens, peak_rps


def _cloud_price_per_m_in_preset(
    difficulty: float,
    min_success: float,
    quality_floor: float,
    profile: dict,
    prefix_hit_rate: float,
    preset_name: str,
    required_capabilities: frozenset[str] = frozenset(),
    quality_domain: str = "general",
) -> tuple[Optional[dict], float]:
    """Cheapest cloud model in the active corpo preset that can serve a project with the
    given (difficulty, min_success_rate). ``quality_domain`` is accepted so the local and
    cloud paths share one project contract; cloud entries currently fall back to their
    global catalog quality until provider-domain anchors are added. Effective $/M is computed apples-to-apples with
    on-prem: sticker price × (1 / token_efficiency). A cloud is eligible only if
    success_rate(cloud.quality, difficulty) ≥ min_success_rate.

    Returns (cloud_info_or_None, effective_price_per_m). None when no compatible cloud
    exists in the catalog — i.e. spillover is *blocked* for this project."""
    in_len = float(profile["in_len"])
    out_len = float(profile["out_len"])
    cached = in_len * min(max(prefix_hit_rate, 0.0), 1.0)
    uncached = max(0.0, in_len - cached)
    tokens_per_req = max(1.0, in_len + out_len)

    best: Optional[tuple[float, dict]] = None
    for key, cloud in cloud_policy.effective_corpo_models(preset_name):
        if not (required_capabilities <= frozenset(cloud.get("capabilities", ()))):
            continue
        cloud_quality = float(cloud.get("quality", 0.5))
        cloud_eff = max(float(cloud.get("token_efficiency", 1.0)), 1e-6)
        if cloud_quality + 1e-9 < quality_floor:
            continue
        cloud_success = success_rate(cloud_quality, difficulty)
        if cloud_success + 1e-9 < min_success:
            continue
        threshold = max(float(cloud.get("long_context_threshold_tokens", 0.0) or 0.0), 0.0)
        long_context_pricing = threshold > 0.0 and in_len > threshold

        def tier_price(field: str, base_field: str) -> float:
            value = cloud.get(field) if long_context_pricing else None
            return max(float(cloud[base_field] if value is None else value), 0.0)

        input_price = tier_price("long_context_in_per_m", "in_per_m")
        cached_input_price = tier_price("long_context_cached_in_per_m", "cached_in_per_m")
        output_price = tier_price("long_context_out_per_m", "out_per_m")
        sticker = (
            (uncached / 1e6) * input_price
            + (cached / 1e6) * cached_input_price
            + ((out_len / cloud_eff) / 1e6) * output_price
        )
        # Token efficiency affects generated tokens, not the fixed prompt payload.
        # Retry-adjust cloud and on-prem routes symmetrically. A route with
        # success probability p consumes 1/p attempts per completed useful task.
        price_pm = sticker / (tokens_per_req / 1e6) / max(cloud_success, 1e-6)
        if best is None or price_pm < best[0]:
            best = (
                price_pm,
                cloud
                | {
                    "key": key,
                    "success_rate": cloud_success,
                    "long_context_pricing_applied": long_context_pricing,
                    "effective_in_per_m": input_price,
                    "effective_cached_in_per_m": cached_input_price,
                    "effective_out_per_m": output_price,
                },
            )

    if best is None:
        return None, math.inf
    return best[1], best[0]


def _marginal_gpu_recommendations(
    state,
    base_margin: float,
    base_cloud: float,
    base_destroyed: float,
    base_served_tokens: float,
) -> list[dict]:
    """Estimate the best one-GPU expansions for currently deployed models.

    This stays inside calc.py to avoid importing state.py back into the module that state.py
    already imports. It therefore simulates only growth of existing assignments and retunes
    their topology with calc.py's local strategy helper.
    """
    seen: set[tuple[int, int]] = set()
    rows: list[dict] = []
    for am, gpu in _iter_resolved_models(state):
        if getattr(am.model, "embedding_profile", None) is not None:
            continue
        key = (am.uid, am.gpu_uid)
        if key in seen:
            continue
        seen.add(key)

        sim = copy.deepcopy(state)
        sim_am = next((m for m in sim.models if m.uid == am.uid), None)
        sim_gp = next((gp for gp in sim.gpus if gp.uid == am.gpu_uid), None)
        if sim_am is None or sim_gp is None:
            continue

        sim_gp.count += 1
        sim_am.gpu_count += 1
        sim_spec = spec_runtime_for(sim, sim_am, sim_am.model)
        strategy = default_strategy(
            sim_am.model,
            sim_am.gpu_count,
            sim_gp.gpu,
            sim.mu,
            sim.profiled_non_kv_gb,
            sim_am.prec,
            sim_spec,
        )
        if not valid_strategies(
            sim_am.model,
            sim_am.gpu_count,
            sim_gp.gpu,
            sim.mu,
            sim.profiled_non_kv_gb,
            sim_am.prec,
            sim_spec,
        ):
            continue
        sim_am.tp, sim_am.pp, sim_am.dp = strategy
        sim_am.prefill_tp, sim_am.prefill_pp, sim_am.prefill_dp = strategy

        projected = compute_revenue_projection(sim, include_recommendations=False)
        margin_gain = projected["margin_day"] - base_margin
        cloud_reduced = max(0.0, base_cloud - projected["value_cloud_day"])
        destroyed_reduced = max(0.0, base_destroyed - projected["value_destroyed_day"])
        score = margin_gain + 0.25 * cloud_reduced + 0.50 * destroyed_reduced
        if score <= 0 and margin_gain <= 0 and cloud_reduced <= 0 and destroyed_reduced <= 0:
            continue
        rows.append(
            {
                "model_name": sim_am.model.name,
                "gpu_name": sim_gp.gpu.name,
                "gpu_uid": sim_gp.uid,
                "am_uid": sim_am.uid,
                "added_gpus": 1,
                "new_gpu_count": sim_am.gpu_count,
                "margin_gain_day": margin_gain,
                "cloud_reduced_day": cloud_reduced,
                "destroyed_reduced_day": destroyed_reduced,
                "served_gain_tokens": max(
                    0.0, projected["fates"]["served_tokens"] - base_served_tokens
                ),
                "score": score,
            }
        )

    rows.sort(key=lambda r: (-r["score"], -r["margin_gain_day"], r["model_name"]))
    return rows[:MARGINAL_RECOMMENDATION_LIMIT]


def co2_g_per_task(
    gpu: GPU,
    gpu_count: int,
    tokens_per_task_val: float,
    tokens_per_sec: float,
    gco2_per_kwh: float,
    utilization: float = GPU_POWER_UTILIZATION,
) -> float:
    """Grams CO2-eq per task. Energy = cluster_power × tokens_per_task / tokens_per_sec."""
    if tokens_per_sec <= 0 or tokens_per_task_val <= 0:
        return 0.0
    tdp = float(getattr(gpu, "tdp_watts", 0.0))
    if tdp <= 0:
        return 0.0
    cluster_power_w = tdp * gpu_count * utilization
    task_wall_s = tokens_per_task_val / tokens_per_sec
    energy_j = cluster_power_w * task_wall_s
    # 1 kWh = 3.6e6 J; gCO2/kWh × kWh = grams.
    return energy_j * gco2_per_kwh / 3.6e6


def tokens_per_task(model: Model, task_il: int, task_ol: int) -> float:
    """Output tokens scale by 1/token_efficiency (verbose models emit more to finish a task)."""
    eff = max(float(getattr(model, "token_efficiency", 1.0)), 1e-6)
    return float(task_il) + float(task_ol) / eff


def latent_activation_share(cheapest_pm: float, unlock_price: float) -> float:
    """Smooth latent-demand activation around the configured unlock price.

    A hard threshold makes portfolio demand jump discontinuously when a model becomes
    barely cheap enough. This curve keeps the same midpoint semantics: at the unlock
    price, half the latent pool is active; materially cheaper routes approach 100%.
    """
    if unlock_price <= 0 or math.isinf(cheapest_pm) or cheapest_pm <= 0:
        return 0.0
    ratio = unlock_price / cheapest_pm
    return min(max(1.0 / (1.0 + math.exp(-LATENT_UNLOCK_STEEPNESS * (ratio - 1.0))), 0.0), 1.0)


def _portfolio_domain_quality(model: Model, projects) -> tuple[float, str, float]:
    """Demand/value-weighted quality across the active workload domain mix.

    The score is used only to shortlist and explain swap candidates; the full
    projection simulation still decides the recommendation economics.
    """
    weights: dict[str, float] = defaultdict(float)
    for project in projects:
        domain = normalize_quality_domain(getattr(project, "quality_domain", "general"))
        project_weights = normalize_quality_weights(
            getattr(project, "quality_weights", None),
            domain,
        )
        demand = max(0.0, float(getattr(project, "tokens_day", 0.0) or 0.0))
        latent = 0.25 * max(0.0, float(getattr(project, "latent_jobs_day", 0.0) or 0.0))
        value = max(0.01, float(getattr(project, "wtp_per_m", 0.0) or 0.0))
        project_weight = (demand + latent) * value
        for component_domain, component_weight in project_weights.items():
            weights[component_domain] += project_weight * component_weight
    if not weights or sum(weights.values()) <= 0:
        return effective_quality(model), QUALITY_DOMAIN_LABELS["general"], 0.0

    total = sum(weights.values())
    score = (
        sum(effective_quality(model, domain) * weight for domain, weight in weights.items()) / total
    )
    anchored = (
        sum(
            weight
            for domain, weight in weights.items()
            if model_domain_anchor(model, domain) is not None
        )
        / total
    )
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    labels = [
        f"{QUALITY_DOMAIN_LABELS[domain]} {weight / total:.0%}" for domain, weight in ranked[:3]
    ]
    return score, " · ".join(labels), anchored


def _best_deployment_result_for_model(
    state,
    am,
    gpu: GPU,
    in_len: int,
    out_len: int,
    batch_sizes: list[int],
    prefix_hit_rate: Optional[float] = None,
) -> Optional[DeploymentPeakResult]:
    prefix_rate = (
        state.prefix_hit_rate
        if prefix_hit_rate is None
        else min(max(float(prefix_hit_rate), 0.0), 1.0)
    )
    best: Optional[DeploymentPeakResult] = None
    spec = spec_runtime_for(state, am, am.model)
    for bs in batch_sizes:
        result = compute_data(
            am.model,
            (am.prefill_tp, am.prefill_pp, am.prefill_dp),
            (am.tp, am.pp, am.dp),
            bs,
            in_len,
            out_len,
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            prefix_rate,
            state.prefill_efficiency,
            state.decode_efficiency,
            spec,
        )
        if result is None:
            continue

        candidate = DeploymentPeakResult(
            tps=result.tps,
            rps=result.rps,
            batch_size=bs,
            prefill_frac=result.prefill_frac,
        )
        if best is None:
            best = candidate
            continue

        if candidate.tps > best.tps:
            best = candidate
            continue
        if candidate.tps == best.tps and candidate.batch_size < best.batch_size:
            best = candidate
    return best
