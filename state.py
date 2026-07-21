"""Server-side state management for the vLLM planner."""

from __future__ import annotations

import copy
import math
import os
import re
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Optional

from data import (
    GPUS,
    MODELS,
    DIST_PRESETS,
    EMBEDDING_DOC_BUCKETS,
    EMBEDDING_DOC_PRESETS,
    INPUT_BUCKETS,
    OUTPUT_BUCKETS,
    DAY_SHAPES,
    GPU,
    Model,
    PROJECT_PRESETS,
    CORPO_CLOUD_DEFAULT,
    MODEL_CAPABILITIES,
    SCALE_MODELS,
    PRECISIONS,
    PRECISION_LABELS,
    normalize_gpu_count,
    normalize_precision,
)
import cloud_policy
from calc import (
    EfficiencyParams,
    avg_dist,
    valid_strategies,
)


_uid_counter = 0
_uid_lock = threading.Lock()
PROJECT_FIELD_BOUNDS = {
    "tokens_day":          (0.0, 1e12),     # 0 to 1T tokens/day — generous, the UI enforces slider range
    "wtp_per_m":           (0.0, 200.0),    # $/M tokens ceiling
    # Task difficulty ∈ [0,1]. Paired with each model's quality via success_rate() to get a
    # per-(project, model) success probability. Higher = harder task, needs smarter model.
    "difficulty":          (0.0, 1.0),
    # Job-level success-rate floor. Candidates whose success_rate(model.quality, difficulty)
    # falls below this are rejected.
    "min_success_rate":    (0.50, 1.0),
    "quality_floor":       (0.0, 1.0),
    # Latent demand that unlocks only when on-prem $/M drops below the project's unlock threshold.
    "latent_jobs_day":     (0.0, 1e12),
    "unlock_price_per_m":  (0.0, 200.0),
}
ALLOWED_CAPABILITIES = frozenset(MODEL_CAPABILITIES)
ALLOWED_PROJECT_KINDS = frozenset(p["key"] for p in PROJECT_PRESETS) | {"custom"}
VISIBLE_PLOT_MODES = (
    ("userpareto", "User Pareto"),
    ("processingpareto", "Processing Pareto"),
    ("embedquality", "Embedding Quality"),
    ("asrquality", "ASR Quality"),
)
DEFAULT_PLOT_MODE = VISIBLE_PLOT_MODES[0][0]
LEGACY_PLOT_MODE_REDIRECTS = {"realtime": "asrquality", "embedding": "embedquality"}
ALLOWED_PLOT_MODES = frozenset(mode for mode, _ in VISIBLE_PLOT_MODES)
DEFAULT_DAY_SHAPE = "workday"
ALLOWED_DAY_SHAPES = frozenset(DAY_SHAPES)
AUTO_MODEL_STRATEGIES = (
    ("balanced", "Best value / GPU", "Picks the compatible models that capture the most WTP-weighted workload value per assigned GPU."),
    ("coverage", "Most use cases", "Prefers models that satisfy the largest number of active use cases and capability gates."),
    ("quality", "Highest quality", "Prefers the highest effective model quality and SLO margin among models that fit."),
    ("lean", "Fewest GPUs", "Picks the smallest viable model set and leaves unused GPUs free instead of filling every pool."),
    ("throughput", "Most throughput", "Prefers smaller active-parameter and token-efficient models after quality gates are met."),
)
DEFAULT_AUTO_MODEL_STRATEGY = AUTO_MODEL_STRATEGIES[0][0]
AUTO_MODEL_STRATEGY_LABELS = {key: label for key, label, _ in AUTO_MODEL_STRATEGIES}
ALLOWED_AUTO_MODEL_STRATEGIES = frozenset(AUTO_MODEL_STRATEGY_LABELS)
DEFAULT_SCALE_KIND = {
    "model": "linear",
    "label": "Token demand",
    "unit": "M tokens/day",
    "token_multiplier": 1e6,
    "min": 0.0,
    "max": 5000.0,
    "step": 10.0,
    "formula": "millions of tokens/day",
}
PROJECTION_PCT_BOUNDS = {
    # Average share of peak capacity that internal users actually book. Values above 100%
    # represent oversubscribed demand that routing will spill once modeled capacity is full.
    "projection_demand_level":      (0.05, 1.20),
    # Discount offered to users who batch overnight. Drives demand shift via elasticity.
    "projection_night_discount":    (0.0, 0.80),
    # Fraction of demand that is batch-eligible (not real-time/interactive).
    "projection_batch_eligible":    (0.0, 1.0),
    # How responsive internal users are to the discount. shift = min(1, elasticity * discount).
    "projection_elasticity":        (0.0, 4.0),
}


def _next_uid() -> int:
    global _uid_counter
    with _uid_lock:
        _uid_counter += 1
        return _uid_counter


def _env_nonnegative_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def normalize_plot_mode(mode: Optional[str]) -> str:
    if mode in LEGACY_PLOT_MODE_REDIRECTS:
        return LEGACY_PLOT_MODE_REDIRECTS[mode]
    return mode if mode in ALLOWED_PLOT_MODES else DEFAULT_PLOT_MODE


def normalize_day_shape(shape: Optional[str]) -> str:
    return shape if shape in ALLOWED_DAY_SHAPES else DEFAULT_DAY_SHAPE


def normalize_corpo_cloud(name: Optional[str]) -> str:
    return name if name in cloud_policy.corpo_presets() else CORPO_CLOUD_DEFAULT


def normalize_auto_strategy(strategy: Optional[str]) -> str:
    return strategy if strategy in ALLOWED_AUTO_MODEL_STRATEGIES else DEFAULT_AUTO_MODEL_STRATEGY


def _embedding_doc_dist_from_length(seq_len: int) -> list[int]:
    length = max(int(seq_len or 0), 1)
    nearest = min(range(len(EMBEDDING_DOC_BUCKETS)), key=lambda i: abs(EMBEDDING_DOC_BUCKETS[i].length - length))
    dist = [0] * len(EMBEDDING_DOC_BUCKETS)
    dist[nearest] = 100
    return dist


def normalize_embedding_doc_distribution(state: "PlannerState"):
    dist = getattr(state, "embedding_doc_dist", None)
    if not isinstance(dist, list):
        dist = _embedding_doc_dist_from_length(getattr(state, "task_il", 2048))

    values = []
    for i in range(len(EMBEDDING_DOC_BUCKETS)):
        raw = dist[i] if i < len(dist) else 0
        values.append(max(0, int(raw or 0)))
    if not any(values):
        values = list(EMBEDDING_DOC_PRESETS["Doc"])

    state.embedding_doc_dist = values
    state.embedding_doc_pre = (
        getattr(state, "embedding_doc_pre", "Doc")
        if getattr(state, "embedding_doc_pre", "Doc") in EMBEDDING_DOC_PRESETS
        else ""
    )


@dataclass
class GpuPool:
    uid: int
    gpu_type: str
    count: int
    cost_per_gpu_hour: float = 0.0
    country: str = "FR"

    @property
    def gpu(self) -> GPU:
        return GPUS[self.gpu_type]

    @property
    def cost_per_gpu_day(self) -> float:
        return self.cost_per_gpu_hour * 24.0

    @property
    def pool_cost_day(self) -> float:
        return self.cost_per_gpu_day * self.count


@dataclass
class Project:
    """Demand-side input: a workload stream (tokens/day) with a task-difficulty axis and a
    ceiling price ($/M tokens). Drives the internal-market routing. Candidate models are
    scored by success_rate(model.quality, project.difficulty) and filtered by min_success_rate."""
    uid: int
    name: str
    difficulty: float         # ∈ [0,1]; paired with model.quality via success_rate()
    tokens_day: float         # total daily token demand
    wtp_per_m: float          # willingness-to-pay, $/M tokens
    scale_value: Optional[float] = None  # organization-specific scale in the use-case's native unit
    scale_kind: dict[str, Any] = field(default_factory=dict)
    # Built-in use-case definition this instance follows. "custom" means the card owns
    # its definition directly. Scale remains on the instance either way.
    kind_key: str = "custom"
    batch_eligible: bool = False  # if True, batch-shiftable off-peak (works with night batching)
    # Hard capability gates: a model must supply ALL listed capabilities to be eligible.
    requires: frozenset[str] = frozenset()
    # Quality SLO: project rejects any candidate whose success_rate(model.quality, difficulty)
    # falls below this floor.
    min_success_rate: float = 0.85
    # Absolute effective-quality floor. This prevents tiny/uncertain models from clearing very
    # easy tasks only because the sigmoid threshold is low.
    quality_floor: float = 0.0
    # Latent demand — hidden workload that only materializes when on-prem $/M falls at or
    # below unlock_price_per_m. Hard threshold: the pool is all-or-nothing per routing pass.
    latent_jobs_day: float = 0.0
    unlock_price_per_m: float = 0.0
    # Per-project input / output length preset. The aggregate state.in_dist / state.out_dist
    # used by calc.py are a demand-weighted blend across all projects' presets.
    in_pre: str = "Chat"
    out_pre: str = "Chat"

    def __post_init__(self):
        if not self.kind_key:
            self.kind_key = "custom"
        if not isinstance(self.requires, frozenset):
            self.requires = frozenset(c for c in (self.requires or ()) if c in ALLOWED_CAPABILITIES)
        self.difficulty = min(max(float(self.difficulty), 0.0), 1.0)
        self.scale_kind = _normalize_scale_kind({"scale_kind": getattr(self, "scale_kind", {})})
        if getattr(self, "scale_value", None) is None:
            self.scale_value = tokens_to_scale_value(float(self.tokens_day), self.scale_kind)
        else:
            self.scale_value = max(0.0, float(self.scale_value))
            self.tokens_day = scale_value_to_tokens(self.scale_value, self.scale_kind)
        self.quality_floor = min(max(float(getattr(self, "quality_floor", 0.0)), 0.0), 1.0)
        if self.in_pre not in DIST_PRESETS:
            self.in_pre = "Chat"
        if self.out_pre not in DIST_PRESETS:
            self.out_pre = "Chat"


@dataclass
class ModelAssignment:
    uid: int
    model_key: str
    gpu_uid: int
    gpu_count: int
    tp: int
    dp: int
    prec: str
    pp: int = 1
    prefill_tp: Optional[int] = None
    prefill_pp: Optional[int] = None
    prefill_dp: Optional[int] = None

    def __post_init__(self):
        self.prec = normalize_precision(self.prec)
        if self.prefill_tp is None:
            self.prefill_tp = self.tp
        if self.prefill_pp is None:
            self.prefill_pp = self.pp
        if self.prefill_dp is None:
            self.prefill_dp = self.dp

    @property
    def model(self) -> Model:
        return MODELS[self.model_key]

    @property
    def gpu_spec(self) -> Optional[GPU]:
        return None


class ModelAssignmentProxy:
    """Wrap ModelAssignment with resolved GPU metadata."""

    def __init__(
        self,
        assignment: ModelAssignment,
        gpu: Optional[GPU],
        phase: str = "decode",
        prefill_mem=None,
        decode_mem=None,
    ):
        self._assignment = assignment
        self._gpu = gpu
        self._phase = phase
        self.prefill_mem = prefill_mem
        self.decode_mem = decode_mem

    @property
    def assignment(self) -> ModelAssignment:
        return self._assignment

    def __getattr__(self, name):
        if name == "gpu_spec":
            return self._gpu
        if self._phase == "prefill":
            if name == "tp":
                return self._assignment.prefill_tp
            if name == "pp":
                return self._assignment.prefill_pp
            if name == "dp":
                return self._assignment.prefill_dp
        return getattr(self._assignment, name)


@dataclass
class PlannerState:
    gpus: list[GpuPool] = field(default_factory=list)
    models: list[ModelAssignment] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    auto_excluded: list[str] = field(default_factory=list)
    auto_mode: bool = False
    auto_strategy: str = DEFAULT_AUTO_MODEL_STRATEGY
    use_case_defs: list[dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(PROJECT_PRESETS))
    in_dist: list[int] = field(default_factory=lambda: list(DIST_PRESETS["Chat"]["in"]))
    out_dist: list[int] = field(default_factory=lambda: list(DIST_PRESETS["Chat"]["out"]))
    in_pre: str = "Chat"
    out_pre: str = "Chat"
    embedding_doc_dist: list[int] = field(default_factory=lambda: list(EMBEDDING_DOC_PRESETS["Doc"]))
    embedding_doc_pre: str = "Doc"
    mu: float = 0.90
    profiled_non_kv_gb: float = 4.0

    kv_slack: float = 0.02
    moe_imbalance: float = 1.15
    pd_interference: float = 0.0

    prefill_bw_eff: float = 0.80
    prefill_comp_eff: float = 0.75
    prefill_overhead: float = 0.08
    prefill_paged_oh: float = 0.10
    prefill_ar_overlap: float = 0.30

    decode_bw_eff: float = 0.80
    decode_comp_eff: float = 0.75
    decode_overhead: float = 0.08
    decode_paged_oh: float = 0.10
    decode_ar_overlap: float = 0.30
    decode_sched_budget: int = 16384
    
    prefix_hit_rate: float = 0.0
    task_il: int = 2048
    task_ol: int = 32
    mode: str = DEFAULT_PLOT_MODE
    projection_day_shape: str = DEFAULT_DAY_SHAPE
    # Which corpo cloud catalog projects can spill to. "current" = today's procurement reality;
    # "advocated" = what we'd unlock by getting more vendors approved (drives the demand-
    # destruction story when no compatible cloud exists).
    corpo_cloud: str = CORPO_CLOUD_DEFAULT
    # Average booked demand as fraction of planner peak capacity.
    projection_demand_level: float = 0.65
    # Night-batching lever (the "tick the box in LiteLLM" scenario).
    projection_night_batching: bool = False
    projection_night_discount: float = 0.30
    projection_batch_eligible: float = 0.35
    projection_elasticity: float = 2.0

    def __post_init__(self):
        self.mode = normalize_plot_mode(self.mode)
        self.projection_day_shape = normalize_day_shape(self.projection_day_shape)
        self.corpo_cloud = normalize_corpo_cloud(self.corpo_cloud)
        self.auto_strategy = normalize_auto_strategy(self.auto_strategy)

    @property
    def prefill_efficiency(self) -> EfficiencyParams:
        return EfficiencyParams(
            bw_eff=self.prefill_bw_eff,
            comp_eff=self.prefill_comp_eff,
            overhead=self.prefill_overhead,
            kv_slack=self.kv_slack,
            paged_oh=self.prefill_paged_oh,
            ar_overlap=self.prefill_ar_overlap,
            moe_imbalance=self.moe_imbalance,
            pd_interference=self.pd_interference,
        )

    @property
    def decode_efficiency(self) -> EfficiencyParams:
        return EfficiencyParams(
            bw_eff=self.decode_bw_eff,
            comp_eff=self.decode_comp_eff,
            overhead=self.decode_overhead,
            kv_slack=self.kv_slack,
            paged_oh=self.decode_paged_oh,
            ar_overlap=self.decode_ar_overlap,
            moe_imbalance=self.moe_imbalance,
            sched_budget=self.decode_sched_budget,
            pd_interference=self.pd_interference,
        )

    def find_gpu(self, uid: int) -> Optional[GpuPool]:
        return next((g for g in self.gpus if g.uid == uid), None)

    def find_model(self, uid: int) -> Optional[ModelAssignment]:
        return next((m for m in self.models if m.uid == uid), None)

    def used_gpu_for_pool(self, gpu_uid: int) -> int:
        return sum(m.gpu_count for m in self.models if m.gpu_uid == gpu_uid)

    def free_gpu_for_pool(self, gpu_uid: int) -> int:
        gp = self.find_gpu(gpu_uid)
        return gp.count - self.used_gpu_for_pool(gpu_uid) if gp else 0

    def total_gpus(self) -> int:
        return sum(g.count for g in self.gpus)

    def find_project(self, uid: int) -> Optional[Project]:
        return next((p for p in self.projects if p.uid == uid), None)


def create_default_state() -> PlannerState:
    from placement import _retune_model

    state = PlannerState()
    gpu_uid = _next_uid()
    state.gpus.append(GpuPool(gpu_uid, "MI355X", 8))
    state.models.append(ModelAssignment(_next_uid(), "q122", gpu_uid, 4, 2, 2, "bf16"))
    state.models.append(ModelAssignment(_next_uid(), "l70", gpu_uid, 2, 1, 2, "bf16"))
    state.models.append(ModelAssignment(_next_uid(), "q35", gpu_uid, 2, 1, 2, "bf16"))
    for am in state.models:
        _retune_model(state, am)
    # A small, opinionated default project mix so the internal-market story lands immediately.
    for preset_key in ("classify", "chatbot", "coding", "research"):
        _add_project_from_preset(state, preset_key)
    _sync_aggregate_distribution(state)
    return state


def _find_preset(key: str) -> Optional[dict]:
    return next((p for p in PROJECT_PRESETS if p["key"] == key), None)


def _slugify_key(value: str, fallback: str = "use_case") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return slug or fallback


def _unique_use_case_key(state: PlannerState, base: str) -> str:
    existing = {str(d.get("key", "")) for d in get_use_case_defs(state)}
    key = _slugify_key(base)
    if key not in existing:
        return key
    i = 2
    while f"{key}_{i}" in existing:
        i += 1
    return f"{key}_{i}"


def _payload_optional_float(source: dict, key: str) -> Optional[float]:
    try:
        value = source.get(key)
    except AttributeError:
        return None
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _positive_payload_float(source: dict, key: str, default: float) -> float:
    value = _payload_optional_float(source, key)
    if value is None or value <= 0.0:
        return float(default)
    return float(value)


def _normalize_scale_kind(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    nested = raw.get("scale_kind") if isinstance(raw.get("scale_kind"), dict) else {}
    source = {**nested}
    for flat_key, nested_key in (
        ("scale_model", "model"),
        ("scale_label", "label"),
        ("scale_unit", "unit"),
        ("scale_formula", "formula"),
        ("scale_token_multiplier", "token_multiplier"),
        ("tokens_per_scale_unit", "token_multiplier"),
        ("scale_min", "min"),
        ("scale_max", "max"),
        ("scale_step", "step"),
    ):
        if flat_key in raw and nested_key not in source:
            source[nested_key] = raw[flat_key]

    model = str(source.get("model") or DEFAULT_SCALE_KIND["model"]).strip()
    if model not in SCALE_MODELS:
        model = "custom"
    unit = str(source.get("unit") or DEFAULT_SCALE_KIND["unit"]).strip()[:48] or DEFAULT_SCALE_KIND["unit"]
    label = str(source.get("label") or "Scale").strip()[:48] or "Scale"
    formula = str(source.get("formula") or raw.get("scale_hint") or DEFAULT_SCALE_KIND["formula"]).strip()[:180]
    token_multiplier = _positive_payload_float(source, "token_multiplier", DEFAULT_SCALE_KIND["token_multiplier"])
    min_value = _payload_optional_float(source, "min")
    max_value = _payload_optional_float(source, "max")
    step = _positive_payload_float(source, "step", DEFAULT_SCALE_KIND["step"])
    if min_value is None:
        min_value = float(DEFAULT_SCALE_KIND["min"])
    if max_value is None or max_value <= min_value:
        max_value = max(float(DEFAULT_SCALE_KIND["max"]), min_value + step)
    return {
        "model": model,
        "label": label,
        "unit": unit,
        "token_multiplier": token_multiplier,
        "min": float(min_value),
        "max": float(max_value),
        "step": float(step),
        "formula": formula,
    }


def scale_value_to_tokens(scale_value: float, scale_kind: dict[str, Any] | None) -> float:
    kind = _normalize_scale_kind({"scale_kind": scale_kind or {}})
    value = max(0.0, float(scale_value or 0.0))
    factor = max(float(kind["token_multiplier"]), 0.0)
    if kind["model"] == "quadratic":
        return value * value * factor
    return value * factor


def tokens_to_scale_value(tokens_day: float, scale_kind: dict[str, Any] | None) -> float:
    kind = _normalize_scale_kind({"scale_kind": scale_kind or {}})
    tokens = max(0.0, float(tokens_day or 0.0))
    factor = max(float(kind["token_multiplier"]), 1e-9)
    if kind["model"] == "quadratic":
        return math.sqrt(tokens / factor)
    return tokens / factor


def scale_decimals(scale_kind: dict[str, Any] | None) -> int:
    step = _normalize_scale_kind({"scale_kind": scale_kind or {}})["step"]
    if step >= 1:
        return 0
    text = f"{step:.6f}".rstrip("0")
    return max(0, len(text.partition(".")[2]))


def format_scale_value(value: float, scale_kind: dict[str, Any] | None = None) -> str:
    decimals = scale_decimals(scale_kind)
    return f"{float(value):.{decimals}f}"


def project_scale_config(state: PlannerState, proj: Project) -> dict[str, Any]:
    preset = _find_use_case_def(state, getattr(proj, "kind_key", "custom"))
    kind = preset.get("scale_kind") if preset else getattr(proj, "scale_kind", {})
    kind = _normalize_scale_kind({"scale_kind": kind})
    value = getattr(proj, "scale_value", None)
    if value is None:
        value = tokens_to_scale_value(getattr(proj, "tokens_day", 0.0), kind)
    max_value = max(float(kind["max"]), float(value or 0.0))
    return {
        **kind,
        "value": float(value or 0.0),
        "display_value": format_scale_value(float(value or 0.0), kind),
        "decimals": scale_decimals(kind),
        "max": max_value,
        "model_label": SCALE_MODELS.get(kind["model"], SCALE_MODELS["custom"]),
    }


def _bounded_def_value(field_name: str, value: float) -> float:
    return _bounded_project_value(field_name, value)


def _coerce_requires(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    else:
        values = value or ()
    return tuple(c for c in values if c in ALLOWED_CAPABILITIES)


def _normalize_use_case_def(raw: dict[str, Any], fallback_key: str | None = None) -> dict[str, Any]:
    base_key = str(raw.get("key") or fallback_key or raw.get("name") or "use_case")
    name = str(raw.get("name") or base_key.replace("_", " ").title()).strip()[:80] or "Use case"
    in_pre = str(raw.get("in_pre", "Chat"))
    out_pre = str(raw.get("out_pre", "Chat"))
    preset_fallback = _find_preset(_slugify_key(base_key))
    has_scale_metadata = "scale_kind" in raw or any(
        key in raw
        for key in ("scale_model", "scale_label", "scale_unit", "scale_formula", "scale_token_multiplier")
    )
    scale_source = raw if has_scale_metadata or preset_fallback is None else preset_fallback
    scale_kind = _normalize_scale_kind(scale_source)
    raw_tokens_day = _bounded_def_value("tokens_day", _payload_float(raw, "tokens_day", 500e6))
    scale_payload = raw.get("scale") if isinstance(raw.get("scale"), dict) else {}
    scale_value = _payload_optional_float(raw, "scale_value")
    if scale_value is None:
        scale_value = _payload_optional_float(scale_payload, "value")
    if scale_value is None:
        scale_value = tokens_to_scale_value(raw_tokens_day, scale_kind)
    scale_kind["max"] = max(float(scale_kind["max"]), float(scale_value))
    return {
        "key": _slugify_key(base_key),
        "name": name,
        "difficulty": _bounded_def_value("difficulty", _payload_float(raw, "difficulty", 0.3)),
        "tokens_day": _bounded_def_value("tokens_day", scale_value_to_tokens(scale_value, scale_kind)),
        "scale_value": max(0.0, float(scale_value)),
        "scale_kind": scale_kind,
        "wtp_per_m": _bounded_def_value("wtp_per_m", _payload_float(raw, "wtp_per_m", 1.0)),
        "requires": _coerce_requires(raw.get("requires", ())),
        "min_success_rate": _bounded_def_value("min_success_rate", _payload_float(raw, "min_success_rate", 0.85)),
        "quality_floor": _bounded_def_value("quality_floor", _payload_float(raw, "quality_floor", 0.0)),
        "batch_eligible": bool(raw.get("batch_eligible", False)),
        "latent_jobs_day": _bounded_def_value("latent_jobs_day", _payload_float(raw, "latent_jobs_day", 0.0)),
        "unlock_price_per_m": _bounded_def_value("unlock_price_per_m", _payload_float(raw, "unlock_price_per_m", 0.0)),
        "in_pre": in_pre if in_pre in DIST_PRESETS else "Chat",
        "out_pre": out_pre if out_pre in DIST_PRESETS else "Chat",
        "scale_hint": str(raw.get("scale_hint", "")).strip()[:240],
    }


def normalize_use_case_defs(state: PlannerState):
    raw_defs = getattr(state, "use_case_defs", None)
    if not isinstance(raw_defs, list) or not raw_defs:
        raw_defs = copy.deepcopy(PROJECT_PRESETS)

    normalized = []
    seen = set()
    for idx, raw in enumerate(raw_defs):
        if not isinstance(raw, dict):
            continue
        item = _normalize_use_case_def(raw, fallback_key=f"use_case_{idx + 1}")
        base_key = item["key"]
        if base_key in seen:
            i = 2
            while f"{base_key}_{i}" in seen:
                i += 1
            item["key"] = f"{base_key}_{i}"
        seen.add(item["key"])
        normalized.append(item)

    state.use_case_defs = normalized or copy.deepcopy(PROJECT_PRESETS)


def get_use_case_defs(state: PlannerState) -> list[dict[str, Any]]:
    normalize_use_case_defs(state)
    return state.use_case_defs


def _find_use_case_def(state: PlannerState, key: str) -> Optional[dict[str, Any]]:
    return next((d for d in get_use_case_defs(state) if d["key"] == key), None)


def _default_project_name(name: str) -> bool:
    clean = (name or "").strip()
    return clean in {"", "New project", "New use case"} or any(clean == p["name"] for p in PROJECT_PRESETS)


def _apply_preset_definition(proj: Project, preset: dict, preserve_scale: bool = True):
    old_kind = getattr(proj, "kind_key", "custom")
    tokens_day = proj.tokens_day
    scale_value = getattr(proj, "scale_value", None)
    latent_jobs_day = proj.latent_jobs_day
    scale_kind = _normalize_scale_kind(preset)

    proj.kind_key = str(preset["key"])
    proj.name = str(preset["name"])
    proj.difficulty = float(preset["difficulty"])
    proj.wtp_per_m = float(preset["wtp_per_m"])
    proj.scale_kind = copy.deepcopy(scale_kind)
    proj.batch_eligible = bool(preset.get("batch_eligible", False))
    proj.requires = frozenset(preset.get("requires", ()))
    proj.min_success_rate = float(preset.get("min_success_rate", 0.85))
    proj.quality_floor = float(preset.get("quality_floor", 0.0))
    proj.unlock_price_per_m = float(preset.get("unlock_price_per_m", 0.0))
    proj.in_pre = str(preset.get("in_pre", "Chat"))
    proj.out_pre = str(preset.get("out_pre", "Chat"))
    if preserve_scale:
        if old_kind == preset["key"] and scale_value is not None:
            proj.scale_value = max(0.0, float(scale_value))
        else:
            proj.scale_value = tokens_to_scale_value(tokens_day, scale_kind)
        proj.tokens_day = scale_value_to_tokens(proj.scale_value, scale_kind)
        proj.latent_jobs_day = latent_jobs_day
    else:
        proj.scale_value = float(preset.get("scale_value", tokens_to_scale_value(preset.get("tokens_day", tokens_day), scale_kind)))
        proj.tokens_day = scale_value_to_tokens(proj.scale_value, scale_kind)
        proj.latent_jobs_day = float(preset.get("latent_jobs_day", latent_jobs_day))
    proj.__post_init__()


def _add_project_from_preset(state: PlannerState, preset_key: str) -> Optional[Project]:
    preset = _find_use_case_def(state, preset_key)
    if preset is None:
        return None
    scale_kind = _normalize_scale_kind(preset)
    scale_value = float(preset.get("scale_value", tokens_to_scale_value(preset.get("tokens_day", 500e6), scale_kind)))
    proj = Project(
        uid=_next_uid(),
        name=preset["name"],
        difficulty=float(preset["difficulty"]),
        tokens_day=scale_value_to_tokens(scale_value, scale_kind),
        wtp_per_m=float(preset["wtp_per_m"]),
        scale_value=scale_value,
        scale_kind=copy.deepcopy(scale_kind),
        kind_key=str(preset["key"]),
        batch_eligible=bool(preset.get("batch_eligible", False)),
        requires=frozenset(preset.get("requires", ())),
        min_success_rate=float(preset.get("min_success_rate", 0.85)),
        quality_floor=float(preset.get("quality_floor", 0.0)),
        latent_jobs_day=float(preset.get("latent_jobs_day", 0.0)),
        unlock_price_per_m=float(preset.get("unlock_price_per_m", 0.0)),
        in_pre=str(preset.get("in_pre", "Chat")),
        out_pre=str(preset.get("out_pre", "Chat")),
    )
    state.projects.append(proj)
    _sync_aggregate_distribution(state)
    return proj


def add_project(state: PlannerState, preset_key: Optional[str] = None) -> Project:
    if preset_key:
        proj = _add_project_from_preset(state, preset_key)
        if proj is not None:
            return proj
    # Fallback blank project
    scale_kind = copy.deepcopy(DEFAULT_SCALE_KIND)
    proj = Project(
        uid=_next_uid(),
        name="New use case",
        difficulty=0.3,
        tokens_day=500e6,
        wtp_per_m=1.0,
        scale_value=tokens_to_scale_value(500e6, scale_kind),
        scale_kind=scale_kind,
        kind_key="custom",
        batch_eligible=False,
        requires=frozenset(),
        min_success_rate=0.85,
        quality_floor=0.0,
        latent_jobs_day=0.0,
        unlock_price_per_m=0.0,
        in_pre="Chat",
        out_pre="Chat",
    )
    state.projects.append(proj)
    _sync_aggregate_distribution(state)
    return proj


def set_project_kind(state: PlannerState, project_uid: int, kind_key: str):
    proj = state.find_project(project_uid)
    if proj is None:
        return
    if kind_key == "custom":
        proj.kind_key = "custom"
        return
    preset = _find_use_case_def(state, kind_key)
    if preset is None:
        return
    _apply_preset_definition(proj, preset, preserve_scale=True)
    _sync_aggregate_distribution(state)


def remove_project(state: PlannerState, project_uid: int):
    state.projects = [p for p in state.projects if p.uid != project_uid]
    _sync_aggregate_distribution(state)


def set_project_field(state: PlannerState, project_uid: int, field_name: str, value: float):
    proj = state.find_project(project_uid)
    if proj is None:
        return
    bounds = PROJECT_FIELD_BOUNDS.get(field_name)
    if not bounds:
        return
    lo, hi = bounds
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    setattr(proj, field_name, min(max(numeric, lo), hi))
    if field_name == "tokens_day":
        proj.scale_value = tokens_to_scale_value(proj.tokens_day, getattr(proj, "scale_kind", {}))
        _sync_aggregate_distribution(state)


def set_project_scale_value(state: PlannerState, project_uid: int, value: float):
    proj = state.find_project(project_uid)
    if proj is None:
        return
    scale_kind = project_scale_config(state, proj)
    proj.scale_kind = {
        k: scale_kind[k]
        for k in ("model", "label", "unit", "token_multiplier", "min", "max", "step", "formula")
    }
    proj.scale_value = max(0.0, float(value or 0.0))
    proj.tokens_day = _bounded_project_value("tokens_day", scale_value_to_tokens(proj.scale_value, proj.scale_kind))
    _sync_aggregate_distribution(state)


def set_project_dist_preset(state: PlannerState, project_uid: int, kind: str, preset_key: str):
    proj = state.find_project(project_uid)
    if proj is None or preset_key not in DIST_PRESETS:
        return
    if kind == "in":
        proj.in_pre = preset_key
    elif kind == "out":
        proj.out_pre = preset_key
    _sync_aggregate_distribution(state)


def _sync_aggregate_distribution(state: "PlannerState"):
    """Recompute state.in_dist / state.out_dist as a demand-weighted blend of each
    project's in_pre / out_pre preset. Shared capacity views consume this aggregate, while
    routing economics can still use each project's declared shape directly."""
    in_len = len(INPUT_BUCKETS)
    out_len = len(OUTPUT_BUCKETS)
    in_agg = [0.0] * in_len
    out_agg = [0.0] * out_len
    total = 0.0
    for p in state.projects:
        w = max(float(p.tokens_day), 0.0)
        if w <= 0.0:
            continue
        in_preset = DIST_PRESETS.get(p.in_pre) or DIST_PRESETS["Chat"]
        out_preset = DIST_PRESETS.get(p.out_pre) or DIST_PRESETS["Chat"]
        for i in range(in_len):
            in_agg[i] += w * float(in_preset["in"][i])
        for i in range(out_len):
            out_agg[i] += w * float(out_preset["out"][i])
        total += w
    if total <= 0.0:
        return
    s_in = sum(in_agg)
    s_out = sum(out_agg)
    if s_in > 0.0:
        state.in_dist = [max(0, int(round(100 * x / s_in))) for x in in_agg]
    if s_out > 0.0:
        state.out_dist = [max(0, int(round(100 * x / s_out))) for x in out_agg]


def set_project_name(state: PlannerState, project_uid: int, name: str):
    proj = state.find_project(project_uid)
    if proj is None:
        return
    proj.name = (name or "").strip()[:60] or proj.name


def set_project_batch_eligible(state: PlannerState, project_uid: int, value: bool):
    proj = state.find_project(project_uid)
    if proj is None:
        return
    proj.batch_eligible = bool(value)


def set_project_capability(state: PlannerState, project_uid: int, capability: str, required: bool):
    proj = state.find_project(project_uid)
    if proj is None or capability not in ALLOWED_CAPABILITIES:
        return
    if required:
        proj.requires = proj.requires | {capability}
    else:
        proj.requires = proj.requires - {capability}


def _sync_projects_from_use_case_defs(state: PlannerState):
    for proj in state.projects:
        if getattr(proj, "kind_key", "custom") == "custom":
            continue
        preset = _find_use_case_def(state, proj.kind_key)
        if preset is None:
            proj.kind_key = "custom"
            continue
        _apply_preset_definition(proj, preset, preserve_scale=True)
    _sync_aggregate_distribution(state)


def add_use_case_def(state: PlannerState) -> dict[str, Any]:
    key = _unique_use_case_key(state, "new_use_case")
    item = _normalize_use_case_def({
        "key": key,
        "name": "New use case",
        "difficulty": 0.3,
        "tokens_day": 500e6,
        "scale_value": 500,
        "scale_kind": copy.deepcopy(DEFAULT_SCALE_KIND),
        "wtp_per_m": 1.0,
        "requires": (),
        "min_success_rate": 0.85,
        "quality_floor": 0.0,
        "batch_eligible": False,
        "latent_jobs_day": 0.0,
        "unlock_price_per_m": 0.0,
        "in_pre": "Chat",
        "out_pre": "Chat",
        "scale_hint": "Set this from the organization's real usage driver.",
    })
    state.use_case_defs.append(item)
    return item


def remove_use_case_def(state: PlannerState, key: str):
    get_use_case_defs(state)
    state.use_case_defs = [d for d in state.use_case_defs if d["key"] != key]
    for proj in state.projects:
        if getattr(proj, "kind_key", "custom") == key:
            proj.kind_key = "custom"


def _set_use_case_scale_kind_field(item: dict[str, Any], field_name: str, value: Any):
    scale_kind = _normalize_scale_kind(item)
    if field_name == "scale_model":
        model = str(value or "").strip()
        scale_kind["model"] = model if model in SCALE_MODELS else "custom"
    elif field_name == "scale_label":
        scale_kind["label"] = str(value or "").strip()[:48] or scale_kind["label"]
    elif field_name == "scale_unit":
        scale_kind["unit"] = str(value or "").strip()[:48] or scale_kind["unit"]
    elif field_name == "scale_formula":
        scale_kind["formula"] = str(value or "").strip()[:180] or scale_kind["formula"]
    elif field_name == "scale_token_multiplier":
        scale_kind["token_multiplier"] = max(1e-9, float(value or scale_kind["token_multiplier"]))
    elif field_name == "scale_max":
        scale_kind["max"] = max(float(value or scale_kind["max"]), float(scale_kind["min"]) + float(scale_kind["step"]))
    elif field_name == "scale_step":
        scale_kind["step"] = max(1e-9, float(value or scale_kind["step"]))
    else:
        return
    item["scale_kind"] = scale_kind
    item["tokens_day"] = _bounded_def_value(
        "tokens_day",
        scale_value_to_tokens(float(item.get("scale_value", 0.0)), scale_kind),
    )


def set_use_case_def_field(state: PlannerState, key: str, field_name: str, value: Any):
    item = _find_use_case_def(state, key)
    if item is None:
        return

    if field_name == "name":
        item["name"] = (str(value or "").strip()[:80] or item["name"])
    elif field_name == "scale_hint":
        item["scale_hint"] = str(value or "").strip()[:240]
    elif field_name == "scale_value":
        item["scale_value"] = max(0.0, float(value or 0.0))
        item["tokens_day"] = _bounded_def_value("tokens_day", scale_value_to_tokens(item["scale_value"], item.get("scale_kind", {})))
    elif field_name in {"scale_model", "scale_label", "scale_unit", "scale_formula", "scale_token_multiplier", "scale_max", "scale_step"}:
        _set_use_case_scale_kind_field(item, field_name, value)
    elif field_name == "batch_eligible":
        item["batch_eligible"] = bool(value)
    elif field_name in PROJECT_FIELD_BOUNDS:
        item[field_name] = _bounded_def_value(field_name, float(value or 0.0))
        if field_name == "tokens_day":
            item["scale_value"] = tokens_to_scale_value(item["tokens_day"], item.get("scale_kind", {}))
    elif field_name == "in_pre" and value in DIST_PRESETS:
        item["in_pre"] = str(value)
    elif field_name == "out_pre" and value in DIST_PRESETS:
        item["out_pre"] = str(value)
    else:
        return
    _sync_projects_from_use_case_defs(state)


def set_use_case_def_capability(state: PlannerState, key: str, capability: str, required: bool):
    item = _find_use_case_def(state, key)
    if item is None or capability not in ALLOWED_CAPABILITIES:
        return
    caps = set(item.get("requires", ()))
    if required:
        caps.add(capability)
    else:
        caps.discard(capability)
    item["requires"] = tuple(c for c in MODEL_CAPABILITIES if c in caps)
    _sync_projects_from_use_case_defs(state)


def _payload_float(source: dict, key: str, default: float) -> float:
    try:
        value = float(source.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _bounded_project_value(field_name: str, value: float) -> float:
    lo, hi = PROJECT_FIELD_BOUNDS[field_name]
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    return min(max(numeric, lo), hi)


def _infer_project_kind(state: PlannerState, proj: Project) -> str:
    current = getattr(proj, "kind_key", "custom")
    if current == "custom" or _find_use_case_def(state, current) is not None:
        return current
    for preset in get_use_case_defs(state):
        if getattr(proj, "name", "") == preset["name"]:
            return str(preset["key"])
    return "custom"


def normalize_projects(state: PlannerState):
    normalize_use_case_defs(state)
    for proj in state.projects:
        proj.kind_key = _infer_project_kind(state, proj)
        preset = _find_use_case_def(state, proj.kind_key) if proj.kind_key != "custom" else None
        if preset is not None:
            _apply_preset_definition(proj, preset, preserve_scale=True)
        else:
            proj.scale_kind = _normalize_scale_kind({"scale_kind": getattr(proj, "scale_kind", {})})
            if getattr(proj, "scale_value", None) is None:
                proj.scale_value = tokens_to_scale_value(getattr(proj, "tokens_day", 0.0), proj.scale_kind)
            else:
                proj.scale_value = max(0.0, float(proj.scale_value))
                proj.tokens_day = scale_value_to_tokens(proj.scale_value, proj.scale_kind)
        if not isinstance(proj.requires, frozenset):
            proj.requires = frozenset(c for c in (proj.requires or ()) if c in ALLOWED_CAPABILITIES)
        if proj.in_pre not in DIST_PRESETS:
            proj.in_pre = "Chat"
        if proj.out_pre not in DIST_PRESETS:
            proj.out_pre = "Chat"
        proj.difficulty = _bounded_project_value("difficulty", getattr(proj, "difficulty", 0.3))
        proj.tokens_day = _bounded_project_value("tokens_day", getattr(proj, "tokens_day", 0.0))
        proj.scale_value = tokens_to_scale_value(proj.tokens_day, getattr(proj, "scale_kind", {}))
        proj.wtp_per_m = _bounded_project_value("wtp_per_m", getattr(proj, "wtp_per_m", 1.0))
        proj.min_success_rate = _bounded_project_value("min_success_rate", getattr(proj, "min_success_rate", 0.85))
        proj.quality_floor = _bounded_project_value("quality_floor", getattr(proj, "quality_floor", 0.0))
        proj.latent_jobs_day = _bounded_project_value("latent_jobs_day", getattr(proj, "latent_jobs_day", 0.0))
        proj.unlock_price_per_m = _bounded_project_value("unlock_price_per_m", getattr(proj, "unlock_price_per_m", 0.0))
    _sync_aggregate_distribution(state)


def add_gpu(state: PlannerState, gpu_type: str, count: int = 8):
    count = normalize_gpu_count(gpu_type, count)
    existing = next((g for g in state.gpus if g.gpu_type == gpu_type), None)
    if existing:
        existing.count = normalize_gpu_count(gpu_type, existing.count + count)
    else:
        state.gpus.append(GpuPool(_next_uid(), gpu_type, count))


def remove_gpu(state: PlannerState, gpu_uid: int):
    state.models = [m for m in state.models if m.gpu_uid != gpu_uid]
    state.gpus = [g for g in state.gpus if g.uid != gpu_uid]


def change_gpu_qty(state: PlannerState, gpu_uid: int, delta: int):
    from placement import _retune_model

    gp = state.find_gpu(gpu_uid)
    if gp is None:
        return

    new_count = normalize_gpu_count(gp.gpu_type, gp.count + delta, allow_zero=True)
    used = state.used_gpu_for_pool(gpu_uid)
    if new_count < used:
        excess = used - new_count
        for am in reversed(state.models):
            if am.gpu_uid != gpu_uid or excess <= 0:
                continue
            take = min(am.gpu_count, excess)
            am.gpu_count -= take
            excess -= take
            _retune_model(state, am)

    gp.count = new_count
    if new_count == 0:
        state.models = [m for m in state.models if m.gpu_uid != gpu_uid]
        state.gpus = [g for g in state.gpus if g.uid != gpu_uid]


def add_model(state: PlannerState, model_key: str):
    from placement import (
        _best_precision_need,
        _finite_gpu_need,
        _min_gpu_count_for_pool,
        _retune_model,
    )

    if not state.gpus:
        raise ValueError("Add a GPU pool before adding a model.")
    if model_key not in MODELS or MODELS[model_key].hidden:
        raise ValueError("Invalid model key.")
    model = MODELS[model_key]

    def fit_needs(gp: GpuPool) -> tuple[dict[str, float], dict[str, float]]:
        avail = state.free_gpu_for_pool(gp.uid)
        needs_now = {
            prec: _min_gpu_count_for_pool(model, gp.gpu, state.mu, state.profiled_non_kv_gb, prec, avail)
            for prec in PRECISIONS
        }
        needs_full = {
            prec: _min_gpu_count_for_pool(model, gp.gpu, state.mu, state.profiled_non_kv_gb, prec, gp.count)
            for prec in PRECISIONS
        }
        return needs_now, needs_full

    def sort_key(gp: GpuPool) -> tuple[bool, float, bool, float, bool, float, int, int]:
        avail = state.free_gpu_for_pool(gp.uid)
        needs_now, needs_full = fit_needs(gp)
        best_now = _finite_gpu_need(*needs_now.values())
        best_full = _finite_gpu_need(*needs_full.values())
        bf16_now = needs_now["bf16"]
        return (
            math.isinf(bf16_now),
            bf16_now,
            math.isinf(best_now),
            best_now,
            math.isinf(best_full),
            best_full,
            -avail,
            -gp.count,
        )

    gp = min(state.gpus, key=sort_key)
    avail = state.free_gpu_for_pool(gp.uid)
    needs_now, needs_full = fit_needs(gp)
    best_full = _finite_gpu_need(*needs_full.values())
    if math.isinf(best_full):
        labels = ", ".join(PRECISION_LABELS[p] for p in PRECISIONS)
        raise ValueError(f"{model.name} does not fit on any configured GPU pool under the current memory cap in {labels}.")

    bf16_now = needs_now["bf16"]
    if not math.isinf(bf16_now):
        selected_prec = "bf16"
        gpu_count = int(bf16_now)
    else:
        selected_prec, best_now = _best_precision_need(needs_now)
        if selected_prec is not None and not math.isinf(best_now):
            gpu_count = int(best_now)
        else:
            selected_prec, _ = _best_precision_need(needs_full)
            selected_prec = selected_prec or "bf16"
            gpu_count = avail

    am = ModelAssignment(_next_uid(), model_key, gp.uid, gpu_count, 1, 1, selected_prec)
    state.models.append(am)
    _retune_model(state, am)
    state.auto_mode = False
    state.auto_excluded = []


def add_models(state: PlannerState, model_keys: list[str]) -> list[str]:
    existing = {am.model_key for am in state.models}
    added: list[str] = []
    for model_key in model_keys:
        if model_key in existing:
            continue
        add_model(state, model_key)
        existing.add(model_key)
        added.append(model_key)
    return added


def auto_exclude_model(state: PlannerState, model_uid: int):
    am = state.find_model(model_uid)
    if am is None:
        return
    model_key = am.model_key
    state.models = [m for m in state.models if m.uid != model_uid]
    if model_key in state.auto_excluded:
        return
    state.auto_excluded.append(model_key)


def auto_reallow_model(state: PlannerState, model_key: str):
    if model_key not in state.auto_excluded:
        return
    state.auto_excluded = [k for k in state.auto_excluded if k != model_key]


def remove_model(state: PlannerState, model_uid: int):
    state.models = [m for m in state.models if m.uid != model_uid]


def set_model_prec(state: PlannerState, model_uid: int, prec: str):
    from placement import _retune_model

    am = state.find_model(model_uid)
    if am is None:
        return
    am.prec = normalize_precision(prec)
    _retune_model(state, am, preserve_existing=True)


def set_model_gpu_count(state: PlannerState, model_uid: int, count: int):
    from placement import _retune_model

    am = state.find_model(model_uid)
    if am is None:
        return
    gp = state.find_gpu(am.gpu_uid)
    if gp is None:
        am.gpu_count = 0
        return
    others_used = sum(x.gpu_count for x in state.models if x.uid != am.uid and x.gpu_uid == am.gpu_uid)
    max_avail = max(0, gp.count - others_used)
    am.gpu_count = min(count, max_avail)
    _retune_model(state, am)


def set_model_strat(state: PlannerState, model_uid: int, tp: int, pp: int, dp: int, phase: str = "decode"):
    am = state.find_model(model_uid)
    if am is None:
        return
    
    # Validate the strategy before setting
    if am.gpu_count <= 0:
        return
    
    gp = state.find_gpu(am.gpu_uid)
    if gp is None:
        return
    
    model = MODELS[am.model_key]
    valid = valid_strategies(
        model,
        am.gpu_count,
        gp.gpu,
        state.mu,
        state.profiled_non_kv_gb,
        am.prec,
    )
    
    strategy = (tp, pp, dp)
    if strategy not in valid:
        # Optionally could set to default or keep current
        return

    if getattr(model, "embedding_profile", None) is not None:
        am.prefill_tp = tp
        am.prefill_pp = pp
        am.prefill_dp = dp
        am.tp = tp
        am.pp = pp
        am.dp = dp
        return
    
    if phase == "prefill":
        am.prefill_tp = tp
        am.prefill_pp = pp
        am.prefill_dp = dp
    else:
        am.tp = tp
        am.pp = pp
        am.dp = dp


def set_model_gpu_pool(state: PlannerState, model_uid: int, gpu_uid: int):
    from placement import _retune_model

    am = state.find_model(model_uid)
    if am is None:
        return
    am.gpu_uid = gpu_uid
    others_used = sum(x.gpu_count for x in state.models if x.uid != am.uid and x.gpu_uid == gpu_uid)
    gp = state.find_gpu(gpu_uid)
    max_avail = gp.count - others_used if gp else 0
    am.gpu_count = min(am.gpu_count, max_avail)
    _retune_model(state, am)


def set_dist_preset(state: PlannerState, kind: str, preset_key: str):
    if kind == "embedding_doc":
        preset = EMBEDDING_DOC_PRESETS.get(preset_key)
        if not preset:
            return
        state.embedding_doc_dist = list(preset)
        state.embedding_doc_pre = preset_key
        state.task_il = avg_dist(state.embedding_doc_dist, EMBEDDING_DOC_BUCKETS)
        state.task_ol = 0
        return

    preset = DIST_PRESETS.get(preset_key)
    if not preset:
        return
    if kind == "in":
        state.in_dist = list(preset["in"])
        state.in_pre = preset_key
    else:
        state.out_dist = list(preset["out"])
        state.out_pre = preset_key


def set_dist_value(state: PlannerState, kind: str, index: int, value: int):
    value = min(max(int(value), 0), 1_000_000)
    if kind == "embedding_doc":
        if 0 <= index < len(state.embedding_doc_dist):
            state.embedding_doc_dist[index] = value
        state.embedding_doc_pre = ""
        state.task_il = avg_dist(state.embedding_doc_dist, EMBEDDING_DOC_BUCKETS)
        state.task_ol = 0
    elif kind == "in":
        if 0 <= index < len(state.in_dist):
            state.in_dist[index] = value
        state.in_pre = ""
    else:
        if 0 <= index < len(state.out_dist):
            state.out_dist[index] = value
        state.out_pre = ""


def set_prefix_hit_rate(state: PlannerState, value: float):
    state.prefix_hit_rate = min(max(value, 0.0), 1.0)


def set_projection_choice(state: PlannerState, key: str, value: str):
    if key == "projection_day_shape":
        state.projection_day_shape = normalize_day_shape(value)
    elif key == "corpo_cloud":
        state.corpo_cloud = normalize_corpo_cloud(value)


def set_projection_pct(state: PlannerState, key: str, value: float):
    bounds = PROJECTION_PCT_BOUNDS.get(key)
    if not bounds:
        return
    lo, hi = bounds
    setattr(state, key, min(max(value, lo), hi))


def set_projection_toggle(state: PlannerState, key: str, value: bool):
    if key == "projection_night_batching":
        state.projection_night_batching = bool(value)


def set_gpu_cost(state: PlannerState, gpu_uid: int, cost: float):
    gp = state.find_gpu(gpu_uid)
    if gp is None:
        return
    gp.cost_per_gpu_hour = max(0.0, cost)


_states: dict[str, PlannerState] = {}
_compare_states: dict[str, PlannerState] = {}
_state_last_seen: dict[str, float] = {}
_state_guard = threading.RLock()
_scope_locks: weakref.WeakValueDictionary[str, threading.RLock] = weakref.WeakValueDictionary()
_STATE_TTL_SECONDS = _env_nonnegative_int("PLANNER_STATE_TTL_SECONDS", 86400)
_STATE_MAX_SCOPES = max(1, _env_nonnegative_int("PLANNER_STATE_MAX_SCOPES", 5000))


def get_scope_lock(session_id: str) -> threading.RLock:
    """Return the process-local lock serializing one browser-tab scope."""
    with _state_guard:
        lock = _scope_locks.get(session_id)
        if lock is None:
            lock = threading.RLock()
            _scope_locks[session_id] = lock
        return lock


def allow_visitor_scope(session_id: str, visitor_id: str, max_scopes: int) -> bool:
    """Return whether a visitor may reuse/create this tab scope under its cap."""
    prefix = f"{visitor_id}:"
    with _state_guard:
        _prune_states_locked(time.monotonic(), preserve=session_id)
        if session_id in _states or session_id in _compare_states:
            return True
        existing = sum(1 for key in _states if key.startswith(prefix))
        return existing < max(1, int(max_scopes))


def _prune_states_locked(now: float, preserve: Optional[str] = None) -> None:
    if _STATE_TTL_SECONDS > 0:
        stale = [
            key for key, touched in _state_last_seen.items()
            if key != preserve and now - touched > _STATE_TTL_SECONDS
        ]
        for key in stale:
            _states.pop(key, None)
            _compare_states.pop(key, None)
            _state_last_seen.pop(key, None)

    excess = len(_states) - _STATE_MAX_SCOPES
    if excess > 0:
        oldest = sorted(
            (touched, key) for key, touched in _state_last_seen.items() if key != preserve
        )[:excess]
        for _, key in oldest:
            _states.pop(key, None)
            _compare_states.pop(key, None)
            _state_last_seen.pop(key, None)


def reset_state(session_id: str, *, blank: bool = False) -> PlannerState:
    with _state_guard:
        state = PlannerState() if blank else create_default_state()
        _states[session_id] = state
        _compare_states.pop(session_id, None)
        _state_last_seen[session_id] = time.monotonic()
        _prune_states_locked(_state_last_seen[session_id], preserve=session_id)
        return state


def delete_visitor_states(visitor_id: str) -> int:
    prefix = f"{visitor_id}:"
    with _state_guard:
        keys = {key for key in (*_states.keys(), *_compare_states.keys()) if key.startswith(prefix)}
        for key in keys:
            _states.pop(key, None)
            _compare_states.pop(key, None)
            _state_last_seen.pop(key, None)
        return len(keys)


def _normalize_loaded_state(s: PlannerState) -> PlannerState:
    for am in s.models:
        am.prec = normalize_precision(getattr(am, "prec", "bf16"))
    s.mode = normalize_plot_mode(s.mode)
    s.projection_day_shape = normalize_day_shape(s.projection_day_shape)
    s.corpo_cloud = normalize_corpo_cloud(getattr(s, "corpo_cloud", CORPO_CLOUD_DEFAULT))
    if not hasattr(s, "auto_excluded"):
        s.auto_excluded = []
    if not hasattr(s, "auto_mode"):
        s.auto_mode = False
    s.auto_strategy = normalize_auto_strategy(getattr(s, "auto_strategy", DEFAULT_AUTO_MODEL_STRATEGY))
    normalize_embedding_doc_distribution(s)
    normalize_projects(s)
    return s


def get_state(session_id: str) -> PlannerState:
    now = time.monotonic()
    with _state_guard:
        _prune_states_locked(now, preserve=session_id)
        if session_id not in _states:
            _states[session_id] = create_default_state()
        _state_last_seen[session_id] = now
        _prune_states_locked(now, preserve=session_id)
        s = _states[session_id]
    return _normalize_loaded_state(s)


def get_compare_state(session_id: str) -> Optional[PlannerState]:
    with _state_guard:
        state = _compare_states.get(session_id)
        if state is not None:
            _state_last_seen[session_id] = time.monotonic()
    if state is not None:
        _normalize_loaded_state(state)
    return state


def duplicate_compare_state(session_id: str) -> PlannerState:
    # Clone the current primary configuration so panel B starts from panel A.
    with _state_guard:
        _compare_states[session_id] = copy.deepcopy(get_state(session_id))
        _state_last_seen[session_id] = time.monotonic()
        return _compare_states[session_id]


def clear_compare_state(session_id: str) -> bool:
    with _state_guard:
        return _compare_states.pop(session_id, None) is not None


def replace_scope_states(session_id: str, state_a: PlannerState, state_b: Optional[PlannerState]) -> None:
    with _state_guard:
        _states[session_id] = state_a
        if state_b is None:
            _compare_states.pop(session_id, None)
        else:
            _compare_states[session_id] = state_b
        _state_last_seen[session_id] = time.monotonic()
        _prune_states_locked(_state_last_seen[session_id], preserve=session_id)
