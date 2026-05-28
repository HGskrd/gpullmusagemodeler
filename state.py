"""Server-side state management for the vLLM planner."""

from __future__ import annotations

import copy
import math
import re
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
    CORPO_CLOUD_PRESETS,
    CORPO_CLOUD_DEFAULT,
    MODEL_CAPABILITIES,
    SCALE_MODELS,
    PRECISIONS,
    PRECISION_LABELS,
    normalize_gpu_count,
    normalize_precision,
    effective_quality,
    model_success_rate,
    required_quality,
)
from calc import (
    EfficiencyParams,
    avg_dist,
    communication_breakdown,
    compute_decode,
    compute_embedding_distribution,
    compute_memory,
    compute_prefill,
    compute_realtime_capacity,
    compute_realtime_max_users,
    default_strategy,
    embedding_doc_stats,
    embedding_sequence_length,
    effective_prefill_length,
    gpu_supports_mxfp4,
    gpu_supports_nvfp4,
    per_replica_kv_cache_bytes,
    strategy_label,
    valid_strategies,
)


_uid_counter = 0
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
ALLOWED_CORPO_CLOUDS = frozenset(CORPO_CLOUD_PRESETS)
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
    # Average share of peak capacity that internal users actually book. Max day-shape hour
    # can run this above 100% → thrash/stall zone (see calc._stall_curve).
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
    _uid_counter += 1
    return _uid_counter


def normalize_plot_mode(mode: Optional[str]) -> str:
    if mode in LEGACY_PLOT_MODE_REDIRECTS:
        return LEGACY_PLOT_MODE_REDIRECTS[mode]
    return mode if mode in ALLOWED_PLOT_MODES else DEFAULT_PLOT_MODE


def normalize_day_shape(shape: Optional[str]) -> str:
    return shape if shape in ALLOWED_DAY_SHAPES else DEFAULT_DAY_SHAPE


def normalize_corpo_cloud(name: Optional[str]) -> str:
    return name if name in ALLOWED_CORPO_CLOUDS else CORPO_CLOUD_DEFAULT


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


def _min_gpu_count_for_pool(
    m: Model,
    g: GPU,
    mu: float,
    profiled_non_kv_gb: float,
    prec: str,
    max_gpu_count: int,
) -> float:
    for gpu_count in range(1, max_gpu_count + 1):
        if valid_strategies(m, gpu_count, g, mu, profiled_non_kv_gb, prec):
            return gpu_count
    return math.inf


def _finite_gpu_need(*needs: float) -> float:
    finite = [need for need in needs if not math.isinf(need)]
    return min(finite) if finite else math.inf


def _best_precision_need(needs: dict[str, float]) -> tuple[Optional[str], float]:
    finite = [(prec, need) for prec, need in needs.items() if not math.isinf(need)]
    if not finite:
        return None, math.inf
    return min(finite, key=lambda item: (item[1], PRECISIONS.index(item[0])))


def _gpu_count_options(max_avail: int, current_count: int, gpu: Optional[GPU]) -> list[int]:
    max_count = max(0, int(max_avail or 0))
    current = min(max(0, int(current_count or 0)), max_count)
    options = {0, current, max_count}

    options.update(range(1, min(max_count, 8) + 1))
    options.update(range(10, min(max_count, 16) + 1, 2))

    for count in (24, 32, 48, 64, 96, 128, 192, 256):
        if count <= max_count:
            options.add(count)

    if gpu is not None:
        node_size = max(int(getattr(gpu, "node_size", 1) or 1), 1)
        for count in range(node_size, max_count + 1, node_size):
            options.add(count)

    return sorted(count for count in options if 0 <= count <= max_count)


def _probe_batch_sizes(dp: int) -> list[int]:
    values = {max(1, dp)}
    while max(values) < 128:
        values.add(max(values) * 2)
    return sorted(values)


def _preferred_strategy(state: PlannerState, am: ModelAssignment, gpu: GPU, phase: str) -> tuple[int, int, int]:
    model = MODELS[am.model_key]
    candidates = valid_strategies(model, am.gpu_count, gpu, state.mu, state.profiled_non_kv_gb, am.prec)
    if not candidates:
        return default_strategy(model, am.gpu_count, gpu, state.mu, state.profiled_non_kv_gb, am.prec)

    best = candidates[0]
    best_score = None
    probe_prefill_len = max(1, effective_prefill_length(max(state.task_il, avg_dist(state.in_dist, INPUT_BUCKETS)), state.prefix_hit_rate))
    is_embedding = getattr(model, "embedding_profile", None) is not None
    for tp, pp, dp in candidates:
        mem = compute_memory(
            model,
            tp,
            pp,
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.prefill_efficiency if phase == "prefill" else state.decode_efficiency,
        )
        kv_headroom = mem.kv_budget if mem else 0.0
        local_tp = 1 if tp <= gpu.node_size else 0
        peak_tps = -1
        aux = float("-inf")

        for bs in _probe_batch_sizes(dp):
            if is_embedding:
                result = compute_embedding_distribution(
                    model,
                    (tp, pp, dp),
                    bs,
                    state.embedding_doc_dist,
                    EMBEDDING_DOC_BUCKETS,
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.prefill_efficiency,
                )
                if result is None:
                    continue
                metric = (result.tps, result.rps)
            elif phase == "prefill":
                result = compute_prefill(
                    model,
                    tp,
                    pp,
                    bs,
                    dp,
                    probe_prefill_len,
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.prefill_efficiency,
                )
                if result is None:
                    continue
                metric = (result.tps, result.rps)
            else:
                result = compute_decode(
                    model,
                    tp,
                    pp,
                    bs,
                    dp,
                    gpu,
                    state.mu,
                    state.profiled_non_kv_gb,
                    am.prec,
                    state.in_dist,
                    state.out_dist,
                    state.decode_efficiency,
                )
                if result is None:
                    continue
                metric = (result.tps, -result.lat)

            if metric[0] > peak_tps or (metric[0] == peak_tps and metric[1] > aux):
                peak_tps = metric[0]
                aux = metric[1]

        if peak_tps < 0:
            score = (local_tp, min(tp, gpu.node_size), dp, -pp, kv_headroom)
        else:
            score = (peak_tps, aux, local_tp, min(tp, gpu.node_size), dp, -pp, kv_headroom)

        if best_score is None or score > best_score:
            best = (tp, pp, dp)
            best_score = score

    return best


def _retune_model(state: PlannerState, am: ModelAssignment, preserve_existing: bool = False):
    if am.gpu_count <= 0:
        am.tp = 1
        am.pp = 1
        am.dp = 1
        am.prefill_tp = 1
        am.prefill_pp = 1
        am.prefill_dp = 1
        return

    gp = state.find_gpu(am.gpu_uid)
    if gp is None:
        return

    model = MODELS[am.model_key]
    if getattr(model, "embedding_profile", None) is not None:
        embedding_default = _preferred_strategy(state, am, gp.gpu, "prefill")
        if not preserve_existing:
            am.tp, am.pp, am.dp = embedding_default
            am.prefill_tp, am.prefill_pp, am.prefill_dp = embedding_default
            return

        valid = valid_strategies(
            model,
            am.gpu_count,
            gp.gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
        )
        if (am.prefill_tp, am.prefill_pp, am.prefill_dp) not in valid:
            am.prefill_tp, am.prefill_pp, am.prefill_dp = embedding_default
        am.tp, am.pp, am.dp = am.prefill_tp, am.prefill_pp, am.prefill_dp
        return

    decode_default = _preferred_strategy(state, am, gp.gpu, "decode")
    prefill_default = _preferred_strategy(state, am, gp.gpu, "prefill")
    if not preserve_existing:
        am.tp, am.pp, am.dp = decode_default
        am.prefill_tp, am.prefill_pp, am.prefill_dp = prefill_default
        return

    decode_valid = valid_strategies(
        model,
        am.gpu_count,
        gp.gpu,
        state.mu,
        state.profiled_non_kv_gb,
        am.prec,
    )
    if (am.tp, am.pp, am.dp) not in decode_valid:
        am.tp, am.pp, am.dp = decode_default

    prefill_valid = valid_strategies(
        model,
        am.gpu_count,
        gp.gpu,
        state.mu,
        state.profiled_non_kv_gb,
        am.prec,
    )
    if (am.prefill_tp, am.prefill_pp, am.prefill_dp) not in prefill_valid:
        am.prefill_tp, am.prefill_pp, am.prefill_dp = prefill_default


def retune_models(state: PlannerState, preserve_existing: bool = True):
    for am in state.models:
        if am.gpu_count > 0:
            _retune_model(state, am, preserve_existing=preserve_existing)


def _assignment_memories(state: PlannerState, am: ModelAssignment, gpu: GPU):
    model = MODELS[am.model_key]
    prefill_mem = compute_memory(
        model,
        am.prefill_tp,
        am.prefill_pp,
        gpu,
        state.mu,
        state.profiled_non_kv_gb,
        am.prec,
        state.prefill_efficiency,
    )
    decode_mem = compute_memory(
        model,
        am.tp,
        am.pp,
        gpu,
        state.mu,
        state.profiled_non_kv_gb,
        am.prec,
        state.decode_efficiency,
    )
    return prefill_mem, decode_mem


def get_deployed(state: PlannerState, phase: str = "decode") -> list[ModelAssignmentProxy]:
    deployed = []
    for am in state.models:
        if am.gpu_count <= 0:
            continue
        gp = state.find_gpu(am.gpu_uid)
        if gp is None:
            continue
        prefill_mem, decode_mem = _assignment_memories(state, am, gp.gpu)
        mem = prefill_mem if phase == "prefill" else decode_mem
        if mem is None:
            continue
        deployed.append(ModelAssignmentProxy(am, gp.gpu, phase, prefill_mem, decode_mem))
    return deployed


def create_default_state() -> PlannerState:
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
    setattr(proj, field_name, min(max(float(value), lo), hi))
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


def serialize_use_case_defs(state: PlannerState) -> dict:
    return {
        "type": "gpullm-use-case-library",
        "version": 1,
        "use_cases": copy.deepcopy(get_use_case_defs(state)),
    }


def replace_use_case_defs(state: PlannerState, payload: Any) -> int:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("use_cases")
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("Use-case JSON must contain a use_cases array.")

    normalized = []
    seen = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("Each use case must be a JSON object.")
        raw = dict(item.get("definition", item)) if isinstance(item.get("definition"), dict) else dict(item)
        # Accept the earlier selected-instance export shape by folding scale into seed values.
        scale = item.get("scale") if isinstance(item.get("scale"), dict) else {}
        if "tokens_day" not in raw and "tokens_day" in scale:
            raw["tokens_day"] = scale["tokens_day"]
        if "scale_value" not in raw and "value" in scale:
            raw["scale_value"] = scale["value"]
        if "latent_jobs_day" not in raw and "latent_jobs_day" in scale:
            raw["latent_jobs_day"] = scale["latent_jobs_day"]
        raw["key"] = item.get("kind_key") or item.get("key") or raw.get("key") or item.get("name") or f"use_case_{idx + 1}"
        raw["name"] = item.get("name") or raw.get("name") or raw["key"]
        normalized_item = _normalize_use_case_def(raw, fallback_key=f"use_case_{idx + 1}")
        base_key = normalized_item["key"]
        if base_key in seen:
            i = 2
            while f"{base_key}_{i}" in seen:
                i += 1
            normalized_item["key"] = f"{base_key}_{i}"
        seen.add(normalized_item["key"])
        normalized.append(normalized_item)

    state.use_case_defs = normalized
    _sync_projects_from_use_case_defs(state)
    return len(normalized)


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


def _project_definition_payload(proj: Project) -> dict:
    return {
        "difficulty": float(proj.difficulty),
        "scale_kind": copy.deepcopy(getattr(proj, "scale_kind", DEFAULT_SCALE_KIND)),
        "wtp_per_m": float(proj.wtp_per_m),
        "requires": sorted(proj.requires),
        "min_success_rate": float(proj.min_success_rate),
        "quality_floor": float(getattr(proj, "quality_floor", 0.0)),
        "batch_eligible": bool(proj.batch_eligible),
        "unlock_price_per_m": float(proj.unlock_price_per_m),
        "in_pre": proj.in_pre,
        "out_pre": proj.out_pre,
    }


def _project_scale_payload(proj: Project) -> dict:
    return {
        "value": float(getattr(proj, "scale_value", tokens_to_scale_value(proj.tokens_day, getattr(proj, "scale_kind", {})))),
        "tokens_day": float(proj.tokens_day),
        "latent_jobs_day": float(proj.latent_jobs_day),
    }


def serialize_project_set(state: PlannerState) -> dict:
    """JSON-save format for the demand side only.

    Each row keeps definition and scale separate so a saved file can contain many
    use-case kinds while each organization chooses its own deployment scale.
    """
    return {
        "type": "gpullm-use-case-set",
        "version": 1,
        "use_cases": [
            {
                "name": proj.name,
                "kind_key": getattr(proj, "kind_key", "custom"),
                "scale": _project_scale_payload(proj),
                "definition": _project_definition_payload(proj),
            }
            for proj in state.projects
        ],
    }


def _payload_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _payload_float(source: dict, key: str, default: float) -> float:
    try:
        return float(source.get(key, default))
    except (TypeError, ValueError):
        return default


def _bounded_project_value(field_name: str, value: float) -> float:
    lo, hi = PROJECT_FIELD_BOUNDS[field_name]
    return min(max(float(value), lo), hi)


def _project_from_payload(state: PlannerState, item: dict) -> Project:
    kind_key = str(item.get("kind_key") or item.get("kind") or "custom")
    preset = _find_use_case_def(state, kind_key) if kind_key != "custom" else None
    if preset is None and kind_key != "custom":
        preset = _find_preset(kind_key)
    base = preset or {
        "key": "custom",
        "name": "Custom use case",
        "difficulty": 0.3,
        "tokens_day": 500e6,
        "scale_value": 500.0,
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
    }
    definition = _payload_dict(item.get("definition")) or item
    scale = _payload_dict(item.get("scale")) or item
    requires_raw = definition.get("requires", base.get("requires", ()))
    if isinstance(requires_raw, str):
        requires_iter = (requires_raw,)
    else:
        requires_iter = requires_raw or ()
    requires = frozenset(c for c in requires_iter if c in ALLOWED_CAPABILITIES)
    scale_kind_source = definition if isinstance(definition.get("scale_kind"), dict) else base
    scale_kind = _normalize_scale_kind(scale_kind_source)
    scale_value = _payload_optional_float(scale, "value")
    if scale_value is None:
        scale_value = _payload_optional_float(definition, "scale_value")
    if scale_value is None:
        scale_value = tokens_to_scale_value(
            _payload_float(scale, "tokens_day", float(base.get("tokens_day", 500e6))),
            scale_kind,
        )

    proj = Project(
        uid=_next_uid(),
        name=str(item.get("name") or base["name"])[:60],
        difficulty=_bounded_project_value(
            "difficulty",
            _payload_float(definition, "difficulty", float(base["difficulty"])),
        ),
        tokens_day=_bounded_project_value(
            "tokens_day",
            scale_value_to_tokens(scale_value, scale_kind),
        ),
        wtp_per_m=_bounded_project_value(
            "wtp_per_m",
            _payload_float(definition, "wtp_per_m", float(base["wtp_per_m"])),
        ),
        scale_value=max(0.0, float(scale_value)),
        scale_kind=copy.deepcopy(scale_kind),
        kind_key=str(base["key"]) if preset else "custom",
        batch_eligible=bool(definition.get("batch_eligible", base.get("batch_eligible", False))),
        requires=requires,
        min_success_rate=_bounded_project_value(
            "min_success_rate",
            _payload_float(definition, "min_success_rate", float(base.get("min_success_rate", 0.85))),
        ),
        quality_floor=_bounded_project_value(
            "quality_floor",
            _payload_float(definition, "quality_floor", float(base.get("quality_floor", 0.0))),
        ),
        latent_jobs_day=_bounded_project_value(
            "latent_jobs_day",
            _payload_float(scale, "latent_jobs_day", float(base.get("latent_jobs_day", 0.0))),
        ),
        unlock_price_per_m=_bounded_project_value(
            "unlock_price_per_m",
            _payload_float(definition, "unlock_price_per_m", float(base.get("unlock_price_per_m", 0.0))),
        ),
        in_pre=str(definition.get("in_pre", base.get("in_pre", "Chat"))),
        out_pre=str(definition.get("out_pre", base.get("out_pre", "Chat"))),
    )
    return proj


def replace_project_set(state: PlannerState, payload: Any) -> int:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("use_cases")
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("Use-case JSON must contain a use_cases array.")

    projects = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each use case must be a JSON object.")
        projects.append(_project_from_payload(state, item))

    state.projects = projects
    _sync_aggregate_distribution(state)
    return len(projects)


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


def _model_serves_project(model: Model, project: Project) -> bool:
    if getattr(model, "is_realtime_only", False) or getattr(model, "embedding_profile", None) is not None:
        return False
    return (
        project.requires <= model.capabilities
        and effective_quality(model) + 1e-9 >= float(getattr(project, "quality_floor", 0.0))
        and model_success_rate(model, project.difficulty) >= project.min_success_rate
    )


def _active_project_demand(project: Project) -> float:
    return max(0.0, float(project.tokens_day or 0.0)) + 0.25 * max(0.0, float(project.latent_jobs_day or 0.0))


def _best_available_placement(state: PlannerState, model: Model) -> Optional[tuple[GpuPool, int, str]]:
    placements: list[tuple[int, int, float, int, GpuPool, str]] = []
    for gp in state.gpus:
        avail = state.free_gpu_for_pool(gp.uid)
        if avail <= 0:
            continue
        for prec in PRECISIONS:
            need = _min_gpu_count_for_pool(model, gp.gpu, state.mu, state.profiled_non_kv_gb, prec, avail)
            if math.isinf(need):
                continue
            placements.append((int(need), PRECISIONS.index(prec), -gp.gpu.mem, -avail, gp, prec))
    if not placements:
        return None
    need, _, _, _, gp, prec = min(placements)
    return gp, need, prec


def _best_available_placement_on_pool(
    state: PlannerState,
    model: Model,
    gp: GpuPool,
) -> Optional[tuple[int, str]]:
    avail = state.free_gpu_for_pool(gp.uid)
    if avail <= 0:
        return None

    placements: list[tuple[int, int, str]] = []
    for prec in PRECISIONS:
        need = _min_gpu_count_for_pool(model, gp.gpu, state.mu, state.profiled_non_kv_gb, prec, avail)
        if not math.isinf(need):
            placements.append((int(need), PRECISIONS.index(prec), prec))
    if not placements:
        return None

    need, _, prec = min(placements)
    return need, prec


def _auto_assignment_demand(state: PlannerState, am: ModelAssignment) -> float:
    model = MODELS[am.model_key]
    demand = sum(_active_project_demand(project) for project in state.projects if _model_serves_project(model, project))
    return demand or model.quality * 1e6


def _auto_model_value(model: Model, projects: list[Project]) -> float:
    value = 0.0
    for project in projects:
        if not _model_serves_project(model, project):
            continue
        sr = model_success_rate(model, project.difficulty)
        value += _active_project_demand(project) * max(0.0, float(project.wtp_per_m or 0.0)) * sr
    return value


def _auto_model_value_density(model: Model, projects: list[Project], gpu_count: int) -> float:
    return _auto_model_value(model, projects) / max(int(gpu_count or 0), 1)


def _auto_served_projects(model: Model, projects: list[Project]) -> list[Project]:
    return [project for project in projects if _model_serves_project(model, project)]


def _auto_weighted_success(model: Model, projects: list[Project]) -> float:
    served = _auto_served_projects(model, projects)
    total = sum(_active_project_demand(project) for project in served)
    if total <= 0:
        return 0.0
    return sum(
        _active_project_demand(project) * model_success_rate(model, project.difficulty)
        for project in served
    ) / total


def _auto_quality_margin(model: Model, projects: list[Project]) -> float:
    served = _auto_served_projects(model, projects)
    if not served:
        return 0.0
    return min(
        model_success_rate(model, project.difficulty) - float(project.min_success_rate)
        for project in served
    )


def _auto_covered_demand(model: Model, projects: list[Project]) -> float:
    return sum(_active_project_demand(project) for project in _auto_served_projects(model, projects))


def _auto_required_capability_count(projects: list[Project]) -> int:
    required: set[str] = set()
    for project in projects:
        required.update(getattr(project, "requires", frozenset()) or frozenset())
    return len(required)


def _auto_model_work_size(model: Model) -> float:
    return model.active_params / max(float(model.token_efficiency), 1e-6)


def _auto_model_kv_size(model: Model) -> float:
    return max(model.kv_layer_count, 1) * max(model.kv_heads, 1) * max(model.head_dim, 1)


def _auto_candidate_key(
    model: Model,
    projects: list[Project],
    gpu_count: int,
    prec: str,
    strategy: str,
) -> tuple:
    strategy = normalize_auto_strategy(strategy)
    value = _auto_model_value(model, projects)
    value_density = value / max(int(gpu_count or 0), 1)
    served = _auto_served_projects(model, projects)
    served_count = len(served)
    covered_demand = _auto_covered_demand(model, projects)
    weighted_success = _auto_weighted_success(model, served)
    quality_margin = _auto_quality_margin(model, served)
    quality = effective_quality(model)
    work_size = _auto_model_work_size(model)
    kv_size = _auto_model_kv_size(model)
    prec_idx = PRECISIONS.index(prec) if prec in PRECISIONS else len(PRECISIONS)

    if strategy == "coverage":
        return (
            -served_count,
            -covered_demand,
            -_auto_required_capability_count(served),
            -value_density,
            -weighted_success,
            gpu_count,
            work_size,
            model.total_params,
            prec_idx,
            model.key,
        )
    if strategy == "quality":
        return (
            -quality,
            -weighted_success,
            -quality_margin,
            -value_density,
            gpu_count,
            work_size,
            model.total_params,
            prec_idx,
            model.key,
        )
    if strategy == "lean":
        return (
            gpu_count,
            work_size,
            model.total_params,
            -quality_margin,
            -weighted_success,
            -value_density,
            prec_idx,
            model.key,
        )
    if strategy == "throughput":
        return (
            work_size,
            kv_size,
            gpu_count,
            model.total_params,
            -value_density,
            -weighted_success,
            -quality,
            prec_idx,
            model.key,
        )

    return (
        -value_density,
        -value,
        -quality,
        work_size,
        gpu_count,
        model.total_params,
        prec_idx,
        model.key,
    )


def _seed_empty_auto_pools(state: PlannerState, projects: list[Project], strategy: str):
    for gp in state.gpus:
        if state.free_gpu_for_pool(gp.uid) <= 0:
            continue
        if any(am.gpu_uid == gp.uid for am in state.models):
            continue

        candidates = []
        for model in MODELS.values():
            if model.hidden or model.key in state.auto_excluded:
                continue
            value = _auto_model_value(model, projects)
            if value <= 0:
                continue
            placement = _best_available_placement_on_pool(state, model, gp)
            if placement is None:
                continue
            gpu_count, prec = placement
            candidates.append((
                _auto_candidate_key(model, projects, gpu_count, prec, strategy),
                model,
                prec,
            ))

        if not candidates:
            continue

        _, model, prec = min(candidates)
        gpu_count, _ = _best_available_placement_on_pool(state, model, gp) or (0, prec)
        if gpu_count <= 0:
            continue
        state.models.append(ModelAssignment(_next_uid(), model.key, gp.uid, gpu_count, 1, 1, prec))
        _retune_model(state, state.models[-1])


def _auto_assignment_growth_key(state: PlannerState, am: ModelAssignment, strategy: str) -> tuple:
    model = MODELS[am.model_key]
    demand = _auto_assignment_demand(state, am)
    served_projects = [project for project in state.projects if _model_serves_project(model, project)]
    if strategy == "coverage":
        return (-len(served_projects), -demand, am.gpu_count, am.uid)
    if strategy == "quality":
        return (-effective_quality(model), -demand, am.gpu_count, am.uid)
    if strategy == "throughput":
        demand_per_work = demand / max(_auto_model_work_size(model), 1.0)
        return (-demand_per_work, -demand, am.gpu_count, am.uid)
    return (-demand, am.gpu_count, am.uid)


def _grow_auto_assignments(state: PlannerState, strategy: str):
    for gp in state.gpus:
        while state.free_gpu_for_pool(gp.uid) > 0:
            candidates = [am for am in state.models if am.gpu_uid == gp.uid]
            candidates.sort(key=lambda am: _auto_assignment_growth_key(state, am, strategy))
            grew = False
            for am in candidates:
                next_count = am.gpu_count + 1
                if not valid_strategies(MODELS[am.model_key], next_count, gp.gpu, state.mu, state.profiled_non_kv_gb, am.prec):
                    continue
                am.gpu_count = next_count
                _retune_model(state, am)
                grew = True
                break
            if not grew:
                break


def auto_select_models(state: PlannerState, strategy: Optional[str] = None):
    if not state.gpus:
        raise ValueError("Add a GPU pool before auto-selecting models.")

    strategy = normalize_auto_strategy(strategy or getattr(state, "auto_strategy", DEFAULT_AUTO_MODEL_STRATEGY))
    state.auto_strategy = strategy
    original_models = list(state.models)
    state.models = []
    projects = [project for project in state.projects if _active_project_demand(project) > 0]
    if not projects:
        projects = [
            Project(_next_uid(), "Balanced chat", 0.30, 1.0, 1.0, min_success_rate=0.90, quality_floor=0.55),
            Project(_next_uid(), "Coding / reasoning", 0.55, 1.0, 4.0, requires=frozenset({"tools", "ctx_128k"}), min_success_rate=0.85, quality_floor=0.70),
            Project(_next_uid(), "Frontier reasoning", 0.90, 1.0, 20.0, requires=frozenset({"tools", "reasoning"}), min_success_rate=0.95, quality_floor=0.90),
        ]

    selected_keys: set[str] = set()
    ordered_projects = sorted(
        projects,
        key=lambda project: (
            -required_quality(project.difficulty, project.min_success_rate, quality_floor=getattr(project, "quality_floor", 0.0)),
            -_active_project_demand(project),
            -len(project.requires),
        ),
    )

    for project in ordered_projects:
        if strategy in {"coverage", "lean"} and any(
            _model_serves_project(MODELS[am.model_key], project) for am in state.models
        ):
            continue
        candidates = []
        for model in MODELS.values():
            if model.hidden or model.key in selected_keys or model.key in state.auto_excluded or not _model_serves_project(model, project):
                continue
            placement = _best_available_placement(state, model)
            if placement is None:
                continue
            _, gpu_count, prec = placement
            candidates.append((
                _auto_candidate_key(model, projects if strategy == "coverage" else [project], gpu_count, prec, strategy),
                model,
                placement,
            ))
        if not candidates:
            continue

        _, model, placement = min(candidates)
        gp, gpu_count, prec = placement
        state.models.append(ModelAssignment(_next_uid(), model.key, gp.uid, gpu_count, 1, 1, prec))
        selected_keys.add(model.key)
        _retune_model(state, state.models[-1])

    if not state.models:
        state.models = original_models
        raise ValueError("No eligible model fits the configured GPU pools and use-case SLOs.")

    if strategy != "lean":
        _seed_empty_auto_pools(state, projects, strategy)
        _grow_auto_assignments(state, strategy)
    state.auto_mode = True


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
    am = state.find_model(model_uid)
    if am is None:
        return
    am.prec = normalize_precision(prec)
    _retune_model(state, am, preserve_existing=True)


def set_model_gpu_count(state: PlannerState, model_uid: int, count: int):
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


def get_state(session_id: str) -> PlannerState:
    if session_id not in _states:
        _states[session_id] = create_default_state()
    s = _states[session_id]
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


def get_compare_state(session_id: str) -> Optional[PlannerState]:
    state = _compare_states.get(session_id)
    if state is not None:
        for am in state.models:
            am.prec = normalize_precision(getattr(am, "prec", "bf16"))
        state.mode = normalize_plot_mode(state.mode)
        state.projection_day_shape = normalize_day_shape(state.projection_day_shape)
        state.corpo_cloud = normalize_corpo_cloud(getattr(state, "corpo_cloud", CORPO_CLOUD_DEFAULT))
        if not hasattr(state, "auto_excluded"):
            state.auto_excluded = []
        if not hasattr(state, "auto_mode"):
            state.auto_mode = False
        state.auto_strategy = normalize_auto_strategy(getattr(state, "auto_strategy", DEFAULT_AUTO_MODEL_STRATEGY))
        normalize_embedding_doc_distribution(state)
        normalize_projects(state)
    return state


def duplicate_compare_state(session_id: str) -> PlannerState:
    # Clone the current primary configuration so panel B starts from panel A.
    _compare_states[session_id] = copy.deepcopy(get_state(session_id))
    return _compare_states[session_id]


def clear_compare_state(session_id: str) -> bool:
    return _compare_states.pop(session_id, None) is not None


def _comm_summary(tp: int, pp: int) -> str:
    terms = []
    if tp > 1:
        terms.append("dense TP reductions")
    if pp > 1:
        terms.append("PP stage boundaries")
    return "Comm model: " + " + ".join(terms) if terms else ""


def _comm_alerts(model: Model, tp: int, pp: int, dp: int, gpu: Optional[GPU], avg_seq: float, eff: EfficiencyParams) -> list[str]:
    if gpu is None:
        return []

    batch_tokens = max(1, min(32, math.ceil(32 / max(dp, 1))))
    comm = communication_breakdown(model, tp, pp, batch_tokens, avg_seq, gpu, eff)
    alerts: list[str] = []
    if comm.tp_cross_node:
        alerts.append(
            f"{strategy_label(tp, pp, dp)} uses cross-node TP (node size {gpu.node_size}). Prefer TP within a node and scale with PP/DP."
        )
    if comm.ep_advisory:
        alerts.append("MoE multi-node expert traffic is not modeled, so throughput may be optimistic.")
    if comm.dcp_advisory:
        alerts.append("Long-context KV sharding can shift real capacity versus this simplified estimate.")
    return alerts


def _precision_alerts(prec: str, gpu: Optional[GPU]) -> list[str]:
    if gpu is None:
        return []
    if prec == "nvfp4" and not gpu_supports_nvfp4(gpu):
        return [f"NVFP4 is not native on {gpu.name}; compute is discounted for dequant/packing fallback."]
    if prec == "mxfp4" and not gpu_supports_mxfp4(gpu):
        return [f"MXFP4 is not native on {gpu.name}; compute is discounted for dequant/packing fallback."]
    return []


def _quantization_profile_alerts(model: Model, prec: str) -> list[str]:
    profile = model.quantization_profile(prec)
    if profile is None:
        return []
    if profile.source_kind == "family":
        return [f"{profile.label} uses a family proxy from {profile.source_repo}; exact artifact tensor headers are not pinned yet."]
    return []


def _build_model_info(state: PlannerState, am: ModelAssignment, gpu_pool: Optional[GpuPool], prefill_mem, decode_mem) -> dict:
    model = MODELS[am.model_key]
    gpu = gpu_pool.gpu if gpu_pool else None
    quant_profiles_by_precision = {
        prec: profile
        for prec in PRECISIONS
        if (profile := model.quantization_profile(prec)) is not None
    }
    quant_profile = model.quantization_profile(am.prec)

    strats: list[tuple[int, int, int]] = []
    recommended_label = ""
    alt_prec = None
    alt_fits_now = False
    selected_min_gpu_count = None
    selected_pool_min_gpu_count = None
    alt_min_gpu_count = None
    alt_pool_min_gpu_count = None
    if gpu and am.gpu_count > 0:
        strats = valid_strategies(model, am.gpu_count, gpu, state.mu, state.profiled_non_kv_gb, am.prec)
        recommended_label = strategy_label(*_preferred_strategy(state, am, gpu, "decode"))

    avg_in = avg_dist(state.in_dist, INPUT_BUCKETS)
    avg_out = avg_dist(state.out_dist, OUTPUT_BUCKETS)
    avg_seq = avg_in + avg_out / 2.0
    realtime_profile = getattr(model, "realtime_profile", None)
    embedding_profile = getattr(model, "embedding_profile", None)
    embedding_stats = None
    if realtime_profile is not None:
        avg_seq = float(realtime_profile.state_tokens)
    if embedding_profile is not None:
        embedding_stats = embedding_doc_stats(model, state.embedding_doc_dist, EMBEDDING_DOC_BUCKETS, am.prec)
        avg_seq = float(embedding_stats.mean_seq_len)

    decode_max_slots = 0
    if decode_mem and avg_seq > 0:
        decode_kv = per_replica_kv_cache_bytes(model, avg_seq, am.prec, am.pp, am.tp)
        decode_per_replica = int(decode_mem.kv_budget / decode_kv) if decode_kv > 0 else 0
        if state.decode_efficiency.sched_budget > 0:
            decode_per_replica = min(decode_per_replica, state.decode_efficiency.sched_budget)
        decode_max_slots = decode_per_replica * am.dp

    prefill_probe_len = max(1, effective_prefill_length(max(state.task_il, avg_in), state.prefix_hit_rate))
    prefill_max_batch = 0
    if prefill_mem and prefill_probe_len > 0:
        prefill_kv = per_replica_kv_cache_bytes(model, prefill_probe_len, am.prec, am.prefill_pp, am.prefill_tp)
        prefill_max_batch = (int(prefill_mem.kv_budget / prefill_kv) if prefill_kv > 0 else 0) * am.prefill_dp

    others_used = sum(x.gpu_count for x in state.models if x.uid != am.uid and x.gpu_uid == am.gpu_uid)
    max_avail = gpu_pool.count - others_used if gpu_pool else 0
    if gpu:
        needs_now = {
            prec: _min_gpu_count_for_pool(model, gpu, state.mu, state.profiled_non_kv_gb, prec, max_avail)
            for prec in PRECISIONS
        }
        needs_pool = {
            prec: _min_gpu_count_for_pool(model, gpu, state.mu, state.profiled_non_kv_gb, prec, gpu_pool.count)
            for prec in PRECISIONS
        }
        selected_need = needs_now[am.prec]
        selected_pool_need = needs_pool[am.prec]
        fit_now = [
            prec for prec in PRECISIONS
            if prec != am.prec and am.gpu_count > 0
            and valid_strategies(model, am.gpu_count, gpu, state.mu, state.profiled_non_kv_gb, prec)
        ]
        if fit_now:
            alt_prec = fit_now[0]
            alt_fits_now = True
            alt_need = needs_now[alt_prec]
            alt_pool_need = needs_pool[alt_prec]
        else:
            alt_prec, alt_pool_need = _best_precision_need({prec: need for prec, need in needs_pool.items() if prec != am.prec})
            alt_need = needs_now.get(alt_prec, math.inf) if alt_prec else math.inf
        if not math.isinf(selected_need):
            selected_min_gpu_count = int(selected_need)
        if not math.isinf(selected_pool_need):
            selected_pool_min_gpu_count = int(selected_pool_need)
        if alt_prec and not math.isinf(alt_need):
            alt_min_gpu_count = int(alt_need)
        if alt_prec and not math.isinf(alt_pool_need):
            alt_pool_min_gpu_count = int(alt_pool_need)
    topology_label = strategy_label(am.tp, am.pp, am.dp)
    mem = decode_mem
    prefill_kv_gb = f"{prefill_mem.kv_budget / 1e9:.0f}" if prefill_mem else "0"
    decode_kv_gb = f"{decode_mem.kv_budget / 1e9:.0f}" if decode_mem else "0"
    realtime = None
    if realtime_profile is not None and gpu and am.gpu_count > 0 and decode_mem is not None:
        max_realtime_users = compute_realtime_max_users(
            model,
            (am.tp, am.pp, am.dp),
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )
        sample_users = max(max_realtime_users, 1)
        sample = compute_realtime_capacity(
            model,
            (am.tp, am.pp, am.dp),
            sample_users,
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )
        realtime = {
            "profile": realtime_profile,
            "max_users": max_realtime_users,
            "sample": sample,
        }

    embedding = None
    if embedding_profile is not None and gpu and am.gpu_count > 0 and prefill_mem is not None:
        if embedding_stats is None:
            embedding_stats = embedding_doc_stats(model, state.embedding_doc_dist, EMBEDDING_DOC_BUCKETS, am.prec)
        best_embedding = None
        for bs in _probe_batch_sizes(max(am.prefill_dp, 1)):
            sample = compute_embedding_distribution(
                model,
                (am.prefill_tp, am.prefill_pp, am.prefill_dp),
                bs,
                state.embedding_doc_dist,
                EMBEDDING_DOC_BUCKETS,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.prefill_efficiency,
            )
            if sample is None:
                continue
            if best_embedding is None or sample.rps > best_embedding.rps:
                best_embedding = sample
        doc_distribution = []
        weights = []
        total = sum(max(int(v or 0), 0) for v in state.embedding_doc_dist) or 1
        for i, bucket in enumerate(EMBEDDING_DOC_BUCKETS):
            raw = state.embedding_doc_dist[i] if i < len(state.embedding_doc_dist) else 0
            share = max(int(raw or 0), 0) / total
            weights.append(share)
            if share <= 0:
                continue
            clipped = embedding_sequence_length(model, bucket.length)
            doc_distribution.append({
                "label": bucket.label,
                "length": bucket.length,
                "clipped_length": clipped,
                "share": share,
                "color": bucket.color,
            })
        embedding = {
            "profile": embedding_profile,
            "seq_len": round(embedding_stats.mean_seq_len),
            "p50_seq_len": embedding_stats.p50_seq_len,
            "p90_seq_len": embedding_stats.p90_seq_len,
            "p99_seq_len": embedding_stats.p99_seq_len,
            "vectors_per_input": embedding_stats.mean_vectors_per_input,
            "output_kb_per_input": embedding_stats.mean_output_bytes_per_input / 1e3,
            "doc_distribution": doc_distribution,
            "doc_distribution_weights": weights,
            "sample": best_embedding,
        }

    return {
        "am": am,
        "model": model,
        "gpu_pool": gpu_pool,
        "gpu": gpu,
        "mem": mem,
        "prefill_mem": prefill_mem,
        "decode_mem": decode_mem,
        "strats": strats,
        "kv_gb": f"{mem.kv_budget / 1e9:.0f}" if mem else "0",
        "prefill_kv_gb": prefill_kv_gb,
        "decode_kv_gb": decode_kv_gb,
        "max_slots": decode_max_slots,
        "decode_max_slots": decode_max_slots,
        "requested_gb": f"{mem.requested / 1e9:.0f}" if mem else "0",
        "profiled_non_kv_total_gb": f"{mem.profiled_non_kv / 1e9:.0f}" if mem else "0",
        "kv_reserved_gb": f"{mem.kv_reserved / 1e9:.0f}" if mem else "0",
        "prefill_max_batch": prefill_max_batch,
        "prefill_probe_len": prefill_probe_len,
        "max_avail": max_avail,
        "gpu_count_options": _gpu_count_options(max_avail, am.gpu_count, gpu),
        "weight_bpp": model.weight_bytes_per_param(am.prec),
        "quant_profiles_by_precision": quant_profiles_by_precision,
        "quant_profile": quant_profile,
        "realtime": realtime,
        "embedding": embedding,
        "mixed_weight_precision": model.uses_mixed_weight_precision(am.prec),
        "fits": mem is not None,
        "decode_fits": decode_mem is not None,
        "prefill_fits": prefill_mem is not None,
        "runnable": bool(strats),
        "recommended_label": recommended_label,
        "topology_label": topology_label,
        "selected_min_gpu_count": selected_min_gpu_count,
        "selected_pool_min_gpu_count": selected_pool_min_gpu_count,
        "alt_prec": alt_prec,
        "alt_fits_now": alt_fits_now,
        "alt_min_gpu_count": alt_min_gpu_count,
        "alt_pool_min_gpu_count": alt_pool_min_gpu_count,
        "comm_summary": _comm_summary(am.tp, am.pp),
        "precision_alerts": _precision_alerts(am.prec, gpu) + _quantization_profile_alerts(model, am.prec),
        "alerts": _comm_alerts(model, am.tp, am.pp, am.dp, gpu, avg_seq, state.decode_efficiency),
        "prefill_comm_summary": _comm_summary(am.prefill_tp, am.prefill_pp),
        "decode_comm_summary": _comm_summary(am.tp, am.pp),
        "prefill_alerts": _comm_alerts(model, am.prefill_tp, am.prefill_pp, am.prefill_dp, gpu, prefill_probe_len, state.prefill_efficiency),
        "decode_alerts": _comm_alerts(model, am.tp, am.pp, am.dp, gpu, avg_seq, state.decode_efficiency),
        "decode_exceeds_node": bool(gpu and am.tp > gpu.node_size),
        "prefill_exceeds_node": bool(gpu and am.prefill_tp > gpu.node_size),
    }


def get_model_info(state: PlannerState, am: ModelAssignment) -> dict:
    gpu_pool = state.find_gpu(am.gpu_uid)
    gpu = gpu_pool.gpu if gpu_pool else None
    prefill_mem = None
    decode_mem = None
    if gpu and am.gpu_count > 0:
        prefill_mem, decode_mem = _assignment_memories(state, am, gpu)
    return _build_model_info(state, am, gpu_pool, prefill_mem, decode_mem)


def get_model_infos(state: PlannerState) -> list[dict]:
    infos = []
    for am in state.models:
        gpu_pool = state.find_gpu(am.gpu_uid)
        gpu = gpu_pool.gpu if gpu_pool else None
        prefill_mem = None
        decode_mem = None
        if gpu and am.gpu_count > 0:
            prefill_mem, decode_mem = _assignment_memories(state, am, gpu)
        infos.append(_build_model_info(state, am, gpu_pool, prefill_mem, decode_mem))
    return infos
