"""Server-side state management for the vLLM planner."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

from data import (
    GPUS,
    MODELS,
    DIST_PRESETS,
    INPUT_BUCKETS,
    OUTPUT_BUCKETS,
    DAY_SHAPES,
    GPU,
    Model,
    PROJECT_PRESETS,
    CORPO_CLOUD_PRESETS,
    CORPO_CLOUD_DEFAULT,
    MODEL_CAPABILITIES,
    PRECISIONS,
    PRECISION_LABELS,
    normalize_precision,
)
from calc import (
    EfficiencyParams,
    avg_dist,
    communication_breakdown,
    compute_decode,
    compute_memory,
    compute_prefill,
    default_strategy,
    effective_prefill_length,
    gpu_supports_mxfp4,
    gpu_supports_nvfp4,
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
    # Latent demand that unlocks only when on-prem $/M drops below the project's unlock threshold.
    "latent_jobs_day":     (0.0, 1e12),
    "unlock_price_per_m":  (0.0, 200.0),
}
ALLOWED_CAPABILITIES = frozenset(MODEL_CAPABILITIES)
VISIBLE_PLOT_MODES = (
    ("userpareto", "User Pareto"),
    ("processingpareto", "Processing Pareto"),
)
DEFAULT_PLOT_MODE = VISIBLE_PLOT_MODES[0][0]
ALLOWED_PLOT_MODES = frozenset(mode for mode, _ in VISIBLE_PLOT_MODES)
DEFAULT_DAY_SHAPE = "workday"
ALLOWED_DAY_SHAPES = frozenset(DAY_SHAPES)
ALLOWED_CORPO_CLOUDS = frozenset(CORPO_CLOUD_PRESETS)
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
    return mode if mode in ALLOWED_PLOT_MODES else DEFAULT_PLOT_MODE


def normalize_day_shape(shape: Optional[str]) -> str:
    return shape if shape in ALLOWED_DAY_SHAPES else DEFAULT_DAY_SHAPE


def normalize_corpo_cloud(name: Optional[str]) -> str:
    return name if name in ALLOWED_CORPO_CLOUDS else CORPO_CLOUD_DEFAULT


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
    batch_eligible: bool = False  # if True, batch-shiftable off-peak (works with night batching)
    # Hard capability gates: a model must supply ALL listed capabilities to be eligible.
    requires: frozenset[str] = frozenset()
    # Quality SLO: project rejects any candidate whose success_rate(model.quality, difficulty)
    # falls below this floor.
    min_success_rate: float = 0.85
    # Latent demand — hidden workload that only materializes when on-prem $/M falls at or
    # below unlock_price_per_m. Hard threshold: the pool is all-or-nothing per routing pass.
    latent_jobs_day: float = 0.0
    unlock_price_per_m: float = 0.0
    # Per-project input / output length preset. The aggregate state.in_dist / state.out_dist
    # used by calc.py are a demand-weighted blend across all projects' presets.
    in_pre: str = "Chat"
    out_pre: str = "Chat"

    def __post_init__(self):
        if not isinstance(self.requires, frozenset):
            self.requires = frozenset(c for c in (self.requires or ()) if c in ALLOWED_CAPABILITIES)
        self.difficulty = min(max(float(self.difficulty), 0.0), 1.0)
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
    in_dist: list[int] = field(default_factory=lambda: list(DIST_PRESETS["Chat"]["in"]))
    out_dist: list[int] = field(default_factory=lambda: list(DIST_PRESETS["Chat"]["out"]))
    in_pre: str = "Chat"
    out_pre: str = "Chat"
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
            if phase == "prefill":
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


def _add_project_from_preset(state: PlannerState, preset_key: str) -> Optional[Project]:
    preset = _find_preset(preset_key)
    if preset is None:
        return None
    proj = Project(
        uid=_next_uid(),
        name=preset["name"],
        difficulty=float(preset["difficulty"]),
        tokens_day=float(preset["tokens_day"]),
        wtp_per_m=float(preset["wtp_per_m"]),
        batch_eligible=bool(preset.get("batch_eligible", False)),
        requires=frozenset(preset.get("requires", ())),
        min_success_rate=float(preset.get("min_success_rate", 0.85)),
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
    proj = Project(
        uid=_next_uid(),
        name="New project",
        difficulty=0.3,
        tokens_day=500e6,
        wtp_per_m=1.0,
        batch_eligible=False,
        requires=frozenset(),
        min_success_rate=0.85,
        latent_jobs_day=0.0,
        unlock_price_per_m=0.0,
        in_pre="Chat",
        out_pre="Chat",
    )
    state.projects.append(proj)
    _sync_aggregate_distribution(state)
    return proj


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
    project's in_pre / out_pre preset. calc.py still consumes the single aggregate
    distribution; this just keeps it in sync with the projects' declared shapes."""
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


def add_gpu(state: PlannerState, gpu_type: str, count: int = 8):
    existing = next((g for g in state.gpus if g.gpu_type == gpu_type), None)
    if existing:
        existing.count += count
    else:
        state.gpus.append(GpuPool(_next_uid(), gpu_type, count))


def remove_gpu(state: PlannerState, gpu_uid: int):
    state.models = [m for m in state.models if m.gpu_uid != gpu_uid]
    state.gpus = [g for g in state.gpus if g.uid != gpu_uid]


def change_gpu_qty(state: PlannerState, gpu_uid: int, delta: int):
    gp = state.find_gpu(gpu_uid)
    if gp is None:
        return

    new_count = max(0, gp.count + delta)
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
    if kind == "in":
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
    return s


def get_compare_state(session_id: str) -> Optional[PlannerState]:
    state = _compare_states.get(session_id)
    if state is not None:
        for am in state.models:
            am.prec = normalize_precision(getattr(am, "prec", "bf16"))
        state.mode = normalize_plot_mode(state.mode)
        state.projection_day_shape = normalize_day_shape(state.projection_day_shape)
        state.corpo_cloud = normalize_corpo_cloud(getattr(state, "corpo_cloud", CORPO_CLOUD_DEFAULT))
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


def _build_model_info(state: PlannerState, am: ModelAssignment, gpu_pool: Optional[GpuPool], prefill_mem, decode_mem) -> dict:
    model = MODELS[am.model_key]
    gpu = gpu_pool.gpu if gpu_pool else None

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

    decode_max_slots = 0
    if decode_mem and avg_seq > 0 and decode_mem.kv_per_token > 0:
        decode_per_replica = int(decode_mem.kv_budget / (avg_seq * decode_mem.kv_per_token))
        if state.decode_efficiency.sched_budget > 0:
            decode_per_replica = min(decode_per_replica, state.decode_efficiency.sched_budget)
        decode_max_slots = decode_per_replica * am.dp

    prefill_probe_len = max(1, effective_prefill_length(max(state.task_il, avg_in), state.prefix_hit_rate))
    prefill_max_batch = 0
    if prefill_mem and prefill_probe_len > 0 and prefill_mem.kv_per_token > 0:
        prefill_max_batch = int(prefill_mem.kv_budget / (prefill_probe_len * prefill_mem.kv_per_token)) * am.prefill_dp

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
        "weight_bpp": model.weight_bytes_per_param(am.prec),
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
        "precision_alerts": _precision_alerts(am.prec, gpu),
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
