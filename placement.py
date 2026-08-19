"""Placement, retune, and auto-select engines for the GPU/LLM Usage Modeler."""

from __future__ import annotations

import math
from typing import Optional

from calc import (
    avg_dist,
    compute_decode,
    compute_embedding_distribution,
    compute_memory,
    compute_prefill,
    default_strategy,
    effective_prefill_length,
    resolve_spec_runtime,
    valid_strategies,
)
from data import (
    EMBEDDING_DOC_BUCKETS,
    GPU,
    INPUT_BUCKETS,
    MODELS,
    PRECISIONS,
    Model,
    effective_quality,
    model_profile_quality,
    model_profile_success_rate,
    required_quality,
)
from state import (
    DEFAULT_AUTO_MODEL_STRATEGY,
    GpuPool,
    ModelAssignment,
    ModelAssignmentProxy,
    PlannerState,
    Project,
    _next_uid,
    normalize_auto_strategy,
)


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


def _preferred_strategy(
    state: PlannerState, am: ModelAssignment, gpu: GPU, phase: str
) -> tuple[int, int, int]:
    model = MODELS[am.model_key]
    spec = resolve_spec_runtime(model, am.spec_method, am.spec_k, state.spec_acceptance, am.prec)
    candidates = valid_strategies(
        model, am.gpu_count, gpu, state.mu, state.profiled_non_kv_gb, am.prec, spec
    )
    if not candidates:
        return default_strategy(
            model, am.gpu_count, gpu, state.mu, state.profiled_non_kv_gb, am.prec, spec
        )

    best = candidates[0]
    best_score: tuple[float, ...] | None = None
    probe_prefill_len = max(
        1,
        effective_prefill_length(
            max(state.task_il, avg_dist(state.in_dist, INPUT_BUCKETS)), state.prefix_hit_rate
        ),
    )
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
            spec,
        )
        kv_headroom = mem.kv_budget if mem else 0.0
        local_tp = 1 if tp <= gpu.node_size else 0
        peak_tps = -1
        aux = float("-inf")

        for bs in _probe_batch_sizes(dp):
            if is_embedding:
                embedding_result = compute_embedding_distribution(
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
                if embedding_result is None:
                    continue
                metric = (embedding_result.tps, embedding_result.rps)
            elif phase == "prefill":
                prefill_result = compute_prefill(
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
                    spec,
                )
                if prefill_result is None:
                    continue
                metric = (prefill_result.tps, prefill_result.rps)
            else:
                decode_result = compute_decode(
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
                    spec,
                )
                if decode_result is None:
                    continue
                metric = (decode_result.tps, -decode_result.lat)

            if metric[0] > peak_tps or (metric[0] == peak_tps and metric[1] > aux):
                peak_tps = metric[0]
                aux = metric[1]

        if peak_tps < 0:
            score: tuple[float, ...] = (local_tp, min(tp, gpu.node_size), dp, -pp, kv_headroom)
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
    spec = resolve_spec_runtime(model, am.spec_method, am.spec_k, state.spec_acceptance, am.prec)
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
            spec,
        )
        if (am.prefill_tp, am.prefill_pp, am.prefill_dp) not in valid:
            am.prefill_tp, am.prefill_pp, am.prefill_dp = embedding_default
        am.tp, am.pp, am.dp = am.prefill_tp, am.prefill_pp, am.prefill_dp
        return

    decode_default = _preferred_strategy(state, am, gp.gpu, "decode")
    # A ModelAssignment has one physical GPU allocation. Until separate prefill and
    # decode pools are represented explicitly, both phases must share one resident
    # TP/PP/DP layout or the planner would double-count the same GPUs and VRAM.
    prefill_default = decode_default
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
        spec,
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
        spec,
    )
    if (am.prefill_tp, am.prefill_pp, am.prefill_dp) not in prefill_valid or (
        am.prefill_tp,
        am.prefill_pp,
        am.prefill_dp,
    ) != (am.tp, am.pp, am.dp):
        am.prefill_tp, am.prefill_pp, am.prefill_dp = am.tp, am.pp, am.dp


def retune_models(state: PlannerState, preserve_existing: bool = True):
    for am in state.models:
        if am.gpu_count > 0:
            _retune_model(state, am, preserve_existing=preserve_existing)


def _assignment_memories(state: PlannerState, am: ModelAssignment, gpu: GPU):
    if (am.prefill_tp, am.prefill_pp, am.prefill_dp) != (am.tp, am.pp, am.dp):
        return None, None
    model = MODELS[am.model_key]
    spec = resolve_spec_runtime(model, am.spec_method, am.spec_k, state.spec_acceptance, am.prec)
    prefill_mem = compute_memory(
        model,
        am.prefill_tp,
        am.prefill_pp,
        gpu,
        state.mu,
        state.profiled_non_kv_gb,
        am.prec,
        state.prefill_efficiency,
        spec,
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
        spec,
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


def _model_serves_project(model: Model, project: Project) -> bool:
    if (
        getattr(model, "is_realtime_only", False)
        or getattr(model, "embedding_profile", None) is not None
    ):
        return False
    domain = getattr(project, "quality_domain", "general")
    weights = getattr(project, "quality_weights", None)
    return (
        project.requires <= model.capabilities
        and model_profile_quality(model, weights, domain) + 1e-9
        >= float(getattr(project, "quality_floor", 0.0))
        and model_profile_success_rate(model, project.difficulty, weights, domain) + 1e-9
        >= project.min_success_rate
    )


def _active_project_demand(project: Project) -> float:
    return max(0.0, float(project.tokens_day or 0.0)) + 0.25 * max(
        0.0, float(project.latent_jobs_day or 0.0)
    )


def _best_available_placement(
    state: PlannerState, model: Model
) -> Optional[tuple[GpuPool, int, str]]:
    placements: list[tuple[tuple[int, int, float, int, int, int], GpuPool, str]] = []
    for pool_order, gp in enumerate(state.gpus):
        avail = state.free_gpu_for_pool(gp.uid)
        if avail <= 0:
            continue
        for prec in PRECISIONS:
            need = _min_gpu_count_for_pool(
                model, gp.gpu, state.mu, state.profiled_non_kv_gb, prec, avail
            )
            if math.isinf(need):
                continue
            placements.append(
                (
                    (int(need), PRECISIONS.index(prec), -gp.gpu.mem, -avail, gp.uid, pool_order),
                    gp,
                    prec,
                )
            )
    if not placements:
        return None
    key, gp, prec = min(placements, key=lambda placement: placement[0])
    need = key[0]
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
        need = _min_gpu_count_for_pool(
            model, gp.gpu, state.mu, state.profiled_non_kv_gb, prec, avail
        )
        if not math.isinf(need):
            placements.append((int(need), PRECISIONS.index(prec), prec))
    if not placements:
        return None

    need, _, prec = min(placements)
    return need, prec


def _auto_assignment_demand(state: PlannerState, am: ModelAssignment) -> float:
    model = MODELS[am.model_key]
    demand = sum(
        _active_project_demand(project)
        for project in state.projects
        if _model_serves_project(model, project)
    )
    return demand or model.quality * 1e6


def _auto_model_value(model: Model, projects: list[Project]) -> float:
    value = 0.0
    for project in projects:
        if not _model_serves_project(model, project):
            continue
        sr = model_profile_success_rate(
            model,
            project.difficulty,
            getattr(project, "quality_weights", None),
            getattr(project, "quality_domain", "general"),
        )
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
    return (
        sum(
            _active_project_demand(project)
            * model_profile_success_rate(
                model,
                project.difficulty,
                getattr(project, "quality_weights", None),
                getattr(project, "quality_domain", "general"),
            )
            for project in served
        )
        / total
    )


def _auto_quality_margin(model: Model, projects: list[Project]) -> float:
    served = _auto_served_projects(model, projects)
    if not served:
        return 0.0
    return min(
        model_profile_success_rate(
            model,
            project.difficulty,
            getattr(project, "quality_weights", None),
            getattr(project, "quality_domain", "general"),
        )
        - float(project.min_success_rate)
        for project in served
    )


def _auto_covered_demand(model: Model, projects: list[Project]) -> float:
    return sum(
        _active_project_demand(project) for project in _auto_served_projects(model, projects)
    )


def _auto_portfolio_quality(model: Model, projects: list[Project]) -> float:
    weighted = [
        (
            _active_project_demand(project) * max(0.01, float(project.wtp_per_m or 0.0)),
            model_profile_quality(
                model,
                getattr(project, "quality_weights", None),
                getattr(project, "quality_domain", "general"),
            ),
        )
        for project in projects
    ]
    total = sum(weight for weight, _ in weighted)
    if total <= 0:
        return effective_quality(model)
    return sum(weight * quality for weight, quality in weighted) / total


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
    quality = _auto_portfolio_quality(model, projects)
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
            candidates.append(
                (
                    _auto_candidate_key(model, projects, gpu_count, prec, strategy),
                    model,
                    prec,
                )
            )

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
    served_projects = [
        project for project in state.projects if _model_serves_project(model, project)
    ]
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
                model = MODELS[am.model_key]
                spec = resolve_spec_runtime(
                    model, am.spec_method, am.spec_k, state.spec_acceptance, am.prec
                )
                if not valid_strategies(
                    model, next_count, gp.gpu, state.mu, state.profiled_non_kv_gb, am.prec, spec
                ):
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

    strategy = normalize_auto_strategy(
        strategy or getattr(state, "auto_strategy", DEFAULT_AUTO_MODEL_STRATEGY)
    )
    state.auto_strategy = strategy
    original_models = list(state.models)
    state.models = []
    projects = [project for project in state.projects if _active_project_demand(project) > 0]
    if not projects:
        projects = [
            Project(
                _next_uid(),
                "Balanced chat",
                0.30,
                1.0,
                1.0,
                min_success_rate=0.90,
                quality_floor=0.55,
            ),
            Project(
                _next_uid(),
                "Coding / reasoning",
                0.55,
                1.0,
                4.0,
                requires=frozenset({"tools", "ctx_128k"}),
                min_success_rate=0.85,
                quality_floor=0.70,
            ),
            Project(
                _next_uid(),
                "Frontier reasoning",
                0.90,
                1.0,
                20.0,
                requires=frozenset({"tools", "reasoning"}),
                min_success_rate=0.95,
                quality_floor=0.90,
            ),
        ]

    selected_keys: set[str] = set()
    ordered_projects = sorted(
        projects,
        key=lambda project: (
            -required_quality(
                project.difficulty,
                project.min_success_rate,
                quality_floor=getattr(project, "quality_floor", 0.0),
            ),
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
            if (
                model.hidden
                or model.key in selected_keys
                or model.key in state.auto_excluded
                or not _model_serves_project(model, project)
            ):
                continue
            placement = _best_available_placement(state, model)
            if placement is None:
                continue
            _, gpu_count, prec = placement
            candidates.append(
                (
                    _auto_candidate_key(
                        model,
                        projects if strategy == "coverage" else [project],
                        gpu_count,
                        prec,
                        strategy,
                    ),
                    model,
                    placement,
                )
            )
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
