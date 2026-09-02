"""Presentation view-model builders for model assignment cards."""

from __future__ import annotations

import math
from typing import Optional

from calc import (
    EfficiencyParams,
    avg_dist,
    communication_breakdown,
    compute_decode,
    compute_embedding_distribution,
    compute_realtime_capacity,
    compute_realtime_max_users,
    effective_prefill_length,
    embedding_doc_stats,
    embedding_sequence_length,
    gpu_supports_mxfp4,
    gpu_supports_nvfp4,
    per_replica_kv_cache_bytes,
    resolve_spec_runtime,
    spec_optimization_for,
    strategy_label,
    valid_strategies,
)
from data import (
    EMBEDDING_DOC_BUCKETS,
    GPU,
    INPUT_BUCKETS,
    MODELS,
    OUTPUT_BUCKETS,
    PRECISION_SPECS,
    PRECISIONS,
    Model,
)
from placement import (
    _assignment_memories,
    _best_precision_need,
    _gpu_count_options,
    _min_gpu_count_for_pool,
    _preferred_strategy,
    _probe_batch_sizes,
)
from state import (
    GpuPool,
    ModelAssignment,
    PlannerState,
)


def _comm_summary(tp: int, pp: int) -> str:
    terms = []
    if tp > 1:
        terms.append("dense TP reductions")
    if pp > 1:
        terms.append("PP stage boundaries")
    return "Comm model: " + " + ".join(terms) if terms else ""


def _comm_alerts(
    model: Model,
    tp: int,
    pp: int,
    dp: int,
    gpu: Optional[GPU],
    avg_seq: float,
    eff: EfficiencyParams,
) -> list[str]:
    if gpu is None:
        return []

    batch_tokens = max(1, min(32, math.ceil(32 / max(dp, 1))))
    comm = communication_breakdown(model, tp, pp, batch_tokens, avg_seq, gpu, eff)
    alerts: list[str] = []
    if comm.tp_cross_node:
        alerts.append(
            f"{strategy_label(tp, pp, dp)} uses cross-node TP (node size {gpu.node_size}). Prefer TP within a node and scale with PP/DP."
        )
    if comm.expert_parallel_unmodeled:
        scope = "Multi-node " if comm.ep_advisory else ""
        alerts.append(
            f"{scope}MoE expert dispatch/combine traffic is excluded, so throughput may be optimistic."
        )
    if comm.dcp_advisory:
        alerts.append(
            "Long-context KV sharding can shift real capacity versus this simplified estimate."
        )
    return alerts


def _precision_alerts(prec: str, gpu: Optional[GPU]) -> list[str]:
    if gpu is None:
        return []
    if prec == "nvfp4" and not gpu_supports_nvfp4(gpu):
        return [
            f"NVFP4 is not native on {gpu.name}; compute is discounted for dequant/packing fallback."
        ]
    if prec == "mxfp4" and not gpu_supports_mxfp4(gpu):
        return [
            f"MXFP4 is not native on {gpu.name}; compute is discounted for dequant/packing fallback."
        ]
    return []


def _quantization_profile_alerts(model: Model, prec: str) -> list[str]:
    profile = model.quantization_profile(prec)
    if profile is None:
        return []
    if profile.source_kind == "family":
        return [
            f"{profile.label} uses a family proxy from {profile.source_repo}; exact artifact tensor headers are not pinned yet."
        ]
    return []


def _precision_options(model: Model, selected: str) -> list[dict[str, str | bool]]:
    """Native release format first, followed by strictly smaller choices.

    A larger precision from an imported scenario is kept as a legacy selected
    option so the browser never silently rewrites persisted planner state.
    """
    native = model.native_precision_key
    native_bpp = model.weight_bytes_per_param(native)
    keys = [native]
    keys.extend(
        sorted(
            (
                prec
                for prec in PRECISIONS
                if prec != native and model.weight_bytes_per_param(prec) < native_bpp - 1e-9
            ),
            key=model.weight_bytes_per_param,
            reverse=True,
        )
    )
    if selected not in keys:
        keys.append(selected)

    options: list[dict[str, str | bool]] = []
    for prec in keys:
        profile = model.quantization_profile(prec)
        is_native = prec == native
        label = (
            f"Native · {model.native_precision_display}"
            if is_native
            else PRECISION_SPECS[prec].label
        )
        if profile is not None and not is_native:
            label += " · artifact"
        elif not is_native:
            label += " · estimated"
        if prec == selected and prec not in (native, *PRECISIONS):
            label += " · legacy"
        description = (
            model.native_precision_description if is_native else PRECISION_SPECS[prec].description
        )
        if profile is not None:
            description += f" Artifact: {profile.source_repo}. {profile.notes}"
        elif not is_native:
            description = "Estimated conversion, not a released artifact. " + description
        options.append(
            {
                "key": prec,
                "label": label,
                "description": description,
                "native": is_native,
            }
        )
    return options


def _build_model_info(
    state: PlannerState, am: ModelAssignment, gpu_pool: Optional[GpuPool], prefill_mem, decode_mem
) -> dict:
    model = MODELS[am.model_key]
    gpu = gpu_pool.gpu if gpu_pool else None
    quant_profiles_by_precision = {
        prec: profile
        for prec in PRECISIONS
        if (profile := model.quantization_profile(prec)) is not None
    }
    quant_profile = model.quantization_profile(am.prec)
    precision_options = _precision_options(model, am.prec)

    strats: list[tuple[int, int, int]] = []
    recommended_label = ""
    alt_prec = None
    alt_fits_now = False
    selected_min_gpu_count = None
    selected_pool_min_gpu_count = None
    alt_min_gpu_count = None
    alt_pool_min_gpu_count = None
    spec_runtime = resolve_spec_runtime(
        model, am.spec_method, am.spec_k, state.spec_acceptance, am.prec
    )
    if gpu and am.gpu_count > 0:
        strats = valid_strategies(
            model, am.gpu_count, gpu, state.mu, state.profiled_non_kv_gb, am.prec, spec_runtime
        )
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
        embedding_stats = embedding_doc_stats(
            model, state.embedding_doc_dist, EMBEDDING_DOC_BUCKETS, am.prec
        )
        avg_seq = float(embedding_stats.mean_seq_len)

    decode_max_slots = 0
    if decode_mem and avg_seq > 0:
        decode_kv = per_replica_kv_cache_bytes(model, avg_seq, am.prec, am.pp, am.tp)
        if spec_runtime is not None:
            # The drafter's own small KV cache shares the sequence's KV budget.
            decode_kv *= 1.0 + spec_runtime.profile.kv_overhead
        decode_per_replica = int(decode_mem.kv_budget / decode_kv) if decode_kv > 0 else 0
        if state.decode_efficiency.sched_budget > 0:
            decode_per_replica = min(decode_per_replica, state.decode_efficiency.sched_budget)
        decode_max_slots = decode_per_replica * am.dp

    spec_info = None
    if spec_runtime is not None:
        probe = None
        probe_bs = max(1, min(32, decode_max_slots or 1))
        optimization = None
        if am.spec_k == 0 and gpu and am.gpu_count > 0:
            optimization = spec_optimization_for(state, am, model)
            if optimization.runtime is not None:
                spec_runtime = optimization.runtime
                probe_bs = optimization.probe_concurrency
        if gpu and am.gpu_count > 0 and optimization is None:
            probe = compute_decode(
                model,
                am.tp,
                am.pp,
                probe_bs,
                am.dp,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                am.prec,
                state.in_dist,
                state.out_dist,
                state.decode_efficiency,
                spec_runtime,
            )
        note = str(getattr(spec_runtime.profile, "note", "") or "").lower()
        prior_is_unmeasured = any(
            marker in note
            for marker in (
                "unmeasured",
                "not a measured",
                "no acceptance benchmark",
                "assumption",
                "family proxy",
            )
        )
        if state.spec_acceptance > 0:
            alpha_source = "user override"
            unmeasured_prior = True
        elif prior_is_unmeasured:
            alpha_source = "unmeasured profile prior"
            unmeasured_prior = True
        elif spec_runtime.k in dict(getattr(spec_runtime.profile, "acceptance_alpha_by_k", ())):
            alpha_source = f"measured per-k calibration (k={spec_runtime.k})"
            unmeasured_prior = False
        elif spec_runtime.k != spec_runtime.profile.default_k:
            alpha_source = f"calibration extrapolated from k={spec_runtime.profile.default_k}"
            unmeasured_prior = False
        else:
            alpha_source = "measured profile calibration"
            unmeasured_prior = False
        spec_info = {
            "profile": spec_runtime.profile,
            "k": spec_runtime.k,
            "alpha": spec_runtime.alpha,
            "alpha_source": alpha_source,
            "unmeasured_prior": unmeasured_prior,
            "tau": spec_runtime.tau,
            "draft_gb": spec_runtime.draft_weight_bytes / 1e9,
            "probe_bs": probe_bs,
            "speedup": optimization.speedup
            if optimization is not None
            else (probe.spec_speedup if probe else 0.0),
            "beneficial": optimization.beneficial
            if optimization is not None
            else bool(probe and probe.spec_speedup >= 1.0),
            "active": optimization.beneficial if optimization is not None else True,
            "reason": optimization.reason if optimization is not None else "manual k",
        }

    prefill_probe_len = max(
        1, effective_prefill_length(max(state.task_il, avg_in), state.prefix_hit_rate)
    )
    prefill_max_batch = 0
    if prefill_mem and prefill_probe_len > 0:
        prefill_kv = per_replica_kv_cache_bytes(
            model, prefill_probe_len, am.prec, am.prefill_pp, am.prefill_tp
        )
        prefill_max_batch = (
            int(prefill_mem.kv_budget / prefill_kv) if prefill_kv > 0 else 0
        ) * am.prefill_dp

    others_used = sum(
        x.gpu_count for x in state.models if x.uid != am.uid and x.gpu_uid == am.gpu_uid
    )
    max_avail = gpu_pool.count - others_used if gpu_pool else 0
    # gpu is derived from gpu_pool above, so a truthy gpu already implies a
    # pool; name the dependency rather than leaving it to inference.
    if gpu and gpu_pool:
        needs_now = {
            prec: _min_gpu_count_for_pool(
                model, gpu, state.mu, state.profiled_non_kv_gb, prec, max_avail
            )
            for prec in PRECISIONS
        }
        needs_pool = {
            prec: _min_gpu_count_for_pool(
                model, gpu, state.mu, state.profiled_non_kv_gb, prec, gpu_pool.count
            )
            for prec in PRECISIONS
        }
        selected_need = needs_now[am.prec]
        selected_pool_need = needs_pool[am.prec]
        fit_now = [
            prec
            for prec in PRECISIONS
            if prec != am.prec
            and am.gpu_count > 0
            and valid_strategies(
                model,
                am.gpu_count,
                gpu,
                state.mu,
                state.profiled_non_kv_gb,
                prec,
                resolve_spec_runtime(model, am.spec_method, am.spec_k, state.spec_acceptance, prec),
            )
        ]
        if fit_now:
            alt_prec = fit_now[0]
            alt_fits_now = True
            alt_need = needs_now[alt_prec]
            alt_pool_need = needs_pool[alt_prec]
        else:
            alt_prec, alt_pool_need = _best_precision_need(
                {prec: need for prec, need in needs_pool.items() if prec != am.prec}
            )
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
            embedding_stats = embedding_doc_stats(
                model, state.embedding_doc_dist, EMBEDDING_DOC_BUCKETS, am.prec
            )
        best_embedding = None
        for bs in _probe_batch_sizes(max(am.prefill_dp, 1)):
            # Distinct from the realtime `sample` bound earlier in this
            # function: two different result types, so two different names.
            embedding_sample = compute_embedding_distribution(
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
            if embedding_sample is None:
                continue
            if best_embedding is None or embedding_sample.rps > best_embedding.rps:
                best_embedding = embedding_sample
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
            doc_distribution.append(
                {
                    "label": bucket.label,
                    "length": bucket.length,
                    "clipped_length": clipped,
                    "share": share,
                    "color": bucket.color,
                }
            )
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
        "precision_options": precision_options,
        "precision_description": next(
            str(option["description"]) for option in precision_options if option["key"] == am.prec
        ),
        "realtime": realtime,
        "embedding": embedding,
        "spec": spec_info,
        "spec_options": model.available_spec_profiles,
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
        "precision_alerts": _precision_alerts(am.prec, gpu)
        + _quantization_profile_alerts(model, am.prec),
        "alerts": _comm_alerts(model, am.tp, am.pp, am.dp, gpu, avg_seq, state.decode_efficiency),
        "prefill_comm_summary": _comm_summary(am.prefill_tp, am.prefill_pp),
        "decode_comm_summary": _comm_summary(am.tp, am.pp),
        "prefill_alerts": _comm_alerts(
            model,
            am.prefill_tp,
            am.prefill_pp,
            am.prefill_dp,
            gpu,
            prefill_probe_len,
            state.prefill_efficiency,
        ),
        "decode_alerts": _comm_alerts(
            model, am.tp, am.pp, am.dp, gpu, avg_seq, state.decode_efficiency
        ),
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
