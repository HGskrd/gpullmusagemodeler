"""Machine placement view: how a model's TP/PP/DP layout lands on machines.

Mirrors the planner's own assumptions, it does not add new physics:
- a pool of N GPUs is ceil(N / node_size) machines of ``GPU.node_size`` GPUs;
- GPUs are filled copy by copy, stage by stage, TP rank by TP rank;
- inside a machine collectives use ``GPU.scale_up_collective_bw``; between
  machines ``GPU.inter_node_bw`` (pool override, default 200 Gb/s per GPU);
- MoE expert dispatch/combine traffic is not modelled anywhere.
"""

from __future__ import annotations

import math
from typing import Optional

from calc import (
    _pp_boundary_counts,
    communication_breakdown,
    compute_decode,
    compute_memory,
    resolve_spec_runtime,
)
from data import GPU
from placement import _preferred_strategy, _probe_batch_sizes
from presentation.formatting import layout_label
from state import GpuPool, ModelAssignment, PlannerState, pool_gpu

COPY_COLORS = [
    "#3266ad",
    "#1D9E75",
    "#BA7517",
    "#7F77DD",
    "#D85A30",
    "#A32D2D",
    "#5F8A8B",
    "#8C6D46",
]


def _scale_up_label(g: GPU) -> str:
    # Catalog has no interconnect name; PCIe-class peer bandwidth is <= 128 GB/s bidir.
    return "PCIe" if g.scale_up_p2p_bw_bidir <= 128e9 else "NVLink"


def _machines(tp: int, pp: int, dp: int, node_size: int, gpu_count: int) -> list[list[dict]]:
    slots = []
    for rep in range(dp):
        for stage in range(pp):
            for rank in range(tp):
                slots.append({"rep": rep, "stage": stage, "rank": rank})
    n_machines = max(1, math.ceil(gpu_count / node_size))
    machines: list[list[dict]] = [[] for _ in range(n_machines)]
    for i, slot in enumerate(slots):
        machines[min(i // node_size, n_machines - 1)].append(slot)
    for machine in machines:
        while len(machine) < node_size:
            machine.append({"rep": None, "stage": None, "rank": None})
    return machines


def _layout_row(
    state: PlannerState, am: ModelAssignment, g: GPU, tp: int, pp: int, dp: int
) -> dict:
    m = am.model
    spec = resolve_spec_runtime(m, am.spec_method, am.spec_k, state.spec_acceptance, am.prec)
    machines = _machines(tp, pp, dp, g.node_size, am.gpu_count)
    copy_machines: dict[int, set[int]] = {}
    first_stage_machine: dict[int, int] = {}
    for mi, machine in enumerate(machines):
        for slot in machine:
            if slot["rep"] is None:
                continue
            copy_machines.setdefault(slot["rep"], set()).add(mi)
            if slot["stage"] == 0 and slot["rank"] == 0:
                first_stage_machine[slot["rep"]] = mi
    span = max((len(v) for v in copy_machines.values()), default=1)
    # Machines that share a split copy with each machine (network actually used).
    partner_sets: list[set[int]] = [set() for _ in machines]
    for ms in copy_machines.values():
        for mi in ms:
            partner_sets[mi] |= ms - {mi}
    partners = [", ".join(f"M{p + 1}" for p in sorted(ps)) for ps in partner_sets]
    _, pp_cross = _pp_boundary_counts(tp, pp, g)
    tp_cross = tp > g.node_size
    host_per_copy = m.system_ram_weight_bytes
    host_by_machine = [0.0] * len(machines)
    for mi in first_stage_machine.values():
        host_by_machine[mi] += host_per_copy

    mem = compute_memory(
        m, tp, pp, g, state.mu, state.profiled_non_kv_gb, am.prec, state.decode_efficiency, spec
    )
    peak_tps, peak_bs, step_ms = 0, 0, 0.0
    for bs in _probe_batch_sizes(dp):
        r = compute_decode(
            m,
            tp,
            pp,
            bs,
            dp,
            g,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.in_dist,
            state.out_dist,
            state.decode_efficiency,
            spec,
        )
        if r and r.tps > peak_tps:
            peak_tps, peak_bs, step_ms = r.tps, bs, r.step_ms
    light = compute_decode(
        m,
        tp,
        pp,
        dp,
        dp,
        g,
        state.mu,
        state.profiled_non_kv_gb,
        am.prec,
        state.in_dist,
        state.out_dist,
        state.decode_efficiency,
        spec,
    )
    per_user = (light.tps / dp) if light else 0.0
    max_slots = light.max_slots if light else 0
    batch_tokens = max(1, min(32, math.ceil(32 / max(dp, 1))))
    comm = communication_breakdown(m, tp, pp, batch_tokens, 2000, g, state.decode_efficiency)
    comm_ms = (comm.dense_tp + comm.pp_boundary) * 1e3
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "label": layout_label(tp, pp, dp),
        "copy_gpus": tp * pp,
        "machines": machines,
        "span": span,
        "partners": partners,
        "copies_per_machine": (g.node_size // (tp * pp)) if tp * pp <= g.node_size else 0,
        "tp_cross": tp_cross,
        "pp_cross": pp_cross,
        "net_hops": pp_cross + (1 if tp_cross else 0),
        "ep_cross": bool(m.is_moe and span > 1),
        "kv_gb": (mem.kv_budget / 1e9) if mem else 0.0,
        "max_slots": max_slots,
        "peak_tps": peak_tps,
        "peak_bs": peak_bs,
        "step_ms": step_ms,
        "per_user": per_user,
        "comm_ms": comm_ms,
        "host_by_machine_gb": [b / 1e9 for b in host_by_machine],
        "host_max_gb": max(host_by_machine) / 1e9 if host_by_machine else 0.0,
    }


def placement_view(state: PlannerState, info: dict) -> Optional[dict]:
    am: ModelAssignment = info["am"]
    g: Optional[GPU] = info.get("gpu")
    if g is None or am.gpu_count <= 0 or not info.get("strats"):
        return None
    cur = _layout_row(state, am, g, am.tp, am.pp, am.dp)
    return {
        "cur": cur,
        "node_size": g.node_size,
        "n_machines": max(1, math.ceil(am.gpu_count / g.node_size)),
        "scale_up_label": _scale_up_label(g),
        "scale_up_gbs": g.scale_up_collective_bw / 1e9,
        "network_gbs": g.inter_node_bw / 1e9,
        "network_gbps": g.inter_node_bw * 8 / 1e9,
        "host_per_copy_gb": am.model.system_ram_weight_bytes / 1e9,
        "colors": COPY_COLORS,
    }


NETWORK_GBPS = (100.0, 200.0, 400.0, 800.0)
SHAPE_SIZES = (1, 2, 4, 8, 16)


def _best_on(state: PlannerState, am: ModelAssignment, g: GPU) -> Optional[dict]:
    if am.gpu_count <= 0:
        return None
    tp, pp, dp = _preferred_strategy(state, am, g, "decode")
    row = _layout_row(state, am, g, tp, pp, dp)
    return row if row["peak_tps"] > 0 else None


def machine_options(state: PlannerState, gp: GpuPool) -> dict:
    """Machine shapes for one pool and what each does to its deployed models.

    Each cell re-runs the planner's own layout pick (decode, <=128 users) on the
    candidate shape; the current cell also carries the deployed layout's own
    throughput, and every delta is relative to that. Expert dispatch/combine
    traffic is still not modelled.
    """
    base = pool_gpu(gp.gpu_type)
    cur = gp.gpu
    sizes = sorted({s for s in SHAPE_SIZES if s <= gp.count} | {base.node_size, cur.node_size})
    sizes = [s for s in sizes if s <= max(gp.count, cur.node_size)]
    networks = sorted(set(NETWORK_GBPS) | {cur.inter_node_bw * 8 / 1e9})
    models = [am for am in state.models if am.gpu_uid == gp.uid and am.gpu_count > 0]
    llms = [
        am
        for am in models
        if not (am.model.is_embedding_model or am.model.is_asr_model or am.model.is_decision_model)
    ]
    grid: list[dict] = []
    best = None
    current_cell = None
    for size in sizes:
        cells = []
        for net in networks:
            g = pool_gpu(gp.gpu_type, size, net)
            rows = [_best_on(state, am, g) for am in llms]
            ok = [r for r in rows if r]
            tps = sum(r["peak_tps"] for r in ok)
            cell = {
                "size": size,
                "net": net,
                "machines": max(1, math.ceil(gp.count / size)),
                "tps": tps,
                "per_user": min((r["per_user"] for r in ok), default=0.0),
                "span": max((r["span"] for r in ok), default=1),
                "layouts": " + ".join(r["label"] for r in ok),
                "host_max_gb": max((r["host_max_gb"] for r in ok), default=0.0),
                "is_current": size == cur.node_size and abs(net * 1e9 / 8 - cur.inter_node_bw) < 1,
                "is_default": size == base.node_size
                and abs(net * 1e9 / 8 - base.inter_node_bw) < 1,
            }
            if cell["is_current"]:
                # The deployment as configured now, which may be a layout the
                # user chose rather than the planner's pick for this shape.
                own = [_layout_row(state, am, g, am.tp, am.pp, am.dp) for am in llms]
                own_ok = [r for r in own if r["peak_tps"] > 0]
                cell["own_tps"] = sum(r["peak_tps"] for r in own_ok)
                cell["own_layouts"] = " + ".join(r["label"] for r in own_ok)
                cell["own_is_pick"] = cell["own_layouts"] == cell["layouts"]
                current_cell = cell
            cells.append(cell)
        grid.append({"size": size, "machines": max(1, math.ceil(gp.count / size)), "cells": cells})
    cells_all = [c for row in grid for c in row["cells"] if c["tps"] > 0]
    if cells_all:
        top = max(c["tps"] for c in cells_all)
        # Within 0.5% is a tie: keep the current shape, then the fewest network
        # crossings, then the catalog default, then fewer, larger machines.
        near = [c for c in cells_all if c["tps"] >= top * 0.995]
        best = min(
            near,
            key=lambda c: (
                not c["is_current"],
                c["span"],
                not c["is_default"],
                c["machines"],
                c["net"],
            ),
        )
    for row in grid:
        for cell in row["cells"]:
            cell["is_best"] = cell is best
            # Gains are measured against what is deployed now (own_tps).
            base_tps = current_cell.get("own_tps", 0.0) if current_cell else 0.0
            cell["delta"] = (cell["tps"] / base_tps - 1) if base_tps else 0.0
    return {
        "sizes": sizes,
        "networks": networks,
        "grid": grid,
        "best": best,
        "current": current_cell,
        "default_size": base.node_size,
        "default_net": base.inter_node_bw * 8 / 1e9,
        "cur_size": cur.node_size,
        "cur_net": cur.inter_node_bw * 8 / 1e9,
        "n_llms": len(llms),
        "has_models": bool(llms),
    }
