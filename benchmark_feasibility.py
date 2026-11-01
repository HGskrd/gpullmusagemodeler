"""Bounded benchmark candidates screened by the planner's own calculation paths."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from calc import (
    EfficiencyParams,
    compute_data,
    compute_decision,
    compute_embedding,
    compute_memory,
    compute_prefill,
    compute_realtime_capacity,
    context_supported,
    resolve_execution,
    valid_strategies,
)
from data import GPU, GPUS, Model, configured_model, normalize_gpu_count
from data.artifacts import validate_runtime


@dataclass(frozen=True)
class ResearchGrid:
    gpu_counts: tuple[int, ...] = (1, 2, 4, 8)
    batch_sizes: tuple[int, ...] = (1, 16)
    input_tokens: int = 2048
    output_tokens: int = 128
    execution_backends: tuple[str, ...] = ("auto",)
    serving_runtime: str = "generic"
    runtime_version: str = ""
    memory_utilization: float = 0.90
    non_kv_gb: float = 4.0

    def __post_init__(self) -> None:
        if not self.gpu_counts or min(self.gpu_counts) < 1:
            raise ValueError("GPU counts must be positive")
        if not self.batch_sizes or min(self.batch_sizes) < 1:
            raise ValueError("Batch sizes must be positive")
        if self.input_tokens < 1 or self.output_tokens < 1:
            raise ValueError("Input and output tokens must be positive")
        if (
            not 0 < self.memory_utilization <= 1
            or not isfinite(self.non_kv_gb)
            or self.non_kv_gb < 0
        ):
            raise ValueError("Require 0 < memory-utilization <= 1 and non-kv-gb >= 0")
        if not self.execution_backends or any(
            backend not in {"auto", "native", "w4a16", "expanded"}
            for backend in self.execution_backends
        ):
            raise ValueError("Invalid execution backend for screening")
        validate_runtime(self.serving_runtime, self.runtime_version)

    def assumptions(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "efficiency": asdict(EfficiencyParams()),
            "kv_precision": "auto",
            "recurrent_precision": "auto",
            "prefix_hit_rate": 0,
            "speculative_decoding": "off",
            "extra_offload": "none; catalog-native host tables are preserved",
            "topology_selection": "First workload-feasible planner-sorted layout per grid cell; DP <= batch",
            "gpu_count_policy": "Normalize each requested count using planner purchase constraints",
            "context_policy": "Exact requested length; no silent truncation",
            "runtime_support": "Explicit unsupported paths rejected; estimated paths need web verification",
        }


def _workload(m: Model, batch: int, grid: ResearchGrid) -> dict[str, Any]:
    if m.is_asr_model:
        profile = m.realtime_profile
        assert profile is not None
        return {
            "kind": "asr",
            "concurrent_audio_streams_or_chunks": batch,
            "streaming": profile.streaming,
            "audio_chunk_seconds": profile.audio_chunk_seconds or None,
            "state_tokens": profile.state_tokens,
            "required_positions_per_audio_second": profile.tokens_per_second,
        }
    return {
        "kind": "embedding"
        if m.is_embedding_model
        else "decision"
        if m.is_decision_model
        else "llm",
        "batch_size": batch,
        "input_tokens": grid.input_tokens,
        "output_tokens": 0 if m.is_embedding_model or m.is_decision_model else grid.output_tokens,
        **({"questions_per_request": 1, "options_per_question": 4} if m.is_decision_model else {}),
    }


def _context_fits(m: Model, grid: ResearchGrid) -> bool:
    if m.is_asr_model:
        return True  # Audio state/chunk lengths are defined by its workload profile.
    if m.embedding_profile:
        return grid.input_tokens <= m.embedding_profile.max_sequence_length
    if m.decision_profile:
        return grid.input_tokens <= m.decision_profile.max_sequence_length
    return context_supported(m, grid.input_tokens, grid.output_tokens)


def _estimate(
    m: Model, g: GPU, precision: str, strat: tuple[int, int, int], batch: int, grid: ResearchGrid
) -> dict[str, float] | None:
    eff = EfficiencyParams()
    mu, non_kv = grid.memory_utilization, grid.non_kv_gb
    if m.is_asr_model:
        speech = compute_realtime_capacity(m, strat, batch, g, mu, non_kv, precision, eff)
        if speech is None:
            return None
        if m.is_streaming_asr and speech.realtime_factor < 1:
            return None  # A streaming candidate must keep up with incoming audio.
        return {
            "aggregate_audio_seconds_per_second": batch * speech.realtime_factor,
            "per_stream_capacity_ratio_to_realtime": speech.realtime_factor,
        }
    if m.is_embedding_model:
        embedding = compute_embedding(
            m, strat, batch, grid.input_tokens, g, mu, non_kv, precision, eff
        )
        if embedding is None:
            return None
        return {"inputs_per_second": embedding.rps, "batch_service_seconds": embedding.service_time}
    if m.is_decision_model:
        decision = compute_decision(
            m, strat, batch, grid.input_tokens, g, mu, non_kv, precision, eff
        )
        if decision is None:
            return None
        return {
            "decisions_per_second": decision.decisions_per_second,
            "batch_service_seconds": decision.service_time,
        }
    tp, pp, dp = strat
    # compute_data uses average decode residency. Additionally require the final
    # context to fit so the candidate can finish generating, not merely start.
    peak_fit = compute_prefill(
        m,
        tp,
        pp,
        batch,
        dp,
        grid.input_tokens,
        g,
        mu,
        non_kv,
        precision,
        eff,
        kv_residency_seq_len=grid.input_tokens + grid.output_tokens,
    )
    if peak_fit is None:
        return None
    text = compute_data(
        m,
        strat,
        strat,
        batch,
        grid.input_tokens,
        grid.output_tokens,
        g,
        mu,
        non_kv,
        precision,
        0.0,
        eff,
        eff,
    )
    if text is None:
        return None
    return {
        "requests_per_second": text.rps,
        "aggregate_input_plus_output_tokens_per_second": text.tps,
    }


def screen_pair(
    hardware_key: str, model_key: str, artifacts: list[dict[str, Any]], grid: ResearchGrid
) -> dict[str, Any]:
    """One representative topology per feasible artifact/backend/count/batch cell.

    All topology alternatives are tried until one passes. Rejection tallies refer
    to grid cells, not a claim that the pair can never run outside these bounds.
    """
    gpu = GPUS[hardware_key]
    counts = sorted({normalize_gpu_count(hardware_key, count) for count in grid.gpu_counts})
    batches = sorted(set(grid.batch_sizes))
    rejections: Counter[str] = Counter()
    configurations = []
    for artifact in artifacts:
        precision = artifact["precision"]
        for backend in dict.fromkeys(grid.execution_backends):
            model = configured_model(
                model_key,
                artifact["key"],
                backend,
                "auto",
                "auto",
                grid.serving_runtime,
                grid.runtime_version,
            )
            execution = resolve_execution(gpu, model, precision)
            cells = len(counts) * len(batches)
            if execution.status == "unsupported":
                rejections["execution_unsupported"] += cells
                continue
            if not _context_fits(model, grid):
                rejections["context_exceeded"] += cells
                continue
            for count in counts:
                strategies = valid_strategies(
                    model, count, gpu, grid.memory_utilization, grid.non_kv_gb, precision
                )
                if not strategies:
                    rejections["no_weight_or_topology_fit"] += len(batches)
                    continue
                for batch in batches:
                    for strat in strategies:
                        tp, pp, dp = strat
                        if dp > batch:
                            continue
                        estimate = _estimate(model, gpu, precision, strat, batch, grid)
                        if estimate is None:
                            continue
                        memory = compute_memory(
                            model,
                            tp,
                            pp,
                            gpu,
                            grid.memory_utilization,
                            grid.non_kv_gb,
                            precision,
                            EfficiencyParams(),
                        )
                        assert memory is not None
                        configurations.append(
                            {
                                "artifact_id": artifact["key"],
                                "precision": precision,
                                "gpu_count": count,
                                "tp": tp,
                                "pp": pp,
                                "dp": dp,
                                "crosses_scale_up_domain": tp > gpu.node_size
                                or tp * pp > gpu.node_size,
                                "execution_backend": backend,
                                "execution_path": execution.path,
                                "runtime_evidence": execution.status,
                                "runtime_note": execution.note,
                                "workload": _workload(model, batch, grid),
                                "estimated_gpu_weights_bytes_per_rank": memory.weights,
                                "estimated_kv_or_work_buffer_budget_bytes_per_rank": memory.kv_budget,
                                "planner_estimate": estimate,
                                "catalog_host_table_bytes_per_copy": model.system_ram_weight_bytes,
                            }
                        )
                        break
                    else:
                        rejections["workload_memory_or_capacity"] += 1
    return {
        "classification": "plausible" if configurations else "rejected_within_grid",
        "effective_gpu_counts": counts,
        "rejected_grid_cells": dict(sorted(rejections.items())),
        "configurations": configurations,
        "runtime_support_verified": False,
    }
