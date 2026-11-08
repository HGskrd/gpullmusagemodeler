"""Export catalog hardware/model pairs for an external benchmark-search agent.

No network requests, paid inference, benchmark execution, or catalog writes.
Run from the repository root: python -m benchmark_research --help
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import heapq
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from benchmark_feasibility import ResearchGrid, screen_pair
from data import GPUS, MODELS, OPEN_MODEL_ARCHITECTURE_SOURCES, artifact_options
from data.hardware_picker import hardware_choices

SCHEMA_VERSION = 2
TERMINAL_STATUSES = frozenset({"found", "not_found", "unsupported"})
RESEARCH_INSTRUCTIONS = """Research measured inference benchmarks for the exact hardware/model pair.
Search broadly first, then use artifact/repository names to disambiguate. The supplied
catalog values are unverified planner inputs, never benchmark evidence. Prefer original
benchmark reports, reproducible repositories, runtime maintainers and vendor measurements;
record independent measurements too, with provenance. Open the source; search snippets
alone are insufficient. Record URLs, access dates, publication dates, quoted evidence or
table/figure locations, and each measurement's exact configuration. Keep SXM/PCIe, memory
variants, model versions, base/instruct, artifacts, GPU counts, topology, runtime/version,
execution format and KV format distinct. Never substitute a related model or GPU silently.
Record adjacent evidence separately, identifying the mismatch. Do not infer runtime support
from memory fit, or infer incompatibility from missing search results. Do not download weights.
Record input/output lengths, batch/concurrency, prefix caching, speculative decoding, TP/PP/DP,
offload and latency/SLO constraints when given; otherwise use null. Separate per-user from
aggregate throughput, prefill from decode, TTFT from inter-token/end-to-end latency, and
device from cluster memory. Preserve raw metric names, values, units and scope. For speech
record audio duration, streaming mode, real-time factor or audio-seconds/second; embeddings
need sequence length and inputs/second; decisions need decisions/second. Tokens/second is
not interchangeable with any of these. FLOPS, theoretical rooflines, quality scores, and
planner estimates are not measured inference benchmarks. Comparability requires a matching
configuration, not merely a matching pair. Return found only with exact-pair measurements;
not_found means the bounded search found none, unsupported requires explicit source evidence,
and error is retryable. Record queries tried and search limitations. Never invent values or
automatically update planner coefficients from heterogeneous web results.
Plausible tasks contain concrete calculator-screened configurations and uncalibrated
estimates. Prioritize these configurations, but record the source's actual settings.
Passing screening does not verify runtime support or benchmark comparability. Exploratory
tasks failed only the stated finite grid: investigate the rejection without treating it
as proof of impossibility. Do not copy a planner estimate into observed metrics.
"""

RESULT_TEMPLATE: dict[str, Any] = {
    "task_id": "copy from task",
    "input_fingerprint": "copy from task",
    "status": "found | not_found | unsupported | error",
    "searched_at": "ISO-8601 UTC",
    "queries_tried": [],
    "observations": [
        {
            "match": "exact_pair | adjacent",
            "mismatches": [],
            "source_url": None,
            "source_kind": "vendor | runtime_maintainer | independent | other",
            "published_at": None,
            "accessed_at": None,
            "evidence_location": None,
            "hardware_name": None,
            "hardware_count": None,
            "topology": None,
            "model_repository": None,
            "model_revision": None,
            "artifact_id": None,
            "weight_format": None,
            "execution_format": None,
            "kv_format": None,
            "runtime": None,
            "runtime_version": None,
            "tensor_parallel": None,
            "pipeline_parallel": None,
            "data_parallel": None,
            "input_tokens": None,
            "output_tokens": None,
            "batch_size": None,
            "concurrency": None,
            "prefix_cache": None,
            "speculative_decoding": None,
            "offload": None,
            "latency_constraint": None,
            "audio_duration_seconds": None,
            "streaming": None,
            "metrics": [{"name": None, "value": None, "unit": None, "scope": None}],
        }
    ],
    "limitations": [],
}


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _iter_pairs(
    *,
    hardware_keys: list[str] | None = None,
    model_keys: list[str] | None = None,
    visible_only: bool = False,
    include_estimated_artifacts: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield every selected pair, including unlikely fits and unverified backends.

    Counts, workloads and runtimes are discovered within each pair's search rather
    than expanded into an unbounded Cartesian product or silently guessed.
    """
    choices = {choice.key: choice for choice in hardware_choices()}
    selected_hardware = sorted(set(GPUS if hardware_keys is None else hardware_keys))
    selected_models = sorted(set(MODELS if model_keys is None else model_keys))
    for label, keys, catalog in (
        ("hardware", selected_hardware, GPUS),
        ("model", selected_models, MODELS),
    ):
        unknown = set(keys) - catalog.keys()
        if unknown:
            raise ValueError(f"Unknown {label} keys: {', '.join(sorted(unknown))}")
    model_records = {}
    for key in selected_models:
        model = MODELS[key]
        if visible_only and model.hidden:
            continue
        source = OPEN_MODEL_ARCHITECTURE_SOURCES.get(key)
        kind = (
            "asr"
            if model.is_asr_model
            else "embedding"
            if model.is_embedding_model
            else "decision"
            if model.is_decision_model
            else "llm"
        )
        model_records[key] = {
            "key": key,
            "name": model.name,
            "kind": kind,
            "hidden": model.hidden,
            "streaming": model.is_streaming_asr if model.is_asr_model else None,
            "total_parameters": model.total_params,
            "active_parameters": model.active_params,
            "max_context_tokens": model.max_context_tokens,
            "architecture_note": model.architecture_note,
            "source": {
                "repository": source.repository,
                "revision": source.revision,
                "config_url": source.config_url,
                "note": source.note,
            }
            if source
            else None,
            "artifacts": [
                option
                for option in artifact_options(model)
                if include_estimated_artifacts or not option["key"].startswith("estimate:")
            ],
        }
    for key in selected_hardware:
        gpu = GPUS[key]
        choice = choices.get(key)
        if visible_only and choice is None:
            continue
        hardware = {
            "key": key,
            "name": gpu.name,
            "vendor": gpu.vendor_label,
            "picker_visible": choice is not None,
            "memory_bytes_per_device": gpu.mem,
            "peak_memory_bandwidth_bytes_per_second": gpu.bw,
            "planner_memory_bandwidth_bytes_per_second": gpu.effective_bw,
            "catalog_dense_flops_per_second": {
                "bf16": gpu.bf16,
                "fp8": gpu.fp8,
                "fp4": gpu.fp4,
            },
            "scale_up_p2p_bandwidth_bytes_per_second_bidirectional": gpu.scale_up_p2p_bw_bidir,
            "scale_up_domain_devices": gpu.node_size,
            "minimum_purchase_devices": gpu.min_count,
            "purchase_device_multiple": gpu.count_multiple,
            "specs_note": choice.specs if choice else "Non-picker compatibility/reference profile.",
            "runtime_note": choice.runtime if choice else "Runtime support requires verification.",
            "source_url": choice.source if choice else None,
            "reviewed_at": choice.reviewed_at if choice else None,
        }
        for model_key, model_record in model_records.items():
            task = {
                "schema_version": SCHEMA_VERSION,
                "task_id": f"benchmark-v2:{key}:{model_key}",
                "hardware": hardware,
                "model": model_record,
                "search_queries": [
                    f'"{gpu.name}" "{model_record["name"]}" inference benchmark',
                    f'"{gpu.name}" "{model_record["name"]}" throughput latency',
                ],
            }
            yield task


def iter_tasks(
    *,
    hardware_keys: list[str] | None = None,
    model_keys: list[str] | None = None,
    visible_only: bool = False,
    include_estimated_artifacts: bool = False,
    queue: str = "plausible",
    grid: ResearchGrid | None = None,
    exploratory_limit: int = 100,
    screening_counts: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Default to planner-feasible pairs; preserve exhaustive mode for audits."""
    if queue not in {"plausible", "exploratory", "all"}:
        raise ValueError(f"Unknown queue: {queue}")
    if exploratory_limit < 1:
        raise ValueError("Exploratory limit must be positive")
    grid = grid or ResearchGrid()
    assumptions = grid.assumptions()
    stats = screening_counts if screening_counts is not None else {}
    stats.update(candidate_pairs=0, plausible_pairs=0, rejected_pairs=0, feasible_configurations=0)

    def classified() -> Iterator[dict[str, Any]]:
        for task in _iter_pairs(
            hardware_keys=hardware_keys,
            model_keys=model_keys,
            visible_only=visible_only,
            include_estimated_artifacts=include_estimated_artifacts,
        ):
            stats["candidate_pairs"] += 1
            if queue != "all":
                screening = screen_pair(
                    task["hardware"]["key"],
                    task["model"]["key"],
                    task["model"]["artifacts"],
                    grid,
                )
                task["screening"] = screening
                plausible = bool(screening["configurations"])
                stats["plausible_pairs" if plausible else "rejected_pairs"] += 1
                stats["feasible_configurations"] += len(screening["configurations"])
                if plausible != (queue == "plausible"):
                    continue
            task["queue"] = queue
            task["input_fingerprint"] = fingerprint(
                [
                    task,
                    assumptions if queue != "all" else None,
                    RESEARCH_INSTRUCTIONS,
                    RESULT_TEMPLATE,
                ]
            )
            yield task

    if queue == "exploratory":
        # A fixed-size deterministic sample across the complete rejected set;
        # avoid spending the entire exploration budget on alphabetical neighbors.
        yield from heapq.nsmallest(
            exploratory_limit, classified(), key=lambda row: fingerprint(row["task_id"])
        )
    else:
        yield from classified()


def completed_tasks(path: Path | None) -> set[tuple[str, str]]:
    """Only skip terminal results for the same inputs; errors remain retryable."""
    completed: set[tuple[str, str]] = set()
    if path is None:
        return completed
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id, digest, status = (
                    row["task_id"],
                    row["input_fingerprint"],
                    row["status"],
                )
                if not all(isinstance(item, str) for item in (task_id, digest, status)):
                    raise ValueError("task_id, input_fingerprint and status must be strings")
                if status not in TERMINAL_STATUSES | {"error"}:
                    raise ValueError(f"Unknown status: {status}")
                if status in TERMINAL_STATUSES:
                    completed.add((task_id, digest))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: invalid completion record: {exc}") from exc
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL task file")
    parser.add_argument(
        "--hardware", action="append", help="Exact catalog key; repeat to select more"
    )
    parser.add_argument("--model", action="append", help="Exact catalog key; repeat to select more")
    parser.add_argument("--visible-only", action="store_true", help="Only current picker entries")
    parser.add_argument("--include-estimated-artifacts", action="store_true")
    parser.add_argument("--queue", choices=("plausible", "exploratory", "all"), default="plausible")
    parser.add_argument("--exploratory-limit", type=int, default=100)
    parser.add_argument("--gpu-count", type=int, action="append", help="Repeat; default 1,2,4,8")
    parser.add_argument("--batch-size", type=int, action="append", help="Repeat; default 1,16")
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument(
        "--execution-backend", action="append", choices=("auto", "native", "w4a16", "expanded")
    )
    parser.add_argument(
        "--runtime", default="generic", choices=("generic", "vllm", "sglang", "trtllm")
    )
    parser.add_argument("--runtime-version", default="")
    parser.add_argument("--memory-utilization", type=float, default=0.90)
    parser.add_argument("--non-kv-gb", type=float, default=4.0)
    parser.add_argument(
        "--completed", type=Path, help="Agent result JSONL to skip unchanged completions"
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index")
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if manifest_path.exists():
        parser.error("Output or manifest already exists; choose a new output path")
    try:
        grid = ResearchGrid(
            gpu_counts=tuple(sorted(set(args.gpu_count or (1, 2, 4, 8)))),
            batch_sizes=tuple(sorted(set(args.batch_size or (1, 16)))),
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            execution_backends=tuple(sorted(set(args.execution_backend or ("auto",)))),
            serving_runtime=args.runtime,
            runtime_version=args.runtime_version,
            memory_utilization=args.memory_utilization,
            non_kv_gb=args.non_kv_gb,
        )
        screening_counts: dict[str, int] = {}
        completed = completed_tasks(args.completed)
        # Validate keys before creating files, even for an empty selection.
        tasks = iter_tasks(
            hardware_keys=args.hardware,
            model_keys=args.model,
            visible_only=args.visible_only,
            include_estimated_artifacts=args.include_estimated_artifacts,
            queue=args.queue,
            grid=grid,
            exploratory_limit=args.exploratory_limit,
            screening_counts=screening_counts,
        )
        first = next(tasks, None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"selected_pairs": 0, "shard_pairs": 0, "skipped_completed": 0, "written": 0}
    dataset_hash = hashlib.sha256()

    def rows() -> Iterator[dict[str, Any]]:
        if first is not None:
            yield first
        yield from tasks

    with tempfile.TemporaryDirectory(
        prefix=".benchmark-export-", dir=args.output.parent
    ) as staging:
        staged_data = Path(staging) / args.output.name
        staged_manifest = Path(staging) / manifest_path.name
        with staged_data.open("x", encoding="utf-8") as output:
            for task in rows():
                counts["selected_pairs"] += 1
                dataset_hash.update(str(task["input_fingerprint"]).encode())
                shard = int(hashlib.sha256(task["task_id"].encode()).hexdigest(), 16)
                if shard % args.shard_count != args.shard_index:
                    continue
                counts["shard_pairs"] += 1
                if (task["task_id"], task["input_fingerprint"]) in completed:
                    counts["skipped_completed"] += 1
                    continue
                output.write(json.dumps(task, ensure_ascii=False, allow_nan=False) + "\n")
                counts["written"] += 1
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "catalog_values_are": "Unverified planner inputs, not measured benchmark evidence",
            "pair_scope": "GPUS × MODELS; separate retired records and hosted-only APIs are excluded",
            "configuration_scope": "One workload-feasible planner-sorted topology per artifact/backend/count/batch cell",
            "queue": args.queue,
            "screening_counts": screening_counts,
            "screening_assumptions": grid.assumptions() if args.queue != "all" else None,
            "selection": {
                "hardware": sorted(set(args.hardware)) if args.hardware else None,
                "models": sorted(set(args.model)) if args.model else None,
                "visible_only": args.visible_only,
                "include_estimated_artifacts": args.include_estimated_artifacts,
                "shard_count": args.shard_count,
                "shard_index": args.shard_index,
                "exploratory_limit": args.exploratory_limit
                if args.queue == "exploratory"
                else None,
            },
            "selected_inputs_sha256": dataset_hash.hexdigest(),
            "counts": counts,
            "agent_instructions": RESEARCH_INSTRUCTIONS,
            "result_template": RESULT_TEMPLATE,
        }
        with staged_manifest.open("x", encoding="utf-8") as output:
            output.write(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        published = []
        try:
            # Hard links publish complete files without overwriting another writer.
            for source, destination in (
                (staged_data, args.output),
                (staged_manifest, manifest_path),
            ):
                if destination == args.output and destination.exists():
                    # An uncatchable interruption may leave the complete data file
                    # without its completion manifest. Resume only identical bytes.
                    if not filecmp.cmp(source, destination, shallow=False):
                        raise ValueError(
                            "Output already exists with different content; choose a new path."
                        )
                    continue
                os.link(source, destination)
                published.append(destination)
        except BaseException:
            for destination in reversed(published):
                destination.unlink()
            raise
    print(json.dumps({**counts, "screening": screening_counts}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
