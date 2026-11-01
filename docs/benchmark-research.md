# Web benchmark research queue

Generate a queue screened by the calculator, using the project environment from
the repository root:

```bash
.venv/bin/python -m benchmark_research --output instance/benchmark-research/plausible-v2.jsonl
```

This is an offline export. It makes no search or paid LLM calls and does not change
the planner. Give the agent the accompanying `*.jsonl.manifest.json` once as
shared context, then one task JSON object per job. The manifest includes research
instructions and a result template. Have the agent append one result JSON object
per line to a separate `results.jsonl`, copying `task_id` and `input_fingerprint`.
The template's observation is a shape example: use an empty observations list when
there is no evidence, and as many observations as the sources support.

The default `--queue plausible` evaluates a bounded grid for every selected pair:

- Native artifacts and pinned conversions; `--include-estimated-artifacts` also
  tests generic estimated conversions.
- 1, 2, 4, and 8 GPUs, normalized with the planner's purchase constraints. A
  system-only profile can therefore require more than eight devices. Repeat
  `--gpu-count` to replace the count list.
- Batches of 1 and 16; repeat `--batch-size` to replace the batch list.
- 2,048 input tokens and 128 output tokens for text. `--input-tokens` and
  `--output-tokens` override these. Embeddings and decisions use zero output tokens;
  decisions assume one question with four options. Inputs beyond their sequence
  limits are rejected, never silently truncated.
- Speech uses each catalog profile's audio chunk/state and streaming mode, with
  the batch values interpreted as concurrent streams/chunks. Streaming candidates
  must sustain real time according to the estimate.
- Auto execution, auto KV/recurrent formats, generic runtime, 90% device memory,
  and 4 GB non-KV runtime reservation using the planner's `EfficiencyParams`
  defaults. Repeat `--execution-backend` for auto/native/W4A16/expanded paths;
  `--runtime`, `--runtime-version`, `--memory-utilization`, and `--non-kv-gb` override
  the corresponding assumptions. Prefix caching and speculation are off. Only
  catalog-native host table offload is included.

`configured_model`, `resolve_execution`, `valid_strategies`, and the modality's
actual calculation function screen candidates. Within each artifact/backend/count/
batch cell, the first workload-feasible planner-sorted TP/PP/DP layout is retained;
all alternatives are tried if earlier layouts fail. Idle DP replicas are excluded.
Text additionally checks end-of-generation KV residency, rather than relying on
average decode context alone. This exports representative feasible topologies,
not every equivalent layout.

A pair gets one job only if at least one cell passes. The job's `screening` object
contains its concrete configurations, estimated memory and modality-specific
throughput values, and rejected-cell reason counts. The manifest records the full
grid and candidate/plausible/rejected/configuration totals. Cells stay grouped so
the agent can reuse sources across related configurations.

These are **calculator-plausible**, not verified runtime configurations. Explicitly
unsupported execution paths are rejected; estimated paths still require backend/
version/kernel evidence. The calculator does not establish kernel support for every
vendor, host RAM fit, multimodal encoder execution, loader workspace, or latency
SLOs. Cross-domain layouts retain the planner's interconnect assumptions. A
rejection means no fit within this grid, not that the pair can never run.

Keep a small independent exploratory queue to check those assumptions:

```bash
.venv/bin/python -m benchmark_research --queue exploratory --exploratory-limit 100 \
  --output instance/benchmark-research/exploratory-v2.jsonl
```

This deterministically samples at most 100 rejected pairs across the selected
catalog, with their rejection reasons. It is disjoint from the plausible queue
for the same grid. Sampling happens before sharding, so the limit applies globally.
Use `--queue all` to recover the exhaustive, unscreened Cartesian product when
needed; it does not run the feasibility checks.

By default this includes hidden and compatibility profiles in the live catalogs.
Retired records outside `MODELS` and hosted-only API offerings are outside this
hardware audit. Add `--visible-only` for current picker entries. Use repeated
`--hardware` and `--model` flags for exact catalog keys:

```bash
.venv/bin/python -m benchmark_research \
  --hardware H100 --hardware RTX4090 --model qwen38-27b \
  --output instance/benchmark-research/pilot.jsonl
```

Start with a small pilot to establish search quality and actual cost before paying
for the full catalog. Bound searches/time/tokens in the consumer. A pair with no
published evidence is a valid outcome, not a reason for endless searches. Results
should use `found`, `not_found`, `unsupported`, or retryable `error` status. Only
exact-pair measured evidence qualifies for `found`; preserve nearby configurations
as adjacent evidence and retain missing configuration fields as null.

Resume to a new output file; existing outputs are never overwritten:

```bash
.venv/bin/python -m benchmark_research \
  --completed instance/benchmark-research/results.jsonl \
  --output instance/benchmark-research/remaining.jsonl
```

Only terminal results with matching IDs **and** input fingerprints are skipped.
Catalog inputs, screening grid, or evidence-instruction changes trigger research again; errors are
retried. Completion loading checks record identity and status, not scientific
validity: review the agent's evidence before accepting results. To refresh a stale
`not_found` result, remove it from the completion input or omit `--completed`.

For independent workers use `--shard-count N --shard-index I` (zero-based index),
with a different output filename per shard. Hash-based shard assignment is stable
when unrelated catalog entries are added; fixed-N shards are disjoint and cover all
selected pairs. Each manifest reports selected pairs, shard pairs, skipped results,
written jobs, and a fingerprint of the selected inputs.

The exported values are current planner assumptions, **not benchmark measurements**.
Memory, bandwidth and FLOPS have explicit units and per-device scope. A source's
tokens/second cannot validate a planner estimate without matching model artifact,
hardware variant/count/topology, runtime/version, workload, concurrency, cache,
speculation and offload. Speech, embeddings and decisions have separate units.
Schema version 2 uses new task IDs so previous unscreened completions cannot
silently suppress the new configuration research. This tool prepares evidence
collection; it does not fit coefficients or change
estimator math. See [calibration-benchmark.md](calibration-benchmark.md) for controlled
measurement and calibration design.
