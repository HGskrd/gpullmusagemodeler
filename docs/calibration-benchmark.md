# Calibration benchmark specification

Status: proposal, written 2026-10-20 against commit `3820dd6`. Line numbers
below refer to that commit and will drift; the function names are the stable
reference. Nothing here has been measured yet. Every planner number today
comes from the hand-set priors in `state.py:401-424`, so it is a closed-form
estimate, not a benchmark result (AGENTS.md, "Catalog and planner correctness").

This document specifies:

1. what each efficiency knob does in the planner math, and which measurement
   identifies it;
2. the smallest experiment matrix that identifies every knob;
3. what to record for each run;
4. the record format the benchmark must emit;
5. how the backend should turn records into fitted knob values, and how to
   accept or reject a fit;
6. measurement pitfalls.

Statements about vLLM flags, log lines and metric names are written from
general knowledge of vLLM V1 (0.8-0.11 era), not from this repository. They
are marked **[check version]** and must be confirmed against the exact runtime
version under test before the harness is written.

---

## 1. Knobs, where they act, and what identifies them

### 1.1 Where the knobs live

- Defaults: `PlannerState` fields, `state.py:401-424`.
- They are packed into two `EfficiencyParams` objects (`calc.py:113-123`) by
  `PlannerState.prefill_efficiency` (`state.py:504`) and
  `PlannerState.decode_efficiency` (`state.py:517`). `kv_slack`,
  `moe_imbalance` and `pd_interference` go into **both** objects;
  `sched_budget` goes into the decode object only.
- UI bounds: `EFFICIENCY_SETTING_BOUNDS` and `INTEGER_SETTING_BOUNDS`
  (`web/config.py:95-112`), enforced in `web/planner.py:425-450`. Scenario
  import uses its own, wider bounds (`scenarios.py:474-496`). Fits must use the
  `web/config.py` bounds, because the UI cannot represent a value outside them.
- Scope: the knobs are **global per planner state**. Every assignment, model
  and GPU pool in a scenario shares one `EfficiencyParams` pair (see the
  callers in `placement.py:128-199`, `engine/economics.py:317`,
  `presentation/charts.py:123-319`). Per-(model, GPU) fitted values therefore
  need a code change before the planner can use them. See section 5.6.

### 1.2 The planner equations the fit is compared against

Decode step time, one replica, `pr` running sequences
(`_decode_step_time`, `calc.py:1475-1583`; spec terms omitted):

```
mb      = ceil(pr / pp)
wt      = (W_active * f_pp / tp) / (BW * bw_eff)                      # calc.py:1500
kv_time = mb * KVread(avg_seq) / (BW * bw_eff)                         # calc.py:1515
ct      = compute_seconds(2*P_active*mb*f_pp, attn) / (tp * comp_eff)  # calc.py:1528
stage   = (max(wt + kv_time, ct) + comm_tp) * pp + comm_pp             # calc.py:1547
step    = stage * (1 + overhead + paged_oh_eff) * moe_mult             # calc.py:1582-1583
```

- `BW` is `GPU.effective_bw` (`data/hardware.py:49`): `planner_bw` if set,
  else the published `bw`.
- `avg_seq = avg_in + avg_out/2` (`calc.py:1638`), the mean resident context
  of a sequence over its lifetime.
- `comm_tp` is `_dense_tp_oh` (`calc.py:1413-1426`): two ring all-reduces per
  layer at `scale_up_collective_bw * bw_eff`, plus 3 µs latency each, scaled
  by `(1 - ar_overlap)`. The memory-bandwidth efficiency is also applied to the
  interconnect bandwidth. That is a modelling simplification, and it couples
  `bw_eff` and `ar_overlap`.
- `moe_mult = moe_imbalance` for MoE models, else 1 (`_moe_tail_multiplier`,
  `calc.py:1467`).
- Decode throughput is `pr * tau / step` and per-token latency is
  `step / tau` (`calc.py:1659-1664`). The returned `DecodeResult.lat` and
  `step_ms` are rounded to 0.01 ms, and `tps` to an integer
  (`calc.py:1671-1678`). The fitter must call the unrounded internals.

Prefill service time for `pr` prompts of `seq_len` uncached tokens
(`compute_prefill`, `calc.py:1871-1957`):

```
ct   = compute_seconds(ffn, attn, "prefill") / (tp * comp_eff)            # calc.py:1938
mt   = (W_active_prefill * f_pp / tp) / (BW * bw_eff)                     # calc.py:1945
base = max(ct, mt) * (1 + (pp-1)/pr)                                       # calc.py:1948
t    = (base + comm_total) * (1 + 1.3*overhead + fixed_paged_oh(L, 0.35))  # calc.py:1949
t   *= moe_mult
```

`t` is the makespan of the whole prompt batch. The planner reports it as TTFT,
which is the last user's first token, a p100-style number
(`calc.py:2431-2434`). Prefill has no chunk-size term: vLLM's
`--max-num-batched-tokens` chunking is not modelled.

Memory (`compute_memory`, `calc.py:1274-1307`):

```
requested   = GPU.mem * mu                                     # GPU.mem is decimal bytes, e.g. 80e9
non_kv      = weights_per_gpu + profiled_non_kv_gb*1e9*(1 + 0.12*log2(tp))   # calc.py:1207-1212
kv_reserved = requested - non_kv
kv_budget   = kv_reserved * (1 - kv_slack)                     # calc.py:1299
max_slots   = min(floor(kv_budget / KV(avg_seq)), sched_budget)  # calc.py:1644-1646
```

Mixed prefill and decode (`compute_data`, `calc.py:2235-2307`;
`compute_user_experience`, `calc.py:2372-2447`):

```
total = max(t_pf, t_dec) + (t_pf + t_dec - max(t_pf, t_dec)) * pd_interference
rps   = bs / total
```

`pd_interference = 0` means prefill fully overlaps decode; `1` means they run
serially. Note that `compute_data` uses `fixed_paged_oh(in+out/2)`
(`calc.py:2292`), while `compute_decode` uses the distribution-aware
`decode_paged_oh` (`calc.py:1183-1191`). The same fixed-length workload
therefore gets slightly different paged overhead on the two paths. The fitter
must use the path that matches the view being calibrated.

Prefix reuse: `effective_prefill_length = ceil(L * (1 - hit_rate))`
(`calc.py:1199-1204`). Compute is reduced; KV residency is still charged at
the full prompt length (`calc.py:1897-1902`).

Speculative decoding: `tau = (1 - α^(k+1)) / (1 - α)` (`spec_acceptance_len`,
`calc.py:153`), corrected for finite outputs by `spec_finite_output_tau`
(`calc.py:167`). A non-zero global `spec_acceptance` overrides the profile's
`acceptance_alpha` and `acceptance_alpha_by_k` (`calc.py:212-214`).

### 1.3 Knob table

| Knob (state field) | Default | UI bound | Applied at | Physical meaning | Identifying measurement |
|---|---|---|---|---|---|
| `mu` | 0.90 | n/a (scenario 0.01-1.0) | `calc.py:1287` | vLLM `--gpu-memory-utilization` | **Set, not fit.** Copy the configured flag value. |
| `profiled_non_kv_gb` | 4.0 | n/a (scenario 0-4096) | `calc.py:1207-1212`, `1294` | Activation peak, CUDA graphs, NCCL buffers, allocator and non-torch memory, per GPU at TP=1 | vLLM startup log: available KV memory vs. `mu * total` minus weights (section 1.4.1). |
| `kv_slack` | 0.02 | 0.00-0.10 | `calc.py:1299` | KV that is reserved but unusable (block rounding, fragmentation, watermark) | Max concurrency reached without preemption vs. reported KV token capacity (1.4.1). |
| `decode_sched_budget` | 16384 | 2048-65536 (int) | `calc.py:1645-1646` | Cap on running sequences per replica | **Set, not fit.** Maps to `--max-num-seqs` [check version]. See open question Q1. |
| `decode_bw_eff` | 0.80 | 0.50-1.00 | `calc.py:1500`, `1515`, `1418` | Achieved fraction of `effective_bw` in decode GEMV, KV reads, and (by coupling) collectives | Low-concurrency decode, short context, TP=1, dense (1.4.2). |
| `decode_comp_eff` | 0.75 | 0.50-1.00 | `calc.py:1528-1530` | Achieved fraction of tensor peak in decode | High-concurrency decode past the roofline knee (1.4.2). |
| `decode_overhead` | 0.08 | 0.00-0.25 | `calc.py:1582` | Proportional non-GPU-work tax per step: launch, scheduling, sampling, gaps | GPU busy fraction from a profiler trace; otherwise held at the prior (1.4.2, gauge). |
| `decode_paged_oh` | 0.10 | 0.00-0.25 | `calc.py:1183-1191`, `1582` | Paged-attention indirection cost, scaled by `_paged_kv_pressure` (0.45-2.0) | Residual vs. context length and length heterogeneity at fixed batch (1.4.2). |
| `decode_ar_overlap` | 0.30 | 0.00-0.80 | `calc.py:1426` | Fraction of TP all-reduce hidden under compute | TP>1 decode minus the TP=1-fitted prediction (1.4.4). |
| `prefill_bw_eff` | 0.80 | 0.50-1.00 | `calc.py:1945` | Weight-streaming efficiency in prefill | Weight-bound prefill: 1 request, ≤256 tokens (1.4.3). |
| `prefill_comp_eff` | 0.75 | 0.50-1.00 | `calc.py:1938` | Achieved fraction of tensor peak in prefill | Long-prompt prefill, 4k-16k tokens (1.4.3). |
| `prefill_overhead` | 0.08 | 0.00-0.25 | `calc.py:1949` (×1.3) | Proportional prefill tax | Profiler trace; otherwise prior (gauge). |
| `prefill_paged_oh` | 0.10 | 0.00-0.25 | `calc.py:1194-1196`, `1949` (×0.35) | Paged-KV write overhead in prefill | Weakly identified; fit only if residuals trend with L (1.4.3). |
| `prefill_ar_overlap` | 0.30 | 0.00-0.80 | `calc.py:1426` via `communication_breakdown` | All-reduce overlap in prefill | TP>1 long-prompt prefill (1.4.4). |
| `moe_imbalance` | 1.15 | 1.00-2.00 | `calc.py:1467`, `1583`, `1950` | Expert load-imbalance tail multiplier | MoE runs after the dense knobs are fixed on the same GPU, precision and runtime (1.4.5). |
| `pd_interference` | 0.0 | 0.00-1.00 | `calc.py:2299-2301`, `2437-2439` | Serialisation fraction between prefill and decode | Mixed open-loop run vs. isolated prefill and decode times (1.4.6). |
| `spec_acceptance` | 0.0 (use profile) | 0.00-0.99 | `calc.py:214` | Per-token acceptance α of the chain model | vLLM spec-decode acceptance counters (1.4.7). |
| `prefix_hit_rate` (per use case, `data/use_cases.py:523`) | 0.10-0.50 | 0-1 | `calc.py:1199` | Fraction of prompt tokens served from the prefix cache | `prefix_cache_hits / prefix_cache_queries` from production-like traffic (1.4.8). |

### 1.4 Identification, knob by knob

#### 1.4.1 Memory: `profiled_non_kv_gb`, `kv_slack`

Gate check before any fitting (a catalog check, not a knob): compare the
planner's KV bytes per token, `MemoryResult.kv_per_token` (`calc.py:1306`),
with vLLM's figure, `available_kv_bytes / kv_capacity_tokens`. vLLM logs both
at startup ("Available KV cache memory: X GiB" and "GPU KV cache size: N
tokens") [check version]. If they disagree by more than 2%, the model geometry,
KV dtype or artifact is wrong in the catalog. Fix that first, because no knob
should absorb a geometry error. Do the same for weights: compare the planner's
`weights` (`calc.py:1289`) with "Model loading took X GiB" [check version].

`profiled_non_kv_gb`. vLLM computes
`kv_reserved_measured = mu * total_device_memory - (weights + peak_activation + non_torch + cudagraph)`.
Solve the planner's equation for the knob:

```
profiled_non_kv_gb = (mu * GPU.mem - weights_planner - kv_reserved_measured) / 1e9 / (1 + 0.12*log2(tp))
```

Two unit traps:

- vLLM logs GiB (2^30). Convert to bytes.
- `GPU.mem` is the catalog's decimal nominal capacity (H100: `80e9`,
  `data/hardware.py:92`). The physical device reports its own total
  (`torch.cuda.mem_get_info()[1]`, roughly 85.0e9 bytes on an H100 80GB
  [check on hardware]).

Because the planner multiplies `mu` by the catalog number, the fitted
`profiled_non_kv_gb` absorbs the difference between nominal and physical
capacity, and can come out negative. Record both totals. Store the fit
together with the `GPU.mem` value it was fit against (section 5.4). This must
hold: TP=1 and TP>1 fits agree within 10% once the `1 + 0.12*log2(tp)` scaling
is applied. If they do not, the TP scaling term is wrong and needs a separate
look.

`kv_slack`. The planner fills `kv_budget / KV(avg_seq)` sequences. vLLM
allocates whole blocks: each sequence uses `ceil(len / block_size)` blocks,
and the scheduler preempts when blocks run out. Run fixed-length requests
(in = 4096, out = 512, `--ignore-eos`) at increasing concurrency until
`vllm:num_preemptions_total` starts increasing or `vllm:num_requests_waiting`
stays above zero. Let `C_max` be the largest concurrency with zero preemptions
and `vllm:num_requests_running == C`. Then:

```
kv_slack = 1 - (C_max * KV_planner(in + out)) / kv_reserved_measured
```

Peak residency (in + out) is used here, not `avg_seq`. A synchronized cohort
peaks at in + out, and the planner's slot count is an average-occupancy figure
(`calc.py:1632-1637`). Report both `C_max` and the planner's
`max_slots(avg_seq)`. The difference is a modelling choice, not slack. It is
inference that block rounding dominates the slack: with block_size 16 and
4608-token sequences, rounding alone is under 0.4%.

#### 1.4.2 Decode: `decode_bw_eff`, `decode_comp_eff`, `decode_overhead`, `decode_paged_oh`

Isolation method, the synchronized cohort:

1. Submit `C` requests at once (in = L_in, out = L_out, `--ignore-eos`,
   temperature 0).
2. Measure only the window after the last request has received its first
   token, up to the first request finishing.

No prefill runs inside that window, so `pd_interference` does not apply.
Context grows linearly from L_in to L_in + L_out. The planner's step time is
linear in context (KV read term), so the window-mean ITL equals the planner's
step at `avg_seq = L_in + L_out/2`. This is exact for the linear model.

- Weight-dominated regime (C ≤ 8, L_in = 256, TP=1, dense model):
  `step ≈ wt * (1 + overhead + paged_oh_eff)`. This identifies
  `bw_eff / (1 + overhead + paged_oh_eff)`.
- KV-read regime (C = 32-128, L_in = 4096-16384): the slope of step vs. `C·KV`
  checks the same `bw_eff` on KV traffic.
- Compute regime (C ≥ the planner's roofline knee, where `ct > wt + kv_time`):
  identifies `comp_eff / (1 + overhead + paged_oh_eff)`. Planner inference for
  BF16 8B on H100: the knee sits near C ≈ 200-300 at short context, so it may
  be unreachable before KV or `--max-num-seqs` limits. In that case
  `decode_comp_eff` stays at the prior and is marked "not identified".

**Gauge problem.** Scaling `bw_eff` and `comp_eff` by the same factor `s`
while setting `(1 + overhead + p) → s·(1 + overhead + p)` leaves every
decode prediction unchanged. The only exceptions are the fixed spec
constants (`calc.py:105-107`) and the 3 µs collective latency. Throughput and
latency data alone therefore **cannot** separate `overhead` from the two
efficiencies. Resolve it in one of two ways:

- **Preferred:** measure `overhead` directly. Take an Nsight Systems or
  `torch.profiler` trace of 50 decode steps and compute
  `overhead = wall_step / sum(GPU kernel time on the critical stream) - 1`.
  Then fit `bw_eff` and `comp_eff` with `overhead` fixed.
- **Fallback:** hold `overhead` at the 0.08 prior, fit the efficiencies, and
  label the result "efficiencies conditional on overhead = 0.08".

Batch-1 latency is still recorded. Regress measured step against predicted
step across C. A significant positive intercept means a fixed per-step cost,
which the planner does not model: it only has proportional overhead. Report
that intercept as a structural gap instead of folding it into `overhead`.

`decode_paged_oh`: its effective value is `paged_oh * _paged_kv_pressure(...)`
(`calc.py:1176-1181`). The pressure term varies with mean context and length
heterogeneity between 0.45 and 2.0. With the efficiencies fixed, fit
`paged_oh` from the residual trend across L_in ∈ {256, 4096, 16384}, plus one
mixed-length run (section 2, S-dec-mix). If the trend is not significant,
keep the prior.

#### 1.4.3 Prefill: `prefill_bw_eff`, `prefill_comp_eff`, `prefill_overhead`, `prefill_paged_oh`

Isolation: submit N requests at once with `max_tokens = 1`, unique prompts,
and prefix caching off. Measured service time = the last request's TTFT
(server-side, see section 3.2), minus queue time before the first scheduled
step. Set `--max-num-batched-tokens ≥ N·L` where memory allows, so the batch is
not chunked. Otherwise record the chunk size and treat the run as
validation-only.

- `prefill_bw_eff`: N = 1, L ∈ {32, 128, 256}. Here `mt > ct` (planner
  inference for BF16 8B on H100: mt ≈ 6 ms, ct ≈ 1.4 ms at 64 tokens).
- `prefill_comp_eff`: N ∈ {1, 4}, L ∈ {4096, 16384}. Here `ct` dominates.
  Attention FLOPs are charged at `g.bf16` (`calc.py:1131-1132`), so long-L
  points also test the attention-rate assumption.
- The overhead gauge is the same as in decode, with factor 1.3
  (`calc.py:1949`). Use a profiler trace or the prior.
- `prefill_paged_oh` enters at ×0.35 times pressure. Fit it only when residuals
  trend with L after the efficiencies are fixed. Expect "not identified".

#### 1.4.4 Tensor parallel: `decode_ar_overlap`, `prefill_ar_overlap`

Fix the TP=1 knobs for the model, GPU, precision and runtime. Run the same
model at TP ∈ {2, 4, 8}, within one node. Then:

```
measured_step = stage_noncomm(tp) * (1+oh+p) + comm_raw(tp) * (1 - ar_overlap) * (1+oh+p)
```

This is linear in `ar_overlap`, so it has a closed-form least-squares
solution. Decode uses the C = 1-64 points; prefill uses L = 4096. If the model
does not fit at TP=1, use the smallest TP that fits for the base fit, and only
larger TPs for overlap. Cross-node TP uses the fixed 25 GB/s
`INTER_NODE_COLLECTIVE_BW` (`calc.py:30`). That is out of scope for the
minimum matrix.

Because `bw_eff` multiplies the interconnect bandwidth (`calc.py:1418`), a
fitted overlap is conditional on the fitted `bw_eff`. Store it with that
dependency recorded.

#### 1.4.5 MoE: `moe_imbalance`

Fit the dense knobs on a dense model with the same GPU, precision, runtime and
TP. Then run the MoE model through the same decode and prefill sweeps, compute
the planner prediction with `moe_imbalance = 1`, and set
`moe_imbalance = measured / predicted`. Fit decode and prefill points
separately, even though the planner has one shared knob.

"Matched active parameters" is not required. The planner already uses active
weight bytes (`calc.py:1471`). Pick a dense model with similar active params
only to keep the regime (weight-bound vs. compute-bound) comparable.

Expected structural gap (inference): decode weight bytes are
batch-independent (`Model.active_weight_bytes`, `data/model_class.py:663`).
Real MoE decode reads more distinct experts as batch grows, up to all experts.
The fitted ratio will therefore rise with C. Report the ratio per C bucket. If
it spans more than 1.3× across C, the single multiplier is the wrong shape;
document that and do not tune it away.

Expert parallel is flagged as unmodelled (`calc.py:1458`). Run MoE without
`--enable-expert-parallel` for the fit, or record that it was on and treat
the run as validation-only.

#### 1.4.6 Prefill/decode interference: `pd_interference`

1. Run the isolated prefill (1.4.3) and synchronized-cohort decode (1.4.2)
   for the workload shape, to get measured `t_pf` and `t_dec` (not planner
   predictions, so other knob errors do not leak in).
2. Run a closed-loop mixed run: `--max-concurrency C`, at least 10·C
   requests, the same lengths. Measure `total = C / measured_request_rate`.
3. Compute `pd_interference = (total - max(t_pf, t_dec)) / min(t_pf, t_dec)`,
   clipped to [0, 1].

This depends on chunked prefill, `--max-num-batched-tokens` and async
scheduling. Store the fit keyed on those flags (section 5.4). Use the planner's
closed-loop interpretation (`bs` in flight), not open-loop Poisson, for the
fit. Poisson runs are validation.

#### 1.4.7 Speculative acceptance: `spec_acceptance` and profile alphas

vLLM V1 counters [check version]: `vllm:spec_decode_num_drafts_total`,
`vllm:spec_decode_num_draft_tokens_total`,
`vllm:spec_decode_num_accepted_tokens_total`,
`vllm:spec_decode_num_accepted_tokens_per_pos_total` (label `position`).

```
mean_acceptance_length (tau_meas) = 1 + accepted / drafts
```

Solve `spec_acceptance_len(α, k) = tau_meas` for α by bisection on [0, 0.99];
the function is monotone in α. Use `spec_finite_output_tau` instead when
outputs are short (under about 8k tokens). Per-position counters give
conditional acceptance per position. Record them; the chain model assumes a
constant α, and the per-position spread measures how wrong that is.

α depends on the dataset, sampling temperature and k. It does not depend on
GPU. Store it per (model, spec method, k, dataset class, temperature) as
`acceptance_alpha_by_k`-style data. The global `spec_acceptance` override is
a single number and the wrong place for it. Measure throughput speedup at
C ∈ {1, 8, 32, 128} as a validation of the whole spec cost model
(`calc.py:1548-1581`), not as a fit target.

#### 1.4.8 Prefix reuse: `prefix_hit_rate`

Hit rate is a workload property, not a hardware knob. vLLM V1 counters
[check version]: `vllm:prefix_cache_queries_total` and
`vllm:prefix_cache_hits_total`, both in tokens. The V0 gauge
`vllm:gpu_prefix_cache_hit_rate` is deprecated. Then:

```
hit_rate = Δhits / Δqueries
```

Compute it from production or replayed traffic per use case, and use it to
replace `USE_CASE_PREFIX_HIT_RATES` (`data/use_cases.py:523`). The benchmark
only validates `effective_prefill_length`: with synthetic prefix-share
s ∈ {0, 0.5, 0.9}, the measured prefill time should match the planner's time
at `ceil(L·(1-hit))`, with `hit` taken from the counters. It should not be
taken from `s`, because block granularity means hit ≤ s.

---

## 2. Experiment matrix

### 2.1 Cell definition

A cell is one tuple of: `model_key`, `artifact_id`, `gpu_key`, `gpu_count`,
`(tp, pp, dp)`, `serving_runtime`, `runtime_version`, `kv_precision`
(`--kv-cache-dtype`), and `execution_backend`. Fits are per cell, with
fallbacks (section 5.5).

### 2.2 Minimum cells for one GPU type

| Cell | Example (H100) | Purpose | Required |
|---|---|---|---|
| D1 | `l8`, `native:bf16`, TP=1 | Base decode/prefill/memory/slack knobs | Yes |
| D2 | `l8`, an FP8 artifact (or `estimate:fp8`), TP=1 | Precision-specific efficiencies | Yes, if FP8 is served |
| D3 | `llama33-70b`, FP8 or BF16, TP ∈ {2,4,8} | `ar_overlap`, TP scaling of non-KV memory | Yes, if multi-GPU TP is planned |
| M1 | an MoE such as `q35` or `mx87` at the same precision as D1/D2 | `moe_imbalance` | Yes, if MoE is planned |
| S1 | `l8` with its EAGLE-3 profile (`data/models_text.py:79-89`) | α, spec cost validation | Yes, if spec decode is planned |
| P1 | D3 model at PP=2 | PP bubble and boundary validation | Optional |
| R1 | D1 with DP=2 | Replica independence validation | Optional |

Use the same runtime version for every cell on a GPU type. A runtime upgrade
invalidates fits (section 5.5).

### 2.3 Sweeps per cell

Input and output lengths are chosen from `INPUT_BUCKETS` / `OUTPUT_BUCKETS`
(`data/presets.py:13-30`). That way a one-hot distribution reproduces the
exact planner input, because `avg_dist` returns the bucket length
(`calc.py:1139-1141`).

| Sweep | Cells | Variables | Fits / validates | Required |
|---|---|---|---|---|
| S-mem | all | startup only; `mu` ∈ {0.90} (+0.80 optional) | geometry gate, `profiled_non_kv_gb` | Yes |
| S-cap | D1, D3, M1 | in=4096, out=512; C rising until first preemption | `kv_slack` | Yes |
| S-dec | D1, D2, D3, M1 | cohort decode; C ∈ {1,2,4,8,16,32,64,128,256} up to the cap; L_in ∈ {256, 4096}; L_out=512 | decode `bw_eff`, `comp_eff`, `paged_oh`, `ar_overlap`, `moe_imbalance` | Yes |
| S-dec-long | D1 | L_in=16384, C ∈ {1,8,32} | KV-read `bw_eff`, `paged_oh` trend | Yes |
| S-dec-mix | D1 | in_dist one-hot mix 256/4096/16384 equal weights, C=32 | `decode_paged_oh` heterogeneity | Optional |
| S-pre | D1, D2, D3, M1 | N ∈ {1,4,16}; L ∈ {256,1024,4096,16384}; max_tokens=1; plus N=1, L ∈ {32,128} | prefill `bw_eff`, `comp_eff`, `ar_overlap`, `moe_imbalance` | Yes |
| S-mix | D1 (+D3) | closed loop C ∈ {8,32,128}; (in,out) ∈ {(1024,128),(4096,512)} | `pd_interference` | Yes |
| S-mix-poisson | D1 | Poisson at 50/75/90% of the S-mix rate | held-out validation | Optional |
| S-prefix | D1 | in=4096, out=128, C=16, share s ∈ {0,0.5,0.9} | `effective_prefill_length` validation | Yes |
| S-spec | S1 | spec off and on; k ∈ {default−1, default, default+1}; C ∈ {1,8,32,128}; real text dataset, temperature 0 and 0.7 | α per k, spec speedup validation | Yes, if S1 |
| S-real | D1 | replay of a real dataset with natural lengths | end-to-end held-out validation | Optional |
| S-trace | D1 | profiler trace of 50 decode and 5 prefill steps at C=1 and C=64 | `overhead` (gauge fix) | Strongly recommended |

Repeat each fit point 3 times, or once if the coefficient of variation of the
first two runs is under 2%. Rough size for one GPU type with D1+D2+D3+M1+S1:
about 350-450 runs of 1-3 minutes each, under one day of machine time. This is
an estimate, not a measurement.

---

## 3. What to record per run

### 3.1 Workload descriptors

- Request count, concurrency mode (`closed` with `max_concurrency`, or
  `poisson` with `request_rate_rps`), and arrival seed.
- Prompt and output tokens per request, as server-reported (`usage` in the
  response). Record the requested lengths separately.
- `ignore_eos`, `min_tokens`, temperature, top_p, and the dataset id (with a
  hash of the prompt file).
- Prefix-share setting and prefix length; whether prompts were unique per run.
- Spec config: method (one of `SPEC_METHODS`, `data/model_class.py:90`), k,
  and draft artifact.
- Isolation mode: `cohort_decode`, `prefill_only`, `mixed_closed`,
  `mixed_poisson`, `capacity_probe`, `startup_only`.

### 3.2 Timing (client side, per request, then aggregated)

Record per-request raw arrays in a sidecar file, and in the record:

- TTFT: mean, p50, p90, p99, max (max is the planner's makespan TTFT).
- TPOT per request, `(e2e - ttft)/(out-1)`: mean, p50, p90, p99.
- ITL over all tokens: mean, p50, p90, p99.
- End-to-end latency: mean, p50, p90, p99.
- Request throughput; output token throughput; prompt token throughput.
- For cohort runs: `window_start_s`, `window_end_s`, and window-mean ITL.
- For prefill-only runs: makespan.

`vllm bench serve` (formerly `benchmarks/benchmark_serving.py`) reports most
of these with `--save-result --percentile-metrics ttft,tpot,itl,e2el`
[check version]. Cohort-window ITL needs per-token timestamps, which come from
the streaming client.

Server-side histograms [check version]: `vllm:time_to_first_token_seconds`,
`vllm:inter_token_latency_seconds` (older: `vllm:time_per_output_token_seconds`),
`vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`,
`vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds`,
`vllm:iteration_tokens_total`. Prefer the server-side prefill and decode times
for fitting, because they exclude HTTP and tokenization.

### 3.3 Runtime metrics (Prometheus `/metrics`)

Scrape before the run, at 1 s intervals during it, and after it. Store counter
deltas and gauge time series summarized as mean, max, and the value
in-window. All names are [check version]:

| Metric | Use |
|---|---|
| `vllm:num_requests_running` | Confirms the actual batch equals the target C |
| `vllm:num_requests_waiting` | Detects queueing and capacity limits |
| `vllm:kv_cache_usage_perc` (older: `vllm:gpu_cache_usage_perc`) | KV occupancy, as a fraction 0-1 despite the name |
| `vllm:num_preemptions_total` | Capacity-limit detection; retried work |
| `vllm:prompt_tokens_total` | Prefill token throughput (whether cached tokens are included is [check version]) |
| `vllm:generation_tokens_total` | Decode token throughput |
| `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total` | Prefix hit rate (tokens) |
| `vllm:spec_decode_num_drafts_total`, `..._num_draft_tokens_total`, `..._num_accepted_tokens_total`, `..._num_accepted_tokens_per_pos_total` | α |
| `vllm:cache_config_info` (labels include `num_gpu_blocks`, `block_size`, `gpu_memory_utilization`, `cache_dtype`) | Memory facts |
| `vllm:request_success_total` (label `finished_reason`) | Confirms `length` finishes under `ignore_eos` |

For SGLang and TRT-LLM, record the analogous metrics under a `runtime_metrics`
map with the native names. The fitter needs only the derived quantities
listed in section 4.

### 3.4 Memory facts (from startup log and `cache_config_info`)

- Model weights in bytes ("Model loading took X GiB"; convert GiB to bytes).
- Peak activation bytes and non-torch bytes, if logged.
- CUDA graph memory bytes ("Graph capturing finished in … took X GiB").
- Available KV memory bytes; `num_gpu_blocks`; `block_size`; KV capacity in
  tokens.
- Device total memory as reported (`torch.cuda.mem_get_info`).
- Draft-model weights, when spec decode is on.

### 3.5 Environment

- GPU model string, count, PCI and NVLink topology (`nvidia-smi topo -m`), and
  GPU serials or cloud instance id plus region.
- Driver version, CUDA version, NCCL version, PyTorch version.
- Runtime and version, plus the git commit if built from source.
- Container image digest.
- Full server command line, including at least: `--tensor-parallel-size`,
  `--pipeline-parallel-size`, `--data-parallel-size`,
  `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`,
  `--max-num-batched-tokens`, `--enable-chunked-prefill`,
  `--enable-prefix-caching` / `--no-enable-prefix-caching`,
  `--kv-cache-dtype`, `--block-size`, `--async-scheduling`,
  `--enforce-eager`, `--compilation-config` (with `cudagraph_capture_sizes`),
  `--speculative-config`, `--enable-expert-parallel`, `--quantization`
  [check version].
- Relevant environment variables (`VLLM_*`, `NCCL_*`, `CUDA_VISIBLE_DEVICES`).
- Client host, client CPU count, client process count, and whether the client
  shares the server host.

### 3.6 Power and clocks (optional, recommended)

Sample every 500 ms. Use `nvidia-smi
--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active
--format=csv -lms 500`, or DCGM (`DCGM_FI_DEV_POWER_USAGE`,
`DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_PROF_DRAM_ACTIVE`,
`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`) [check version]. `DRAM_ACTIVE` in the
decode window is an independent check on `decode_bw_eff`. Energy per run in
joules feeds the CO2 math, which uses `tdp_watts` today
(`data/hardware.py:27`).

---

## 4. Record format

### 4.1 Conventions

- One JSON object per line (JSONL), one line per run. Raw per-request arrays
  go in a sidecar file named by `run_id`.
- `schema_version` is a string, `"gpucalc.bench/1"`. Breaking changes bump the
  integer.
- SI base units everywhere, with the unit in the field name: seconds (`_s`),
  bytes (`_bytes`, decimal integers, never GiB), tokens (`_tokens`), tokens per
  second (`_tok_per_s`), requests per second (`_rps`), watts (`_w`), joules
  (`_j`), fractions in [0, 1] (`_frac`). No milliseconds.
- Timestamps are ISO 8601 UTC.
- Keys must match planner catalog keys exactly:
  - `model_key` is in `MODELS` (`data/models.py`).
  - `gpu_key` is in `GPUS` (`data/hardware.py:84`).
  - `precision` is in `PRECISIONS` = `bf16|fp8|nvfp4|mxfp4`
    (`data/specs.py:229-262`).
  - `artifact_id` is `native:<prec>`, `estimate:<prec>` or
    `hf:<repo>@<revision>` (`data/artifacts.py:26-29`, `data/specs.py:183`).
  - `serving_runtime` is in `SERVING_RUNTIMES` = `generic|vllm|sglang|trtllm`
    (`data/specs.py:89`).
  - `kv_precision` is in `CACHE_PRECISIONS`, `execution_backend` in
    `EXECUTION_BACKENDS` (`data/specs.py:90-91`).
  - `spec.method` is in `SPEC_METHODS` (`data/model_class.py:90`).
- `measured` and `derived` are separate objects. Anything computed by the
  harness rather than read from the client, server or log goes under
  `derived`, with a `method` string. This keeps measured facts apart from
  inference, as AGENTS.md requires.

### 4.2 Fields

R = required, O = optional, C = required when the condition holds.

| Path | Type | R/O | Notes |
|---|---|---|---|
| `schema_version` | str | R | `"gpucalc.bench/1"` |
| `run_id` | str | R | UUID |
| `campaign_id` | str | R | Groups runs fitted together |
| `provenance.started_at`, `.finished_at` | str | R | ISO 8601 UTC |
| `provenance.operator` | str | O | |
| `provenance.harness_commit` | str | R | Benchmark harness git SHA |
| `provenance.planner_commit` | str | R | Planner SHA whose catalog the keys refer to |
| `provenance.host` | obj | R | `{cloud, instance_type, instance_id, region}` or `{on_prem: true, hostname}` |
| `provenance.gpu_serials` | [str] | R | `nvidia-smi --query-gpu=serial` |
| `provenance.notes` | str | O | |
| `deployment.model_key` | str | R | |
| `deployment.artifact_id` | str | R | |
| `deployment.hf_repo`, `.hf_revision` | str | R | What was actually loaded |
| `deployment.precision` | str | R | |
| `deployment.kv_precision` | str | R | |
| `deployment.execution_backend` | str | O | Default `auto` |
| `deployment.gpu_key` | str | R | |
| `deployment.gpu_count`, `.tp`, `.pp`, `.dp` | int | R | `tp*pp*dp == gpu_count` |
| `deployment.expert_parallel` | bool | R | |
| `deployment.serving_runtime`, `.runtime_version`, `.runtime_commit` | str | R (commit O) | |
| `deployment.flags` | obj | R | Parsed server flags (3.5); unknown flags kept verbatim |
| `deployment.spec` | obj | C | `{method, k, draft_repo, draft_revision}` when spec is on, else `null` |
| `environment` | obj | R | `{driver, cuda, nccl, torch, image_digest, env_vars, topology}` |
| `client` | obj | R | `{tool, tool_version, host_same_as_server, processes, cpu_count}` |
| `workload.mode` | str | R | One of the isolation modes (3.1) |
| `workload.num_requests` | int | R | |
| `workload.max_concurrency` | int | C | closed loop |
| `workload.request_rate_rps` | float | C | poisson |
| `workload.input_tokens_requested`, `.output_tokens_requested` | int | R | |
| `workload.input_dist` | [[int, float]] | O | `[[len, weight]]` for mixed lengths |
| `workload.prefix_share_frac`, `.prefix_tokens` | float, int | R | 0 if unused |
| `workload.ignore_eos`, `.temperature`, `.top_p`, `.seed` | | R | |
| `workload.dataset`, `.dataset_sha256` | str | R | `random` is allowed for dense, but not for MoE or spec |
| `workload.warmup_requests`, `.prefix_cache_reset` | int, bool | R | |
| `measured.window_s` | [float, float] | R | Analysis window, relative to start |
| `measured.input_tokens_mean`, `.output_tokens_mean` | float | R | From server `usage` |
| `measured.requests_completed`, `.requests_failed` | int | R | |
| `measured.ttft_s` | obj | R | `{mean,p50,p90,p99,max}` |
| `measured.tpot_s`, `.itl_s`, `.e2e_s` | obj | R | `{mean,p50,p90,p99}` |
| `measured.cohort_window_itl_mean_s` | float | C | `cohort_decode` |
| `measured.prefill_makespan_s` | float | C | `prefill_only` |
| `measured.server_prefill_time_s`, `.server_decode_time_s`, `.server_queue_time_s` | obj | O | From server histograms |
| `measured.request_throughput_rps` | float | R | |
| `measured.prompt_tok_per_s`, `.generation_tok_per_s` | float | R | Counter deltas over window |
| `measured.runtime.num_requests_running` | obj | R | `{mean, max, in_window_min}` |
| `measured.runtime.num_requests_waiting` | obj | R | `{mean, max}` |
| `measured.runtime.kv_cache_usage_frac` | obj | R | `{mean, max}` |
| `measured.runtime.preemptions` | int | R | Counter delta |
| `measured.runtime.prefix_cache_queries_tokens`, `.prefix_cache_hits_tokens` | int | R | Deltas; 0 if disabled |
| `measured.runtime.spec_drafts`, `.spec_draft_tokens`, `.spec_accepted_tokens` | int | C | spec |
| `measured.runtime.spec_accepted_per_pos` | [int] | C | spec |
| `measured.runtime.raw_metric_names_version` | str | R | Which names were scraped |
| `measured.memory.device_total_bytes` | int | R | Physical, from torch |
| `measured.memory.weights_bytes` | int | R | Startup log |
| `measured.memory.cudagraph_bytes` | int | O | |
| `measured.memory.peak_activation_bytes`, `.non_torch_bytes` | int | O | |
| `measured.memory.kv_available_bytes` | int | R | |
| `measured.memory.num_gpu_blocks`, `.block_size_tokens`, `.kv_capacity_tokens` | int | R | |
| `measured.memory.draft_weights_bytes` | int | C | spec |
| `measured.trace` | obj | C | S-trace: `{steps, wall_step_mean_s, gpu_busy_mean_s, file}` |
| `measured.power` | obj | O | `{mean_w, max_w, energy_j, sm_clock_mhz_min, throttle_reasons}` |
| `derived` | obj | O | e.g. `{"capacity_C_max": 61, "method": "first preemption"}` |
| `validity` | obj | R | `{valid: bool, reasons: [str]}`, set by the harness checks in section 6 |

### 4.3 Example record

Values are illustrative placeholders showing shape and units, not
measurements.

```json
{
  "schema_version": "gpucalc.bench/1",
  "run_id": "0b6f6f0e-5c1e-4d55-9a4f-3c2b0f8d7a11",
  "campaign_id": "h100-vllm-0.11-2026-11",
  "provenance": {
    "started_at": "2026-11-03T14:02:11Z",
    "finished_at": "2026-11-03T14:04:37Z",
    "harness_commit": "a1b2c3d",
    "planner_commit": "3820dd6",
    "host": {"cloud": "aws", "instance_type": "p5.48xlarge", "instance_id": "i-0abc123", "region": "us-east-1"},
    "gpu_serials": ["1654323012345"]
  },
  "deployment": {
    "model_key": "l8",
    "artifact_id": "native:bf16",
    "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
    "hf_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    "precision": "bf16",
    "kv_precision": "bf16",
    "execution_backend": "auto",
    "gpu_key": "H100",
    "gpu_count": 1, "tp": 1, "pp": 1, "dp": 1,
    "expert_parallel": false,
    "serving_runtime": "vllm",
    "runtime_version": "0.11.0",
    "runtime_commit": null,
    "flags": {
      "gpu_memory_utilization": 0.90,
      "max_model_len": 32768,
      "max_num_seqs": 256,
      "max_num_batched_tokens": 8192,
      "enable_chunked_prefill": true,
      "enable_prefix_caching": false,
      "kv_cache_dtype": "auto",
      "block_size": 16,
      "async_scheduling": true,
      "enforce_eager": false,
      "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256]
    },
    "spec": null
  },
  "environment": {
    "driver": "580.65.06", "cuda": "12.9", "nccl": "2.27.3", "torch": "2.8.0",
    "image_digest": "sha256:4f1c…", "env_vars": {"VLLM_USE_V1": "1"},
    "topology": "single GPU"
  },
  "client": {"tool": "vllm bench serve", "tool_version": "0.11.0", "host_same_as_server": false, "processes": 1, "cpu_count": 32},
  "workload": {
    "mode": "cohort_decode",
    "num_requests": 32, "max_concurrency": 32, "request_rate_rps": null,
    "input_tokens_requested": 4096, "output_tokens_requested": 512,
    "prefix_share_frac": 0.0, "prefix_tokens": 0,
    "ignore_eos": true, "temperature": 0.0, "top_p": 1.0, "seed": 7,
    "dataset": "random", "dataset_sha256": "9c1e…",
    "warmup_requests": 64, "prefix_cache_reset": true
  },
  "measured": {
    "window_s": [1.92, 7.10],
    "input_tokens_mean": 4096.0, "output_tokens_mean": 512.0,
    "requests_completed": 32, "requests_failed": 0,
    "ttft_s": {"mean": 1.05, "p50": 1.07, "p90": 1.80, "p99": 1.89, "max": 1.90},
    "tpot_s": {"mean": 0.01012, "p50": 0.01011, "p90": 0.01020, "p99": 0.01031},
    "itl_s": {"mean": 0.01012, "p50": 0.01008, "p90": 0.01025, "p99": 0.01090},
    "e2e_s": {"mean": 6.22, "p50": 6.24, "p90": 6.95, "p99": 7.03},
    "cohort_window_itl_mean_s": 0.01009,
    "request_throughput_rps": 4.55,
    "prompt_tok_per_s": 0.0,
    "generation_tok_per_s": 3171.0,
    "runtime": {
      "num_requests_running": {"mean": 31.6, "max": 32, "in_window_min": 32},
      "num_requests_waiting": {"mean": 0.0, "max": 0},
      "kv_cache_usage_frac": {"mean": 0.33, "max": 0.36},
      "preemptions": 0,
      "prefix_cache_queries_tokens": 0, "prefix_cache_hits_tokens": 0,
      "raw_metric_names_version": "vllm-0.11"
    },
    "memory": {
      "device_total_bytes": 85029158912,
      "weights_bytes": 16060522496,
      "cudagraph_bytes": 536870912,
      "kv_available_bytes": 57982058496,
      "num_gpu_blocks": 28311, "block_size_tokens": 16, "kv_capacity_tokens": 452976
    },
    "power": {"mean_w": 612.0, "max_w": 688.0, "energy_j": 3260.0, "sm_clock_mhz_min": 1755, "throttle_reasons": []}
  },
  "derived": {},
  "validity": {"valid": true, "reasons": []}
}
```

---

## 5. Turning records into knob values (proposal)

### 5.1 Prediction function

The fitter imports `calc` and calls the internal, unrounded functions for
each valid record:

- `cohort_decode`: `_decode_step_time(m, tp, pp, C/dp, g, prec, L_in + L_out/2, eff, paged_oh, spec=...)`,
  with `paged_oh = decode_paged_oh(one_hot(L_in), one_hot(L_out), eff)`.
  Compare with `cohort_window_itl_mean_s · tau` (`tau = 1` without spec).
- `prefill_only`: `compute_prefill(...).service_time` with
  `seq_len = effective_prefill_length(L, hit)`. Compare with
  `prefill_makespan_s` or the server prefill time.
- `capacity_probe` / `startup_only`: `compute_memory(...)` fields.
- `mixed_closed`: `compute_data(...)`; compare `rps`.

`m` is built with `configured_model(model_key, artifact_id, execution_backend,
kv_precision, ...)` (`data/artifacts.py:61`), so artifact-specific weight bytes
and compute shares are used. Records whose keys do not resolve are rejected.

### 5.2 Loss and optimizer

- Residual: `r = ln(predicted / measured)`. This is scale-free and treats
  over- and under-prediction symmetrically.
- Loss: robust sum, `Σ huber(r, δ = 0.05)`, weighted by `1/n` per sweep so the
  long C sweeps do not dominate.
- Bounded least squares, e.g. `scipy.optimize.least_squares` with `bounds`
  taken from `EFFICIENCY_SETTING_BOUNDS`. The parameter counts are small (2-3
  per stage), so a dense grid at 0.01 resolution is an acceptable,
  dependency-free alternative.
- Uncertainty: bootstrap over runs (200 resamples). Report the 5th and 95th
  percentiles.

### 5.3 Staged fit order

Fit stages run in order; each later stage holds the earlier results fixed.

| Stage | Data | Fit | Held fixed |
|---|---|---|---|
| 0 | S-mem | gate: KV bytes/token and weights within 2%; else stop | n/a |
| 1 | S-mem, S-cap | `profiled_non_kv_gb` (closed form), `kv_slack` (closed form); `mu` and `sched_budget` copied from flags | n/a |
| 2 | S-trace | `decode_overhead`, `prefill_overhead` (closed form) | n/a |
| 3 | S-dec, S-dec-long, D1/D2 (TP=1) | `decode_bw_eff`, `decode_comp_eff` jointly; then `decode_paged_oh` | stage 1-2 |
| 4 | S-pre, D1/D2 | `prefill_bw_eff`, `prefill_comp_eff` jointly; then `prefill_paged_oh` | stage 1-2 |
| 5 | S-dec, S-pre, D3 | `decode_ar_overlap`, `prefill_ar_overlap` (linear) | stages 1-4 for the same GPU, precision and runtime |
| 6 | S-dec, S-pre, M1 | `moe_imbalance` (ratio; report per C and phase) | stages 1-5 |
| 7 | S-mix | `pd_interference` (closed form from measured isolated times) | n/a |
| 8 | S-spec | α per (method, k, dataset, temperature) | n/a |
| 9 | all held-out runs | acceptance (5.7) | everything |

Rules:

- A parameter whose bootstrap interval spans more than half its bound range
  is stored as `"identified": false`, and the prior stays in effect.
- A fitted value sitting on a bound is stored with `"at_bound": true` and is
  not accepted automatically. It means the planner's structure, not the knob,
  is wrong for that cell.

### 5.4 Storage

Proposed files (not created by this document):

- `data/calibration/measurements/<campaign_id>.jsonl`: raw records,
  version-controlled. Sidecars live outside git, referenced by hash.
- `data/calibration/fits.json`: fitted values, reviewed like catalog data.

Shape of `fits.json`:

```json
{
  "schema_version": "gpucalc.fits/1",
  "generated_at": "2026-11-05T09:00:00Z",
  "fitter_commit": "…",
  "entries": [
    {
      "key": {
        "model_key": "l8", "artifact_id": "native:bf16", "gpu_key": "H100",
        "precision": "bf16", "kv_precision": "bf16", "tp": 1,
        "serving_runtime": "vllm", "runtime_version_minor": "0.11",
        "flags_class": "chunked+async+cudagraph"
      },
      "baseline": {"gpu_mem_bytes": 80e9, "effective_bw_bytes_per_s": 3.35e12, "bf16_flops": 989e12},
      "knobs": {
        "decode_bw_eff": {"value": 0.86, "ci90": [0.84, 0.88], "identified": true, "at_bound": false, "n_runs": 54},
        "decode_overhead": {"value": 0.05, "source": "trace", "identified": true},
        "decode_comp_eff": {"value": 0.75, "identified": false, "note": "knee not reached"}
      },
      "residuals": {"decode_mae_log": 0.04, "prefill_mae_log": 0.07, "heldout_mae_log": 0.06},
      "campaign_ids": ["h100-vllm-0.11-2026-11"],
      "measured_on": "2026-11-03"
    }
  ],
  "workload": {
    "prefix_hit_rate": {"coding": {"value": 0.62, "source": "prod telemetry 2026-10", "n_tokens": 1.2e9}},
    "spec_alpha": [{"model_key": "l8", "method": "eagle3", "k": 3, "dataset": "sharegpt-like", "temperature": 0.0, "alpha": 0.74}]
  }
}
```

`baseline` records the catalog values the fit was taken against. If
`GPUS[gpu_key].effective_bw`, `.mem` or `.bf16` later changes, the loader must
mark the entry stale. Otherwise a corrected catalog value would silently
double-correct.

### 5.5 Fallback order

For each knob, use the first match:

1. exact key;
2. same model, GPU, precision and runtime minor version, any `tp`
   (hardware-only knobs: `bw_eff`, `comp_eff`, `overhead`, `paged_oh`);
3. same GPU, precision and runtime, model class (dense or MoE): the median of
   fitted entries;
4. same GPU and precision, any runtime: median, flagged "cross-runtime";
5. `PlannerState` default prior.

`ar_overlap` falls back only within the same GPU and TP width.
`profiled_non_kv_gb` falls back only within the same runtime minor version,
because CUDA graph and activation reservations change between releases. A
different runtime minor version never matches levels 1-3.

### 5.6 Integration (code change, out of scope here)

`EfficiencyParams` is global per state today (`state.py:504-528`). To use the
fits, the planner needs:

- a resolver `efficiency_for(state, assignment, gpu, phase)` that merges
  defaults, then fitted values, then explicit user overrides;
- call sites in `placement.py`, `presentation/charts.py`,
  `engine/economics.py` and `planner_service.py` switched from
  `state.prefill_efficiency` / `state.decode_efficiency` to that resolver.

`pd_interference` is resolved per assignment as `max(prefill, decode)` today
(`calc.py:2299`), which keeps working. The UI should label each knob's source
("measured fit, 2026-11-03", "default prior", "user override"), following the
existing `alpha_source` pattern in `presentation/model_cards.py:274-289`. A
planner-math change like this needs before-and-after planner reports
(AGENTS.md, "Math changes").

### 5.7 Acceptance criteria

Hold out 20% of runs, stratified so that at least one full (L_in, C) row and
all S-mix-poisson and S-real runs are held out. Error is
`|predicted/measured - 1|`.

| Quantity | Median error | Max error |
|---|---|---|
| Decode per-token latency (cohort) | ≤ 8% | ≤ 20% |
| Decode token throughput | ≤ 8% | ≤ 20% |
| Prefill makespan / prompt throughput, L ≥ 1024 | ≤ 10% | ≤ 25% |
| KV capacity in tokens (planner `kv_budget / kv_per_token` vs. `kv_capacity_tokens`) | ≤ 3% | ≤ 5% |
| Max concurrency before preemption (`C_max`) | n/a | ≤ 5% |
| Mixed closed-loop request rate | ≤ 12% | ≤ 25% |
| Spec speedup (on vs. off, same C) | ≤ 10% | ≤ 20% |

A fit is also rejected unless its held-out error is lower than the default
priors' held-out error on the same runs. These thresholds are proposals; they
can be tightened after the first campaign shows the achievable noise floor
(run-to-run coefficient of variation).

---

## 6. Pitfalls and harness checks

The harness should check each of these automatically and write failures into
`validity.reasons`.

- **Warmup.** Send at least 2·C requests of the same shape before each
  measured run, and discard the first run after a server start. JIT,
  autotuning and allocator growth happen on first use.
- **CUDA graph capture.** Decode batch sizes not in `cudagraph_capture_sizes`
  are padded up to the next captured size or run eagerly, which creates steps
  in the latency-vs-C curve. Record the capture sizes, use C values that are
  captured sizes, and do not fit across a padding step without noting it.
  `--enforce-eager` runs belong to a different `flags_class`.
- **Async scheduling.** It hides CPU scheduling time and changes the overhead
  knob. Fit with and without it as separate `flags_class` values. Do not mix
  them.
- **Prefix-cache contamination.** Disable prefix caching for all non-prefix
  sweeps, or reset it between runs (`POST /reset_prefix_cache`, which may need
  `VLLM_SERVER_DEV_MODE=1` [check version]) and use unique random prompt
  prefixes per run. Check: `prefix_cache_hits_tokens == 0` on runs that
  should have no reuse.
- **Chunked prefill.** With `--max-num-batched-tokens` below N·L, prefill is
  split across steps and mixed with decode. Prefill-only runs must either
  raise the limit or be marked validation-only.
- **Client bottleneck.** Python streaming clients become CPU-bound at a few
  thousand tokens per second per process. Check: `num_requests_running.in_window_min == C`
  for closed-loop runs, and client CPU below 70%. Otherwise add client
  processes or move the client to another host. Measure the client's own
  overhead once against a mock server.
- **Tokenizer differences.** The client tokenizer may disagree with the
  server's (chat templates, BOS, special tokens). Always use server-reported
  `usage.prompt_tokens` / `completion_tokens`. Random-token datasets can
  re-tokenize to a different length; reject runs whose measured mean deviates
  from the requested length by more than 1%.
- **Useful vs. retried tokens.** Preemption by recompute re-runs prefill, so
  server prompt-token counters include retried work. Fitting throughput uses
  client-side useful tokens. Any run with `preemptions > 0` is excluded from
  decode and prefill fits (it is a capacity datum only). Count only requests
  with `finished_reason == length` under `ignore_eos`; exclude errors and
  aborts.
- **Early stop.** Without `--ignore-eos`, output lengths vary and the
  synchronized cohort breaks. Fit runs require `ignore_eos = true`. Spec runs
  should use a real dataset, because acceptance on forced post-EOS text is not
  representative.
- **MoE routing variance.** Random token ids give unrealistic expert routing.
  Use a fixed real-text dataset for M1 and S-spec, repeat with 3 seeds, and
  report the spread.
- **Thermal and power throttling.** Check that `clocks_throttle_reasons` shows
  no power or thermal cap during the window and that SM clock stays within 5%
  of its value at run start. Leave at least 30 s idle between long runs.
  Record ambient conditions for on-prem hardware.
- **Measurement window.** Exclude ramp-up and drain. For closed-loop and
  Poisson runs, use only the interval where `num_requests_running` is within
  5% of its plateau.
- **Units.** vLLM logs GiB; the catalog uses decimal bytes (`GPU.mem = 80e9`);
  `kv_cache_usage_perc` is a 0-1 fraction despite its name. Convert at
  ingestion, never in the fitter.
- **Planner rounding.** `DecodeResult` and `UserExperienceResult` are rounded
  (`calc.py:1671-1678`, `2440-2446`). Fit against the unrounded internals.
- **Multi-GPU topology.** Record `nvidia-smi topo -m`. TP over PCIe vs.
  NVLink on the same `gpu_key` will produce very different `ar_overlap`, and
  the catalog's `scale_up_p2p_bw_bidir` assumes the catalog topology.
- **Runtime drift.** vLLM metric names, defaults (`max_num_seqs`,
  `max_num_batched_tokens`, chunked prefill, async scheduling) and memory
  profiling change between minor versions. Pin the version per campaign and
  store `raw_metric_names_version`.

---

## 7. Open questions

- **Q1. `decode_sched_budget` semantics.** The code uses it as a cap on running
  sequences per replica (`calc.py:1645-1646`), which corresponds to vLLM
  `--max-num-seqs`. The name suggests a token budget
  (`--max-num-batched-tokens`), and the UI bound starts at 2048
  (`web/config.py:112`), above vLLM's usual `max_num_seqs` of 256-1024
  [check version]. A measured config therefore cannot be entered through the
  UI. This needs a decision: either lower the bound and document it as
  max-num-seqs, or add a separate token-budget term.
- **Q2. The overhead gauge** (1.4.2). This proposal fixes it with a profiler
  trace. If traces are not available on the target hardware, all
  efficiencies are relative to the 0.08 prior.
- **Q3. Coupling of `bw_eff` and interconnect** (`calc.py:1418`, `1436-1437`).
  Fitted `ar_overlap` is conditional on `bw_eff`. A separate interconnect
  efficiency would make the fit cleaner, but that is a math change.
- **Q4. Batch-dependent MoE weight reads** (1.4.5). If the fitted
  `moe_imbalance` rises strongly with C, the planner needs an expert-coverage
  term rather than a constant multiplier.
- **Q5. The two paged-overhead paths** (`decode_paged_oh` vs.
  `fixed_paged_oh` for the same fixed-length workload, `calc.py:1711` vs.
  `2292`). Should these be unified before fitting `paged_oh`?
- **Q6. Whether vLLM's `prompt_tokens_total` includes prefix-cached tokens**
  in the target version. This decides whether prefill throughput comes from
  that counter or from client `usage` minus hits.
