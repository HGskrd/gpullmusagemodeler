# Decision models — 17 October 2026

These checkpoints score choices in one forward pass. The planner has a separate Decision picker and Decision throughput chart. No autoregressive decode, generated-token revenue, prefix-cache discount or speculative speedup is credited. This does not connect to an inference service or download weights.

The chart assumes one question with four options per request. Input positions include the complete formatted state, question and choices. The Python estimator also accepts multiple questions and reports decisions/s separately from requests/s; increasing question count must include its additional input tokens. Over-limit requests are rejected, never silently truncated.

| Checkpoint | Stored parameters | Native GB | Input cap | Pinned config |
|---|---:|---:|---:|---|
| Liquid d1 3B | 3,123,483,888 | 6.247 | 32,768 | [051bcc46](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/config.json) |
| Liquid d1 Omni 600M | 587,161,089 | 2.349 | 16,384 | [02b55d70](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/config.json) |
| Perplexity Decider v1.1 27B | 26,086,635,760 | 52.173 | 262,144 | [14c9d901](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/config.json) |
| Perplexity Decider Flash v1.2 (4B) | 4,539,918,336 | 9.080 | 262,144 | [6d771ad7](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d/config.json) |
| Cloudflare Clef | 27,356,728,560 | 54.713 | 16,384 | [0b331204](https://huggingface.co/Cloudflare/clef/blob/0b331204bb13fbd2ca93a64df1956f5b55478ce5/config.json) |
| Cloudflare Clef Flash | 9,409,813,744 | 18.820 | 16,384 | [8b2e5fd1](https://huggingface.co/Cloudflare/clef-flash/blob/8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3/config.json) |
| Mapika Decider 2B | 1,881,825,088 | 3.764 | 32,768 | [ff99089b](https://huggingface.co/Mapika/decider-2b/blob/ff99089b6810691bd79134ec6febb5d9afa6e619/config.json) |
| Mapika Decider 4B | 4,205,751,296 | 8.412 | 32,768 | [99d7fd5b](https://huggingface.co/Mapika/decider-4b/blob/99d7fd5bfcc4d11ba37239dafe0ed653803819d3/config.json) |
| Mapika Decider 35B A3B | 34,660,610,688 | 69.321 | 32,768 | [91470e65](https://huggingface.co/Mapika/decider-35b-a3b/blob/91470e65478656eb68fde8c0ccdfe575d69d0309/config.json) |
| Kev 27B | 25,624,600,064 | 51.249 | 65,536 | [af0e6d55](https://huggingface.co/jaredpalmer/kev-27b/blob/af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf/config.json) |
| Intern-Decision 0.8B | 852,985,920 | 1.706 | 8,192 | [85a0cc5a](https://huggingface.co/internlm/Intern-Decision-0.8B/blob/85a0cc5a99d67ea8d56dfe98115689212867171d/config.json) |
| Intern-Decision 2B | 2,213,241,664 | 4.426 | 8,192 | [8797836c](https://huggingface.co/internlm/Intern-Decision-2B/blob/8797836c65fc91a2435b1fb6850b5f0aabd75cc3/config.json) |
| Intern-Decision 4B | 4,539,265,536 | 9.079 | 8,192 | [1e65a988](https://huggingface.co/internlm/Intern-Decision-4B/blob/1e65a988dacbcf00b84aee7366f9e47563407000/config.json) |

Caps use published serving defaults for Clef, Mapika and Intern, d1 Omni's combined-position limit, Kev's validated envelope, and the config ceiling for d1 3B and Perplexity. Perplexity v1.1 uses noncausal full attention and native linear attention. Follow each pinned model card's custom inference implementation, calibration and access/license requirements; the planner is not proof that a generic generative runtime supports the checkpoint.

Native storage comes from pinned Hub tensor dtype inventories, including modality components. Perplexity v1.1 adds the separately published BF16 255×5120 readout (2,611,200 bytes) outside its backbone shard index. d1 Omni retains FP32 storage while BF16 compute is an explicit proxy. Other precision choices are hypothetical conversions unless backed by a named artifact. No released quantized decision artifact is asserted here.

Capacity uses a prefill roofline: two FLOPs per active parameter per input position plus attention, one active-weight read, existing communication and efficiency assumptions. Fused attention avoids a materialized quadratic score matrix. Each rank reserves four BF16 activation planes and FP32 linear recurrent state per request, conservatively without TP/PP activation sharding. This is a closed-form, text-only capacity estimate requiring deployment calibration. Multimodal encoder execution, image/audio preprocessing, tokenization, CPU/custom heads and calibrated decision quality are unmodeled. Full modality weights remain resident; dense active-parameter counts conservatively use the entire checkpoint in the text FLOP proxy. Published Decision Index scores are not comparable to the planner's language-model quality scale and are not fabricated into it.

Primary release context: [Liquid d1](https://www.liquid.ai/blog/d1-open), [Mapika inference](https://github.com/Mapika/decider), [Intern-Decision paper](https://arxiv.org/abs/2610.22131). Per-checkpoint cards and configs are available at the pinned repository links above.


## Verification

The final combined checkout passes compile, Ruff lint/format, mypy, and 470 tests with 933 subtests. All 13 decision models execute through the public calculation API and emit single-forward reports. Focused numerical checks cover the independent causal-attention roofline, FP32 native residency, activation memory limits, data-parallel scaling, serving caps, zero generated tokens, and exclusion from generation supply and quality ranking.

Headless browser acceptance starts blank, adds H200 hardware, selects d1 from the 13-entry Decision tab, and checks assignment, native storage, forward work-buffer labels and the Decision throughput chart. The actual UI export is re-imported after a local server restart. Raising input above the cap displays no capacity; returning to 64 positions restores a chart with nine positive points. Estimates range from 552.509 to 1,667.009 decisions/s over batches 1–256 on one assigned H200 with the displayed default efficiencies. These are planner estimates, not measured benchmark results. The 2,048-token API report with two H200 GPUs estimates 102.83 requests/s and zero generated tokens.

The existing Qwen3.8 27B FP8 control report remains unchanged: cost/day 31.68 and estimated capacity 1,009,461,325.2 tokens/day. Work remains local and uncommitted; no production deployment was performed.


## Perplexity Decider Flash v1.2 (4B)

Added from official revision `6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d`. The Qwen3.5-4B backbone has 32 layers: eight noncausal full-attention layers and 24 causal linear-attention layers. Its indexed 4,539,265,536 BF16 parameters occupy 9,078,531,072 bytes. The separate 255×2560 BF16 readout adds 652,800 parameters and 1,305,600 bytes, yielding **4,539,918,336 parameters and 9,079,836,672 bytes** in total.

[decision_config.json](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d/decision_config.json) specifies noncausal full attention; [joint_config.json](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d/joint_config.json) specifies protocol v3 and per-question inline markers. The [official card](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2/blob/6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d/README.md) gives a default 262,144-position limit, but training covered about 32k tokens and longer inputs need validation. The planner rejects requests above its input cap; it does not simulate the library's recursive splitting of multi-question requests. Multimodal encoder latency, compile/CUDA-graph packing, last-layer pruning, and optional FP8 runtime speedups remain unmodeled. Native storage includes the vision components. No quality score or fast-path benchmark is credited to the capacity proxy.

Validation for Flash v1.2: native bytes and separate-head accounting, hybrid attention geometry, noncausal profile, input cap and zero-output behavior are covered by focused tests. Its single-H200, 2,048-position report estimates 34.63 requests/s (one question per request) and zero generated tokens; this is a capacity proxy, not a measured benchmark. The complete suite passes 470 tests and 933 subtests; compile, Ruff lint/format, mypy and whitespace checks pass.

Flash v1.2 browser acceptance also passes from a blank scenario: H200 added, model chosen from the 13-entry Decision tab, native BF16 plus separate-head footprint inspected, one-GPU assignment confirmed, and the Decision throughput chart shows nine positive points (34.633 decisions/s at 2,048 input positions under the default efficiencies).
