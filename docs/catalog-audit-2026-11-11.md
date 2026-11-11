# November 11, 2026 catalog and capacity corrections

This implements the approved **Add/update now** audit findings. Sources were
captured November 11. Exact tensor storage, inferred active traffic, and runtime
performance are independent evidence. No model weights were downloaded: tensor
inventories used metadata and HTTP range requests for safetensors headers.

## Corrected capacity

- **MiMo V2.6 Flash/Pro RL:** separate K192 and V128 in KV residency and reads,
  and QK/V widths in local/global attention FLOPs. At 131,072 positions in BF16,
  Flash needs 3,045,457,920 bytes and Pro 6,750,208,000 bytes, rather than
  2,436,366,336 and 5,400,166,400. The hidden V2.5 parents retain their previous
  assumptions pending their own pinned implementation review.
  [Flash implementation](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/f657ba2f7d24c9adc5730675baaa2d34de878a49/modeling_mimo_v2.py),
  [Pro config](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/a066ba0057192126bd5d255f37138265fc1a435e/config.json).
- **Gemma E2B/E4B:** full resident BF16 tensors are 10,246,357,958 and
  15,992,314,836 bytes. This refines the audit's Hub count with 928 additional
  stored BF16 calibration scalars per checkpoint. PLE remains GPU-resident
  BF16 (4,697,620,480 / 5,637,144,576 bytes) independently of generic backbone
  precision. Text compute excludes PLE and inactive modality towers; lookups
  separately charge 17,920 / 21,504 bytes per token. Complete checkpoint
  residency and retained consumer projections are conservative loader bounds.
  [E2B pin](https://huggingface.co/google/gemma-4-E2B-it/tree/4adf15748de9e1a0097b6274cee9c15836c41c37),
  [E4B pin](https://huggingface.co/google/gemma-4-E4B-it/tree/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c).
- E2B has 35 executed layers but 15 KV producers (12 local/3 global); E4B has
  42 executed layers but 24 producers (20 local/4 global). Reads and attention
  still charge every consumer. E4B uses hidden2560, Q8/KV2, local256/global512
  head dimensions, and 35 sliding512 plus seven global layers. Shared pools
  conservatively remain resident on each PP stage. Their ASR aliases inherit
  both corrected geometry and exact artifacts. Gemma12B and its ASR alias now
  allow 262,144 positions, supported by both card and config.
  [Shared-KV implementation](https://github.com/huggingface/transformers/blob/cc1451df3b5b218295bc16803569256c53c9d6e2/src/transformers/models/gemma4/modeling_gemma4.py),
  [12B card](https://huggingface.co/google/gemma-4-12B-it/blob/43c9d014b231474fa7212dbb48a762566c2b255d/README.md).
- **Qwen3.5 small models:** exact native target subsets retain vision while
  excluding optional MTP. Loaders must omit the drafter until enabled; enabling
  MTP adds its separately pinned BF16 tensors. Acceptance remains a prior.

| Key | Target resident bytes | Optional MTP bytes | Pinned revision |
| --- | ---: | ---: | --- |
| q08 | 1,705,977,024 | 40,905,728 | `2fc06364715b967f1860aea9cf38778875588b17` |
| q2 | 4,426,488,512 | 121,656,320 | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| q4 | 9,078,538,752 | 241,199,104 | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| q9 | 18,819,635,168 | 486,581,248 | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

Sources: official Qwen/Qwen3.5-{0.8B,2B,4B,9B} repositories at those revisions;
for example the [2B tensor](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/model.safetensors).

## New local models and artifacts

| Entry | Exact source and tensor bytes | Serving assumptions |
| --- | --- | --- |
| Mistral Large4 FP8 | [pin](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/tree/af26e7869af601441d5cdf75a1272469ddc3d059), 1,060,548,427,328 | Full package includes vision and 4.20GB inactive DSpark. No speculative speedup. |
| Mistral Large4 NVFP4/FP8 | [pin](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4/tree/bb6d00fe249b0e876f0cffc6b85014be02de3e68), 610,030,457,408 | Mixed operators, retained BF16 islands; not uniform four-bit compute. |
| PPLX v2 contextual0.6B | [pin](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b/tree/6d1fddab0a18478acc0a51edd1e81333fe441095), 2,381,480,704 | FP32 weights; BF16 compute proxy. One full-document forward emits chunk vectors. |
| PPLX v2 contextual9B | [pin](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-9b/tree/dab89ff203fc943c4ca16e51c5882ff92790d758), 33,604,334,528 | 2048-dimensional FP32 arrays containing INT8-valued elements. |
| Granite470M TurboCTC multilingual | [pin](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/tree/91dcb0605daf65f88dbb04fef015c423241d941d), 946,249,856 | Encoder-only CTC; no generated tokens or transcript KV. |
| PPLX Decider v1.2 | [pin](https://huggingface.co/perplexity-ai/pplx-decider-v1.2/tree/cce0f3f80033e9d89d80831d5281fd72124aacce), 52,173,271,520 | Joint-question forward; noncausal global and causal linear layers. Separate readout included. |
| LFM3VL MLX4bit | [pin](https://huggingface.co/LiquidAI/LFM3-VL-2B-MLX-4bit/tree/d0d7edaf1b462161720898e46687c6fdaae6caee), 1,590,866,944 | Group32 affine INT4, INT8 tied embedding, BF16 vision. Requires Apple Silicon/MLX-VLM≥0.7.11. |
| Nanbeige4.3 FP8 | [pin](https://huggingface.co/Nanbeige/Nanbeige4.3-33B-FP8/tree/0b1acaa19bae10fec482600ada7c9defcb7c4884), 35,516,330,752 | E4M3, block128×128, BF16 scales/head/embeddings. |
| Nanbeige4.3 INT8 | [pin](https://huggingface.co/Nanbeige/Nanbeige4.3-33B-INT8/tree/879049b69c5f64384f3067095924b63e9125e84f), 35,992,669,888 | Symmetric group128 W8A16; distinct artifact beside FP8. |
| Nanbeige4.3 INT4 | [pin](https://huggingface.co/Nanbeige/Nanbeige4.3-33B-INT4/tree/d834b414cf1c47e42640de33c147b7cf8a9321ef), 22,550,315,872 | Asymmetric group32 W4A16, packed zero points. Never claims native FP4 arithmetic. |

MistralLarge4 uses 72 consumers, 63 unique MLA producers and 44 indexers. A single
compressed latent is replicated across TP. Its default cache bound is 83,840
bytes/token: all63 producers retain64 RoPE channels and index keys use a fixed
BF16 upper bound. This is 1.39% above the exact source producer RoPE schedule;
runtime FP8 cache layouts/scales, padding and kernel gains remain uncalibrated.
[Pinned implementation](https://github.com/huggingface/transformers/blob/c5bd76ec51a70c303be110cc04be3d78cf755299/src/transformers/models/mistral_large4/modeling_mistral_large4.py).

Contextual embeddings assume **512 formatted positions/chunk** as an explicit
workload default. Prefixes and separators consume positions. Full document work
is charged once, output vectors are counted per chunk, and overlength documents
are rejected rather than truncated. Packed INT8 vector storage is not assumed.

Granite's source-derived staged bound charges one block at50Hz, one at25Hz,
and fourteen at12.5Hz. The tied CTC projection is resident once and executed
twice. Feature extraction, softmax and exact buffers remain unmodeled.
Its quality plot uses the publisher's **5.51% English WER over nine short-form
Open ASR sets**, explicitly marked as an unofficial submission at October21.
Japanese CER is kept separate. Contextual embeddings without comparable quality
evidence use the existing capacity-only surface instead of a fabricated score.
PPLX Decider retains its262K positional ceiling separately from32K training
coverage; recursive splitting repeats state and changes joint semantics.

## Hosted pricing and billing

Added a separate `november_2026` gateway preset; the established `current` preset
and corporate allowlists are preserved. New hosted identities are Fable5.2,
Gemini4Carbon/Argon, GPT6.1Sol/Terra, GPT6Luna, Grok4.7 and MiniMaxM3.1Flash.
Their quality/verbosity values are explicitly predecessor/deployment-class
proxies, not new independent benchmark measurements.

- Enforce input, output and combined limits using **actual generated output**
  (`useful_output / token_efficiency`); keep useful units in the cost denominator.
- Select `cache_write_ttl` before long-context pricing. Fable supports5m/1h;
  OpenAI supports30m/12h. Unsupported explicit TTLs are excluded. Google4
  implicit-only caching rejects explicit write/storage workloads.
- Service token discounts no longer discount storage. Per-component maps model
  GeminiPro's unchanged Batch cache-read price and FlashLite's distinct Batch
  versus Flex cache prices. Negotiated scalar discounts and region modifiers
  remain separate; negotiated prices are not overwritten by public histories.
- Mistral current rates are **$0.70/$0.07/$2.10 per million input/cache/output**.
  Explicit dates preserve captured October6 and20 snapshots and the October29
  reset. Gemini4's December1 doubling is explicit and date-testable.
- DeepSeek date/hour pricing uses the published2026 Chinese holiday calendar;
  unknown years are flagged unless an explicit holiday override is supplied.
  Undated workloads retain the ordinary-week blend. Weekends stay off-peak.
- Gemini long-decode continuation/hop billing is not represented by ordinary
  synchronous requests. Kimi3.1 rates remain unpriced because official tables
  were empty. Mythos access and unsupported MiniMax service tiers are not assumed.

Sources: [OpenAI models/pricing](https://developers.openai.com/api/docs/pricing),
[OpenAI October20 cache update](https://developers.openai.com/api/docs/changelog),
[Anthropic](https://platform.claude.com/docs/en/about-claude/pricing),
[Google](https://aistudio.google.com/docs/pricing),
[Mistral October29](https://mistral.ai/news/mistral-large-4-oss/),
[xAI](https://docs.x.ai/developers/pricing),
[MiniMax native OpenAPI](https://platform.minimax.io/docs/api-reference/text/api/openapi-chat-openai.json),
[DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/),
[holiday notice](https://www.beijing.gov.cn/cs/gncs/zcwj/202603/t20260327_4568275.html),
[Cohere CommandA](https://docs.cohere.com/docs/command-a),
[Cohere R7B](https://docs.cohere.com/docs/command-r7b).

## Hardware snapshot

Six-R9700 Tinybox configurations use210W per GPU in both cases, matching the
[official configurator JavaScript](https://tinygrad.org/assets/tinybox.js?v=red-whale-2)
(captured November11, publication date unavailable; SHA256
`95886e5d8289c12622bca2238ee6f920e3f14b9bccdede1c0fc028385ce1a109`).
Prices and physical device boundaries remain unchanged. Compute remains an
uncalibrated power-scaled proxy; PSU capacity is not measured wall draw.

[Galaxy](https://tenstorrent.com/en/hardware/galaxy) acquisition cost is
$160,000/32 ASICs. [Intel's October12 release](https://www.intc.com/news-events/press-releases/detail/1783/intel-accelerates-open-infrastructure-for-the-agentic-ai-era)
updates D-series/D270 bandwidth to1.5TB/s. Its published1.3PF FP8 lacks resolved
dense/sparse semantics, so tensor rooflines remain conservative proxies. Sampling
does not establish a purchasable configuration. Rubin availability is refreshed;
its existing288GB/19.2TB/s/3TB/s specifications already matched.

## Representative before/after calculations

One request, 2048 input/128 output tokens,90% memory utilization,4GB non-KV.
Small models use one RTX4090; MiMo uses eight H200s. These are estimator outputs,
not runtime benchmarks. KV below is the selected artifact's cache format.

| Model | Resident GB before→after | KV at131072 positions, GB before→after | Estimated requests/s before→after |
| --- | --- | --- | --- |
| Gemma E2B BF16 | 4.000→10.246 | 0.954→0.812 | 1.33→1.15 |
| Gemma E4B BF16 | 8.000→15.992 | 18.254→2.168 | 0.64→0.56 |
| Qwen3.5 2B BF16 | 4.000→4.426 | 1.611→1.611 | 1.32→1.40 |
| Qwen3.5 4B BF16 | 8.000→9.079 | 4.295→4.295 | 0.65→0.62 |
| MiMo Flash MXFP4 | 172.923→172.923 | 1.218→1.523 | 11.00→11.00 |
| MiMo Pro MXFP4 | 566.028→566.028 | 2.700→3.375 | 4.82→4.82 |

Gemma's shared producers reduce cache despite larger full weights. Qwen2B's
text-active correction can raise estimated throughput while residency rises.
MiMo's short-workload throughput is unchanged after rounding; long-context
capacity correctly decreases. Gemma12B remains unable to fit BF16 on one4090
under these reservations. New entries have public planner/report roundtrip tests.

## Deferred evidence

Watch items remain deferred: runtime release speedups, unmeasured speculative
acceptance, Kimi hosted prices, FP4 cache formats, Apple maximum-memory quotes,
additional decision/speculator families, and broader carryover releases. No
closed-form estimate is relabeled as a benchmark. Production was not modified.
