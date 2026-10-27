# Open-weight additions — 27 October 2026

This batch adds 47 entries: 13 text-generating modes, 14 embeddings, 12 ASR modes,
and eight decision models. The text modes include two Granite Switch chat aliases;
the ASR modes include separate Qwen offline/streaming workload profiles. All modes
are independently allocated; the planner does not credit cross-assignment model-pool sharing.
Previously landed LFM3 work is separate from this batch.

The selector update also restores Mistral Small 3.2 and hides superseded choices
while retaining their keys for saved scenarios. The restored entry has no retained
comparable AA anchor, so quality confidence is explicitly zero.

## Evidence and limits

Every entry pins an official checkpoint or upstream reference manifest. Native sizes
use safetensors tensor headers, official index metadata or checkpoint ZIP metadata;
no weight tensor payload was downloaded. Model parameters, packed tensor elements,
checkpoint storage and runtime VRAM are separate quantities. Nominal active counts
remain labeled publisher estimates where exact active operator accounting is unavailable.

All capacity results are closed-form estimates, not benchmarks. Quality confidence
and token efficiency remain uncalibrated for new text models. No speculative speedup
is credited for the optional MiniMax DSpark or Ling MTP checkpoints; these are excluded
from target-only residency. MiniMax's optional drafter is 2.302 GB; Ling's is 8.212 GB.

Visual models include full resident towers but their throughput is a text-input proxy.
Image/video preprocessing, visual token expansion and tower execution are unmodeled.
Jina Code weights are CC-BY-NC-4.0. Other license names appear beside the exact source;
open weights do not imply identical commercial terms or runtime support.

## Catalog

GB below are decimal checkpoint storage, not per-GPU memory. A runtime cast can change
resident bytes independently. Immutable source links identify the reviewed revisions.

| Kind | Key | Parameters (B total / active) | Checkpoint GB | License | Pinned source |
|---|---|---:|---:|---|---|
| embedding | `qwen3-embedding-0.6b` | 0.5958 / 0.5958 | 1.192 | apache-2.0 | [97b0c614be4d](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) |
| embedding | `qwen3-embedding-4b` | 4.0218 / 4.0218 | 8.044 | apache-2.0 | [5cf2132abc99](https://huggingface.co/Qwen/Qwen3-Embedding-4B/tree/5cf2132abc99cad020ac570b19d031efec650f2b) |
| embedding | `qwen3-embedding-8b` | 7.5673 / 7.5673 | 15.135 | apache-2.0 | [1d8ad4ca9b3d](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af) |
| embedding | `nomic-embed-text-v2-moe` | 0.4753 / 0.3054 | 1.901 | apache-2.0 | [1066b6599d09](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe/tree/1066b6599d099fbb93dfcb64f9c37a7c9e503e85) |
| embedding | `harrier-oss-v1-270m` | 0.2681 / 0.2681 | 0.536 | mit | [31de22b67391](https://huggingface.co/microsoft/harrier-oss-v1-270m/tree/31de22b673913c7d658c0f03f792d77c2dcf8ebd) |
| embedding | `harrier-oss-v1-0.6b` | 0.5960 / 0.5960 | 1.192 | mit | [f9b9dc8d367d](https://huggingface.co/microsoft/harrier-oss-v1-0.6b/tree/f9b9dc8d367d443f2479d27aa5d8d2850c0774ee) |
| embedding | `harrier-oss-v1-27b` | 27.0093 / 27.0093 | 54.019 | mit | [0c0fc62f6d8a](https://huggingface.co/microsoft/harrier-oss-v1-27b/tree/0c0fc62f6d8af9e8604cb818c412301b103a0093) |
| embedding | `jina-code-embeddings-0.5b` | 0.4940 / 0.4940 | 0.988 | cc-by-nc-4.0 | [4db235132daf](https://huggingface.co/jinaai/jina-code-embeddings-0.5b/tree/4db235132dafbe56a8b9c5f59b59795ecf58a4a7) |
| embedding | `jina-code-embeddings-1.5b` | 1.5437 / 1.5437 | 3.087 | cc-by-nc-4.0 | [39aeb4fb9b60](https://huggingface.co/jinaai/jina-code-embeddings-1.5b/tree/39aeb4fb9b60f930934c78ae5d749a46287c248a) |
| embedding | `qwen3-vl-embedding-2b` | 2.1275 / 1.7206 | 4.255 | apache-2.0 | [9f2f7e710d6d](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/tree/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda) |
| embedding | `qwen3-vl-embedding-8b` | 8.1448 / 7.5684 | 16.290 | apache-2.0 | [2c4565515e0f](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/tree/2c4565515e0f265c6511776e7193b22c0968ddc7) |
| embedding | `vultronretrieverflash-qwen3.5-0.8b` | 0.8533 / 0.7527 | 1.707 | apache-2.0 | [6f6b6757e9ea](https://huggingface.co/vultr/VultronRetrieverFlash-Qwen3.5-0.8B/tree/6f6b6757e9ea3ad93f4f2cbd839bc6f2f0b93739) |
| embedding | `vultronretrievercore-qwen3.5-4.5b` | 4.5401 / 4.2066 | 9.080 | apache-2.0 | [b0857609ad6d](https://huggingface.co/vultr/VultronRetrieverCore-Qwen3.5-4.5B/tree/b0857609ad6db741009a7026e02cf30bd384f54d) |
| embedding | `vultronretrieverprime-qwen3.5-8b` | 8.3940 / 7.9380 | 16.788 | apache-2.0 | [ca8bc3c904d8](https://huggingface.co/vultr/VultronRetrieverPrime-Qwen3.5-8B/tree/ca8bc3c904d845761a942965e4c27734a3e45ac1) |
| decision | `d3` | 26.0866 / 25.6259 | 52.176 | apache-2.0 | [20f37b9a892a](https://huggingface.co/vllm-sr/d3/tree/20f37b9a892acc13d4b07a4f3c4f11b2518c615c) |
| decision | `d3-flash` | 8.3937 / 7.9377 | 16.790 | apache-2.0 | [13dc1cc8ef23](https://huggingface.co/vllm-sr/d3-flash/tree/13dc1cc8ef23caefa765907e1d764751e5629f7d) |
| decision | `julia-1` | 0.1443 / 0.1443 | 0.577 | apache-2.0 | [a85b127321d5](https://huggingface.co/SupersonicLabs/Julia-1/tree/a85b127321d580d65176c89ced8273f305745d85) |
| decision | `laya-multilingual` | 0.3219 / 0.3219 | 0.644 | apache-2.0 | [1720e3e3357c](https://huggingface.co/convaiinnovations/laya-multilingual/tree/1720e3e3357cfe1e281542e223f8273b0890ca34) |
| decision | `strands-decider-2b-qwen35-v1` | 1.9354 / 1.9354 | 3.978 | apache-2.0 | [216d1ccb46a9](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/tree/216d1ccb46a9e9b036eb9187b248db98480f77d4) |
| decision | `kev-4b` | 4.2071 / 4.2071 | 8.417 | apache-2.0 | [6cfce5c2fa4b](https://huggingface.co/jaredpalmer/kev-4b/tree/6cfce5c2fa4b4bd64026336ab649c5ca78857d52) |
| decision | `granite-switch-4.1-8b-decision` | 8.7764 / 8.5785 | 17.554 | apache-2.0 | [83a46a256ecd](https://huggingface.co/ibm-granite/granite-switch-4.1-8b/tree/83a46a256ecda57e9e89e0f688b7c4ed20f35713) |
| llm | `granite-switch-4.1-8b` | 8.7764 / 8.5785 | 17.554 | apache-2.0 | [83a46a256ecd](https://huggingface.co/ibm-granite/granite-switch-4.1-8b/tree/83a46a256ecda57e9e89e0f688b7c4ed20f35713) |
| decision | `granite-switch-4.1-30b-decision` | 29.9898 / 29.4278 | 59.981 | apache-2.0 | [d0a8805f1c03](https://huggingface.co/ibm-granite/granite-switch-4.1-30b/tree/d0a8805f1c0348f17c0ccb8f7352f149b80103b4) |
| llm | `granite-switch-4.1-30b` | 29.9898 / 29.4278 | 59.981 | apache-2.0 | [d0a8805f1c03](https://huggingface.co/ibm-granite/granite-switch-4.1-30b/tree/d0a8805f1c0348f17c0ccb8f7352f149b80103b4) |
| asr | `qwen3-asr-0.6b` | 0.9380 / 0.9380 | 1.876 | apache-2.0 | [5eb144179a02](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) |
| asr | `qwen3-asr-0.6b-streaming` | 0.9380 / 0.9380 | 1.876 | apache-2.0 | [5eb144179a02](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) |
| asr | `qwen3-asr-1.7b` | 2.3492 / 2.3492 | 4.698 | apache-2.0 | [7278e1e70fe2](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) |
| asr | `qwen3-asr-1.7b-streaming` | 2.3492 / 2.3492 | 4.698 | apache-2.0 | [7278e1e70fe2](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) |
| asr | `granite-speech-5.0-2b` | 2.3132 / 2.3132 | 4.626 | apache-2.0 | [c684620bb54f](https://huggingface.co/ibm-granite/granite-speech-5.0-2b/tree/c684620bb54fc89d14ba27256b23bbc9e09ba764) |
| asr | `whisper-large-v3` | 1.5435 / 1.5435 | 3.087 | apache-2.0 | [06f233fe06e7](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1) |
| asr | `whisper-large-v3-turbo` | 0.8089 / 0.8089 | 1.618 | mit | [41f01f3fe87f](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9) |
| asr | `moss-transcribe-diarize-0.9b` | 0.9085 / 0.9085 | 1.817 | apache-2.0 | [704aa4a9c304](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/tree/704aa4a9c304e8520be88901e0d1960158ef5b15) |
| asr | `ark-asr-0.6b` | 1.2995 / 1.2995 | 2.599 | apache-2.0 | [45776b56d58c](https://huggingface.co/Edge0/ARK-ASR-0.6B/tree/45776b56d58cdfb2e2eb632f7e110f38684633e0) |
| asr | `ark-asr-3b` | 4.0634 / 4.0634 | 8.127 | apache-2.0 | [1e28271b79ed](https://huggingface.co/Edge0/ARK-ASR-3B/tree/1e28271b79edc97635783bea65abc89195a09ed3) |
| asr | `meta-omniasr-ctc-300m-v2` | 0.3260 / 0.3260 | 1.304 | Apache-2.0 | [81f51e224ce9](https://github.com/facebookresearch/omnilingual-asr/tree/81f51e224ce9e74b02cc2a3eaf21b2d91d743455) |
| asr | `meta-omniasr-llm-7b-v2` | 7.8049 / 7.8049 | 31.220 | Apache-2.0 | [81f51e224ce9](https://github.com/facebookresearch/omnilingual-asr/tree/81f51e224ce9e74b02cc2a3eaf21b2d91d743455) |
| llm | `minimax-m3.1-flash` | 427.0519 / 23.0000 | 250.103 | minimax-community | [262ae6830934](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/262ae6830934f3512211081b8a3f16425b93a239) |
| llm | `nanbeige43-33b` | 33.1190 / 33.1190 | 66.238 | apache-2.0 | [84c54883993f](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/tree/84c54883993f9053b058f95ddc8f591b203e92bf) |
| llm | `nemotron35-bolt-e4b` | 7.9962 / 4.0000 | 15.992 | openmdw-1.1 | [49843eed8db1](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16/tree/49843eed8db1f1c616c47af75cfd36e07aec0667) |
| llm | `gpt-oss-20b` | 20.9148 / 3.6000 | 13.761 | apache-2.0 | [6cee5e81ee83](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee) |
| llm | `gpt-oss-120b` | 116.8292 / 5.1000 | 65.249 | apache-2.0 | [b5c939de8f75](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a) |
| llm | `recursive-flint-40b-a2.5b` | 39.7853 / 2.5000 | 79.571 | apache-2.0 | [2a247be0c0ae](https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288) |
| llm | `kolibri-1-78b-a3.46b` | 78.1031 / 3.4576 | 78.827 | apache-2.0 | [5bb7075ba22d](https://huggingface.co/Aleph-Alpha/Kolibri-1/tree/5bb7075ba22db733e3834b1c71d2245537cc14ed) |
| llm | `apertus15-8b` | 8.9035 / 8.6091 | 18.396 | apache-2.0 | [0be762a6a58d](https://huggingface.co/swiss-ai/Apertus-v1.5-8B/tree/0be762a6a58dad2a3b3f71b2e0f46d34339fa836) |
| llm | `apertus15-70b` | 72.0058 / 71.7114 | 144.601 | apache-2.0 | [18af3fe32946](https://huggingface.co/swiss-ai/Apertus-v1.5-70B/tree/18af3fe329469e7ed0060e82dc0229bdb99a3af0) |
| llm | `ling31-flash-560b-a25b` | 563.9465 / 25.0000 | 565.541 | mit | [276cca72bbd8](https://huggingface.co/inclusionAI/Ling-3.1-flash/tree/276cca72bbd8b95693032848bfd638460b529b8b) |
| llm | `dory1-4b-2loop` | 3.9752 / 3.9752 | 4.751 | openmdw-1.1 | [bba10abb88a8](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8/tree/bba10abb88a8940f85bb9ef2ae6814b156142ab4) |

## Architecture and workload assumptions

- Causal decoder embeddings use triangular prompt attention. Existing bidirectional
  embeddings retain rectangular attention. Pooling, output dimensions, Matryoshka
  serving limits and native FP32 Nomic storage remain explicit.
- Published text, code and visual retrieval scores keep their exact protocol labels.
  They are percentage-scaled publisher values, not a cross-benchmark normalization.
  Harrier has no compatible sourced retrieval aggregate and shows capacity only.
- ASR uses a representative 30-second chunk and assumed six transcript tokens/s;
  rich MOSS output uses 12. Streaming Qwen adds 2.5 rollback tokens/s and an eightfold
  accumulated-audio replay prior for a 30-second session. These are workload/implementation
  proxies, not measured latency or a streaming SLA. Longer recordings need a matching
  chunk/context policy; whole-recording state growth is not simulated.
- Speech rooflines split encoder frames, decoder audio prefill, transcript decoding,
  encoder attention and Whisper cross-attention. Residency includes each component once.
  Work buffers reserve activations, self-KV and cross-KV conservatively without buffer
  sharding. Convolution/projector FLOPs use a component-level approximation. Fused
  processing, overlap, rollback efficiency and kernel scheduling require calibration.
- Meta v2 native files are FP32. Its BF16 serving residency is an explicit conversion
  estimate. Laya's FP16 file expands into FP32 in its pinned reference loader; the model
  counts that runtime memory and traffic while using BF16 autocast compute.
- d3, Julia and Laya charge independent state encodings for each question. Strands/Kev
  use the same conservative independent-question fallback; their default runtime's
  shared prefix optimization is not credited. Decision models generate zero tokens.
- Julia/Laya include 22 encoder layers plus two decision-head attention layers:
  10 global and 14 local. Strands counts the text torso plus adapter/head dependencies;
  Kev's merged BF16 serving footprint is a derived conversion, not an exact released artifact.
- Granite Switch has 40/64 learned decoder layers; two additional config slots are
  analytic switch caches. Full adapter banks remain resident. Its external TurboCTC
  speech dependency is not silently included in text-only decisions or chat; a shared
  audio-enabled pool would additionally need 946,249,856 bytes plus ASR runtime memory.
- Bolt computes 42 attention layers but stores 24 unique KV producers (20 local/four
  global). Every attention consumer still incurs a conservative read. Cross-stage
  cache sharing conservatively reserves the shared producer pool on each PP stage
  until a runtime-specific placement contract is available. Its official Q4
  GGUF is 6,180,446,208 bytes and uses BF16 arithmetic as a W4A16 proxy. It occupies the
  planner's existing four-bit artifact bucket but is explicitly encoded Q4_K_M/Q6_K/Q8_0,
  not native MXFP4. The advertised 4GB envelope needs table offload/workload assumptions
  that are not credited. The BF16 artifact additionally includes modality towers.
- Dory uses its default two loops and per-loop KV: 56 executed blocks, 42 local/14 global,
  with 36 physical blocks stored. Parameter work/rereads multiply by
  `1 + 1,835,560,960 / 3,975,229,440`; cache geometry is independent of that multiplier.
  W8A16 arithmetic is charged as BF16. Last-loop cache optimization is uncredited.
  Runtime support is documented in upstream PRs, not asserted as a stable release.
- Kolibri and Ling initially use the pinned 262,144-position config limit. Card-level
  1M claims are not substituted for a validated extension. Flint uses its documented
  256K envelope. Apertus includes expanded vocabulary plus FP32 modality tokenizers;
  gated official metadata was checked against copies with identical Git blob identities.
- MiniMax uses all-layer sparse attention with full-context KV residency and bounded
  selected reads. Packed expert and attention storage is exact; active compute shares
  use routed-expert geometry and a conservative BF16 prior for the remaining operators.
  gpt-oss similarly separates MXFP4 experts from retained BF16 operators.

## Scores without a quality estimate

Capacity remains visible as a bar chart when only unrated checkpoints are selected,
or as a separate text readout beside rated models. No invented WER or retrieval score
is placed on a quality axis. Meta v2 publishes CER evidence and is not relabeled WER;
Whisper Turbo has no verified checkpoint-specific WER in this review. ASR protocols
retain their dataset, language, offline/streaming mode, date and primary URL. Older
leaderboard aggregates are not directly compared with the changed October suite.

## Watchlist and scope

Mistral Large 4 and Kimi K3.1 Pro remain pending weight delivery in the reviewed
primary announcements. Qwen-Audio-3.1-ASR has no verified public checkpoint here.
Rerankers, TTS and image/video generation remain separate workload extensions.

## Verification

The focused release tests exercise all 47 entries through the public calculation API,
plus numerical invariants for causal attention, shared KV producers/consumers, repeated
parameter work, reference-loader storage casts, cross-attention, streaming replay and
per-question forwards. The repository's compile, Ruff, mypy and pytest matrix is required
against the final tree. Browser acceptance covers blank-scenario setup, H200 selection,
all four picker tabs, assignment, artifact selection, rated quality plots and unrated
capacity plots. Export/import preserves the selected GGUF artifact.

No production deployment is included in this change.


### Existing-model controls before and after

These are planner outputs on H200, not hardware measurements. All four controls use
four requests/users; embedding/decision inputs are 2,048 tokens, while text uses the
existing Chat distribution and FP8. The other controls use BF16.

| Control | Before | After |
|---|---:|---:|
| Qwen3.8 27B decode tokens/s | 381 | 381 |
| Nemotron3 Embed1B documents/s | 121.01 | 121.01 |
| Liquid d1 3B requests/s | 51.4133548 | 51.4133548 |
| Voxtral realtime factor at four users | 22.2 | 22.2 |

Desktop browser acceptance passed for all four picker tabs, native assignments,
MiniMax and d3 charts, sourced Qwen embedding/ASR plots, and capacity-only Harrier/
Whisper Turbo plots. The selected Bolt GGUF artifact survived export, server restart
and scenario import. Speech cards use chunk/work-buffer labels; the technical summary
shows input-token/audio-second units for native modalities even without generation demand.
The attempted narrow-viewport check could not complete because the collaborative browser
resize automation timed out; no mobile-layout acceptance is claimed.

### Final checkout results

- Compile, Ruff lint/format, mypy and JavaScript syntax checks pass.
- Final local pytest run: **704 passed, 1,282 subtests passed** (including the
  separate benchmark-export work present in the checkout).
- The restored `ms32` entry explicitly records its missing comparable quality
  anchor instead of inheriting full-confidence defaults.
- The characterization fixture changes only by adding six copies of four new
  model metadata fields; no existing numerical/control value changes.
- Owned local acceptance server stopped; production deployment is outside this delivery.
