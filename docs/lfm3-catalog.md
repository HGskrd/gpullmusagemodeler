# LFM3 catalog snapshot

Sources checked 2026-10-27: [official launch](https://www.liquid.ai/blog/lfm3),
official model cards/configs, and safetensors headers at the immutable revisions
in `data/model_sources.py`. No weights were downloaded. Tensor totals include
all modality components, rather than the rounded marketing model size.

| Checkpoint | Parameters counted | Native resident tensor bytes | Context limit |
|---|---:|---:|---:|
| LFM3 1.7B / Base | 1,728,801,536 | 3,457,603,072 | 262,144 |
| LFM3 VL 2B / d1 2B | 1,918,181,888 | 3,836,363,776 | 128,000 |
| LFM3 Audio 2B | 2,075,892,018 | 4,151,786,596 | 32,768 |
| LFM3 Omni 2B | 2,240,487,474 | 4,480,977,508 | 32,768 |
| d1 Omni 2B | 2,166,928,384 | 4,333,856,768 | 128,000 |

Audio and Omni include the separate `audio_tokenizer/model.safetensors`:
73,557,810 BF16 and 1,280 FP32 elements, totaling 147,120,740 bytes. The decision
checkpoints do not require a speech-output codec. Native storage excludes loader
workspace, runtime memory, activations and KV, which the planner treats separately.
The Audio/Omni cards' 32K limit takes precedence over their 128,000-position configs;
32K is interpreted as 32,768 tokens. VL/d1 use their exact config ceiling of 128,000.

All backbones have 24 layers, hidden size 2,048, 16 query heads, four KV heads and
128-dimensional heads. Eleven layers retain KV: six global and five with a
256-token sliding window. The remaining 13 are gated short convolutions, kernel
size seven. Their cache reserves seven values per channel (372,736 BF16 bytes per
sequence), conservatively covering implementations retaining six or seven. No
recurrent matrix is invented. Convolution state follows the independent state
precision convention; head-aligned TP sharding and proportional PP placement are
planning approximations.

The text model also has two official mixed-precision artifacts:

| Artifact | Revision | Tensor bytes | Retained operators |
|---|---|---:|---|
| [FP8](https://huggingface.co/LiquidAI/LFM3-1.7B-FP8/blob/243955da95b1cd68026f293b29de4b4c13f3638f/config.json) | `243955da95b1cd68026f293b29de4b4c13f3638f` | 2,209,949,952 | BF16 convolution, gates, norms, embeddings |
| [NVFP4](https://huggingface.co/LiquidAI/LFM3-1.7B-NVFP4/blob/b30025bbbbcbf62f6c110b40d51d36af6579f6a2/config.json) | `b30025bbbbcbf62f6c110b40d51d36af6579f6a2` | 1,829,689,408 | BF16 attention, convolution, norms, embeddings |

FP8 uses 128×128 weight blocks; NVFP4 quantizes FFN linears with group size 16.
Both totals include scales and keep BF16 KV/convolution caches. Compute shares
come from quantized versus retained parameter counts, not measured kernel time.
Unpinned formats remain explicitly labeled conversion estimates.

Audio/Omni remain audio-capable LLMs with text-token throughput estimates, rather
than receiving an invented ASR/voice capacity calibration. Full checkpoint weights
are charged as a conservative text-compute proxy. Image/audio encoding, video,
speech generation, codec latency and multimodal activation memory are unmodeled.
d1 entries use the Decision tab and single-forward request units with zero generated
tokens. Their `d1_attention="set"` mask is conservatively approximated as noncausal
attention; detailed question-set sharing is not simulated. Quality and token
efficiency are uncalibrated placeholders throughout this new family.
