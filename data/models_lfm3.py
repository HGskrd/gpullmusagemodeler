"""LFM3 launch family, pinned on 2026-10-27; see docs/lfm3-catalog.md."""

from .model_class import (
    DECISION_PROFILES,
    MODEL_QUANTIZATION_PROFILES,
    DecisionProfile,
    Model,
    _artifact_profile,
)
from .model_sources import OPEN_MODEL_ARCHITECTURE_SOURCES

LFM3_CAPTURED_AT = "2026-10-27"
LFM3_CODEC_PARAMS = 73_559_090
LFM3_CODEC_BYTES = 147_120_740


def _lfm3_model(
    key: str,
    name: str,
    color: str,
    params: int,
    context: int,
    capabilities: frozenset[str],
    *,
    codec: bool = False,
    decision: bool = False,
    note: str = "",
) -> Model:
    source = OPEN_MODEL_ARCHITECTURE_SOURCES[key]
    # Hub's top-level parameter total excludes the separate LiquidCodec. Include
    # it for speech-generating models; d1 never generates speech or text tokens.
    total = params + (LFM3_CODEC_PARAMS if codec else 0)
    weight_bytes = params * 2 + (LFM3_CODEC_BYTES if codec else 0)
    caveat = (
        "Text-only throughput proxy; full checkpoint residency included. "
        "Vision/audio encoding, speech generation and codec latency are not simulated. "
        "Quality and token efficiency are uncalibrated placeholders. "
        "Throughput is a closed-form estimate, not a measured benchmark. "
        "Convolution cache reserves seven positions per channel; TP/PP placement is approximate. "
    ) + note
    if decision:
        DECISION_PROFILES[key] = DecisionProfile(
            context,
            f"{source.repository}/blob/{source.revision}/README.md",
            "Typed single-forward decisions. Noncausal set mask approximated by full "
            "bidirectional attention; question-set sharing and multimodal latency unmodeled.",
            causal_attention=False,
        )
    MODEL_QUANTIZATION_PROFILES.update(
        [
            _artifact_profile(
                model_key=key,
                precision_key="bf16",
                label="Official BF16" + (" + LiquidCodec" if codec else ""),
                source_repo=source.repository.removeprefix("https://huggingface.co/"),
                source_revision=source.revision,
                total_weight_bytes=weight_bytes,
                compute_precision_shares={"bf16": 1.0},
                quantized=(),
                retained=("Entire native checkpoint, including modality components.",),
                notes="Exact pinned tensor-header inventory. "
                + ("Includes LiquidCodec BF16 weights and 1,280 FP32 elements. " if codec else "")
                + "Other unpinned precisions are conversion estimates, not verified artifacts.",
                quant_algo="Unquantized",
                kv_cache_format="BF16",
                captured_at=LFM3_CAPTURED_AT,
            )
        ]
    )
    return Model(
        key=key,
        name=name,
        cat="LiquidAI" if decision else "LFM",
        color=color,
        total_params=total,
        active_params=total,
        prefill_active_params=total,
        is_moe=False,
        layers=24,
        num_heads=16,
        kv_heads=4,
        head_dim=128,
        is_mla=False,
        hidden_dim=2048,
        attention_layers=11,
        kv_layers=11,
        local_attention_layers=5,
        local_attention_window=256,
        short_conv_state_elements=13 * 2048 * 7,
        attention_label="13 gated conv (k=7) + 5 SWA (256) + 6 global GQA",
        max_context_tokens=context,
        capabilities_override=capabilities,
        quality_confidence=0.0,
        native_precision_label="Official BF16" + (" + LiquidCodec" if codec else ""),
        architecture_note=caveat,
    )


LFM3_MODELS = {
    "lfm3-1.7b": _lfm3_model(
        "lfm3-1.7b",
        "LFM3 1.7B",
        "#087F9C",
        1_728_801_536,
        262144,
        frozenset({"tools", "reasoning"}),
    ),
    "lfm3-1.7b-base": _lfm3_model(
        "lfm3-1.7b-base",
        "LFM3 1.7B Base",
        "#0F766E",
        1_728_801_536,
        262144,
        frozenset(),
        note="Pre-trained base for specialization; use the post-trained model for agents.",
    ),
    "lfm3-vl-2b": _lfm3_model(
        "lfm3-vl-2b",
        "LFM3 VL 2B",
        "#155E75",
        1_918_181_888,
        128000,
        frozenset({"tools", "images", "reasoning"}),
    ),
    "lfm3-audio-2b": _lfm3_model(
        "lfm3-audio-2b",
        "LFM3 Audio 2B",
        "#7C3AED",
        2_002_332_928,
        32768,
        frozenset({"tools", "audio", "reasoning"}),
        codec=True,
        note="32K card limit overrides the larger config ceiling. Audio capability is not a calibrated ASR capacity profile.",
    ),
    "lfm3-omni-2b": _lfm3_model(
        "lfm3-omni-2b",
        "LFM3 Omni 2B",
        "#A21CAF",
        2_166_928_384,
        32768,
        frozenset({"tools", "images", "audio"}),
        codec=True,
        note="Non-reasoning model. 32K card limit overrides the larger config ceiling. Audio capability is not a calibrated ASR capacity profile.",
    ),
    "d1-2b": _lfm3_model(
        "d1-2b",
        "Liquid d1 2B",
        "#0D9488",
        1_918_181_888,
        128000,
        frozenset(),
        decision=True,
        note="Single-forward typed decisions; no generation economics.",
    ),
    "d1-omni-2b": _lfm3_model(
        "d1-omni-2b",
        "Liquid d1 Omni 2B",
        "#9333EA",
        2_166_928_384,
        128000,
        frozenset(),
        decision=True,
        note="Single-forward typed decisions; no generation economics.",
    ),
}

# Physical storage is exact; operator shares are dense parameter-count proxies,
# not measured kernel time. Both releases retain substantial BF16 islands.
for _precision, _repo, _revision, _bytes, _quantized_params, _counts, _group in (
    (
        "fp8",
        "LFM3-1.7B-FP8",
        "243955da95b1cd68026f293b29de4b4c13f3638f",
        2_209_949_952,
        1_247_805_440,
        {"BF16": 481_072_256, "F8_E4M3": 1_247_805_440},
        128,
    ),
    (
        "nvfp4",
        "LFM3-1.7B-NVFP4",
        "b30025bbbbcbf62f6c110b40d51d36af6579f6a2",
        1_829_689_408,
        1_132_462_080,
        {"BF16": 596_339_456, "F8_E4M3": 70_778_880, "F32": 144, "U8": 566_231_040},
        16,
    ),
):
    _share = _quantized_params / 1_728_801_536
    MODEL_QUANTIZATION_PROFILES.update(
        [
            _artifact_profile(
                model_key="lfm3-1.7b",
                precision_key=_precision,
                label="Official " + _precision.upper(),
                source_repo="LiquidAI/" + _repo,
                source_revision=_revision,
                total_weight_bytes=_bytes,
                compute_precision_shares={_precision: _share, "bf16": 1 - _share},
                quantized=(
                    "FFN linears",
                    "FP8 additionally quantizes Q/K/V/output attention projections",
                ),
                retained=(
                    "Embeddings, convolution operators, norms and gates BF16; NVFP4 retains all attention BF16.",
                ),
                notes="Pinned config and exact tensor-header storage, including scales. Compute shares are uncalibrated dense parameter-count proxies. KV and convolution cache remain BF16.",
                quant_algo="block FP8" if _precision == "fp8" else "ModelOpt NVFP4",
                kv_cache_format="BF16",
                group_size=_group,
                storage_format_counts=_counts,
                captured_at=LFM3_CAPTURED_AT,
            )
        ]
    )
