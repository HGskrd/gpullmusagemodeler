"""StartLux decision checkpoints, pinned from official metadata on 2026-11-11."""

from typing import NamedTuple

from .model_class import (
    DECISION_PROFILES,
    MODEL_QUANTIZATION_PROFILES,
    DecisionProfile,
    Model,
    _artifact_profile,
)
from .model_sources import OPEN_MODEL_ARCHITECTURE_SOURCES, ModelArchitectureSource


class _Checkpoint(NamedTuple):
    size: str
    version: str
    variant: str
    revision: str
    params: int
    fp32_params: int = 0


# Physical safetensors headers, including vision and any stored MTP weights.
# AutoThink index total_size is stale; use the actual tensor payload sizes.
_CHECKPOINTS = (
    _Checkpoint(
        "0.8B", "v1.1", "Instant", "6496ae641cc71b149ce9dad69bd97f2a44e7bf11", 873438784, 2592
    ),
    _Checkpoint(
        "2B", "v1.1", "Instant", "f1e6cd147dda82c9ffa3e8e725d5c1f175566fb1", 2274069824, 2592
    ),
    _Checkpoint(
        "4B", "v1.1", "Instant", "416088f77cfce09633906a3675c7cd2e5dabcf10", 4659865088, 3840
    ),
    _Checkpoint(
        "9B", "v1.1", "Instant", "18a17f734966b213b2f00fa9c915e8b9af2a9b20", 9653104368, 3840
    ),
    _Checkpoint("27B", "v1.1", "Instant", "32e7d35d74db615c4a059181efbf21c5421f3561", 27781427952),
    _Checkpoint("4B", "v1.1", "AutoThink", "23dd4c513ef387705bb13fd8caadc2d5c863082f", 4539265536),
    _Checkpoint("9B", "v1.1", "AutoThink", "efc649d8493b5b7353cf2ed6da0cdcbc215958ef", 9409813744),
    _Checkpoint(
        "27B", "v1.1", "AutoThink", "dd62d5cd84a483d75bdd548880bbf08bdd3f67a4", 27356728560
    ),
    _Checkpoint("0.8B", "v1", "", "56996b552616bb21990f8ed7b473a7d1ed636822", 873438784, 2592),
    _Checkpoint("2B", "v1", "", "e6f8d5b4bedffa2df82bf3e11ff6eb7c7495b3c2", 2274069824, 2592),
    _Checkpoint("4B", "v1", "", "9c0a32e9809d6a3f612434dd0bce150295dcc274", 4659865088, 3840),
    _Checkpoint("9B", "v1", "", "c030630a94f8146c4d08e60f4da1b168e17f2e06", 9653104368, 3840),
    _Checkpoint("27B", "v1", "", "5de3f54a7e0300b986d7cccbd3bf1cd59085dbac", 27781427952),
    _Checkpoint("35B-A3B", "v1", "", "81e59afd1a05942444bf08805c1f258611b8cdac", 35951822704),
)

# layers, hidden width, attention heads, KV heads, linear value heads
_GEOMETRY = {
    "0.8B": (24, 1024, 8, 2, 16),
    "2B": (24, 2048, 8, 2, 16),
    "4B": (32, 2560, 16, 4, 32),
    "9B": (32, 4096, 16, 4, 32),
    "27B": (64, 5120, 24, 4, 48),
    "35B-A3B": (40, 2048, 16, 2, 32),
}

STARTLUX_MODELS: dict[str, Model] = {}
for _checkpoint in _CHECKPOINTS:
    _suffix = f"{_checkpoint.version}-{_checkpoint.size}"
    if _checkpoint.variant:
        _suffix += f"-{_checkpoint.variant}"
    _key = f"startlux-decision-{_suffix.lower()}"
    _repo = f"startlux-models/StartLux-Decision-{_suffix}"
    _source = f"https://huggingface.co/{_repo}"
    _layers, _hidden, _heads, _kv_heads, _linear_heads = _GEOMETRY[_checkpoint.size]
    _instant = _checkpoint.variant == "Instant"
    _autothink = _checkpoint.variant == "AutoThink"
    _moe = _checkpoint.size == "35B-A3B"
    # Native positional ceiling is 262,144; current Instant engine stops at 131,072.
    _cap = 131072 if _instant else 262144
    _mode_note = (
        "Instant requires StartLux bidirectional full-attention support; linear layers stay causal. "
        "Serving cap 131,072 despite native 262,144 positions; no prefix/chunked-prefill credit. "
        if _instant
        else "Causal attention; native/serving ceiling 262,144 on vLLM 0.31. "
    )
    if _autothink:
        _mode_note += (
            "Estimates require think_mode=never. Auto/always reasoning, gate frequency, thought "
            "tokens and second readout latency are unmodeled. "
        )
    if _moe:
        _mode_note += "Rounded 3B active-parameter proxy; expert routing uncalibrated. "
    _note = (
        _mode_note
        + "One complete prompt forward per question, at most 26 listed options; group rounds "
        "unmodeled. Text-only closed-form estimate, not a benchmark; vision and stored MTP "
        "weights stay resident, encoder execution and MTP speedups unmodeled. Native mixed "
        "BF16/FP32 tensor bytes retained; other precisions are hypothetical conversions. "
        "Quality uncalibrated; no generation economics. Weights: CC BY-NC 4.0; commercial "
        "use requires a separate StartLux license."
    )
    _name = f"StartLux-Decision {_suffix.replace('-', ' ')}"
    if _autothink:
        _name += " (think_mode=never)"
    STARTLUX_MODELS[_key] = Model(
        key=_key,
        name=_name,
        cat="StartLux",
        color="#7C3AED",
        total_params=_checkpoint.params,
        active_params=3e9 if _moe else _checkpoint.params,
        is_moe=_moe,
        layers=_layers,
        hidden_dim=_hidden,
        num_heads=_heads,
        kv_heads=_kv_heads,
        head_dim=256,
        is_mla=False,
        attention_layers=_layers // 4,
        kv_layers=_layers // 4,
        linear_attention_layers=_layers * 3 // 4,
        linear_attention_heads=_linear_heads,
        linear_attention_head_dim=128,
        linear_attention_k_heads=16,
        linear_attention_k_head_dim=128,
        linear_attention_conv_kernel=4,
        moe_routed_experts=256 if _moe else 0,
        moe_active_experts=8 if _moe else 0,
        max_context_tokens=_cap,
        quality_confidence=0.0,
        capabilities_override=frozenset(),
        attention_label=(
            f"{_layers // 4} {'noncausal' if _instant else 'causal'} full attention + "
            f"{_layers * 3 // 4} causal linear · one forward/question"
        ),
        native_precision_label="Native BF16/FP32" if _checkpoint.fp32_params else "Native BF16",
        architecture_note=_note,
    )
    DECISION_PROFILES[_key] = DecisionProfile(
        _cap,
        f"{_source}/blob/{_checkpoint.revision}/README.md",
        _note,
        causal_attention=not _instant,
        forward_per_question=True,
        max_options=26,
    )
    OPEN_MODEL_ARCHITECTURE_SOURCES[_key] = ModelArchitectureSource(
        _source,
        _checkpoint.revision,
        "Captured 2026-11-11: official card/config and physical safetensors headers. "
        "AutoThink shard-index total_size is stale; header payloads determine residency. " + _note,
    )
    MODEL_QUANTIZATION_PROFILES.update(
        [
            _artifact_profile(
                model_key=_key,
                precision_key="bf16",
                label="Native checkpoint (vision/MTP included)",
                source_repo=_repo,
                source_revision=_checkpoint.revision,
                total_weight_bytes=2 * (_checkpoint.params + _checkpoint.fp32_params),
                storage_format_counts={
                    "BF16": _checkpoint.params - _checkpoint.fp32_params,
                    "F32": _checkpoint.fp32_params,
                },
                compute_precision_shares={"bf16": 1.0},
                quantized=(),
                retained=(
                    "Entire physical checkpoint including vision, stored MTP and FP32 tensors.",
                ),
                notes="Pinned physical tensor payload bytes, not stale index metadata. "
                "BF16 compute uses the existing uncalibrated text-only proxy. " + _mode_note,
                quant_algo="Unquantized",
                kv_cache_format="BF16",
                captured_at="2026-11-11",
            )
        ]
    )
