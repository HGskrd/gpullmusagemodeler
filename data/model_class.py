"""Model, workload-profile, and quantization-profile domain types."""

import math
from dataclasses import dataclass

from .model_archive import ARCHIVED_MODELS
from .specs import (
    PRECISION_SPECS,
    QuantizationProfile,
    bytes_per_param,
    kv_cache_bytes_per_elem,
    normalize_precision,
)


@dataclass(frozen=True)
class RealtimeProfile:
    label: str
    tokens_per_second: float
    audio_ms_per_token: float
    target_delay_ms: int
    state_tokens: int
    source: str
    note: str
    audio_encoder_params: float = 0.0
    audio_tokens_per_step: int = 1
    audio_attention_layers: int = 0
    audio_attention_heads: int = 0
    audio_attention_head_dim: int = 0
    audio_attention_window: int = 0
    streaming: bool = True


@dataclass(frozen=True)
class EmbeddingProfile:
    label: str
    kind: str  # "single", "late", or "hybrid"
    output_dim: int
    max_sequence_length: int
    source: str
    note: str
    late_interaction_dim: int = 0
    query_length: int = 0
    document_length: int = 0
    vector_bytes_per_elem: float = 4.0
    storage_format: str = "FP32"
    pooling: str = ""

    @property
    def supports_single_vector(self) -> bool:
        return self.kind in {"single", "hybrid"}

    @property
    def supports_late_interaction(self) -> bool:
        return self.kind in {"late", "hybrid"}

    @property
    def mode_label(self) -> str:
        if self.kind == "hybrid":
            return "single-vector + late interaction"
        if self.kind == "late":
            return "late interaction"
        return "single-vector"


# Speculative decoding.  A SpeculativeProfile describes one drafter that can run
# against a model: a native MTP head trained into the checkpoint, an attachable
# speculator (EAGLE-3 / DFlash), or a training-free n-gram proposer.
# acceptance_alpha is the per-token acceptance probability; the planner derives the
# acceptance length with the chain formula tau = (1 - a^(k+1)) / (1 - a).  Real
# drafters verify trees or blocks rather than chains, so alpha is fitted to
# reproduce the measured acceptance length at default_k — extrapolating to other k
# with the chain formula is a documented conservative approximation.
SPEC_METHODS: tuple[str, ...] = ("mtp", "eagle3", "dflash", "dspark", "draft_model", "ngram")


@dataclass(frozen=True)
class SpeculativeProfile:
    label: str
    method: str  # one of SPEC_METHODS
    draft_params: float  # drafter weight footprint read per draft pass; 0 for ngram
    draft_layers: int  # draft KV layers; 0 for ngram
    parallel_draft: bool  # True: whole block in one pass (DFlash); False: k autoregressive passes
    default_k: int  # default number of speculative tokens per cycle
    acceptance_alpha: (
        float  # per-token acceptance probability, fitted to measured acceptance length
    )
    kv_overhead: float  # draft KV cache as a fraction of target KV bytes per token
    source: str
    note: str
    # Attached draft checkpoints keep their own storage precision. Native MTP
    # modules may leave this at zero to inherit the target checkpoint format.
    exact_weight_bytes: float = 0.0
    # MoE draft modules have many resident expert parameters but activate only
    # a subset per token. Zero means the resident count is also the active count.
    active_params: float = 0.0
    # Empty preserves the generic 1..32 planner range. Non-empty profiles encode
    # the only depths that are supported or calibrated well enough for Auto mode.
    supported_ks: tuple[int, ...] = ()
    # Optional per-depth alpha fits prevent extrapolating one measured accept
    # length across structurally different draft depths.
    acceptance_alpha_by_k: tuple[tuple[int, float], ...] = ()
    # Exact resident bytes by target precision for mixed-format native draft
    # modules. This avoids scaling a measured FP8 artifact with average model BPP.
    exact_weight_bytes_by_precision: tuple[tuple[str, float], ...] = ()


def _mtp_profile(
    alpha: float,
    active_params: float,
    default_k: int,
    source: str,
    note: str,
    exact_weight_bytes: float = 0.0,
    resident_params: float = 0.0,
    draft_layers: int = 1,
    supported_ks: tuple[int, ...] = (),
    acceptance_alpha_by_k: tuple[tuple[int, float], ...] = (),
    label: str = "Native MTP",
    exact_weight_bytes_by_precision: tuple[tuple[str, float], ...] = (),
) -> SpeculativeProfile:
    resident_params = resident_params or active_params
    return SpeculativeProfile(
        label,
        "mtp",
        resident_params,
        draft_layers,
        False,
        default_k,
        alpha,
        0.03,
        source,
        note,
        exact_weight_bytes,
        active_params,
        supported_ks,
        acceptance_alpha_by_k,
        exact_weight_bytes_by_precision,
    )


def _eagle3_profile(
    draft_params: float, alpha: float, default_k: int, source: str, note: str
) -> SpeculativeProfile:
    return SpeculativeProfile(
        "EAGLE-3 speculator",
        "eagle3",
        draft_params,
        1,
        False,
        default_k,
        alpha,
        0.05,
        source,
        note,
        draft_params * 2.0,
        draft_params,
    )


def _dflash_profile(
    draft_params: float,
    alpha: float,
    default_k: int,
    source: str,
    note: str,
    draft_layers: int = 5,
    exact_weight_bytes: float = 0.0,
    supported_ks: tuple[int, ...] = (),
    acceptance_alpha_by_k: tuple[tuple[int, float], ...] = (),
) -> SpeculativeProfile:
    return SpeculativeProfile(
        "DFlash block-diffusion speculator",
        "dflash",
        draft_params,
        draft_layers,
        True,
        default_k,
        alpha,
        0.08,
        source,
        note,
        exact_weight_bytes or draft_params * 2.0,
        draft_params,
        supported_ks,
        acceptance_alpha_by_k,
    )


def _dspark_profile(
    draft_params: float,
    alpha: float,
    source: str,
    note: str,
    *,
    exact_weight_bytes: float = 0.0,
    draft_layers: int = 5,
    kv_overhead: float = 0.0,
    default_k: int = 7,
) -> SpeculativeProfile:
    """Published fixed-depth parallel DSpark profile.

    DSpark attached checkpoints execute their draft layers in one block pass.
    ``kv_overhead`` remains explicit because hybrid/recurrent draft state is not
    equivalent to an ordinary full-attention KV cache.
    """
    return SpeculativeProfile(
        "DSpark parallel speculator",
        "dspark",
        draft_params,
        draft_layers,
        True,
        default_k,
        alpha,
        kv_overhead,
        source,
        note,
        exact_weight_bytes,
        draft_params,
        (default_k,),
        ((default_k, alpha),),
    )


NGRAM_SPECULATIVE_PROFILE = SpeculativeProfile(
    "N-gram (training-free)",
    "ngram",
    0.0,
    0,
    False,
    5,
    0.35,
    0.0,
    "https://specdecode-bench.github.io/",
    "Zero extra memory. Acceptance is workload-dependent: wins on high prompt-output overlap "
    "(code editing, BLEU-4 > 0.6, up to ~2.75x), modest elsewhere. Conservative alpha.",
)

# Capability flags. Projects can require one or more; models must supply them to be eligible.
# Kept deliberately coarse — the planner isn't a model quality benchmark, it's a capacity model.
MODEL_CAPABILITIES: tuple[str, ...] = ("tools", "ctx_128k", "images", "audio", "reasoning")
CAPABILITY_LABELS = {
    "tools": "Tool use",
    "ctx_128k": "≥128k ctx",
    "images": "Image input",
    "audio": "Audio input",
    "reasoning": "Thinking / reasoning",
}
# Tool use remains the common default. Long-context eligibility is derived from the
# model's explicit context limit below instead of being granted to every text model.
DEFAULT_MODEL_CAPABILITIES: frozenset[str] = frozenset({"tools"})


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    cat: str
    color: str
    total_params: float  # parameter count
    active_params: float  # activated parameter count
    is_moe: bool
    layers: int
    num_heads: int
    kv_heads: int
    head_dim: int
    is_mla: bool
    mla_kv_dim: int = 0
    mla_rope_dim: int = 0
    mla_tp_supported: bool = False
    kv_layers: int = -1
    bf16_weight_bytes_per_param: float = 2.0
    fp8_weight_bytes_per_param: float = 1.0
    hidden: bool = False
    extra_capabilities: frozenset[str] = frozenset()
    # Benchmark-anchored capability axes used by the revenue projection.
    # quality ∈ [0,1]: abstract success axis paired with task difficulty via success_rate().
    # token_efficiency > 0: per-model token-budget multiplier baseline — 1.0 = 10M output
    # tokens on Artificial Analysis' Intelligence Index, >1 = uses fewer tokens, <1 = verbose.
    quality: float = 0.5
    quality_confidence: float = 1.0
    token_efficiency: float = 1.0
    hidden_dim: int = 0
    attention_layers: int = -1
    local_attention_layers: int = 0
    local_attention_window: int = 0
    local_attention_heads: int = 0
    local_attention_head_dim: int = 0
    local_kv_heads: int = 0
    local_kv_head_dim: int = 0
    global_kv_heads: int = 0
    global_head_dim: int = 0
    shared_key_value: bool = False
    linear_attention_layers: int = 0
    linear_attention_heads: int = 0
    linear_attention_head_dim: int = 0
    linear_attention_k_heads: int = 0
    linear_attention_k_head_dim: int = 0
    linear_attention_conv_kernel: int = 0
    attention_query_heads: int = 0
    # Some MLA checkpoints expose a wider Q/K head than the value head. The
    # legacy ``head_dim`` remains the Q/K width; zero preserves equal QK/V.
    attention_value_head_dim: int = 0
    # Sparse attention keeps a bounded set of selected KV positions but still
    # runs a lightweight indexer over the available context. Index-sharing
    # architectures can evaluate that indexer in fewer layers than the main
    # attention stack and reuse the selected positions in the remaining layers.
    sparse_attention_top_k: int = 0
    sparse_indexer_heads: int = 0
    sparse_indexer_head_dim: int = 0
    sparse_indexer_layers: int = 0
    # IndexPool-style architectures pool source keys for the indexer without
    # compressing the main attention KV. The compact key width is expressed in
    # elements so its storage follows the selected KV-cache precision.
    sparse_indexer_compression_ratio: int = 0
    sparse_indexer_cache_elements_per_compressed_token: int = 0
    # Per-target-layer compressed-attention ratios. A zero ratio denotes a
    # sliding-window-only layer; positive ratios retain one compressed KV row
    # per ``ratio`` source tokens in addition to the live window. Architectures
    # such as DeepSeek V4 mix several ratios, so a single global top-k cannot
    # represent either their KV residency or their decode read traffic.
    attention_compression_ratios: tuple[int, ...] = ()
    compressed_attention_window: int = 0
    compressed_attention_indexer_ratio: int = 0
    # Runtime-format bytes read for one compressed indexer key. This is kept
    # separate from indexer FLOP geometry because serving kernels can store a
    # compact FP8/FP4 key rather than ``heads * head_dim`` elements.
    sparse_indexer_cache_bytes_per_token: float = 0.0
    attention_label: str = ""
    # Optional architecture metadata for models whose depth mixing and sparse
    # channel mixing cannot be inferred from the ordinary transformer fields.
    attention_residual_block_size: int = 0
    moe_latent_dim: int = 0
    moe_intermediate_dim: int = 0
    moe_routed_experts: int = 0
    moe_active_experts: int = 0
    moe_shared_experts: int = 0
    activation_label: str = ""
    capabilities_override: frozenset[str] | None = None
    realtime_profile: RealtimeProfile | None = None
    embedding_profile: EmbeddingProfile | None = None
    # Maximum combined prompt + generated tokens accepted by the model. 131072 is
    # the catalog default for current text models; legacy/short-context families
    # override it at their definitions. Embedding models retain their profile cap.
    max_context_tokens: int = 131072
    speculative_profiles: tuple[SpeculativeProfile, ...] = ()
    # The released checkpoint format shown first in the precision menu.  The
    # stored assignment remains the real calculation precision (bf16/fp8/fp4),
    # so existing scenarios and every roofline formula remain compatible.
    native_precision: str = "bf16"
    native_precision_label: str = ""
    native_precision_note: str = ""

    @property
    def capabilities(self) -> frozenset[str]:
        base = (
            self.capabilities_override
            if self.capabilities_override is not None
            else DEFAULT_MODEL_CAPABILITIES
        )
        capabilities = (base | self.extra_capabilities) - {"ctx_128k"}
        if (
            not self.is_asr_model
            and not self.is_embedding_model
            and self.max_context_tokens >= 131072
        ):
            capabilities = capabilities | {"ctx_128k"}
        return capabilities

    @property
    def is_realtime_only(self) -> bool:
        """Backward-compatible alias for older ASR classification call sites."""
        return self.is_asr_model

    @property
    def is_asr_model(self) -> bool:
        return self.realtime_profile is not None

    @property
    def is_streaming_asr(self) -> bool:
        return self.realtime_profile is not None and self.realtime_profile.streaming

    @property
    def asr_mode_label(self) -> str:
        return "Realtime" if self.is_streaming_asr else "Non-realtime"

    @property
    def is_embedding_model(self) -> bool:
        return self.embedding_profile is not None

    @property
    def available_spec_profiles(self) -> tuple[SpeculativeProfile, ...]:
        # N-gram needs no drafter and is always available to plain text models.
        if self.is_asr_model or self.is_embedding_model:
            return ()
        return self.speculative_profiles + (NGRAM_SPECULATIVE_PROFILE,)

    @property
    def size_label(self) -> str:
        def fmt_b(params: float) -> str:
            b = params / 1e9
            if b >= 1000:
                return f"{b / 1000:.1f}T"
            if b < 1:
                return f"{b:.2f}".rstrip("0").rstrip(".")
            if b < 10:
                return f"{b:.1f}".rstrip("0").rstrip(".")
            return f"{b:.0f}"

        tp_b = fmt_b(self.total_params)
        ap_b = fmt_b(self.active_params)
        if self.is_moe:
            return f"{tp_b}B-A{ap_b}B"
        return f"{tp_b}B"

    @property
    def hidden_size(self) -> int:
        return self.hidden_dim or self.num_heads * self.head_dim

    @property
    def attention_layer_count(self) -> int:
        return self.layers if self.attention_layers < 0 else self.attention_layers

    @property
    def kv_layer_count(self) -> int:
        return self.attention_layer_count if self.kv_layers < 0 else self.kv_layers

    @property
    def local_attention_head_count(self) -> int:
        return self.local_attention_heads or self.attention_query_head_count

    @property
    def local_attention_head_size(self) -> int:
        return self.local_attention_head_dim or self.head_dim

    @property
    def local_kv_head_count(self) -> int:
        return self.local_kv_heads or self.kv_heads

    @property
    def local_kv_head_size(self) -> int:
        return self.local_kv_head_dim or self.local_attention_head_size

    @property
    def attention_query_head_count(self) -> int:
        return self.attention_query_heads or self.num_heads

    @property
    def attention_value_head_size(self) -> int:
        return self.attention_value_head_dim or self.head_dim

    @property
    def linear_attention_layer_count(self) -> int:
        return max(self.linear_attention_layers, 0)

    @property
    def linear_attention_head_count(self) -> int:
        return self.linear_attention_heads or self.num_heads

    @property
    def linear_attention_head_size(self) -> int:
        return self.linear_attention_head_dim or self.head_dim

    @property
    def linear_attention_k_head_count(self) -> int:
        return self.linear_attention_k_heads or self.linear_attention_head_count

    @property
    def linear_attention_k_head_size(self) -> int:
        return self.linear_attention_k_head_dim or self.linear_attention_head_size

    @property
    def linear_attention_kernel_size(self) -> int:
        return max(self.linear_attention_conv_kernel, 1)

    @property
    def attention_residual_block_count(self) -> int:
        block_size = max(int(self.attention_residual_block_size), 0)
        if block_size <= 0:
            return 0
        return math.ceil(max(int(self.layers), 0) / block_size)

    @property
    def attention_residual_source_count(self) -> int:
        # Block AttnRes also keeps the embedding representation as a source.
        blocks = self.attention_residual_block_count
        return blocks + 1 if blocks else 0

    def weight_bytes_per_param(self, prec: str) -> float:
        prec = normalize_precision(prec)
        profile = get_quantization_profile(self.key, prec)
        if profile is not None:
            return profile.weight_bytes_per_param(self.total_params)
        if prec == "bf16":
            return self.bf16_weight_bytes_per_param
        if prec == "fp8":
            return self.fp8_weight_bytes_per_param
        # Keep model-specific high-precision islands when a native FP8 catalog entry is
        # already above 1 B/param. This avoids pretending FP4 converts every tensor.
        retained_bpp = max(
            0.0,
            self.fp8_weight_bytes_per_param
            - PRECISION_SPECS["fp8"].effective_weight_bytes_per_param,
        )
        return PRECISION_SPECS[prec].effective_weight_bytes_per_param + retained_bpp

    def uses_mixed_weight_precision(self, prec: str) -> bool:
        if get_quantization_profile(self.key, prec) is not None:
            return True
        return not math.isclose(
            self.weight_bytes_per_param(prec), bytes_per_param(prec), rel_tol=1e-9, abs_tol=1e-9
        )

    def weight_bytes(self, prec: str) -> float:
        return self.total_params * self.weight_bytes_per_param(prec)

    def weight_gb(self, prec: str) -> float:
        return self.weight_bytes(prec) / 1e9

    def active_weight_bytes(self, prec: str) -> float:
        profile = get_quantization_profile(self.key, prec)
        if profile is not None:
            params = self.active_params if self.is_moe else self.total_params
            return params * profile.active_weight_bytes_per_param(self.total_params)
        params = self.active_params if self.is_moe else self.total_params
        return params * self.weight_bytes_per_param(prec)

    def kv_cache_bytes_per_elem(self, prec: str) -> float:
        profile = get_quantization_profile(self.key, prec)
        if profile is not None:
            return profile.kv_cache_bytes_per_elem
        return kv_cache_bytes_per_elem(prec)

    def quantization_profile(self, prec: str) -> QuantizationProfile | None:
        return get_quantization_profile(self.key, prec)

    @property
    def native_precision_key(self) -> str:
        return normalize_precision(self.native_precision)

    @property
    def native_precision_display(self) -> str:
        return self.native_precision_label or PRECISION_SPECS[self.native_precision_key].label

    @property
    def native_precision_description(self) -> str:
        if self.native_precision_note:
            return self.native_precision_note
        return PRECISION_SPECS[self.native_precision_key].description


QUANTIZATION_CAPTURED_AT = "2026-05-22"


def _nvfp4_profile(
    *,
    model_key: str,
    source_repo: str,
    source_revision: str,
    source_downloads: int,
    source_kind: str = "exact",
    storage_format_counts: dict[str, int] | None = None,
    compute_precision_shares: dict[str, float] | None = None,
    quantized: tuple[str, ...] = (),
    retained: tuple[str, ...] = (),
    total_weight_bytes_override: float | None = None,
    notes: str = "",
    captured_at: str = QUANTIZATION_CAPTURED_AT,
) -> tuple[tuple[str, str], QuantizationProfile]:
    return (
        (model_key, "nvfp4"),
        QuantizationProfile(
            precision_key="nvfp4",
            label="NVFP4",
            source_repo=source_repo,
            source_revision=source_revision,
            source_downloads=source_downloads,
            captured_at=captured_at,
            source_kind=source_kind,
            quant_algo="NVFP4",
            kv_cache_format="FP8",
            kv_cache_bytes_per_elem=1.0,
            group_size=16,
            storage_format_counts=storage_format_counts or {},
            compute_precision_shares=compute_precision_shares or {"nvfp4": 1.0},
            quantized=quantized,
            retained=retained,
            total_weight_bytes_override=total_weight_bytes_override,
            notes=notes,
        ),
    )


def _artifact_profile(
    *,
    model_key: str,
    precision_key: str,
    label: str,
    source_repo: str,
    source_revision: str,
    total_weight_bytes: float,
    compute_precision_shares: dict[str, float],
    quantized: tuple[str, ...],
    retained: tuple[str, ...],
    notes: str,
    quant_algo: str,
    kv_cache_format: str = "FP8",
    group_size: int | None = None,
    storage_format_counts: dict[str, int] | None = None,
    captured_at: str = "2026-09-02",
) -> tuple[tuple[str, str], QuantizationProfile]:
    return (
        (model_key, precision_key),
        QuantizationProfile(
            precision_key=precision_key,
            label=label,
            source_repo=source_repo,
            source_revision=source_revision,
            source_downloads=0,
            captured_at=captured_at,
            source_kind="exact",
            quant_algo=quant_algo,
            kv_cache_format=kv_cache_format,
            kv_cache_bytes_per_elem=1.0,
            group_size=group_size,
            storage_format_counts=storage_format_counts or {},
            compute_precision_shares=compute_precision_shares,
            quantized=quantized,
            retained=retained,
            total_weight_bytes_override=total_weight_bytes,
            notes=notes,
        ),
    )


MODEL_QUANTIZATION_PROFILES: dict[tuple[str, str], QuantizationProfile] = dict(
    [
        _artifact_profile(
            model_key="qwen38-27b",
            precision_key="fp8",
            label="Official FP8",
            source_repo="Qwen/Qwen3.8-27B-FP8",
            source_revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            total_weight_bytes=30_863_648_224,
            storage_format_counts={"BF16": 3_082_220_272, "F8_E4M3": 24_699_207_680},
            compute_precision_shares={"fp8": 0.889, "bf16": 0.111},
            quantized=("language linear weights FP8",),
            retained=("vision, embeddings, norms, and excluded tensors BF16",),
            quant_algo="official block FP8",
            notes="Exact official FP8 shard inventory; 30.864 GB of indexed tensor data.",
        ),
        _artifact_profile(
            model_key="qwen38-2.4t-a95b",
            precision_key="fp8",
            label="Official FP8",
            source_repo="Qwen/Qwen3.8-2.4T-A95B-FP8",
            source_revision="d2dc35658bcf77e66643428cb52e774cc3b5bd29",
            total_weight_bytes=2_495_773_699_840,
            storage_format_counts={"BF16": 49_590_974_336, "F8_E4M3": 2_396_591_751_168},
            compute_precision_shares={"fp8": 0.98, "bf16": 0.02},
            quantized=("transformer linear weights FP8",),
            retained=("embeddings, norms, routers, and excluded tensors BF16",),
            quant_algo="official block FP8",
            notes="Exact official FP8 shard inventory; 2.496 TB of indexed tensor data.",
        ),
        _artifact_profile(
            model_key="qwen38-2.4t-a95b",
            precision_key="nvfp4",
            label="NVIDIA NVFP4",
            source_repo="nvidia/Qwen3.8-2.4T-A95B-NVFP4",
            source_revision="049fa3b549c6d0cbca8355dc7e5d73585e41d2d0",
            total_weight_bytes=1_444_420_107_432,
            storage_format_counts={
                "BF16": 35_470_849_920,
                "U8": 1_185_410_973_696,
                "F8_E4M3": 39_889_928_192,
            },
            compute_precision_shares={"nvfp4": 0.88, "fp8": 0.10, "bf16": 0.02},
            quantized=("MoE experts NVFP4", "attention and GDN projections FP8"),
            retained=("embeddings, norms, routers, and MTP tensors BF16",),
            quant_algo="NVIDIA mixed NVFP4/FP8",
            group_size=16,
            notes="Exact 200-shard artifact footprint; optional NVFP4 KV is not selected by this FP8-KV profile.",
        ),
        _artifact_profile(
            model_key="qwen38-flash-next",
            precision_key="fp8",
            label="Official FP8",
            source_repo="Qwen/Qwen3.8-Flash-Next-FP8",
            source_revision="236dfdf285828023ca3bcd3f37366c58a3469b13",
            total_weight_bytes=185_487_179_488,
            storage_format_counts={"BF16": 5_487_198_064, "F8_E4M3": 174_512_783_360},
            compute_precision_shares={"fp8": 0.97, "bf16": 0.03},
            quantized=("main, N-gram embedding, and supported linear tensors FP8",),
            retained=("excluded tensors BF16",),
            quant_algo="official block FP8",
            notes="Exact official FP8 inventory, including the 51B N-gram table and bundled MTP tensors.",
        ),
        _artifact_profile(
            model_key="deepseek-v4-pro",
            precision_key="mxfp4",
            label="Native MXFP4/FP8",
            source_repo="deepseek-ai/DeepSeek-V4-Pro-0813",
            source_revision="72e1d3230f6c080a530b0a1d46f8eb4602340597",
            total_weight_bytes=892_727_580_904,
            compute_precision_shares={"mxfp4": 0.85, "fp8": 0.14, "bf16": 0.01},
            quantized=("routed experts MXFP4", "attention and shared-expert projections FP8"),
            retained=("norms, embeddings, and excluded tensors BF16",),
            quant_algo="native mixed MXFP4/FP8",
            group_size=32,
            notes="Exact 0813 indexed checkpoint footprint; bundled DSpark tensors remain part of resident storage.",
        ),
        _artifact_profile(
            model_key="deepseek-v4-flash",
            precision_key="mxfp4",
            label="Native MXFP4/FP8",
            source_repo="deepseek-ai/DeepSeek-V4-Flash-0731",
            source_revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
            total_weight_bytes=166_878_536_440,
            compute_precision_shares={"mxfp4": 0.84, "fp8": 0.15, "bf16": 0.01},
            quantized=("routed experts MXFP4", "supported dense projections FP8"),
            retained=(
                "attention, norms, embeddings, and DSpark exclusions at published higher precision",
            ),
            quant_algo="native mixed MXFP4/FP8",
            group_size=32,
            notes="Exact 0731 indexed checkpoint footprint.",
        ),
        _artifact_profile(
            model_key="deepseek-v4-pro",
            precision_key="nvfp4",
            label="NVIDIA NVFP4 compatibility",
            source_repo="nvidia/DeepSeek-V4-Pro-0813-NVFP4",
            source_revision="a6ed51e09c9f9ae455c424cc7b83a322d6744485",
            total_weight_bytes=892_727_580_904,
            compute_precision_shares={"nvfp4": 0.85, "fp8": 0.14, "bf16": 0.01},
            quantized=("routed MoE operators NVFP4",),
            retained=("attention, shared experts, head, and DSpark unquantized",),
            quant_algo="NVIDIA MXFP4-to-NVFP4 compatibility cast",
            group_size=16,
            notes="Indexed metadata matches the source footprint; NVIDIA notes the served artifact is slightly larger, so this is a compatibility/performance option rather than a smaller quant.",
        ),
        _artifact_profile(
            model_key="deepseek-v4-flash",
            precision_key="nvfp4",
            label="NVIDIA NVFP4 compatibility",
            source_repo="nvidia/DeepSeek-V4-Flash-0731-NVFP4",
            source_revision="f1caa71142bd0be02f728c79f75042ac1e461579",
            total_weight_bytes=166_878_536_440,
            compute_precision_shares={"nvfp4": 0.84, "fp8": 0.15, "bf16": 0.01},
            quantized=("routed experts losslessly cast from MXFP4 to NVFP4",),
            retained=("DSpark and unsupported tensors at source precision",),
            quant_algo="NVIDIA MXFP4-to-NVFP4 compatibility cast",
            group_size=16,
            notes="Compatibility/performance artifact; the source checkpoint was already four-bit for routed experts, so this is not modeled as a smaller footprint.",
        ),
        _artifact_profile(
            model_key="kimi-k3",
            precision_key="nvfp4",
            label="NVIDIA NVFP4 compatibility",
            source_repo="nvidia/Kimi-K3-NVFP4",
            source_revision="b2428a0b83a8b712ff2e1a8448a103d4175341f1",
            total_weight_bytes=1_609_777_087_808,
            compute_precision_shares={"nvfp4": 0.47, "fp8": 0.18, "bf16": 0.35},
            quantized=("routed experts NVFP4", "supported KDA/MLA projections block FP8"),
            retained=(
                "latent/shared experts, routers, towers, and unsupported tensors higher precision",
            ),
            quant_algo="NVIDIA mixed NVFP4/FP8 compatibility",
            group_size=16,
            notes="Validated at 196,608 tokens on 8×B300. This Blackwell alternative is larger than Kimi's native MXFP4 artifact and is not listed as a smaller quant.",
        ),
        _artifact_profile(
            model_key="nemotron35-lightning",
            precision_key="nvfp4",
            label="NVIDIA NVFP4 QAD",
            source_repo="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            source_revision="cc84af2fe71647d87f4486c064f320e1e7535243",
            total_weight_bytes=21_559_589_596,
            compute_precision_shares={"nvfp4": 0.74, "fp8": 0.12, "bf16": 0.14},
            quantized=("MoE/shared/head and Mamba linears NVFP4/FP8",),
            retained=(
                "attention projections, embeddings, routers, norms, convolutions, and MTP BF16",
            ),
            quant_algo="NVIDIA QAD four_over_six NVFP4",
            group_size=16,
            notes="Exact 52-shard QAD artifact; NVIDIA reports 98.97% median score recovery versus BF16.",
        ),
        _artifact_profile(
            model_key="glm53f",
            precision_key="fp8",
            label="Native block FP8",
            source_repo="zai-org/GLM-5.3-Flash",
            source_revision="c54b8d14c81437589ce7db2bece34f157bd90203",
            total_weight_bytes=320_833_372_408,
            compute_precision_shares={"fp8": 0.96, "bf16": 0.04},
            quantized=("most language weights block FP8",),
            retained=("6.93B BF16 parameters and small FP32 islands",),
            quant_algo="native block FP8",
            notes="Exact base-model safetensors inventory; the optional next-token module is charged separately when native MTP is enabled.",
        ),
        _nvfp4_profile(
            model_key="g31",
            source_repo="nvidia/Gemma-4-31B-IT-NVFP4",
            source_revision="e5ef03afa233c35cb000323ff098d4291e1dd07c",
            source_downloads=2_281_570,
            storage_format_counts={
                "BF16": 10_464_098_156,
                "U8": 10_404_495_360,
                "F8_E4M3": 1_300_561_920,
                "F32": 360,
            },
            compute_precision_shares={"nvfp4": 0.62, "bf16": 0.38},
            quantized=("language MLP weights: packed FP4 payload + FP8 scales",),
            retained=(
                "language self-attention BF16",
                "embeddings BF16",
                "vision tower BF16",
                "lm_head BF16",
            ),
            notes="HF quant config excludes every language self-attention block, the vision tower, embed_vision, and lm_head.",
        ),
        _nvfp4_profile(
            model_key="g26",
            source_repo="nvidia/Gemma-4-26B-A4B-NVFP4",
            source_revision="a19cfe00be84568a6867111c9a68c9c44fdcffe6",
            source_downloads=923_412,
            storage_format_counts={
                "BF16": 2_967_950_926,
                "U8": 11_418_992_640,
                "F8_E4M3": 1_427_374_080,
            },
            compute_precision_shares={"nvfp4": 0.72, "bf16": 0.28},
            quantized=("later language MoE/MLP tensors: packed FP4 payload + FP8 scales",),
            retained=(
                "early language layers BF16",
                "routers BF16",
                "vision tower BF16",
                "lm_head BF16",
            ),
            notes="HF quant config excludes language layers 0-29 plus routers/self-attention, vision tower, embed_vision, and lm_head.",
        ),
        _nvfp4_profile(
            model_key="q35",
            source_repo="txn545/Qwen3.5-35B-A3B-NVFP4",
            source_revision="63ffbd1d5ca18043b67ea5302238afe3929fddb2",
            source_downloads=26_399,
            storage_format_counts={
                "F32": 61_700,
                "BF16": 3_613_738_864,
                "F8_E4M3": 2_021_130_240,
                "U8": 16_169_041_920,
            },
            compute_precision_shares={"nvfp4": 0.82, "bf16": 0.18},
            quantized=(
                "MoE expert weights: packed FP4 payload + FP8 scales",
                "selected self-attention layers",
            ),
            retained=(
                "linear attention BF16",
                "router gates BF16",
                "embeddings BF16",
                "vision modules BF16",
                "lm_head BF16",
            ),
            notes="Top exact Qwen3.5-35B-A3B NVFP4 artifact by HF downloads when captured.",
        ),
        _nvfp4_profile(
            model_key="q122",
            source_repo="Sehyo/Qwen3.5-122B-A10B-NVFP4",
            source_revision="56a6bdda33285ba2d5688e4f71f6c714649497b4",
            source_downloads=198_104,
            storage_format_counts={
                "F32": 74_112,
                "BF16": 7_725_676_784,
                "F8_E4M3": 7_335_051_264,
                "U8": 58_680_410_112,
            },
            compute_precision_shares={"nvfp4": 0.84, "bf16": 0.16},
            quantized=("Linear MoE/expert tensors: packed FP4 payload + FP8 scales",),
            retained=(
                "linear attention BF16",
                "router gates BF16",
                "visual modules BF16",
                "lm_head BF16",
            ),
            notes="Recipe targets Linear and ignores lm_head, router gates, shared expert gates, linear attention, and visual modules.",
        ),
        _nvfp4_profile(
            model_key="q397",
            source_repo="Sehyo/Qwen3.5-122B-A10B-NVFP4",
            source_revision="56a6bdda33285ba2d5688e4f71f6c714649497b4",
            source_downloads=198_104,
            source_kind="family",
            total_weight_bytes_override=265_101_993_628,
            compute_precision_shares={"nvfp4": 0.84, "bf16": 0.16},
            quantized=("Qwen3.5 MoE Linear tensors by family proxy",),
            retained=(
                "linear attention BF16",
                "router gates BF16",
                "visual modules BF16",
                "lm_head BF16",
            ),
            notes="Family proxy until the larger Qwen3.5-397B safetensors headers are captured locally.",
        ),
        (
            ("kimi-k3", "mxfp4"),
            QuantizationProfile(
                precision_key="mxfp4",
                label="MXFP4",
                source_repo="moonshotai/Kimi-K3",
                source_revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
                source_downloads=2_850,
                captured_at="2026-07-27",
                source_kind="exact",
                quant_algo="native MXFP4 QAT",
                kv_cache_format="FP8",
                kv_cache_bytes_per_elem=1.0,
                group_size=32,
                storage_format_counts={},
                # The report's 104.2B active count includes 48.62B active routed-
                # expert parameters; the remaining attention, latent projections,
                # shared experts, routers, and dense layer stay in BF16.
                compute_precision_shares={
                    "mxfp4": 0.466606256890595,
                    "bf16": 0.533393743109405,
                },
                quantized=(
                    "92 layers of routed MoE expert weights in MXFP4 with MXFP8 activations",
                ),
                retained=(
                    "attention and Block AttnRes projections BF16",
                    "latent MoE projections, shared experts, and routers BF16",
                    "dense layer, embeddings, MoonViT-V2, projector, and lm_head BF16",
                ),
                # Exact aggregate from the release's 96-shard safetensors index.
                total_weight_bytes_override=1_560_860_324_864,
                # Derived from the exact routed-expert geometry and the release
                # artifact: active experts use 0.53125 B/param; all other active
                # weights average 1.998 B/param.
                active_weight_bytes_per_param_override=1.313609348872837,
                notes=(
                    "Exact 96-shard release footprint. Only routed expert weights are MXFP4; "
                    "all non-expert components remain in higher precision per the technical report."
                ),
            ),
        ),
        _nvfp4_profile(
            model_key="k25",
            source_repo="nvidia/Kimi-K2.5-NVFP4",
            source_revision="0fd0a5e6879298d3476e3b61852a79792a35ae3d",
            source_downloads=1_227_250,
            total_weight_bytes_override=590_850_735_131,
            compute_precision_shares={"nvfp4": 0.80, "fp8": 0.10, "bf16": 0.10},
            quantized=("MoE experts NVFP4", "selected dense projections FP8"),
            retained=("self-attention BF16", "vision/projector modules BF16", "lm_head BF16"),
            notes="HF quant config is mixed precision with NVFP4 experts and FP8 dense projections; bytes use repository storage.",
        ),
        _nvfp4_profile(
            model_key="inkling",
            source_repo="thinkingmachines/Inkling-NVFP4",
            source_revision="d11961f515e883e37796edb9dd6ec1bf0e0e8212",
            source_downloads=4,
            storage_format_counts={
                "I64": 378,
                "F32": 48_704,
                "BF16": 39_160_185_992,
                "F8_E4M3": 57_076_088_832,
                "U8": 456_608_710_656,
            },
            compute_precision_shares={"nvfp4": 0.82, "fp8": 0.10, "bf16": 0.08},
            quantized=("MoE/feed-forward tensors: packed NVFP4 payload + FP8 scales",),
            retained=(
                "attention and SConv tensors BF16/FP8",
                "routers BF16",
                "multimodal towers BF16",
                "lm_head BF16",
            ),
            notes="Exact base-checkpoint repository storage; excludes the optional 10.5 GB MTP drafter artifact.",
        ),
        _nvfp4_profile(
            model_key="minimax25",
            source_repo="nvidia/MiniMax-M2.5-NVFP4",
            source_revision="b6220d658389629b9d507d4b2bb314f41fea7898",
            source_downloads=137_435,
            storage_format_counts={
                "BF16": 1_278_796_288,
                "F32": 2_730_491_904,
                "F8_E4M3": 14_042_529_792,
                "U8": 112_340_238_336,
            },
            compute_precision_shares={"nvfp4": 0.86, "bf16": 0.14},
            quantized=("MoE/feed-forward weights: packed FP4 payload + FP8 scales",),
            retained=("self-attention BF16", "MoE gates BF16", "lm_head BF16"),
        ),
        _nvfp4_profile(
            model_key="minimax27",
            source_repo="nvidia/MiniMax-M2.7-NVFP4",
            source_revision="e79701cb1f9dce8fe5395b9ed2b20170beebecde",
            source_downloads=195_984,
            storage_format_counts={
                "BF16": 1_278_796_288,
                "F32": 2_730_491_904,
                "F8_E4M3": 14_042_529_792,
                "U8": 112_340_238_336,
            },
            compute_precision_shares={"nvfp4": 0.86, "bf16": 0.14},
            quantized=("MoE/feed-forward weights: packed FP4 payload + FP8 scales",),
            retained=("self-attention BF16", "MoE gates BF16", "lm_head BF16"),
        ),
        _nvfp4_profile(
            model_key="minimax3",
            source_repo="nvidia/MiniMax-M3-NVFP4",
            source_revision="901464083161bf8612a29ff7ad29914cd4ab4a85",
            source_downloads=0,
            total_weight_bytes_override=250_103_762_320,
            compute_precision_shares={"nvfp4": 0.86, "bf16": 0.14},
            quantized=("MoE/feed-forward weights: packed FP4 payload + FP8 scales",),
            retained=(
                "attention, routing-sensitive tensors, embeddings, multimodal modules, and lm_head in higher precision",
            ),
            notes="Exact 88-shard official artifact footprint. The 86/14 compute split is a MiniMax-family planning proxy pending tensor-header classification.",
            captured_at="2026-06-26",
        ),
        _nvfp4_profile(
            model_key="nvidia-nemotron-3-embed-1b",
            source_repo="nvidia/Nemotron-3-Embed-1B-NVFP4",
            source_revision="c01600056187dba44bd712346cedb1e57fa50220",
            source_downloads=0,
            total_weight_bytes_override=1_027_789_672,
            compute_precision_shares={"nvfp4": 1.0},
            quantized=("encoder linear weights: packed NVFP4 payload + FP8 scales",),
            retained=(
                "normalization, embedding, and pooling-sensitive tensors in higher precision",
            ),
            notes="Exact official artifact footprint captured from the safetensors index; download count was not used as provenance.",
            captured_at="2026-07-16",
        ),
        _nvfp4_profile(
            model_key="nem3s",
            source_repo="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            source_revision="4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6",
            source_downloads=1_017_905,
            storage_format_counts={
                "F32": 20_992,
                "BF16": 6_020_553_728,
                "F8_E4M3": 11_873_353_728,
                "U8": 56_382_455_808,
            },
            compute_precision_shares={"nvfp4": 0.82, "fp8": 0.08, "bf16": 0.10},
            quantized=("latent-MoE experts NVFP4", "some dense mixer projections FP8"),
            retained=("attention and routing-sensitive tensors BF16",),
            notes="HF quant config is mixed precision and KV cache FP8.",
        ),
        _nvfp4_profile(
            model_key="nem3n",
            source_repo="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            source_revision="ce1b118ae66ec705d02c241525192832eb045fd3",
            source_downloads=532_640,
            storage_format_counts={
                "F32": 7_916_416,
                "BF16": 1_078_212_032,
                "F8_E4M3": 1_905_738_240,
                "U8": 15_245_905_920,
            },
            compute_precision_shares={"nvfp4": 0.82, "bf16": 0.18},
            quantized=("latent-MoE experts NVFP4",),
            retained=("routing-sensitive and attention tensors BF16",),
        ),
        _nvfp4_profile(
            model_key="nem3no",
            source_repo="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            source_revision="dc5f0b0bfddf8b6e0f5891475be9af05b80126fe",
            source_downloads=1_281_803,
            storage_format_counts={
                "F32": 7_916_416,
                "BF16": 2_217_567_168,
                "F8_E4M3": 3_251_232_768,
                "U8": 14_687_404_032,
            },
            compute_precision_shares={"nvfp4": 0.76, "bf16": 0.24},
            quantized=("language latent-MoE experts NVFP4",),
            retained=(
                "omni/multimodal towers BF16",
                "routing-sensitive and attention tensors BF16",
            ),
        ),
        _nvfp4_profile(
            model_key="glm5",
            source_repo="nvidia/GLM-5-NVFP4",
            source_revision="dc54ff55a7e9e71b85db953d8bc22eca894b44c6",
            source_downloads=107_715,
            storage_format_counts={
                "BF16": 25_577_755_904,
                "U8": 364_143_181_824,
                "F8_E4M3": 45_517_897_728,
                "F32": 19_456,
            },
            compute_precision_shares={"nvfp4": 0.84, "bf16": 0.16},
            quantized=("GLM MoE expert tensors NVFP4",),
            retained=("dense/routing-sensitive tensors BF16",),
        ),
        _nvfp4_profile(
            model_key="glm51",
            source_repo="nvidia/GLM-5-NVFP4",
            source_revision="dc54ff55a7e9e71b85db953d8bc22eca894b44c6",
            source_downloads=107_715,
            source_kind="family",
            storage_format_counts={
                "BF16": 25_577_755_904,
                "U8": 364_143_181_824,
                "F8_E4M3": 45_517_897_728,
                "F32": 19_456,
            },
            compute_precision_shares={"nvfp4": 0.84, "bf16": 0.16},
            quantized=("GLM MoE expert tensors NVFP4 by family proxy",),
            retained=("dense/routing-sensitive tensors BF16",),
        ),
        _nvfp4_profile(
            model_key="glm52",
            source_repo="nvidia/GLM-5-NVFP4",
            source_revision="dc54ff55a7e9e71b85db953d8bc22eca894b44c6",
            source_downloads=107_715,
            source_kind="family",
            storage_format_counts={
                "BF16": 25_577_755_904,
                "U8": 364_143_181_824,
                "F8_E4M3": 45_517_897_728,
                "F32": 19_456,
            },
            compute_precision_shares={"nvfp4": 0.84, "bf16": 0.16},
            quantized=("GLM-5 expert tensors NVFP4 by family proxy",),
            retained=("IndexShare/MTP and routing-sensitive tensors BF16",),
        ),
    ]
)

# Retired artifact data stays recoverable with the retirement record, but cannot
# leak into current placement, UI, or source-quality checks.
ARCHIVED_QUANTIZATION_PROFILES = {
    key: profile
    for key, profile in MODEL_QUANTIZATION_PROFILES.items()
    if key[0] in ARCHIVED_MODELS
}
MODEL_QUANTIZATION_PROFILES = {
    key: profile
    for key, profile in MODEL_QUANTIZATION_PROFILES.items()
    if key[0] not in ARCHIVED_MODELS
}


def get_quantization_profile(model_key: str, prec: str) -> QuantizationProfile | None:
    return MODEL_QUANTIZATION_PROFILES.get((model_key, normalize_precision(prec)))
