"""GPU, model, and distribution data definitions."""

import math
from dataclasses import dataclass


FP8_SPEEDUP_DEFAULT = 1.4
FP8_SPEEDUP_OPTIONS = [1.0, 1.2, 1.4, 1.6]
MIXED_NATIVE_BF16_WEIGHT_BPP = 1.35
MIXED_NATIVE_FP8_WEIGHT_BPP = 1.10
FP4_FP8_MOE_WEIGHT_BPP = 0.70
MXFP4_WEIGHT_BPP = (4 + 8 / 32) / 8
NVFP4_WEIGHT_BPP = (4 + 8 / 16) / 8
VENDOR_LABELS = {
    "nv": "NVIDIA",
    "amd": "AMD",
    "intel": "Intel",
    "apple": "Apple",
}


@dataclass(frozen=True)
class PrecisionSpec:
    key: str
    label: str
    nominal_weight_bytes_per_param: float
    effective_weight_bytes_per_param: float
    kv_cache_bytes_per_elem: float
    description: str


@dataclass(frozen=True)
class NumberFormatSpec:
    key: str
    label: str
    bytes_per_elem: float
    description: str


NUMBER_FORMAT_SPECS: dict[str, NumberFormatSpec] = {
    "BF16": NumberFormatSpec("BF16", "BF16", 2.0, "Brain floating point 16-bit tensor."),
    "F32": NumberFormatSpec("F32", "FP32", 4.0, "Float32 tensor, usually tiny global scale auxiliaries."),
    "F8_E4M3": NumberFormatSpec("F8_E4M3", "FP8 E4M3", 1.0, "FP8 E4M3 scale or activation tensor."),
    "U8": NumberFormatSpec("U8", "Packed FP4", 1.0, "Unsigned byte storage for packed FP4 payloads."),
}


def _storage_bytes(format_counts: dict[str, int]) -> float:
    total = 0.0
    for fmt, elems in format_counts.items():
        spec = NUMBER_FORMAT_SPECS.get(fmt)
        if spec is None:
            continue
        total += elems * spec.bytes_per_elem
    return total


@dataclass(frozen=True)
class QuantizationProfile:
    """Offline-captured model artifact profile.

    Counts are safetensors storage element counts, not logical parameter counts:
    U8 entries are already packed FP4 bytes, while F8_E4M3 entries are scale tensors.
    """

    precision_key: str
    label: str
    source_repo: str
    source_revision: str
    source_downloads: int
    captured_at: str
    source_kind: str
    quant_algo: str
    kv_cache_format: str
    kv_cache_bytes_per_elem: float
    group_size: int | None
    storage_format_counts: dict[str, int]
    compute_precision_shares: dict[str, float]
    quantized: tuple[str, ...]
    retained: tuple[str, ...]
    total_weight_bytes_override: float | None = None
    active_weight_bytes_per_param_override: float | None = None
    notes: str = ""

    @property
    def total_weight_bytes(self) -> float:
        if self.total_weight_bytes_override is not None:
            return self.total_weight_bytes_override
        return _storage_bytes(self.storage_format_counts)

    def weight_bytes_per_param(self, total_params: float) -> float:
        return self.total_weight_bytes / max(float(total_params), 1.0)

    def active_weight_bytes_per_param(self, total_params: float) -> float:
        if self.active_weight_bytes_per_param_override is not None:
            return self.active_weight_bytes_per_param_override
        return self.weight_bytes_per_param(total_params)

    @property
    def source_label(self) -> str:
        if self.source_kind == "exact":
            return f"HF {self.source_repo}"
        return f"HF family proxy {self.source_repo}"

    @property
    def storage_summary(self) -> str:
        parts = []
        for fmt, count in sorted(self.storage_format_counts.items()):
            spec = NUMBER_FORMAT_SPECS.get(fmt)
            label = spec.label if spec else fmt
            parts.append(f"{label} {count / 1e9:.2f}B")
        return " · ".join(parts)

    @property
    def compute_summary(self) -> str:
        parts = []
        for prec, share in self.compute_precision_shares.items():
            label = PRECISION_SPECS[prec].label if prec in PRECISION_SPECS else prec.upper()
            parts.append(f"{label} {share * 100:.0f}%")
        return " / ".join(parts)


PRECISION_SPECS: dict[str, PrecisionSpec] = {
    "bf16": PrecisionSpec(
        "bf16",
        "BF16",
        2.0,
        2.0,
        2.0,
        "BF16 weights and KV cache.",
    ),
    "fp8": PrecisionSpec(
        "fp8",
        "FP8",
        1.0,
        1.0,
        1.0,
        "FP8 weights and FP8 KV cache.",
    ),
    "nvfp4": PrecisionSpec(
        "nvfp4",
        "NVFP4",
        0.5,
        NVFP4_WEIGHT_BPP,
        1.0,
        "E2M1 FP4 weights with FP8 scale per 16 values and tensor scaling; KV cache stays FP8.",
    ),
    "mxfp4": PrecisionSpec(
        "mxfp4",
        "MXFP4",
        0.5,
        MXFP4_WEIGHT_BPP,
        1.0,
        "OCP MXFP4 E2M1 weights with E8M0 scale per 32 values; KV cache stays FP8.",
    ),
}
PRECISIONS = tuple(PRECISION_SPECS.keys())
PRECISION_LABELS = {key: spec.label for key, spec in PRECISION_SPECS.items()}
PRECISION_DESCRIPTIONS = {key: spec.description for key, spec in PRECISION_SPECS.items()}


def normalize_precision(prec: str | None) -> str:
    return prec if prec in PRECISION_SPECS else "bf16"


def bytes_per_param(prec: str) -> float:
    return PRECISION_SPECS[normalize_precision(prec)].nominal_weight_bytes_per_param


def kv_cache_bytes_per_elem(prec: str) -> float:
    return PRECISION_SPECS[normalize_precision(prec)].kv_cache_bytes_per_elem


@dataclass
class GPU:
    key: str
    name: str
    vendor: str  # 'nv', 'amd', 'intel', or 'apple'
    mem: float  # bytes
    bw: float  # bytes/s, published peak memory bandwidth shown in the UI
    bf16: float  # FLOP/s
    fp8: float  # FLOP/s
    scale_up_p2p_bw_bidir: float  # bytes/s, per-GPU aggregate bidirectional peer BW for node_size topology
    node_size: int = 8
    planner_bw: float | None = None  # bytes/s, optional sustained bandwidth proxy used by planner math
    fp4: float | None = None  # FLOP/s for native dense FP4/MXFP4/NVFP4 tensor paths, when available
    tdp_watts: float = 0.0  # published board TDP — used with a utilization factor for CO2 math
    min_count: int = 1  # minimum pool size when this profile is only sold as a system/rack
    count_multiple: int = 1  # pool sizes snap to this multiple for system/rack-only profiles

    @property
    def mem_gb(self) -> float:
        return self.mem / 1e9

    @property
    def bw_tbs(self) -> float:
        return self.bw / 1e12

    @property
    def vendor_label(self) -> str:
        return VENDOR_LABELS.get(self.vendor, self.vendor.title())

    @property
    def effective_bw(self) -> float:
        return self.planner_bw if self.planner_bw is not None else self.bw

    @property
    def scale_up_collective_bw(self) -> float:
        """One-direction per-GPU bandwidth used by the ring collective model."""
        return self.scale_up_p2p_bw_bidir / 2


@dataclass(frozen=True)
class GPUPlannerOption:
    label: str
    gpu_key: str


@dataclass(frozen=True)
class GPUCard:
    name: str
    vendor: str
    architecture: str
    vram: str
    use_case: str
    planner_options: tuple[GPUPlannerOption, ...] = ()
    note: str | None = None


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


# Capability flags. Projects can require one or more; models must supply them to be eligible.
# Kept deliberately coarse — the planner isn't a model quality benchmark, it's a capacity model.
MODEL_CAPABILITIES: tuple[str, ...] = ("tools", "ctx_128k", "images", "audio", "reasoning")
CAPABILITY_LABELS = {
    "tools":     "Tool use",
    "ctx_128k":  "≥128k ctx",
    "images":    "Image input",
    "audio":     "Audio input",
    "reasoning": "Thinking / reasoning",
}
# Every modern open-weights model supports tool-calling and long context. Multimodal
# and reasoning flags are the ones that actually discriminate.
DEFAULT_MODEL_CAPABILITIES: frozenset[str] = frozenset({"tools", "ctx_128k"})


@dataclass
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
    attention_label: str = ""
    capabilities_override: frozenset[str] | None = None
    realtime_profile: RealtimeProfile | None = None
    embedding_profile: EmbeddingProfile | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        if self.capabilities_override is not None:
            return self.capabilities_override | self.extra_capabilities
        return DEFAULT_MODEL_CAPABILITIES | self.extra_capabilities

    @property
    def is_realtime_only(self) -> bool:
        return self.realtime_profile is not None

    @property
    def is_embedding_model(self) -> bool:
        return self.embedding_profile is not None

    @property
    def size_label(self) -> str:
        def fmt_b(params: float) -> str:
            b = params / 1e9
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
    def attention_query_head_count(self) -> int:
        return self.attention_query_heads or self.num_heads

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
        retained_bpp = max(0.0, self.fp8_weight_bytes_per_param - PRECISION_SPECS["fp8"].effective_weight_bytes_per_param)
        return PRECISION_SPECS[prec].effective_weight_bytes_per_param + retained_bpp

    def uses_mixed_weight_precision(self, prec: str) -> bool:
        if get_quantization_profile(self.key, prec) is not None:
            return True
        return not math.isclose(self.weight_bytes_per_param(prec), bytes_per_param(prec), rel_tol=1e-9, abs_tol=1e-9)

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


AA_INTELLIGENCE_INDEX_MIN = 7.0
AA_INTELLIGENCE_INDEX_MAX = 51.0
AA_QUALITY_MIN = 0.30
AA_QUALITY_MAX = 0.95
AA_TOKEN_EFFICIENCY_REF_OUTPUT_TOKENS_M = 10.0
QUALITY_CONFIDENCE_PENALTY = 0.12


def aa_intelligence_to_quality(score: float) -> float:
    clipped = min(max(score, AA_INTELLIGENCE_INDEX_MIN), AA_INTELLIGENCE_INDEX_MAX)
    span = max(AA_INTELLIGENCE_INDEX_MAX - AA_INTELLIGENCE_INDEX_MIN, 1.0)
    t = (clipped - AA_INTELLIGENCE_INDEX_MIN) / span
    return AA_QUALITY_MIN + (AA_QUALITY_MAX - AA_QUALITY_MIN) * t


def aa_output_tokens_to_efficiency(output_tokens_m: float) -> float:
    return AA_TOKEN_EFFICIENCY_REF_OUTPUT_TOKENS_M / max(output_tokens_m, 0.1)


def effective_quality(model: Model) -> float:
    """Conservative quality used for routing.

    Direct benchmark scores keep their catalog quality. Proxy or uncertain scores are
    discounted so unknown tiny models do not pass workload gates only because their
    throughput is attractive.
    """
    confidence = min(max(float(getattr(model, "quality_confidence", 1.0)), 0.0), 1.0)
    return min(max(float(model.quality) - QUALITY_CONFIDENCE_PENALTY * (1.0 - confidence), 0.0), 1.0)


def model_success_rate(model: Model, difficulty: float) -> float:
    return success_rate(effective_quality(model), difficulty)


GPUS: dict[str, GPU] = {
    # Normalized to per-GPU aggregate bidirectional peer-to-peer bandwidth for each local topology.
    # AMD values use aggregate peer-to-peer bandwidth rather than raw per-link or total-transport IF figures.
    # Public NVIDIA product pages often quote sparse tensor throughput; use dense rates for planner math.
    # Ampere GPUs without native FP8 use BF16/FP16 tensor throughput for the FP8 planner path.
    "A100": GPU("A100", "A100 80GB SXM", "nv", 80e9, 2.039e12, 312e12, 312e12, 600e9, 8),
    "A100_40": GPU("A100_40", "A100 40GB PCIe", "nv", 40e9, 1.555e12, 156e12, 156e12, 64e9, 8),
    "A10": GPU("A10", "A10 24GB PCIe", "nv", 24e9, 600e9, 125e12, 125e12, 64e9, 8),
    "H100": GPU("H100", "H100 80GB SXM", "nv", 80e9, 3.35e12, 989e12, 1979e12, 900e9, 8),
    "H200": GPU("H200", "H200 141GB SXM", "nv", 141e9, 4.8e12, 989e12, 1979e12, 900e9, 8),
    "L40S": GPU("L40S", "L40S 48GB", "nv", 48e9, 864e9, 362.05e12, 733e12, 64e9, 8),
    "L4": GPU("L4", "L4 24GB", "nv", 24e9, 300e9, 121e12, 242.5e12, 64e9, 8),
    "RTXPRO6000_BSE": GPU("RTXPRO6000_BSE", "RTX PRO 6000 Blackwell Server Edition 96GB", "nv", 96e9, 1.597e12, 1e15, 2e15, 128e9, 8),
    "RTXPRO6000_BW_WS": GPU("RTXPRO6000_BW_WS", "RTX PRO 6000 Blackwell Workstation 96GB", "nv", 96e9, 1.792e12, 1e15, 2e15, 128e9, 4),
    "RTXPRO5000_BW_72": GPU("RTXPRO5000_BW_72", "RTX PRO 5000 Blackwell 72GB", "nv", 72e9, 1.344e12, 535.5e12, 1071e12, 128e9, 4),
    "RTX6000_ADA": GPU("RTX6000_ADA", "RTX 6000 Ada Generation 48GB", "nv", 48e9, 960e9, 364.25e12, 728.5e12, 64e9, 4),
    "RTX5090": GPU("RTX5090", "GeForce RTX 5090 32GB", "nv", 32e9, 1.792e12, 838e12, 1676e12, 128e9, 1),
    "RTX4090": GPU("RTX4090", "GeForce RTX 4090 24GB", "nv", 24e9, 1.008e12, 330.25e12, 660.5e12, 64e9, 1),
    "RTX3090": GPU("RTX3090", "GeForce RTX 3090 24GB", "nv", 24e9, 936e9, 142e12, 142e12, 64e9, 1),
    "DGX_SPARK": GPU("DGX_SPARK", "DGX Spark GB10 128GB", "nv", 128e9, 273e9, 125e12, 250e12, 25e9, 1),
    "GB200": GPU("GB200", "GB200 NVL72 Grace Blackwell 186GB/GPU", "nv", 186e9, 8e12, 2.5e15, 5e15, 3.6e12, 72, min_count=72, count_multiple=72),
    "B200": GPU("B200", "B200 180GB HGX/DGX", "nv", 180e9, 8e12, 2.25e15, 4.5e15, 1.8e12, 8, min_count=8, count_multiple=8),
    "B300": GPU("B300", "B300 Blackwell Ultra 288GB HGX/DGX", "nv", 288e9, 8e12, 2.5e15, 5e15, 1.8e12, 8, min_count=8, count_multiple=8),
    "GB300": GPU("GB300", "GB300 NVL72 Blackwell Ultra 288GB/GPU", "nv", 288e9, 8e12, 2.5e15, 5e15, 3.6e12, 72, min_count=72, count_multiple=72),
    "DGX_STATION_GB300": GPU("DGX_STATION_GB300", "DGX Station GB300 Blackwell Ultra 252GB", "nv", 252e9, 7.1e12, 2.5e15, 5e15, 100e9, 2),
    "A40": GPU("A40", "A40 48GB", "nv", 48e9, 696e9, 149.7e12, 149.7e12, 112.5e9, 2),
    "A30": GPU("A30", "A30 24GB", "nv", 24e9, 933e9, 165e12, 165e12, 200e9, 2),
    "A6000": GPU("A6000", "RTX A6000 48GB", "nv", 48e9, 768e9, 154.85e12, 154.85e12, 112.5e9, 2),
    "A4000": GPU("A4000", "RTX A4000 16GB", "nv", 16e9, 448e9, 76.7e12, 76.7e12, 64e9, 1),
    "A2000_MOBILE": GPU("A2000_MOBILE", "RTX A2000 Laptop GPU 8GB", "nv", 8e9, 192e9, 37.5e12, 37.5e12, 64e9, 1),
    "T4": GPU("T4", "T4 16GB", "nv", 16e9, 320e9, 65e12, 65e12, 32e9, 8),
    "V100": GPU("V100", "V100 32GB SXM2", "nv", 32e9, 900e9, 125e12, 125e12, 300e9, 8),
    "JETSON_AGX_THOR": GPU("JETSON_AGX_THOR", "Jetson AGX Thor 128GB", "nv", 128e9, 273e9, 258.75e12, 517.5e12, 25e9, 1),
    "MI250X": GPU("MI250X", "MI250X 128GB", "amd", 128e9, 3.2e12, 383e12, 383e12, 800e9, 8),
    "MI300X": GPU("MI300X", "MI300X 192GB", "amd", 192e9, 5.3e12, 1307e12, 2615e12, 896e9, 8),
    "MI325X": GPU("MI325X", "MI325X 256GB", "amd", 256e9, 6e12, 1307e12, 2615e12, 896e9, 8),
    "MI350X": GPU("MI350X", "MI350X 288GB", "amd", 288e9, 8e12, 2010e12, 4020e12, 1075.2e9, 8),
    "MI355X": GPU("MI355X", "MI355X 288GB", "amd", 288e9, 8e12, 2512e12, 5037e12, 1075.2e9, 8),
    "MI400": GPU("MI400", "MI400 Series Preview 432GB", "amd", 432e9, 19.6e12, 10e15, 20e15, 260e12 / 72.0, 72),
    "RadeonProW7900": GPU("RadeonProW7900", "Radeon PRO W7900 48GB", "amd", 48e9, 864e9, 123e12, 123e12, 64e9, 1),
    "RadeonAIProR9700": GPU("RadeonAIProR9700", "Radeon AI PRO R9700 32GB", "amd", 32e9, 640e9, 96e12, 96e12, 128e9, 1),
    # Intel does not publish dense BF16/FP8 peak figures for these public pages, so the planner
    # uses transparent proxy rooflines derived from the nearest available official disclosures.
    "Gaudi2": GPU("Gaudi2", "Gaudi 2 96GB", "intel", 96e9, 2.45e12, 432e12, 865e12, 300e9, 8),
    "Gaudi3": GPU("Gaudi3", "Gaudi 3 128GB", "intel", 128e9, 3.7e12, 1.3e15, 2.6e15, 900e9, 4),
    "CrescentIsland": GPU("CrescentIsland", "Crescent Island Preview 160GB", "intel", 160e9, 1.0e12, 183.5e12, 367e12, 128e9, 8),
    "ArcProB70": GPU("ArcProB70", "Arc Pro B70 32GB", "intel", 32e9, 608e9, 183.5e12, 367e12, 128e9, 8),
    "ArcProB60": GPU("ArcProB60", "Arc Pro B60 24GB", "intel", 24e9, 456e9, 98.5e12, 197e12, 64e9, 8),
    "ArcProB50": GPU("ArcProB50", "Arc Pro B50 16GB", "intel", 16e9, 224e9, 85e12, 170e12, 64e9, 8),
    # Apple publishes peak unified-memory bandwidth and GPU-core counts, but not dense BF16/FP8
    # tensor rooflines or sustained inference bandwidth. We therefore keep Apple's peak bandwidth
    # for display, while planner_bw + BF16/FP8 proxies are calibrated conservatively against
    # whatcani.run Apple-device decode/prefill scaling across MLX/GGUF runs.
    "MAC_MINI_M4_PRO": GPU("MAC_MINI_M4_PRO", "Mac mini M4 Pro 64GB", "apple", 64e9, 273e9, 16e12, 16e12, 50e9, 1, 273e9),
    "MAC_STUDIO_M4_MAX": GPU("MAC_STUDIO_M4_MAX", "Mac Studio M4 Max 128GB", "apple", 128e9, 546e9, 26e12, 26e12, 50e9, 1, 410e9),
    "MAC_STUDIO_M3_ULTRA": GPU("MAC_STUDIO_M3_ULTRA", "Mac Studio M3 Ultra 512GB", "apple", 512e9, 819e9, 48e12, 48e12, 50e9, 1, 560e9),
}

GPU_FP4_FLOPS = {
    # Native dense FP4 tensor paths. Sparse marketing figures are intentionally not used.
    "RTXPRO6000_BSE": 4e15,
    "RTXPRO6000_BW_WS": 4e15,
    "RTXPRO5000_BW_72": 2142e12,
    "RTX5090": 3352e12,
    "DGX_SPARK": 500e12,
    "GB200": 10e15,
    "B200": 9e15,
    "B300": 15e15,
    "GB300": 15e15,
    "DGX_STATION_GB300": 15e15,
    "JETSON_AGX_THOR": 1035e12,
    "MI350X": 9.2e15,
    "MI355X": 10.1e15,
    "MI400": 40e15,
}
for _k, _fp4 in GPU_FP4_FLOPS.items():
    if _k in GPUS:
        GPUS[_k].fp4 = float(_fp4)

# Published board TDPs (watts). Used with a utilization factor to estimate per-task energy.
# Sources: vendor product pages; Mac figures use whole-system measured peak.
GPU_TDP_WATTS = {
    "A100": 400, "A100_40": 250, "A10": 150, "H100": 700, "H200": 700, "L40S": 350, "L4": 72,
    "RTXPRO6000_BSE": 600, "RTXPRO6000_BW_WS": 600, "RTXPRO5000_BW_72": 300, "RTX6000_ADA": 300,
    "RTX5090": 575, "RTX4090": 450, "RTX3090": 350, "DGX_SPARK": 140, "GB200": 1200, "B200": 1000, "B300": 1400,
    "GB300": 1400, "DGX_STATION_GB300": 1600,
    "A40": 300, "A30": 165, "A6000": 300, "A4000": 140, "A2000_MOBILE": 95, "T4": 70, "V100": 300, "JETSON_AGX_THOR": 130,
    "MI250X": 560, "MI300X": 750, "MI325X": 1000, "MI350X": 1000, "MI355X": 1400, "MI400": 1500,
    "RadeonProW7900": 295, "RadeonAIProR9700": 300,
    "Gaudi2": 600, "Gaudi3": 900, "CrescentIsland": 300, "ArcProB70": 230, "ArcProB60": 200, "ArcProB50": 70,
    "MAC_MINI_M4_PRO": 140, "MAC_STUDIO_M4_MAX": 270, "MAC_STUDIO_M3_ULTRA": 480,
}
for _k, _w in GPU_TDP_WATTS.items():
    if _k in GPUS:
        GPUS[_k].tdp_watts = float(_w)


def normalize_gpu_count(gpu_type: str, count: int, allow_zero: bool = False) -> int:
    """Snap pool sizes to set-only hardware constraints."""
    try:
        normalized = int(count)
    except (TypeError, ValueError):
        normalized = 0
    if allow_zero and normalized <= 0:
        return 0

    gpu = GPUS.get(gpu_type)
    if gpu is None:
        return max(normalized, 0)

    normalized = max(normalized, max(int(gpu.min_count), 1))
    multiple = max(int(gpu.count_multiple), 1)
    if multiple > 1:
        normalized = math.ceil(normalized / multiple) * multiple
    return normalized


GPU_CARDS: list[GPUCard] = [
    # ── NVIDIA: flagship Blackwell → Hopper → Ampere datacenter → Ada → professional → desktop ──
    GPUCard(
        "GB300 NVL72",
        "NVIDIA",
        "Blackwell Ultra",
        "72-GPU rack: 288 GB HBM3e/GPU",
        "Rack-scale AI reasoning, training, and high-density inference",
        (GPUPlannerOption("Add 72-GPU Rack", "GB300"),),
        "Rack-only profile: models one Blackwell Ultra GPU inside the 72-GPU GB300 NVL72 domain. Pool sizes snap to multiples of 72; CPU LPDDR5X memory is not counted as GPU memory.",
    ),
    GPUCard(
        "DGX Station GB300",
        "NVIDIA",
        "Blackwell Ultra",
        "252 GB HBM3e + 496 GB LPDDR5X",
        "Deskside AI development, local fine-tuning, and large-model inference",
        (GPUPlannerOption("Add Station", "DGX_STATION_GB300"),),
        "System-only GB300 desktop superchip profile. Planner capacity uses the 252GB GPU HBM pool; coherent CPU memory is noted but not counted for model weights or KV cache.",
    ),
    GPUCard(
        "GB200 NVL72",
        "NVIDIA",
        "Blackwell",
        "72-GPU rack: 186 GB HBM3e/GPU",
        "AI supercomputing, large-model training",
        (GPUPlannerOption("Add 72-GPU Rack", "GB200"),),
        "Rack-only profile: models one Blackwell GPU inside the 72-GPU GB200 NVL72 domain. Pool sizes snap to multiples of 72.",
    ),
    GPUCard(
        "B300 / Blackwell Ultra",
        "NVIDIA",
        "Blackwell Ultra",
        "8-GPU system: 288 GB HBM3e/GPU",
        "HGX/DGX Blackwell Ultra systems and GB300 rack components",
        (GPUPlannerOption("Add 8-GPU System", "B300"),),
        "System-only profile for HGX/DGX B300 class servers. Pool sizes snap to multiples of 8; use GB300 NVL72 for the rack-scale Grace Blackwell Ultra domain.",
    ),
    GPUCard(
        "B200",
        "NVIDIA",
        "Blackwell",
        "8-GPU system: 180 GB HBM3e/GPU",
        "AI training/inference, scaling beyond Hopper",
        (GPUPlannerOption("Add 8-GPU System", "B200"),),
        "System-only profile for HGX/DGX B200 class servers. Pool sizes snap to multiples of 8.",
    ),
    GPUCard(
        "H200 SXM/PCIe",
        "NVIDIA",
        "Hopper",
        "141 GB HBM3e",
        "Large-scale LLM inference, HPC, memory-bound workloads",
        (GPUPlannerOption("Add", "H200"),),
        "Planner uses the calibrated H200 141 GB SXM profile.",
    ),
    GPUCard(
        "H100 SXM/PCIe",
        "NVIDIA",
        "Hopper",
        "80 GB HBM3",
        "AI training & inference, general-purpose accelerator",
        (GPUPlannerOption("Add", "H100"),),
        "Planner uses the calibrated H100 80 GB SXM profile.",
    ),
    GPUCard(
        "A100 80GB SXM",
        "NVIDIA",
        "Ampere",
        "80 GB HBM2e",
        "Training, inference, ML workloads (still widely available)",
        (GPUPlannerOption("Add", "A100"),),
        "Calibrated A100 80 GB SXM planner profile.",
    ),
    GPUCard(
        "A100 40GB PCIe",
        "NVIDIA",
        "Ampere",
        "40 GB HBM2e",
        "Lower-cost Ampere option, PCIe slot-in",
        (GPUPlannerOption("Add", "A100_40"),),
        "Calibrated A100 40 GB PCIe planner profile.",
    ),
    GPUCard(
        "V100 32GB SXM2",
        "NVIDIA",
        "Volta",
        "32 GB HBM2",
        "Legacy NVLink-connected training and budget inference nodes",
        (GPUPlannerOption("Add", "V100"),),
        "Uses NVIDIA's 32GB SXM2 tensor and NVLink profile; FP8 planner path falls back to FP16 tensor throughput.",
    ),
    GPUCard(
        "A40",
        "NVIDIA",
        "Ampere",
        "48 GB GDDR6 ECC",
        "Data center visual compute, vGPU, and large-memory inference",
        (GPUPlannerOption("Add", "A40"),),
        "Uses NVIDIA's dense BF16/FP16 tensor peak with the 48GB GDDR6 and 2-way NVLink profile.",
    ),
    GPUCard(
        "A30",
        "NVIDIA",
        "Ampere",
        "24 GB HBM2",
        "Mainstream data center training/inference with MIG and NVLink",
        (GPUPlannerOption("Add", "A30"),),
        "Uses NVIDIA's dense BF16/FP16 tensor peak; FP8 planner path falls back to Ampere tensor throughput.",
    ),
    GPUCard(
        "A10",
        "NVIDIA",
        "Ampere",
        "24 GB GDDR6",
        "Mainstream enterprise inference, vGPU, graphics, and video workloads",
        (GPUPlannerOption("Add", "A10"),),
        "Uses NVIDIA's dense BF16/FP16 tensor peak with the 24GB GDDR6 and PCIe Gen4 profile.",
    ),
    GPUCard(
        "L40S",
        "NVIDIA",
        "Ada Lovelace",
        "48 GB GDDR6",
        "Mixed AI/graphics, rendering, video, digital twins",
        (GPUPlannerOption("Add", "L40S"),),
        "Uses the public NVIDIA L40S dense tensor and memory specs.",
    ),
    GPUCard(
        "L4",
        "NVIDIA",
        "Ada Lovelace",
        "24 GB GDDR6",
        "Video transcoding, light inference, virtual desktops",
        (GPUPlannerOption("Add", "L4"),),
        "Uses the public NVIDIA L4 dense tensor and memory specs.",
    ),
    GPUCard(
        "T4",
        "NVIDIA",
        "Turing",
        "16 GB GDDR6",
        "Low-power legacy cloud inference and video workloads",
        (GPUPlannerOption("Add", "T4"),),
        "Uses NVIDIA's FP16 tensor peak as the planner proxy; BF16/FP8 are not native on Turing.",
    ),
    GPUCard(
        "RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA",
        "Blackwell",
        "96 GB GDDR7",
        "Graphics-intensive AI, virtual workstations (GCP)",
        (GPUPlannerOption("Add", "RTXPRO6000_BSE"),),
        "Planner uses the NVIDIA Server Edition memory and tensor figures; modeled as an 8-GPU PCIe server topology.",
    ),
    GPUCard(
        "RTX PRO 6000 Blackwell Workstation Edition",
        "NVIDIA",
        "Blackwell",
        "96 GB GDDR7 ECC",
        "High-end local AI, rendering, and workstation model serving",
        (GPUPlannerOption("Add", "RTXPRO6000_BW_WS"),),
        "Uses the workstation 96GB/1.792TB/s profile with Blackwell FP4 tensor support; modeled as a 4-GPU PCIe workstation topology.",
    ),
    GPUCard(
        "RTX PRO 5000 Blackwell 72GB",
        "NVIDIA",
        "Blackwell",
        "72 GB GDDR7 ECC",
        "Large local inference and agentic AI workstations below RTX PRO 6000",
        (GPUPlannerOption("Add", "RTXPRO5000_BW_72"),),
        "Uses the 72GB RTX PRO 5000 memory profile and published Blackwell AI throughput ratio.",
    ),
    GPUCard(
        "RTX 6000 Ada Generation",
        "NVIDIA",
        "Ada Lovelace",
        "48 GB GDDR6 ECC",
        "Professional workstation AI, rendering, simulation, and visualization",
        (GPUPlannerOption("Add", "RTX6000_ADA"),),
        "Uses the dense half of NVIDIA's effective sparse FP8 tensor figure and the public 960GB/s memory spec.",
    ),
    GPUCard(
        "GeForce RTX 5090",
        "NVIDIA",
        "Blackwell",
        "32 GB GDDR7",
        "Prosumer/local AI inference, experimentation, and high-end desktop workloads",
        (GPUPlannerOption("Add", "RTX5090"),),
        "Uses the 32GB/1.792TB/s Founders Edition memory profile and Blackwell FP4 AI throughput ratio; modeled as a single-GPU desktop card.",
    ),
    GPUCard(
        "GeForce RTX 4090",
        "NVIDIA",
        "Ada Lovelace",
        "24 GB GDDR6X",
        "Common local inference/development baseline",
        (GPUPlannerOption("Add", "RTX4090"),),
        "Uses the 24GB/1.008TB/s Founders Edition memory profile and dense Ada tensor proxy.",
    ),
    GPUCard(
        "GeForce RTX 3090",
        "NVIDIA",
        "Ampere",
        "24 GB GDDR6X",
        "Common local inference/development baseline with broad used-market availability",
        (GPUPlannerOption("Add", "RTX3090"),),
        "Uses the 24GB/936GB/s Founders Edition memory profile; FP8 planner path falls back to the Ampere FP16 tensor proxy.",
    ),
    GPUCard(
        "RTX A6000",
        "NVIDIA",
        "Ampere",
        "48 GB GDDR6 ECC",
        "Workstation inference and development, with 2-way NVLink",
        (GPUPlannerOption("Add", "A6000"),),
        "Planner uses the dense half of NVIDIA's sparse tensor figure and the 2-way NVLink bandwidth.",
    ),
    GPUCard(
        "RTX A4000",
        "NVIDIA",
        "Ampere",
        "16 GB GDDR6",
        "Entry workstation GPU for lightweight inference and development",
        (GPUPlannerOption("Add", "A4000"),),
    ),
    GPUCard(
        "RTX A2000 Laptop GPU",
        "NVIDIA",
        "Ampere",
        "up to 8 GB GDDR6",
        "Mobile workstation GPU for lightweight local inference and development",
        (GPUPlannerOption("Add", "A2000_MOBILE"),),
        "Planner uses the top-bin 8GB/192GB/s/95W mobile profile and dense half of the published tensor peak; OEM configs also include 4GB and lower-TGP variants.",
    ),
    GPUCard(
        "DGX Spark",
        "NVIDIA",
        "Grace Blackwell",
        "128 GB LPDDR5x unified memory",
        "Desktop AI supercomputer for local prototyping, inference, and fine-tuning",
        (GPUPlannerOption("Add", "DGX_SPARK"),),
        "Planner uses the GB10 128GB/273GB/s profile. Dense BF16/FP8 rooflines are derived from NVIDIA's published sparse FP4 figure.",
    ),
    # ── AMD: newest generation first ────────────────────────────────────────
    GPUCard(
        "MI400 series",
        "AMD",
        "CDNA 5 (projected)",
        "432 GB HBM4, ~19.6 TB/s",
        "Expected H2 2026, 2× perf over MI355X",
        (GPUPlannerOption("Add Preview", "MI400"),),
        "Preview profile based on AMD's public MI400-series roadmap disclosures.",
    ),
    GPUCard(
        "MI355X",
        "AMD",
        "CDNA 4",
        "288 GB HBM3e, 8 TB/s",
        "Higher FP8/FP4 throughput variant of MI350X",
        (GPUPlannerOption("Add", "MI355X"),),
        "Calibrated MI355X planner profile.",
    ),
    GPUCard(
        "MI350X",
        "AMD",
        "CDNA 4",
        "288 GB HBM3e, 8 TB/s",
        "Generative AI & HPC, FP4/FP6 support (June 2025)",
        (GPUPlannerOption("Add", "MI350X"),),
        "Calibrated MI350X planner profile.",
    ),
    GPUCard(
        "MI325X",
        "AMD",
        "CDNA 3",
        "256 GB HBM3e, 6 TB/s",
        "Extra capacity for LLM serving",
        (GPUPlannerOption("Add", "MI325X"),),
    ),
    GPUCard(
        "MI300X",
        "AMD",
        "CDNA 3",
        "192 GB HBM3, 5.3 TB/s",
        "H100 competitor, large model serving",
        (GPUPlannerOption("Add", "MI300X"),),
    ),
    GPUCard(
        "MI250X",
        "AMD",
        "CDNA 2",
        "64 GB HBM2e",
        "General-purpose HPC and AI inference",
        (GPUPlannerOption("Add", "MI250X"),),
        "Planner models the full MI250X accelerator at 128GB; the 64GB figure commonly refers to one GCD.",
    ),
    GPUCard(
        "Radeon AI PRO R9700",
        "AMD",
        "RDNA 4",
        "32 GB GDDR6",
        "Affordable local AI workstation and multi-GPU inference builds",
        (GPUPlannerOption("Add", "RadeonAIProR9700"),),
        "Uses AMD's public 32GB/640GB/s profile; BF16/FP8 planner paths use the published FP16 matrix throughput proxy.",
    ),
    GPUCard(
        "Radeon PRO W7900",
        "AMD",
        "RDNA 3",
        "48 GB GDDR6",
        "Large-memory workstation graphics, visualization, and local inference",
        (GPUPlannerOption("Add", "RadeonProW7900"),),
        "Uses AMD's public 48GB profile and FP16 matrix throughput as the planner proxy.",
    ),
    # ── Intel ────────────────────────────────────────────────────────────────
    GPUCard(
        "Gaudi 3",
        "Intel",
        "Gaudi",
        "8.2 TB rack-scale HBM",
        "Scalable enterprise/cloud inference, up to 64 accelerators per rack",
        (GPUPlannerOption("Add", "Gaudi3"),),
        "Uses Intel's public 128GB/3.7TB/s Gaudi 3 card specs with a provisional BF16/FP8 roofline.",
    ),
    GPUCard(
        "Gaudi 2",
        "Intel",
        "Gaudi",
        "96 GB HBM2e",
        "Prior-generation AI training/inference accelerator with Ethernet scale-out",
        (GPUPlannerOption("Add", "Gaudi2"),),
        "Uses Intel's 96GB/2.45TB/s Gaudi 2 profile and published BF16/FP8 matrix throughput.",
    ),
    GPUCard(
        "GPU Crescent Island",
        "Intel",
        "Xe3P",
        "160 GB LPDDR5X",
        "Inference & tokens-as-a-service, air-cooled (announced Oct 2025)",
        (GPUPlannerOption("Add Preview", "CrescentIsland"),),
        "Preview proxy profile: Intel has announced memory capacity, but not a full public roofline yet.",
    ),
    GPUCard(
        "Arc Pro B70",
        "Intel",
        "Xe2",
        "32 GB GDDR6",
        "High-memory local AI workstation GPU",
        (GPUPlannerOption("Add", "ArcProB70"),),
        "Uses public Intel 32GB/608GB/s specs; BF16/FP8 planner rooflines are inferred from Intel's published INT8 XMX throughput.",
    ),
    GPUCard(
        "Arc Pro B60",
        "Intel",
        "Xe2",
        "24 GB",
        "Edge-cloud/multi-GPU server, up to 8× for 150B param models",
        (GPUPlannerOption("Add", "ArcProB60"),),
        "Uses public Intel memory specs; BF16/FP8 planner rooflines are inferred from Intel's published INT8 XMX throughput.",
    ),
    GPUCard(
        "Arc Pro B50",
        "Intel",
        "Xe2",
        "16 GB",
        "Lighter edge inference option",
        (GPUPlannerOption("Add", "ArcProB50"),),
        "Uses public Intel memory specs; BF16/FP8 planner rooflines are inferred from Intel's published INT8 XMX throughput.",
    ),
    # ── Apple: most memory first ─────────────────────────────────────────────
    GPUCard(
        "Mac Studio M3 Ultra",
        "Apple",
        "Apple silicon",
        "up to 512 GB unified memory",
        "Largest-memory Apple desktop for local large-model serving and experimentation",
        (GPUPlannerOption("Add 512GB", "MAC_STUDIO_M3_ULTRA"),),
        "Planner uses the 80-core GPU / 512GB Mac Studio M3 Ultra top bin. Peak bandwidth comes from Apple specs; planner math is conservatively scaled from whatcani.run's benchmarked 60-core M3 Ultra runs.",
    ),
    GPUCard(
        "Mac Studio M4 Max",
        "Apple",
        "Apple silicon",
        "up to 128 GB unified memory",
        "Single-box model serving, creative/ML workstation, local team node",
        (GPUPlannerOption("Add 128GB", "MAC_STUDIO_M4_MAX"),),
        "Planner uses the 40-core GPU / 128GB Mac Studio M4 Max config. Peak bandwidth comes from Apple specs; planner math uses a lower sustained-bandwidth proxy to match observed M4 Pro to M4 Max scaling on whatcani.run.",
    ),
    GPUCard(
        "Mac mini M4 Pro",
        "Apple",
        "Apple silicon",
        "up to 64 GB unified memory",
        "Local inference, eval runners, compact dev/prototyping box",
        (GPUPlannerOption("Add 64GB", "MAC_MINI_M4_PRO"),),
        "Planner uses the top-bin 64GB Mac mini M4 Pro profile. Peak bandwidth comes from Apple specs; compute and sustained-bandwidth math are conservative proxies cross-checked against whatcani.run.",
    ),
    # ── Edge / Embedded ──────────────────────────────────────────────────────
    GPUCard(
        "Jetson AGX Thor Developer Kit",
        "Edge / Embedded",
        "Blackwell",
        "128 GB LPDDR5X",
        "Robotics and edge AI in a 40-130 W power envelope",
        (GPUPlannerOption("Add", "JETSON_AGX_THOR"),),
        "Kept outside the main NVIDIA accelerator list. Dense BF16/FP8 rooflines are derived from NVIDIA's published sparse FP4 figure.",
    ),
]


RWKV7_G1_HEAD_DIM = 64
RWKV7_G1_CONTEXT = 8192
RWKV7_G1_BASE_CAPABILITIES = frozenset({"reasoning"})
RWKV7_G1_TOOL_CAPABILITIES = frozenset({"tools", "reasoning"})
VOXTRAL_REALTIME_PROFILE = RealtimeProfile(
    label="Realtime ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=480,
    state_tokens=8192,
    source="mistralai/Voxtral-Mini-4B-Realtime-2602",
    note="Realtime stream demand uses 6 delay tokens over 480 ms, i.e. 12.5 streaming ticks/sec. Each 80 ms tick also runs 4 causal audio-encoder tokens.",
    audio_encoder_params=0.97e9,
    audio_tokens_per_step=4,
    audio_attention_layers=32,
    audio_attention_heads=32,
    audio_attention_head_dim=64,
    audio_attention_window=750,
)
MIMO_V25_ASR_PROFILE = RealtimeProfile(
    label="ASR",
    tokens_per_second=6.25,
    audio_ms_per_token=160.0,
    target_delay_ms=1600,
    state_tokens=8192,
    source="XiaomiMiMo/MiMo-V2.5-ASR + XiaomiMiMo/MiMo-Audio-Tokenizer",
    note="ASR stream demand approximates MiMo-Audio-Tokenizer's 25 Hz RVQ stream after the 4-frame patch grouping used by the ASR model, i.e. 6.25 audio patches/sec. The companion tokenizer is modeled as extra audio work.",
    audio_encoder_params=1.2e9,
    audio_tokens_per_step=4,
    audio_attention_layers=32,
    audio_attention_heads=20,
    audio_attention_head_dim=64,
    audio_attention_window=25,
)
NEMOTRON_SPEECH_STREAMING_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=1000.0 / 560.0,
    audio_ms_per_token=560.0,
    target_delay_ms=560,
    state_tokens=70,
    source="nvidia/nemotron-speech-streaming-en-0.6b",
    note="Cache-aware FastConformer-RNNT profile uses the 560 ms streaming chunk setting; the cached left context is 70 80 ms frames.",
)
NEMOTRON_35_ASR_STREAMING_PROFILE = RealtimeProfile(
    label="Multilingual Streaming ASR",
    tokens_per_second=1000.0 / 560.0,
    audio_ms_per_token=560.0,
    target_delay_ms=560,
    state_tokens=56,
    source="nvidia/nemotron-3.5-asr-streaming-0.6b",
    note="Prompt-conditioned cache-aware FastConformer-RNNT profile uses the 560 ms streaming chunk setting from att_context_size [56,6]; the cached left context is 56 80 ms frames.",
)
PARAKEET_UNIFIED_STREAMING_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=1000.0 / 560.0,
    audio_ms_per_token=560.0,
    target_delay_ms=560,
    state_tokens=70,
    source="nvidia/parakeet-unified-en-0.6b",
    note="Unified FastConformer-RNNT profile uses the published 0.56 s streaming latency point with 5.6 s left context.",
)
PARAKEET_REALTIME_EOU_PROFILE = RealtimeProfile(
    label="Streaming ASR + EOU",
    tokens_per_second=6.25,
    audio_ms_per_token=160.0,
    target_delay_ms=160,
    state_tokens=70,
    source="nvidia/parakeet_realtime_eou_120m-v1",
    note="Voice-agent streaming profile uses the 160 ms setting from the model card and keeps the 70-frame cache-aware left context.",
)
MULTITALKER_PARAKEET_STREAMING_PROFILE = RealtimeProfile(
    label="Streaming Multitalker ASR",
    tokens_per_second=1000.0 / 1120.0,
    audio_ms_per_token=1120.0,
    target_delay_ms=1120,
    state_tokens=70,
    source="nvidia/multitalker-parakeet-streaming-0.6b-v1",
    note="Multitalker profile uses the 1.12 s setting; NVIDIA documents one ASR model instance per active speaker, so planner users should scale assignments by target speaker count.",
)
KYUTAI_STT_1B_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=500,
    state_tokens=750,
    source="kyutai/stt-1b-en_fr",
    note="Kyutai delayed-streams profile uses the published 12.5 Hz Mimi frame rate, 32 audio tokens per frame, and 0.5 s text delay.",
)
KYUTAI_STT_2_6B_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=2500,
    state_tokens=375,
    source="kyutai/stt-2.6b-en",
    note="Kyutai delayed-streams profile uses the published 12.5 Hz Mimi frame rate and 2.5 s text delay; the Transformers config uses a 375-token sliding window.",
)
MOONSHINE_STREAMING_TINY_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=80,
    state_tokens=16,
    source="UsefulSensors/moonshine-streaming-tiny",
    note="Moonshine Streaming uses 50 Hz frontend features with stride-4 downsampling and bounded 16-frame encoder windows; the first/last layers add 80 ms lookahead.",
)
MOONSHINE_STREAMING_SMALL_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=80,
    state_tokens=16,
    source="UsefulSensors/moonshine-streaming-small",
    note="Moonshine Streaming uses 50 Hz frontend features with stride-4 downsampling and bounded 16-frame encoder windows; the first/last layers add 80 ms lookahead.",
)
MOONSHINE_STREAMING_MEDIUM_PROFILE = RealtimeProfile(
    label="Streaming ASR",
    tokens_per_second=12.5,
    audio_ms_per_token=80.0,
    target_delay_ms=80,
    state_tokens=16,
    source="UsefulSensors/moonshine-streaming-medium",
    note="Moonshine Streaming uses 50 Hz frontend features with stride-4 downsampling and bounded 16-frame encoder windows; the first/last layers add 80 ms lookahead.",
)
FUN_ASR_NANO_PROFILE = RealtimeProfile(
    label="Realtime ASR",
    tokens_per_second=6.25,
    audio_ms_per_token=160.0,
    target_delay_ms=160,
    state_tokens=8192,
    source="FunAudioLLM/Fun-ASR-Nano-2512",
    note="Fun-ASR-Nano is published as a low-latency realtime 800M ASR model; planner timing uses a conservative 160 ms streaming tick because no comparable chunk table is published.",
)
GRANITE_4_1B_SPEECH_PROFILE = RealtimeProfile(
    label="Offline ASR",
    tokens_per_second=1.0,
    audio_ms_per_token=1000.0,
    target_delay_ms=30000,
    state_tokens=8192,
    source="ibm-granite/granite-4.0-1b-speech",
    note="Granite Speech is modeled as high-throughput chunked ASR rather than native streaming; one planner tick approximates one audio-second equivalent of chunk processing.",
    streaming=False,
)
PARAKEET_TDT_06B_V3_PROFILE = RealtimeProfile(
    label="Offline ASR",
    tokens_per_second=1.0,
    audio_ms_per_token=1000.0,
    target_delay_ms=30000,
    state_tokens=8192,
    source="nvidia/parakeet-tdt-0.6b-v3",
    note="Parakeet TDT v3 is modeled as high-throughput non-streaming ASR; one planner tick approximates one audio-second equivalent of chunk processing.",
    streaming=False,
)


# ---------------------------------------------------------------------------
# Published ASR quality: word error rate (WER) by benchmark/language, in
# percent; lower is better. Used by the "ASR Quality" plot (max streams vs
# WER). Capacity is benchmark-independent in this closed-form model, so each
# benchmark point for a model sits at the same max-stream height.
#
# Sources:
# - https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
#   FLEURS row for 480 ms streaming WER.
# - https://mimo.xiaomi.com/mimo-v2-5-asr
#   Xiaomi-published Open ASR English average. MiMo is documented as
#   Chinese-English; no primary French WER was found.
# - https://artificialanalysis.ai/speech-to-text/batch#error-rate
#   AA-WER v2 is an English batch score over AA-AgentTalk, VoxPopuli-Cleaned-AA,
#   and Earnings22-Cleaned-AA. The closest source-backed French public rows
#   used here are CoVoST for short-form prompted speech, FLEURS for formal
#   read-speech, and MLS for long-form speech. VoxPopuli-fr would be the closest
#   French parliamentary match, but the current public model table used below
#   does not publish matching rows for this catalog's models.
# - NVIDIA, Kyutai, Moonshine, IBM, and Hugging Face model cards / Open ASR
#   leaderboard rows for the added open/self-hosted ASR models below.
# - https://huggingface.co/datasets/Steveeeeeeen/multilingual_evals
#   Hugging Face Open ASR multilingual CSV for French CoVoST/MLS/FLEURS rows
#   where the catalog model has a current public result.
# - https://arxiv.org/abs/2603.11243
#   IBM Granite 4.0 Speech paper, CommonVoice French full-AR WER.
# French rows are filled only when a source-backed French WER was found; models
# that are English-only or have no published French table intentionally omit the
# French benchmark keys unless the whole model is marked placeholder.
# ---------------------------------------------------------------------------
ASR_WER_LANGUAGES: tuple[str, ...] = (
    "en",
    "fr_covost",
    "fr_fleurs",
    "fr_mls",
    "fr_commonvoice",
)
ASR_WER_LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    # Legacy aggregate label kept for imports/tests; not included in
    # ASR_WER_LANGUAGES because the chart now plots the component French rows.
    "fr": "French aggregate",
    "fr_covost": "French CoVoST",
    "fr_fleurs": "French FLEURS",
    "fr_mls": "French MLS",
    "fr_commonvoice": "French Common Voice",
}
ASR_WER_LANGUAGE_SOURCES: dict[str, dict[str, str]] = {
    "voxtral-realtime-mini-4b": {
        "en": "Mistral FLEURS benchmark, 480 ms streaming delay, English WER.",
        "fr": "Legacy aggregate mean over French CoVoST, FLEURS, and MLS rows.",
        "fr_covost": "Hugging Face Open ASR multilingual eval, French CoVoST row; closest current source-backed proxy for AA-AgentTalk-style short prompted speech.",
        "fr_fleurs": "Hugging Face Open ASR multilingual eval, French FLEURS row; closest current source-backed French formal/read-speech proxy.",
        "fr_mls": "Hugging Face Open ASR multilingual eval, French MLS row; closest current source-backed long-form French proxy.",
    },
    "mimo-v2.5-asr": {
        "en": "Xiaomi MiMo General English Recognition Open ASR average WER.",
    },
    "nvidia-nemotron-speech-streaming-0.6b": {
        "en": "NVIDIA comparison table, HuggingFace OpenASR average WER at 0.56 s streaming latency.",
    },
    "nvidia-nemotron-3.5-asr-streaming-0.6b": {
        "en": "NVIDIA Nemotron 3.5 ASR model card, FLEURS English WER at 560 ms LangID streaming chunk.",
        "fr_fleurs": "NVIDIA Nemotron 3.5 ASR model card, FLEURS French WER at 560 ms LangID streaming chunk.",
    },
    "nvidia-parakeet-unified-0.6b": {
        "en": "NVIDIA comparison table, HuggingFace OpenASR average WER at 0.56 s streaming latency.",
    },
    "nvidia-parakeet-realtime-eou-120m": {
        "en": "NVIDIA model card, HuggingFace OpenASR average WER at 160 ms streaming setting.",
    },
    "nvidia-multitalker-parakeet-streaming-0.6b": {
        "en": "NVIDIA model card, single-speaker-mode HuggingFace OpenASR average WER.",
    },
    "kyutai-stt-1b-en-fr": {
        "en": "Placeholder: Kyutai publishes model latency and throughput, but no text WER table was found.",
        "fr": "Placeholder legacy aggregate: Kyutai publishes model latency and throughput, but no text WER table was found.",
        "fr_covost": "Placeholder French CoVoST proxy: Kyutai publishes model latency and throughput, but no text WER table was found.",
    },
    "kyutai-stt-2.6b-en": {
        "en": "Kyutai Hugging Face evaluation, HuggingFace OpenASR mean WER.",
    },
    "moonshine-streaming-tiny": {
        "en": "Useful Sensors Moonshine Streaming model card, Open ASR average WER.",
    },
    "moonshine-streaming-small": {
        "en": "Useful Sensors Moonshine Streaming model card, Open ASR average WER.",
    },
    "moonshine-streaming-medium": {
        "en": "Useful Sensors Moonshine Streaming model card, Open ASR average WER.",
    },
    "fun-asr-nano-2512": {
        "en": "Placeholder: FunAudioLLM publishes realtime capability and parameter count, but no comparable OpenASR WER was found.",
    },
    "granite-4.0-1b-speech": {
        "en": "IBM Granite model card, HuggingFace OpenASR average WER.",
        "fr": "Legacy aggregate alias for IBM Granite 4.0 Speech paper, CommonVoice French full-AR WER.",
        "fr_commonvoice": "IBM Granite 4.0 Speech paper, CommonVoice French full-AR WER.",
    },
    "nvidia-parakeet-tdt-0.6b-v3": {
        "en": "NVIDIA model card, HuggingFace OpenASR average WER.",
        "fr": "Legacy aggregate mean over French CoVoST, FLEURS, and MLS rows.",
        "fr_covost": "Hugging Face Open ASR multilingual eval, French CoVoST row; closest current source-backed proxy for AA-AgentTalk-style short prompted speech.",
        "fr_fleurs": "Hugging Face Open ASR multilingual eval, French FLEURS row; closest current source-backed French formal/read-speech proxy.",
        "fr_mls": "Hugging Face Open ASR multilingual eval, French MLS row; closest current source-backed long-form French proxy.",
    },
}
PUBLISHED_ASR_WER: dict[str, dict[str, float]] = {
    "voxtral-realtime-mini-4b": {
        "en": 4.90,
        "fr": 7.92,
        "fr_covost": 9.68,
        "fr_fleurs": 8.44,
        "fr_mls": 5.64,
    },
    "mimo-v2.5-asr": {
        "en": 5.73,
    },
    "nvidia-nemotron-speech-streaming-0.6b": {
        "en": 7.09,
    },
    "nvidia-nemotron-3.5-asr-streaming-0.6b": {
        "en": 7.99,
        "fr_fleurs": 9.45,
    },
    "nvidia-parakeet-unified-0.6b": {
        "en": 6.52,
    },
    "nvidia-parakeet-realtime-eou-120m": {
        "en": 9.30,
    },
    "nvidia-multitalker-parakeet-streaming-0.6b": {
        "en": 7.44,
    },
    "kyutai-stt-1b-en-fr": {
        "en": 7.00,
        "fr": 7.50,
        "fr_covost": 7.50,
    },
    "kyutai-stt-2.6b-en": {
        "en": 6.40,
    },
    "moonshine-streaming-tiny": {
        "en": 12.01,
    },
    "moonshine-streaming-small": {
        "en": 7.84,
    },
    "moonshine-streaming-medium": {
        "en": 6.65,
    },
    "fun-asr-nano-2512": {
        "en": 7.00,
    },
    "granite-4.0-1b-speech": {
        "en": 5.52,
        "fr": 7.15,
        "fr_commonvoice": 7.15,
    },
    "nvidia-parakeet-tdt-0.6b-v3": {
        "en": 6.34,
        "fr": 5.42,
        "fr_covost": 6.38,
        "fr_fleurs": 4.76,
        "fr_mls": 5.12,
    },
}
ASR_WER_PLACEHOLDER: frozenset[str] = frozenset({
    "kyutai-stt-1b-en-fr",
    "fun-asr-nano-2512",
})

# Backward-compatible aliases for older chart/import names.
ASR_WER_BENCHMARKS = ASR_WER_LANGUAGES
ASR_WER_BENCHMARK_LABELS = ASR_WER_LANGUAGE_LABELS
ASR_WER_BENCHMARK_SOURCES = ASR_WER_LANGUAGE_SOURCES
STREAMING_WER_LANGUAGES = ASR_WER_LANGUAGES
STREAMING_WER_LANGUAGE_LABELS = ASR_WER_LANGUAGE_LABELS
STREAMING_ASR_WER = PUBLISHED_ASR_WER
STREAMING_ASR_WER_PLACEHOLDER = ASR_WER_PLACEHOLDER


# ---------------------------------------------------------------------------
# Published embedding retrieval quality, in [0, 1]. Used by the "Embedding
# Quality" plot (quality vs docs/s, one dot per model). Scores are published
# average retrieval benchmarks converted from percent to [0, 1]. Source labels
# name the exact benchmark because English-only, multilingual, dense, and
# late-interaction models do not all publish the same aggregate or metric set.
# ---------------------------------------------------------------------------
EMBEDDING_QUALITY_SOURCES: dict[str, str] = {
    "denseon": "LightOn DenseOn HF model card, BEIR average nDCG@10 table.",
    "lateon": "LightOn LateOn HF model card, BEIR average nDCG@10 table.",
    "bge-m3": "MTEB results repo, BGE-M3 MTEB(Multilingual, v2) retrieval average over the 18 pplx report tasks.",
    "mxbai-embed-large-v1": "Mixedbread mxbai-embed-large-v1 HF model card, MTEB Retrieval (15) nDCG@10.",
    "mxbai-embed-2d-large-v1": "Mixedbread mxbai-embed-large-v1 HF model card comparison table, MTEB Retrieval (15) nDCG@10.",
    "mxbai-embed-xsmall-v1": "Mixedbread xsmall release blog, MTEB retrieval average nDCG@10.",
    "deepset-mxbai-embed-de-large-v1": "Mixedbread/deepset German-English release blog, German retrieval benchmark average NDCG@10.",
    "mxbai-edge-colbert-v0-17m": "Mixedbread mxbai-edge ColBERT HF model card, BEIR subset average nDCG@10.",
    "mxbai-edge-colbert-v0-32m": "Mixedbread mxbai-edge ColBERT HF model card, BEIR subset average nDCG@10.",
    "modernbert-embed-base": "Nomic ModernBERT Embed Base HF model card, MTEB Retrieval (15) nDCG@10.",
    "kalm-mini-it-v15": "KaLM v1.5 HF model-index, MTEB English Retrieval (15) average nDCG@10.",
    "pplx-embed-v1-0.6b": "Perplexity pplx-embed technical report, MTEB Multilingual v2 retrieval average nDCG@10, INT8.",
    "pplx-embed-v1-4b": "Perplexity pplx-embed technical report, MTEB Multilingual v2 retrieval average nDCG@10, INT8.",
    "pplx-embed-v1-late-0.6b": "Perplexity late-interaction HF model card, BEIR (15 tasks) average nDCG@10.",
}
PUBLISHED_EMBEDDING_QUALITY: dict[str, float] = {
    "denseon":                 0.5620,
    "lateon":                  0.5722,
    "bge-m3":                  0.5288,
    "mxbai-embed-large-v1":     0.5439,
    "mxbai-embed-2d-large-v1":  0.5142,
    "mxbai-embed-xsmall-v1":    0.4280,
    "deepset-mxbai-embed-de-large-v1": 0.5170,
    "mxbai-edge-colbert-v0-17m": 0.4900,
    "mxbai-edge-colbert-v0-32m": 0.5210,
    "modernbert-embed-base":   0.5289,
    "kalm-mini-it-v15":        0.5165,
    "pplx-embed-v1-0.6b":      0.6541,
    "pplx-embed-v1-4b":        0.6966,
    "pplx-embed-v1-late-0.6b": 0.5661,
}
EMBEDDING_QUALITY_PLACEHOLDER: frozenset[str] = frozenset()

# Optional hover-detail metric. Keep separate from PUBLISHED_EMBEDDING_QUALITY
# so models without decontaminated BEIR still remain visible in the plot.
EMBEDDING_DECONTAMINATED_BEIR_SOURCES: dict[str, str] = {
    "denseon": "LightOn DenseOn/LateOn blog, Full Decontaminated BEIR Results, DenseOn row, average nDCG@10 over 12 datasets.",
    "lateon": "LightOn DenseOn/LateOn blog, Full Decontaminated BEIR Results, LateOn row, average nDCG@10 over 12 datasets.",
    "modernbert-embed-base": "LightOn DenseOn/LateOn blog, Full Decontaminated BEIR Results, MBEmb.-base row, average nDCG@10 over 12 datasets.",
    "pplx-embed-v1-0.6b": "LightOn DenseOn/LateOn blog, Full Decontaminated BEIR Results, pplx-v1-0.6b row, average nDCG@10 over 12 datasets.",
}
PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR: dict[str, float] = {
    "denseon":                 0.5771,
    "lateon":                  0.6036,
    "modernbert-embed-base":   0.5442,
    "pplx-embed-v1-0.6b":      0.5850,
}


def _modernbert_embed_model(
    key: str,
    name: str,
    color: str,
    profile: EmbeddingProfile,
) -> Model:
    return Model(
        key,
        name,
        "Embeddings",
        color,
        149e6,
        149e6,
        False,
        22,
        12,
        12,
        64,
        False,
        hidden_dim=768,
        local_attention_layers=14,
        local_attention_window=128,
        attention_label="ModernBERT local/global encoder",
        capabilities_override=frozenset(),
        embedding_profile=profile,
    )


def _pplx_embed_model(
    key: str,
    name: str,
    color: str,
    params: float,
    layers: int,
    hidden_dim: int,
    num_heads: int,
    kv_heads: int,
    output_dim: int,
    profile: EmbeddingProfile,
) -> Model:
    return Model(
        key,
        name,
        "Embeddings",
        color,
        params,
        params,
        False,
        layers,
        num_heads,
        kv_heads,
        128,
        False,
        hidden_dim=hidden_dim,
        attention_query_heads=num_heads,
        attention_label="bidirectional Qwen3 encoder",
        capabilities_override=frozenset(),
        embedding_profile=profile,
    )


def _rwkv7_g1_model(
    key: str,
    name: str,
    color: str,
    params: float,
    layers: int,
    hidden_dim: int,
    capabilities: frozenset[str],
) -> Model:
    heads = hidden_dim // RWKV7_G1_HEAD_DIM
    return Model(
        key,
        name,
        "RWKV",
        color,
        params,
        params,
        False,
        layers,
        heads,
        0,
        RWKV7_G1_HEAD_DIM,
        False,
        kv_layers=0,
        hidden_dim=hidden_dim,
        attention_layers=0,
        linear_attention_layers=layers,
        linear_attention_heads=heads,
        linear_attention_head_dim=RWKV7_G1_HEAD_DIM,
        linear_attention_k_heads=heads,
        linear_attention_k_head_dim=RWKV7_G1_HEAD_DIM,
        attention_label=f"RWKV recurrent state, ctx {RWKV7_G1_CONTEXT // 1024}k",
        capabilities_override=capabilities,
    )


LFM_TEXT_CAPABILITIES = frozenset({"tools"})


def _lfm_text_model(
    key: str,
    name: str,
    color: str,
    total_params: float,
    active_params: float,
    layers: int,
    attention_layers: int,
    hidden_dim: int,
    num_heads: int,
    kv_heads: int,
    capabilities: frozenset[str] = LFM_TEXT_CAPABILITIES,
) -> Model:
    conv_layers = max(layers - attention_layers, 0)
    head_dim = hidden_dim // max(num_heads, 1)
    return Model(
        key,
        name,
        "LFM",
        color,
        total_params,
        active_params,
        not math.isclose(total_params, active_params, rel_tol=1e-9, abs_tol=1.0),
        layers,
        num_heads,
        kv_heads,
        head_dim,
        False,
        kv_layers=attention_layers,
        hidden_dim=hidden_dim,
        attention_layers=attention_layers,
        attention_query_heads=num_heads,
        attention_label=f"{conv_layers} LIV conv + {attention_layers} GQA, ctx 32k",
        capabilities_override=capabilities,
    )


MODELS: dict[str, Model] = {
    "denseon": _modernbert_embed_model(
        "denseon",
        "DenseOn",
        "#0F766E",
        EmbeddingProfile(
            label="DenseOn",
            kind="single",
            output_dim=768,
            max_sequence_length=8192,
            source="lightonai/DenseOn",
            note="ModernBERT-base dense retrieval model; single-vector counterpart to LateOn.",
            pooling="CLS",
        ),
    ),
    "lateon": _modernbert_embed_model(
        "lateon",
        "LateOn",
        "#7C3AED",
        EmbeddingProfile(
            label="LateOn",
            kind="late",
            output_dim=128,
            late_interaction_dim=128,
            max_sequence_length=300,
            query_length=32,
            document_length=300,
            source="lightonai/LateOn",
            note="ModernBERT-base ColBERT model with MaxSim scoring; document length 300 and query length 32 in the model card.",
            pooling="token vectors",
        ),
    ),
    "bge-m3": Model(
        "bge-m3",
        "BGE-M3",
        "Embeddings",
        "#3266AD",
        569e6,
        569e6,
        False,
        24,
        16,
        16,
        64,
        False,
        hidden_dim=1024,
        attention_label="XLM-R encoder; dense/sparse/ColBERT heads",
        capabilities_override=frozenset(),
        embedding_profile=EmbeddingProfile(
            label="BGE-M3",
            kind="hybrid",
            output_dim=1024,
            late_interaction_dim=1024,
            max_sequence_length=8192,
            source="BAAI/bge-m3",
            note="Unified dense, sparse lexical, and ColBERT-style multi-vector embedding model.",
            pooling="CLS dense + token vectors",
        ),
    ),
    "mxbai-embed-large-v1": Model(
        "mxbai-embed-large-v1",
        "MXBAI Embed Large v1",
        "Embeddings",
        "#B45309",
        0.3e9,
        0.3e9,
        False,
        24,
        16,
        16,
        64,
        False,
        hidden_dim=1024,
        attention_label="BERT-large encoder",
        capabilities_override=frozenset(),
        embedding_profile=EmbeddingProfile(
            label="MXBAI Embed Large",
            kind="single",
            output_dim=1024,
            max_sequence_length=512,
            source="mixedbread-ai/mxbai-embed-large-v1",
            note="English retrieval embedder with CLS pooling, Matryoshka truncation, and binary quantization support.",
            pooling="CLS",
        ),
    ),
    "mxbai-embed-2d-large-v1": Model(
        "mxbai-embed-2d-large-v1",
        "MXBAI Embed 2D Large v1",
        "Embeddings",
        "#9A3412",
        0.3e9,
        0.3e9,
        False,
        24,
        16,
        16,
        64,
        False,
        hidden_dim=1024,
        attention_label="BERT-large encoder with adaptive layers",
        capabilities_override=frozenset(),
        embedding_profile=EmbeddingProfile(
            label="MXBAI Embed 2D Large",
            kind="single",
            output_dim=1024,
            max_sequence_length=512,
            source="mixedbread-ai/mxbai-embed-2d-large-v1",
            note="2D Matryoshka English embedder with adaptive layer count and embedding dimension.",
            pooling="CLS",
        ),
    ),
    "mxbai-embed-xsmall-v1": Model(
        "mxbai-embed-xsmall-v1",
        "MXBAI Embed XSmall v1",
        "Embeddings",
        "#4D7C0F",
        24.1e6,
        24.1e6,
        False,
        6,
        12,
        12,
        32,
        False,
        hidden_dim=384,
        attention_label="MiniLM encoder",
        capabilities_override=frozenset(),
        embedding_profile=EmbeddingProfile(
            label="MXBAI Embed XSmall",
            kind="single",
            output_dim=384,
            max_sequence_length=4096,
            source="mixedbread-ai/mxbai-embed-xsmall-v1",
            note="Compact English retrieval embedder with long-context support, MRL, and binary quantization support.",
            pooling="mean",
        ),
    ),
    "deepset-mxbai-embed-de-large-v1": Model(
        "deepset-mxbai-embed-de-large-v1",
        "Deepset MXBAI Embed DE Large v1",
        "Embeddings",
        "#047857",
        0.5e9,
        0.5e9,
        False,
        24,
        16,
        16,
        64,
        False,
        hidden_dim=1024,
        attention_label="XLM-R encoder",
        capabilities_override=frozenset(),
        embedding_profile=EmbeddingProfile(
            label="MXBAI Embed DE Large",
            kind="single",
            output_dim=1024,
            max_sequence_length=512,
            source="mixedbread-ai/deepset-mxbai-embed-de-large-v1",
            note="German-English retrieval embedder initialized from multilingual-e5-large; query/passages require explicit prefixes.",
            pooling="mean",
        ),
    ),
    "mxbai-edge-colbert-v0-17m": Model(
        "mxbai-edge-colbert-v0-17m",
        "MXBAI Edge ColBERT v0 17M",
        "Embeddings",
        "#6D28D9",
        16.8e6,
        16.8e6,
        False,
        7,
        4,
        4,
        64,
        False,
        hidden_dim=256,
        local_attention_layers=4,
        local_attention_window=128,
        attention_label="Ettin/ModernBERT ColBERT, 3 global + 4 local",
        capabilities_override=frozenset(),
        embedding_profile=EmbeddingProfile(
            label="MXBAI Edge ColBERT 17M",
            kind="late",
            output_dim=48,
            late_interaction_dim=48,
            max_sequence_length=32000,
            query_length=32,
            document_length=32000,
            source="mixedbread-ai/mxbai-edge-colbert-v0-17m",
            note="Lightweight ColBERT retriever with 48-d token vectors and documented 32k-token document support.",
            pooling="token vectors",
        ),
    ),
    "mxbai-edge-colbert-v0-32m": Model(
        "mxbai-edge-colbert-v0-32m",
        "MXBAI Edge ColBERT v0 32M",
        "Embeddings",
        "#4338CA",
        31.9e6,
        31.9e6,
        False,
        10,
        6,
        6,
        64,
        False,
        hidden_dim=384,
        local_attention_layers=6,
        local_attention_window=128,
        attention_label="Ettin/ModernBERT ColBERT, 4 global + 6 local",
        capabilities_override=frozenset(),
        embedding_profile=EmbeddingProfile(
            label="MXBAI Edge ColBERT 32M",
            kind="late",
            output_dim=64,
            late_interaction_dim=64,
            max_sequence_length=32000,
            query_length=32,
            document_length=32000,
            source="mixedbread-ai/mxbai-edge-colbert-v0-32m",
            note="Lightweight ColBERT retriever with 64-d token vectors and documented 32k-token document support.",
            pooling="token vectors",
        ),
    ),
    "modernbert-embed-base": _modernbert_embed_model(
        "modernbert-embed-base",
        "ModernBERT Embed Base",
        "#BA7517",
        EmbeddingProfile(
            label="ModernBERT Embed Base",
            kind="single",
            output_dim=768,
            max_sequence_length=8192,
            source="nomic-ai/modernbert-embed-base",
            note="Nomic single-vector embedder with Matryoshka truncation support down to 256 dimensions.",
            pooling="mean",
        ),
    ),
    "kalm-mini-it-v15": Model(
        "kalm-mini-it-v15",
        "KaLM Mini Instruct v1.5",
        "Embeddings",
        "#D85A30",
        494e6,
        494e6,
        False,
        24,
        14,
        2,
        64,
        False,
        hidden_dim=896,
        attention_query_heads=14,
        attention_label="Qwen2-0.5B embedding adapter",
        capabilities_override=frozenset(),
        embedding_profile=EmbeddingProfile(
            label="KaLM Mini IT v1.5",
            kind="single",
            output_dim=896,
            max_sequence_length=512,
            source="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
            note="Multilingual Qwen2-0.5B-based instruct embedding model; model card examples set max_seq_length to 512.",
            pooling="mean",
        ),
    ),
    "pplx-embed-v1-0.6b": _pplx_embed_model(
        "pplx-embed-v1-0.6b",
        "PPLX Embed v1 0.6B",
        "#0891B2",
        0.6e9,
        28,
        1024,
        16,
        8,
        1024,
        EmbeddingProfile(
            label="PPLX Embed v1 0.6B",
            kind="single",
            output_dim=1024,
            max_sequence_length=32768,
            source="perplexity-ai/pplx-embed-v1-0.6b",
            note="Perplexity bidirectional Qwen3 embedder; supports Matryoshka dimensions 128-1024 and native INT8/BINARY embeddings.",
            vector_bytes_per_elem=1.0,
            storage_format="INT8",
            pooling="mean",
        ),
    ),
    "pplx-embed-v1-4b": _pplx_embed_model(
        "pplx-embed-v1-4b",
        "PPLX Embed v1 4B",
        "#1D5276",
        4.0e9,
        36,
        2560,
        32,
        8,
        2560,
        EmbeddingProfile(
            label="PPLX Embed v1 4B",
            kind="single",
            output_dim=2560,
            max_sequence_length=32768,
            source="perplexity-ai/pplx-embed-v1-4b",
            note="Perplexity bidirectional Qwen3 embedder; supports Matryoshka dimensions 128-2560 and native INT8/BINARY embeddings.",
            vector_bytes_per_elem=1.0,
            storage_format="INT8",
            pooling="mean",
        ),
    ),
    "pplx-embed-v1-late-0.6b": _pplx_embed_model(
        "pplx-embed-v1-late-0.6b",
        "PPLX Embed v1 Late 0.6B",
        "#7F77DD",
        0.6e9,
        28,
        1024,
        16,
        8,
        128,
        EmbeddingProfile(
            label="PPLX Embed v1 Late 0.6B",
            kind="late",
            output_dim=128,
            late_interaction_dim=128,
            max_sequence_length=32768,
            source="perplexity-ai/pplx-embed-v1-late-0.6b",
            note="Token-level late-interaction model continued from PPLX Embed v1 0.6B and scored with MaxSim.",
            pooling="token vectors",
        ),
    ),

    "l8": Model("l8", "Llama 3.1 8B", "Meta", "#22976B", 8e9, 8e9, False, 32, 32, 8, 128, False),
    "l70": Model("l70", "Llama 3.1 70B", "Meta", "#2B7A78", 70.6e9, 70.6e9, False, 80, 64, 8, 128, False),

    "ge2": Model("ge2", "Gemma 4 E2B", "Gemma", "#5D8C3C", 2e9, 2e9, False, 26, 16, 8, 128, False),
    "ge4": Model("ge4", "Gemma 4 E4B", "Gemma", "#6FA84A", 4e9, 4e9, False, 34, 24, 8, 128, False),
    "g12": Model(
        "g12",
        "Gemma 4 12B Unified",
        "Gemma",
        "#7DAF52",
        11.95e9,
        11.95e9,
        False,
        48,
        16,
        8,
        256,
        False,
        hidden_dim=3840,
        attention_layers=48,
        local_attention_layers=40,
        local_attention_window=1024,
        global_kv_heads=1,
        global_head_dim=512,
        shared_key_value=True,
        attention_label="40 sliding 1k + 8 global p-RoPE; encoder-free image/audio projection",
    ),
    "g26": Model("g26", "Gemma 4 26B-A4B", "Gemma", "#8AB85C", 26e9, 4e9, True, 48, 32, 8, 128, False),
    "g31": Model("g31", "Gemma 4 31B", "Gemma", "#A2C96E", 31e9, 31e9, False, 48, 40, 8, 128, False),

    "lfm2.5-350m": _lfm_text_model(
        "lfm2.5-350m",
        "LFM2.5 350M",
        "#14B8A6",
        354_483_968,
        354_483_968,
        16,
        6,
        1024,
        16,
        8,
    ),
    "lfm2.5-1.2b-instruct": _lfm_text_model(
        "lfm2.5-1.2b-instruct",
        "LFM2.5 1.2B Instruct",
        "#0891B2",
        1_170_340_608,
        1_170_340_608,
        16,
        6,
        2048,
        32,
        8,
    ),
    "lfm2.5-1.2b-thinking": _lfm_text_model(
        "lfm2.5-1.2b-thinking",
        "LFM2.5 1.2B Thinking",
        "#2563EB",
        1_170_340_608,
        1_170_340_608,
        16,
        6,
        2048,
        32,
        8,
    ),
    "lfm2-700m": _lfm_text_model(
        "lfm2-700m",
        "LFM2 700M",
        "#0F766E",
        742_489_344,
        742_489_344,
        16,
        6,
        1536,
        24,
        8,
    ),
    "lfm2-2.6b": _lfm_text_model(
        "lfm2-2.6b",
        "LFM2 2.6B",
        "#0E7490",
        2_569_272_320,
        2_569_272_320,
        30,
        8,
        2048,
        32,
        8,
    ),
    "lfm2-8b-a1b": _lfm_text_model(
        "lfm2-8b-a1b",
        "LFM2 8B-A1.5B",
        "#1D4ED8",
        8.3e9,
        1.5e9,
        24,
        6,
        2048,
        32,
        8,
    ),
    "lfm2-24b-a2b": _lfm_text_model(
        "lfm2-24b-a2b",
        "LFM2 24B-A2.3B",
        "#1E3A8A",
        24e9,
        2.3e9,
        40,
        10,
        2048,
        32,
        8,
    ),

    "rwkv7-g1d-01b": _rwkv7_g1_model(
        "rwkv7-g1d-01b",
        "RWKV7-G1D 0.1B",
        "#0F766E",
        0.1e9,
        12,
        768,
        RWKV7_G1_BASE_CAPABILITIES,
    ),
    "rwkv7-g1d-04b": _rwkv7_g1_model(
        "rwkv7-g1d-04b",
        "RWKV7-G1D 0.4B",
        "#13806F",
        0.4e9,
        24,
        1024,
        RWKV7_G1_BASE_CAPABILITIES,
    ),
    "rwkv7-g1f-15b": _rwkv7_g1_model(
        "rwkv7-g1f-15b",
        "RWKV7-G1F 1.5B",
        "#168A70",
        1.5e9,
        24,
        2048,
        RWKV7_G1_TOOL_CAPABILITIES,
    ),
    "rwkv7-g1f-29b": _rwkv7_g1_model(
        "rwkv7-g1f-29b",
        "RWKV7-G1F 2.9B",
        "#1D9470",
        2.9e9,
        32,
        2560,
        RWKV7_G1_TOOL_CAPABILITIES,
    ),
    "rwkv7-g1g-72b": _rwkv7_g1_model(
        "rwkv7-g1g-72b",
        "RWKV7-G1G 7.2B",
        "#259E6F",
        7.2e9,
        32,
        4096,
        RWKV7_G1_TOOL_CAPABILITIES,
    ),
    "rwkv7-g1g-133b": _rwkv7_g1_model(
        "rwkv7-g1g-133b",
        "RWKV7-G1G 13.3B",
        "#2EA86E",
        13.3e9,
        61,
        4096,
        RWKV7_G1_TOOL_CAPABILITIES,
    ),

    "q08": Model("q08", "Qwen 3.5 0.8B", "Qwen", "#0E8F66", 0.8e9, 0.8e9, False, 24, 16, 4, 64, False),
    "q2": Model("q2", "Qwen 3.5 2B", "Qwen", "#15986D", 2e9, 2e9, False, 28, 16, 4, 128, False),
    "q4": Model("q4", "Qwen 3.5 4B", "Qwen", "#1AA174", 4e9, 4e9, False, 32, 24, 4, 128, False),
    "q9": Model("q9", "Qwen 3.5 9B", "Qwen", "#1D9E75", 9.2e9, 9.2e9, False, 36, 36, 4, 128, False),
    "q27": Model("q27", "Qwen 3.5 27B", "Qwen", "#3266ad", 27.8e9, 27.8e9, False, 48, 36, 4, 128, False),
    "q35": Model("q35", "Qwen 3.5 35B-A3B", "Qwen", "#7F77DD", 35e9, 3e9, True, 64, 16, 4, 128, False),
    "q122": Model("q122", "Qwen 3.5 122B-A10B", "Qwen", "#D85A30", 122e9, 10e9, True, 96, 32, 8, 128, False),
    "q397": Model("q397", "Qwen 3.5 397B-A17B", "Qwen", "#A6422A", 397e9, 17e9, True, 96, 64, 8, 128, False),

    "glm45a": Model("glm45a", "GLM-4.5-Air 106B-A12B", "GLM", "#2F7E9F", 106e9, 12e9, True, 56, 64, 8, 128, False),
    "glm45": Model("glm45", "GLM-4.5 355B-A32B", "GLM", "#2B6D8A", 355e9, 32e9, True, 62, 96, 8, 128, False),
    "glm46": Model("glm46", "GLM-4.6 357B-A32B", "GLM", "#275C75", 357e9, 32e9, True, 62, 96, 8, 128, False),
    "glm47": Model("glm47", "GLM-4.7 358B-A32B", "GLM", "#214A61", 358e9, 32e9, True, 62, 96, 8, 128, False),
    "glm47f": Model("glm47f", "GLM-4.7-Flash 31B-A3B", "GLM", "#3F93BA", 31e9, 3e9, True, 48, 32, 8, 128, False),
    "glm5": Model("glm5", "GLM-5 744B-A40B", "GLM", "#16354A", 744e9, 40e9, True, 72, 128, 8, 128, False),
    "glm51": Model("glm51", "GLM-5.1 744B-A40B", "GLM", "#0F273A", 744e9, 40e9, True, 72, 128, 8, 128, False),

    "k25": Model(
        "k25",
        "Kimi K2.5 1T-A32B",
        "Kimi",
        "#5B4FE9",
        1e12,
        32e9,
        True,
        61,
        64,
        1,
        112,
        True,
        512,
        64,
        bf16_weight_bytes_per_param=MIXED_NATIVE_BF16_WEIGHT_BPP,
        fp8_weight_bytes_per_param=MIXED_NATIVE_FP8_WEIGHT_BPP,
    ),
    "kimi-linear-48b": Model(
        "kimi-linear-48b",
        "Kimi Linear 48B-A3B",
        "Kimi",
        "#4F46E5",
        48e9,
        3e9,
        True,
        27,
        32,
        32,
        72,
        True,
        512,
        64,
        mla_tp_supported=True,
        kv_layers=7,
        hidden_dim=2304,
        attention_layers=7,
        linear_attention_layers=20,
        linear_attention_heads=32,
        linear_attention_head_dim=128,
        linear_attention_k_heads=32,
        linear_attention_k_head_dim=128,
        linear_attention_conv_kernel=4,
        attention_label="20 KDA + 7 MLA",
    ),

    "command-a-plus-05-2026": Model(
        "command-a-plus-05-2026",
        "Command A+ 05-2026",
        "Cohere",
        "#0F766E",
        218e9,
        25e9,
        True,
        32,
        128,
        8,
        128,
        False,
        hidden_dim=4096,
        local_attention_layers=24,
        local_attention_window=4096,
        local_attention_heads=128,
        attention_label="24 SWA 4k + 8 global",
    ),

    "minimax25": Model("minimax25", "MiniMax M2.5 229B-A10B", "MiniMax", "#2C6D9B", 229e9, 10e9, True, 62, 48, 8, 128, False),
    "minimax27": Model("minimax27", "MiniMax M2.7 229B-A10B", "MiniMax", "#1D5276", 229e9, 10e9, True, 62, 48, 8, 128, False),

    "nem3s": Model("nem3s", "Nemotron 3 Super 120B-A12B", "Nemotron", "#6FA7C9", 120e9, 12e9, True, 88, 32, 2, 128, False, kv_layers=8),
    "nem3n": Model("nem3n", "Nemotron 3 Nano 30B-A3B", "Nemotron", "#98C5DE", 31.6e9, 3.2e9, True, 52, 32, 2, 128, False, kv_layers=6),
    "nem3no": Model("nem3no", "Nemotron 3 Nano Omni 30B-A3B", "Nemotron", "#B7D5E8", 30e9, 3e9, True, 52, 32, 2, 128, False, kv_layers=6),

    "ds3": Model(
        "ds3",
        "DeepSeek V3 671B-A37B",
        "DeepSeek",
        "#A32D2D",
        671e9,
        37e9,
        True,
        61,
        128,
        1,
        288,
        True,
        512,
        64,
        bf16_weight_bytes_per_param=MIXED_NATIVE_BF16_WEIGHT_BPP,
        fp8_weight_bytes_per_param=MIXED_NATIVE_FP8_WEIGHT_BPP,
    ),
    "deepseek-v4-pro": Model(
        "deepseek-v4-pro",
        "DeepSeek V4 Pro 1.6T-A49B",
        "DeepSeek",
        "#7F1D1D",
        1.6e12,
        49e9,
        True,
        72,
        128,
        1,
        288,
        True,
        512,
        64,
        bf16_weight_bytes_per_param=MIXED_NATIVE_BF16_WEIGHT_BPP,
        fp8_weight_bytes_per_param=FP4_FP8_MOE_WEIGHT_BPP,
    ),
    "deepseek-v4-flash": Model(
        "deepseek-v4-flash",
        "DeepSeek V4 Flash 284B-A13B",
        "DeepSeek",
        "#C24132",
        284e9,
        13e9,
        True,
        48,
        96,
        1,
        256,
        True,
        512,
        64,
        bf16_weight_bytes_per_param=MIXED_NATIVE_BF16_WEIGHT_BPP,
        fp8_weight_bytes_per_param=FP4_FP8_MOE_WEIGHT_BPP,
    ),

    "mi7": Model("mi7", "Mistral 7B", "Mistral", "#e07020", 7e9, 7e9, False, 32, 32, 8, 128, False),
    "mx87": Model("mx87", "Mixtral 8×7B (45B-A12B)", "Mistral", "#cc6633", 45e9, 12e9, True, 32, 32, 8, 128, False),
    "cs22": Model("cs22", "Codestral 22B", "Mistral", "#d4882e", 22e9, 22e9, False, 56, 32, 8, 128, False),
    "ms24": Model("ms24", "Mistral Small 3.1 24B", "Mistral", "#b87530", 24e9, 24e9, False, 40, 32, 8, 128, False),
    "ms32": Model("ms32", "Mistral Small 3.2 24B", "Mistral", "#C18438", 24e9, 24e9, False, 40, 32, 8, 128, False),
    "voxtral-realtime-mini-4b": Model(
        "voxtral-realtime-mini-4b",
        "Voxtral Mini Realtime 4B",
        "Audio",
        "#DE7A24",
        4.37e9,
        4.37e9,
        False,
        26,
        32,
        8,
        128,
        False,
        hidden_dim=3072,
        local_attention_layers=26,
        local_attention_window=8192,
        attention_label="Realtime ASR · 8k SWA",
        capabilities_override=frozenset(),
        realtime_profile=VOXTRAL_REALTIME_PROFILE,
    ),
    "mimo-v2.5-asr": Model(
        "mimo-v2.5-asr",
        "MiMo-V2.5-ASR 8B",
        "Audio",
        "#B83280",
        8.0e9,
        8.0e9,
        False,
        36,
        32,
        8,
        128,
        False,
        hidden_dim=4096,
        attention_query_heads=32,
        attention_label="ASR full attention 8k",
        capabilities_override=frozenset(),
        realtime_profile=MIMO_V25_ASR_PROFILE,
    ),
    "nvidia-nemotron-speech-streaming-0.6b": Model(
        "nvidia-nemotron-speech-streaming-0.6b",
        "NVIDIA Nemotron Speech Streaming 0.6B",
        "Audio",
        "#2563EB",
        0.6e9,
        0.6e9,
        False,
        24,
        8,
        8,
        128,
        False,
        hidden_dim=1024,
        local_attention_layers=24,
        local_attention_window=70,
        attention_label="Cache-aware FastConformer 70f",
        capabilities_override=frozenset(),
        realtime_profile=NEMOTRON_SPEECH_STREAMING_PROFILE,
    ),
    "nvidia-nemotron-3.5-asr-streaming-0.6b": Model(
        "nvidia-nemotron-3.5-asr-streaming-0.6b",
        "NVIDIA Nemotron 3.5 ASR Streaming 0.6B",
        "Audio",
        "#76B900",
        0.6e9,
        0.6e9,
        False,
        24,
        8,
        8,
        128,
        False,
        hidden_dim=1024,
        local_attention_layers=24,
        local_attention_window=56,
        attention_label="Prompted cache-aware FastConformer 56f",
        capabilities_override=frozenset(),
        realtime_profile=NEMOTRON_35_ASR_STREAMING_PROFILE,
    ),
    "nvidia-parakeet-unified-0.6b": Model(
        "nvidia-parakeet-unified-0.6b",
        "NVIDIA Parakeet Unified 0.6B",
        "Audio",
        "#0F766E",
        0.6e9,
        0.6e9,
        False,
        24,
        8,
        8,
        128,
        False,
        hidden_dim=1024,
        local_attention_layers=24,
        local_attention_window=70,
        attention_label="Unified FastConformer 70f",
        capabilities_override=frozenset(),
        realtime_profile=PARAKEET_UNIFIED_STREAMING_PROFILE,
    ),
    "nvidia-parakeet-realtime-eou-120m": Model(
        "nvidia-parakeet-realtime-eou-120m",
        "NVIDIA Parakeet Realtime EOU 120M",
        "Audio",
        "#CA8A04",
        120e6,
        120e6,
        False,
        17,
        8,
        8,
        64,
        False,
        hidden_dim=512,
        local_attention_layers=17,
        local_attention_window=70,
        attention_label="Cache-aware FastConformer 70f",
        capabilities_override=frozenset(),
        realtime_profile=PARAKEET_REALTIME_EOU_PROFILE,
    ),
    "nvidia-multitalker-parakeet-streaming-0.6b": Model(
        "nvidia-multitalker-parakeet-streaming-0.6b",
        "NVIDIA Multitalker Parakeet Streaming 0.6B",
        "Audio",
        "#7C3AED",
        0.6e9,
        0.6e9,
        False,
        24,
        8,
        8,
        128,
        False,
        hidden_dim=1024,
        local_attention_layers=24,
        local_attention_window=70,
        attention_label="Multi-instance FastConformer 70f",
        capabilities_override=frozenset(),
        realtime_profile=MULTITALKER_PARAKEET_STREAMING_PROFILE,
    ),
    "kyutai-stt-1b-en-fr": Model(
        "kyutai-stt-1b-en-fr",
        "Kyutai STT 1B EN/FR",
        "Audio",
        "#0891B2",
        1.0e9,
        1.0e9,
        False,
        16,
        16,
        16,
        128,
        False,
        hidden_dim=2048,
        local_attention_layers=16,
        local_attention_window=750,
        attention_label="Mimi delayed streams 750",
        capabilities_override=frozenset(),
        realtime_profile=KYUTAI_STT_1B_PROFILE,
    ),
    "kyutai-stt-2.6b-en": Model(
        "kyutai-stt-2.6b-en",
        "Kyutai STT 2.6B EN",
        "Audio",
        "#0284C7",
        2.6e9,
        2.6e9,
        False,
        48,
        32,
        32,
        64,
        False,
        hidden_dim=2048,
        local_attention_layers=48,
        local_attention_window=375,
        attention_label="Mimi delayed streams 375",
        capabilities_override=frozenset(),
        realtime_profile=KYUTAI_STT_2_6B_PROFILE,
    ),
    "moonshine-streaming-tiny": Model(
        "moonshine-streaming-tiny",
        "Moonshine Streaming Tiny 34M",
        "Audio",
        "#16A34A",
        34e6,
        34e6,
        False,
        12,
        5,
        5,
        64,
        False,
        hidden_dim=320,
        local_attention_layers=6,
        local_attention_window=16,
        attention_label="Streaming seq2seq 16f SWA",
        capabilities_override=frozenset(),
        realtime_profile=MOONSHINE_STREAMING_TINY_PROFILE,
    ),
    "moonshine-streaming-small": Model(
        "moonshine-streaming-small",
        "Moonshine Streaming Small 123M",
        "Audio",
        "#65A30D",
        123e6,
        123e6,
        False,
        20,
        10,
        10,
        62,
        False,
        hidden_dim=620,
        local_attention_layers=10,
        local_attention_window=16,
        attention_label="Streaming seq2seq 16f SWA",
        capabilities_override=frozenset(),
        realtime_profile=MOONSHINE_STREAMING_SMALL_PROFILE,
    ),
    "moonshine-streaming-medium": Model(
        "moonshine-streaming-medium",
        "Moonshine Streaming Medium 245M",
        "Audio",
        "#15803D",
        245e6,
        245e6,
        False,
        28,
        12,
        12,
        64,
        False,
        hidden_dim=768,
        local_attention_layers=14,
        local_attention_window=16,
        attention_label="Streaming seq2seq 16f SWA",
        capabilities_override=frozenset(),
        realtime_profile=MOONSHINE_STREAMING_MEDIUM_PROFILE,
    ),
    "fun-asr-nano-2512": Model(
        "fun-asr-nano-2512",
        "Fun-ASR-Nano 2512 800M",
        "Audio",
        "#DC2626",
        800e6,
        800e6,
        False,
        28,
        16,
        8,
        128,
        False,
        hidden_dim=1024,
        attention_label="Realtime ASR Qwen3-0.6B core",
        capabilities_override=frozenset(),
        realtime_profile=FUN_ASR_NANO_PROFILE,
    ),
    "granite-4.0-1b-speech": Model(
        "granite-4.0-1b-speech",
        "Granite 4.0 1B Speech",
        "Audio",
        "#475569",
        1.0e9,
        1.0e9,
        False,
        40,
        16,
        4,
        128,
        False,
        hidden_dim=2048,
        attention_label="Offline speech LLM 128k ctx",
        capabilities_override=frozenset(),
        realtime_profile=GRANITE_4_1B_SPEECH_PROFILE,
    ),
    "nvidia-parakeet-tdt-0.6b-v3": Model(
        "nvidia-parakeet-tdt-0.6b-v3",
        "NVIDIA Parakeet TDT 0.6B v3",
        "Audio",
        "#059669",
        0.6e9,
        0.6e9,
        False,
        24,
        8,
        8,
        128,
        False,
        hidden_dim=1024,
        attention_label="Offline FastConformer-TDT",
        capabilities_override=frozenset(),
        realtime_profile=PARAKEET_TDT_06B_V3_PROFILE,
    ),
    # Mistral does not publish a parameter count for Medium 3.1; keep a hidden
    # legacy entry so older saved states continue to resolve cleanly.
    "mm31": Model("mm31", "Mistral Medium 3.1 (legacy)", "Mistral", "#AD6A2C", 24e9, 24e9, False, 40, 32, 8, 128, False, hidden=True),
    "mistral-medium-3.5-preview": Model("mistral-medium-3.5-preview", "Mistral Medium 3.5 128B", "Mistral", "#A95F24", 128e9, 128e9, False, 88, 96, 8, 128, False),
    "ms4": Model("ms4", "Mistral Small 4 119B-A6.5B", "Mistral", "#93511F", 119e9, 6.5e9, True, 64, 64, 8, 128, False),
    "ml3": Model("ml3", "Mistral Large 3 675B-A41B", "Mistral", "#7A3B18", 675e9, 41e9, True, 96, 128, 8, 128, False),
    "ml123": Model("ml123", "Mistral Large 2 123B", "Mistral", "#994422", 123e9, 123e9, False, 88, 96, 8, 128, False),

    "n3": Model("n3", "Ministral 3 3B", "Ministral", "#E2A552", 3e9, 3e9, False, 28, 24, 8, 128, False),
    "n8": Model("n8", "Ministral 3 8B", "Ministral", "#D69343", 8e9, 8e9, False, 32, 32, 8, 128, False),
    "n14": Model("n14", "Ministral 3 14B", "Ministral", "#CA8136", 14e9, 14e9, False, 40, 32, 8, 128, False),

    "dv24": Model("dv24", "Devstral Small 2 24B", "Devstral", "#B85F59", 24e9, 24e9, False, 40, 32, 8, 128, False),
    "dv123": Model("dv123", "Devstral 2 123B", "Devstral", "#94423E", 123e9, 123e9, False, 88, 96, 8, 128, False),

    # ZAYA1-base/reasoning-base config: 16 physical heads, CCA attention in 8-query-head
    # latent space, 2 KV heads, and 40 attention-bearing layers.
    "zaya1-8b": Model(
        "zaya1-8b",
        "ZAYA1-8B 8.3B-A0.76B",
        "Zyphra",
        "#5B7CFA",
        8.3e9,
        0.76e9,
        True,
        40,
        16,
        2,
        128,
        False,
        kv_layers=40,
        hidden_dim=2048,
        attention_layers=40,
        attention_query_heads=8,
        attention_label="CCA/CCGQA 2x Q, 8x KV",
    ),
    # Retained only so older saved states can resolve. Zyphra's public ZAYA catalog currently
    # exposes the 8.3B/760M model; no public 74B architecture/config is available to model.
    "zaya1-74b-preview": Model(
        "zaya1-74b-preview",
        "ZAYA1-74B Preview (legacy proxy)",
        "Zyphra",
        "#7553C8",
        74e9,
        4e9,
        True,
        120,
        16,
        2,
        128,
        False,
        kv_layers=60,
        hidden_dim=4096,
        attention_layers=60,
        local_attention_layers=30,
        local_attention_window=4096,
        hidden=True,
        attention_label="Legacy proxy",
    ),

    # Poolside has not published the Laguna M.1 config; keep the public 225B-A23B
    # facts and use a conservative Laguna-family attention proxy for capacity math.
    "laguna-m1": Model(
        "laguna-m1",
        "Laguna M.1 225B-A23B",
        "Poolside",
        "#0E7490",
        225e9,
        23e9,
        True,
        64,
        64,
        8,
        128,
        False,
        hidden_dim=4096,
        local_attention_layers=48,
        local_attention_window=512,
        local_attention_heads=64,
        attention_label="16 global + 48 SWA 512 (proxy)",
    ),
    "laguna-xs2": Model("laguna-xs2", "Laguna XS.2 33B-A3B", "Poolside", "#0891B2", 33e9, 3e9, True, 40, 48, 8, 128, False, hidden_dim=2048, local_attention_layers=30, local_attention_window=512, local_attention_heads=64, attention_label="10 global + 30 SWA 512"),

    "mimo-v2.5-pro": Model(
        "mimo-v2.5-pro",
        "MiMo-V2.5-Pro 1.02T-A42B",
        "MiMo",
        "#C026D3",
        1.02e12,
        42e9,
        True,
        70,
        48,
        8,
        128,
        False,
        bf16_weight_bytes_per_param=MIXED_NATIVE_BF16_WEIGHT_BPP,
        fp8_weight_bytes_per_param=MIXED_NATIVE_FP8_WEIGHT_BPP,
    ),
    "mimo-v2.5": Model(
        "mimo-v2.5",
        "MiMo-V2.5 310B-A15B",
        "MiMo",
        "#E11D48",
        310e9,
        15e9,
        True,
        48,
        32,
        8,
        128,
        False,
        bf16_weight_bytes_per_param=MIXED_NATIVE_BF16_WEIGHT_BPP,
        fp8_weight_bytes_per_param=MIXED_NATIVE_FP8_WEIGHT_BPP,
    ),

    "cr13": Model("cr13", "Croissant 1.3B", "Croissant", "#dda050", 1.3e9, 1.3e9, False, 22, 16, 4, 96, False),
}


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
) -> tuple[tuple[str, str], QuantizationProfile]:
    return (
        (model_key, "nvfp4"),
        QuantizationProfile(
            precision_key="nvfp4",
            label="NVFP4",
            source_repo=source_repo,
            source_revision=source_revision,
            source_downloads=source_downloads,
            captured_at=QUANTIZATION_CAPTURED_AT,
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


MODEL_QUANTIZATION_PROFILES: dict[tuple[str, str], QuantizationProfile] = dict([
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
        retained=("language self-attention BF16", "embeddings BF16", "vision tower BF16", "lm_head BF16"),
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
        retained=("early language layers BF16", "routers BF16", "vision tower BF16", "lm_head BF16"),
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
        quantized=("MoE expert weights: packed FP4 payload + FP8 scales", "selected self-attention layers"),
        retained=("linear attention BF16", "router gates BF16", "embeddings BF16", "vision modules BF16", "lm_head BF16"),
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
        retained=("linear attention BF16", "router gates BF16", "visual modules BF16", "lm_head BF16"),
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
        retained=("linear attention BF16", "router gates BF16", "visual modules BF16", "lm_head BF16"),
        notes="Family proxy until the larger Qwen3.5-397B safetensors headers are captured locally.",
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
        retained=("omni/multimodal towers BF16", "routing-sensitive and attention tensors BF16"),
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
])


def get_quantization_profile(model_key: str, prec: str) -> QuantizationProfile | None:
    return MODEL_QUANTIZATION_PROFILES.get((model_key, normalize_precision(prec)))


# Capability overrides. Vision-enabled and reasoning-first models deviate from the default
# (tools + ctx_128k). Kept conservative — annotate models with well-documented support.
_VISION_MODELS = (
    "ge2", "ge4", "g12", "g26", "g31",
    "command-a-plus-05-2026",
    "ms24", "ms32", "mistral-medium-3.5-preview",
    "minimax25", "minimax27", "nem3no", "mimo-v2.5",
)
_AUDIO_INPUT_MODELS = (
    "ge2", "ge4", "g12",
)
_REASONING_MODELS = (
    "g12", "q35", "q122", "q397",
    "glm45", "glm45a", "glm46", "glm47", "glm47f", "glm5", "glm51",
    "k25", "ds3", "deepseek-v4-pro", "deepseek-v4-flash",
    "lfm2.5-1.2b-thinking",
    "command-a-plus-05-2026",
    "mistral-medium-3.5-preview", "ml3",
    "minimax25", "minimax27",
    "nem3s", "nem3n", "nem3no",
    "zaya1-8b", "zaya1-74b-preview", "laguna-m1", "laguna-xs2",
    "mimo-v2.5-pro", "mimo-v2.5",
)
for _k in _VISION_MODELS:
    if _k in MODELS:
        MODELS[_k].extra_capabilities = MODELS[_k].extra_capabilities | {"images"}
for _k in _AUDIO_INPUT_MODELS:
    if _k in MODELS:
        MODELS[_k].extra_capabilities = MODELS[_k].extra_capabilities | {"audio"}
for _k in _REASONING_MODELS:
    if _k in MODELS:
        MODELS[_k].extra_capabilities = MODELS[_k].extra_capabilities | {"reasoning"}

# Artificial Analysis Intelligence Index score and Intelligence Index output-token usage
# (verbosity) in millions. For models with separate reasoning/non-reasoning AA pages, prefer
# the reasoning page when available. Where AA had no directly usable page for the exact model,
# we use the closest available family proxy and note it inline.
AA_MODEL_METRICS: dict[str, tuple[float, float]] = {
    "l8": (12.0, 5.2),
    "l70": (12.0, 4.7),
    "ge2": (12.0, 8.3),
    "ge4": (15.0, 7.9),
    "g12": (25.0, 12.0),  # Proxy from Google Gemma 4 12B benchmarks; no AA page found at launch.
    "g26": (27.0, 14.0),
    "g31": (32.0, 7.1),
    "lfm2.5-350m": (7.0, 12.0),          # Conservative proxy; no AA page found for LFM2.5-350M.
    "lfm2.5-1.2b-instruct": (8.0, 4.6),
    "lfm2.5-1.2b-thinking": (8.0, 31.0),
    "lfm2-700m": (7.0, 10.0),           # Conservative size proxy; no AA page found for LFM2-700M.
    "lfm2-2.6b": (8.0, 7.8),
    "lfm2-8b-a1b": (7.0, 7.8),
    "lfm2-24b-a2b": (10.0, 11.0),
    "rwkv7-g1d-01b": (7.0, 60.0),     # Low-confidence size proxy until AA publishes RWKV7-G1 rows.
    "rwkv7-g1d-04b": (8.0, 70.0),
    "rwkv7-g1f-15b": (11.0, 90.0),
    "rwkv7-g1f-29b": (14.0, 105.0),
    "rwkv7-g1g-72b": (19.0, 120.0),
    "rwkv7-g1g-133b": (22.0, 130.0),
    "q08": (11.0, 230.0),
    "q2": (16.0, 390.0),
    "q4": (27.0, 240.0),
    "q9": (32.0, 200.0),
    "q27": (42.0, 98.0),
    "q35": (37.0, 100.0),
    "q122": (42.0, 91.0),
    "q397": (45.0, 86.0),
    "glm45a": (23.0, 68.0),
    "glm45": (26.0, 61.0),
    "glm46": (33.0, 57.0),
    "glm47": (42.0, 170.0),
    "glm47f": (30.0, 64.0),
    "glm5": (50.0, 110.0),
    "glm51": (51.0, 110.0),
    "kimi-linear-48b": (37.0, 100.0),  # Proxy from Qwen 3.5 35B-A3B until AA publishes Kimi Linear.
    "command-a-plus-05-2026": (37.0, 66.0),
    "minimax25": (42.0, 56.0),
    "minimax27": (50.0, 87.0),
    "nem3s": (36.0, 110.0),
    "nem3n": (24.0, 140.0),
    "nem3no": (26.0, 130.0),  # Omni preview proxy from Nano reasoning until AA publishes a dedicated page.
    "deepseek-v4-pro": (52.0, 190.0),
    "deepseek-v4-flash": (47.0, 240.0),
    "mi7": (7.0, 2.5),
    "mx87": (8.0, 2.5),    # Proxy verbosity from Mistral 7B; AA exposes score but not token usage.
    "cs22": (15.0, 4.4),   # Proxy from Devstral Small (Jul '25'); no AA page for Codestral 22B found.
    "ms24": (14.0, 4.7),
    "ms32": (15.0, 4.5),
    "mm31": (21.0, 7.6),
    "mistral-medium-3.5-preview": (39.0, 90.0),
    "ms4": (19.0, 3.9),
    "ml3": (23.0, 5.2),
    "ml123": (15.0, 2.6),
    "n3": (11.0, 16.0),
    "n8": (15.0, 13.0),
    "n14": (16.0, 11.0),
    "dv24": (19.0, 8.6),
    "dv123": (22.0, 7.4),
    "zaya1-8b": (24.0, 140.0),       # Proxy from Nemotron 3 Nano reasoning until AA publishes ZAYA1-8B.
    "zaya1-74b-preview": (37.0, 100.0),  # Preview is pre-RL; proxy from Qwen 3.5 35B-A3B until AA publishes it.
    "laguna-m1": (44.0, 95.0),       # Proxy from Qwen 3.5 397B-A17B adjusted against Poolside coding-agent benchmarks; no AA row found.
    "laguna-xs2": (37.0, 100.0),     # Proxy from Qwen 3.5 35B-A3B; no AA page for Laguna XS.2 found.
    "mimo-v2.5-pro": (54.0, 92.0),
    "mimo-v2.5": (49.0, 74.0),
    "cr13": (12.0, 8.3),   # Proxy from Gemma 4 E2B (Non-reasoning); no AA page for Croissant 1.3B found.
}

for _k, (_score, _verbosity_m) in AA_MODEL_METRICS.items():
    if _k in MODELS:
        MODELS[_k].quality = aa_intelligence_to_quality(_score)
        MODELS[_k].token_efficiency = aa_output_tokens_to_efficiency(_verbosity_m)

# Confidence is separate from score: direct benchmark rows stay at 1.0, family/proxy rows
# are discounted by effective_quality(). This is intentionally conservative for models
# whose public benchmark coverage is missing or weak.
AA_MODEL_QUALITY_CONFIDENCE: dict[str, float] = {
    "lfm2.5-350m": 0.45,
    "lfm2-700m": 0.45,
    "g12": 0.65,
    "rwkv7-g1d-01b": 0.35,
    "rwkv7-g1d-04b": 0.35,
    "rwkv7-g1f-15b": 0.35,
    "rwkv7-g1f-29b": 0.35,
    "rwkv7-g1g-72b": 0.35,
    "rwkv7-g1g-133b": 0.35,
    "kimi-linear-48b": 0.55,
    "nem3no": 0.65,
    "mx87": 0.70,
    "cs22": 0.60,
    "mistral-medium-3.5-preview": 0.70,
    "zaya1-8b": 0.45,
    "zaya1-74b-preview": 0.35,
    "laguna-m1": 0.55,
    "laguna-xs2": 0.45,
    "cr13": 0.25,
}
for _k, _confidence in AA_MODEL_QUALITY_CONFIDENCE.items():
    if _k in MODELS:
        MODELS[_k].quality_confidence = _confidence


@dataclass
class Bucket:
    length: int
    label: str
    color: str


INPUT_BUCKETS = [
    Bucket(256, "256", "#10825c"),
    Bucket(1024, "1k", "#1D9E75"),
    Bucket(4096, "4k", "#3266ad"),
    Bucket(16384, "16k", "#7F77DD"),
    Bucket(32768, "32k", "#BA7517"),
    Bucket(65536, "64k", "#D85A30"),
    Bucket(131072, "128k", "#A32D2D"),
]

OUTPUT_BUCKETS = [
    Bucket(32, "32", "#10825c"),
    Bucket(128, "128", "#1D9E75"),
    Bucket(512, "512", "#3266ad"),
    Bucket(2048, "2k", "#7F77DD"),
    Bucket(4096, "4k", "#BA7517"),
    Bucket(8192, "8k", "#D85A30"),
]

EMBEDDING_DOC_BUCKETS = [
    Bucket(32, "32", "#10825c"),
    Bucket(128, "128", "#1D9E75"),
    Bucket(256, "256", "#3266ad"),
    Bucket(1024, "1k", "#7F77DD"),
    Bucket(2048, "2k", "#BA7517"),
    Bucket(8192, "8k", "#D85A30"),
    Bucket(32768, "32k", "#A32D2D"),
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

DIST_PRESETS = {
    "Chat": {"in": [10, 30, 35, 15, 7, 2, 1], "out": [15, 30, 35, 15, 4, 1]},
    "RAG": {"in": [5, 15, 25, 25, 18, 8, 4], "out": [10, 25, 40, 18, 5, 2]},
    "Long doc": {"in": [2, 5, 10, 15, 25, 25, 18], "out": [5, 15, 30, 30, 15, 5]},
    "Code": {"in": [8, 25, 30, 22, 10, 4, 1], "out": [10, 20, 35, 25, 8, 2]},
    "Classify": {"in": [5, 20, 40, 25, 8, 2, 0], "out": [80, 15, 4, 1, 0, 0]},
}

EMBEDDING_DOC_PRESETS = {
    "Query": [90, 10, 0, 0, 0, 0, 0],
    "Passage": [0, 10, 75, 15, 0, 0, 0],
    "Doc": [0, 5, 15, 40, 30, 10, 0],
    "Long doc": [0, 0, 2, 8, 15, 55, 20],
}

TASK_PRESETS = {
    "Classify": {"i": 2048, "o": 32},
    "Extract": {"i": 4096, "o": 256},
    "Summarize": {"i": 8192, "o": 512},
    "Rephrase": {"i": 2048, "o": 2048},
    "Synth gen": {"i": 512, "o": 4096},
    "Score": {"i": 4096, "o": 8},
}

# Cloud model registry with representative public API pricing in $/M tokens.
# `quality` and `token_efficiency` are calibrated below from Artificial Analysis'
# Intelligence Index and Intelligence Index output-token usage (verbosity).
CLOUD_MODELS = {
    "gpt-5": {
        "label": "GPT-5",
        "vendor": "OpenAI",
        "in_per_m": 1.25,
        "cached_in_per_m": 0.125,
        "out_per_m": 10.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gpt-5-mini": {
        "label": "GPT-5 mini",
        "vendor": "OpenAI",
        "in_per_m": 0.25,
        "cached_in_per_m": 0.025,
        "out_per_m": 2.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gpt-5-nano": {
        "label": "GPT-5 nano",
        "vendor": "OpenAI",
        "in_per_m": 0.05,
        "cached_in_per_m": 0.005,
        "out_per_m": 0.40,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "claude-opus": {
        "label": "Claude Opus 4",
        "vendor": "Anthropic",
        "in_per_m": 15.00,
        "cached_in_per_m": 1.50,
        "out_per_m": 75.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "claude-sonnet": {
        "label": "Claude Sonnet 4",
        "vendor": "Anthropic",
        "in_per_m": 3.00,
        "cached_in_per_m": 0.30,
        "out_per_m": 15.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "claude-haiku": {
        "label": "Claude Haiku 4",
        "vendor": "Anthropic",
        "in_per_m": 0.80,
        "cached_in_per_m": 0.08,
        "out_per_m": 4.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gemini-pro": {
        "label": "Gemini 2.5 Pro",
        "vendor": "Google",
        "in_per_m": 1.25,
        "cached_in_per_m": 0.31,
        "out_per_m": 10.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gemini-flash": {
        "label": "Gemini 2.5 Flash",
        "vendor": "Google",
        "in_per_m": 0.30,
        "cached_in_per_m": 0.075,
        "out_per_m": 2.50,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gemini-flash-lite": {
        "label": "Gemini 2.5 Flash Lite",
        "vendor": "Google",
        "in_per_m": 0.10,
        "cached_in_per_m": 0.025,
        "out_per_m": 0.40,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "mistral-medium": {
        "label": "Mistral Medium 3.5",
        "vendor": "Mistral",
        "in_per_m": 1.50,
        "cached_in_per_m": 0.15,
        "out_per_m": 7.50,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "mistral-large": {
        "label": "Mistral Large",
        "vendor": "Mistral",
        "in_per_m": 2.00,
        "cached_in_per_m": 0.50,
        "out_per_m": 6.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "mistral-large-2": {
        "label": "Mistral Large 2",
        "vendor": "Mistral",
        "in_per_m": 2.00,
        "cached_in_per_m": 0.50,
        "out_per_m": 6.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "codestral-2501": {
        "label": "Codestral 2501",
        "vendor": "Mistral",
        "in_per_m": 0.20,
        "cached_in_per_m": 0.05,
        "out_per_m": 0.60,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "deepseek-v3": {
        "label": "DeepSeek V3",
        "vendor": "DeepSeek",
        "in_per_m": 0.27,
        "cached_in_per_m": 0.07,
        "out_per_m": 1.10,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
}

AA_CLOUD_METRICS: dict[str, tuple[float, float]] = {
    "gpt-5": (45.0, 76.0),
    "gpt-5-mini": (41.0, 69.0),
    "gpt-5-nano": (27.0, 110.0),
    "claude-opus": (50.0, 72.0),         # Proxy from Claude Opus 4.5 (Reasoning).
    "claude-sonnet": (39.0, 55.0),       # Claude 4 Sonnet (Reasoning).
    "claude-haiku": (37.0, 87.0),        # Proxy from Claude 4.5 Haiku (Reasoning).
    "gemini-pro": (35.0, 55.0),
    "gemini-flash": (21.0, 17.0),
    "gemini-flash-lite": (13.0, 36.0),
    "mistral-medium": (39.0, 90.0),      # Mistral Medium 3.5.
    "mistral-large": (13.0, 2.6),        # Proxy from Mistral Large 2 (Jul '24), verbosity from Nov '24 refresh.
    "mistral-large-2": (15.0, 2.6),
    "codestral-2501": (15.0, 4.4),       # Proxy from Devstral Small (Jul '25'); no AA page for Codestral 2501 found.
    "deepseek-v3": (16.0, 2.6),
}

for _k, (_score, _verbosity_m) in AA_CLOUD_METRICS.items():
    if _k in CLOUD_MODELS:
        CLOUD_MODELS[_k]["quality"] = aa_intelligence_to_quality(_score)
        CLOUD_MODELS[_k]["token_efficiency"] = aa_output_tokens_to_efficiency(_verbosity_m)

# Vertex availability matrix: GCP regions where each cloud-model family is
# served today. Models not on Vertex Europe (e.g. OpenAI via Azure, DeepSeek)
# have an empty tuple and no grid-intensity estimate. Regions are enriched
# onto CLOUD_MODELS at the bottom of the file, after the carbon-intensity
# tables and helpers are defined.
_GEMINI_PRO_ZONES = (
    "europe-west1", "europe-west4", "europe-west8", "europe-west9",
    "europe-central2", "europe-north1", "europe-southwest1",
)
_GEMINI_FLASH_ZONES = _GEMINI_PRO_ZONES + ("europe-west2", "europe-west3")
_CLAUDE_ZONES = ("europe-west1",)
_MISTRAL_ZONES = ("europe-west4",)

CLOUD_MODEL_ZONES: dict[str, tuple[str, ...]] = {
    "gpt-5": (),
    "gpt-5-mini": (),
    "gpt-5-nano": (),
    "claude-opus": _CLAUDE_ZONES,
    "claude-sonnet": _CLAUDE_ZONES,
    "claude-haiku": _CLAUDE_ZONES,
    "gemini-pro": _GEMINI_PRO_ZONES,
    "gemini-flash": _GEMINI_FLASH_ZONES,
    "gemini-flash-lite": _GEMINI_PRO_ZONES,
    "mistral-medium": _MISTRAL_ZONES,
    "mistral-large": _MISTRAL_ZONES,
    "mistral-large-2": _MISTRAL_ZONES,
    "codestral-2501": _MISTRAL_ZONES,
    "deepseek-v3": (),
}

# Steepness of the quality/difficulty sigmoid used by success_rate(). k=10 gives a
# ~0.1-quality-edge → ~73% success and a 0.2 edge → ~88%. Tune if calibration demands.
SUCCESS_RATE_SIGMOID_K = 10.0


def success_rate(quality: float, difficulty: float, k: float = SUCCESS_RATE_SIGMOID_K) -> float:
    """Probability that a model of given `quality` succeeds on a task of given `difficulty`.

    Continuous replacement for the old discrete tier-distance success curve:
    `sigmoid(k · (quality − difficulty))`. Quality ≫ difficulty → ~1.0; matched → 0.5;
    quality ≪ difficulty → ~0.0.
    """
    x = max(min(k * (quality - difficulty), 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-x))


def required_quality(difficulty: float, min_success_rate: float, k: float = SUCCESS_RATE_SIGMOID_K, quality_floor: float = 0.0) -> float:
    """Inverse of success_rate(): minimum model quality that clears `min_success_rate`
    at the given `difficulty`. Returns a value on the same [0, 1] quality axis the model
    catalog uses (AA Intelligence Index calibrated into 0.30..0.95)."""
    slo = min(max(float(min_success_rate), 1e-4), 1 - 1e-4)
    logit = math.log(slo / (1.0 - slo))
    floor = min(max(float(quality_floor or 0.0), 0.0), 1.0)
    return min(max(float(difficulty) + logit / k, floor, 0.0), 1.0)


# Corporate cloud catalog presets. The cloud isn't an open marketplace — corp procurement
# decides which models flow through their gateway. Today's reality (Gemini-only, no
# Anthropic/OpenAI) is the "current" preset; "advocated" models the realistic ask
# (Claude on Vertex Europe). Anything not in the active preset is unavailable as a
# spillover destination, so demand needing a tier nobody on the list can serve gets
# destroyed (not leaked).
CORPO_CLOUD_PRESETS = {
    "current": {
        "label": "Current corpo gateway (Gemini + Mistral)",
        "models": (
            "gemini-flash-lite",
            "gemini-flash",
            "gemini-pro",
            "codestral-2501",
            "mistral-medium",
            "mistral-large",
            "mistral-large-2",
        ),
    },
    "advocated": {
        "label": "Advocated · + Anthropic on Vertex EU",
        "models": (
            "gemini-flash-lite",
            "gemini-flash",
            "gemini-pro",
            "codestral-2501",
            "mistral-medium",
            "mistral-large",
            "mistral-large-2",
            "claude-haiku",
            "claude-sonnet",
            "claude-opus",
        ),
    },
}
CORPO_CLOUD_DEFAULT = "current"


def corpo_cloud_models(name: str) -> tuple[str, ...]:
    preset = CORPO_CLOUD_PRESETS.get(name) or CORPO_CLOUD_PRESETS[CORPO_CLOUD_DEFAULT]
    return preset["models"]


# Opinionated use-case definitions so the demo is one click from a realistic story.
# These define workload dynamics: difficulty, SLO, price ceiling, capability gates,
# batchability, token shape, and latent-demand unlock economics. tokens_day and
# latent_jobs_day are only starting scales for a newly-added instance; switching a
# selected use case to another kind preserves the organization's scale.
#
# wtp_per_m is the ceiling price ($/M tokens) above which this use case refuses to buy
# (flees to cloud if it's cheaper, else shelves the work). requires lists hard capability
# gates. difficulty ∈ [0,1] is paired with each candidate model's quality by success_rate().
# min_success_rate is the SLO floor. latent_jobs_day is additional demand that is not
# economically viable today but unlocks when the cheapest model reaches unlock_price_per_m.
SCALE_MODELS = {
    "linear": "Linear",
    "quadratic": "Quadratic",
    "network": "Network/graph",
    "corpus": "Corpus/backfill",
    "custom": "Custom formula",
}

PROJECT_PRESETS = [
    {"key": "classify",       "name": "Mass classification",        "difficulty": 0.10, "tokens_day": 3.0e9, "scale_value": 5_000_000, "scale_kind": {"model": "linear", "label": "Records processed", "unit": "records/day", "token_multiplier": 600, "min": 0, "max": 10_000_000, "step": 10_000, "formula": "records/day x average tokens per record"}, "wtp_per_m": 0.25, "requires": (),                    "min_success_rate": 0.80, "quality_floor": 0.35, "batch_eligible": True, "latent_jobs_day": 8.0e9, "unlock_price_per_m": 0.05, "in_pre": "Classify", "out_pre": "Classify", "scale_hint": "Records/day x tokens per record; mostly batchable queues."},
    {"key": "summarize",      "name": "Doc summarization",          "difficulty": 0.25, "tokens_day": 1.5e9, "scale_value": 60_000, "scale_kind": {"model": "linear", "label": "Documents summarized", "unit": "documents/day", "token_multiplier": 25_000, "min": 0, "max": 250_000, "step": 100, "formula": "documents/day x average document+summary tokens"}, "wtp_per_m": 0.90, "requires": ("ctx_128k",),         "min_success_rate": 0.85, "quality_floor": 0.50, "batch_eligible": True, "latent_jobs_day": 3.0e9, "unlock_price_per_m": 0.20, "in_pre": "Long doc", "out_pre": "Long doc", "scale_hint": "Documents/day x document length; periodic backfills can dominate."},
    {"key": "chatbot",        "name": "Customer chatbot",           "difficulty": 0.30, "tokens_day": 2.0e9, "scale_value": 80_000, "scale_kind": {"model": "linear", "label": "Support tickets", "unit": "tickets/day", "token_multiplier": 25_000, "min": 0, "max": 250_000, "step": 100, "formula": "tickets/day x turns x tokens per turn"}, "wtp_per_m": 1.20, "requires": ("tools",),            "min_success_rate": 0.95, "quality_floor": 0.60, "in_pre": "Chat",     "out_pre": "Chat",     "scale_hint": "Users or tickets/day x turns x tokens per turn; interactive peak matters."},
    {"key": "email_corrector","name": "Email correction copilot",   "difficulty": 0.18, "tokens_day": 250e6, "scale_value": 5_000, "scale_kind": {"model": "linear", "label": "Enabled headcount", "unit": "employees", "token_multiplier": 50_000, "min": 0, "max": 25_000, "step": 10, "formula": "employees x messages/day x correction tokens"}, "wtp_per_m": 0.65, "requires": (),                    "min_success_rate": 0.90, "quality_floor": 0.40, "in_pre": "Chat",     "out_pre": "Chat",     "scale_hint": "Headcount x messages/day x correction tokens; roughly linear with staff."},
    {"key": "coding",         "name": "Coding assistant",           "difficulty": 0.55, "tokens_day": 1.2e9, "scale_value": 8_000, "scale_kind": {"model": "linear", "label": "Developer seats", "unit": "developers", "token_multiplier": 150_000, "min": 0, "max": 25_000, "step": 10, "formula": "developer seats x active days x code context"}, "wtp_per_m": 4.00, "requires": ("tools", "ctx_128k"), "min_success_rate": 0.85, "quality_floor": 0.70, "in_pre": "Code",     "out_pre": "Code",     "scale_hint": "Developer seats x active days x code context; bursty during migrations."},
    {"key": "meeting_notes",  "name": "Meeting notes assistant",    "difficulty": 0.35, "tokens_day": 600e6, "scale_value": 6_000, "scale_kind": {"model": "linear", "label": "Recorded meeting time", "unit": "meeting hours/day", "token_multiplier": 100_000, "min": 0, "max": 20_000, "step": 10, "formula": "meeting hours/day x transcript+summary tokens"}, "wtp_per_m": 1.50, "requires": ("ctx_128k",),         "min_success_rate": 0.88, "quality_floor": 0.55, "batch_eligible": True, "latent_jobs_day": 1.0e9, "unlock_price_per_m": 0.35, "in_pre": "Long doc", "out_pre": "Long doc", "scale_hint": "Meeting hours/day x transcript length; can be delayed after calls."},
    {"key": "evals",          "name": "Batch evaluations",          "difficulty": 0.45, "tokens_day": 800e6, "scale_value": 400_000, "scale_kind": {"model": "linear", "label": "Evaluation prompts", "unit": "eval prompts/day", "token_multiplier": 2_000, "min": 0, "max": 2_000_000, "step": 1_000, "formula": "eval prompts/day x average prompt+judgment tokens"}, "wtp_per_m": 2.00, "requires": (),                    "min_success_rate": 0.90, "quality_floor": 0.60, "batch_eligible": True, "latent_jobs_day": 2.0e9, "unlock_price_per_m": 0.50, "in_pre": "RAG",      "out_pre": "Classify", "scale_hint": "Runs/day x eval set size; off-peak capacity is usually acceptable."},
    {"key": "inbox_archive",  "name": "Decade inbox knowledge base","difficulty": 0.50, "tokens_day": 120e6, "scale_value": 50, "scale_kind": {"model": "corpus", "label": "Indexed mailboxes", "unit": "mailboxes indexed", "token_multiplier": 2_400_000, "min": 0, "max": 5_000, "step": 10, "formula": "mailboxes x retained years x messages, dailyized as batch load"}, "wtp_per_m": 1.75, "requires": ("ctx_128k",),         "min_success_rate": 0.86, "quality_floor": 0.62, "batch_eligible": True, "latent_jobs_day": 12.0e9, "unlock_price_per_m": 0.30, "in_pre": "RAG",      "out_pre": "Long doc", "scale_hint": "Mailboxes x retained years x messages; one-time corpus scale dominates."},
    {"key": "longctx",        "name": "Long-ctx analytics",         "difficulty": 0.70, "tokens_day": 400e6, "scale_value": 200, "scale_kind": {"model": "linear", "label": "Large analyses", "unit": "analyses/day", "token_multiplier": 2_000_000, "min": 0, "max": 1_000, "step": 1, "formula": "analyses/day x full-source-pack length"}, "wtp_per_m": 8.00, "requires": ("ctx_128k",),         "min_success_rate": 0.90, "quality_floor": 0.78, "latent_jobs_day": 1.0e9, "unlock_price_per_m": 3.00, "in_pre": "Long doc", "out_pre": "Long doc", "scale_hint": "Analyses/day x full-source-pack length; limited volume, large prompts."},
    {"key": "research",       "name": "Deep research agent",        "difficulty": 0.90, "tokens_day": 150e6, "scale_value": 300, "scale_kind": {"model": "custom", "label": "Research jobs", "unit": "investigations/day", "token_multiplier": 500_000, "min": 0, "max": 1_000, "step": 1, "formula": "analysts x investigations/day x agent depth"}, "wtp_per_m": 20.00,"requires": ("tools", "reasoning"),"min_success_rate": 0.95, "quality_floor": 0.90, "latent_jobs_day": 500e6, "unlock_price_per_m": 5.00, "in_pre": "RAG",      "out_pre": "Long doc", "scale_hint": "Analysts x investigations/day x agent depth; low volume, high value."},
]

DAY_SHAPES = {
    "flat": {
        "label": "Flat 24/7",
        "weights": [1.0] * 24,
        "note": "Even demand across the whole day.",
    },
    "workday": {
        "label": "Enterprise workday",
        "weights": [
            0.30, 0.24, 0.20, 0.18, 0.18, 0.22, 0.35, 0.55,
            0.82, 1.00, 1.10, 1.16, 1.20, 1.18, 1.10, 1.00,
            0.92, 0.82, 0.68, 0.54, 0.44, 0.38, 0.34, 0.32,
        ],
        "note": "Daytime-heavy office demand with a mild evening shoulder.",
    },
    "consumer": {
        "label": "Consumer evening",
        "weights": [
            0.32, 0.28, 0.24, 0.22, 0.20, 0.22, 0.30, 0.42,
            0.58, 0.70, 0.78, 0.84, 0.88, 0.92, 0.98, 1.04,
            1.10, 1.18, 1.28, 1.36, 1.40, 1.24, 0.92, 0.54,
        ],
        "note": "Lower daytime demand with a strong evening consumer peak.",
    },
    "globalsaas": {
        "label": "Follow the sun",
        "weights": [
            0.55, 0.52, 0.50, 0.48, 0.50, 0.56, 0.64, 0.74,
            0.86, 0.96, 1.02, 1.08, 1.12, 1.16, 1.18, 1.16,
            1.12, 1.08, 1.00, 0.90, 0.82, 0.74, 0.68, 0.60,
        ],
        "note": "Broader global SaaS demand with fewer sharp regional peaks.",
    },
    "nightbatch": {
        "label": "Night batch",
        "weights": [
            1.28, 1.34, 1.38, 1.40, 1.34, 1.18, 0.92, 0.68,
            0.48, 0.38, 0.34, 0.32, 0.32, 0.34, 0.38, 0.42,
            0.50, 0.62, 0.78, 0.92, 1.00, 1.08, 1.18, 1.24,
        ],
        "note": "Off-peak-heavy traffic that leans into discounted overnight processing.",
    },
}


# Hour index 0 = local midnight.
# Values are grams CO2-equivalent per kWh delivered.
CARBON_INTENSITY_HOURLY: dict[str, list[float]] = {
    "BE": [290, 280, 270, 265, 270, 290, 320, 340, 330, 310, 290, 270, 260, 260, 270, 290, 320, 350, 380, 400, 390, 360, 330, 310],
    "UK": [230, 220, 210, 205, 210, 230, 260, 280, 270, 250, 230, 220, 210, 210, 220, 240, 260, 290, 320, 340, 330, 300, 270, 250],
    "DE": [420, 410, 400, 395, 400, 420, 450, 480, 470, 450, 430, 410, 400, 395, 405, 430, 460, 500, 540, 560, 550, 520, 480, 450],
    "NL": [410, 400, 390, 380, 385, 410, 440, 470, 460, 440, 420, 400, 390, 385, 395, 420, 450, 490, 530, 550, 540, 510, 470, 440],
    "CH": [120, 115, 110, 110, 115, 120, 130, 135, 130, 125, 120, 115, 110, 110, 115, 120, 130, 140, 150, 155, 150, 140, 130, 125],
    "IT": [360, 350, 340, 330, 335, 360, 400, 420, 410, 380, 360, 340, 330, 330, 340, 360, 390, 430, 470, 490, 480, 450, 420, 390],
    "FR": [60, 55, 55, 55, 55, 60, 65, 70, 65, 60, 55, 50, 50, 50, 50, 55, 60, 70, 75, 80, 80, 75, 70, 65],
    "FI": [140, 135, 130, 130, 135, 140, 150, 155, 150, 145, 140, 135, 130, 130, 135, 140, 150, 160, 170, 180, 175, 165, 155, 150],
    "PL": [650, 640, 630, 625, 630, 650, 700, 730, 720, 700, 680, 660, 650, 645, 655, 680, 720, 760, 800, 820, 810, 780, 740, 700],
    "ES": [250, 240, 230, 220, 230, 260, 300, 320, 290, 240, 200, 180, 150, 160, 170, 200, 260, 320, 380, 400, 380, 350, 310, 280],
}

COUNTRIES = {
    "BE": "Belgium",
    "UK": "United Kingdom",
    "DE": "Germany",
    "NL": "Netherlands",
    "CH": "Switzerland",
    "IT": "Italy",
    "FR": "France",
    "FI": "Finland",
    "PL": "Poland",
    "ES": "Spain",
}
DEFAULT_COUNTRY = "FR"

# GCP region → country code. Only the European Vertex regions from the
# cloud-model availability matrix are mapped; add more as they come into scope.
GCP_ZONE_COUNTRY: dict[str, str] = {
    "europe-west1": "BE",       # St. Ghislain
    "europe-west2": "UK",       # London
    "europe-west3": "DE",       # Frankfurt
    "europe-west4": "NL",       # Eemshaven
    "europe-west8": "IT",       # Milan
    "europe-west9": "FR",       # Paris
    "europe-central2": "PL",    # Warsaw
    "europe-north1": "FI",      # Hamina
    "europe-southwest1": "ES",  # Madrid
}


def carbon_intensity(country: str, hour: int) -> float:
    """Grams CO2-equivalent per kWh for the given country at the given local hour (0-23)."""
    series = CARBON_INTENSITY_HOURLY.get(country) or CARBON_INTENSITY_HOURLY[DEFAULT_COUNTRY]
    return series[hour % 24]


def carbon_intensity_avg(country: str, hour_weights: list[float] | None = None) -> float:
    """Demand-weighted daily average gCO2/kWh for a country. Unweighted if weights is None."""
    series = CARBON_INTENSITY_HOURLY.get(country) or CARBON_INTENSITY_HOURLY[DEFAULT_COUNTRY]
    if not hour_weights:
        return sum(series) / len(series)
    total_w = sum(hour_weights) or 1.0
    return sum(series[h] * hour_weights[h] for h in range(24)) / total_w


def cloud_model_grid_intensity(zones: tuple[str, ...], hour_weights: list[float] | None = None) -> float:
    """Unweighted (or demand-weighted) gCO2/kWh averaged across the zones a cloud model is served from.

    Zones we can't map to a country are ignored; returns 0.0 when none resolve."""
    countries = [GCP_ZONE_COUNTRY[z] for z in zones if z in GCP_ZONE_COUNTRY]
    if not countries:
        return 0.0
    return sum(carbon_intensity_avg(c, hour_weights) for c in countries) / len(countries)


# Enrich CLOUD_MODELS once the carbon-intensity tables are in scope.
for _k, _zones in CLOUD_MODEL_ZONES.items():
    if _k not in CLOUD_MODELS:
        continue
    CLOUD_MODELS[_k]["gcp_zones"] = _zones
    CLOUD_MODELS[_k]["regions"] = tuple(
        GCP_ZONE_COUNTRY[z] for z in _zones if z in GCP_ZONE_COUNTRY
    )
    CLOUD_MODELS[_k]["grid_gco2_per_kwh"] = cloud_model_grid_intensity(_zones)


def models_by_category() -> dict[str, list[Model]]:
    cats: dict[str, list[Model]] = {}
    for m in MODELS.values():
        if m.hidden:
            continue
        cats.setdefault(m.cat, []).append(m)
    return cats


# Three top-level kinds used by the model picker tabs. Order is the tab order;
# the first non-empty kind is the default active tab.
MODEL_KINDS: tuple[tuple[str, str], ...] = (
    ("llm", "LLM"),
    ("embedding", "Embedding"),
    ("asr", "ASR"),
)


def _model_kind(m: Model) -> str:
    if m.is_realtime_only:
        return "asr"
    if m.is_embedding_model:
        return "embedding"
    return "llm"


def models_by_kind() -> dict[str, dict[str, list[Model]]]:
    """Models grouped first by kind (LLM/Embedding/ASR) then by catalog cat.

    Used by the model picker to render one tab per kind, preserving the
    sub-grouping (e.g. Mistral / Qwen / DeepSeek inside LLM) within each tab.
    """
    out: dict[str, dict[str, list[Model]]] = {kind: {} for kind, _ in MODEL_KINDS}
    for m in MODELS.values():
        if m.hidden:
            continue
        out[_model_kind(m)].setdefault(m.cat, []).append(m)
    return out


def gpus_by_vendor() -> dict[str, list[GPU]]:
    cats: dict[str, list[GPU]] = {}
    for g in GPUS.values():
        cats.setdefault(g.vendor_label, []).append(g)
    return cats


def gpu_cards_by_vendor() -> dict[str, list[GPUCard]]:
    cats: dict[str, list[GPUCard]] = {}
    for card in GPU_CARDS:
        cats.setdefault(card.vendor, []).append(card)
    return cats
