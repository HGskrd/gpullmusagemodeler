"""Precision and quantization specifications."""

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
    "tenstorrent": "Tenstorrent",
    "furiosa": "FuriosaAI",
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
    "F32": NumberFormatSpec(
        "F32", "FP32", 4.0, "Float32 tensor, usually tiny global scale auxiliaries."
    ),
    "F8_E4M3": NumberFormatSpec(
        "F8_E4M3", "FP8 E4M3", 1.0, "FP8 E4M3 tensor; its role comes from checkpoint metadata."
    ),
    "U8": NumberFormatSpec(
        "U8", "UINT8", 1.0, "Unsigned byte storage; may contain packed weights or scales."
    ),
}

NUMBER_FORMAT_SPECS.update(
    {
        key: NumberFormatSpec(key, label, size, "Checkpoint storage dtype.")
        for key, label, size in (
            ("F16", "FP16", 2),
            ("F64", "FP64", 8),
            ("F8_E5M2", "FP8 E5M2", 1),
            ("F8_E8M0", "E8M0 scale", 1),
            ("I8", "INT8", 1),
            ("I16", "INT16", 2),
            ("U16", "UINT16", 2),
            ("I32", "INT32", 4),
            ("U32", "UINT32", 4),
            ("I64", "INT64", 8),
            ("U64", "UINT64", 8),
            ("BOOL", "BOOL", 1),
        )
    }
)


@dataclass(frozen=True)
class QuantizationScheme:
    encoding: str
    block_shape: tuple[int, ...]
    scale_format: str
    zero_point_format: str = "none"


QUANTIZATION_SCHEMES = {
    "bf16": QuantizationScheme("BF16", (), "none"),
    "fp8": QuantizationScheme("E4M3", (128, 128), "checkpoint-specific FP32/E8M0"),
    "mxfp4": QuantizationScheme("E2M1", (32,), "E8M0"),
    "nvfp4": QuantizationScheme("E2M1", (16,), "E4M3 block + FP32 tensor scale"),
}
SERVING_RUNTIMES = ("generic", "vllm", "sglang", "trtllm")
EXECUTION_BACKENDS = ("legacy", "auto", "native", "w4a16", "expanded")
CACHE_PRECISIONS = ("legacy", "auto", "bf16", "fp8")
RECURRENT_PRECISIONS = ("legacy", "auto", "bf16", "fp32")


@dataclass(frozen=True)
class WeightComponent:
    role: str
    weight_format: str
    activation_format: str
    active_params: float
    bytes_per_param: float
    evidence: str

    @property
    def active_bytes(self) -> float:
        return self.active_params * self.bytes_per_param


@dataclass(frozen=True)
class WeightMemory:
    checkpoint_bytes: float
    gpu_bytes: float
    host_bytes: float
    repacking_bytes: float
    loading_peak_lower_bound_bytes: float
    loading_workspace_bytes: float | None = None


def _storage_bytes(format_counts: dict[str, int]) -> float:
    total = 0.0
    for fmt, elems in format_counts.items():
        spec = NUMBER_FORMAT_SPECS.get(fmt)
        if spec is None:
            raise ValueError(f"Unknown checkpoint storage dtype: {fmt}")
        if not isinstance(elems, int) or elems < 0:
            raise ValueError(f"Invalid storage element count for {fmt}")
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
    container: str = "safetensors"
    storage_evidence: str = "exact tensor bytes"
    active_traffic_evidence: str = "estimated from active operator shares"
    compute_evidence: str = "estimated operator shares; uncalibrated throughput"
    runtime_evidence: str = "runtime support requires a matching serving recipe"
    scheme: QuantizationScheme | None = None

    def __post_init__(self) -> None:
        _storage_bytes(self.storage_format_counts)
        if self.precision_key not in PRECISION_SPECS:
            raise ValueError(f"Unknown artifact precision: {self.precision_key}")
        if self.total_weight_bytes_override is not None and (
            not math.isfinite(self.total_weight_bytes_override)
            or self.total_weight_bytes_override <= 0
        ):
            raise ValueError("Invalid checkpoint byte total")
        shares = self.compute_precision_shares
        if shares and (
            any(
                k not in PRECISION_SPECS or not math.isfinite(v) or v < 0 for k, v in shares.items()
            )
            or not math.isclose(sum(shares.values()), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("Compute precision shares must be finite and sum to one")

    @property
    def artifact_id(self) -> str:
        return f"hf:{self.source_repo}@{self.source_revision}"

    @property
    def format_scheme(self) -> QuantizationScheme:
        return self.scheme or QUANTIZATION_SCHEMES[self.precision_key]

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
