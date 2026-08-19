"""Precision and quantization specifications."""

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
    "F8_E4M3": NumberFormatSpec("F8_E4M3", "FP8 E4M3", 1.0, "FP8 E4M3 scale or activation tensor."),
    "U8": NumberFormatSpec(
        "U8", "Packed FP4", 1.0, "Unsigned byte storage for packed FP4 payloads."
    ),
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
