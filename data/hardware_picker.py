"""Owned-hardware chooser: families, purchase channels and explicit estimate limits.

Compatibility profiles stay in GPUS for saved scenarios. Only reviewed, released
hardware is offered for a new purchase; cloud-only products are outside this UI.
"""

from dataclasses import dataclass

from .hardware import GPU_CARDS, GPUS, GPUCard

HARDWARE_REVIEWED_AT = "2026-10-17"
# Do not delete these keys: existing exported scenarios must remain readable.
UNAVAILABLE_HARDWARE = {
    "RUBIN_NVL72": "Preliminary platform; purchasable configuration not verified.",
    "MI455X": "Helios deployment profile; independent purchase not verified.",
    "HELIOS_MI455X": "Preliminary reference system; purchase not verified.",
    "MI400": "Compatibility alias for saved plans.",
    "CrescentIsland": "Announced product; purchasable configuration not verified.",
    "MAC_STUDIO_M5_ULTRA": "512GB configuration starts shipping after this review date.",
    "A2000_MOBILE": "Laptop component without a configuration-matched whole-system profile.",
}
LEGACY_HARDWARE = frozenset({"V100", "T4", "A30", "A40", "MI250X", "Gaudi2", "A4000"})

A100_SOURCE = "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf"
H100_SOURCE = "https://www.nvidia.com/en-us/data-center/h100/"
H100_PCIE_SOURCE = "https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c"
H200_SOURCE = "https://www.nvidia.com/en-us/data-center/h200/"
APPLE_SOURCE = "https://support.apple.com/en-sg/128107"
RYZEN_SOURCE = "https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html"
RYZEN_BUY = "https://www.gmktec.com/products/amd-ryzen%E2%84%A2-ai-max-395-evo-x2-ai-mini-pc"
GEFORCE_TENSOR_SOURCE = "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf"
PRO_TENSOR_SOURCE = "https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf"
HARDWARE_SOURCES = {
    **dict.fromkeys(("RTX5090", "RTX4090", "RTX3090"), GEFORCE_TENSOR_SOURCE),
    "RTXPRO6000_BW_WS": PRO_TENSOR_SOURCE,
    **dict.fromkeys(("A100", "A100_40", "A100_80_PCIE", "A100_40_SXM"), A100_SOURCE),
    **dict.fromkeys(("H100", "H100_NVL"), H100_SOURCE),
    "H100_PCIE": H100_PCIE_SOURCE,
    **dict.fromkeys(("H200", "H200_NVL"), H200_SOURCE),
    "RTX3060_12": "https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/",
    "RTX4060TI_16": "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/",
    "RTX5060TI_16": "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/",
    "RYZEN_AI_MAX_395_128": RYZEN_SOURCE,
    **dict.fromkeys(
        ("MAC_STUDIO_M5_MAX", "MAC_STUDIO_M5_MAX_64", "MAC_STUDIO_M5_ULTRA_256"), APPLE_SOURCE
    ),
    "MI250X": "https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi250.html",
    "Gaudi3": "https://cdrdv2-public.intel.com/817488/Gaudi%203%20PCIe%20Product%20Brief_RB_1_V6.pdf",
    "MAC_STUDIO_M4_MAX_64": "https://support.apple.com/en-us/122211",
    "MAC_STUDIO_M3_ULTRA_256": "https://support.apple.com/en-us/122211",
    "MAC_MINI_M4_PRO_48": "https://support.apple.com/en-us/121555",
}
FAMILY_KEYS = {
    **dict.fromkeys(("A100", "A100_40", "A100_80_PCIE", "A100_40_SXM"), "A100"),
    **dict.fromkeys(("H100", "H100_PCIE", "H100_NVL"), "H100"),
    **dict.fromkeys(("H200", "H200_NVL"), "H200"),
    **dict.fromkeys(("RTXPRO6000_BSE", "RTXPRO6000_BW_WS"), "RTX PRO 6000 Blackwell"),
    **dict.fromkeys(("TT_BLACKHOLE_P100A", "TT_BLACKHOLE_P150"), "Tenstorrent Blackhole cards"),
    **dict.fromkeys(("MAC_STUDIO_M5_MAX", "MAC_STUDIO_M5_MAX_64"), "Mac Studio M5 Max"),
    **dict.fromkeys(("MAC_STUDIO_M4_MAX", "MAC_STUDIO_M4_MAX_64"), "Mac Studio M4 Max"),
    **dict.fromkeys(("MAC_STUDIO_M3_ULTRA", "MAC_STUDIO_M3_ULTRA_256"), "Mac Studio M3 Ultra"),
    **dict.fromkeys(("MAC_MINI_M4_PRO", "MAC_MINI_M4_PRO_48"), "Mac mini M4 Pro"),
    "MAC_STUDIO_M5_ULTRA_256": "Mac Studio M5 Ultra",
}


@dataclass(frozen=True)
class HardwareChoice:
    key: str
    card: GPUCard
    family: str
    legacy: bool
    availability: str
    specs: str
    runtime: str
    source: str
    purchase_source: str
    reviewed_at: str = HARDWARE_REVIEWED_AT


def hardware_choices() -> list[HardwareChoice]:
    choices = []
    for card in GPU_CARDS:
        for option in card.planner_options:
            key = option.gpu_key
            if key in UNAVAILABLE_HARDWARE:
                continue
            gpu = GPUS[key]
            source = HARDWARE_SOURCES.get(key, "")
            specs = (
                card.note or "Published hardware specifications; planner throughput is an estimate."
            )
            runtime = "CUDA serving; model, precision and kernel support must match your runtime."
            if gpu.vendor == "apple":
                specs += " Unified memory is shared with the OS and host allocations. Compute and sustained bandwidth are proxies."
                runtime = "MLX / llama.cpp on Metal. CUDA kernel compatibility is not implied."
            elif gpu.vendor == "amd":
                runtime = "ROCm / llama.cpp support varies by device, model, quantization and OS."
            elif gpu.vendor == "intel":
                runtime = "Gaudi requires SynapseAI; Arc requires an Intel-compatible serving backend. Tensor peaks include proxies."
            elif gpu.vendor == "tenstorrent":
                runtime = "TT-Metal / TT-LLM supported models only. BLOCKFP8 is not IEEE FP8; BF16 is a proxy."
            elif gpu.vendor == "furiosa":
                runtime = "Furiosa SDK and compiled supported models; verify model and precision coverage."
            if key == "RYZEN_AI_MAX_395_128":
                specs = "128 GB system RAM; 96 GB GPU allocation budget, not 128 GB dedicated VRAM. Peak bandwidth 256 GB/s; sustained bandwidth 200 GB/s and BF16/FP8 rooflines 29.7 TFLOPS are uncalibrated conservative proxies. 120 W is an APU power budget, not whole-PC wall power."
            elif key in {"A100_80_PCIE", "A100_40", "H100_PCIE"}:
                specs += (
                    " PCIe-only local topology assumed; optional NVLink bridges are not included."
                )
            elif key in {"H100_NVL", "H200_NVL"}:
                specs += (
                    " Per-GPU memory and dense compute. NVLink domain assumes "
                    + ("two" if key == "H100_NVL" else "four")
                    + " bridged GPUs; a larger pool is not one NVLink domain."
                )
            elif key == "RTX3060_12":
                specs += " BF16 uses conservative dense FP32-accumulation throughput; FP8 is a BF16 fallback, not native FP8. PCIe 4.0 x16."
            elif key in {"RTXPRO6000_BSE", "RTXPRO5000_BW_72"}:
                specs += " Dense compute rates are derived proxies from rounded sparse FP4 headlines, not measured throughput."
            elif key in {"RTX5090", "RTX4090", "RTX3090"}:
                specs += " Tensor compute uses dense FP32 accumulation, not sparse marketing rates."
            elif key in {"RTX4060TI_16", "RTX5060TI_16"}:
                specs += " Conservative dense rates use FP32 accumulation, derived from architecture ratios; sparse AI TOPS are not dense FLOPS. PCIe x8 host link; no NVLink."
            legacy = key in LEGACY_HARDWARE
            channel = (
                "Reseller / used"
                if legacy or key in {"RTX3090", "RTX3060_12"}
                else "Retail / hardware vendor"
            )
            choices.append(
                HardwareChoice(
                    key,
                    card,
                    FAMILY_KEYS.get(key, card.name),
                    legacy,
                    channel,
                    specs,
                    runtime,
                    source,
                    RYZEN_BUY if key == "RYZEN_AI_MAX_395_128" else source,
                )
            )
    return choices


# Explicit reading order mirrors the approved compact picker. Each key occurs once.
PICKER_FAMILIES = {
    "NVIDIA": (
        ("H100", (("H100", "80 GB SXM"), ("H100_PCIE", "80 GB PCIe"), ("H100_NVL", "94 GB NVL"))),
        ("H200", (("H200", "141 GB SXM"), ("H200_NVL", "141 GB NVL"))),
        (
            "A100",
            (
                ("A100_40", "40 GB PCIe"),
                ("A100_80_PCIE", "80 GB PCIe"),
                ("A100_40_SXM", "40 GB SXM"),
                ("A100", "80 GB SXM"),
            ),
        ),
        (
            "RTX PRO",
            (
                ("RTXPRO6000_BW_WS", "6000 WS · 96 GB"),
                ("RTXPRO6000_BSE", "6000 Server · 96 GB"),
                ("RTXPRO5000_BW_72", "5000 · 72 GB"),
                ("RTX6000_ADA", "6000 Ada · 48 GB"),
            ),
        ),
        (
            "GeForce",
            (
                ("RTX5090", "5090 · 32 GB"),
                ("RTX4090", "4090 · 24 GB"),
                ("RTX3090", "3090 · 24 GB"),
                ("RTX5060TI_16", "5060 Ti · 16 GB"),
                ("RTX4060TI_16", "4060 Ti · 16 GB"),
                ("RTX3060_12", "3060 · 12 GB"),
            ),
        ),
        ("L series", (("L4", "L4 · 24 GB"), ("L40S", "L40S · 48 GB"))),
        (
            "Blackwell",
            (
                ("B200", "B200 · 180 GB"),
                ("B300", "B300 · 288 GB"),
                ("GB200", "GB200 NVL72"),
                ("GB300", "GB300 NVL72"),
            ),
        ),
        ("Jetson", (("JETSON_AGX_THOR", "AGX Thor · 128 GB"),)),
        ("DGX", (("DGX_SPARK", "Spark · 128 GB"), ("DGX_STATION_GB300", "Station GB300"))),
        ("RTX A", (("A6000", "A6000 · 48 GB"), ("A4000", "A4000 · 16 GB"))),
        ("Ampere", (("A10", "A10 · 24 GB"), ("A30", "A30 · 24 GB"), ("A40", "A40 · 48 GB"))),
        ("V / T", (("V100", "V100 · 32 GB"), ("T4", "T4 · 16 GB"))),
    ),
    "AMD": (
        (
            "Instinct",
            tuple(
                (k, f"{k} · {GPUS[k].mem_gb:g} GB")
                for k in ("MI355X", "MI350X", "MI325X", "MI300X", "MI250X")
            ),
        ),
        (
            "Radeon PRO",
            (("RadeonAIProR9700", "AI R9700 · 32 GB"), ("RadeonProW7900", "W7900 · 48 GB")),
        ),
        (
            "Radeon RX",
            (
                ("RadeonRX9070XT", "9070 XT · 16 GB"),
                ("RadeonRX7900XTX", "7900 XTX · 24 GB"),
                ("RadeonRX7900XT", "7900 XT · 20 GB"),
            ),
        ),
        ("Ryzen AI", (("RYZEN_AI_MAX_395_128", "Max+ 395 · 96 GB GPU / 128 GB RAM"),)),
    ),
    "Intel": (
        ("Gaudi", (("Gaudi3", "3 · 128 GB"), ("Gaudi2", "2 · 96 GB"))),
        (
            "Arc Pro",
            tuple(
                (k, f"{k.removeprefix('ArcPro')} · {GPUS[k].mem_gb:g} GB")
                for k in ("ArcProB70", "ArcProB60", "ArcProB50")
            ),
        ),
    ),
    "Apple": (
        ("Mac Studio M5 Max", (("MAC_STUDIO_M5_MAX_64", "64 GB"), ("MAC_STUDIO_M5_MAX", "128 GB"))),
        ("Mac Studio M5 Ultra", (("MAC_STUDIO_M5_ULTRA_256", "256 GB"),)),
        ("Mac Studio M4 Max", (("MAC_STUDIO_M4_MAX_64", "64 GB"), ("MAC_STUDIO_M4_MAX", "128 GB"))),
        (
            "Mac Studio M3 Ultra",
            (("MAC_STUDIO_M3_ULTRA_256", "256 GB"), ("MAC_STUDIO_M3_ULTRA", "512 GB")),
        ),
        ("Mac mini M4 Pro", (("MAC_MINI_M4_PRO_48", "48 GB"), ("MAC_MINI_M4_PRO", "64 GB"))),
    ),
    "Tenstorrent": (
        (
            "Blackhole",
            (("TT_BLACKHOLE_P100A", "p100a · 28 GB"), ("TT_BLACKHOLE_P150", "p150 · 32 GB")),
        ),
        ("Galaxy", (("TT_GALAXY_BLACKHOLE", "Blackhole · 32 chips"),)),
    ),
    "FuriosaAI": (("RNGD", (("FURIOSA_RNGD", "48 GB"),)),),
}


def picker_vendors():
    choices = {choice.key: choice for choice in hardware_choices()}
    return {
        vendor: [
            (family, [(choices[key], label) for key, label in variants])
            for family, variants in families
        ]
        for vendor, families in PICKER_FAMILIES.items()
    }


def purchasable_gpu_keys() -> frozenset[str]:
    return frozenset(choice.key for choice in hardware_choices())
