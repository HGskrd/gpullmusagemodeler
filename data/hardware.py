"""GPU types, hardware catalog, researched TCO, and picker cards."""

import math
from dataclasses import dataclass

from .specs import VENDOR_LABELS


@dataclass
class GPU:
    key: str
    name: str
    vendor: str  # 'nv', 'amd', 'intel', or 'apple'
    mem: float  # bytes
    bw: float  # bytes/s, published peak memory bandwidth shown in the UI
    bf16: float  # FLOP/s
    fp8: float  # FLOP/s
    scale_up_p2p_bw_bidir: (
        float  # bytes/s, per-GPU aggregate bidirectional peer BW for node_size topology
    )
    node_size: int = 8
    planner_bw: float | None = (
        None  # bytes/s, optional sustained bandwidth proxy used by planner math
    )
    fp4: float | None = None  # FLOP/s for native dense FP4/MXFP4/NVFP4 tensor paths, when available
    tdp_watts: float = 0.0  # published board TDP — used with a utilization factor for CO2 math
    min_count: int = 1  # minimum pool size when this profile is only sold as a system/rack
    count_multiple: int = 1  # pool sizes snap to this multiple for system/rack-only profiles
    default_tco_per_gpu_hour: float = (
        0.0  # researched owned-hardware TCO, USD per committed GPU-hour
    )

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
    "RTXPRO6000_BSE": GPU(
        "RTXPRO6000_BSE",
        "RTX PRO 6000 Blackwell Server Edition 96GB",
        "nv",
        96e9,
        1.597e12,
        1e15,
        2e15,
        128e9,
        8,
    ),
    "RTXPRO6000_BW_WS": GPU(
        "RTXPRO6000_BW_WS",
        "RTX PRO 6000 Blackwell Workstation 96GB",
        "nv",
        96e9,
        1.792e12,
        1e15,
        2e15,
        128e9,
        4,
    ),
    "RTXPRO5000_BW_72": GPU(
        "RTXPRO5000_BW_72",
        "RTX PRO 5000 Blackwell 72GB",
        "nv",
        72e9,
        1.344e12,
        535.5e12,
        1071e12,
        128e9,
        4,
    ),
    "RTX6000_ADA": GPU(
        "RTX6000_ADA",
        "RTX 6000 Ada Generation 48GB",
        "nv",
        48e9,
        960e9,
        364.25e12,
        728.5e12,
        64e9,
        4,
    ),
    "RTX5090": GPU(
        "RTX5090", "GeForce RTX 5090 32GB", "nv", 32e9, 1.792e12, 838e12, 1676e12, 128e9, 1
    ),
    "RTX4090": GPU(
        "RTX4090", "GeForce RTX 4090 24GB", "nv", 24e9, 1.008e12, 330.25e12, 660.5e12, 64e9, 1
    ),
    "RTX3090": GPU("RTX3090", "GeForce RTX 3090 24GB", "nv", 24e9, 936e9, 142e12, 142e12, 64e9, 1),
    "DGX_SPARK": GPU(
        "DGX_SPARK", "DGX Spark GB10 128GB", "nv", 128e9, 273e9, 125e12, 250e12, 25e9, 1
    ),
    "GB200": GPU(
        "GB200",
        "GB200 NVL72 Grace Blackwell 186GB/GPU",
        "nv",
        186e9,
        8e12,
        2.5e15,
        5e15,
        3.6e12,
        72,
        min_count=72,
        count_multiple=72,
    ),
    "B200": GPU(
        "B200",
        "B200 180GB HGX/DGX",
        "nv",
        180e9,
        8e12,
        2.25e15,
        4.5e15,
        1.8e12,
        8,
        min_count=8,
        count_multiple=8,
    ),
    "B300": GPU(
        "B300",
        "B300 Blackwell Ultra 288GB HGX/DGX",
        "nv",
        288e9,
        8e12,
        2.5e15,
        5e15,
        1.8e12,
        8,
        min_count=8,
        count_multiple=8,
    ),
    "GB300": GPU(
        "GB300",
        "GB300 NVL72 Blackwell Ultra 288GB/GPU",
        "nv",
        288e9,
        8e12,
        2.5e15,
        5e15,
        3.6e12,
        72,
        min_count=72,
        count_multiple=72,
    ),
    "DGX_STATION_GB300": GPU(
        "DGX_STATION_GB300",
        "DGX Station GB300 Blackwell Ultra 252GB",
        "nv",
        252e9,
        7.1e12,
        2.5e15,
        5e15,
        100e9,
        2,
    ),
    # NVIDIA labels every Rubin figure below preliminary and subject to change.
    # The profile models one GPU inside the rack-only 72-GPU NVLink 6 domain.
    "RUBIN_NVL72": GPU(
        "RUBIN_NVL72",
        "Vera Rubin NVL72 Preview 288GB/GPU",
        "nv",
        288e9,
        22e12,
        4e15,
        17.5e15,
        3.6e12,
        72,
        min_count=72,
        count_multiple=72,
    ),
    "A40": GPU("A40", "A40 48GB", "nv", 48e9, 696e9, 149.7e12, 149.7e12, 112.5e9, 2),
    "A30": GPU("A30", "A30 24GB", "nv", 24e9, 933e9, 165e12, 165e12, 200e9, 2),
    "A6000": GPU("A6000", "RTX A6000 48GB", "nv", 48e9, 768e9, 154.85e12, 154.85e12, 112.5e9, 2),
    "A4000": GPU("A4000", "RTX A4000 16GB", "nv", 16e9, 448e9, 76.7e12, 76.7e12, 64e9, 1),
    "A2000_MOBILE": GPU(
        "A2000_MOBILE", "RTX A2000 Laptop GPU 8GB", "nv", 8e9, 192e9, 37.5e12, 37.5e12, 64e9, 1
    ),
    "T4": GPU("T4", "T4 16GB", "nv", 16e9, 320e9, 65e12, 65e12, 32e9, 8),
    "V100": GPU("V100", "V100 32GB SXM2", "nv", 32e9, 900e9, 125e12, 125e12, 300e9, 8),
    "JETSON_AGX_THOR": GPU(
        "JETSON_AGX_THOR", "Jetson AGX Thor 128GB", "nv", 128e9, 273e9, 258.75e12, 517.5e12, 25e9, 1
    ),
    "MI250X": GPU("MI250X", "MI250X 128GB", "amd", 128e9, 3.2e12, 383e12, 383e12, 800e9, 8),
    "MI300X": GPU("MI300X", "MI300X 192GB", "amd", 192e9, 5.3e12, 1307e12, 2615e12, 896e9, 8),
    "MI325X": GPU("MI325X", "MI325X 256GB", "amd", 256e9, 6e12, 1307e12, 2615e12, 896e9, 8),
    "MI350X": GPU("MI350X", "MI350X 288GB", "amd", 288e9, 8e12, 2010e12, 4020e12, 1075.2e9, 8),
    "MI355X": GPU("MI355X", "MI355X 288GB", "amd", 288e9, 8e12, 2512e12, 5037e12, 1075.2e9, 8),
    # AMD now identifies the Helios accelerator as MI455X.  Keep MI400 below as
    # a hidden compatibility key so previously saved planner states still load.
    "MI455X": GPU(
        "MI455X", "Instinct MI455X 432GB", "amd", 432e9, 23.3e12, 5e15, 20.1e15, 3.6e12, 72
    ),
    "HELIOS_MI455X": GPU(
        "HELIOS_MI455X",
        "AMD Helios Preview (72× MI455X) 432GB/GPU",
        "amd",
        432e9,
        23.3e12,
        5e15,
        20.1e15,
        3.6e12,
        72,
        min_count=72,
        count_multiple=72,
    ),
    "MI400": GPU(
        "MI400",
        "MI400 Series compatibility profile (MI455X) 432GB",
        "amd",
        432e9,
        23.3e12,
        5e15,
        20.1e15,
        3.6e12,
        72,
    ),
    "RadeonProW7900": GPU(
        "RadeonProW7900", "Radeon PRO W7900 48GB", "amd", 48e9, 864e9, 123e12, 123e12, 64e9, 1
    ),
    "RadeonAIProR9700": GPU(
        "RadeonAIProR9700", "Radeon AI PRO R9700 32GB", "amd", 32e9, 640e9, 96e12, 96e12, 128e9, 1
    ),
    # Tenstorrent publishes BLOCKFP8, rather than IEEE FP8, peak performance.
    # BF16 is a conservative half-rate planner proxy until a native BF16 peak is published.
    "TT_BLACKHOLE_P100A": GPU(
        "TT_BLACKHOLE_P100A",
        "Tenstorrent Blackhole p100a 28GB",
        "tenstorrent",
        28e9,
        448e9,
        332e12,
        664e12,
        64e9,
        1,
    ),
    "TT_BLACKHOLE_P150": GPU(
        "TT_BLACKHOLE_P150",
        "Tenstorrent Blackhole p150 32GB",
        "tenstorrent",
        32e9,
        512e9,
        332e12,
        664e12,
        800e9,
        8,
    ),
    "TT_GALAXY_BLACKHOLE": GPU(
        "TT_GALAXY_BLACKHOLE",
        "Tenstorrent Galaxy Blackhole (32×) 32GB/ASIC",
        "tenstorrent",
        32e9,
        512e9,
        359.375e12,
        718.75e12,
        1e12,
        32,
        min_count=32,
        count_multiple=32,
    ),
    "FURIOSA_RNGD": GPU(
        "FURIOSA_RNGD", "FuriosaAI RNGD 48GB", "furiosa", 48e9, 1.5e12, 256e12, 512e12, 64e9, 1
    ),
    # Intel does not publish dense BF16/FP8 peak figures for these public pages, so the planner
    # uses transparent proxy rooflines derived from the nearest available official disclosures.
    "Gaudi2": GPU("Gaudi2", "Gaudi 2 96GB", "intel", 96e9, 2.45e12, 432e12, 865e12, 300e9, 8),
    "Gaudi3": GPU("Gaudi3", "Gaudi 3 128GB", "intel", 128e9, 3.7e12, 1.3e15, 2.6e15, 900e9, 4),
    "CrescentIsland": GPU(
        "CrescentIsland",
        "Crescent Island Preview 160GB",
        "intel",
        160e9,
        1.0e12,
        183.5e12,
        367e12,
        128e9,
        8,
    ),
    "ArcProB70": GPU(
        "ArcProB70", "Arc Pro B70 32GB", "intel", 32e9, 608e9, 183.5e12, 367e12, 128e9, 8
    ),
    "ArcProB60": GPU(
        "ArcProB60", "Arc Pro B60 24GB", "intel", 24e9, 456e9, 98.5e12, 197e12, 64e9, 8
    ),
    "ArcProB50": GPU("ArcProB50", "Arc Pro B50 16GB", "intel", 16e9, 224e9, 85e12, 170e12, 64e9, 8),
    # Apple publishes peak unified-memory bandwidth and GPU-core counts, but not dense BF16/FP8
    # tensor rooflines or sustained inference bandwidth. We therefore keep Apple's peak bandwidth
    # for display, while planner_bw + BF16/FP8 proxies are calibrated conservatively against
    # whatcani.run Apple-device decode/prefill scaling across MLX/GGUF runs.
    "MAC_MINI_M4_PRO": GPU(
        "MAC_MINI_M4_PRO",
        "Mac mini M4 Pro 64GB",
        "apple",
        64e9,
        273e9,
        16e12,
        16e12,
        50e9,
        1,
        273e9,
    ),
    "MAC_STUDIO_M4_MAX": GPU(
        "MAC_STUDIO_M4_MAX",
        "Mac Studio M4 Max 128GB",
        "apple",
        128e9,
        546e9,
        26e12,
        26e12,
        50e9,
        1,
        410e9,
    ),
    "MAC_STUDIO_M3_ULTRA": GPU(
        "MAC_STUDIO_M3_ULTRA",
        "Mac Studio M3 Ultra 512GB",
        "apple",
        512e9,
        819e9,
        48e12,
        48e12,
        50e9,
        1,
        560e9,
    ),
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
    "RUBIN_NVL72": 50e15,
    "JETSON_AGX_THOR": 1035e12,
    "MI350X": 9.2e15,
    "MI355X": 10.1e15,
    "MI455X": 40.3e15,
    "HELIOS_MI455X": 40.3e15,
    "MI400": 40.3e15,
}
for _k, _fp4 in GPU_FP4_FLOPS.items():
    if _k in GPUS:
        GPUS[_k].fp4 = float(_fp4)

# Published board TDPs (watts). Used with a utilization factor to estimate per-task energy.
# Sources: vendor product pages; Mac figures use whole-system measured peak.
GPU_TDP_WATTS = {
    "A100": 400,
    "A100_40": 250,
    "A10": 150,
    "H100": 700,
    "H200": 700,
    "L40S": 350,
    "L4": 72,
    "RTXPRO6000_BSE": 600,
    "RTXPRO6000_BW_WS": 600,
    "RTXPRO5000_BW_72": 300,
    "RTX6000_ADA": 300,
    "RTX5090": 575,
    "RTX4090": 450,
    "RTX3090": 350,
    "DGX_SPARK": 140,
    "GB200": 1200,
    "B200": 1000,
    "B300": 1400,
    "GB300": 1400,
    "DGX_STATION_GB300": 1600,
    "A40": 300,
    "A30": 165,
    "A6000": 300,
    "A4000": 140,
    "A2000_MOBILE": 95,
    "T4": 70,
    "V100": 300,
    "JETSON_AGX_THOR": 130,
    "MI250X": 560,
    "MI300X": 750,
    "MI325X": 1000,
    "MI350X": 1000,
    "MI355X": 1400,
    "MI455X": 1500,
    "HELIOS_MI455X": 1500,
    "MI400": 1500,
    "RadeonProW7900": 295,
    "RadeonAIProR9700": 300,
    "TT_BLACKHOLE_P100A": 300,
    "TT_BLACKHOLE_P150": 300,
    "TT_GALAXY_BLACKHOLE": 350,
    "FURIOSA_RNGD": 180,
    "Gaudi2": 600,
    "Gaudi3": 900,
    "CrescentIsland": 300,
    "ArcProB70": 230,
    "ArcProB60": 200,
    "ArcProB50": 70,
    "MAC_MINI_M4_PRO": 140,
    "MAC_STUDIO_M4_MAX": 270,
    "MAC_STUDIO_M3_ULTRA": 480,
}
for _k, _w in GPU_TDP_WATTS.items():
    if _k in GPUS:
        GPUS[_k].tdp_watts = float(_w)


# Owned-hardware TCO defaults.  Acquisition prices are USD per GPU (or per
# accelerator inside a complete system) researched from current vendor, retailer,
# and market-tracker listings.  Board-only prices receive a 15% host/network/power
# infrastructure allocation; complete systems and racks already include it.
# Amortization is deliberately conservative at four years: hyperscalers use longer
# schedules, while AI-native operators commonly use four-to-five-year cycles.
GPU_TCO_PRICING_CAPTURED_AT = "2026-07-21"
GPU_TCO_USEFUL_LIFE_HOURS = 4 * 365 * 24
GPU_TCO_BOARD_INFRA_UPLIFT = 1.15
GPU_TCO_FACILITY_OPS_UPLIFT = 1.10
GPU_TCO_AVG_POWER_UTILIZATION = 0.80
GPU_TCO_PUE = 1.50
GPU_TCO_ELECTRICITY_USD_PER_KWH = 0.20

GPU_TCO_PRICE_USD: dict[str, float] = {
    # NVIDIA datacenter: current new/refurbished market or reported system pricing.
    "A100": 15_000.0,
    "A100_40": 8_000.0,
    "A10": 2_500.0,
    "H100": 32_000.0,
    "H200": 38_000.0,
    "L40S": 8_600.0,
    "L4": 2_800.0,
    "GB200": 3_100_000.0 / 72.0,
    "B200": 45_000.0,
    "B300": 50_000.0,  # proxy: GB300 rack premium over GB200
    "GB300": 3_900_000.0 / 72.0,
    "DGX_STATION_GB300": 110_000.0,  # channel-reported midpoint; no public list price
    "RUBIN_NVL72": 7_800_000.0 / 72.0,  # Morgan Stanley rack estimate; preview
    "A40": 4_500.0,
    "A30": 4_600.0,
    "T4": 542.0,  # current used-market typical price, not launch MSRP
    "V100": 699.0,  # current used-market typical price, not launch MSRP
    # NVIDIA workstation, desktop, and edge.
    "RTXPRO6000_BSE": 13_250.0,  # proxy from official workstation-edition marketplace price
    "RTXPRO6000_BW_WS": 13_250.0,
    "RTXPRO5000_BW_72": 8_700.0,
    "RTX6000_ADA": 7_237.0,
    "RTX5090": 3_766.0,
    "RTX4090": 2_264.0,
    "RTX3090": 1_144.0,
    "DGX_SPARK": 3_999.0,
    "A6000": 3_500.0,
    "A4000": 900.0,
    "A2000_MOBILE": 600.0,  # proxy: no standalone mobile-GPU price
    "JETSON_AGX_THOR": 3_499.0,
    # AMD.  MI455X uses the reported Helios rack midpoint divided by 72 GPUs.
    "MI250X": 1_732.0,  # current refurbished OAM market; EOL part
    "MI300X": 12_500.0,
    "MI325X": 15_000.0,  # proxy from reported MI300X premium
    "MI350X": 25_000.0,
    "MI355X": 27_500.0,  # proxy: liquid-cooled MI350X variant
    "MI455X": 5_250_000.0 / 72.0,
    "HELIOS_MI455X": 5_250_000.0 / 72.0,
    "MI400": 5_250_000.0 / 72.0,
    "RadeonProW7900": 3_999.0,
    "RadeonAIProR9700": 1_299.0,
    # Specialist accelerators, Intel, and Apple.
    "TT_BLACKHOLE_P100A": 999.0,
    "TT_BLACKHOLE_P150": 1_399.0,
    "TT_GALAXY_BLACKHOLE": 110_000.0 / 32.0,
    "FURIOSA_RNGD": 7_000.0,  # proxy: no public card price; L40S-class
    "Gaudi2": 65_000.0 / 8.0,
    "Gaudi3": 125_000.0 / 8.0,
    "CrescentIsland": 8_500.0,  # proxy: preview card, RTX PRO 6000 class
    "ArcProB70": 949.0,
    "ArcProB60": 599.0,
    "ArcProB50": 299.0,
    "MAC_MINI_M4_PRO": 2_199.0,
    "MAC_STUDIO_M4_MAX": 3_499.0,
    "MAC_STUDIO_M3_ULTRA": 9_499.0,
}

GPU_TCO_COMPLETE_SYSTEMS = frozenset(
    {
        "DGX_SPARK",
        "GB200",
        "GB300",
        "DGX_STATION_GB300",
        "RUBIN_NVL72",
        "JETSON_AGX_THOR",
        "MI455X",
        "HELIOS_MI455X",
        "MI400",
        "TT_GALAXY_BLACKHOLE",
        "MAC_MINI_M4_PRO",
        "MAC_STUDIO_M4_MAX",
        "MAC_STUDIO_M3_ULTRA",
    }
)

# Rubin board power is intentionally unpublished in the performance catalog.  The
# TCO default still needs an energy term, so use a conservative 2 kW per-GPU rack
# proxy without turning that estimate into a published TDP claim.
GPU_TCO_POWER_WATTS_OVERRIDE = {"RUBIN_NVL72": 2_000.0}

GPU_TCO_PRICE_SOURCES = {
    "NVIDIA current market": "https://gpupoet.com/gpu/price-compare",
    "NVIDIA datacenter pricing guide": "https://intuitionlabs.ai/articles/nvidia-ai-gpu-pricing-guide",
    "NVIDIA Blackwell rack pricing": "https://www.spheron.network/blog/gb300-nvl72-vs-gb200-nvl72-pricing-availability-2026/",
    "NVIDIA Rubin rack estimate": "https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-memory-costs-soar-485-percent-latest-ai-systems-now-cost-usd7-8-million-to-build-memory-now-comprises-25-percent-of-the-total-cost-rubin-gpus-a-mere-usd50-000-apiece",
    "NVIDIA DGX Station channel pricing": "https://www.servethehome.com/nvidia-dgx-station-systems-available-at-last-gb300-gb200-workstations-for-your-desktop/",
    "NVIDIA RTX PRO official marketplace": "https://marketplace.nvidia.com/en-us/enterprise/laptops-workstations/nvidia-rtx-pro-6000-blackwell-workstation-edition/",
    "NVIDIA DGX Spark official marketplace": "https://marketplace.nvidia.com/en-us/enterprise/personal-ai-supercomputers/dgx-spark/",
    "NVIDIA Jetson Thor official marketplace": "https://marketplace.nvidia.com/en-us/enterprise/robotics-edge/jetson-thor-developer-kit/",
    "AMD MI250X current refurbished market": "https://harddiskdirect.com/p41933-001-1722798.html",
    "AMD Helios estimate": "https://www.trendforce.com/news/2026/07/21/news-amds-first-rack-scale-ai-system-helios-challenges-nvidia-with-hbm4-memory-edge-but-reportedly-comes-at-a-higher-price/",
    "AMD MI300X estimate": "https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-h100-ai-gpus-cost-up-to-four-times-more-than-amds-competing-mi300x-amds-chips-cost-dollar10-to-dollar15k-apiece-nvidias-h100-has-peaked-beyond-dollar40000",
    "AMD MI350X reported price": "https://www.investors.com/news/technology/amd-stock-ai-chip-price-increase-nvidia/",
    "AMD MI325X pricing context": "https://deploybase.ai/articles/amd-mi325x-price",
    "AMD Radeon PRO W7900 launch pricing": "https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html",
    "AMD Radeon AI PRO R9700 MSRP": "https://www.techpowerup.com/342203/amd-radeon-ai-pro-r9700-gpu-arrives-october-27-at-usd-1-299-for-retail",
    "Tenstorrent official card pricing": "https://tenstorrent.com/en/hardware/cards",
    "Tenstorrent Galaxy official pricing": "https://tenstorrent.com/en/hardware/galaxy",
    "FuriosaAI RNGD availability": "https://furiosa.ai/blog/rngd-enters-mass-production-the-high-performance-ai-accelerator-for-any-data-center",
    "Intel Gaudi official UBB pricing": "https://www.servethehome.com/intel-gaudi-2-8x-oam-ubb-65k-gaudi-3-125k-and-includes-networking/",
    "Intel Crescent Island preview": "https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu",
    "Intel Arc Pro B50 MSRP": "https://www.tomshardware.com/pc-components/gpus/intel-launches-usd299-arc-pro-b50-with-16gb-of-memory-project-battlematrix-workstations-with-24gb-arc-pro-b60-gpus",
    "Intel Arc Pro B60 retail pricing": "https://videocardz.com/newz/intel-arc-pro-b60-24gb-professional-gpu-listed-at-599-in-stock-and-shipping",
    "Intel Arc Pro B70 launch pricing": "https://www.phoronix.com/news/Intel-Arc-Pro-B70-Announced",
    "Apple official Mac pricing": "https://www.apple.com/shop/buy-mac",
    "Electricity price assumption": "https://ec.europa.eu/eurostat/en/web/products-eurostat-news/w/ddn-20260508-2",
    "PUE assumption context": "https://datacenter.uptimeinstitute.com/rs/711-RIA-145/images/2025.Annual.Survey.Report.pdf?version=0",
    "Useful-life assumption context": "https://siliconangle.com/2025/11/22/resetting-gpu-depreciation-ai-factories-bend-dont-break-useful-life-assumptions/",
}


def _researched_gpu_tco_per_hour(gpu_key: str) -> float:
    price = GPU_TCO_PRICE_USD[gpu_key]
    allocated_capex = (
        price if gpu_key in GPU_TCO_COMPLETE_SYSTEMS else price * GPU_TCO_BOARD_INFRA_UPLIFT
    )
    amortized = allocated_capex / GPU_TCO_USEFUL_LIFE_HOURS * GPU_TCO_FACILITY_OPS_UPLIFT
    power_watts = GPU_TCO_POWER_WATTS_OVERRIDE.get(gpu_key, GPUS[gpu_key].tdp_watts)
    energy = (
        power_watts
        / 1000.0
        * GPU_TCO_AVG_POWER_UTILIZATION
        * GPU_TCO_PUE
        * GPU_TCO_ELECTRICITY_USD_PER_KWH
    )
    return round(amortized + energy, 2)


GPU_TCO_DEFAULTS: dict[str, float] = {_key: _researched_gpu_tco_per_hour(_key) for _key in GPUS}
for _key, _tco in GPU_TCO_DEFAULTS.items():
    GPUS[_key].default_tco_per_gpu_hour = _tco


# Announced/preview catalog entries must keep their uncertainty reviewable.
# Kimi K3 is now an open-weight release with a pinned config and technical report.
PREVIEW_ASSUMPTIONS_CAPTURED_AT = "2026-08-09"
PREVIEW_ASSUMPTIONS: dict[str, dict[str, object]] = {
    "gpu:RUBIN_NVL72": {
        "status": "full-production ramp; production shipments announced for fall 2026",
        "source": "https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/",
        "assumptions": (
            "all per-GPU performance, memory, bandwidth, and NVLink figures remain preliminary",
            "rack-only pool sizes are constrained to multiples of 72",
            "board power is omitted until NVIDIA publishes a per-GPU figure",
        ),
    },
    "gpu:MI455X": {
        "status": "launched 2026-07-23; Helios volume deployments expected in 2H 2026",
        "source": "https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html",
        "assumptions": (
            "AMD publishes 432 GB HBM4, 23.3 TB/s, 5.0 PF dense BF16, 20.1 PF FP8, 40.3 PF MXFP4, and 3.6 TB/s bidirectional scale-up per GPU",
            "MI455X is the named accelerator in AMD's 72-GPU Helios reference design",
            "1.5 kW remains a planner power proxy because AMD does not publish accelerator TBP",
        ),
    },
    "gpu:HELIOS_MI455X": {
        "status": "AMD reference design; OEM volume deployments expected in 2H 2026",
        "source": "https://www.amd.com/en/products/rackscale-solutions/helios.html",
        "assumptions": (
            "the selectable system represents the complete 72-GPU Helios rack with Venice CPUs and Vulcano networking",
            "AMD publishes 432 GB HBM4, 23.3 TB/s, 5.0 PF dense BF16, 20.1 PF FP8, 40.3 PF MXFP4, and 3.6 TB/s bidirectional scale-up per GPU",
            "AMD publishes 31 TB HBM4, 2.9 EF FP4, 1.4 EF FP8, 260 TB/s scale-up, and 43 TB/s scale-out at rack level; 1.5 kW/GPU remains a planner proxy",
        ),
    },
    "gpu:TT_GALAXY_BLACKHOLE": {
        "status": "commercial 32-ASIC, 6U air-cooled server",
        "source": "https://tenstorrent.com/hardware/galaxy",
        "assumptions": (
            "BLOCKFP8 peak is used for the planner FP8 path; it is not an IEEE FP8 equivalence claim",
            "BF16 uses a conservative half-rate proxy because Tenstorrent does not publish a BF16 peak",
            "per-ASIC fabric is derived from the published ten 400GbE links per ASIC",
        ),
    },
    "gpu:MI400": {
        "status": "legacy compatibility key for the MI455X Helios profile",
        "source": "https://www.amd.com/en/products/rackscale-solutions/helios.html",
        "assumptions": (
            "existing saved plans using MI400 retain the MI455X hardware assumptions",
            "AMD publishes 432 GB HBM4, 23.3 TB/s, 5.0 PF dense BF16, 20.1 PF FP8, 40.3 PF MXFP4, and 3.6 TB/s bidirectional scale-up per GPU",
            "1.5 kW remains a planner power proxy rather than a published MI455X TBP",
        ),
    },
    "gpu:CrescentIsland": {
        "status": "announced inference GPU; public launch configuration pending",
        "source": "https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu",
        "assumptions": (
            "160 GB LPDDR5X is public; bandwidth, tensor rooflines, topology, and power are planner proxies",
        ),
    },
    "model:inkling-small-preview": {
        "status": "preview weights and architecture configuration not released",
        "source": "https://thinkingmachines.ai/news/introducing-inkling/",
        "assumptions": (
            "48-layer layout scales the released Inkling local/global attention pattern",
            "276B total and 12B active parameters are public preview facts",
        ),
    },
    "model:mistral-medium-3.5-preview": {
        "status": "API and model card live; full architecture configuration not public",
        "source": "https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04",
        "assumptions": ("128B dense architecture fields remain a capacity proxy",),
    },
    "model:deepseek-v4-pro": {
        "status": "preview; final open checkpoint and architecture configuration not published",
        "source": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731",
        "assumptions": (
            "1.6T total, 49B active, layer, attention, and MTP fields remain planning proxies",
            "the preview comparison advertises a 1M context window but does not publish the Pro config",
        ),
    },
    "cloud:deepseek-v4-pro": {
        "status": "preview API entry; lifecycle and final API identity may change",
        "source": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731",
        "assumptions": (
            "the final Flash card still identifies V4 Pro as Preview",
            "catalog pricing is retained until DeepSeek publishes a dated hosted-API mapping",
        ),
    },
    "cloud:gemini-pro": {
        "status": "Gemini 3.1 Pro preview; production lifecycle may change with short notice",
        "source": "https://ai.google.dev/gemini-api/docs/models",
        "assumptions": (
            "planner uses standard pricing for prompts at or below 200k tokens",
            "requests above 200k use the explicit long-context input, cached-input, and output rates",
        ),
    },
}
for _preview in PREVIEW_ASSUMPTIONS.values():
    _preview["captured_at"] = PREVIEW_ASSUMPTIONS_CAPTURED_AT


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
        "Vera Rubin NVL72",
        "NVIDIA",
        "Rubin",
        "72-GPU rack: 288 GB HBM4/GPU (preliminary)",
        "Next-generation rack-scale agentic inference and training; production shipments announced for fall 2026",
        (GPUPlannerOption("Add 72-GPU Preview", "RUBIN_NVL72"),),
        "NVIDIA says the platform is in full-production ramp, but still labels the 288GB/22TB/s/50PF NVFP4 per-GPU specifications preliminary and subject to change. Pool sizes snap to multiples of 72.",
    ),
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
        "AMD Helios (72× MI455X + Venice)",
        "AMD",
        "CDNA 5 rackscale reference design (preliminary)",
        "31 TB HBM4 total · 432 GB / GPU",
        "72-GPU ORW rack with EPYC Venice, UALink and Pensando Vulcano networking; OEM volume deployments expected in 2H 2026",
        (GPUPlannerOption("Add Preview", "HELIOS_MI455X"),),
        "System-only profile: 2.9 EF FP4, 1.4 EF FP8, 260 TB/s scale-up, and 43 TB/s scale-out. The count is constrained to one or more 72-GPU racks.",
    ),
    GPUCard(
        "Instinct MI455X",
        "AMD",
        "CDNA 5",
        "432 GB HBM4, 23.3 TB/s",
        "Helios volume deployments expected in 2H 2026",
        (GPUPlannerOption("Add", "MI455X"),),
        "AMD publishes dense BF16, FP8, MXFP4, memory, bandwidth, and scale-up specifications. Power remains a planner proxy; the MI400 key is retained only for saved-plan compatibility.",
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
    GPUCard(
        "MI440X",
        "AMD",
        "MI400 series (announced)",
        "8-GPU enterprise system; memory specification pending",
        "Enterprise AI training, fine-tuning, and inference",
        (),
        "Reference-only: AMD has announced the compact 8-GPU system but has not published planner-grade memory, bandwidth, precision-throughput, or power specifications.",
    ),
    # ── Specialist inference accelerators ──────────────────────────────────
    GPUCard(
        "Tenstorrent Galaxy Blackhole",
        "Tenstorrent",
        "Blackhole Tensix",
        "32 ASICs · 1 TB GDDR6 · 16 TB/s",
        "6U air-cooled server for scalable LLM inference and training",
        (GPUPlannerOption("Add", "TT_GALAXY_BLACKHOLE"),),
        "32-ASIC system profile constrained to full Galaxy servers. Uses published 23 PFLOPS BLOCKFP8 system throughput; BF16 is a conservative proxy.",
    ),
    GPUCard(
        "Tenstorrent Blackhole p150",
        "Tenstorrent",
        "Blackhole Tensix",
        "32 GB GDDR6 · 512 GB/s",
        "Single- or multi-card workstation/server inference",
        (GPUPlannerOption("Add", "TT_BLACKHOLE_P150"),),
        "300W card with four 800Gbps passive QSFP-DD ports for direct Blackhole connections. Published peak is BLOCKFP8; BF16 is a conservative proxy.",
    ),
    GPUCard(
        "Tenstorrent Blackhole p100a",
        "Tenstorrent",
        "Blackhole Tensix",
        "28 GB GDDR6 · 448 GB/s",
        "Single-card workstation evaluation and local inference",
        (GPUPlannerOption("Add", "TT_BLACKHOLE_P100A"),),
        "300W actively cooled PCIe Gen5 card. It lacks the p150's direct QSFP-DD fabric.",
    ),
    GPUCard(
        "FuriosaAI RNGD",
        "FuriosaAI",
        "Tensor Contraction Processor",
        "48 GB HBM3 · 1.5 TB/s",
        "Efficient LLM and multimodal inference; commercially deployed PCIe accelerator",
        (GPUPlannerOption("Add", "FURIOSA_RNGD"),),
        "PCIe Gen5 x16, passive dual-slot 180W card. No native FP4 peak is claimed; all listed tensor rooflines are vendor-published.",
    ),
    GPUCard(
        "VSORA Jotunn 8",
        "VSORA",
        "Data-center inference accelerator (announced)",
        "Memory and throughput specifications pending",
        "LLM inference accelerator",
        (),
        "Reference-only: VSORA identifies Jotunn 8 as a data-center inference chip but does not publish the memory, bandwidth, precision, power, or host-interface figures required by this planner.",
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
