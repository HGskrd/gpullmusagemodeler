"""tinygrad.org configurator snapshot; prices are complete-system USD, before tax.

Reviewed 2026-11-05 against the homepage and assets/tinybox.js?v=red-whale-2.
Keep option IDs stable: exported scenarios retain configuration-specific GPU keys.
"""

from dataclasses import asdict, dataclass
from itertools import product

TINYBOX_REVIEWED_AT = "2026-11-11"
TINYBOX_SOURCE = "https://tinygrad.org/"
TINYBOX_CONFIG_SOURCE = "https://tinygrad.org/assets/tinybox.js?v=red-whale-2"
TINYBOX_OPTIONS = {
    "power": (("1700", "1700 W · standard"), ("3400", "3400 W · tall")),
    "platform": (("G4", "G4 · EPYC · PCIe 4 · DDR4"), ("G5", "G5 · GENOA · PCIe 5 · DDR5")),
    "memory": (("32", "32 GB"), ("128", "128 GB · G4"), ("192", "192 GB · G5")),
    "storage": (
        ("BOOT", "1 TB boot SSD only"),
        ("RAID4", "4 TB RAID + boot SSD"),
        ("RAID16", "16 TB RAID + boot SSD"),
        ("RAID4X4", "4× 4 TB RAID · G5"),
    ),
    "gpu": (
        ("NONE", "None · bring your own GPUs"),
        ("4X9070XT", "4× AMD 9070 XT · 64 GB"),
        ("4XR9700", "4× AMD R9700 · 128 GB"),
        ("6XR9700", "6× AMD R9700 · 192 GB"),
        ("6X9070XT", "6× AMD 9070 XT · 96 GB · tall"),
        ("14XR9700", "14× AMD R9700 · 448 GB · tall"),
    ),
}
# The source selects its first six-GPU option for red flash: R9700, not 9070 XT.
_PRESETS = {
    ("1700", "G4", "32", "BOOT", "6XR9700"): ("TINYBOX_RED_FLASH", "tinybox red flash"),
    ("1700", "G5", "32", "RAID4X4", "4XR9700"): ("TINYBOX_RED_KIMI", "tinybox red kimi"),
    ("3400", "G5", "32", "RAID4", "14XR9700"): ("TINYBOX_RED_WHALE", "tinybox red whale"),
}


@dataclass(frozen=True)
class TinyboxConfig:
    key: str
    name: str
    power: str
    platform: str
    memory: str
    storage: str
    gpu: str
    count: int
    price: int

    @property
    def base_gpu(self) -> str:
        return "RadeonRX9070XT" if "9070XT" in self.gpu else "RadeonAIProR9700"

    @property
    def board_watts(self) -> int:
        if self.count == 14:
            return 210
        if self.gpu == "6XR9700":
            return 210
        return 304 if "9070XT" in self.gpu else 300

    @property
    def details(self) -> str:
        storage = dict(TINYBOX_OPTIONS["storage"])[self.storage]
        lanes = "x8" if self.count == 14 else "x16"
        network = "2× 1GbE" if self.platform == "G4" else "2× 10GbE"
        ocp = "" if self.storage == "RAID4X4" or self.count == 14 else " + OCP3.0"
        usb = (
            " USB boot."
            if self.storage == "RAID4X4" or (self.storage != "BOOT" and self.count >= 6)
            else ""
        )
        return (
            f"{self.platform} · 32-core CPU · {self.memory} GB host RAM · {storage} · "
            f"{self.power} W PSU capacity · {network}{ocp}. PCIe {lanes} per GPU.{usb} "
            f"${self.price:,} per complete system before shipping/tax; {self.count} GPUs. "
            f"{self.board_watts} W/GPU power limit. "
            "Per-GPU VRAM is separate; aggregate bandwidth is not shared memory. "
            "Compute is an uncalibrated power-scaled catalog proxy; PCIe peer bandwidth "
            "is a conservative link proxy, not a measured collective. Host RAM/storage "
            "do not increase GPU capacity; CPU/SSD offload is not simulated. "
            "tinygrad / ROCm model and precision support requires validation. "
            "TCO energy uses a PSU-capacity-per-GPU planning proxy, not measured wall power; "
            "task energy uses the GPU power limit and excludes host overhead."
        )


def _configurations() -> tuple[TinyboxConfig, ...]:
    configs = []
    for values in product(
        *(tuple(key for key, _ in options) for options in TINYBOX_OPTIONS.values())
    ):
        power, platform, memory, storage, gpu = values
        count = 0 if gpu == "NONE" else int(gpu.split("X", 1)[0])
        if memory not in ("32", "128" if platform == "G4" else "192"):
            continue
        if storage == "RAID4X4" and (platform != "G5" or count >= 6):
            continue
        if gpu in {"6X9070XT", "14XR9700"} and power != "3400":
            continue
        price = (
            (7000 if platform == "G4" else 12000)
            + (1000 if power == "3400" else 0)
            + {"32": 0, "128": 3000, "192": 8000}[memory]
            + {"BOOT": 0, "RAID4": 2000, "RAID16": 6000, "RAID4X4": 8000}[storage]
            + count * (1000 if "9070XT" in gpu else 2500)
        )
        key, name = _PRESETS.get(
            (power, platform, memory, storage, gpu), ("TINYBOX_" + "_".join(values), "")
        )
        if not name:
            name = f"tinybox {platform} · {count}× {gpu.split('X', 1)[-1]} · {memory}GB · {storage} · {power}W"
        configs.append(
            TinyboxConfig(key, name, power, platform, memory, storage, gpu, count, price)
        )
    return tuple(configs)


TINYBOX_CONFIGS = _configurations()
TINYBOX_GPU_CONFIGS = {config.key: config for config in TINYBOX_CONFIGS if config.count}
TINYBOX_PRESET_KEYS = tuple(key for key, _ in _PRESETS.values())


def tinybox_picker_configs() -> list[dict]:
    return [dict(asdict(config), details=config.details) for config in TINYBOX_CONFIGS]
