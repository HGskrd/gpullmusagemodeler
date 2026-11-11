"""Complete-system pricing, physical topology and tinygrad configurator constraints."""

import pytest

from data import GPU_TCO_COMPLETE_SYSTEMS, GPU_TCO_PRICE_USD, GPUS
from data.tinybox import TINYBOX_CONFIGS, TINYBOX_GPU_CONFIGS
from scenarios import deserialize_scenario, serialize_scenario
from state import PlannerState, add_gpu


@pytest.mark.parametrize(
    "key,price,count,mem,watts,peer_bw",
    [
        ("TINYBOX_RED_FLASH", 22000, 6, 32e9, 210, 32e9),
        ("TINYBOX_RED_KIMI", 30000, 4, 32e9, 300, 64e9),
        ("TINYBOX_RED_WHALE", 50000, 14, 32e9, 210, 32e9),
    ],
)
def test_presets_keep_per_gpu_geometry_and_complete_system_cost(
    key, price, count, mem, watts, peer_bw
):
    config = TINYBOX_GPU_CONFIGS[key]
    gpu = GPUS[key]
    assert config.price == price
    assert gpu.key == key
    assert gpu.mem == mem
    assert gpu.bw == 640e9
    assert (gpu.node_size, gpu.min_count, gpu.count_multiple) == (count, count, count)
    assert gpu.tdp_watts == watts
    assert gpu.scale_up_p2p_bw_bidir == peer_bw
    assert GPU_TCO_PRICE_USD[key] * count == pytest.approx(price)
    assert key in GPU_TCO_COMPLETE_SYSTEMS
    assert gpu.bf16 == pytest.approx(GPUS[config.base_gpu].bf16 * watts / 300)


def test_all_offered_options_and_only_compatible_combinations_are_represented():
    assert len(TINYBOX_CONFIGS) == 132
    assert len(TINYBOX_GPU_CONFIGS) == 104
    assert len({config.key for config in TINYBOX_CONFIGS}) == len(TINYBOX_CONFIGS)
    for config in TINYBOX_CONFIGS:
        assert config.memory in ("32", "128" if config.platform == "G4" else "192")
        if config.storage == "RAID4X4":
            assert config.platform == "G5"
            assert config.count < 6
        if config.gpu in {"6X9070XT", "14XR9700"}:
            assert config.power == "3400"
        if not config.count:
            assert config.key not in GPUS  # Barebones is not zero-cost GPU capacity.
    base = next(c for c in TINYBOX_CONFIGS if c.key == "TINYBOX_1700_G4_32_BOOT_NONE")
    assert base.price == 7000
    upgraded = TINYBOX_GPU_CONFIGS["TINYBOX_3400_G5_192_RAID16_6X9070XT"]
    assert upgraded.price == 33000
    assert GPUS[upgraded.key].mem == 16e9
    assert GPUS[upgraded.key].system_memory_gb == 192


def test_system_purchase_boundaries_survive_scenario_roundtrip():
    state = PlannerState()
    add_gpu(state, "TINYBOX_RED_WHALE", 1)
    assert state.gpus[0].count == 14
    add_gpu(state, "TINYBOX_RED_WHALE", 14)
    assert state.gpus[0].count == 28
    restored, _ = deserialize_scenario(serialize_scenario(state, None))
    assert restored.gpus[0].gpu_type == "TINYBOX_RED_WHALE"
    assert restored.gpus[0].count == 28
    assert restored.gpus[0].gpu.mem == 32e9
