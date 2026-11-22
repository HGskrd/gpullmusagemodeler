"""Complete-system pricing, physical topology and tinygrad configurator constraints."""

import copy

import pytest
from app_factory import create_test_app

from data import GPU_TCO_COMPLETE_SYSTEMS, GPU_TCO_PRICE_USD, GPUS
from data.hardware_picker import hardware_choices
from data.tinybox import TINYBOX_CONFIGS, TINYBOX_GPU_CONFIGS, tinybox_picker_configs
from scenarios import deserialize_scenario, serialize_scenario
from state import PlannerState, add_gpu
from web.agent_contract import EXAMPLE_REQUEST


@pytest.mark.parametrize(
    "key,price,count,mem,watts,peer_bw",
    [
        ("TINYBOX_RED_FLASH", 22000, 6, 32e9, 210, 32e9),
        ("TINYBOX_RED_KIMI", 30000, 4, 32e9, 300, 64e9),
        ("TINYBOX_RED_WHALE", 51000, 14, 32e9, 210, 32e9),
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
    assert len(TINYBOX_GPU_CONFIGS) == 105
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
    assert upgraded.price == 34000
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


def test_mi300x_fixed_system_preserves_topology_price_and_preorder_evidence():
    key = "TINYBOX_RED_MI300X"
    config = TINYBOX_GPU_CONFIGS[key]
    gpu = GPUS[key]
    assert gpu.key == key
    assert (gpu.node_size, gpu.min_count, gpu.count_multiple) == (4, 4, 4)
    assert (gpu.mem, gpu.bw, gpu.tdp_watts) == (192e9, 5.3e12, 500)
    assert gpu.bf16 == pytest.approx(GPUS["MI300X"].bf16 * 500 / 750)
    assert gpu.fp8 == pytest.approx(GPUS["MI300X"].fp8 * 500 / 750)
    assert gpu.scale_up_p2p_bw_bidir == 64e9  # PCIe proxy, not 896 GB/s XGMI.
    assert gpu.inter_node_bw * 4 == 1.25e9  # Conservative single 10GbE path.
    assert gpu.system_memory_gb == 192
    assert GPU_TCO_PRICE_USD[key] * 4 == config.price == 89_000
    assert key in GPU_TCO_COMPLETE_SYSTEMS
    choice = next(c for c in hardware_choices() if c.key == key)
    assert choice.card.architecture == "CDNA 3 · G5"
    assert choice.reviewed_at == "2026-11-19"
    assert choice.source == choice.purchase_source == config.source
    assert "preorder" in choice.availability
    assert "no XGMI" in choice.specs
    assert "no AMD warranty/support" in choice.specs
    assert key not in {c["key"] for c in tinybox_picker_configs()}


def test_mi300x_picker_and_planner_report_roundtrip():
    key = "TINYBOX_RED_MI300X"
    state = PlannerState()
    add_gpu(state, key, 1)
    assert state.gpus[0].count == 4
    restored, _ = deserialize_scenario(serialize_scenario(state, None))
    assert restored.gpus[0].gpu_type == key
    assert restored.gpus[0].count == 4
    payload = copy.deepcopy(EXAMPLE_REQUEST)
    panel = payload["scenario"]["panel_a"]
    panel["gpus"] = [{"uid": 1, "gpu_type": key, "count": 4}]
    panel["models"][0]["gpu_count"] = 4
    with create_test_app(REQUEST_RATE_LIMIT=100_000).test_client() as client:
        picker = client.get("/picker/gpu?panel=A")
        assert picker.status_code == 200
        assert f'data-gpu-key="{key}"' in picker.text
        assert "4× MI300X · 768 GB total" in picker.text
        assert "Batch 2 preorder" in picker.text
        result = client.post("/api/v1/calculate", json=payload)
        assert result.status_code == 200, result.get_json()
        data = result.get_json()
        assert data["results"]["panel_a"]["models"][0]["runnable"]
        assert "tinybox red 4× MI300X" in data["report"]
        replay = client.post("/api/v1/calculate", json={**payload, "scenario": data["scenario"]})
        assert replay.status_code == 200
        assert replay.get_json()["report"] == data["report"]
