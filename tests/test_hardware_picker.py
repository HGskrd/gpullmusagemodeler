"""Owned hardware, purchase-unit boundaries and saved-plan compatibility."""

import uuid

import pytest
from app_factory import create_test_app

import web.session_store as session_store
from data import GPU_CARDS, GPU_TCO_DEFAULTS, GPU_TCO_PRICE_USD, GPUS
from data.hardware_picker import (
    HARDWARE_SOURCES,
    PICKER_FAMILIES,
    UNAVAILABLE_HARDWARE,
    hardware_choices,
    picker_vendors,
    purchasable_gpu_keys,
)
from scenarios import deserialize_planner_state


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(session_store, "_states", {})
    monkeypatch.setattr(session_store, "_compare_states", {})
    app = create_test_app(REQUEST_RATE_LIMIT=100_000)
    client = app.test_client()
    client.environ_base["HTTP_X_TAB_ID"] = str(uuid.uuid4())
    assert client.post("/session/reset").status_code == 200
    yield client


def pool(client):
    return client.get("/scenario/export").get_json()["panel_a"]["gpus"][0]


def test_chooser_has_unique_purchasable_profiles_and_keeps_compatibility_keys():
    choices = hardware_choices()
    keys = [choice.key for choice in choices]
    assert len(keys) == len(set(keys))
    assert not set(keys) & UNAVAILABLE_HARDWARE.keys()
    assert set(keys) < GPUS.keys()
    assert set(UNAVAILABLE_HARDWARE) <= GPUS.keys()
    rendered = [
        key
        for families in PICKER_FAMILIES.values()
        for _, variants in families
        for key, _ in variants
    ]
    assert len(rendered) == len(set(rendered))
    assert set(rendered) == set(keys)
    assert list(picker_vendors()) == ["NVIDIA", "AMD", "Intel", "Apple", "Tenstorrent", "FuriosaAI"]


def test_catalog_corrections_and_new_variant_invariants():
    assert GPUS["A100_40"].bf16 == 312e12
    cards = {option.gpu_key: card for card in GPU_CARDS for option in card.planner_options}
    assert "128 GB" in cards["MI250X"].vram
    assert "128 GB" in cards["Gaudi3"].vram
    assert "PCIe" not in cards["H100"].name
    assert "PCIe" not in cards["H200"].name
    expected = {
        "RTX3060_12": (12e9, 360e9, 170),
        "RTX4060TI_16": (16e9, 288e9, 165),
        "RTX5060TI_16": (16e9, 448e9, 180),
        "A100_80_PCIE": (80e9, 1.935e12, 300),
        "A100_40_SXM": (40e9, 1.555e12, 400),
        "H100_PCIE": (80e9, 2e12, 350),
        "H100_NVL": (94e9, 3.9e12, 400),
        "H200_NVL": (141e9, 4.8e12, 600),
        "RYZEN_AI_MAX_395_128": (96e9, 256e9, 120),
    }
    for key, (mem, bw, power) in expected.items():
        gpu = GPUS[key]
        assert (gpu.mem, gpu.bw, gpu.tdp_watts) == (mem, bw, power)
        assert key in purchasable_gpu_keys()
        assert HARDWARE_SOURCES[key].startswith("https://")
    assert GPUS["RTX5060TI_16"].fp4 == 379.5e12
    assert GPUS["RTX5060TI_16"].bf16 == 759e12 / 16
    assert GPUS["RTX5060TI_16"].fp8 == 759e12 / 8
    assert GPUS["RTX5090"].fp4 == 1676e12
    assert GPUS["RTX5090"].bf16 == 209.5e12
    assert GPUS["RTXPRO6000_BW_WS"].fp4 == 2015.2e12
    assert GPUS["H100_NVL"].bf16 == 835.5e12  # half the sparse headline
    assert GPUS["H200_NVL"].node_size == 4
    ryzen = GPUS["RYZEN_AI_MAX_395_128"]
    assert ryzen.system_memory_gb == 128
    assert ryzen.mem_gb < ryzen.system_memory_gb
    assert ryzen.effective_bw < ryzen.bw
    for key, gpu in GPUS.items():
        assert gpu.key == key


def test_unknown_quotes_are_not_starting_prices_or_energy_only_tco():
    for key in (
        "MAC_STUDIO_M5_MAX",
        "MAC_STUDIO_M5_ULTRA",
        "MAC_STUDIO_M5_ULTRA_256",
        "RTX5060TI_16",
    ):
        assert GPU_TCO_PRICE_USD[key] == 0
        assert GPU_TCO_DEFAULTS[key] == 0
        assert "Quote required" in GPUS[key].price_note


def test_picker_has_compact_families_and_no_unavailable_options(client):
    response = client.get("/picker/gpu?panel=A")
    assert response.status_code == 200
    html = response.text
    for text in (
        "data-hardware-search",
        'role="tablist"',
        "gpu-picker-family",
        "Click a variant to add",
        "Quantity",
        "80 GB PCIe",
        "94 GB NVL",
    ):
        assert text in html
    for key in UNAVAILABLE_HARDWARE:
        assert f'data-gpu-key="{key}"' not in html
    assert "Jotunn" not in html
    assert "MI440X" not in html
    assert "<details" not in html
    assert "Edge" not in html


@pytest.mark.parametrize(
    "key,count,expected", [("RTX5060TI_16", 8, 8), ("MAC_STUDIO_M5_MAX_64", 1, 1), ("GB300", 8, 72)]
)
def test_variant_count_and_saved_scenario_contract(client, key, count, expected):
    response = client.post("/gpu/add", data={"gpu_type": key, "count": count})
    assert response.status_code == 200
    assert pool(client)["count"] == expected
    assert pool(client)["gpu_type"] == key


def test_unavailable_profiles_cannot_be_added_but_saved_plans_still_load(client):
    assert client.post("/gpu/add", data={"gpu_type": "RUBIN_NVL72", "count": 72}).status_code == 400
    state = deserialize_planner_state(
        {"projects": [], "models": [], "gpus": [{"uid": 1, "gpu_type": "RUBIN_NVL72", "count": 72}]}
    )
    assert state.gpus[0].count == 72
