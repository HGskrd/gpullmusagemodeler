import unittest

from data import (
    GPU_CARDS,
    GPU_TCO_DEFAULTS,
    GPU_TCO_PRICE_USD,
    GPUS,
    MODELS,
    PREVIEW_ASSUMPTIONS,
    RADEON_GPU_SPEC_SOURCES,
    RADEON_GPU_SPECS_CAPTURED_AT,
)
from planner_service import add_model, add_models, change_gpu_qty, set_model_gpu_count
from presentation.model_cards import get_model_info
from scenarios import deserialize_planner_state
from state import (
    PlannerState,
    add_gpu,
)


class GPUCatalogTests(unittest.TestCase):
    def test_every_gpu_has_researched_tco_default(self):
        self.assertEqual(set(GPU_TCO_PRICE_USD), set(GPUS))
        self.assertEqual(set(GPU_TCO_DEFAULTS), set(GPUS))
        self.assertEqual(GPU_TCO_DEFAULTS["H100"], 1.32)
        self.assertEqual(GPU_TCO_DEFAULTS["MI400"], GPU_TCO_DEFAULTS["MI455X"])
        self.assertGreater(GPU_TCO_DEFAULTS["RUBIN_NVL72"], GPU_TCO_DEFAULTS["GB300"])

        for key, gpu in GPUS.items():
            with self.subTest(gpu=key):
                self.assertGreater(GPU_TCO_DEFAULTS[key], 0.0)
                self.assertEqual(gpu.default_tco_per_gpu_hour, GPU_TCO_DEFAULTS[key])

    def test_new_gpu_pools_prefill_researched_tco(self):
        state = PlannerState()

        add_gpu(state, "H100", 1)

        self.assertEqual(state.gpus[0].cost_per_gpu_hour, GPUS["H100"].default_tco_per_gpu_hour)

    def test_scenario_import_defaults_missing_tco_but_preserves_explicit_zero(self):
        defaulted = deserialize_planner_state(
            {
                "projects": [],
                "gpus": [{"uid": 1, "gpu_type": "H100", "count": 1}],
                "models": [],
            }
        )
        suppressed = deserialize_planner_state(
            {
                "projects": [],
                "gpus": [{"uid": 1, "gpu_type": "H100", "count": 1, "cost_per_gpu_hour": 0.0}],
                "models": [],
            }
        )

        self.assertEqual(defaulted.gpus[0].cost_per_gpu_hour, GPUS["H100"].default_tco_per_gpu_hour)
        self.assertEqual(suppressed.gpus[0].cost_per_gpu_hour, 0.0)

    def test_vera_rubin_preview_profile_uses_preliminary_vendor_specs(self):
        gpu = GPUS["RUBIN_NVL72"]
        picker_keys = {option.gpu_key for card in GPU_CARDS for option in card.planner_options}

        self.assertEqual(gpu.name, "Vera Rubin NVL72 Preview 288GB/GPU")
        self.assertEqual(gpu.mem, 288e9)
        self.assertEqual(gpu.bw, 22e12)
        self.assertEqual(gpu.bf16, 4e15)
        self.assertEqual(gpu.fp8, 17.5e15)
        self.assertEqual(gpu.fp4, 50e15)
        self.assertEqual(gpu.scale_up_p2p_bw_bidir, 3.6e12)
        self.assertEqual(gpu.node_size, 72)
        self.assertEqual(gpu.min_count, 72)
        self.assertEqual(gpu.count_multiple, 72)
        self.assertEqual(gpu.tdp_watts, 0.0)
        self.assertIn("RUBIN_NVL72", picker_keys)
        self.assertIn("gpu:RUBIN_NVL72", PREVIEW_ASSUMPTIONS)
        self.assertIn("preliminary", PREVIEW_ASSUMPTIONS["gpu:RUBIN_NVL72"]["assumptions"][0])

    def test_mi455x_profile_uses_named_helios_accelerator(self):
        gpu = GPUS["MI455X"]
        card = next(
            card
            for card in GPU_CARDS
            if any(option.gpu_key == "MI455X" for option in card.planner_options)
        )
        assumptions = PREVIEW_ASSUMPTIONS["gpu:MI455X"]

        self.assertIn("MI455X", gpu.name)
        self.assertEqual(gpu.mem, 432e9)
        self.assertEqual(gpu.bw, 23.3e12)
        self.assertEqual(gpu.bf16, 5e15)
        self.assertEqual(gpu.fp8, 20.1e15)
        self.assertEqual(gpu.fp4, 40.3e15)
        self.assertEqual(gpu.scale_up_p2p_bw_bidir, 3.6e12)
        self.assertIn("Helios", card.use_case)
        self.assertIn("2H 2026", assumptions["status"])
        self.assertTrue(str(assumptions["source"]).startswith("https://www.amd.com/"))
        assumption_text = " ".join(str(item) for item in assumptions["assumptions"])
        self.assertIn("23.3 TB/s", assumption_text)
        self.assertNotIn("19.6 TB/s", assumption_text)
        self.assertNotIn("10 PF BF16", assumption_text)

    def test_helios_profile_uses_official_per_gpu_mi455x_fields(self):
        helios = GPUS["HELIOS_MI455X"]
        assumptions = PREVIEW_ASSUMPTIONS["gpu:HELIOS_MI455X"]

        self.assertEqual(helios.mem, 432e9)
        self.assertEqual(helios.bw, 23.3e12)
        self.assertEqual(helios.bf16, 5e15)
        self.assertEqual(helios.fp8, 20.1e15)
        self.assertEqual(helios.fp4, 40.3e15)
        self.assertEqual(helios.scale_up_p2p_bw_bidir, 3.6e12)
        self.assertEqual(helios.node_size, 72)
        self.assertEqual((helios.min_count, helios.count_multiple), (72, 72))
        self.assertIn("reference design", assumptions["status"].lower())
        self.assertIn("2H 2026", assumptions["status"])
        self.assertTrue(str(assumptions["source"]).startswith("https://www.amd.com/"))
        assumption_text = " ".join(str(item) for item in assumptions["assumptions"])
        self.assertIn("23.3 TB/s", assumption_text)
        self.assertNotIn("19.6 TB/s", assumption_text)
        self.assertNotIn("10 PF BF16", assumption_text)

    def test_mi400_legacy_key_remains_available_for_saved_plans(self):
        legacy = GPUS["MI400"]
        mi455x = GPUS["MI455X"]
        assumptions = PREVIEW_ASSUMPTIONS["gpu:MI400"]

        for field in (
            "mem",
            "bw",
            "bf16",
            "fp8",
            "fp4",
            "scale_up_p2p_bw_bidir",
            "node_size",
            "tdp_watts",
        ):
            with self.subTest(field=field):
                self.assertEqual(getattr(legacy, field), getattr(mi455x, field))
        self.assertIn("compatibility", legacy.name)
        self.assertIn("compatibility", assumptions["status"])
        self.assertTrue(str(assumptions["source"]).startswith("https://www.amd.com/"))
        assumption_text = " ".join(str(item) for item in assumptions["assumptions"])
        self.assertIn("23.3 TB/s", assumption_text)
        self.assertNotIn("19.6 TB/s", assumption_text)
        self.assertNotIn("10 PF BF16", assumption_text)

    def test_new_system_and_specialist_accelerator_entries_are_exposed(self):
        picker_keys = {option.gpu_key for card in GPU_CARDS for option in card.planner_options}

        for key in {
            "HELIOS_MI455X",
            "TT_GALAXY_BLACKHOLE",
            "TT_BLACKHOLE_P150",
            "TT_BLACKHOLE_P100A",
            "FURIOSA_RNGD",
        }:
            self.assertIn(key, GPUS)
            self.assertIn(key, picker_keys)

        helios = GPUS["HELIOS_MI455X"]
        self.assertEqual((helios.min_count, helios.count_multiple), (72, 72))
        self.assertEqual(GPUS["TT_GALAXY_BLACKHOLE"].min_count, 32)
        self.assertEqual(GPUS["FURIOSA_RNGD"].mem, 48e9)
        self.assertEqual(GPUS["FURIOSA_RNGD"].bw, 1.5e12)

    def test_unpublished_accelerators_are_reference_only(self):
        cards = {card.name: card for card in GPU_CARDS}

        self.assertFalse(cards["MI440X"].planner_options)
        self.assertFalse(cards["VSORA Jotunn 8"].planner_options)

    def test_preview_gpu_picker_entries_have_assumption_records(self):
        preview_keys = {
            option.gpu_key
            for card in GPU_CARDS
            for option in card.planner_options
            if "preview" in option.label.lower()
        }

        self.assertTrue({f"gpu:{key}" for key in preview_keys} <= set(PREVIEW_ASSUMPTIONS))

    def test_nvidia_a10_catalog_entry_matches_public_specs(self):
        gpu = GPUS["A10"]

        self.assertEqual(gpu.name, "A10 24GB PCIe")
        self.assertEqual(gpu.vendor_label, "NVIDIA")
        self.assertEqual(gpu.mem, 24e9)
        self.assertEqual(gpu.bw, 600e9)
        self.assertEqual(gpu.bf16, 125e12)
        self.assertEqual(gpu.fp8, 125e12)
        self.assertEqual(gpu.scale_up_p2p_bw_bidir, 64e9)
        self.assertEqual(gpu.node_size, 8)
        self.assertEqual(gpu.tdp_watts, 150.0)

        self.assertTrue(
            any(
                card.name == "A10"
                and any(option.gpu_key == "A10" for option in card.planner_options)
                for card in GPU_CARDS
            )
        )

    def test_requested_local_and_legacy_hardware_is_exposed(self):
        requested_keys = {
            "RTX5090",
            "RTX4090",
            "RTX3090",
            "RTXPRO6000_BW_WS",
            "RTXPRO5000_BW_72",
            "RTX6000_ADA",
            "RadeonProW7900",
            "RadeonAIProR9700",
            "RadeonRX9070XT",
            "RadeonRX7900XT",
            "RadeonRX7900XTX",
            "ArcProB70",
            "T4",
            "V100",
            "A30",
            "A40",
            "Gaudi2",
        }
        picker_keys = {option.gpu_key for card in GPU_CARDS for option in card.planner_options}

        for key in requested_keys:
            self.assertIn(key, GPUS)
            self.assertIn(key, picker_keys)

        self.assertEqual(GPUS["RTX5090"].mem, 32e9)
        self.assertEqual(GPUS["RTX5090"].bw, 1.792e12)
        self.assertEqual(GPUS["RTX4090"].mem, 24e9)
        self.assertEqual(GPUS["RTX3090"].mem, 24e9)
        self.assertEqual(GPUS["RTX3090"].bw, 936e9)
        self.assertEqual(GPUS["RTX3090"].tdp_watts, 350.0)
        self.assertEqual(GPUS["RTXPRO6000_BW_WS"].mem, 96e9)
        self.assertEqual(GPUS["RTXPRO5000_BW_72"].mem, 72e9)
        self.assertEqual(GPUS["RadeonAIProR9700"].bw, 640e9)
        self.assertEqual(GPUS["ArcProB70"].bw, 608e9)
        self.assertEqual(GPUS["Gaudi2"].mem, 96e9)

    def test_requested_radeon_desktop_entries_match_amd_reference_specs(self):
        expected = {
            "RadeonRX9070XT": (16e9, 640e9, 195e12, 389e12, 128e9, 304.0),
            "RadeonRX7900XT": (20e9, 800e9, 103e12, 103e12, 64e9, 315.0),
            "RadeonRX7900XTX": (24e9, 960e9, 123e12, 123e12, 64e9, 355.0),
            "RadeonAIProR9700": (32e9, 640e9, 191e12, 383e12, 128e9, 300.0),
        }

        self.assertEqual(RADEON_GPU_SPECS_CAPTURED_AT, "2026-08-19")
        self.assertEqual(set(RADEON_GPU_SPEC_SOURCES), set(expected))

        for key, specs in expected.items():
            with self.subTest(gpu=key):
                gpu = GPUS[key]
                self.assertEqual(
                    (
                        gpu.mem,
                        gpu.bw,
                        gpu.bf16,
                        gpu.fp8,
                        gpu.scale_up_p2p_bw_bidir,
                        gpu.tdp_watts,
                    ),
                    specs,
                )
                self.assertEqual(gpu.vendor, "amd")
                self.assertEqual(gpu.node_size, 1)
                self.assertTrue(RADEON_GPU_SPEC_SOURCES[key].startswith("https://www.amd.com/"))

    def test_blackwell_ultra_catalog_entries_and_set_constraints(self):
        picker_keys = {option.gpu_key for card in GPU_CARDS for option in card.planner_options}

        for key in {"GB300", "DGX_STATION_GB300", "B300", "B200", "GB200"}:
            self.assertIn(key, GPUS)
            self.assertIn(key, picker_keys)

        gb300 = GPUS["GB300"]
        self.assertEqual(gb300.mem, 288e9)
        self.assertEqual(gb300.bw, 8e12)
        self.assertEqual(gb300.bf16, 2.5e15)
        self.assertEqual(gb300.fp8, 5e15)
        self.assertEqual(gb300.fp4, 15e15)
        self.assertEqual(gb300.node_size, 72)
        self.assertEqual(gb300.min_count, 72)
        self.assertEqual(gb300.count_multiple, 72)

        station = GPUS["DGX_STATION_GB300"]
        self.assertEqual(station.mem, 252e9)
        self.assertEqual(station.bw, 7.1e12)
        self.assertEqual(station.fp4, 15e15)
        self.assertEqual(station.node_size, 2)
        self.assertEqual(station.tdp_watts, 1600.0)

        b300 = GPUS["B300"]
        self.assertEqual(b300.bf16, 2.5e15)
        self.assertEqual(b300.fp8, 5e15)
        self.assertEqual(b300.tdp_watts, 1400.0)
        self.assertEqual(b300.min_count, 8)
        self.assertEqual(b300.count_multiple, 8)

    def test_set_only_gpu_counts_snap_to_valid_system_sizes(self):
        state = PlannerState()

        add_gpu(state, "GB300", 8)
        self.assertEqual(state.gpus[0].count, 72)

        change_gpu_qty(state, state.gpus[0].uid, 72)
        self.assertEqual(state.gpus[0].count, 144)

        change_gpu_qty(state, state.gpus[0].uid, -72)
        self.assertEqual(state.gpus[0].count, 72)

        change_gpu_qty(state, state.gpus[0].uid, -72)
        self.assertEqual(state.gpus, [])

        add_gpu(state, "B300", 1)
        self.assertEqual(state.gpus[0].count, 8)

    def test_model_gpu_count_options_reach_full_nvl72_pool(self):
        state = PlannerState()

        add_gpu(state, "GB300", 8)
        add_model(state, "q27")

        info = get_model_info(state, state.models[0])
        self.assertIn(72, info["gpu_count_options"])

        set_model_gpu_count(state, state.models[0].uid, 72)
        self.assertEqual(state.models[0].gpu_count, 72)
        self.assertIn(72, get_model_info(state, state.models[0])["gpu_count_options"])

    def test_bulk_add_models_adds_visible_embedding_models_once(self):
        state = PlannerState()
        add_gpu(state, "H100", 64)
        embedding_keys = [
            key for key, model in MODELS.items() if model.is_embedding_model and not model.hidden
        ]

        added = add_models(state, embedding_keys)

        self.assertEqual(added, embedding_keys)
        self.assertEqual([am.model_key for am in state.models], embedding_keys)

        added_again = add_models(state, embedding_keys)

        self.assertEqual(added_again, [])
        self.assertEqual(len(state.models), len(embedding_keys))


if __name__ == "__main__":
    unittest.main()
