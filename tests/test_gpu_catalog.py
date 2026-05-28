import unittest

from data import GPU_CARDS, GPUS, MODELS
from state import (
    PlannerState,
    add_gpu,
    add_model,
    add_models,
    change_gpu_qty,
    get_model_info,
    set_model_gpu_count,
)


class GPUCatalogTests(unittest.TestCase):
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
            "ArcProB70",
            "T4",
            "V100",
            "A30",
            "A40",
            "Gaudi2",
        }
        picker_keys = {
            option.gpu_key
            for card in GPU_CARDS
            for option in card.planner_options
        }

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

    def test_blackwell_ultra_catalog_entries_and_set_constraints(self):
        picker_keys = {
            option.gpu_key
            for card in GPU_CARDS
            for option in card.planner_options
        }

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
            key
            for key, model in MODELS.items()
            if model.is_embedding_model and not model.hidden
        ]

        added = add_models(state, embedding_keys)

        self.assertEqual(added, embedding_keys)
        self.assertEqual([am.model_key for am in state.models], embedding_keys)

        added_again = add_models(state, embedding_keys)

        self.assertEqual(added_again, [])
        self.assertEqual(len(state.models), len(embedding_keys))


if __name__ == "__main__":
    unittest.main()
