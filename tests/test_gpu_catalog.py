import unittest

from data import GPU_CARDS, GPUS


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


if __name__ == "__main__":
    unittest.main()
