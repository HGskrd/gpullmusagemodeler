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


if __name__ == "__main__":
    unittest.main()
