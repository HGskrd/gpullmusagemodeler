import unittest

from calc import (
    kv_cache_bytes_for_sequence,
    linear_attention_state_bytes,
    valid_strategies,
)
from data import (
    GPUS,
    MODELS,
    aa_intelligence_to_quality,
    aa_output_tokens_to_efficiency,
    get_quantization_profile,
)


class ModelCatalogTests(unittest.TestCase):
    def test_command_a_plus_catalog_entry_matches_public_specs(self):
        model = MODELS["command-a-plus-05-2026"]

        self.assertEqual(model.name, "Command A+ 05-2026")
        self.assertEqual(model.cat, "Cohere")
        self.assertEqual(model.size_label, "218B-A25B")
        self.assertTrue(model.is_moe)
        self.assertEqual(model.layers, 32)
        self.assertEqual(model.num_heads, 128)
        self.assertEqual(model.kv_heads, 8)
        self.assertEqual(model.head_dim, 128)
        self.assertEqual(model.hidden_size, 4096)
        self.assertEqual(model.local_attention_layers, 24)
        self.assertEqual(model.local_attention_window, 4096)
        self.assertTrue({"tools", "ctx_128k", "images", "reasoning"} <= model.capabilities)
        self.assertAlmostEqual(model.quality, aa_intelligence_to_quality(37.0))
        self.assertAlmostEqual(model.token_efficiency, aa_output_tokens_to_efficiency(66.0))

    def test_command_a_plus_w4a4_min_hardware_shapes_fit(self):
        model = MODELS["command-a-plus-05-2026"]

        self.assertTrue(valid_strategies(model, 1, GPUS["B200"], 0.90, 4.0, "nvfp4"))
        self.assertTrue(valid_strategies(model, 2, GPUS["H100"], 0.90, 4.0, "nvfp4"))

    def test_nvfp4_profile_lut_uses_artifact_storage_for_gemma_31b(self):
        model = MODELS["g31"]
        profile = get_quantization_profile("g31", "nvfp4")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.source_repo, "nvidia/Gemma-4-31B-IT-NVFP4")
        self.assertEqual(profile.source_kind, "exact")
        self.assertIn("language self-attention BF16", profile.retained)
        self.assertAlmostEqual(model.weight_bytes_per_param("nvfp4"), 1.052685646, places=6)
        self.assertAlmostEqual(model.kv_cache_bytes_per_elem("nvfp4"), 1.0)

    def test_qwen_moe_nvfp4_profiles_are_model_specific(self):
        q35 = MODELS["q35"]
        q122 = MODELS["q122"]
        q397 = MODELS["q397"]

        self.assertEqual(get_quantization_profile("q35", "nvfp4").source_kind, "exact")
        self.assertAlmostEqual(q35.weight_bytes_per_param("nvfp4"), 0.726225620, places=6)
        self.assertAlmostEqual(q122.weight_bytes_per_param("nvfp4"), 0.667763208, places=6)
        self.assertAlmostEqual(q397.weight_bytes_per_param("nvfp4"), 0.667763208, places=6)
        self.assertNotAlmostEqual(q35.weight_bytes_per_param("nvfp4"), q122.weight_bytes_per_param("nvfp4"), places=3)

    def test_rwkv7_g1_catalog_family_uses_recurrent_state(self):
        expected = {
            "rwkv7-g1d-01b": (0.1e9, 12, 768, False),
            "rwkv7-g1d-04b": (0.4e9, 24, 1024, False),
            "rwkv7-g1f-15b": (1.5e9, 24, 2048, True),
            "rwkv7-g1f-29b": (2.9e9, 32, 2560, True),
            "rwkv7-g1g-72b": (7.2e9, 32, 4096, True),
            "rwkv7-g1g-133b": (13.3e9, 61, 4096, True),
        }

        for key, (params, layers, hidden_dim, supports_tools) in expected.items():
            with self.subTest(key=key):
                model = MODELS[key]

                self.assertEqual(model.cat, "RWKV")
                self.assertEqual(model.total_params, params)
                self.assertEqual(model.active_params, params)
                self.assertFalse(model.is_moe)
                self.assertEqual(model.layers, layers)
                self.assertEqual(model.hidden_size, hidden_dim)
                self.assertEqual(model.head_dim, 64)
                self.assertEqual(model.num_heads, hidden_dim // 64)
                self.assertEqual(model.kv_heads, 0)
                self.assertEqual(model.attention_layer_count, 0)
                self.assertEqual(model.kv_layer_count, 0)
                self.assertEqual(model.linear_attention_layer_count, layers)
                self.assertEqual(model.linear_attention_head_count, hidden_dim // 64)
                self.assertEqual(model.linear_attention_head_size, 64)
                self.assertIn("reasoning", model.capabilities)
                self.assertNotIn("ctx_128k", model.capabilities)
                self.assertEqual("tools" in model.capabilities, supports_tools)
                self.assertEqual(kv_cache_bytes_for_sequence(model, 8192, "bf16"), 0.0)
                self.assertGreater(linear_attention_state_bytes(model, "bf16"), 0.0)

    def test_voxtral_realtime_catalog_entry_uses_streaming_profile(self):
        model = MODELS["voxtral-realtime-mini-4b"]
        profile = model.realtime_profile

        self.assertEqual(model.name, "Voxtral Mini Realtime 4B")
        self.assertEqual(model.cat, "Audio")
        self.assertEqual(model.total_params, 4.37e9)
        self.assertEqual(model.layers, 26)
        self.assertEqual(model.num_heads, 32)
        self.assertEqual(model.kv_heads, 8)
        self.assertEqual(model.head_dim, 128)
        self.assertEqual(model.hidden_size, 3072)
        self.assertEqual(model.local_attention_layers, 26)
        self.assertEqual(model.local_attention_window, 8192)
        self.assertTrue(model.is_realtime_only)
        self.assertEqual(model.capabilities, frozenset())
        self.assertIsNotNone(profile)
        self.assertEqual(profile.target_delay_ms, 480)
        self.assertAlmostEqual(profile.audio_ms_per_token, 80.0)
        self.assertAlmostEqual(profile.tokens_per_second, 12.5)
        self.assertEqual(profile.state_tokens, 8192)
        self.assertEqual(profile.audio_encoder_params, 0.97e9)
        self.assertEqual(profile.audio_tokens_per_step, 4)
        self.assertEqual(profile.audio_attention_layers, 32)
        self.assertEqual(profile.audio_attention_heads, 32)
        self.assertEqual(profile.audio_attention_head_dim, 64)
        self.assertEqual(profile.audio_attention_window, 750)


if __name__ == "__main__":
    unittest.main()
