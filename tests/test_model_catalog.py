import unittest

from calc import (
    EfficiencyParams,
    compute_embedding,
    compute_embedding_distribution,
    embedding_doc_stats,
    embedding_sequence_length,
    embedding_vectors_per_input,
    kv_bytes_per_token,
    kv_cache_bytes_for_sequence,
    linear_attention_state_bytes,
    valid_strategies,
)
from data import (
    EMBEDDING_DOC_BUCKETS,
    EMBEDDING_DOC_PRESETS,
    EMBEDDING_DECONTAMINATED_BEIR_SOURCES,
    EMBEDDING_QUALITY_PLACEHOLDER,
    EMBEDDING_QUALITY_SOURCES,
    GPUS,
    MODELS,
    PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR,
    PUBLISHED_EMBEDDING_QUALITY,
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

    def test_laguna_m1_catalog_entry_uses_public_poolside_specs(self):
        model = MODELS["laguna-m1"]

        self.assertEqual(model.name, "Laguna M.1 225B-A23B")
        self.assertEqual(model.cat, "Poolside")
        self.assertEqual(model.size_label, "225B-A23B")
        self.assertTrue(model.is_moe)
        self.assertEqual(model.total_params, 225e9)
        self.assertEqual(model.active_params, 23e9)
        self.assertEqual(model.layers, 64)
        self.assertEqual(model.num_heads, 64)
        self.assertEqual(model.kv_heads, 8)
        self.assertEqual(model.head_dim, 128)
        self.assertEqual(model.hidden_size, 4096)
        self.assertEqual(model.local_attention_layers, 48)
        self.assertEqual(model.local_attention_window, 512)
        self.assertIn("proxy", model.attention_label)
        self.assertTrue({"tools", "ctx_128k", "reasoning"} <= model.capabilities)
        self.assertNotIn("images", model.capabilities)
        self.assertAlmostEqual(model.quality, aa_intelligence_to_quality(44.0))
        self.assertAlmostEqual(model.token_efficiency, aa_output_tokens_to_efficiency(95.0))
        self.assertAlmostEqual(model.quality_confidence, 0.55)

    def test_gemma4_12b_unified_catalog_entry_uses_encoder_free_specs(self):
        model = MODELS["g12"]

        self.assertEqual(model.name, "Gemma 4 12B Unified")
        self.assertEqual(model.cat, "Gemma")
        self.assertEqual(model.size_label, "12B")
        self.assertFalse(model.is_moe)
        self.assertEqual(model.total_params, 11.95e9)
        self.assertEqual(model.active_params, 11.95e9)
        self.assertEqual(model.layers, 48)
        self.assertEqual(model.attention_layer_count, 48)
        self.assertEqual(model.local_attention_layers, 40)
        self.assertEqual(model.local_attention_window, 1024)
        self.assertEqual(model.hidden_size, 3840)
        self.assertEqual(model.num_heads, 16)
        self.assertEqual(model.kv_heads, 8)
        self.assertEqual(model.global_kv_heads, 1)
        self.assertEqual(model.global_head_dim, 512)
        self.assertEqual(model.head_dim, 256)
        self.assertTrue(model.shared_key_value)
        self.assertIn("encoder-free image/audio projection", model.attention_label)
        self.assertTrue({"tools", "ctx_128k", "images", "audio", "reasoning"} <= model.capabilities)
        self.assertAlmostEqual(model.quality, aa_intelligence_to_quality(25.0))
        self.assertAlmostEqual(model.token_efficiency, aa_output_tokens_to_efficiency(12.0))
        self.assertAlmostEqual(model.quality_confidence, 0.65)

        self.assertEqual(kv_bytes_per_token(model, "bf16"), 172_032)
        self.assertEqual(kv_cache_bytes_for_sequence(model, 32_768, "bf16"), 436_207_616)

    def test_lfm_catalog_entries_use_hybrid_attention_specs(self):
        expected = {
            "lfm2.5-350m": ("LFM2.5 350M", 354_483_968, 354_483_968, 16, 6, 1024, 16, 8, False),
            "lfm2.5-1.2b-instruct": ("LFM2.5 1.2B Instruct", 1_170_340_608, 1_170_340_608, 16, 6, 2048, 32, 8, False),
            "lfm2.5-1.2b-thinking": ("LFM2.5 1.2B Thinking", 1_170_340_608, 1_170_340_608, 16, 6, 2048, 32, 8, True),
            "lfm2-700m": ("LFM2 700M", 742_489_344, 742_489_344, 16, 6, 1536, 24, 8, False),
            "lfm2-2.6b": ("LFM2 2.6B", 2_569_272_320, 2_569_272_320, 30, 8, 2048, 32, 8, False),
            "lfm2-8b-a1b": ("LFM2 8B-A1.5B", 8.3e9, 1.5e9, 24, 6, 2048, 32, 8, False),
            "lfm2-24b-a2b": ("LFM2 24B-A2.3B", 24e9, 2.3e9, 40, 10, 2048, 32, 8, False),
        }

        for key, (name, total, active, layers, attn_layers, hidden_dim, heads, kv_heads, reasoning) in expected.items():
            with self.subTest(key=key):
                model = MODELS[key]

                self.assertEqual(model.name, name)
                self.assertEqual(model.cat, "LFM")
                self.assertEqual(model.total_params, total)
                self.assertEqual(model.active_params, active)
                self.assertEqual(model.is_moe, total != active)
                self.assertEqual(model.layers, layers)
                self.assertEqual(model.attention_layer_count, attn_layers)
                self.assertEqual(model.kv_layer_count, attn_layers)
                self.assertEqual(model.hidden_size, hidden_dim)
                self.assertEqual(model.num_heads, heads)
                self.assertEqual(model.kv_heads, kv_heads)
                self.assertEqual(model.head_dim, 64)
                self.assertIn("LIV conv", model.attention_label)
                self.assertIn("GQA", model.attention_label)
                self.assertIn("tools", model.capabilities)
                self.assertNotIn("ctx_128k", model.capabilities)
                self.assertEqual("reasoning" in model.capabilities, reasoning)
                self.assertGreater(kv_cache_bytes_for_sequence(model, 32768, "bf16"), 0.0)

        self.assertAlmostEqual(MODELS["lfm2.5-1.2b-instruct"].quality, aa_intelligence_to_quality(8.0))
        self.assertAlmostEqual(MODELS["lfm2.5-1.2b-instruct"].token_efficiency, aa_output_tokens_to_efficiency(4.6))
        self.assertAlmostEqual(MODELS["lfm2.5-1.2b-thinking"].token_efficiency, aa_output_tokens_to_efficiency(31.0))

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

    def test_mimo_v25_asr_catalog_entry_uses_hf_config(self):
        model = MODELS["mimo-v2.5-asr"]
        profile = model.realtime_profile

        self.assertEqual(model.name, "MiMo-V2.5-ASR 8B")
        self.assertEqual(model.cat, "Audio")
        self.assertEqual(model.size_label, "8B")
        self.assertFalse(model.is_moe)
        self.assertEqual(model.total_params, 8.0e9)
        self.assertEqual(model.layers, 36)
        self.assertEqual(model.num_heads, 32)
        self.assertEqual(model.kv_heads, 8)
        self.assertEqual(model.head_dim, 128)
        self.assertEqual(model.hidden_size, 4096)
        self.assertEqual(model.attention_layer_count, 36)
        self.assertEqual(model.local_attention_layers, 0)
        self.assertTrue(model.is_realtime_only)
        self.assertEqual(model.capabilities, frozenset())
        self.assertIsNotNone(profile)
        self.assertEqual(profile.source, "XiaomiMiMo/MiMo-V2.5-ASR + XiaomiMiMo/MiMo-Audio-Tokenizer")
        self.assertEqual(profile.state_tokens, 8192)
        self.assertAlmostEqual(profile.audio_ms_per_token, 160.0)
        self.assertAlmostEqual(profile.tokens_per_second, 6.25)
        self.assertEqual(profile.audio_encoder_params, 1.2e9)
        self.assertEqual(profile.audio_tokens_per_step, 4)
        self.assertEqual(profile.audio_attention_layers, 32)
        self.assertEqual(profile.audio_attention_heads, 20)
        self.assertEqual(profile.audio_attention_head_dim, 64)

    def test_added_asr_catalog_entries_have_profiles(self):
        expected = {
            "nvidia-nemotron-speech-streaming-0.6b": ("NVIDIA Nemotron Speech Streaming 0.6B", 0.6e9, True),
            "nvidia-nemotron-3.5-asr-streaming-0.6b": ("NVIDIA Nemotron 3.5 ASR Streaming 0.6B", 0.6e9, True),
            "nvidia-parakeet-unified-0.6b": ("NVIDIA Parakeet Unified 0.6B", 0.6e9, True),
            "nvidia-parakeet-realtime-eou-120m": ("NVIDIA Parakeet Realtime EOU 120M", 120e6, True),
            "nvidia-multitalker-parakeet-streaming-0.6b": ("NVIDIA Multitalker Parakeet Streaming 0.6B", 0.6e9, True),
            "kyutai-stt-1b-en-fr": ("Kyutai STT 1B EN/FR", 1.0e9, True),
            "kyutai-stt-2.6b-en": ("Kyutai STT 2.6B EN", 2.6e9, True),
            "moonshine-streaming-tiny": ("Moonshine Streaming Tiny 34M", 34e6, True),
            "moonshine-streaming-small": ("Moonshine Streaming Small 123M", 123e6, True),
            "moonshine-streaming-medium": ("Moonshine Streaming Medium 245M", 245e6, True),
            "fun-asr-nano-2512": ("Fun-ASR-Nano 2512 800M", 800e6, True),
            "granite-4.0-1b-speech": ("Granite 4.0 1B Speech", 1.0e9, False),
            "nvidia-parakeet-tdt-0.6b-v3": ("NVIDIA Parakeet TDT 0.6B v3", 0.6e9, False),
        }

        for key, (name, params, streaming) in expected.items():
            with self.subTest(model=key):
                model = MODELS[key]
                profile = model.realtime_profile

                self.assertEqual(model.name, name)
                self.assertEqual(model.cat, "Audio")
                self.assertEqual(model.total_params, params)
                self.assertTrue(model.is_realtime_only)
                self.assertEqual(model.capabilities, frozenset())
                self.assertIsNotNone(profile)
                self.assertEqual(profile.streaming, streaming)
                self.assertGreater(profile.tokens_per_second, 0)
                self.assertGreater(profile.target_delay_ms, 0)

    def test_nemotron_35_asr_streaming_catalog_entry_uses_hf_card(self):
        model = MODELS["nvidia-nemotron-3.5-asr-streaming-0.6b"]
        profile = model.realtime_profile

        self.assertEqual(model.name, "NVIDIA Nemotron 3.5 ASR Streaming 0.6B")
        self.assertEqual(model.cat, "Audio")
        self.assertEqual(model.size_label, "0.6B")
        self.assertFalse(model.is_moe)
        self.assertEqual(model.total_params, 0.6e9)
        self.assertEqual(model.layers, 24)
        self.assertEqual(model.hidden_size, 1024)
        self.assertEqual(model.local_attention_layers, 24)
        self.assertEqual(model.local_attention_window, 56)
        self.assertIn("Prompted", model.attention_label)
        self.assertTrue(model.is_realtime_only)
        self.assertEqual(model.capabilities, frozenset())
        self.assertIsNotNone(profile)
        self.assertEqual(profile.source, "nvidia/nemotron-3.5-asr-streaming-0.6b")
        self.assertEqual(profile.state_tokens, 56)
        self.assertEqual(profile.target_delay_ms, 560)
        self.assertAlmostEqual(profile.audio_ms_per_token, 560.0)
        self.assertAlmostEqual(profile.tokens_per_second, 1000.0 / 560.0)

    def test_embedding_catalog_entries_cover_dense_and_late_modes(self):
        expected = {
            "denseon": ("single", 768, 8192),
            "lateon": ("late", 128, 300),
            "bge-m3": ("hybrid", 1024, 8192),
            "mxbai-embed-large-v1": ("single", 1024, 512),
            "mxbai-embed-2d-large-v1": ("single", 1024, 512),
            "mxbai-embed-xsmall-v1": ("single", 384, 4096),
            "deepset-mxbai-embed-de-large-v1": ("single", 1024, 512),
            "mxbai-edge-colbert-v0-17m": ("late", 48, 32000),
            "mxbai-edge-colbert-v0-32m": ("late", 64, 32000),
            "modernbert-embed-base": ("single", 768, 8192),
            "kalm-mini-it-v15": ("single", 896, 512),
            "pplx-embed-v1-0.6b": ("single", 1024, 32768),
            "pplx-embed-v1-4b": ("single", 2560, 32768),
            "pplx-embed-v1-late-0.6b": ("late", 128, 32768),
        }

        for key, (kind, dim, max_len) in expected.items():
            with self.subTest(key=key):
                model = MODELS[key]
                profile = model.embedding_profile

                self.assertEqual(model.cat, "Embeddings")
                self.assertTrue(model.is_embedding_model)
                self.assertEqual(model.capabilities, frozenset())
                self.assertIsNotNone(profile)
                self.assertEqual(profile.kind, kind)
                self.assertEqual(profile.output_dim, dim)
                self.assertEqual(profile.max_sequence_length, max_len)

        self.assertEqual(MODELS["lateon"].embedding_profile.late_interaction_dim, 128)
        self.assertEqual(MODELS["bge-m3"].embedding_profile.late_interaction_dim, 1024)
        self.assertEqual(MODELS["mxbai-edge-colbert-v0-17m"].embedding_profile.late_interaction_dim, 48)
        self.assertEqual(MODELS["mxbai-edge-colbert-v0-32m"].embedding_profile.late_interaction_dim, 64)

    def test_embedding_quality_scores_are_sourced(self):
        expected_quality = {
            "denseon": 0.5620,
            "lateon": 0.5722,
            "bge-m3": 0.5288,
            "mxbai-embed-large-v1": 0.5439,
            "mxbai-embed-2d-large-v1": 0.5142,
            "mxbai-embed-xsmall-v1": 0.4280,
            "deepset-mxbai-embed-de-large-v1": 0.5170,
            "mxbai-edge-colbert-v0-17m": 0.4900,
            "mxbai-edge-colbert-v0-32m": 0.5210,
            "modernbert-embed-base": 0.5289,
            "kalm-mini-it-v15": 0.5165,
            "pplx-embed-v1-0.6b": 0.6541,
            "pplx-embed-v1-4b": 0.6966,
            "pplx-embed-v1-late-0.6b": 0.5661,
        }
        expected_decontaminated_beir = {
            "denseon": 0.5771,
            "lateon": 0.6036,
            "modernbert-embed-base": 0.5442,
            "pplx-embed-v1-0.6b": 0.5850,
        }

        self.assertEqual(EMBEDDING_QUALITY_PLACEHOLDER, frozenset())
        self.assertEqual(set(PUBLISHED_EMBEDDING_QUALITY), set(expected_quality))
        self.assertEqual(set(EMBEDDING_QUALITY_SOURCES), set(expected_quality))
        self.assertEqual(set(PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR), set(expected_decontaminated_beir))
        self.assertEqual(set(EMBEDDING_DECONTAMINATED_BEIR_SOURCES), set(expected_decontaminated_beir))
        for key, score in expected_quality.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(PUBLISHED_EMBEDDING_QUALITY[key], score)
                self.assertGreater(len(EMBEDDING_QUALITY_SOURCES[key]), 20)
        for key, score in expected_decontaminated_beir.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR[key], score)
                self.assertIn("Decontaminated BEIR", EMBEDDING_DECONTAMINATED_BEIR_SOURCES[key])

    def test_embedding_estimator_caps_sequence_and_counts_vectors(self):
        model = MODELS["lateon"]

        self.assertEqual(embedding_sequence_length(model, 1024), 300)
        self.assertEqual(embedding_vectors_per_input(model, 300), 300)

        result = compute_embedding(
            model,
            (1, 1, 1),
            8,
            1024,
            GPUS["L4"],
            0.90,
            2.0,
            "bf16",
            EfficiencyParams(),
        )

        self.assertIsNotNone(result)
        self.assertGreater(result.rps, 0)
        self.assertEqual(result.seq_len, 300)
        self.assertEqual(result.vectors_per_input, 300)

    def test_embedding_estimator_uses_document_size_distribution(self):
        model = MODELS["denseon"]
        stats = embedding_doc_stats(model, EMBEDDING_DOC_PRESETS["Doc"], EMBEDDING_DOC_BUCKETS, "bf16")

        self.assertGreater(stats.mean_seq_len, 1000)
        self.assertGreaterEqual(stats.p90_seq_len, stats.p50_seq_len)
        self.assertGreater(stats.mean_output_bytes_per_input, 0)

        result = compute_embedding_distribution(
            model,
            (1, 1, 1),
            8,
            EMBEDDING_DOC_PRESETS["Doc"],
            EMBEDDING_DOC_BUCKETS,
            GPUS["L4"],
            0.90,
            2.0,
            "bf16",
            EfficiencyParams(),
        )

        self.assertIsNotNone(result)
        self.assertGreater(result.rps, 0)
        self.assertEqual(result.seq_len, round(stats.mean_seq_len))
        self.assertEqual(result.p90_seq_len, stats.p90_seq_len)


if __name__ == "__main__":
    unittest.main()
