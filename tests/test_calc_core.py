import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from calc import (
    EfficiencyParams,
    SpecRuntime,
    _active_weight_bytes,
    _cloud_price_per_m_in_preset,
    _compute_decode_core,
    _decode_attention_work,
    _decode_step_time,
    _dense_tp_oh,
    _deployment_capacity_for_profile,
    _pp_peak_fraction,
    _prefill_attention_work,
    attention_residual_scratch_bytes,
    chart_user_pareto,
    communication_breakdown,
    compute_data,
    compute_data_capacity,
    compute_decode,
    compute_embedding,
    compute_memory,
    compute_prefill,
    compute_revenue_projection,
    fixed_paged_oh,
    kv_cache_bytes_for_sequence,
    model_gpu_flops,
    optimize_spec_k,
    per_replica_kv_cache_bytes,
    per_replica_recurrent_state_bytes,
    per_replica_token_kv_cache_bytes,
    per_tp_linear_attention_state_bytes,
    resolve_spec_runtime,
    spec_acceptance_len,
    spec_finite_output_tau,
    valid_strategies,
)
from data import (
    DIST_PRESETS,
    GPU,
    GPUS,
    MODELS,
    EmbeddingProfile,
    Model,
    SpeculativeProfile,
    success_rate,
)
from placement import get_deployed, retune_models
from state import GpuPool, ModelAssignment, PlannerState, Project


class CoreCapacityMathTests(unittest.TestCase):
    def setUp(self):
        self.eff = EfficiencyParams()
        self.chat_in = DIST_PRESETS["Chat"]["in"]
        self.chat_out = DIST_PRESETS["Chat"]["out"]

    def test_mla_cache_stores_one_joint_latent_plus_rope_key(self):
        model = MODELS["ds3"]
        seq_len = 131_072
        expected = model.kv_layer_count * seq_len * (model.mla_kv_dim + model.mla_rope_dim) * 2

        self.assertEqual(kv_cache_bytes_for_sequence(model, seq_len, "bf16"), expected)

        roomy_gpu = GPU("roomy", "Roomy", "nv", 2e12, 1e12, 1e12, 1e12, 1e12, 8)
        mem = compute_memory(model, 1, 1, roomy_gpu, 0.90, 2.0, "bf16", self.eff)
        self.assertIsNotNone(mem)
        self.assertEqual(mem.kv_per_token, kv_cache_bytes_for_sequence(model, 1, "bf16"))

    def test_linear_attention_state_uses_schema_head_shards(self):
        model = Model(
            "linear",
            "Linear",
            "Test",
            "#000",
            1,
            1,
            False,
            2,
            4,
            1,
            8,
            False,
            attention_layers=0,
            linear_attention_layers=2,
            linear_attention_heads=4,
            linear_attention_head_dim=8,
            linear_attention_k_heads=2,
            linear_attention_k_head_dim=4,
            linear_attention_conv_kernel=3,
        )

        recurrent = (4 * 8 * 8) / 4
        convolution = 2 * ((4 * 8) / 4 + (2 * 2 * 4) / 2)
        expected = 2 * (recurrent + convolution) * 2
        self.assertEqual(per_tp_linear_attention_state_bytes(model, "bf16", 4), expected)

    def test_recurrent_state_is_split_from_token_growing_kv(self):
        model = Model(
            "hybrid",
            "Hybrid",
            "Test",
            "#000",
            1,
            1,
            False,
            2,
            4,
            1,
            8,
            False,
            attention_layers=1,
            linear_attention_layers=1,
            linear_attention_heads=4,
            linear_attention_head_dim=8,
            linear_attention_k_heads=2,
            linear_attention_k_head_dim=4,
            linear_attention_conv_kernel=3,
        )
        token_kv = per_replica_token_kv_cache_bytes(model, 100, "bf16", 1, 1)
        recurrent = per_replica_recurrent_state_bytes(model, "bf16", 1, 1)

        self.assertGreater(token_kv, 0)
        self.assertGreater(recurrent, 0)
        self.assertEqual(per_replica_kv_cache_bytes(model, 100, "bf16", 1, 1), token_kv + recurrent)

    def test_spec_verification_does_not_multiply_fixed_recurrent_state_by_k(self):
        model = Model(
            "recurrent",
            "Recurrent",
            "Test",
            "#000",
            1,
            1,
            False,
            1,
            1,
            1,
            4,
            False,
            attention_layers=0,
            linear_attention_layers=1,
            linear_attention_heads=1,
            linear_attention_head_dim=4,
        )
        gpu = GPU("fast", "Fast", "nv", 1e12, 1e12, 1e30, 1e30, 1e12, 8)
        eff = EfficiencyParams(
            bw_eff=1.0, comp_eff=1.0, overhead=0.0, paged_oh=0.0, moe_imbalance=1.0
        )
        profile = SpeculativeProfile(
            "parallel",
            "dflash",
            0.0,
            0,
            True,
            4,
            0.8,
            0.0,
            "https://example.com",
            "test",
        )
        spec = SpecRuntime(profile, 4, 0.8, 1.0, 1, 0.0, 0.0)
        recurrent = per_replica_recurrent_state_bytes(model, "bf16", 1, 1)

        with (
            patch("calc.SPEC_DRAFT_LAUNCH_OVERHEAD_S", 0.0),
            patch("calc.SPEC_SCHEDULER_OVERHEAD_S", 0.0),
            patch("calc.SPEC_REJECTION_SYNC_OVERHEAD_S", 0.0),
        ):
            cycle = _decode_step_time(model, 1, 1, 1, gpu, "bf16", 100, eff, spec=spec)

        # No attention KV exists, so k=4 still incurs exactly one recurrent load
        # and one final store, plus one target-weight read.
        expected = (model.active_weight_bytes("bf16") + 2 * recurrent) / gpu.effective_bw
        self.assertAlmostEqual(cycle, expected)

    def test_attention_residual_scratch_is_tp_and_pp_sharded(self):
        model = Model(
            "attnres",
            "AttnRes",
            "Test",
            "#000",
            1,
            1,
            False,
            4,
            4,
            4,
            8,
            False,
            hidden_dim=8,
            attention_residual_block_size=2,
        )
        # Four layers in two blocks retain three residual sources. PP2 owns half
        # the layer sources and TP2 sequence-shards the token activations.
        expected = 3 * 100 * 8 * 2 * 0.5 / 2
        self.assertEqual(attention_residual_scratch_bytes(model, 100, "bf16", 2, 2), expected)

    def test_attention_residual_scratch_reduces_prefill_batch_fit(self):
        base = Model(
            "plain",
            "Plain",
            "Test",
            "#000",
            1,
            1,
            False,
            4,
            4,
            4,
            8,
            False,
            hidden_dim=8192,
        )
        attnres = replace(base, key="attnres-fit", attention_residual_block_size=1)
        gpu = GPU("small", "Small", "nv", 2e8, 1e12, 1e30, 1e30, 1e12, 8)
        eff = EfficiencyParams(overhead=0.0, paged_oh=0.0)
        plain = compute_prefill(base, 1, 1, 1, 1, 1024, gpu, 1.0, 0.0, "bf16", eff)
        guarded = compute_prefill(attnres, 1, 1, 1, 1, 1024, gpu, 1.0, 0.0, "bf16", eff)

        self.assertIsNotNone(plain)
        self.assertIsNotNone(guarded)
        self.assertLess(guarded.max_batch, plain.max_batch)

    def test_sparse_attention_bounds_selected_tokens_and_models_indexer_work(self):
        model = Model(
            "sparse",
            "Sparse",
            "Test",
            "#000",
            1,
            1,
            False,
            4,
            8,
            8,
            16,
            False,
            sparse_attention_top_k=32,
            sparse_indexer_heads=2,
            sparse_indexer_head_dim=4,
            sparse_indexer_layers=2,
        )
        pr = 3
        seq = 128
        selected_attention = 4 * pr * 4 * (8 * 16) * 32
        decode_indexer = 2 * pr * 2 * (2 * 4) * seq
        prefill_indexer = decode_indexer * seq

        self.assertEqual(
            _decode_attention_work(model, pr, seq, 1),
            selected_attention + decode_indexer,
        )
        self.assertEqual(
            _prefill_attention_work(model, pr, seq, 1),
            selected_attention * seq + prefill_indexer,
        )

        short_seq = 16
        short_selected_attention = 4 * pr * 4 * (8 * 16) * short_seq
        short_indexer = 2 * pr * 2 * (2 * 4) * short_seq
        self.assertEqual(
            _decode_attention_work(model, pr, short_seq, 1),
            short_selected_attention + short_indexer,
        )

    def test_local_and_global_kv_heads_are_sharded_independently(self):
        model = Model(
            "split-kv",
            "Split KV",
            "Test",
            "#000",
            1,
            1,
            False,
            2,
            8,
            8,
            128,
            False,
            local_attention_layers=1,
            local_attention_window=10,
            local_kv_heads=2,
            local_kv_head_dim=64,
            global_kv_heads=8,
            global_head_dim=128,
        )
        # TP4 shards the 8-head global cache four ways, but the 2-head local
        # cache only two ways and therefore duplicates it across two ranks.
        global_cache = 1000 * (2 * 8 * 128) / 4
        local_cache = 10 * (2 * 2 * 64) / 2
        self.assertEqual(
            per_replica_kv_cache_bytes(model, 1000, "bf16", 1, 4),
            (global_cache + local_cache) * 2,
        )

    def test_glm52_indexshare_matches_published_long_context_flop_reduction(self):
        glm51 = MODELS["glm51"]
        glm52 = MODELS["glm52"]
        seq = 1024 * 1024
        glm51_flops = 2 * glm51.active_params + _decode_attention_work(glm51, 1, seq, 1)
        glm52_flops = 2 * glm52.active_params + _decode_attention_work(glm52, 1, seq, 1)

        self.assertGreater(glm51_flops, glm52_flops)
        self.assertAlmostEqual(glm51_flops / glm52_flops, 2.9, delta=0.2)

    def test_dense_tp_models_two_all_reduces_per_layer(self):
        model = Model("tiny", "Tiny", "Test", "#000", 1, 1, False, 4, 4, 4, 8, False)
        gpu = GPU("gpu", "GPU", "nv", 1e12, 1e12, 1e12, 1e12, 200e9, 8)
        batch_tokens = 7
        tp = 2
        collective_bw = gpu.scale_up_collective_bw * self.eff.bw_eff
        msg = batch_tokens * model.hidden_size * 2
        per_collective = msg * 2 * (tp - 1) / (tp * collective_bw)
        expected = model.layers * 2 * (per_collective + 3e-6) * (1 - self.eff.ar_overlap)

        self.assertAlmostEqual(
            _dense_tp_oh(tp, 1, batch_tokens, model, gpu, self.eff.bw_eff, self.eff.ar_overlap),
            expected,
        )

    def test_moe_communication_discloses_unmodeled_expert_parallelism(self):
        model = Model("moe", "MoE", "Test", "#000", 8, 2, True, 2, 2, 2, 8, False)
        comm = communication_breakdown(model, 1, 1, 1, 128, GPUS["H100"], self.eff)

        self.assertTrue(comm.expert_parallel_unmodeled)
        self.assertFalse(comm.ep_advisory)

    def test_attention_counts_qk_and_av_matmuls(self):
        model = Model("tiny", "Tiny", "Test", "#000", 1, 1, False, 1, 1, 1, 8, False)

        self.assertEqual(_decode_attention_work(model, 3, 16, 1), 4 * 3 * 1 * 8 * 16)
        self.assertEqual(_prefill_attention_work(model, 3, 16, 1), 4 * 3 * 1 * 8 * 16 * 16)

    def test_decode_does_not_activate_idle_dp_replicas(self):
        model = MODELS["q08"]
        gpu = GPUS["H100"]
        dp1 = compute_decode(
            model, 1, 1, 1, 1, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff
        )
        dp8 = compute_decode(
            model, 1, 1, 1, 8, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff
        )

        self.assertIsNotNone(dp1)
        self.assertIsNotNone(dp8)
        self.assertEqual(dp8.tps, dp1.tps)
        self.assertEqual(dp8.step_ms, dp1.step_ms)
        self.assertEqual(dp8.max_slots, dp1.max_slots * 8)

    def test_decode_sums_uneven_replica_loads(self):
        model = MODELS["q08"]
        gpu = GPUS["H100"]
        one = compute_decode(
            model, 1, 1, 1, 1, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff
        )
        two = compute_decode(
            model, 1, 1, 2, 1, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff
        )
        nine_on_eight = compute_decode(
            model, 1, 1, 9, 8, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff
        )

        self.assertIsNotNone(one)
        self.assertIsNotNone(two)
        self.assertIsNotNone(nine_on_eight)
        self.assertAlmostEqual(nine_on_eight.tps, 7 * one.tps + two.tps, delta=5)
        self.assertEqual(nine_on_eight.step_ms, two.step_ms)

    def test_decode_latency_is_full_inter_token_step(self):
        result = compute_decode(
            MODELS["q08"],
            1,
            1,
            64,
            1,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            self.chat_in,
            self.chat_out,
            self.eff,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.lat, result.step_ms)
        self.assertAlmostEqual(result.lat, 1000.0 / (result.tps / 64), delta=0.02)

    def test_pipeline_decode_does_not_accelerate_each_user_with_concurrency(self):
        model = MODELS["kimi-k3"]
        gpu = GPUS["B300"]
        per_user_tps = []

        for users in (1, 2, 4, 8, 16, 32, 64):
            result = compute_decode(
                model,
                4,
                18,
                users,
                1,
                gpu,
                0.90,
                2.0,
                "bf16",
                self.chat_in,
                self.chat_out,
                self.eff,
            )
            self.assertIsNotNone(result)
            per_user_tps.append(result.tps / users)

        self.assertTrue(
            all(later <= earlier for earlier, later in zip(per_user_tps, per_user_tps[1:]))
        )

    def test_pipeline_fill_drain_prevents_batch_one_speedup(self):
        model = MODELS["q08"]
        gpu = GPUS["H100"]
        pp1 = compute_prefill(model, 1, 1, 1, 1, 8192, gpu, 0.90, 2.0, "bf16", self.eff)
        pp2 = compute_prefill(model, 1, 2, 1, 1, 8192, gpu, 0.90, 2.0, "bf16", self.eff)

        self.assertIsNotNone(pp1)
        self.assertIsNotNone(pp2)
        self.assertGreaterEqual(pp2.service_time, pp1.service_time)

    def test_pipeline_fit_uses_busiest_remainder_stage(self):
        model = Model("three", "Three layer", "Test", "#000", 9, 9, False, 3, 1, 1, 8, False)
        gpu = GPU("tiny", "Tiny", "nv", 10, 1e12, 1e12, 1e12, 1e12, 2)

        # Average PP2 weights would be 9 bytes/GPU, but the two-layer stage owns 12.
        self.assertNotIn((1, 2, 1), valid_strategies(model, 2, gpu, 1.0, 0.0, "bf16"))

    def test_context_limit_drives_capability_and_rejection(self):
        legacy = MODELS["mi7"]
        self.assertEqual(legacy.max_context_tokens, 32768)
        self.assertNotIn("ctx_128k", legacy.capabilities)
        self.assertIn("ctx_128k", MODELS["q08"].capabilities)
        self.assertIsNone(
            compute_prefill(legacy, 1, 1, 1, 1, 32769, GPUS["H100"], 0.90, 2.0, "bf16", self.eff)
        )

    def test_end_to_end_context_rejects_prompt_plus_output(self):
        legacy = MODELS["mi7"]
        result = compute_data(
            legacy,
            (1, 1, 1),
            (1, 1, 1),
            1,
            32_000,
            1_000,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            0.0,
            self.eff,
            self.eff,
        )
        self.assertIsNone(result)

    def test_full_prefix_hit_is_finite(self):
        result = compute_prefill(
            MODELS["q08"],
            1,
            1,
            8,
            1,
            0,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            self.eff,
        )

        self.assertIsNotNone(result)
        self.assertTrue(math.isfinite(result.rps))
        self.assertEqual(result.service_time, 0.0)

        end_to_end = compute_data(
            MODELS["q08"],
            (1, 1, 1),
            (1, 1, 1),
            8,
            2048,
            32,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            1.0,
            self.eff,
            self.eff,
        )
        self.assertIsNotNone(end_to_end)
        self.assertTrue(math.isfinite(end_to_end.rps))

    def test_independent_pd_layouts_are_rejected_without_separate_pools(self):
        args = (MODELS["q08"], (1, 1, 2), (2, 1, 1), 2, 2048, 32, GPUS["H100"], 0.90, 2.0, "bf16")

        self.assertIsNone(compute_data(*args, 0.0, self.eff, self.eff))
        self.assertEqual(
            compute_data_capacity(
                MODELS["q08"],
                (1, 1, 2),
                (2, 1, 1),
                2048,
                32,
                GPUS["H100"],
                0.90,
                2.0,
                "bf16",
                0.0,
                self.eff,
                self.eff,
            ),
            0,
        )

    def test_retune_co_locates_prefill_and_decode_layouts(self):
        assignment = ModelAssignment(
            2, "q08", 1, 2, 2, 1, "bf16", prefill_tp=1, prefill_pp=1, prefill_dp=2
        )
        state = PlannerState(gpus=[GpuPool(1, "H100", 2)], models=[assignment])

        self.assertEqual(get_deployed(state, "prefill"), [])
        self.assertEqual(get_deployed(state, "decode"), [])
        retune_models(state, preserve_existing=True)

        self.assertEqual(
            (assignment.prefill_tp, assignment.prefill_pp, assignment.prefill_dp),
            (assignment.tp, assignment.pp, assignment.dp),
        )
        self.assertEqual(len(get_deployed(state, "prefill")), 1)
        self.assertEqual(len(get_deployed(state, "decode")), 1)

    def test_embedding_pp_uses_busiest_stage_fraction(self):
        model = Model(
            "embed",
            "Embed",
            "Test",
            "#000",
            3e6,
            3e6,
            False,
            3,
            3,
            3,
            8,
            False,
            embedding_profile=EmbeddingProfile("Embed", "single", 8, 128, "test", "test"),
        )
        gpu = GPUS["H100"]
        eff = EfficiencyParams(overhead=0.0, paged_oh=0.0)
        seq = 16
        result = compute_embedding(model, (1, 2, 1), 1, seq, gpu, 1.0, 0.0, "bf16", eff)

        self.assertIsNotNone(result)
        pp_fraction = _pp_peak_fraction(model, 2)
        ffn = 2 * model.active_params * seq * pp_fraction
        att = _prefill_attention_work(model, 1, seq, 2)
        compute_time = (ffn + att) / (model_gpu_flops(gpu, model, "bf16") * eff.comp_eff)
        memory_time = (_active_weight_bytes(model, "bf16") * pp_fraction) / (
            gpu.effective_bw * eff.bw_eff
        )
        output_time = result.output_bytes_per_input / (gpu.effective_bw * eff.bw_eff)
        comm = communication_breakdown(model, 1, 2, seq, seq, gpu, eff)
        expected = (max(compute_time, memory_time) + output_time + comm.total) * (
            1 + fixed_paged_oh(seq, eff, 0.20)
        )
        self.assertAlmostEqual(result.service_time, expected)


class ProjectionMathTests(unittest.TestCase):
    def test_cloud_price_includes_expected_retry_attempts(self):
        profile = {"in_len": 1000, "out_len": 200, "tokens_per_request": 1200}
        info, effective_pm = _cloud_price_per_m_in_preset(0.10, 0.80, 0.0, profile, 0.0, "current")

        self.assertIsNotNone(info)
        raw_pm = (
            1000 * info["in_per_m"] + (200 / info["token_efficiency"]) * info["out_per_m"]
        ) / 1200
        expected_success = success_rate(info["quality"], 0.10)
        self.assertAlmostEqual(info["success_rate"], expected_success)
        self.assertAlmostEqual(effective_pm, raw_pm / expected_success)

    def test_cloud_long_context_tier_uses_input_threshold_and_absolute_prices(self):
        cloud = {
            "vendor": "Test",
            "quality": 1.0,
            "token_efficiency": 1.0,
            "capabilities": (),
            "in_per_m": 1.0,
            "cached_in_per_m": 0.5,
            "out_per_m": 2.0,
            "long_context_threshold_tokens": 1000,
            "long_context_in_per_m": 3.0,
            "long_context_cached_in_per_m": 1.5,
            "long_context_out_per_m": 4.0,
        }
        at_threshold = {"in_len": 1000, "out_len": 10000, "tokens_per_request": 11000}
        above_threshold = {"in_len": 1001, "out_len": 9999, "tokens_per_request": 11000}

        with patch("calc.cloud_policy.effective_corpo_models", return_value=[("tiered", cloud)]):
            normal, normal_pm = _cloud_price_per_m_in_preset(
                0.0, 0.0, 0.0, at_threshold, 0.5, "test"
            )
            tiered, tiered_pm = _cloud_price_per_m_in_preset(
                0.0, 0.0, 0.0, above_threshold, 0.5, "test"
            )

        self.assertFalse(normal["long_context_pricing_applied"])
        self.assertTrue(tiered["long_context_pricing_applied"])
        self.assertEqual(tiered["effective_in_per_m"], 3.0)
        self.assertEqual(tiered["effective_cached_in_per_m"], 1.5)
        self.assertEqual(tiered["effective_out_per_m"], 4.0)
        self.assertGreater(tiered_pm, normal_pm)

    def test_completed_retry_adjusted_work_is_not_value_discounted_twice(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 1, cost_per_gpu_hour=0.01)],
            models=[ModelAssignment(2, "q08", 1, 1, 1, 1, "bf16")],
            projects=[Project(3, "Easy", 0.10, 1_000_000, 100.0, min_success_rate=0.50)],
        )
        retune_models(state, preserve_existing=False)

        row = compute_revenue_projection(state, include_recommendations=False)["projects"][0]
        value_basis = row["wtp_per_m"] if row["cloud_blocked"] else row["cloud_pm"]
        self.assertAlmostEqual(row["value_served"], row["served"] / 1e6 * value_basis)

    def test_shape_specific_capacity_is_recomputed(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 1, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "q08", 1, 1, 1, 1, "bf16")],
        )
        retune_models(state, preserve_existing=False)
        am = state.models[0]
        gpu = state.gpus[0].gpu
        short = {"in_len": 256, "out_len": 32, "tokens_per_request": 288}
        long = {"in_len": 16_384, "out_len": 1024, "tokens_per_request": 17_408}

        short_cap, short_rps = _deployment_capacity_for_profile(state, am, gpu, short, 1.0)
        long_cap, long_rps = _deployment_capacity_for_profile(state, am, gpu, long, 1.0)
        self.assertGreater(short_cap, 0)
        self.assertGreater(long_cap, 0)
        self.assertNotEqual(short_cap, long_cap)
        self.assertGreater(short_rps, long_rps)

        prefill_heavy = {"in_len": 16_384, "out_len": 32, "tokens_per_request": 16_416}
        uncached_cap, _ = _deployment_capacity_for_profile(state, am, gpu, prefill_heavy, 1.0, 0.0)
        cached_cap, _ = _deployment_capacity_for_profile(state, am, gpu, prefill_heavy, 1.0, 0.8)
        self.assertGreater(cached_cap, uncached_cap)


class SpeculativeDecodingMathTests(unittest.TestCase):
    def setUp(self):
        self.eff = EfficiencyParams()
        self.chat_in = DIST_PRESETS["Chat"]["in"]
        self.chat_out = DIST_PRESETS["Chat"]["out"]

    def decode(self, model, tp, pp, bs, dp, gpu, prec, spec=None):
        return compute_decode(
            model,
            tp,
            pp,
            bs,
            dp,
            gpu,
            0.90,
            2.0,
            prec,
            self.chat_in,
            self.chat_out,
            self.eff,
            spec,
        )

    def test_acceptance_len_chain_formula(self):
        # DeepSeek-V3: 87.5% single-token acceptance -> 1.875 tokens per cycle.
        self.assertAlmostEqual(spec_acceptance_len(0.875, 1), 1.875)
        self.assertAlmostEqual(spec_acceptance_len(0.0, 5), 1.0)
        self.assertAlmostEqual(spec_acceptance_len(0.5, 3), (1 - 0.5**4) / 0.5)
        self.assertGreater(spec_acceptance_len(0.8, 3), spec_acceptance_len(0.6, 3))
        self.assertGreater(spec_acceptance_len(0.6, 5), spec_acceptance_len(0.6, 3))
        self.assertEqual(spec_acceptance_len(1.0, 3), 4.0)

    def test_finite_outputs_do_not_receive_asymptotic_spec_speedup(self):
        model = MODELS["l8"]
        eagle = resolve_spec_runtime(model, "eagle3", 0, 0.0, "bf16")
        one = _compute_decode_core(
            model,
            1,
            1,
            1,
            1,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            512,
            1,
            self.eff,
            spec=eagle,
        )
        short = _compute_decode_core(
            model,
            1,
            1,
            1,
            1,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            512,
            4,
            self.eff,
            spec=eagle,
        )
        long = _compute_decode_core(
            model,
            1,
            1,
            1,
            1,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            512,
            256,
            self.eff,
            spec=eagle,
        )
        self.assertEqual(spec_finite_output_tau(0.8, 3, 1), 1.0)
        self.assertEqual(one.spec_speedup, 1.0)
        self.assertLess(short.spec_speedup, long.spec_speedup)

    def test_spec_speedup_erodes_with_long_context_kv_verification(self):
        l8 = MODELS["l8"]
        eagle = resolve_spec_runtime(l8, "eagle3", 0, 0.0, "bf16")
        short = _compute_decode_core(
            l8,
            1,
            1,
            1,
            1,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            128,
            256,
            self.eff,
            spec=eagle,
        )
        long = _compute_decode_core(
            l8,
            1,
            1,
            1,
            1,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            65536,
            256,
            self.eff,
            spec=eagle,
        )
        self.assertLess(long.spec_speedup, short.spec_speedup)

    def test_resolve_spec_runtime_off_unknown_and_overrides(self):
        ds3 = MODELS["ds3"]
        self.assertIsNone(resolve_spec_runtime(ds3, "off", 0, 0.0, "fp8"))
        self.assertIsNone(resolve_spec_runtime(ds3, "bogus", 0, 0.0, "fp8"))

        mtp = resolve_spec_runtime(ds3, "mtp", 0, 0.0, "fp8")
        self.assertEqual(mtp.k, 1)
        self.assertEqual(mtp.passes, 1)
        self.assertAlmostEqual(mtp.alpha, 0.875)
        self.assertGreater(mtp.draft_weight_bytes, 0.0)

        # Explicit k drives autoregressive passes; alpha override replaces the profile value.
        eagle = resolve_spec_runtime(MODELS["l8"], "eagle3", 8, 0.5, "bf16")
        self.assertEqual(eagle.k, 8)
        self.assertEqual(eagle.passes, 8)
        self.assertAlmostEqual(eagle.alpha, 0.5)

        # Block-diffusion drafters emit the whole block in one pass at any k.
        dflash = resolve_spec_runtime(MODELS["q397"], "dflash", 16, 0.0, "fp8")
        self.assertEqual(dflash.passes, 1)
        self.assertEqual(dflash.k, 16)

    def test_spec_off_matches_baseline_exactly(self):
        l8 = MODELS["l8"]
        base = self.decode(l8, 1, 1, 4, 1, GPUS["H100"], "bf16")
        off = self.decode(
            l8, 1, 1, 4, 1, GPUS["H100"], "bf16", resolve_spec_runtime(l8, "off", 0, 0.0, "bf16")
        )
        self.assertEqual((base.tps, base.lat, base.max_slots), (off.tps, off.lat, off.max_slots))
        self.assertEqual(off.spec_tau, 0.0)
        self.assertEqual(off.spec_speedup, 1.0)

    def test_speedup_monotone_in_acceptance_alpha(self):
        l8 = MODELS["l8"]
        speedups = []
        for alpha in (0.4, 0.6, 0.8):
            spec = resolve_spec_runtime(l8, "eagle3", 0, alpha, "bf16")
            speedups.append(self.decode(l8, 1, 1, 1, 1, GPUS["H100"], "bf16", spec).spec_speedup)
        self.assertLess(speedups[0], speedups[1])
        self.assertLess(speedups[1], speedups[2])
        self.assertGreater(speedups[0], 1.0)

    def test_ngram_adds_zero_memory_and_may_slow_unrepetitive_workloads(self):
        l8 = MODELS["l8"]
        ngram = resolve_spec_runtime(l8, "ngram", 0, 0.0, "bf16")
        self.assertEqual(ngram.draft_weight_bytes, 0.0)
        self.assertEqual(ngram.profile.kv_overhead, 0.0)

        base_mem = compute_memory(l8, 1, 1, GPUS["H100"], 0.90, 2.0, "bf16", self.eff)
        spec_mem = compute_memory(l8, 1, 1, GPUS["H100"], 0.90, 2.0, "bf16", self.eff, ngram)
        self.assertEqual(base_mem.weights, spec_mem.weights)
        self.assertEqual(base_mem.kv_budget, spec_mem.kv_budget)

        base = self.decode(l8, 1, 1, 4, 1, GPUS["H100"], "bf16")
        sp = self.decode(l8, 1, 1, 4, 1, GPUS["H100"], "bf16", ngram)
        self.assertEqual(base.max_slots, sp.max_slots)
        # The profile's low cross-workload acceptance prior is not guaranteed
        # to beat baseline once k-position verification work is charged.
        self.assertNotEqual(sp.tps, base.tps)

    def test_draft_weights_and_draft_kv_shrink_slots(self):
        ds3 = MODELS["ds3"]
        mtp = resolve_spec_runtime(ds3, "mtp", 0, 0.0, "fp8")
        base_mem = compute_memory(ds3, 8, 1, GPUS["B200"], 0.90, 2.0, "fp8", self.eff)
        spec_mem = compute_memory(ds3, 8, 1, GPUS["B200"], 0.90, 2.0, "fp8", self.eff, mtp)
        self.assertAlmostEqual(
            spec_mem.weights - base_mem.weights,
            mtp.draft_weight_bytes / 8,
            delta=1.0,
        )

        base = self.decode(ds3, 8, 1, 1, 1, GPUS["B200"], "fp8")
        sp = self.decode(ds3, 8, 1, 1, 1, GPUS["B200"], "fp8", mtp)
        self.assertLess(sp.max_slots, base.max_slots)

    def test_speedup_erodes_as_batch_turns_compute_bound(self):
        l8 = MODELS["l8"]
        eagle = resolve_spec_runtime(l8, "eagle3", 0, 0.0, "bf16")
        low = self.decode(l8, 8, 1, 8, 8, GPUS["H100"], "bf16", eagle)
        high = self.decode(l8, 8, 1, 1024, 8, GPUS["H100"], "bf16", eagle)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertGreater(low.spec_speedup, 1.5)
        self.assertLess(high.spec_speedup, low.spec_speedup)

    def test_ds3_mtp_parity_with_vendor_claim(self):
        # DeepSeek reports 1.8x TPS from MTP speculative decoding; the planner
        # should land in that neighborhood at batch 1, not at the batch-1 marketing
        # numbers of tree drafters.
        ds3 = MODELS["ds3"]
        mtp = resolve_spec_runtime(ds3, "mtp", 0, 0.0, "fp8")
        result = self.decode(ds3, 8, 1, 1, 1, GPUS["B200"], "fp8", mtp)
        # Finite-response waste and full KV/collective verification accounting
        # make this more conservative than the vendor's long-run TPS headline.
        self.assertGreaterEqual(result.spec_speedup, 1.4)
        self.assertLessEqual(result.spec_speedup, 1.95)

    def test_compute_data_carries_spec_speedup(self):
        l8 = MODELS["l8"]
        eagle = resolve_spec_runtime(l8, "eagle3", 0, 0.0, "bf16")
        base = compute_data(
            l8,
            (1, 1, 1),
            (1, 1, 1),
            4,
            512,
            256,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            0.0,
            self.eff,
            self.eff,
        )
        sp = compute_data(
            l8,
            (1, 1, 1),
            (1, 1, 1),
            4,
            512,
            256,
            GPUS["H100"],
            0.90,
            2.0,
            "bf16",
            0.0,
            self.eff,
            self.eff,
            eagle,
        )
        self.assertGreater(sp.rps, base.rps)
        self.assertGreater(sp.tps, base.tps)

    def test_auto_k_searches_supported_calibrated_depths_at_declared_probe(self):
        base_model = MODELS["l8"]
        profile = replace(
            base_model.available_spec_profiles[0],
            supported_ks=(1, 3),
            acceptance_alpha_by_k=((1, 0.45), (3, 0.75)),
        )
        model = replace(base_model, speculative_profiles=(profile,))
        selection = optimize_spec_k(
            model,
            "eagle3",
            0.0,
            "bf16",
            1,
            1,
            1,
            GPUS["H100"],
            0.90,
            2.0,
            self.chat_in,
            self.chat_out,
            self.eff,
            probe_concurrency=7,
        )

        self.assertIsNotNone(selection.runtime)
        self.assertIn(selection.selected_k, (1, 3))
        self.assertEqual(selection.probe_concurrency, 7)
        self.assertTrue(selection.runtime.auto_selected)
        self.assertAlmostEqual(
            selection.runtime.alpha,
            dict(profile.acceptance_alpha_by_k)[selection.selected_k],
        )

    def test_auto_k_bounds_search_by_finite_output_length(self):
        selection = optimize_spec_k(
            MODELS["l8"],
            "eagle3",
            0.0,
            "bf16",
            1,
            1,
            1,
            GPUS["H100"],
            0.90,
            2.0,
            self.chat_in,
            [100, 0, 0, 0, 0, 0],
            self.eff,
        )

        self.assertIsNotNone(selection.runtime)
        self.assertLessEqual(selection.selected_k, 31)

    def test_auto_k_reports_when_spec_off_is_better(self):
        selection = optimize_spec_k(
            MODELS["l8"],
            "eagle3",
            0.001,
            "bf16",
            1,
            1,
            1,
            GPUS["H100"],
            0.90,
            2.0,
            self.chat_in,
            self.chat_out,
            self.eff,
        )

        self.assertIsNotNone(selection.runtime)
        self.assertFalse(selection.beneficial)
        self.assertLess(selection.speedup, 1.0)
        self.assertIn("spec off", selection.reason)

    def test_spec_cycle_charges_fixed_runtime_overheads(self):
        model = MODELS["l8"]
        runtime = resolve_spec_runtime(model, "eagle3", 3, 0.7, "bf16")
        args = (model, 1, 1, 1, GPUS["H100"], "bf16", 1024, self.eff)
        charged = _decode_step_time(*args, spec=runtime)
        with (
            patch("calc.SPEC_DRAFT_LAUNCH_OVERHEAD_S", 0.0),
            patch("calc.SPEC_SCHEDULER_OVERHEAD_S", 0.0),
            patch("calc.SPEC_REJECTION_SYNC_OVERHEAD_S", 0.0),
        ):
            without_fixed_cost = _decode_step_time(*args, spec=runtime)

        self.assertGreater(charged, without_fixed_cost)

    def test_user_pareto_discloses_fixed_auto_spec_configuration(self):
        assignment = ModelAssignment(
            2,
            "l8",
            1,
            1,
            1,
            1,
            "bf16",
            spec_method="eagle3",
            spec_k=0,
        )
        state = PlannerState(gpus=[GpuPool(1, "H100", 1)], models=[assignment])
        retune_models(state, preserve_existing=True)
        datasets = chart_user_pareto(state, [1, 4])

        self.assertEqual(len(datasets), 1)
        dataset = datasets[0]
        self.assertIn("Auto", dataset["label"])
        self.assertTrue(dataset["spec_auto"])
        self.assertGreater(dataset["spec_k"], 0)
        self.assertIn("spec_alpha", dataset)
        self.assertTrue(all(point["spec_k"] == dataset["spec_k"] for point in dataset["data"]))


if __name__ == "__main__":
    unittest.main()
