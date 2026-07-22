import math
import unittest

from calc import (
    EfficiencyParams,
    _active_weight_bytes,
    _cloud_price_per_m_in_preset,
    _decode_attention_work,
    _dense_tp_oh,
    _deployment_capacity_for_profile,
    _pp_peak_fraction,
    _prefill_attention_work,
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
    per_tp_linear_attention_state_bytes,
    valid_strategies,
)
from data import DIST_PRESETS, GPUS, MODELS, EmbeddingProfile, GPU, Model, success_rate
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
            "linear", "Linear", "Test", "#000", 1, 1, False, 2, 4, 1, 8, False,
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

    def test_dense_tp_models_two_all_reduces_per_layer(self):
        model = Model("tiny", "Tiny", "Test", "#000", 1, 1, False, 4, 4, 4, 8, False)
        gpu = GPU("gpu", "GPU", "nv", 1e12, 1e12, 1e12, 1e12, 200e9, 8)
        batch_tokens = 7
        tp = 2
        collective_bw = gpu.scale_up_collective_bw * self.eff.bw_eff
        msg = batch_tokens * model.hidden_size * 2
        per_collective = msg * 2 * (tp - 1) / (tp * collective_bw)
        expected = model.layers * 2 * (per_collective + 3e-6) * (1 - self.eff.ar_overlap)

        self.assertAlmostEqual(_dense_tp_oh(tp, 1, batch_tokens, model, gpu, self.eff.bw_eff, self.eff.ar_overlap), expected)

    def test_attention_counts_qk_and_av_matmuls(self):
        model = Model("tiny", "Tiny", "Test", "#000", 1, 1, False, 1, 1, 1, 8, False)

        self.assertEqual(_decode_attention_work(model, 3, 16, 1), 4 * 3 * 1 * 8 * 16)
        self.assertEqual(_prefill_attention_work(model, 3, 16, 1), 4 * 3 * 1 * 8 * 16 * 16)

    def test_decode_does_not_activate_idle_dp_replicas(self):
        model = MODELS["q08"]
        gpu = GPUS["H100"]
        dp1 = compute_decode(model, 1, 1, 1, 1, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff)
        dp8 = compute_decode(model, 1, 1, 1, 8, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff)

        self.assertIsNotNone(dp1)
        self.assertIsNotNone(dp8)
        self.assertEqual(dp8.tps, dp1.tps)
        self.assertEqual(dp8.step_ms, dp1.step_ms)
        self.assertEqual(dp8.max_slots, dp1.max_slots * 8)

    def test_decode_sums_uneven_replica_loads(self):
        model = MODELS["q08"]
        gpu = GPUS["H100"]
        one = compute_decode(model, 1, 1, 1, 1, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff)
        two = compute_decode(model, 1, 1, 2, 1, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff)
        nine_on_eight = compute_decode(model, 1, 1, 9, 8, gpu, 0.90, 2.0, "bf16", self.chat_in, self.chat_out, self.eff)

        self.assertIsNotNone(one)
        self.assertIsNotNone(two)
        self.assertIsNotNone(nine_on_eight)
        self.assertAlmostEqual(nine_on_eight.tps, 7 * one.tps + two.tps, delta=5)
        self.assertEqual(nine_on_eight.step_ms, two.step_ms)

    def test_decode_latency_is_full_inter_token_step(self):
        result = compute_decode(
            MODELS["q08"], 1, 1, 64, 1, GPUS["H100"], 0.90, 2.0, "bf16",
            self.chat_in, self.chat_out, self.eff,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.lat, result.step_ms)
        self.assertAlmostEqual(result.lat, 1000.0 / (result.tps / 64), delta=0.02)

    def test_pipeline_decode_does_not_accelerate_each_user_with_concurrency(self):
        model = MODELS["kimi-k3-preview"]
        gpu = GPUS["B300"]
        per_user_tps = []

        for users in (1, 2, 4, 8, 16, 32, 64):
            result = compute_decode(
                model, 4, 18, users, 1, gpu, 0.90, 2.0, "bf16",
                self.chat_in, self.chat_out, self.eff,
            )
            self.assertIsNotNone(result)
            per_user_tps.append(result.tps / users)

        self.assertTrue(all(
            later <= earlier
            for earlier, later in zip(per_user_tps, per_user_tps[1:])
        ))

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
            legacy, (1, 1, 1), (1, 1, 1), 1, 32_000, 1_000, GPUS["H100"],
            0.90, 2.0, "bf16", 0.0, self.eff, self.eff,
        )
        self.assertIsNone(result)

    def test_full_prefix_hit_is_finite(self):
        result = compute_prefill(
            MODELS["q08"], 1, 1, 8, 1, 0, GPUS["H100"], 0.90, 2.0, "bf16", self.eff,
        )

        self.assertIsNotNone(result)
        self.assertTrue(math.isfinite(result.rps))
        self.assertEqual(result.service_time, 0.0)

        end_to_end = compute_data(
            MODELS["q08"], (1, 1, 1), (1, 1, 1), 8, 2048, 32, GPUS["H100"],
            0.90, 2.0, "bf16", 1.0, self.eff, self.eff,
        )
        self.assertIsNotNone(end_to_end)
        self.assertTrue(math.isfinite(end_to_end.rps))

    def test_independent_pd_layouts_are_rejected_without_separate_pools(self):
        args = (MODELS["q08"], (1, 1, 2), (2, 1, 1), 2, 2048, 32, GPUS["H100"], 0.90, 2.0, "bf16")

        self.assertIsNone(compute_data(*args, 0.0, self.eff, self.eff))
        self.assertEqual(
            compute_data_capacity(
                MODELS["q08"], (1, 1, 2), (2, 1, 1), 2048, 32, GPUS["H100"],
                0.90, 2.0, "bf16", 0.0, self.eff, self.eff,
            ),
            0,
        )

    def test_retune_co_locates_prefill_and_decode_layouts(self):
        assignment = ModelAssignment(2, "q08", 1, 2, 2, 1, "bf16", prefill_tp=1, prefill_pp=1, prefill_dp=2)
        state = PlannerState(gpus=[GpuPool(1, "H100", 2)], models=[assignment])

        self.assertEqual(get_deployed(state, "prefill"), [])
        self.assertEqual(get_deployed(state, "decode"), [])
        retune_models(state, preserve_existing=True)

        self.assertEqual((assignment.prefill_tp, assignment.prefill_pp, assignment.prefill_dp), (assignment.tp, assignment.pp, assignment.dp))
        self.assertEqual(len(get_deployed(state, "prefill")), 1)
        self.assertEqual(len(get_deployed(state, "decode")), 1)

    def test_embedding_pp_uses_busiest_stage_fraction(self):
        model = Model(
            "embed", "Embed", "Test", "#000", 3e6, 3e6, False, 3, 3, 3, 8, False,
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
        memory_time = (_active_weight_bytes(model, "bf16") * pp_fraction) / (gpu.effective_bw * eff.bw_eff)
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
        raw_pm = (1000 * info["in_per_m"] + (200 / info["token_efficiency"]) * info["out_per_m"]) / 1200
        expected_success = success_rate(info["quality"], 0.10)
        self.assertAlmostEqual(info["success_rate"], expected_success)
        self.assertAlmostEqual(effective_pm, raw_pm / expected_success)

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


if __name__ == "__main__":
    unittest.main()
