import unittest

from calc import (
    EfficiencyParams,
    _cloud_price_per_m_in_preset,
    _decode_attention_work,
    _deployment_capacity_for_profile,
    _prefill_attention_work,
    compute_data,
    compute_decode,
    compute_prefill,
    compute_revenue_projection,
    kv_cache_bytes_for_sequence,
    valid_strategies,
)
from data import DIST_PRESETS, GPUS, MODELS, GPU, Model, success_rate
from state import GpuPool, ModelAssignment, PlannerState, Project, retune_models


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
