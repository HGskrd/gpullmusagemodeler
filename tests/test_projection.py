import unittest

from calc import compute_revenue_projection, latent_activation_share
from state import GpuPool, ModelAssignment, PlannerState, Project


class RevenueProjectionTests(unittest.TestCase):
    def test_projection_exposes_distinct_coverage_metrics(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "q27", 1, 1, 1, 1, "bf16")],
            projects=[
                Project(
                    3,
                    "Coding",
                    difficulty=0.55,
                    tokens_day=20_000_000,
                    wtp_per_m=4.0,
                    min_success_rate=0.80,
                    quality_floor=0.60,
                )
            ],
        )

        projection = compute_revenue_projection(state)

        self.assertIn("token_coverage", projection)
        self.assertIn("value_capture_rate", projection)
        self.assertIn("revenue_multiple", projection)
        self.assertEqual(projection["coverage"], projection["revenue_multiple"])
        self.assertGreaterEqual(projection["token_coverage"], 0.0)
        self.assertLessEqual(projection["token_coverage"], 1.0)

    def test_zero_capacity_assignment_is_not_runnable(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 1, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "q122", 1, 1, 1, 1, "bf16")],
            projects=[
                Project(
                    3,
                    "Long context",
                    difficulty=0.70,
                    tokens_day=10_000_000,
                    wtp_per_m=8.0,
                    requires=frozenset({"ctx_128k"}),
                    min_success_rate=0.80,
                    quality_floor=0.70,
                )
            ],
        )

        projection = compute_revenue_projection(state)

        self.assertEqual(projection["models"][0]["daily_tokens_cap"], 0.0)
        self.assertFalse(projection["models"][0]["runnable"])
        self.assertEqual(projection["models"][0]["status"], "NOT RUNNABLE")

    def test_latent_activation_is_smooth_around_unlock_price(self):
        self.assertAlmostEqual(latent_activation_share(1.0, 1.0), 0.5)
        self.assertGreater(latent_activation_share(0.5, 1.0), 0.95)
        self.assertLess(latent_activation_share(2.0, 1.0), 0.15)

    def test_latent_demand_is_reported_separately_from_baseline(self):
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 2, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "q27", 1, 1, 1, 1, "bf16")],
            projects=[
                Project(
                    3,
                    "Summaries",
                    difficulty=0.25,
                    tokens_day=10_000_000,
                    wtp_per_m=4.0,
                    min_success_rate=0.80,
                    quality_floor=0.60,
                    latent_jobs_day=10_000_000,
                    unlock_price_per_m=10.0,
                )
            ],
        )

        projection = compute_revenue_projection(state)

        self.assertEqual(projection["baseline_tokens_day"], 10_000_000)
        self.assertGreater(projection["latent_active_tokens_day"], 0.0)
        self.assertAlmostEqual(
            projection["fates"]["total_tokens"],
            projection["baseline_tokens_day"] + projection["latent_active_tokens_day"],
        )


if __name__ == "__main__":
    unittest.main()
