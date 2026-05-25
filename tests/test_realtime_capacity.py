import copy
import unittest

from calc import chart_realtime_capacity, compute_realtime_capacity, compute_revenue_projection
from data import DIST_PRESETS, MODELS
from state import GpuPool, ModelAssignment, PlannerState, Project, retune_models


class RealtimeCapacityTests(unittest.TestCase):
    def _state(self) -> PlannerState:
        state = PlannerState(
            gpus=[GpuPool(1, "H100", 1, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "voxtral-realtime-mini-4b", 1, 1, 1, 1, "bf16")],
        )
        retune_models(state, preserve_existing=False)
        return state

    def test_voxtral_realtime_capacity_ignores_use_case_distribution(self):
        state = self._state()
        am = state.models[0]
        gpu = state.gpus[0].gpu
        model = MODELS[am.model_key]

        base = compute_realtime_capacity(
            model,
            (am.tp, am.pp, am.dp),
            8,
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )

        state.in_dist = copy.deepcopy(DIST_PRESETS["Long doc"]["in"])
        state.out_dist = copy.deepcopy(DIST_PRESETS["Long doc"]["out"])
        shifted = compute_realtime_capacity(
            model,
            (am.tp, am.pp, am.dp),
            8,
            gpu,
            state.mu,
            state.profiled_non_kv_gb,
            am.prec,
            state.decode_efficiency,
        )

        self.assertIsNotNone(base)
        self.assertIsNotNone(shifted)
        self.assertEqual(base.realtime_factor, shifted.realtime_factor)
        self.assertEqual(base.max_slots, shifted.max_slots)

    def test_realtime_chart_reports_concurrent_stream_capacity(self):
        state = self._state()

        datasets = chart_realtime_capacity(state, [1, 8, 16])

        self.assertEqual(len(datasets), 1)
        point = datasets[0]["data"][1]
        self.assertEqual(point["users"], 8)
        self.assertGreater(point["required_tps"], 0.0)
        self.assertIn("max_users", point)

    def test_realtime_models_do_not_enter_use_case_projection_supply(self):
        state = self._state()
        state.projects = [
            Project(
                3,
                "Easy classification",
                difficulty=0.10,
                tokens_day=10_000_000,
                wtp_per_m=1.0,
                min_success_rate=0.80,
                quality_floor=0.0,
            )
        ]

        projection = compute_revenue_projection(state)

        self.assertFalse(projection["has_supply"])
        self.assertEqual(projection["models"], [])


if __name__ == "__main__":
    unittest.main()
