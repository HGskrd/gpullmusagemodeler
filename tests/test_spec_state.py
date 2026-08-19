import unittest
from types import SimpleNamespace

from data import MODELS
from placement import get_deployed
from scenarios import deserialize_scenario, serialize_scenario
from state import (
    GpuPool,
    ModelAssignment,
    PlannerState,
    set_model_spec,
    set_spec_acceptance,
)
from viewmodels import get_model_info


def _fake_profile(method, supported_ks=()):
    return SimpleNamespace(
        label=f"Fake {method}",
        method=method,
        draft_params=1e9,
        draft_layers=1,
        parallel_draft=False,
        default_k=3,
        acceptance_alpha=0.7,
        kv_overhead=0.0,
        source="test",
        note="test fixture",
        supported_ks=supported_ks,
        acceptance_alpha_by_k=(),
    )


def _state_with_model(model_key="q08"):
    return PlannerState(
        gpus=[GpuPool(1, "H100", 4, cost_per_gpu_hour=1.0)],
        models=[ModelAssignment(2, model_key, 1, 1, 1, 1, "bf16")],
    )


class SpecStateTests(unittest.TestCase):
    def _patch_profiles(self, model_key, profiles):
        model = MODELS[model_key]
        had = hasattr(model, "speculative_profiles")
        old = getattr(model, "speculative_profiles", None)
        model.speculative_profiles = profiles

        def restore():
            if had:
                model.speculative_profiles = old
            else:
                try:
                    del model.speculative_profiles
                except AttributeError:
                    pass

        self.addCleanup(restore)

    def test_set_model_spec_accepts_valid_method(self):
        self._patch_profiles("q08", (_fake_profile("eagle3"),))
        state = _state_with_model()

        set_model_spec(state, 2, "eagle3", 4)

        self.assertEqual(state.models[0].spec_method, "eagle3")
        self.assertEqual(state.models[0].spec_k, 4)

    def test_set_model_spec_preserves_auto_k_sentinel(self):
        self._patch_profiles("q08", (_fake_profile("eagle3"),))
        state = _state_with_model()

        set_model_spec(state, 2, "eagle3", 0)

        self.assertEqual(state.models[0].spec_method, "eagle3")
        self.assertEqual(state.models[0].spec_k, 0)

    def test_auto_k_viewmodel_discloses_effective_choice_without_mutating_state(self):
        self._patch_profiles("q08", (_fake_profile("eagle3"),))
        state = _state_with_model()
        set_model_spec(state, 2, "eagle3", 0)

        info = get_model_info(state, state.models[0])

        self.assertEqual(state.models[0].spec_k, 0)
        self.assertIsNotNone(info["spec"])
        self.assertGreaterEqual(info["spec"]["k"], 1)
        self.assertIn("alpha_source", info["spec"])
        self.assertIn("speedup", info["spec"])
        self.assertEqual(info["spec"]["probe_bs"], 32)

    def test_set_model_spec_rejects_unknown_method(self):
        self._patch_profiles("q08", (_fake_profile("eagle3"),))
        state = _state_with_model()

        set_model_spec(state, 2, "mtp", 4)

        self.assertEqual(state.models[0].spec_method, "off")
        self.assertEqual(state.models[0].spec_k, 0)

    def test_invalid_method_disables_a_previously_enabled_drafter(self):
        self._patch_profiles("q08", (_fake_profile("eagle3"),))
        state = _state_with_model()
        set_model_spec(state, 2, "eagle3", 4)

        set_model_spec(state, 2, "not-a-method", 7)

        self.assertEqual(state.models[0].spec_method, "off")
        self.assertEqual(state.models[0].spec_k, 0)

    def test_enabling_drafter_retunes_topology_for_resident_memory(self):
        state = PlannerState(
            gpus=[GpuPool(1, "A10", 2, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "l8", 1, 2, 1, 2, "bf16")],
        )
        am = state.models[0]
        am.tp, am.pp, am.dp = 1, 1, 2
        am.prefill_tp, am.prefill_pp, am.prefill_dp = 1, 1, 2

        set_model_spec(state, 2, "eagle3", 0)

        self.assertNotEqual((am.tp, am.pp, am.dp), (1, 1, 2))
        self.assertEqual((am.prefill_tp, am.prefill_pp, am.prefill_dp), (am.tp, am.pp, am.dp))
        self.assertEqual(len(get_deployed(state)), 1)

    def test_drafter_is_rejected_when_no_topology_can_hold_it(self):
        state = PlannerState(
            gpus=[GpuPool(1, "A10", 1, cost_per_gpu_hour=1.0)],
            models=[ModelAssignment(2, "l8", 1, 1, 1, 1, "bf16")],
        )

        set_model_spec(state, 2, "eagle3", 0)

        self.assertEqual(state.models[0].spec_method, "off")
        self.assertEqual(len(get_deployed(state)), 1)

    def test_set_model_spec_off_always_allowed(self):
        state = _state_with_model()
        state.models[0].spec_method = "eagle3"
        state.models[0].spec_k = 4

        set_model_spec(state, 2, "off", 0)

        self.assertEqual(state.models[0].spec_method, "off")

    def test_set_model_spec_ngram_allowed_without_profiles(self):
        state = _state_with_model()

        set_model_spec(state, 2, "ngram", 0)

        self.assertEqual(state.models[0].spec_method, "ngram")

    def test_set_model_spec_ngram_rejected_for_non_text_models(self):
        state = _state_with_model("voxtral-realtime-mini-4b")

        set_model_spec(state, 2, "ngram", 0)

        self.assertEqual(state.models[0].spec_method, "off")

    def test_set_model_spec_clamps_spec_k(self):
        self._patch_profiles("q08", (_fake_profile("eagle3"),))
        state = _state_with_model()

        set_model_spec(state, 2, "eagle3", 100)
        self.assertEqual(state.models[0].spec_k, 32)

        set_model_spec(state, 2, "eagle3", -5)
        self.assertEqual(state.models[0].spec_k, 0)

    def test_set_model_spec_rejects_unsupported_manual_k_without_snapping(self):
        self._patch_profiles("q08", (_fake_profile("eagle3", supported_ks=(3, 7, 15)),))
        state = _state_with_model()
        set_model_spec(state, 2, "eagle3", 7)

        set_model_spec(state, 2, "eagle3", 8)

        self.assertEqual(state.models[0].spec_method, "eagle3")
        self.assertEqual(state.models[0].spec_k, 7)

    def test_set_spec_acceptance_clamps(self):
        state = PlannerState()

        set_spec_acceptance(state, 0.65)
        self.assertAlmostEqual(state.spec_acceptance, 0.65)

        set_spec_acceptance(state, 1.5)
        self.assertAlmostEqual(state.spec_acceptance, 0.99)

        set_spec_acceptance(state, -0.2)
        self.assertAlmostEqual(state.spec_acceptance, 0.0)

    def test_scenario_round_trip_preserves_spec_fields(self):
        self._patch_profiles("q08", (_fake_profile("eagle3"),))
        state = _state_with_model()
        set_model_spec(state, 2, "eagle3", 5)
        set_spec_acceptance(state, 0.7)

        payload = serialize_scenario(state, None)
        restored, restored_b = deserialize_scenario(payload)

        self.assertIsNone(restored_b)
        self.assertEqual(restored.models[0].spec_method, "eagle3")
        self.assertEqual(restored.models[0].spec_k, 5)
        self.assertAlmostEqual(restored.spec_acceptance, 0.7)

    def test_scenario_round_trip_preserves_auto_k_sentinel(self):
        self._patch_profiles("q08", (_fake_profile("eagle3"),))
        state = _state_with_model()
        set_model_spec(state, 2, "eagle3", 0)

        payload = serialize_scenario(state, None)
        restored, _ = deserialize_scenario(payload)

        self.assertEqual(restored.models[0].spec_method, "eagle3")
        self.assertEqual(restored.models[0].spec_k, 0)

    def test_imported_unsupported_manual_k_migrates_to_auto(self):
        state = PlannerState(
            models=[
                ModelAssignment(2, "q27", 0, 1, 1, 1, "bf16", spec_method="mtp", spec_k=5),
            ]
        )

        self.assertEqual(state.models[0].spec_method, "mtp")
        self.assertEqual(state.models[0].spec_k, 0)

    def test_legacy_payload_without_spec_keys_loads_defaults(self):
        state = _state_with_model()
        payload = serialize_scenario(state, None)
        for row in payload["panel_a"]["models"]:
            row.pop("spec_method", None)
            row.pop("spec_k", None)
        payload["panel_a"].pop("spec_acceptance", None)

        restored, _ = deserialize_scenario(payload)

        self.assertEqual(restored.models[0].spec_method, "off")
        self.assertEqual(restored.models[0].spec_k, 0)
        self.assertAlmostEqual(restored.spec_acceptance, 0.0)


if __name__ == "__main__":
    unittest.main()
