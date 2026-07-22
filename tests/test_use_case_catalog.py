import copy
import unittest

import app as app_module
from data import MODELS, PROJECT_PRESETS, effective_quality, required_quality
from state import _normalize_use_case_def
from use_case_evidence import (
    USE_CASE_EVIDENCE,
    USE_CASE_RESEARCH_CAPTURED_AT,
    USE_CASE_SOURCES,
)


class UseCaseCatalogTests(unittest.TestCase):
    def test_catalog_is_expanded_and_scale_math_is_internally_consistent(self):
        self.assertGreaterEqual(len(PROJECT_PRESETS), 19)
        keys = [preset["key"] for preset in PROJECT_PRESETS]
        self.assertEqual(len(keys), len(set(keys)))

        for preset in PROJECT_PRESETS:
            with self.subTest(key=preset["key"]):
                expected = preset["scale_value"] * preset["scale_kind"]["token_multiplier"]
                self.assertAlmostEqual(preset["tokens_day"], expected)
                self.assertTrue(preset["scale_kind"]["formula"])
                self.assertTrue(preset["scale_hint"])

    def test_every_scenario_has_resolvable_dated_evidence(self):
        preset_keys = {preset["key"] for preset in PROJECT_PRESETS}
        self.assertEqual(set(USE_CASE_EVIDENCE), preset_keys)
        self.assertEqual(USE_CASE_RESEARCH_CAPTURED_AT, "2026-07-22")

        for key, evidence in USE_CASE_EVIDENCE.items():
            with self.subTest(key=key):
                self.assertIn(evidence["confidence"], {"low", "medium", "high"})
                self.assertTrue(evidence["assumption"])
                self.assertTrue(evidence["source_ids"])
                for source_id in evidence["source_ids"]:
                    self.assertIn(source_id, USE_CASE_SOURCES)

        for source_id, source in USE_CASE_SOURCES.items():
            with self.subTest(source=source_id):
                self.assertTrue(source["url"].startswith("https://"))
                self.assertTrue(source["title"])
                self.assertTrue(source["publisher"])
                self.assertTrue(source["published"])

    def test_every_preset_has_at_least_one_eligible_local_model(self):
        for preset in PROJECT_PRESETS:
            threshold = required_quality(
                preset["difficulty"],
                preset["min_success_rate"],
                quality_floor=preset.get("quality_floor", 0.0),
            )
            required_caps = set(preset.get("requires", ()))
            eligible = [
                model
                for model in MODELS.values()
                if not model.hidden
                and not model.is_realtime_only
                and not model.is_embedding_model
                and required_caps <= model.capabilities
                and effective_quality(model) >= threshold
            ]
            with self.subTest(key=preset["key"], threshold=threshold):
                self.assertTrue(eligible)

    def test_evidence_is_hidden_after_edit_or_key_collision(self):
        canonical = _normalize_use_case_def(PROJECT_PRESETS[0])
        self.assertTrue(app_module.use_case_detail_for(canonical).get("source_ids"))

        edited = copy.deepcopy(canonical)
        edited["wtp_per_m"] += 1.0
        self.assertEqual(app_module.use_case_detail_for(edited), {})

        collision = copy.deepcopy(canonical)
        collision["name"] = "Imported custom definition"
        self.assertEqual(app_module.use_case_detail_for(collision), {})

    def test_use_case_page_renders_evidence_and_new_presets(self):
        app_module.app.config.update(TESTING=True)
        response = app_module.app.test_client().get("/use-cases")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Evidence &amp; assumptions", response.data)
        self.assertIn(b"Invoice &amp; claims extraction", response.data)
        self.assertIn(b"illustrative large-enterprise scenarios", response.data)


if __name__ == "__main__":
    unittest.main()
