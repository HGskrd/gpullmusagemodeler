import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import cloud_policy
import app as app_module
from calc import _cloud_price_per_m_in_preset
from state import normalize_corpo_cloud


class CloudPolicyTests(unittest.TestCase):
    def tearDown(self):
        cloud_policy.configure(None)

    def test_allowlist_and_negotiated_prices_reach_projection_math(self):
        policy = cloud_policy.validate_policy({
            "allowed_models": ["gemini-flash-lite"],
            "price_overrides": {
                "gemini-flash-lite": {
                    "in_per_m": 0,
                    "cached_in_per_m": 0,
                    "out_per_m": 0,
                },
            },
        })
        cloud_policy.configure(policy)

        cloud, price_per_m = _cloud_price_per_m_in_preset(
            difficulty=0,
            min_success=0.5,
            quality_floor=0,
            profile={"in_len": 1000, "out_len": 100, "tokens_per_request": 1100},
            prefix_hit_rate=0,
            preset_name="current",
        )

        self.assertEqual(cloud["key"], "gemini-flash-lite")
        self.assertEqual(cloud["price_source"], "override")
        self.assertEqual(price_per_m, 0)
        self.assertEqual(set(cloud_policy.effective_catalog()), {"gemini-flash-lite"})

    def test_custom_preset_is_selectable_by_state_normalizer(self):
        policy = cloud_policy.validate_policy({
            "allowed_models": ["gemini-flash-lite"],
            "corpo_presets": {
                "negotiated": {
                    "label": "Negotiated gateway",
                    "models": ["gemini-flash-lite"],
                },
            },
        })
        cloud_policy.configure(policy)

        self.assertEqual(normalize_corpo_cloud("negotiated"), "negotiated")

    def test_custom_preset_cannot_bypass_allowlist(self):
        with self.assertRaisesRegex(ValueError, "outside allowed_models"):
            cloud_policy.validate_policy({
                "allowed_models": ["gemini-flash-lite"],
                "corpo_presets": {
                    "invalid": {
                        "label": "Invalid gateway",
                        "models": ["claude-sonnet"],
                    },
                },
            })

    def test_cloud_routing_enforces_required_capabilities(self):
        cloud_policy.configure(cloud_policy.validate_policy({
            "allowed_models": ["gemini-flash-lite"],
        }))

        cloud, price_per_m = _cloud_price_per_m_in_preset(
            difficulty=0,
            min_success=0.5,
            quality_floor=0,
            profile={"in_len": 1000, "out_len": 100, "tokens_per_request": 1100},
            prefix_hit_rate=0,
            preset_name="current",
            required_capabilities=frozenset({"capability-not-in-catalog"}),
        )

        self.assertIsNone(cloud)
        self.assertTrue(math.isinf(price_per_m))

    def test_policy_sections_must_be_objects(self):
        for section in ("price_overrides", "corpo_presets"):
            with self.subTest(section=section):
                with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                    cloud_policy.validate_policy({section: []})

    def test_policy_file_loads_from_environment(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as policy_file:
            json.dump({"allowed_models": ["gemini-flash-lite"]}, policy_file)
            policy_file.flush()
            with patch.dict(os.environ, {cloud_policy.POLICY_ENV_VAR: policy_file.name}):
                loaded = cloud_policy.configure_from_env()

        self.assertEqual(loaded.allowed_models, frozenset({"gemini-flash-lite"}))
        self.assertTrue(cloud_policy.policy_active())

    def test_custom_gateway_and_policy_status_render(self):
        cloud_policy.configure(cloud_policy.validate_policy({
            "allowed_models": ["gemini-flash-lite"],
            "price_overrides": {
                "gemini-flash-lite": {"out_per_m": 0.5},
            },
            "corpo_presets": {
                "negotiated": {
                    "label": "Negotiated gateway",
                    "models": ["gemini-flash-lite"],
                },
            },
        }))

        response = app_module.app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Negotiated gateway", response.data)
        self.assertIn(b"Corporate cloud policy active", response.data)


if __name__ == "__main__":
    unittest.main()
