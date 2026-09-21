import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch

from app_factory import create_test_app

import cloud_policy
from data import CLOUD_MODELS
from engine.economics import _cloud_price_per_m_in_preset
from state import normalize_corpo_cloud


class CloudPolicyTests(unittest.TestCase):
    def tearDown(self):
        cloud_policy.configure(None)

    def test_allowlist_and_negotiated_prices_reach_projection_math(self):
        policy = cloud_policy.validate_policy(
            {
                "allowed_models": ["gemini-flash-lite"],
                "price_overrides": {
                    "gemini-flash-lite": {
                        "in_per_m": 0,
                        "cached_in_per_m": 0,
                        "out_per_m": 0,
                    },
                },
            }
        )
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
        policy = cloud_policy.validate_policy(
            {
                "allowed_models": ["gemini-flash-lite"],
                "corpo_presets": {
                    "negotiated": {
                        "label": "Negotiated gateway",
                        "models": ["gemini-flash-lite"],
                    },
                },
            }
        )
        cloud_policy.configure(policy)

        self.assertEqual(normalize_corpo_cloud("negotiated"), "negotiated")

    def test_custom_preset_cannot_bypass_allowlist(self):
        with self.assertRaisesRegex(ValueError, "outside allowed_models"):
            cloud_policy.validate_policy(
                {
                    "allowed_models": ["gemini-flash-lite"],
                    "corpo_presets": {
                        "invalid": {
                            "label": "Invalid gateway",
                            "models": ["claude-sonnet"],
                        },
                    },
                }
            )

    def test_cloud_routing_enforces_required_capabilities(self):
        cloud_policy.configure(
            cloud_policy.validate_policy(
                {
                    "allowed_models": ["gemini-flash-lite"],
                }
            )
        )

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

    def test_cloud_routing_enforces_combined_context_and_output_limits(self):
        cloud = {
            "vendor": "Test",
            "quality": 1.0,
            "token_efficiency": 1.0,
            "capabilities": (),
            "in_per_m": 1.0,
            "cached_in_per_m": 0.1,
            "out_per_m": 2.0,
            "max_context_tokens": 1_000,
            "max_output_tokens": 200,
        }
        with patch(
            "engine.economics.cloud_policy.effective_corpo_models",
            return_value=[("limited", cloud)],
        ):
            over_context, _ = _cloud_price_per_m_in_preset(
                0, 0, 0, {"in_len": 900, "out_len": 101, "tokens_per_request": 1001}, 0, "test"
            )
            over_output, _ = _cloud_price_per_m_in_preset(
                0, 0, 0, {"in_len": 700, "out_len": 201, "tokens_per_request": 901}, 0, "test"
            )

        self.assertIsNone(over_context)
        self.assertIsNone(over_output)

    def test_inclusive_tier_cache_lifecycle_and_multiplier_are_charged(self):
        cloud = {
            "vendor": "Test",
            "quality": 1.0,
            "token_efficiency": 1.0,
            "capabilities": (),
            "in_per_m": 1.0,
            "cached_in_per_m": 0.1,
            "cache_write_per_m": 2.0,
            "cache_storage_per_m_hour": 0.5,
            "out_per_m": 3.0,
            "long_context_threshold_tokens": 1_000,
            "long_context_threshold_inclusive": True,
            "long_context_in_per_m": 4.0,
            "long_context_cached_in_per_m": 0.4,
            "long_context_cache_write_per_m": 5.0,
            "long_context_out_per_m": 6.0,
        }
        profile = {
            "in_len": 1_000,
            "out_len": 100,
            "tokens_per_request": 1_100,
            "cache_write_tokens": 200,
            "cache_storage_token_hours": 400,
            "cloud_price_multiplier": 1.25,
        }
        with patch(
            "engine.economics.cloud_policy.effective_corpo_models",
            return_value=[("tiered", cloud)],
        ):
            info, price_per_m = _cloud_price_per_m_in_preset(0, 0, 0, profile, 0.5, "test")

        expected_sticker = 1.25 * (
            500 / 1e6 * 4.0 + 500 / 1e6 * 0.4 + 200 / 1e6 * 5.0 + 400 / 1e6 * 0.5 + 100 / 1e6 * 6.0
        )
        self.assertTrue(info["long_context_pricing_applied"])
        self.assertEqual(info["effective_cache_write_per_m"], 5.0)
        self.assertAlmostEqual(
            price_per_m,
            expected_sticker / (1_100 / 1e6) / info["success_rate"],
        )

    def test_scheduled_pricing_selects_peak_and_off_peak_utc_bands(self):
        cloud = dict(CLOUD_MODELS["deepseek-v4-flash"])
        with patch(
            "engine.economics.cloud_policy.effective_corpo_models",
            return_value=[("deepseek-v4-flash", cloud)],
        ):
            peak, _ = _cloud_price_per_m_in_preset(
                0,
                0,
                0,
                {
                    "in_len": 1000,
                    "out_len": 1,
                    "tokens_per_request": 1001,
                    "pricing_weekday_utc": 0,
                    "pricing_hour_utc": 2,
                },
                0,
                "test",
            )
            off_peak, _ = _cloud_price_per_m_in_preset(
                0,
                0,
                0,
                {
                    "in_len": 1000,
                    "out_len": 1,
                    "tokens_per_request": 1001,
                    "pricing_weekday_utc": 6,
                    "pricing_hour_utc": 2,
                },
                0,
                "test",
            )

        self.assertEqual((peak["pricing_period"], peak["effective_in_per_m"]), ("peak", 0.30))
        self.assertEqual(
            (off_peak["pricing_period"], off_peak["effective_in_per_m"]), ("off_peak", 0.15)
        )

    def test_future_dated_pricing_and_named_multipliers_are_explicit(self):
        cloud = dict(CLOUD_MODELS["gemini-3.8-flash"])
        cloud["service_tier_price_multipliers"] = {"priority": 1.5}
        cloud["region_price_multipliers"] = {"global": 1.2}
        profile = {
            "in_len": 1_000,
            "out_len": 0,
            "tokens_per_request": 1_000,
            "pricing_as_of": "2027-01-01",
            "cloud_service_tier": "priority",
            "cloud_region": "global",
        }
        with patch(
            "engine.economics.cloud_policy.effective_corpo_models",
            return_value=[("gemini-3.8-flash", cloud)],
        ):
            info, price_per_m = _cloud_price_per_m_in_preset(0, 0, 0, profile, 0, "test")

        self.assertEqual(info["price_effective_at"], "2027-01-01")
        self.assertEqual(info["effective_in_per_m"], 1.50)
        self.assertEqual(info["effective_service_tier"], "priority")
        self.assertEqual(info["effective_region"], "global")
        self.assertAlmostEqual(price_per_m, 1.50 * 1.5 * 1.2 / info["success_rate"])

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
        cloud_policy.configure(
            cloud_policy.validate_policy(
                {
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
                }
            )
        )

        response = create_test_app().test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Negotiated gateway", response.data)
        self.assertIn(b"Corporate cloud policy active", response.data)


if __name__ == "__main__":
    unittest.main()
