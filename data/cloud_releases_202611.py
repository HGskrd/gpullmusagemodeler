"""Primary-source hosted pricing snapshot captured 2026-11-11.

Quality/verbosity are explicit predecessor proxies, assigned by cloud.py.
Rates are USD per million tokens; storage is USD per million token-hours.
"""

CLOUD_RELEASE_PROXIES = {
    "gpt-6.1-sol": "gpt-5.6-sol",
    "gpt-6.1-terra": "gpt-5.6-terra",
    "gpt-6-luna": "gpt-5.6-luna",
    "claude-fable-5.2": "claude-fable",
    "gemini-4-carbon": "gemini-pro",
    "gemini-4-argon": "gemini-pro",
    "grok-4.7": "grok-4.6",
    "minimax-m3.1-flash": "deepseek-v4-flash",
}


def apply_cloud_release_updates(models: dict, preview: dict) -> None:
    """Keep stable old identities and current/historical price snapshots separate."""
    captured = "2026-11-11"
    proxy_note = (
        "Quality and verbosity use an uncalibrated predecessor/deployment-class proxy; "
        "they are not measurements of this release."
    )
    for suffix, parent, released, inp, cached, out in (
        ("6.1-sol", "gpt-5.6-sol", "2026-09-29", 2.0, 0.10, 10.0),
        ("6.1-terra", "gpt-5.6-terra", "2026-10-26", 1.0, 0.05, 5.0),
        ("6-luna", "gpt-5.6-luna", "2026-09-22", 0.10, 0.01, 0.50),
    ):
        key = f"gpt-{suffix}"
        models[key] = {
            "label": "GPT-" + suffix.replace("-", " ").title(),
            "vendor": "OpenAI",
            "api_model": key,
            "released_at": released,
            "in_per_m": inp,
            "cached_in_per_m": cached,
            "out_per_m": out,
            "cache_write_per_m": inp * 1.25,
            "cache_write_12h_per_m": inp * 2.5,
            "default_cache_write_ttl": "30m",
            "long_context_threshold_tokens": 272_000,
            "long_context_in_per_m": inp * 2,
            "long_context_cached_in_per_m": cached * 2,
            "long_context_out_per_m": out * 1.5,
            "long_context_cache_write_per_m": inp * 2.5,
            "long_context_cache_write_12h_per_m": inp * 5,
            "max_context_tokens": 1_050_000,
            "max_input_tokens": 922_000,
            "max_output_tokens": 128_000,
            "service_tier_price_multipliers": {"fast": 2.0, "batch": 0.5, "flex": 0.5},
            "supported_service_tiers": ("standard", "fast", "batch", "flex"),
            "region_price_multipliers": {"regional": 1.1},
            "pricing_source": f"https://developers.openai.com/api/docs/models/{key}",
            "price_note": proxy_note,
        }
        if suffix == "6.1-sol":
            models[key]["service_tier_price_multipliers"]["ultrafast"] = 6.0
            models[key]["supported_service_tiers"] += ("ultrafast",)
        # Existing releases keep their own identities and captured prices.
        models[parent].update(max_input_tokens=922_000, max_output_tokens=128_000)
    for key, row in models.items():
        if row["vendor"] == "OpenAI":
            row.setdefault("default_cache_write_ttl", "30m")
            row.setdefault("cache_write_12h_per_m", row["in_per_m"] * 2.5)
            row.setdefault("long_context_cache_write_12h_per_m", row["in_per_m"] * 5)
        elif row["vendor"] == "Anthropic":
            row.setdefault("default_cache_write_ttl", "5m")
    models["gpt-5.6-sol"]["price_note"] = "Promotional rates published through 2026-11-30."
    models["claude-fable-5.2"] = {
        "label": "Claude Fable 5.2",
        "vendor": "Anthropic",
        "api_model": "claude-fable-5-2",
        "released_at": "2026-11-09",
        "in_per_m": 8.0,
        "cached_in_per_m": 0.20,
        "out_per_m": 40.0,
        "cache_write_per_m": 10.0,
        "cache_write_1h_per_m": 16.0,
        "default_cache_write_ttl": "5m",
        "max_context_tokens": 1_000_000,
        "max_output_tokens": 128_000,
        "supported_service_tiers": ("standard", "fast", "batch"),
        "service_tier_price_multipliers": {"fast": 2.0, "batch": 0.5},
        "pricing_source": "https://platform.claude.com/docs/en/about-claude/pricing",
        "price_note": proxy_note,
    }
    for variant, released in (("argon", "2026-10-13"), ("carbon", "2026-11-10")):
        key = f"gemini-4-{variant}"
        models[key] = {
            "label": f"Gemini 4 {variant.title()}",
            "vendor": "Google",
            "api_model": key,
            "released_at": released,
            "in_per_m": 2.0,
            "cached_in_per_m": 0.10,
            "out_per_m": 10.0,
            "max_context_tokens": 1_048_576,
            "max_output_tokens": 65_536,
            "cache_mode": "implicit_only",
            "supported_service_tiers": ("standard", "batch", "flex", "priority"),
            "service_tier_price_multipliers": {"batch": 0.5, "flex": 0.5, "priority": 1.8},
            "future_pricing": {
                "effective_at": "2026-12-01",
                "in_per_m": 4.0,
                "cached_in_per_m": 0.20,
                "out_per_m": 20.0,
            },
            "pricing_source": "https://aistudio.google.com/docs/pricing",
            "price_note": "Introductory rates through November 30. Unified input/thinking/output "
            "context. Ordinary synchronous output cap; multihop long-decode "
            "continuation and repeated-input billing are not simulated. " + proxy_note,
        }
    models["grok-4.7"] = {
        "label": "Grok 4.7",
        "vendor": "xAI",
        "api_model": "grok-4.7",
        "released_at": "2026-09-21",
        "in_per_m": 2.0,
        "cached_in_per_m": 0.5,
        "out_per_m": 6.0,
        "max_context_tokens": 500_000,
        "long_context_threshold_tokens": 200_000,
        "long_context_threshold_inclusive": True,
        "long_context_in_per_m": 4.0,
        "long_context_cached_in_per_m": 1.0,
        "long_context_out_per_m": 12.0,
        "supported_service_tiers": ("standard", "batch"),
        "service_tier_price_multipliers": {"batch": 0.8},
        "region_price_multipliers": {"us": 1.1},
        "pricing_source": "https://docs.x.ai/developers/pricing",
        "price_note": proxy_note,
    }
    models["minimax-m3.1-flash"] = {
        "label": "MiniMax M3.1 Flash",
        "vendor": "MiniMax",
        "api_model": "MiniMax-M3.1-Flash",
        "released_at": "2026-10-19",
        "in_per_m": 0.15,
        "cached_in_per_m": 0.015,
        "out_per_m": 0.60,
        "max_context_tokens": 1_000_000,
        "max_output_tokens": 524_288,
        "supported_service_tiers": ("standard",),
        "pricing_source": "https://www.minimax.io/blog/minimax-m31f",
        "price_note": "Native MiniMax endpoint; output maximum from its official OpenAPI. "
        + proxy_note,
    }
    for key in CLOUD_RELEASE_PROXIES:
        models[key].update(
            pricing_captured_at=captured,
            quality=0.5,
            token_efficiency=1.0,
            capabilities=("tools", "ctx_128k", "images", "reasoning"),
        )
    models["gemini-pro"].update(
        cache_storage_per_m_hour=4.5,
        service_tier_component_multipliers={
            "batch": {
                "in_per_m": 0.5,
                "out_per_m": 0.5,
                "cached_in_per_m": 1.0,
                "cache_storage_per_m_hour": 1.0,
            }
        },
    )
    models["gemini-flash-lite"].update(
        cache_storage_per_m_hour=1.0,
        service_tier_component_multipliers={
            "batch": {"in_per_m": 0.5, "out_per_m": 0.5, "cached_in_per_m": 0.5},
            "flex": {"in_per_m": 0.5, "out_per_m": 0.5, "cached_in_per_m": 2 / 3},
        },
    )
    models["command-a-03-2025"]["max_output_tokens"] = 8_000
    models["command-r7b-12-2024"]["max_output_tokens"] = 4_000
    holidays = (
        ("2026-01-01", "2026-01-03"),
        ("2026-02-15", "2026-02-23"),
        ("2026-04-04", "2026-04-06"),
        ("2026-05-01", "2026-05-05"),
        ("2026-06-19", "2026-06-21"),
        ("2026-09-25", "2026-09-27"),
        ("2026-10-01", "2026-10-07"),
    )
    for key in ("deepseek-v4-flash", "deepseek-v4-pro"):
        models[key].update(
            max_output_tokens=384_000,
            holiday_date_ranges=holidays,
            holiday_calendar_years=("2026",),
            holiday_calendar_source="https://www.beijing.gov.cn/cs/gncs/zcwj/202603/t20260327_4568275.html",
            price_note="Ordinary-week blend excludes public-holiday variation. An explicit UTC "
            "date/hour or holiday override resolves calendar-specific rates; "
            "unknown calendar years are flagged. Output cap uses decimal K.",
        )
    models["deepseek-v4-pro"]["label"] = "DeepSeek V4 Pro 0813"
    preview.pop("cloud:deepseek-v4-pro", None)
    models["mistral-large-4"].update(
        label="Mistral Large 4",
        in_per_m=0.70,
        cached_in_per_m=0.07,
        out_per_m=2.10,
        pricing_captured_at=captured,
        price_note="Released open weights; current GA API. Historical launch snapshots are "
        "retained for explicit dates. " + proxy_note,
        pricing_history=(
            {
                "effective_at": "2026-10-06",
                "in_per_m": 0.68,
                "cached_in_per_m": 0.07,
                "out_per_m": 2.09,
            },
            {
                "effective_at": "2026-10-20",
                "in_per_m": 1.36,
                "cached_in_per_m": 0.14,
                "out_per_m": 4.18,
            },
            {
                "effective_at": "2026-10-29",
                "in_per_m": 0.70,
                "cached_in_per_m": 0.07,
                "out_per_m": 2.10,
            },
        ),
    )
    models["mistral-large-4"].pop("future_pricing", None)
    preview.pop("cloud:mistral-large-4", None)
