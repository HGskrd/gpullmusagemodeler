"""Cloud model pricing and corporate gateway presets."""

from .quality import aa_intelligence_to_quality, aa_output_tokens_to_efficiency

CLOUD_PRICING_CAPTURED_AT = "2026-08-09"
CLOUD_PRICING_SOURCES = {
    "OpenAI": "https://developers.openai.com/api/docs/models/compare",
    "Anthropic": "https://platform.claude.com/docs/en/about-claude/models/overview",
    "Google": "https://ai.google.dev/gemini-api/docs/pricing",
    "Mistral": "https://mistral.ai/pricing/api/",
    "xAI": "https://x.ai/news/grok-4-1-fast",
    "DeepSeek": "https://api-docs.deepseek.com/quick_start/pricing/",
    "Cohere": "https://cohere.com/pricing",
}
CLOUD_MODELS = {
    "gpt-5.6-sol": {
        "label": "GPT-5.6 Sol",
        "vendor": "OpenAI",
        "api_model": "gpt-5.6-sol",
        "in_per_m": 5.00,
        "cached_in_per_m": 0.50,
        "out_per_m": 30.00,
        "long_context_threshold_tokens": 272_000,
        "long_context_in_per_m": 10.00,
        "long_context_cached_in_per_m": 1.00,
        "long_context_out_per_m": 45.00,
        "max_context_tokens": 1_050_000,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gpt-5.6-terra": {
        "label": "GPT-5.6 Terra",
        "vendor": "OpenAI",
        "api_model": "gpt-5.6-terra",
        "in_per_m": 2.00,
        "cached_in_per_m": 0.20,
        "out_per_m": 12.00,
        "long_context_threshold_tokens": 272_000,
        "long_context_in_per_m": 4.00,
        "long_context_cached_in_per_m": 0.40,
        "long_context_out_per_m": 18.00,
        "max_context_tokens": 1_050_000,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gpt-5.6-luna": {
        "label": "GPT-5.6 Luna",
        "vendor": "OpenAI",
        "api_model": "gpt-5.6-luna",
        "in_per_m": 0.20,
        "cached_in_per_m": 0.02,
        "out_per_m": 1.20,
        "long_context_threshold_tokens": 272_000,
        "long_context_in_per_m": 0.40,
        "long_context_cached_in_per_m": 0.04,
        "long_context_out_per_m": 1.80,
        "max_context_tokens": 1_050_000,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "claude-fable": {
        "label": "Claude Fable 5",
        "vendor": "Anthropic",
        "api_model": "claude-fable-5",
        "in_per_m": 10.00,
        "cached_in_per_m": 1.00,
        "out_per_m": 50.00,
        "max_context_tokens": 1_000_000,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "claude-opus": {
        "label": "Claude Opus 5",
        "vendor": "Anthropic",
        "api_model": "claude-opus-5",
        "in_per_m": 5.00,
        "cached_in_per_m": 0.50,
        "out_per_m": 25.00,
        "max_context_tokens": 1_000_000,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "claude-sonnet": {
        "label": "Claude Sonnet 5",
        "vendor": "Anthropic",
        "api_model": "claude-sonnet-5",
        "in_per_m": 2.00,
        "cached_in_per_m": 0.20,
        "out_per_m": 10.00,
        "price_note": "Introductory price through 2026-08-31; list price is $3/$15 per MTok.",
        "max_context_tokens": 1_000_000,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "claude-haiku": {
        "label": "Claude Haiku 4.5",
        "vendor": "Anthropic",
        "api_model": "claude-haiku-4-5",
        "in_per_m": 1.00,
        "cached_in_per_m": 0.10,
        "out_per_m": 5.00,
        "max_context_tokens": 200_000,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gemini-pro": {
        "label": "Gemini 3.1 Pro Preview",
        "vendor": "Google",
        "api_model": "gemini-3.1-pro-preview",
        "in_per_m": 2.00,
        "cached_in_per_m": 0.20,
        "out_per_m": 12.00,
        "long_context_threshold_tokens": 200_000,
        "long_context_in_per_m": 4.00,
        "long_context_cached_in_per_m": 0.40,
        "long_context_out_per_m": 18.00,
        "price_note": "Standard <=200k-token request; long-context requests use $4/$0.40/$18.",
        "max_context_tokens": 1_048_576,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gemini-flash": {
        "label": "Gemini 3.6 Flash",
        "vendor": "Google",
        "api_model": "gemini-3.6-flash",
        "in_per_m": 1.50,
        "cached_in_per_m": 0.15,
        "out_per_m": 7.50,
        "max_context_tokens": 1_048_576,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "gemini-flash-lite": {
        "label": "Gemini 3.5 Flash-Lite",
        "vendor": "Google",
        "api_model": "gemini-3.5-flash-lite",
        "in_per_m": 0.30,
        "cached_in_per_m": 0.03,
        "out_per_m": 2.50,
        "max_context_tokens": 1_048_576,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "mistral-medium": {
        "label": "Mistral Medium 3.5",
        "vendor": "Mistral",
        "api_model": "mistral-medium-latest",
        "in_per_m": 1.50,
        "cached_in_per_m": 0.15,
        "out_per_m": 7.50,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "mistral-large": {
        "label": "Mistral Large 3",
        "vendor": "Mistral",
        "api_model": "mistral-large-latest",
        "in_per_m": 0.50,
        "cached_in_per_m": 0.05,
        "out_per_m": 1.50,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "mistral-small": {
        "label": "Mistral Small 4",
        "vendor": "Mistral",
        "api_model": "mistral-small-latest",
        "in_per_m": 0.15,
        "cached_in_per_m": 0.015,
        "out_per_m": 0.60,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    # Compatibility-only API row retained for imported policies and presets.
    "mistral-large-2": {
        "label": "Mistral Large 2 (legacy)",
        "vendor": "Mistral",
        "api_model": "mistral-large-2411",
        "in_per_m": 2.00,
        "cached_in_per_m": 0.20,
        "out_per_m": 6.00,
        "legacy": True,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "codestral-2501": {
        "label": "Codestral",
        "vendor": "Mistral",
        "api_model": "codestral-latest",
        "in_per_m": 0.30,
        "cached_in_per_m": 0.03,
        "out_per_m": 0.90,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "grok-4.1-fast": {
        "label": "Grok 4.1 Fast",
        "vendor": "xAI",
        "api_model": "grok-4-1-fast-reasoning",
        "in_per_m": 0.20,
        "cached_in_per_m": 0.05,
        "out_per_m": 0.50,
        "max_context_tokens": 2_000_000,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "deepseek-v4-flash": {
        "label": "DeepSeek V4 Flash 0731",
        "vendor": "DeepSeek",
        "api_model": "deepseek-v4-flash",
        "in_per_m": 0.14,
        "cached_in_per_m": 0.0028,
        "out_per_m": 0.28,
        "max_context_tokens": 1_000_000,
        "price_note": "0731 is the released local-reference checkpoint; DeepSeek has not published a dated statement mapping this hosted API alias to that checkpoint.",
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro Preview",
        "vendor": "DeepSeek",
        "api_model": "deepseek-v4-pro",
        "in_per_m": 0.435,
        "cached_in_per_m": 0.003625,
        "out_per_m": 0.87,
        "max_context_tokens": 1_000_000,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "kimi-k3": {
        "label": "Kimi K3",
        "vendor": "Moonshot AI",
        "in_per_m": 3.00,
        "cached_in_per_m": 0.30,
        "out_per_m": 15.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
        "max_context_tokens": 1_048_576,
        "capabilities": ("tools", "ctx_128k", "images", "reasoning"),
    },
    "command-a-03-2025": {
        "label": "Command A 03-2025",
        "vendor": "Cohere",
        "in_per_m": 2.50,
        "cached_in_per_m": 2.50,
        "out_per_m": 10.00,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
    "command-r7b-12-2024": {
        "label": "Command R7B 12-2024",
        "vendor": "Cohere",
        "in_per_m": 0.0375,
        "cached_in_per_m": 0.0375,
        "out_per_m": 0.15,
        "quality": 0.5,
        "token_efficiency": 1.0,
    },
}

for _cloud in CLOUD_MODELS.values():
    _pricing_source = CLOUD_PRICING_SOURCES.get(str(_cloud["vendor"]))
    if _pricing_source:
        _cloud["pricing_source"] = _pricing_source
        _cloud["pricing_captured_at"] = CLOUD_PRICING_CAPTURED_AT

AA_CLOUD_METRICS: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (51.0, 80.0),  # Frontier-bound proxy pending a directly comparable AA row.
    "gpt-5.6-terra": (47.0, 70.0),  # GPT-5.4-class proxy pending a directly comparable AA row.
    "gpt-5.6-luna": (42.0, 65.0),  # Efficient frontier proxy pending a directly comparable AA row.
    "claude-fable": (51.0, 80.0),  # Frontier-bound proxy pending a directly comparable AA row.
    "claude-opus": (51.0, 72.0),  # Opus 4.7-class proxy pending a direct Opus 4.8 row.
    "claude-sonnet": (49.0, 55.0),  # Sonnet 4.6-class proxy pending a direct Sonnet 5 row.
    "claude-haiku": (37.0, 87.0),  # Proxy from Claude 4.5 Haiku (Reasoning).
    "gemini-pro": (51.0, 55.0),  # Preview benchmark proxy pending a stable AA row.
    "gemini-flash": (47.0, 30.0),  # Gemini 3-family proxy pending a direct 3.5 Flash row.
    "gemini-flash-lite": (35.0, 36.0),  # Compact Gemini 3-family proxy.
    "mistral-medium": (39.0, 90.0),  # Mistral Medium 3.5.
    "mistral-large": (37.0, 60.0),  # Mistral Large 3 family proxy pending a direct row.
    "mistral-small": (31.0, 35.0),  # Size-class proxy pending a direct Mistral Small 4 row.
    "mistral-large-2": (15.0, 2.6),  # Compatibility anchor for imported legacy policies.
    "codestral-2501": (
        20.0,
        10.0,
    ),  # Coding-model proxy; API alias now points to current Codestral.
    "grok-4.1-fast": (41.0, 30.0),  # Grok 4.1-class proxy pending a direct Fast row.
    "deepseek-v4-flash": (42.0, 30.0),  # V4 launch-benchmark proxy pending a direct AA row.
    "deepseek-v4-pro": (49.0, 60.0),  # V4 launch-benchmark proxy pending a direct AA row.
    "kimi-k3": (57.0, 130.0),  # Direct Artificial Analysis K3 reasoning row.
    "command-a-03-2025": (
        32.0,
        70.0,
    ),  # Conservative proxy until AA publishes a directly comparable Command A row.
    "command-r7b-12-2024": (
        12.0,
        8.3,
    ),  # Size-class proxy from compact open instruction models; no AA row found.
}

for _k, (_score, _verbosity_m) in AA_CLOUD_METRICS.items():
    if _k in CLOUD_MODELS:
        CLOUD_MODELS[_k]["quality"] = aa_intelligence_to_quality(_score)
        CLOUD_MODELS[_k]["token_efficiency"] = aa_output_tokens_to_efficiency(_verbosity_m)


def cloud_models_missing_quality_anchors() -> list[str]:
    """Cloud/API models lacking an AA_CLOUD_METRICS row (same silent-0.5 failure mode)."""
    return sorted(key for key in CLOUD_MODELS if key not in AA_CLOUD_METRICS)


# Vertex availability matrix: GCP regions where each cloud-model family is
# served today. Models not on Vertex Europe (e.g. OpenAI via Azure, DeepSeek)
# have an empty tuple and no grid-intensity estimate. Regions are enriched
# onto CLOUD_MODELS at the bottom of the file, after the carbon-intensity
# tables and helpers are defined.
_GEMINI_PRO_ZONES = (
    "europe-west1",
    "europe-west4",
    "europe-west8",
    "europe-west9",
    "europe-central2",
    "europe-north1",
    "europe-southwest1",
)
_GEMINI_FLASH_ZONES = _GEMINI_PRO_ZONES + ("europe-west2", "europe-west3")
_CLAUDE_ZONES = ("europe-west1",)
_MISTRAL_ZONES = ("europe-west4",)

CLOUD_MODEL_ZONES: dict[str, tuple[str, ...]] = {
    "gpt-5.6-sol": (),
    "gpt-5.6-terra": (),
    "gpt-5.6-luna": (),
    "claude-fable": _CLAUDE_ZONES,
    "claude-opus": _CLAUDE_ZONES,
    "claude-sonnet": _CLAUDE_ZONES,
    "claude-haiku": _CLAUDE_ZONES,
    "gemini-pro": _GEMINI_PRO_ZONES,
    "gemini-flash": _GEMINI_FLASH_ZONES,
    "gemini-flash-lite": _GEMINI_PRO_ZONES,
    "mistral-medium": _MISTRAL_ZONES,
    "mistral-large": _MISTRAL_ZONES,
    "mistral-small": _MISTRAL_ZONES,
    "mistral-large-2": _MISTRAL_ZONES,
    "codestral-2501": _MISTRAL_ZONES,
    "grok-4.1-fast": (),
    "deepseek-v4-flash": (),
    "deepseek-v4-pro": (),
    "kimi-k3": (),
    "command-a-03-2025": (),
    "command-r7b-12-2024": (),
}

# Steepness of the quality/difficulty sigmoid used by success_rate(). k=10 gives a
# ~0.1-quality-edge → ~73% success and a 0.2 edge → ~88%. Tune if calibration demands.

CORPO_CLOUD_PRESETS = {
    "current": {
        "label": "Current corpo gateway (Gemini + Mistral)",
        "models": (
            "gemini-flash-lite",
            "gemini-flash",
            "gemini-pro",
            "codestral-2501",
            "mistral-medium",
            "mistral-large",
            "mistral-small",
        ),
    },
    "advocated": {
        "label": "Advocated · + Anthropic on Vertex EU",
        "models": (
            "gemini-flash-lite",
            "gemini-flash",
            "gemini-pro",
            "codestral-2501",
            "mistral-medium",
            "mistral-large",
            "mistral-small",
            "claude-haiku",
            "claude-sonnet",
            "claude-opus",
        ),
    },
    "with_cohere": {
        "label": "Current + Cohere",
        "models": (
            "gemini-flash-lite",
            "gemini-flash",
            "gemini-pro",
            "codestral-2501",
            "mistral-medium",
            "mistral-large",
            "mistral-small",
            "command-r7b-12-2024",
            "command-a-03-2025",
        ),
    },
    "with_kimi": {
        "label": "Current + Kimi K3 API",
        "models": (
            "gemini-flash-lite",
            "gemini-flash",
            "gemini-pro",
            "codestral-2501",
            "mistral-medium",
            "mistral-large",
            "mistral-large-2",
            "kimi-k3",
        ),
    },
}
CORPO_CLOUD_DEFAULT = "current"


def corpo_cloud_models(name: str) -> tuple[str, ...]:
    preset = CORPO_CLOUD_PRESETS.get(name) or CORPO_CLOUD_PRESETS[CORPO_CLOUD_DEFAULT]
    return tuple(preset["models"])


# Opinionated use-case definitions so the demo is one click from a realistic story.
