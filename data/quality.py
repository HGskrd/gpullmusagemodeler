"""Model quality, capability, and benchmark evidence."""

import math
from dataclasses import dataclass, replace

from .model_archive import ARCHIVED_MODELS
from .model_class import Model
from .models import MODELS

AA_INTELLIGENCE_INDEX_MIN = 7.0
# 60 provides a dated headroom ceiling: AA's public leaderboard reports a 59.9
# top score. Keeping this above the current catalog avoids silently tying models
# when a new frontier score arrives.
AA_INTELLIGENCE_INDEX_MAX = 60.0
AA_QUALITY_MIN = 0.30
AA_QUALITY_MAX = 0.95
AA_TOKEN_EFFICIENCY_REF_OUTPUT_TOKENS_M = 10.0
QUALITY_CONFIDENCE_PENALTY = 0.12

QUALITY_DOMAINS: tuple[str, ...] = (
    "general",
    "coding",
    "reasoning",
    "long_context",
    "multilingual",
    "vision",
)
QUALITY_DOMAIN_LABELS = {
    "general": "General",
    "coding": "Coding",
    "reasoning": "Reasoning",
    "long_context": "Long context",
    "multilingual": "Multilingual",
    "vision": "Vision",
}


@dataclass(frozen=True)
class DomainQualityAnchor:
    """One vendor-published benchmark point used as a domain quality signal.

    Scores are stored as fractions on a 0..1 scale, but benchmark identity remains
    explicit because (for example) SWE-bench and LiveCodeBench are not interchangeable.
    """

    quality: float
    benchmark: str
    raw_score: float
    source: str
    confidence: float = 0.90
    note: str = ""


def aa_intelligence_to_quality(score: float) -> float:
    clipped = min(max(score, AA_INTELLIGENCE_INDEX_MIN), AA_INTELLIGENCE_INDEX_MAX)
    span = max(AA_INTELLIGENCE_INDEX_MAX - AA_INTELLIGENCE_INDEX_MIN, 1.0)
    t = (clipped - AA_INTELLIGENCE_INDEX_MIN) / span
    return AA_QUALITY_MIN + (AA_QUALITY_MAX - AA_QUALITY_MIN) * t


def quality_to_aa_intelligence(quality: float) -> float:
    """Return the AA Intelligence Index equivalent of the internal quality axis.

    ``quality`` remains the routing representation for backward-compatible saved
    scenarios; the UI uses this inverse to show the source-scale difficulty rather
    than presenting an unrelated number as an Elo rating.
    """
    clipped = min(max(float(quality), AA_QUALITY_MIN), AA_QUALITY_MAX)
    span = max(AA_QUALITY_MAX - AA_QUALITY_MIN, 1e-9)
    t = (clipped - AA_QUALITY_MIN) / span
    return AA_INTELLIGENCE_INDEX_MIN + (AA_INTELLIGENCE_INDEX_MAX - AA_INTELLIGENCE_INDEX_MIN) * t


def aa_output_tokens_to_efficiency(output_tokens_m: float) -> float:
    return AA_TOKEN_EFFICIENCY_REF_OUTPUT_TOKENS_M / max(output_tokens_m, 0.1)


def normalize_quality_domain(domain: str | None) -> str:
    clean = str(domain or "general").strip().lower()
    return clean if clean in QUALITY_DOMAINS else "general"


def normalize_quality_weights(
    weights: dict[str, float] | None,
    fallback_domain: str | None = "general",
) -> dict[str, float]:
    """Return a compact, normalized task-quality vector.

    Old scenarios stored one ``quality_domain``. Treating that value as a one-hot
    vector keeps their routing unchanged while allowing new use cases to combine
    several independently anchored model capabilities.
    """
    clean: dict[str, float] = {}
    if isinstance(weights, dict):
        for domain, raw_weight in weights.items():
            normalized_domain = normalize_quality_domain(domain)
            try:
                weight = max(0.0, float(raw_weight))
            except (TypeError, ValueError):
                continue
            if weight > 0:
                clean[normalized_domain] = clean.get(normalized_domain, 0.0) + weight
    total = sum(clean.values())
    if total <= 0:
        return {normalize_quality_domain(fallback_domain): 1.0}
    normalized = {domain: round(weight / total, 12) for domain, weight in clean.items()}
    primary = max(normalized, key=lambda domain: (normalized[domain], domain))
    normalized[primary] = round(normalized[primary] + (1.0 - sum(normalized.values())), 12)
    return normalized


def quality_weights_label(
    weights: dict[str, float] | None, fallback_domain: str = "general"
) -> str:
    normalized = normalize_quality_weights(weights, fallback_domain)
    ranked = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    return " · ".join(f"{QUALITY_DOMAIN_LABELS[domain]} {weight:.0%}" for domain, weight in ranked)


def model_domain_anchor(model: Model, domain: str | None = None) -> DomainQualityAnchor | None:
    domain = normalize_quality_domain(domain)
    if domain == "general":
        return None
    return MODEL_DOMAIN_QUALITY_ANCHORS.get(model.key, {}).get(domain)


def model_quality(model: Model, domain: str | None = None) -> float:
    anchor = model_domain_anchor(model, domain)
    return min(max(float(anchor.quality if anchor is not None else model.quality), 0.0), 1.0)


def model_quality_confidence(model: Model, domain: str | None = None) -> float:
    anchor = model_domain_anchor(model, domain)
    raw = anchor.confidence if anchor is not None else getattr(model, "quality_confidence", 1.0)
    return min(max(float(raw), 0.0), 1.0)


def effective_quality(model: Model, domain: str | None = None) -> float:
    """Conservative quality used for routing.

    Direct benchmark scores keep their catalog quality. Proxy or uncertain scores are
    discounted so unknown tiny models do not pass workload gates only because their
    throughput is attractive. A sparse domain anchor replaces the global score only for
    that domain; missing anchors preserve the legacy global result exactly.
    """
    confidence = model_quality_confidence(model, domain)
    return min(
        max(model_quality(model, domain) - QUALITY_CONFIDENCE_PENALTY * (1.0 - confidence), 0.0),
        1.0,
    )


def model_profile_quality(
    model: Model,
    weights: dict[str, float] | None,
    fallback_domain: str | None = "general",
) -> float:
    """Blend a model's domain evidence for a task-quality vector.

    A weighted geometric mean prevents a very strong axis from completely hiding a
    weak required axis, while remaining a smooth closed-form approximation. Sparse
    axes retain the existing conservative global-quality fallback.
    """
    normalized = normalize_quality_weights(weights, fallback_domain)
    log_quality = sum(
        weight * math.log(max(effective_quality(model, domain), 1e-6))
        for domain, weight in normalized.items()
    )
    return min(max(math.exp(log_quality), 0.0), 1.0)


def model_success_rate(model: Model, difficulty: float, domain: str | None = None) -> float:
    return success_rate(effective_quality(model, domain), difficulty)


def model_profile_success_rate(
    model: Model,
    difficulty: float,
    weights: dict[str, float] | None,
    fallback_domain: str | None = "general",
) -> float:
    return success_rate(model_profile_quality(model, weights, fallback_domain), difficulty)


_VISION_MODELS = (
    "ge2",
    "ge4",
    "g12",
    "g26",
    "g31",
    "lfm2.5-vl-3b",
    "glm53f",
    "qwen38-27b",
    "qwen38-flash-next",
    "kimi-k3",
    "inkling",
    "inkling-small-preview",
    "command-a-plus-05-2026",
    "mistral-medium-3.5",
    "ms4",
    "ml3",
    "nem3no",
    "mimo-v2.5",
)
_AUDIO_INPUT_MODELS = (
    "ge2",
    "ge4",
    "g12",
    "inkling",
    "inkling-small-preview",
)
_REASONING_MODELS = (
    "g12",
    "q35",
    "qwen38-27b",
    "qwen38-flash-next",
    "qwen38-2.4t-a95b",
    "glm45",
    "glm45a",
    "glm46",
    "glm47",
    "glm47f",
    "glm53",
    "glm53f",
    "kimi-k3",
    "inkling",
    "inkling-small-preview",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "lfm2.5-1.2b-thinking",
    "command-a-plus-05-2026",
    "mistral-medium-3.5",
    "ml3",
    "nem3s",
    "nemotron35-lightning",
    "nem3no",
    "zaya1-8b",
    "zaya1-74b-preview",
    "laguna-m1",
    "laguna-xs-2-1",
    "laguna-s-2-1",
    "mimo-v2.5-pro",
    "mimo-v2.5",
)
for _k in _VISION_MODELS:
    if _k in MODELS:
        _model = MODELS[_k]
        MODELS[_k] = replace(_model, extra_capabilities=_model.extra_capabilities | {"images"})
for _k in _AUDIO_INPUT_MODELS:
    if _k in MODELS:
        _model = MODELS[_k]
        MODELS[_k] = replace(_model, extra_capabilities=_model.extra_capabilities | {"audio"})
for _k in _REASONING_MODELS:
    if _k in MODELS:
        _model = MODELS[_k]
        MODELS[_k] = replace(_model, extra_capabilities=_model.extra_capabilities | {"reasoning"})

# Artificial Analysis Intelligence Index score and Intelligence Index output-token usage
# (verbosity) in millions. For models with separate reasoning/non-reasoning AA pages, prefer
# the reasoning page when available. Where AA had no directly usable page for the exact model,
# we use the closest available family proxy and note it inline.
AA_MODEL_METRICS: dict[str, tuple[float, float]] = {
    "l8": (12.0, 5.2),
    "llama33-70b": (22.0, 5.0),  # Launch-suite proxy pending a directly comparable AA row.
    "ge2": (12.0, 8.3),
    "ge4": (15.0, 7.9),
    "g12": (25.0, 12.0),  # Proxy from Google Gemma 4 12B benchmarks; no AA page found at launch.
    "g26": (27.0, 14.0),
    "g31": (32.0, 7.1),
    "lfm2.5-350m": (7.0, 12.0),  # Conservative proxy; no AA page found for LFM2.5-350M.
    "lfm2.5-1.2b-instruct": (8.0, 4.6),
    "lfm2.5-1.2b-thinking": (8.0, 31.0),
    "lfm2-700m": (7.0, 10.0),  # Conservative size proxy; no AA page found for LFM2-700M.
    "lfm2.5-2.6b": (10.0, 7.8),
    "lfm2.5-8b-a1b": (14.0, 7.8),
    "lfm2.5-vl-3b": (10.0, 8.0),
    "lfm2-24b-a2b": (10.0, 11.0),
    "rwkv7-g1d-01b": (7.0, 60.0),  # Low-confidence size proxy until AA publishes RWKV7-G1 rows.
    "rwkv7-g1d-04b": (8.0, 70.0),
    "rwkv7-g1f-15b": (11.0, 90.0),
    "rwkv7-g1f-29b": (14.0, 105.0),
    "rwkv7-g1g-72b": (19.0, 120.0),
    "rwkv7-g1g-133b": (22.0, 130.0),
    "q08": (11.0, 230.0),
    "q2": (16.0, 390.0),
    "q4": (27.0, 240.0),
    "q9": (32.0, 200.0),
    "q35": (37.0, 100.0),
    "qwen38-27b": (48.0, 100.0),
    "qwen38-flash-next": (56.0, 100.0),
    "qwen38-2.4t-a95b": (59.0, 110.0),
    "glm45a": (23.0, 68.0),
    "glm45": (26.0, 61.0),
    "glm46": (33.0, 57.0),
    "glm47": (42.0, 170.0),
    "glm47f": (30.0, 64.0),
    "glm53": (58.0, 100.0),  # Launch-benchmark proxy pending a direct AA row.
    # Direct AA Intelligence Index v4.1.1 score from the launch report. AA had
    # not published comparable total output-token usage at capture time, so
    # token efficiency remains neutral instead of borrowing GLM-5.2 verbosity.
    "glm53f": (57.0, 10.0),
    "kimi-k3": (57.0, 130.0),  # Direct Artificial Analysis K3 reasoning row.
    "kimi-linear-48b": (37.0, 100.0),  # Proxy from Qwen 3.5 35B-A3B until AA publishes Kimi Linear.
    "inkling": (
        45.0,
        70.0,
    ),  # Official broad benchmark suite; AA-scale/verbosity proxy pending a direct row.
    "inkling-small-preview": (
        44.0,
        70.0,
    ),  # Official release benchmarks; AA-scale/verbosity proxy pending a direct row.
    "command-a-plus-05-2026": (37.0, 66.0),
    "command-r7b-12-2024": (
        12.0,
        8.3,
    ),  # Size-class proxy from compact open instruction models; no AA row found.
    "north-mini-code-1-0": (
        37.0,
        100.0,
    ),  # Coding-agent proxy from Qwen 3.5 35B-A3B until public benchmark rows exist.
    "minimax3": (48.0, 70.0),  # Low-confidence launch-benchmark proxy; no direct AA row yet.
    "nem3s": (36.0, 110.0),
    "nemotron35-lightning": (35.0, 100.0),
    "nem3no": (
        26.0,
        130.0,
    ),  # Omni preview proxy from Nano reasoning until AA publishes a dedicated page.
    "deepseek-v4-pro": (52.0, 190.0),
    "deepseek-v4-flash": (47.0, 240.0),
    "mi7": (7.0, 2.5),
    "mx87": (8.0, 2.5),  # Proxy verbosity from Mistral 7B; AA exposes score but not token usage.
    "cs22": (
        15.0,
        4.4,
    ),  # Proxy from Devstral Small (Jul '25'); no AA page for Codestral 22B found.
    "mm31": (21.0, 7.6),
    "mistral-medium-3.5": (39.0, 90.0),
    "ms4": (19.0, 3.9),
    "ml3": (23.0, 5.2),
    "granite42-3b": (14.0, 12.0),
    "granite42-8b": (21.0, 11.0),
    "granite42-30b": (30.0, 10.0),
    "n3": (11.0, 16.0),
    "n8": (15.0, 13.0),
    "n14": (16.0, 11.0),
    "dv24": (19.0, 8.6),
    "dv123": (22.0, 7.4),
    "tiny-aya-global": (8.0, 4.6),  # Compact multilingual proxy; gated config and no AA row found.
    "tiny-aya-earth": (8.0, 4.6),
    "tiny-aya-fire": (8.0, 4.6),
    "tiny-aya-water": (8.0, 4.6),
    "zaya1-8b": (24.0, 140.0),  # Proxy from Nemotron 3 Nano reasoning until AA publishes ZAYA1-8B.
    "zaya1-74b-preview": (
        37.0,
        100.0,
    ),  # Preview is pre-RL; proxy from Qwen 3.5 35B-A3B until AA publishes it.
    "laguna-m1": (
        44.0,
        95.0,
    ),  # Proxy from Qwen 3.5 397B-A17B adjusted against Poolside coding-agent benchmarks; no AA row found.
    "laguna-xs-2-1": (
        37.0,
        100.0,
    ),  # Same Qwen 3.5 35B-A3B proxy pending a directly comparable AA row.
    # Quality is a provisional AA-scale proxy. No comparable AA Intelligence Index
    # output-token measurement exists yet, so keep token efficiency neutral instead of
    # imposing the old, unrelated 95M-token family proxy (η=0.11).
    "laguna-s-2-1": (45.0, 10.0),
    "mimo-v2.5-pro": (54.0, 92.0),
    "mimo-v2.5": (49.0, 74.0),
    "cr13": (
        12.0,
        8.3,
    ),  # Proxy from Gemma 4 E2B (Non-reasoning); no AA page for Croissant 1.3B found.
}

for _k, (_score, _verbosity_m) in AA_MODEL_METRICS.items():
    if _k in MODELS:
        MODELS[_k] = replace(
            MODELS[_k],
            quality=aa_intelligence_to_quality(_score),
            token_efficiency=aa_output_tokens_to_efficiency(_verbosity_m),
        )

# Confidence is separate from score: direct benchmark rows stay at 1.0, family/proxy rows
# are discounted by effective_quality(). This is intentionally conservative for models
# whose public benchmark coverage is missing or weak.
AA_MODEL_QUALITY_CONFIDENCE: dict[str, float] = {
    "lfm2.5-350m": 0.45,
    "lfm2.5-2.6b": 0.55,
    "lfm2.5-8b-a1b": 0.55,
    "lfm2.5-vl-3b": 0.50,
    "lfm2-700m": 0.45,
    "llama33-70b": 0.70,
    "g12": 0.65,
    "rwkv7-g1d-01b": 0.35,
    "rwkv7-g1d-04b": 0.35,
    "rwkv7-g1f-15b": 0.35,
    "rwkv7-g1f-29b": 0.35,
    "rwkv7-g1g-72b": 0.35,
    "rwkv7-g1g-133b": 0.35,
    "kimi-linear-48b": 0.55,
    "inkling": 0.65,
    "inkling-small-preview": 0.65,
    "command-r7b-12-2024": 0.60,
    "north-mini-code-1-0": 0.45,
    "nem3no": 0.65,
    "mx87": 0.70,
    "cs22": 0.60,
    "mistral-medium-3.5": 0.70,
    "qwen38-27b": 0.70,
    "qwen38-flash-next": 0.75,
    "qwen38-2.4t-a95b": 0.75,
    "glm53": 0.70,
    "nemotron35-lightning": 0.70,
    "granite42-3b": 0.55,
    "granite42-8b": 0.55,
    "granite42-30b": 0.55,
    "tiny-aya-global": 0.45,
    "tiny-aya-earth": 0.45,
    "tiny-aya-fire": 0.45,
    "tiny-aya-water": 0.45,
    "zaya1-8b": 0.45,
    "zaya1-74b-preview": 0.35,
    "laguna-m1": 0.55,
    "laguna-xs-2-1": 0.45,
    "laguna-s-2-1": 0.5,
    "cr13": 0.25,
}
for _k, _confidence in AA_MODEL_QUALITY_CONFIDENCE.items():
    if _k in MODELS:
        MODELS[_k] = replace(MODELS[_k], quality_confidence=_confidence)

# Sparse per-domain quality anchors. Vendor-reported benchmark fractions are kept
# separate from the AA general-quality axis and carry their evaluation identity and
# provenance. The routing fallback for every missing model/domain pair is the model's
# existing global quality, so catalog coverage can grow without inventing proxies.
_QWEN35_DOMAIN_SOURCE = "https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/README.md"
_QWEN35_397_DOMAIN_SOURCE = "https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8"
_KIMI_K25_DOMAIN_SOURCE = "https://huggingface.co/moonshotai/Kimi-K2.5"
_KIMI_K3_DOMAIN_SOURCE = "https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf"
_DEEPSEEK_V3_DOMAIN_SOURCE = "https://huggingface.co/deepseek-ai/DeepSeek-V3"
_GEMMA4_DOMAIN_SOURCE = "https://ai.google.dev/gemma/docs/core/model_card_4"
_GLM5_DOMAIN_SOURCE = "https://huggingface.co/zai-org/GLM-5"
_GLM52_DOMAIN_SOURCE = "https://huggingface.co/zai-org/GLM-5.2"
_GLM53F_DOMAIN_SOURCE = "https://z.ai/blog/glm-5.3-flash"
_LAGUNA_S21_DOMAIN_SOURCE = "https://huggingface.co/poolside/Laguna-S-2.1"
_NORTH_CODE_DOMAIN_SOURCE = "https://huggingface.co/CohereLabs/North-Mini-Code-1.0"

# Provisional SWE-Bench Pro -> SWE-bench Verified-equivalent crosswalk. The frozen
# overlap cohort and ordinary least-squares fit are documented in the changelog/tests:
# verified_pct = 40.9508834 + 0.6464964 * pro_pct. Keeping the transform explicit is
# preferable to pretending raw percentages from differently difficult harnesses share
# one quality axis. Extrapolated anchors carry reduced confidence.
_SWE_PRO_TO_VERIFIED_INTERCEPT = 40.95088340339454
_SWE_PRO_TO_VERIFIED_SLOPE = 0.6464964126552674


def swebench_pro_to_coding_quality(score: float) -> float:
    equivalent = _SWE_PRO_TO_VERIFIED_INTERCEPT + _SWE_PRO_TO_VERIFIED_SLOPE * float(score)
    return min(max(equivalent / 100.0, 0.0), 1.0)


def _domain_anchor(
    score: float,
    benchmark: str,
    source: str,
    *,
    normalized_quality: float | None = None,
    confidence: float = 0.90,
    note: str = "",
) -> DomainQualityAnchor:
    return DomainQualityAnchor(
        quality=min(
            max(
                float(normalized_quality)
                if normalized_quality is not None
                else float(score) / 100.0,
                0.0,
            ),
            1.0,
        ),
        benchmark=benchmark,
        raw_score=float(score),
        source=source,
        confidence=confidence,
        note=note,
    )


MODEL_DOMAIN_QUALITY_ANCHORS: dict[str, dict[str, DomainQualityAnchor]] = {
    "qwen38-27b": {
        "coding": _domain_anchor(
            61.7,
            "SWE-bench Pro",
            "https://huggingface.co/Qwen/Qwen3.8-27B",
            confidence=0.8,
            note="Official Qwen3.8 post-trained model-card result.",
        ),
    },
    "q27": {
        "coding": _domain_anchor(72.4, "SWE-bench Verified", _QWEN35_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(85.5, "GPQA Diamond", _QWEN35_DOMAIN_SOURCE),
        "long_context": _domain_anchor(60.6, "LongBench v2", _QWEN35_DOMAIN_SOURCE),
        "multilingual": _domain_anchor(82.2, "MMLU-ProX (29 languages)", _QWEN35_DOMAIN_SOURCE),
        "vision": _domain_anchor(75.0, "MMMU-Pro", _QWEN35_DOMAIN_SOURCE),
    },
    "q122": {
        "coding": _domain_anchor(72.0, "SWE-bench Verified", _QWEN35_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(86.6, "GPQA Diamond", _QWEN35_DOMAIN_SOURCE),
        "long_context": _domain_anchor(60.2, "LongBench v2", _QWEN35_DOMAIN_SOURCE),
        "multilingual": _domain_anchor(82.2, "MMLU-ProX (29 languages)", _QWEN35_DOMAIN_SOURCE),
        "vision": _domain_anchor(76.9, "MMMU-Pro", _QWEN35_DOMAIN_SOURCE),
    },
    "q35": {
        "coding": _domain_anchor(69.2, "SWE-bench Verified", _QWEN35_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(84.2, "GPQA Diamond", _QWEN35_DOMAIN_SOURCE),
        "long_context": _domain_anchor(59.0, "LongBench v2", _QWEN35_DOMAIN_SOURCE),
        "multilingual": _domain_anchor(81.0, "MMLU-ProX (29 languages)", _QWEN35_DOMAIN_SOURCE),
        "vision": _domain_anchor(75.1, "MMMU-Pro", _QWEN35_DOMAIN_SOURCE),
    },
    "q397": {
        "coding": _domain_anchor(76.4, "SWE-bench Verified", _QWEN35_397_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(88.4, "GPQA Diamond", _QWEN35_397_DOMAIN_SOURCE),
        "long_context": _domain_anchor(63.2, "LongBench v2", _QWEN35_397_DOMAIN_SOURCE),
    },
    "k25": {
        "coding": _domain_anchor(
            76.8,
            "SWE-bench Verified",
            _KIMI_K25_DOMAIN_SOURCE,
            note="Vendor minimal-tool harness; coding results averaged over five runs.",
        ),
        "reasoning": _domain_anchor(87.6, "GPQA Diamond", _KIMI_K25_DOMAIN_SOURCE),
        "long_context": _domain_anchor(
            61.0,
            "LongBench v2",
            _KIMI_K25_DOMAIN_SOURCE,
            note="Prompts and input contexts standardized to approximately 128k tokens.",
        ),
        "multilingual": _domain_anchor(73.0, "SWE-bench Multilingual", _KIMI_K25_DOMAIN_SOURCE),
        "vision": _domain_anchor(78.5, "MMMU-Pro", _KIMI_K25_DOMAIN_SOURCE),
    },
    "kimi-k3": {
        "coding": _domain_anchor(
            67.5,
            "DeepSWE",
            _KIMI_K3_DOMAIN_SOURCE,
            note="Kimi Code harness; the report also discloses 67.3 with mini-SWE-agent.",
        ),
        "reasoning": _domain_anchor(93.5, "GPQA Diamond", _KIMI_K3_DOMAIN_SOURCE),
        "long_context": _domain_anchor(
            74.7,
            "AA-LCR",
            _KIMI_K3_DOMAIN_SOURCE,
            note="Artificial Analysis long-context reasoning score cited in the release report.",
        ),
        "vision": _domain_anchor(
            81.6,
            "MMMU-Pro",
            _KIMI_K3_DOMAIN_SOURCE,
            note="Without Python tool augmentation; the tool-augmented score is 83.4.",
        ),
    },
    "ds3": {
        "coding": _domain_anchor(42.0, "SWE-bench Verified", _DEEPSEEK_V3_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(59.1, "GPQA Diamond", _DEEPSEEK_V3_DOMAIN_SOURCE),
        "long_context": _domain_anchor(48.7, "LongBench v2", _DEEPSEEK_V3_DOMAIN_SOURCE),
        "multilingual": _domain_anchor(79.4, "MMMLU non-English", _DEEPSEEK_V3_DOMAIN_SOURCE),
    },
    "g31": {
        "coding": _domain_anchor(80.0, "LiveCodeBench v6", _GEMMA4_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(84.3, "GPQA Diamond", _GEMMA4_DOMAIN_SOURCE),
        "long_context": _domain_anchor(
            66.4,
            "MRCR v2 8-needle 128k",
            _GEMMA4_DOMAIN_SOURCE,
            note="Retrieval-style long-context benchmark; not directly comparable to LongBench v2.",
        ),
    },
    "g26": {
        "coding": _domain_anchor(77.1, "LiveCodeBench v6", _GEMMA4_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(82.3, "GPQA Diamond", _GEMMA4_DOMAIN_SOURCE),
        "long_context": _domain_anchor(44.1, "MRCR v2 8-needle 128k", _GEMMA4_DOMAIN_SOURCE),
    },
    "g12": {
        "coding": _domain_anchor(72.0, "LiveCodeBench v6", _GEMMA4_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(78.8, "GPQA Diamond", _GEMMA4_DOMAIN_SOURCE),
        "long_context": _domain_anchor(43.4, "MRCR v2 8-needle 128k", _GEMMA4_DOMAIN_SOURCE),
    },
    "ge4": {
        "coding": _domain_anchor(52.0, "LiveCodeBench v6", _GEMMA4_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(58.6, "GPQA Diamond", _GEMMA4_DOMAIN_SOURCE),
        "long_context": _domain_anchor(25.4, "MRCR v2 8-needle 128k", _GEMMA4_DOMAIN_SOURCE),
    },
    "ge2": {
        "coding": _domain_anchor(44.0, "LiveCodeBench v6", _GEMMA4_DOMAIN_SOURCE),
        "reasoning": _domain_anchor(43.4, "GPQA Diamond", _GEMMA4_DOMAIN_SOURCE),
        "long_context": _domain_anchor(19.1, "MRCR v2 8-needle 128k", _GEMMA4_DOMAIN_SOURCE),
    },
    "glm5": {
        "coding": _domain_anchor(
            77.8,
            "SWE-bench Verified",
            _GLM5_DOMAIN_SOURCE,
            note="OpenHands harness with a tailored prompt and 200k context window.",
        ),
        "reasoning": _domain_anchor(86.0, "GPQA Diamond", _GLM5_DOMAIN_SOURCE),
    },
    "glm51": {
        "coding": _domain_anchor(
            58.4,
            "SWE-Bench Pro → Verified-equivalent",
            _GLM52_DOMAIN_SOURCE,
            normalized_quality=swebench_pro_to_coding_quality(58.4),
            confidence=0.75,
            note="Provisional frozen-cohort linear crosswalk; raw SWE-Bench Pro score retained.",
        ),
        "reasoning": _domain_anchor(86.2, "GPQA Diamond", _GLM52_DOMAIN_SOURCE),
    },
    "glm52": {
        "coding": _domain_anchor(
            62.1,
            "SWE-Bench Pro → Verified-equivalent",
            _GLM52_DOMAIN_SOURCE,
            normalized_quality=swebench_pro_to_coding_quality(62.1),
            confidence=0.75,
            note="Provisional frozen-cohort linear crosswalk; corroborated by DeepSWE 46.2 and Terminal-Bench 2.1 81.0.",
        ),
        "reasoning": _domain_anchor(91.2, "GPQA Diamond", _GLM52_DOMAIN_SOURCE),
    },
    "glm53f": {
        "coding": _domain_anchor(
            63.4,
            "DeepSWE v1.1",
            _GLM53F_DOMAIN_SOURCE,
            note="Vendor mini-swe-agent evaluation with a 400k context limit and six-hour timeout.",
        ),
        "vision": _domain_anchor(
            80.5,
            "MMVU",
            _GLM53F_DOMAIN_SOURCE,
            note="Native video input evaluated at up to 256k context.",
        ),
    },
    "laguna-s-2-1": {
        "coding": _domain_anchor(
            59.4,
            "SWE-Bench Pro → Verified-equivalent",
            _LAGUNA_S21_DOMAIN_SOURCE,
            normalized_quality=swebench_pro_to_coding_quality(59.4),
            confidence=0.75,
            note="Provisional frozen-cohort linear crosswalk; corroborated by DeepSWE 40.4 and Terminal-Bench 2.1 70.2.",
        ),
    },
    "north-mini-code-1-0": {
        "coding": _domain_anchor(
            67.6,
            "SWE-bench Verified",
            _NORTH_CODE_DOMAIN_SOURCE,
            note="SWE-Agent 1.1.0; three-seed vendor evaluation.",
        ),
    },
}

ARCHIVED_DOMAIN_QUALITY_ANCHORS = {
    key: anchors for key, anchors in MODEL_DOMAIN_QUALITY_ANCHORS.items() if key in ARCHIVED_MODELS
}
MODEL_DOMAIN_QUALITY_ANCHORS = {
    key: anchors
    for key, anchors in MODEL_DOMAIN_QUALITY_ANCHORS.items()
    if key not in ARCHIVED_MODELS
}

# Text models whose quality anchor is intentionally not AA-sourced. Membership must be
# justified inline; tests assert every other text model carries an AA_MODEL_METRICS row so
# a new catalog entry cannot silently inherit the quality=0.5 / full-confidence defaults.
AA_QUALITY_PLACEHOLDER: frozenset[str] = frozenset()


def text_models_missing_quality_anchors() -> list[str]:
    """Text models lacking both an AA_MODEL_METRICS row and a placeholder justification."""
    return sorted(
        key
        for key, model in MODELS.items()
        if model.embedding_profile is None
        and not model.is_asr_model
        and key not in AA_MODEL_METRICS
        and key not in AA_QUALITY_PLACEHOLDER
    )


SUCCESS_RATE_SIGMOID_K = 10.0


def success_rate(quality: float, difficulty: float, k: float = SUCCESS_RATE_SIGMOID_K) -> float:
    """Probability that a model of given `quality` succeeds on a task of given `difficulty`.

    Continuous replacement for the old discrete tier-distance success curve:
    `sigmoid(k · (quality − difficulty))`. Quality ≫ difficulty → ~1.0; matched → 0.5;
    quality ≪ difficulty → ~0.0.
    """
    x = max(min(k * (quality - difficulty), 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-x))


def required_quality(
    difficulty: float,
    min_success_rate: float,
    k: float = SUCCESS_RATE_SIGMOID_K,
    quality_floor: float = 0.0,
) -> float:
    """Inverse of success_rate(): minimum model quality that clears `min_success_rate`
    at the given `difficulty`. Returns a value on the same [0, 1] quality axis the model
    catalog uses (AA Intelligence Index calibrated into 0.30..0.95)."""
    slo = min(max(float(min_success_rate), 1e-4), 1 - 1e-4)
    logit = math.log(slo / (1.0 - slo))
    floor = min(max(float(quality_floor or 0.0), 0.0), 1.0)
    return min(max(float(difficulty) + logit / k, floor, 0.0), 1.0)
