"""Compatibility-preserving assembly of model-family catalogs."""

from dataclasses import replace

from .asr_support import (
    GEMMA_4_12B_ASR_PROFILE,
    GEMMA_4_E2B_ASR_PROFILE,
    GEMMA_4_E4B_ASR_PROFILE,
)
from .model_class import Model
from .models_asr import ASR_MODELS
from .models_embedding import EMBEDDING_MODELS
from .models_text import TEXT_MODELS

_INITIAL_MODELS = {**EMBEDDING_MODELS, **TEXT_MODELS, **ASR_MODELS}
MODEL_ORDER = (
    "nvidia-nemotron-3-embed-8b",
    "nvidia-nemotron-3-embed-1b",
    "denseon",
    "lateon",
    "bge-m3",
    "mxbai-embed-large-v1",
    "mxbai-embed-2d-large-v1",
    "mxbai-embed-xsmall-v1",
    "deepset-mxbai-embed-de-large-v1",
    "mxbai-edge-colbert-v0-17m",
    "mxbai-edge-colbert-v0-32m",
    "modernbert-embed-base",
    "kalm-mini-it-v15",
    "pplx-embed-v1-0.6b",
    "pplx-embed-v1-4b",
    "pplx-embed-v1-late-0.6b",
    "cohere-embed-v4-0",
    "l8",
    "l70",
    "ge2",
    "ge4",
    "g12",
    "g26",
    "g31",
    "lfm2.5-350m",
    "lfm2.5-1.2b-instruct",
    "lfm2.5-1.2b-thinking",
    "lfm2-700m",
    "lfm2-2.6b",
    "lfm2-8b-a1b",
    "lfm2-24b-a2b",
    "rwkv7-g1d-01b",
    "rwkv7-g1d-04b",
    "rwkv7-g1f-15b",
    "rwkv7-g1f-29b",
    "rwkv7-g1g-72b",
    "rwkv7-g1g-133b",
    "q08",
    "q2",
    "q4",
    "q9",
    "q27",
    "q35",
    "q122",
    "q397",
    "glm45a",
    "glm45",
    "glm46",
    "glm47",
    "glm47f",
    "glm5",
    "glm51",
    "glm52",
    "k25",
    "kimi-k3",
    "kimi-linear-48b",
    "inkling",
    "inkling-small-preview",
    "command-a-plus-05-2026",
    "command-a-03-2025",
    "command-r7b-12-2024",
    "north-mini-code-1-0",
    "tiny-aya-global",
    "tiny-aya-earth",
    "tiny-aya-fire",
    "tiny-aya-water",
    "minimax25",
    "minimax27",
    "minimax3",
    "nem3s",
    "nem3n",
    "nem3no",
    "ds3",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "mi7",
    "mx87",
    "cs22",
    "ms24",
    "ms32",
    "voxtral-realtime-mini-4b",
    "mimo-v2.5-asr",
    "nvidia-nemotron-speech-streaming-0.6b",
    "nvidia-nemotron-3.5-asr-streaming-0.6b",
    "nvidia-parakeet-unified-0.6b",
    "nvidia-parakeet-realtime-eou-120m",
    "nvidia-multitalker-parakeet-streaming-0.6b",
    "kyutai-stt-1b-en-fr",
    "kyutai-stt-2.6b-en",
    "moonshine-streaming-tiny",
    "moonshine-streaming-small",
    "moonshine-streaming-medium",
    "fun-asr-nano-2512",
    "granite-4.0-1b-speech",
    "cohere-transcribe-03-2026",
    "nvidia-parakeet-tdt-0.6b-v3",
    "mm31",
    "mistral-medium-3.5-preview",
    "ms4",
    "ml3",
    "ml123",
    "n3",
    "n8",
    "n14",
    "dv24",
    "dv123",
    "zaya1-8b",
    "zaya1-74b-preview",
    "laguna-m1",
    "laguna-xs2",
    "laguna-xs-2-1",
    "laguna-s-2-1",
    "mimo-v2.5-pro",
    "mimo-v2.5",
    "cr13",
)
MODELS: dict[str, Model] = {key: _INITIAL_MODELS[key] for key in MODEL_ORDER}

# Dedicated ASR variants intentionally follow the base model catalog. Keep this
# tuple in step with the MODELS.update() block below; the checks at the end of
# the module enforce it.
DERIVED_ASR_ORDER = (
    "gemma-4-e2b-asr",
    "gemma-4-e4b-asr",
    "gemma-4-12b-unified-asr",
)

MODELS.update(
    {
        "gemma-4-e2b-asr": replace(
            MODELS["ge2"],
            key="gemma-4-e2b-asr",
            name="Gemma 4 E2B ASR",
            cat="Audio",
            capabilities_override=frozenset(),
            realtime_profile=GEMMA_4_E2B_ASR_PROFILE,
            speculative_profiles=(),
        ),
        "gemma-4-e4b-asr": replace(
            MODELS["ge4"],
            key="gemma-4-e4b-asr",
            name="Gemma 4 E4B ASR",
            cat="Audio",
            capabilities_override=frozenset(),
            realtime_profile=GEMMA_4_E4B_ASR_PROFILE,
            speculative_profiles=(),
        ),
        "gemma-4-12b-unified-asr": replace(
            MODELS["g12"],
            key="gemma-4-12b-unified-asr",
            name="Gemma 4 12B Unified ASR",
            cat="Audio",
            capabilities_override=frozenset(),
            realtime_profile=GEMMA_4_12B_ASR_PROFILE,
            speculative_profiles=(),
        ),
    }
)

# Structural catalog checks. These raise instead of using `assert` so that
# `python -O` cannot strip the catalog's only load-time guard.
#
# They replace a hand-maintained 118-key tuple literal that duplicated
# MODEL_ORDER. That literal could not catch the likeliest mistake — adding an
# entry to a family module and forgetting MODEL_ORDER, which silently kept the
# model out of MODELS without changing tuple(MODELS) at all.
_UNORDERED = tuple(key for key in _INITIAL_MODELS if key not in frozenset(MODEL_ORDER))
if _UNORDERED:
    raise RuntimeError(
        "Model(s) defined in a family module but missing from MODEL_ORDER, so they "
        f"would never reach MODELS or the picker: {_UNORDERED}"
    )

_EXPECTED_ORDER = MODEL_ORDER + DERIVED_ASR_ORDER
if tuple(MODELS) != _EXPECTED_ORDER:
    raise RuntimeError(
        "MODELS iteration order does not match MODEL_ORDER + DERIVED_ASR_ORDER; "
        "a duplicate key or a stray insertion has reordered the catalog."
    )
