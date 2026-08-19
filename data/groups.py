"""Catalog grouping helpers used by picker presentation."""

from .cloud import CLOUD_MODEL_ZONES, CLOUD_MODELS
from .environment import GCP_ZONE_COUNTRY, cloud_model_grid_intensity
from .hardware import GPU, GPU_CARDS, GPUS, GPUCard
from .model_class import Model
from .models import MODELS

for _k, _zones in CLOUD_MODEL_ZONES.items():
    if _k not in CLOUD_MODELS:
        continue
    CLOUD_MODELS[_k]["gcp_zones"] = _zones
    CLOUD_MODELS[_k]["regions"] = tuple(
        GCP_ZONE_COUNTRY[z] for z in _zones if z in GCP_ZONE_COUNTRY
    )
    CLOUD_MODELS[_k]["grid_gco2_per_kwh"] = cloud_model_grid_intensity(_zones)


def models_by_category() -> dict[str, list[Model]]:
    cats: dict[str, list[Model]] = {}
    for m in MODELS.values():
        if m.hidden:
            continue
        cats.setdefault(m.cat, []).append(m)
    return cats


# Three top-level kinds used by the model picker tabs. Order is the tab order;
# the first non-empty kind is the default active tab.
MODEL_KINDS: tuple[tuple[str, str], ...] = (
    ("llm", "LLM"),
    ("embedding", "Embedding"),
    ("asr", "ASR"),
)


def _model_kind(m: Model) -> str:
    if m.is_realtime_only:
        return "asr"
    if m.is_embedding_model:
        return "embedding"
    return "llm"


def models_by_kind() -> dict[str, dict[str, list[Model]]]:
    """Models grouped first by kind (LLM/Embedding/ASR) then by catalog cat.

    Used by the model picker to render one tab per kind, preserving the
    sub-grouping (e.g. Mistral / Qwen / DeepSeek inside LLM) within each tab.
    """
    out: dict[str, dict[str, list[Model]]] = {kind: {} for kind, _ in MODEL_KINDS}
    for m in MODELS.values():
        if m.hidden:
            continue
        out[_model_kind(m)].setdefault(m.cat, []).append(m)
    return out


def gpus_by_vendor() -> dict[str, list[GPU]]:
    cats: dict[str, list[GPU]] = {}
    for g in GPUS.values():
        cats.setdefault(g.vendor_label, []).append(g)
    return cats


def gpu_cards_by_vendor() -> dict[str, list[GPUCard]]:
    cats: dict[str, list[GPUCard]] = {}
    for card in GPU_CARDS:
        cats.setdefault(card.vendor, []).append(card)
    return cats
