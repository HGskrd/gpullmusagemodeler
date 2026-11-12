"""MetaEncoder geometry and full released embedding-artifact residency."""

import pytest

from calc import (
    EfficiencyParams,
    compute_embedding,
    embedding_sequence_length,
    embedding_vectors_per_input,
)
from data import GPUS, MODELS, OPEN_MODEL_ARCHITECTURE_SOURCES
from data.artifacts import artifact_profiles
from data.groups import models_by_kind


def test_meta_encoder_has_its_own_causal_embedding_geometry():
    model = MODELS["meta-encoder-v1"]
    assert model.key == "meta-encoder-v1"
    assert model in models_by_kind()["embedding"]["Embeddings"]
    assert not model.is_decision_model
    assert (model.layers, model.hidden_size, model.num_heads, model.kv_heads, model.head_dim) == (
        52,
        6656,
        32,
        2,
        128,
    )
    assert (model.local_attention_layers, model.local_attention_window) == (39, 2048)
    assert model.max_context_tokens == 16384
    assert model.embedding_profile.causal_attention
    assert model.embedding_profile.output_dim == 6656
    assert model.embedding_profile.vector_bytes_per_elem == 4
    assert model.quality_confidence == 0


def test_meta_encoder_residency_includes_separate_projection_head():
    model = MODELS["meta-encoder-v1"]
    backbone_params = 28_431_795_200
    head_params = 6656 * 6656
    assert model.total_params == model.active_params == backbone_params + head_params
    profile = model.quantization_profile("bf16")
    assert profile in artifact_profiles(model.key)
    assert profile.source_kind == "exact"
    assert profile.source_repo == "facebook/meta-encoder-v1"
    assert profile.source_revision == OPEN_MODEL_ARCHITECTURE_SOURCES[model.key].revision
    assert profile.total_weight_bytes == (backbone_params + head_params) * 2
    assert profile.total_weight_bytes == 56_863_590_400 + 88_604_672
    assert model.total_params * model.weight_bytes_per_param("bf16") == pytest.approx(
        profile.total_weight_bytes
    )
    assert model.quantization_profile("fp8") is None


def test_meta_encoder_capacity_is_single_vector_and_fits_one_h100():
    model = MODELS["meta-encoder-v1"]
    assert embedding_sequence_length(model, 131072) == 16384
    assert embedding_vectors_per_input(model, 512) == 1
    result = compute_embedding(
        model, (1, 1, 1), 1, 512, GPUS["H100"], 0.9, 2.0, "bf16", EfficiencyParams()
    )
    assert result is not None
    assert result.rps > 0
    assert result.vectors_per_input == 1
    assert result.output_bytes_per_input == 6656 * 4
    assert (
        compute_embedding(
            model, (1, 1, 1), 1, 512, GPUS["L4"], 0.9, 2.0, "bf16", EfficiencyParams()
        )
        is None
    )
