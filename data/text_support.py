"""Text-model family factories."""

import math

from .asr_support import (
    RWKV7_G1_CONTEXT,
    RWKV7_G1_HEAD_DIM,
)
from .model_class import Model, SpeculativeProfile


def _rwkv7_g1_model(
    key: str,
    name: str,
    color: str,
    params: float,
    layers: int,
    hidden_dim: int,
    capabilities: frozenset[str],
) -> Model:
    heads = hidden_dim // RWKV7_G1_HEAD_DIM
    return Model(
        key,
        name,
        "RWKV",
        color,
        params,
        params,
        False,
        layers,
        heads,
        0,
        RWKV7_G1_HEAD_DIM,
        False,
        kv_layers=0,
        hidden_dim=hidden_dim,
        attention_layers=0,
        linear_attention_layers=layers,
        linear_attention_heads=heads,
        linear_attention_head_dim=RWKV7_G1_HEAD_DIM,
        linear_attention_k_heads=heads,
        linear_attention_k_head_dim=RWKV7_G1_HEAD_DIM,
        attention_label=f"RWKV recurrent state, ctx {RWKV7_G1_CONTEXT // 1024}k",
        capabilities_override=capabilities,
        max_context_tokens=RWKV7_G1_CONTEXT,
    )


LFM_TEXT_CAPABILITIES = frozenset({"tools"})


def _lfm_text_model(
    key: str,
    name: str,
    color: str,
    total_params: float,
    active_params: float,
    layers: int,
    attention_layers: int,
    hidden_dim: int,
    num_heads: int,
    kv_heads: int,
    capabilities: frozenset[str] = LFM_TEXT_CAPABILITIES,
    speculative_profiles: tuple[SpeculativeProfile, ...] = (),
) -> Model:
    conv_layers = max(layers - attention_layers, 0)
    head_dim = hidden_dim // max(num_heads, 1)
    return Model(
        key,
        name,
        "LFM",
        color,
        total_params,
        active_params,
        not math.isclose(total_params, active_params, rel_tol=1e-9, abs_tol=1.0),
        layers,
        num_heads,
        kv_heads,
        head_dim,
        False,
        kv_layers=attention_layers,
        hidden_dim=hidden_dim,
        attention_layers=attention_layers,
        attention_query_heads=num_heads,
        attention_label=f"{conv_layers} LIV conv + {attention_layers} GQA, ctx 128k",
        capabilities_override=capabilities,
        max_context_tokens=128000,
        speculative_profiles=speculative_profiles,
    )
