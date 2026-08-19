"""Embedding quality evidence and model factories."""

from .model_class import EmbeddingProfile, Model

EMBEDDING_QUALITY_SOURCES: dict[str, str] = {
    "denseon": "LightOn DenseOn HF model card, BEIR average nDCG@10 table.",
    "lateon": "LightOn LateOn HF model card, BEIR average nDCG@10 table.",
    "bge-m3": "MTEB results repo, BGE-M3 MTEB(Multilingual, v2) retrieval average over the 18 pplx report tasks.",
    "mxbai-embed-large-v1": "Mixedbread mxbai-embed-large-v1 HF model card, MTEB Retrieval (15) nDCG@10.",
    "mxbai-embed-2d-large-v1": "Mixedbread mxbai-embed-large-v1 HF model card comparison table, MTEB Retrieval (15) nDCG@10.",
    "mxbai-embed-xsmall-v1": "Mixedbread xsmall release blog, MTEB retrieval average nDCG@10.",
    "deepset-mxbai-embed-de-large-v1": "Mixedbread/deepset German-English release blog, German retrieval benchmark average NDCG@10.",
    "mxbai-edge-colbert-v0-17m": "Mixedbread mxbai-edge ColBERT HF model card, BEIR subset average nDCG@10.",
    "mxbai-edge-colbert-v0-32m": "Mixedbread mxbai-edge ColBERT HF model card, BEIR subset average nDCG@10.",
    "modernbert-embed-base": "Nomic ModernBERT Embed Base HF model card, MTEB Retrieval (15) nDCG@10.",
    "kalm-mini-it-v15": "KaLM v1.5 HF model-index, MTEB English Retrieval (15) average nDCG@10.",
    "pplx-embed-v1-0.6b": "Perplexity pplx-embed technical report, MTEB Multilingual v2 retrieval average nDCG@10, INT8.",
    "pplx-embed-v1-4b": "Perplexity pplx-embed technical report, MTEB Multilingual v2 retrieval average nDCG@10, INT8.",
    "pplx-embed-v1-late-0.6b": "Perplexity late-interaction HF model card, BEIR (15 tasks) average nDCG@10.",
    "cohere-embed-v4-0": "Placeholder: Cohere Embed v4.0 docs publish multimodal and 128k-context product limits, but no comparable public retrieval aggregate was found.",
    "nvidia-nemotron-3-embed-8b": "NVIDIA Nemotron 3 Embed release card/blog, RTEB retrieval average; this is not cross-walked to decontaminated BEIR.",
    "nvidia-nemotron-3-embed-1b": "Placeholder: official 1B config and serving limit are published, but no catalog-comparable retrieval aggregate was isolated.",
}
PUBLISHED_EMBEDDING_QUALITY: dict[str, float] = {
    "denseon": 0.5620,
    "lateon": 0.5722,
    "bge-m3": 0.5288,
    "mxbai-embed-large-v1": 0.5439,
    "mxbai-embed-2d-large-v1": 0.5142,
    "mxbai-embed-xsmall-v1": 0.4280,
    "deepset-mxbai-embed-de-large-v1": 0.5170,
    "mxbai-edge-colbert-v0-17m": 0.4900,
    "mxbai-edge-colbert-v0-32m": 0.5210,
    "modernbert-embed-base": 0.5289,
    "kalm-mini-it-v15": 0.5165,
    "pplx-embed-v1-0.6b": 0.6541,
    "pplx-embed-v1-4b": 0.6966,
    "pplx-embed-v1-late-0.6b": 0.5661,
    "cohere-embed-v4-0": 0.6000,
    "nvidia-nemotron-3-embed-8b": 0.7846,
    "nvidia-nemotron-3-embed-1b": 0.5000,
}
EMBEDDING_QUALITY_PLACEHOLDER: frozenset[str] = frozenset(
    {
        "cohere-embed-v4-0",
        "nvidia-nemotron-3-embed-1b",
    }
)

# Optional hover-detail metric. Keep separate from PUBLISHED_EMBEDDING_QUALITY
# so models without decontaminated BEIR still remain visible in the plot.
EMBEDDING_DECONTAMINATED_BEIR_SOURCES: dict[str, str] = {
    "denseon": "LightOn DenseOn/LateOn blog, Full Decontaminated BEIR Results, DenseOn row, average nDCG@10 over 12 datasets.",
    "lateon": "LightOn DenseOn/LateOn blog, Full Decontaminated BEIR Results, LateOn row, average nDCG@10 over 12 datasets.",
    "modernbert-embed-base": "LightOn DenseOn/LateOn blog, Full Decontaminated BEIR Results, MBEmb.-base row, average nDCG@10 over 12 datasets.",
    "pplx-embed-v1-0.6b": "LightOn DenseOn/LateOn blog, Full Decontaminated BEIR Results, pplx-v1-0.6b row, average nDCG@10 over 12 datasets.",
}
PUBLISHED_EMBEDDING_DECONTAMINATED_BEIR: dict[str, float] = {
    "denseon": 0.5771,
    "lateon": 0.6036,
    "modernbert-embed-base": 0.5442,
    "pplx-embed-v1-0.6b": 0.5850,
}


def _modernbert_embed_model(
    key: str,
    name: str,
    color: str,
    profile: EmbeddingProfile,
) -> Model:
    return Model(
        key,
        name,
        "Embeddings",
        color,
        149e6,
        149e6,
        False,
        22,
        12,
        12,
        64,
        False,
        hidden_dim=768,
        local_attention_layers=14,
        local_attention_window=128,
        attention_label="ModernBERT local/global encoder",
        capabilities_override=frozenset(),
        embedding_profile=profile,
    )


def _pplx_embed_model(
    key: str,
    name: str,
    color: str,
    params: float,
    layers: int,
    hidden_dim: int,
    num_heads: int,
    kv_heads: int,
    output_dim: int,
    profile: EmbeddingProfile,
) -> Model:
    return Model(
        key,
        name,
        "Embeddings",
        color,
        params,
        params,
        False,
        layers,
        num_heads,
        kv_heads,
        128,
        False,
        hidden_dim=hidden_dim,
        attention_query_heads=num_heads,
        attention_label="bidirectional Qwen3 encoder",
        capabilities_override=frozenset(),
        embedding_profile=profile,
    )


def _cohere_embed_v4_model() -> Model:
    return Model(
        "cohere-embed-v4-0",
        "Cohere Embed v4.0",
        "Embeddings",
        "#0F766E",
        1.0e9,
        1.0e9,
        False,
        32,
        16,
        8,
        128,
        False,
        hidden_dim=2048,
        local_attention_layers=24,
        local_attention_window=4096,
        attention_label="managed multimodal encoder proxy, ctx 128k",
        capabilities_override=frozenset(),
        embedding_profile=EmbeddingProfile(
            label="Cohere Embed v4.0",
            kind="single",
            output_dim=1536,
            max_sequence_length=128000,
            source="Cohere embed-v4.0 docs",
            note="Managed/API multimodal embedder for text, images, and mixed document inputs; Cohere publishes 256/512/1024/1536 output dimensions and 128k context, but no local parameter/config card, so capacity uses a conservative 1B encoder proxy.",
            pooling="API embedding",
        ),
    )


def _tiny_aya_model(key: str, name: str, color: str) -> Model:
    return Model(
        key,
        name,
        "Cohere",
        color,
        3.35e9,
        3.35e9,
        False,
        32,
        20,
        8,
        128,
        False,
        hidden_dim=2560,
        local_attention_layers=24,
        local_attention_window=4096,
        attention_label="Tiny Aya gated-config proxy, 3:1 SWA/global, ctx 8k",
        capabilities_override=frozenset(),
        max_context_tokens=8192,
    )
