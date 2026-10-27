"""Use-case routing families and explicit units for non-generative workloads."""

WORKLOAD_KINDS = {
    "llm": "LLM",
    "asr": "ASR",
    "embedding": "Embedding",
    "decision": "Decision",
}


def normalize_workload_kind(value) -> str:
    return value if isinstance(value, str) and value in WORKLOAD_KINDS else "llm"


def model_workload_kind(model) -> str:
    if model.is_decision_model:
        return "decision"
    if model.is_asr_model:
        return "asr"
    if model.is_embedding_model:
        return "embedding"
    return "llm"


def workload_unit(kind: str) -> str:
    return "audio seconds" if kind == "asr" else "input tokens" if kind != "llm" else "tokens"
