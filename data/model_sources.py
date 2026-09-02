"""Pinned primary-source snapshots for released open-model geometry."""

from dataclasses import dataclass

OPEN_MODEL_ARCHITECTURE_CAPTURED_AT = "2026-09-02"


@dataclass(frozen=True)
class ModelArchitectureSource:
    repository: str
    revision: str
    note: str = "Official release checkpoint and config."

    @property
    def config_url(self) -> str:
        return f"{self.repository}/blob/{self.revision}/config.json"


# These entries cover every model added or materially corrected in the
# September 2026 audit. Revisions are immutable official release snapshots.
OPEN_MODEL_ARCHITECTURE_SOURCES: dict[str, ModelArchitectureSource] = {
    "llama33-70b": ModelArchitectureSource(
        "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
        "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
    ),
    "lfm2.5-2.6b": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/LFM2.5-2.6B", "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
    ),
    "lfm2.5-8b-a1b": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/LFM2.5-8B-A1B", "5dd22602c2e9f6a097b1de4c4efe0658b605015c"
    ),
    "lfm2.5-vl-3b": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/LFM2.5-VL-3B", "a3af5799199acdd2a4f56ac4342816abb46c12a9"
    ),
    "qwen38-27b": ModelArchitectureSource(
        "https://huggingface.co/Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    ),
    "qwen38-flash-next": ModelArchitectureSource(
        "https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8",
        "236dfdf285828023ca3bcd3f37366c58a3469b13",
        "Official FP8 artifact; its config defines the main model, QSA, N-gram, and MTP geometry.",
    ),
    "qwen38-2.4t-a95b": ModelArchitectureSource(
        "https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B", "207bd685a7e3696cfaff12ded7c6a7ea0f88c996"
    ),
    "glm53": ModelArchitectureSource(
        "https://huggingface.co/zai-org/GLM-5.3", "187fb9fff6319062325ff825627ef6db084d9bc6"
    ),
    "glm53f": ModelArchitectureSource(
        "https://huggingface.co/zai-org/GLM-5.3-Flash", "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
    ),
    "nemotron35-lightning": ModelArchitectureSource(
        "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
    ),
    "deepseek-v4-pro": ModelArchitectureSource(
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro",
        "b5968e9190ef611bbf34a7229255be88a0e937c1",
    ),
    "deepseek-v4-flash": ModelArchitectureSource(
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash",
        "60d8d70770c6776ff598c94bb586a859a38244f1",
    ),
    "mistral-medium-3.5": ModelArchitectureSource(
        "https://huggingface.co/mistralai/Mistral-Medium-3.5-128B",
        "22b2b868a15677cfa6061277ed2f653d1349a9ab",
    ),
    "ms4": ModelArchitectureSource(
        "https://huggingface.co/mistralai/Mistral-Small-4-119B-2603",
        "a11f36bebf709121056b1dbcc943d1c6afbe494d",
    ),
    "ml3": ModelArchitectureSource(
        "https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512",
        "383ffea2c7d60dfd44ca960e8e691709d4fdb9cd",
    ),
    "granite42-3b": ModelArchitectureSource(
        "https://huggingface.co/ibm-granite/granite-4.2-3b",
        "b7e947307dd2efb3ad3b853b0e8a7e75f8ad4ac2",
    ),
    "granite42-8b": ModelArchitectureSource(
        "https://huggingface.co/ibm-granite/granite-4.2-8b",
        "41a8a2d41c54ef4a71741b3e62604f4caaec9295",
    ),
    "granite42-30b": ModelArchitectureSource(
        "https://huggingface.co/ibm-granite/granite-4.2-30b",
        "70fc514076785017d93087f3d8b0676426b6b355",
    ),
}
