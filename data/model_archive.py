"""Retired open-model releases kept outside the active planner catalog.

The picker and auto-selection only consume ``MODELS``.  These records preserve
why a historical key disappeared, where its immutable release can be recovered,
and which current deployment class supersedes it.  Only the small, explicitly
safe alias subset is migrated automatically when importing old scenarios.
"""

from dataclasses import dataclass

MODEL_ARCHIVE_CAPTURED_AT = "2026-09-02"


@dataclass(frozen=True)
class ArchivedModel:
    key: str
    name: str
    replaced_by: str
    source: str
    source_revision: str
    reason: str
    automatic_migration: bool = False


def _retired(
    key: str,
    name: str,
    replaced_by: str,
    source: str,
    revision: str,
    reason: str,
    *,
    automatic: bool = False,
) -> ArchivedModel:
    return ArchivedModel(key, name, replaced_by, source, revision, reason, automatic)


ARCHIVED_MODELS: dict[str, ArchivedModel] = {
    record.key: record
    for record in (
        _retired(
            "glm5",
            "GLM-5",
            "glm53",
            "https://huggingface.co/zai-org/GLM-5",
            "22ce68ad9a4707fad5356209fee6c54ae2c4cd87",
            "Superseded in the same 744B flagship deployment class.",
            automatic=True,
        ),
        _retired(
            "glm51",
            "GLM-5.1",
            "glm53",
            "https://huggingface.co/zai-org/GLM-5.1",
            "f5ed80ea7943de93bdb66223a637d7c3af6d8fe4",
            "Superseded in the same 744B flagship deployment class.",
            automatic=True,
        ),
        _retired(
            "glm52",
            "GLM-5.2",
            "glm53",
            "https://huggingface.co/zai-org/GLM-5.2",
            "b4734de4facf877f85769a911abafc5283eab3d9",
            "Superseded in the same 744B flagship deployment class.",
            automatic=True,
        ),
        _retired(
            "q27",
            "Qwen 3.5 27B",
            "qwen38-27b",
            "https://huggingface.co/Qwen/Qwen3.5-27B",
            "fc05daec18b0a78c049392ed2e771dde82bdf654",
            "Direct dense 27B replacement.",
            automatic=True,
        ),
        _retired(
            "q122",
            "Qwen 3.5 122B-A10B",
            "qwen38-flash-next",
            "https://huggingface.co/Qwen/Qwen3.5-122B-A10B",
            "dc4d348443bc740c68e2d77492492c11606384d5",
            "Superseded efficient large-MoE tier; footprint changed materially.",
        ),
        _retired(
            "q397",
            "Qwen 3.5 397B-A17B",
            "qwen38-2.4t-a95b",
            "https://huggingface.co/Qwen/Qwen3.5-397B-A17B",
            "8472618112abcbd45acbcdc58436aff4233c23f7",
            "Superseded open flagship tier; footprint changed materially.",
        ),
        _retired(
            "k25",
            "Kimi K2.5",
            "kimi-k3",
            "https://huggingface.co/moonshotai/Kimi-K2.5",
            "4d01dfe0332d63057c186e0b262165819efb6611",
            "Superseded flagship generation; footprint changed materially.",
        ),
        _retired(
            "minimax25",
            "MiniMax M2.5",
            "minimax3",
            "https://huggingface.co/MiniMaxAI/MiniMax-M2.5",
            "f710177d938eff80b684d42c5aa84b382612f21f",
            "Superseded flagship generation; footprint changed materially.",
        ),
        _retired(
            "minimax27",
            "MiniMax M2.7",
            "minimax3",
            "https://huggingface.co/MiniMaxAI/MiniMax-M2.7",
            "d494266a4affc0d2995ba1fa35c8481cbd84294b",
            "Superseded flagship generation; footprint changed materially.",
        ),
        _retired(
            "nem3n",
            "Nemotron 3 Nano 30B-A3B",
            "nemotron35-lightning",
            "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
            "Direct 30B-A3B agent-model successor.",
            automatic=True,
        ),
        _retired(
            "lfm2-2.6b",
            "LFM2 2.6B",
            "lfm2.5-2.6b",
            "https://huggingface.co/LiquidAI/LFM2-2.6B",
            "36ed799f7024ef169ba92ffe184821835e15cc66",
            "Direct 2.6B text-tier successor.",
            automatic=True,
        ),
        _retired(
            "lfm2-8b-a1b",
            "LFM2 8B-A1B",
            "lfm2.5-8b-a1b",
            "https://huggingface.co/LiquidAI/LFM2-8B-A1B",
            "c1c44ff9fc00db3ebf4516970563f5f383d23670",
            "Direct compact-MoE successor.",
            automatic=True,
        ),
        _retired(
            "command-a-03-2025",
            "Command A",
            "command-a-plus-05-2026",
            "https://huggingface.co/CohereLabs/command-a-03-2025",
            "",
            "Command A+ consolidates and supersedes the Command A tier.",
        ),
        _retired(
            "laguna-xs2",
            "Poolside Laguna XS.2",
            "laguna-xs-2-1",
            "https://huggingface.co/poolsideai/laguna-xs2",
            "",
            "Direct XS release-line successor.",
            automatic=True,
        ),
        _retired(
            "ms24",
            "Mistral Small 3.1 24B",
            "ms4",
            "https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "68faf511d618ef198fef186659617cfd2eb8e33a",
            "Superseded Mistral Small release line; new footprint differs.",
        ),
        _retired(
            "ms32",
            "Mistral Small 3.2 24B",
            "ms4",
            "https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "95a6d26c4bfb886c58daf9d3f7332c857cb27b43",
            "Superseded Mistral Small release line; new footprint differs.",
        ),
        _retired(
            "ml123",
            "Mistral Large 2 123B",
            "ml3",
            "https://huggingface.co/mistralai/Mistral-Large-Instruct-2407",
            "a286006d554cb37a61d13c7ae61bc90cc1d372fc",
            "Superseded Mistral Large release line; new footprint differs.",
        ),
        _retired(
            "mistral-medium-3.5-preview",
            "Mistral Medium 3.5 preview key",
            "mistral-medium-3.5",
            "https://huggingface.co/mistralai/Mistral-Medium-3.5-128B",
            "c4be198050fb5789774a55b92ed697becfbf20ae",
            "Released checkpoint replaces the provisional key.",
            automatic=True,
        ),
        _retired(
            "ds3",
            "DeepSeek V3",
            "deepseek-v4-pro",
            "https://huggingface.co/deepseek-ai/DeepSeek-V3",
            "e815299b0bcbac849fa540c768ef21845365c9eb",
            "Superseded open flagship generation; footprint changed materially.",
        ),
        _retired(
            "l70",
            "Llama 3.1 70B",
            "llama33-70b",
            "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
            "1605565b47bb9346c5515c34102e054115b4f98b",
            "Direct 70B deployment-tier successor.",
            automatic=True,
        ),
    )
}

MODEL_KEY_ALIASES = {
    key: record.replaced_by for key, record in ARCHIVED_MODELS.items() if record.automatic_migration
}
