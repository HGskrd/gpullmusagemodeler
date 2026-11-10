"""Pinned primary-source snapshots for released open-model geometry."""

from dataclasses import dataclass

OPEN_MODEL_ARCHITECTURE_CAPTURED_AT = "2026-11-10"


@dataclass(frozen=True)
class ModelArchitectureSource:
    repository: str
    revision: str
    note: str = "Official release checkpoint and config."
    config_path: str = "config.json"

    @property
    def config_url(self) -> str:
        return f"{self.repository}/blob/{self.revision}/{self.config_path}"


# These entries cover every model added or materially corrected in the
# September 2026 audit. Revisions are immutable official release snapshots.
OPEN_MODEL_ARCHITECTURE_SOURCES: dict[str, ModelArchitectureSource] = {
    "ms32": ModelArchitectureSource(
        "https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "95a6d26c4bfb886c58daf9d3f7332c857cb27b43",
        "Source identity retained from the 2026-09-02 archive snapshot; restored to the picker 2026-10-27.",
    ),
    "lfm3-1.7b": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/LFM3-1.7B",
        "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        "Captured 2026-10-27: card, config, modeling code and BF16 tensor headers; 256K context.",
    ),
    "lfm3-1.7b-base": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/LFM3-1.7B-Base",
        "545fc8f4c7b5fa9f405105274cd29c4f8f44b526",
        "Captured 2026-10-27: pre-trained base checkpoint; same tri-hybrid geometry, no agent capability claim.",
    ),
    "lfm3-vl-2b": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/LFM3-VL-2B",
        "f83d2b0a2296aaa277f394990a9b8b949ddf9691",
        "Captured 2026-10-27: 128000-position config, vision encoder and projector included in residency.",
    ),
    "lfm3-audio-2b": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/LFM3-Audio-2B",
        "9d49701c372d6011d900f6abb40c6b296fb706dd",
        "Captured 2026-10-27: card limits use to 32K despite 128000-position config; separate LiquidCodec included.",
    ),
    "lfm3-omni-2b": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/LFM3-Omni-2B",
        "ce2458cbce4ad970f92032ed94e3d18d30b04f9c",
        "Captured 2026-10-27: non-reasoning, card limits use to 32K; separate LiquidCodec included.",
    ),
    "d1-2b": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/d1-2B",
        "97c57047d5852d77e02dfac211f8768aff77a8b2",
        "Captured 2026-10-27: single-forward typed decisions; noncausal set attention approximated conservatively.",
    ),
    "d1-omni-2b": ModelArchitectureSource(
        "https://huggingface.co/LiquidAI/d1-Omni-2B",
        "4cccbfd5ba3405ee0f973134bd03e0d9d60ef1d8",
        "Captured 2026-10-27: single-forward decisions; complete released checkpoint, no speech codec required.",
    ),
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
    "muse-glimmer-30b": ModelArchitectureSource(
        "https://huggingface.co/meta-models/Muse-Glimmer-30B",
        "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        "Pinned base config and index: 52 layers, 39 local/13 global attention, 131,072-token context, and 29,776,626,688 indexed parameters including vision.",
    ),
    "hy4-preview": ModelArchitectureSource(
        "https://huggingface.co/tencent/Hy4-preview",
        "705d81ee51566a186d645b74c974d642ef2828fe",
        "Pinned preview config and model card; 770B/49B excludes the bundled MTP module, whose exact serving footprint remains unmodeled.",
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
    "deepseek-v4.1-flash": ModelArchitectureSource(
        "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "2026-09-10 report sections 2 and 4.2.1, config and indexed checkpoint. Fixed optimized FP4 global/FP8 local cache; host-resident Engram (placement corrected 2026-09-29 against https://arxiv.org/html/2609.19969v1 section 2.4.2 and https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml); conservative throughput proxies, no DSpark speedup.",
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
    "microsoft-vibevoice-asr-streaming-1.5b": ModelArchitectureSource(
        "https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B",
        "4262d23d8a539a6530cf64fbd0b1751ef9a30853",
        "Official BF16 streaming-ASR checkpoint; config and preprocessor config define the Qwen2.5 decoder and 22-frame plus four-frame-lookahead audio contract.",
    ),
    "microsoft-vibevoice-asr-streaming-7b": ModelArchitectureSource(
        "https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B",
        "60d858b518b4e19d404af3737f848fc185b30177",
        "Official BF16 streaming-ASR checkpoint; config and preprocessor config define the Qwen2.5 decoder and 22-frame plus four-frame-lookahead audio contract.",
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


# October release snapshots. Proxy caveats are also shown on model cards.
OPEN_MODEL_ARCHITECTURE_SOURCES.update(
    {
        "beam-1.0-501b-a23b": ModelArchitectureSource(
            "https://huggingface.co/reflection/Beam-1.0-501B-A23B",
            "78f8f654d40a40919b8e9954b40dd879ecb9a692",
            "Official release config, model card, and checkpoint metadata captured 2026-10-17.",
        ),
        "mimo-v2.6-flash-rl": ModelArchitectureSource(
            "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL",
            "dcd3fd83fb181622fbc606fbb1709901982a62c2",
            "Official release config, model card, and checkpoint metadata captured 2026-10-17.",
        ),
        "mimo-v2.6-pro-rl": ModelArchitectureSource(
            "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL",
            "b61f21852680ee02a29e946e18cf8697f4844061",
            "Official release config, model card, and checkpoint metadata captured 2026-10-17.",
        ),
        "mimo-v2.6-distill-qwen-9b": ModelArchitectureSource(
            "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
            "2367e865d009c13ac81713a2878291d33ab28177",
            "Official release config, model card, and checkpoint metadata captured 2026-10-17.",
        ),
        "ternary-bonsai-2-2b": ModelArchitectureSource(
            "https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf",
            "683f7be2629342a50987627b40881e54761a2430",
            "Official GGUF card pins the unchanged Qwen base architecture; packed sizes include PTQ1_0 language model and Q8 vision projector. Nominal parameter counts; custom-kernel throughput remains a proxy.",
            config_path="README.md",
        ),
        "ternary-bonsai-2-4b": ModelArchitectureSource(
            "https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf",
            "8d6557168168f91ea6697a4a49ed2ef622b53adb",
            "Official GGUF card pins the unchanged Qwen base architecture; packed sizes include PTQ1_0 language model and Q8 vision projector. Nominal parameter counts; custom-kernel throughput remains a proxy.",
            config_path="README.md",
        ),
        "ternary-bonsai-2-9b": ModelArchitectureSource(
            "https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf",
            "5b2abb02362efbed46bbe204014b02837778b2c2",
            "Official GGUF card pins the unchanged Qwen base architecture; packed sizes include PTQ1_0 language model and Q8 vision projector. Nominal parameter counts; custom-kernel throughput remains a proxy.",
            config_path="README.md",
        ),
        "ternary-bonsai-2-27b": ModelArchitectureSource(
            "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf",
            "b072e1d3b35a0a630cece372c2127528e0994386",
            "Official GGUF card pins the unchanged Qwen base architecture; packed sizes include PTQ1_0 language model and Q8 vision projector. Nominal parameter counts; custom-kernel throughput remains a proxy.",
            config_path="README.md",
        ),
        "lfm2.5-24b-a2b": ModelArchitectureSource(
            "https://huggingface.co/LiquidAI/LFM2.5-24B-A2B",
            "1769516ac588c4c558089c94f27d96b0dc67b183",
            "Official release config, model card, and checkpoint metadata captured 2026-10-17.",
        ),
        "mellum2.1-12b-a2.5b-thinking": ModelArchitectureSource(
            "https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking",
            "92ddae9fc7665e9f801d141d2e5a6b2caf2460c4",
            "Official release config, model card, and checkpoint metadata captured 2026-10-17.",
        ),
        "hy-memory-30b-a3b": ModelArchitectureSource(
            "https://huggingface.co/tencent/Hy-Memory-30B-A3B",
            "64a460a94d5498b7316fb02395c72d01c47a7347",
            "Official release config, model card, and checkpoint metadata captured 2026-10-17.",
        ),
        "nemotron35-super-vl": ModelArchitectureSource(
            "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
            "1f8953f7f0935358e3605150d22bc15df062061c",
            "Official release config, model card, and checkpoint metadata captured 2026-10-17.",
        ),
        "nemotron35-bolt-74b": ModelArchitectureSource(
            "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
            "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a",
            "Official release config, model card, and checkpoint metadata captured 2026-10-17.",
        ),
        "muse-spark-open-816b-a42b": ModelArchitectureSource(
            "https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B",
            "0fb02478b02565b2ce7fc57e6051c91172774ac9",
            "Gated config/index unavailable. Pinned public official README and Hub tensor metadata support core geometry and total bytes; full-global KV is an explicit conservative proxy pending config access.",
            config_path="README.md",
        ),
        "pplx-embed-v2-late-0.6b": ModelArchitectureSource(
            "https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b",
            "8fc2de24534aa3610d85fa59c463313a5f096455",
            "Pinned config and sentence_bert_config: pruned 0.6B has 6 linear/6 global layers; serving lengths 1024 query / 4096 document. Full FP32 inventory includes vision; text-input throughput proxy.",
        ),
        "pplx-embed-v2-late-9b": ModelArchitectureSource(
            "https://huggingface.co/perplexity-ai/pplx-embed-v2-late-9b",
            "0f49a9977fe06b83377d598094c5c0204ce18ad9",
            "Pinned config and sentence_bert_config: pruned 0.6B has 6 linear/6 global layers; serving lengths 1024 query / 4096 document. Full FP32 inventory includes vision; text-input throughput proxy.",
        ),
        "embeddinggemma-2": ModelArchitectureSource(
            "https://huggingface.co/google/embeddinggemma-2",
            "914f7f89142e33e77833254d9c9b90c3cef7303b",
            "Pinned config overrides inconsistent prose window/head dimensions; full 744M multimodal inventory, 8k served cap, 768d output.",
        ),
        "jais-asr": ModelArchitectureSource(
            "https://huggingface.co/IFM/Jais-ASR",
            "1323995fa189c479369a3929be0e95039eba5cba",
            "Pinned Jais 5.718B checkpoint; 36-layer GQA text decoder + 32-layer Qwen3-Omni audio encoder. Offline capacity is a conservative 25 audio-frame/s proxy, not measured RTF.",
        ),
    }
)

OPEN_MODEL_ARCHITECTURE_SOURCES.update(
    {
        "d1-3b": ModelArchitectureSource(
            "https://huggingface.co/LiquidAI/d1-3B",
            "051bcc464b01b9f92942b364d9586b0ef5912432",
            "Text-only throughput proxy; vision weights included, image preprocessing unmodeled.",
        ),
        "d1-omni-600m": ModelArchitectureSource(
            "https://huggingface.co/LiquidAI/d1-omni-600M",
            "02b55d7076f15129e59ab3f94783f32c4b088674",
            "Experimental. Released combined-position cap 16,384; text-only BF16 compute proxy with native FP32 storage. Image/audio encoder latency unmodeled.",
        ),
        "pplx-decider-v1.1-27b": ModelArchitectureSource(
            "https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b",
            "14c9d90137962d300f3de95633f7b6535e80c5ce",
            "Noncausal full-attention layers, native linear layers, calibrated readout. Config positional ceiling 262,144; production long-context performance uncalibrated. Custom inference and checkpoint access may be required.",
        ),
        "clef": ModelArchitectureSource(
            "https://huggingface.co/Cloudflare/clef",
            "0b331204bb13fbd2ca93a64df1956f5b55478ce5",
            "Released encode_record default max_length 16,384; text-only estimate, image latency unmodeled.",
        ),
        "clef-flash": ModelArchitectureSource(
            "https://huggingface.co/Cloudflare/clef-flash",
            "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
            "Released encode_record default max_length 16,384; text-only estimate, image latency unmodeled.",
        ),
        "mapika-decider-2b": ModelArchitectureSource(
            "https://huggingface.co/Mapika/decider-2b",
            "ff99089b6810691bd79134ec6febb5d9afa6e619",
            "Released 32k serving cap includes state, questions and answer options.",
        ),
        "mapika-decider-4b": ModelArchitectureSource(
            "https://huggingface.co/Mapika/decider-4b",
            "99d7fd5bfcc4d11ba37239dafe0ed653803819d3",
            "Released 32k serving cap includes state, questions and answer options.",
        ),
        "mapika-decider-35b-a3b": ModelArchitectureSource(
            "https://huggingface.co/Mapika/decider-35b-a3b",
            "91470e65478656eb68fde8c0ccdfe575d69d0309",
            "Released 32k serving cap. Rounded 3B active-parameter proxy; expert routing and multimodal latency uncalibrated.",
        ),
        "kev-27b": ModelArchitectureSource(
            "https://huggingface.co/jaredpalmer/kev-27b",
            "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
            "Conservative 65,536-position validated serving envelope; pointer head, prefill only. Questions consume the same input budget.",
        ),
        "intern-decision-0.8b": ModelArchitectureSource(
            "https://huggingface.co/internlm/Intern-Decision-0.8B",
            "85a0cc5a99d67ea8d56dfe98115689212867171d",
            "DecisionEngine default max_length 8,192 includes image positions. Scored fields are not generated tokens.",
        ),
        "intern-decision-2b": ModelArchitectureSource(
            "https://huggingface.co/internlm/Intern-Decision-2B",
            "8797836c65fc91a2435b1fb6850b5f0aabd75cc3",
            "DecisionEngine default max_length 8,192 includes image positions. Scored fields are not generated tokens.",
        ),
        "intern-decision-4b": ModelArchitectureSource(
            "https://huggingface.co/internlm/Intern-Decision-4B",
            "1e65a988dacbcf00b84aee7366f9e47563407000",
            "DecisionEngine default max_length 8,192 includes image positions. Scored fields are not generated tokens.",
        ),
    }
)

OPEN_MODEL_ARCHITECTURE_SOURCES["pplx-decider-flash-v1.2"] = ModelArchitectureSource(
    "https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2",
    "6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d",
    "Qwen3.5-4B backbone with joint per-question marker readout; noncausal full attention and native causal linear layers. Released max_length defaults to 262,144 positions, including images and question/option templates; training covered about 32k. Longer inputs require task validation. Recursive multi-question splitting, image encoder latency, compilation, CUDA graphs and optional FP8 fast-path gains are not simulated. Text-only closed-form capacity estimate, not a benchmark; quality uncalibrated and no generation economics.",
)


# Provenance backfill. These references do not claim a fresh geometry/calibration audit.
_PROVENANCE_BACKFILL_NOTE = "Captured 2026-11-10: official Hub release identity and pinned metadata; provenance backfill only, existing geometry and throughput assumptions were not re-audited."
OPEN_MODEL_ARCHITECTURE_SOURCES.update(
    {
        "nvidia-nemotron-3-embed-8b": ModelArchitectureSource(
            "https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16",
            "d1f2f25730bbd775b99b29185134bc86653bf2d1",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "nvidia-nemotron-3-embed-1b": ModelArchitectureSource(
            "https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16",
            "c0c9fea93ea424587517f2c59e20db9f1d6bf615",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "denseon": ModelArchitectureSource(
            "https://huggingface.co/lightonai/DenseOn",
            "cb9947ebccb33862d24e3c7ca2edb25e51acd887",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "lateon": ModelArchitectureSource(
            "https://huggingface.co/lightonai/LateOn",
            "62911e105059585d244384c7d17826e35f669c17",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "bge-m3": ModelArchitectureSource(
            "https://huggingface.co/BAAI/bge-m3",
            "5617a9f61b028005a4858fdac845db406aefb181",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "mxbai-embed-large-v1": ModelArchitectureSource(
            "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1",
            "b33106f585b9ce46904ad7443a3b52b7a63e231c",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "mxbai-embed-2d-large-v1": ModelArchitectureSource(
            "https://huggingface.co/mixedbread-ai/mxbai-embed-2d-large-v1",
            "416c45d154ee4e435b6d19f6bd54440f38a12ee9",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "mxbai-embed-xsmall-v1": ModelArchitectureSource(
            "https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1",
            "e6ac24e5d6efb8782b59de1647b3ececb4ece94e",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "deepset-mxbai-embed-de-large-v1": ModelArchitectureSource(
            "https://huggingface.co/mixedbread-ai/deepset-mxbai-embed-de-large-v1",
            "cbbec43ebdf8b2d8304d3093781eee56a8cdfb69",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "mxbai-edge-colbert-v0-17m": ModelArchitectureSource(
            "https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m",
            "592c6417c1c6687572043408ed1ae5196bce16b1",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "mxbai-edge-colbert-v0-32m": ModelArchitectureSource(
            "https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m",
            "bb13a29ec9b1e7edd4ba8f7a0776c48b55cbad66",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "modernbert-embed-base": ModelArchitectureSource(
            "https://huggingface.co/nomic-ai/modernbert-embed-base",
            "d556a88e332558790b210f7bdbe87da2fa94a8d8",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "kalm-mini-it-v15": ModelArchitectureSource(
            "https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5",
            "fcff2f8a54e4cd96b7766fef1ee960a43d42bb3c",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "pplx-embed-v1-0.6b": ModelArchitectureSource(
            "https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b",
            "2c4d510dd4a732063c31a0f70193e35067b51fd8",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "pplx-embed-v1-4b": ModelArchitectureSource(
            "https://huggingface.co/perplexity-ai/pplx-embed-v1-4b",
            "06456497a00540a582918fe8dcd3a5eabb207772",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "l8": ModelArchitectureSource(
            "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
            "0e9e39f249a16976918f6564b8830bc894c89659",
            note=_PROVENANCE_BACKFILL_NOTE
            + " Config access is gated; its contents were not inspected.",
            config_path="config.json",
        ),
        "ge2": ModelArchitectureSource(
            "https://huggingface.co/google/gemma-4-E2B-it",
            "4adf15748de9e1a0097b6274cee9c15836c41c37",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "ge4": ModelArchitectureSource(
            "https://huggingface.co/google/gemma-4-E4B-it",
            "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "g12": ModelArchitectureSource(
            "https://huggingface.co/google/gemma-4-12B-it",
            "43c9d014b231474fa7212dbb48a762566c2b255d",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "g26": ModelArchitectureSource(
            "https://huggingface.co/google/gemma-4-26B-A4B-it",
            "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "g31": ModelArchitectureSource(
            "https://huggingface.co/google/gemma-4-31B-it",
            "842da3794eaa0b77d5f08bae87a17459d91ff475",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "lfm2.5-350m": ModelArchitectureSource(
            "https://huggingface.co/LiquidAI/LFM2.5-350M",
            "9921cd4d82c0e3986887aa4b376f9446729ef8a3",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "lfm2.5-1.2b-instruct": ModelArchitectureSource(
            "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct",
            "b9d816ab510f9a28dca37cb182e9da8b0226e13c",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "lfm2.5-1.2b-thinking": ModelArchitectureSource(
            "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking",
            "fa4681e33c27d2861d1eacc9d29ccb68840344a0",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "q08": ModelArchitectureSource(
            "https://huggingface.co/Qwen/Qwen3.5-0.8B",
            "2fc06364715b967f1860aea9cf38778875588b17",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "q2": ModelArchitectureSource(
            "https://huggingface.co/Qwen/Qwen3.5-2B",
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "q4": ModelArchitectureSource(
            "https://huggingface.co/Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "q9": ModelArchitectureSource(
            "https://huggingface.co/Qwen/Qwen3.5-9B",
            "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "q35": ModelArchitectureSource(
            "https://huggingface.co/Qwen/Qwen3.5-35B-A3B",
            "59d61f3ce65a6d9863b86d2e96597125219dc754",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "glm45a": ModelArchitectureSource(
            "https://huggingface.co/zai-org/GLM-4.5-Air",
            "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "glm47": ModelArchitectureSource(
            "https://huggingface.co/zai-org/GLM-4.7",
            "602d01efcdd332c5238ca4bcede555defbe83eb7",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "glm47f": ModelArchitectureSource(
            "https://huggingface.co/zai-org/GLM-4.7-Flash",
            "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "kimi-k3": ModelArchitectureSource(
            "https://huggingface.co/moonshotai/Kimi-K3",
            "f831ab66814297da540d832a5235f8e904f29d06",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "kimi-linear-48b": ModelArchitectureSource(
            "https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct",
            "e1df551a447157d4658b573f9a695d57658590e9",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "inkling": ModelArchitectureSource(
            "https://huggingface.co/thinkingmachines/Inkling",
            "828496eeae4c243ff1a22f7f28ff83694f2f7bc9",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "inkling-small-preview": ModelArchitectureSource(
            "https://huggingface.co/thinkingmachines/Inkling-Small",
            "8cc5877b44d343f88b92086aa1fb72897950f06a",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "command-a-plus-05-2026": ModelArchitectureSource(
            "https://huggingface.co/CohereLabs/command-a-plus-05-2026",
            "5fb6fde5fd12ff89356aae552e11883bc49f069b",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "command-r7b-12-2024": ModelArchitectureSource(
            "https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024",
            "4f3d0aa6856e322f2f4480fe65420d5d53d297b8",
            note=_PROVENANCE_BACKFILL_NOTE
            + " Config access is gated; its contents were not inspected.",
            config_path="config.json",
        ),
        "north-mini-code-1-0": ModelArchitectureSource(
            "https://huggingface.co/CohereLabs/north-mini-code-1.0",
            "d11e61a842617a22dc328552fa5bb86231ee4f37",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "tiny-aya-global": ModelArchitectureSource(
            "https://huggingface.co/CohereLabs/tiny-aya-global",
            "af89d219b53ed9b13b8a4645f9c8028973510324",
            note=_PROVENANCE_BACKFILL_NOTE
            + " Config access is gated; its contents were not inspected.",
            config_path="config.json",
        ),
        "tiny-aya-earth": ModelArchitectureSource(
            "https://huggingface.co/CohereLabs/tiny-aya-earth",
            "0624fafbe200f930b89566fc4ec4da314ba48d89",
            note=_PROVENANCE_BACKFILL_NOTE
            + " Config access is gated; its contents were not inspected.",
            config_path="config.json",
        ),
        "tiny-aya-fire": ModelArchitectureSource(
            "https://huggingface.co/CohereLabs/tiny-aya-fire",
            "73500c80fcb6c8bea3e830c21d5d2920249a0e1a",
            note=_PROVENANCE_BACKFILL_NOTE
            + " Config access is gated; its contents were not inspected.",
            config_path="config.json",
        ),
        "tiny-aya-water": ModelArchitectureSource(
            "https://huggingface.co/CohereLabs/tiny-aya-water",
            "41a99216d12b3ef4881df762934186e5e4d2bbc1",
            note=_PROVENANCE_BACKFILL_NOTE
            + " Config access is gated; its contents were not inspected.",
            config_path="config.json",
        ),
        "minimax3": ModelArchitectureSource(
            "https://huggingface.co/MiniMaxAI/MiniMax-M3",
            "d95e30cb912fe3021bb9c2ba66a8ce60b6c937a9",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "nem3s": ModelArchitectureSource(
            "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
            "2dc98e2afe4face0e4ce40972a915c45368bd34a",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "nem3no": ModelArchitectureSource(
            "https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
            "e5e9932441de940c9a62185c870ea5bcd4cd24e2",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "mi7": ModelArchitectureSource(
            "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
            "c170c708c41dac9275d15a8fff4eca08d52bab71",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "mx87": ModelArchitectureSource(
            "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
            "eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "cs22": ModelArchitectureSource(
            "https://huggingface.co/mistralai/Codestral-22B-v0.1",
            "28b1c1a51dabe9d86ca8c41420ada1984632498f",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "voxtral-realtime-mini-4b": ModelArchitectureSource(
            "https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602",
            "2769294da9567371363522aac9bbcfdd19447add",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "mimo-v2.5-asr": ModelArchitectureSource(
            "https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR",
            "227ce4c5e64e356007ad948781f0e8b55a1beba0",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "nvidia-nemotron-speech-streaming-0.6b": ModelArchitectureSource(
            "https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b",
            "ebe59e5a817142986528bbbee5dba8db7b38ed50",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "nvidia-nemotron-3.5-asr-streaming-0.6b": ModelArchitectureSource(
            "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
            "ea30d66debe3740a08b573244286791d423d6b3e",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "nvidia-parakeet-unified-0.6b": ModelArchitectureSource(
            "https://huggingface.co/nvidia/parakeet-unified-en-0.6b",
            "fe53cd885760c96b6a5f51a0bfd362cb4584a98b",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="README.md",
        ),
        "nvidia-parakeet-realtime-eou-120m": ModelArchitectureSource(
            "https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1",
            "a7e2b4629593dce0ec19f600e00e9904353fda2d",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="README.md",
        ),
        "nvidia-multitalker-parakeet-streaming-0.6b": ModelArchitectureSource(
            "https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1",
            "8749fc71fd6e2d88ef230159bbf2aea69b524ee1",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="README.md",
        ),
        "kyutai-stt-1b-en-fr": ModelArchitectureSource(
            "https://huggingface.co/kyutai/stt-1b-en_fr",
            "1c34c6b4f7e9299bb61985f145052ff131005dde",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "kyutai-stt-2.6b-en": ModelArchitectureSource(
            "https://huggingface.co/kyutai/stt-2.6b-en",
            "a07aec56d22be5589cd0bc8709c75b6cf3e3039d",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "moonshine-streaming-tiny": ModelArchitectureSource(
            "https://huggingface.co/UsefulSensors/moonshine-streaming-tiny",
            "f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "moonshine-streaming-small": ModelArchitectureSource(
            "https://huggingface.co/UsefulSensors/moonshine-streaming-small",
            "2c036506f23a09c18df5a50057599ba6d9280999",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "moonshine-streaming-medium": ModelArchitectureSource(
            "https://huggingface.co/UsefulSensors/moonshine-streaming-medium",
            "57b843633a8c183cadf6699ffa761377a933a866",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "fun-asr-nano-2512": ModelArchitectureSource(
            "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512",
            "272c57b82523ada6fd87095e955f8e29100979ab",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="README.md",
        ),
        "granite-4.0-1b-speech": ModelArchitectureSource(
            "https://huggingface.co/ibm-granite/granite-4.0-1b-speech",
            "bd87ab862416353633ea431fe49b1614003623c5",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "cohere-transcribe-03-2026": ModelArchitectureSource(
            "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026",
            "d552dadc2922ba6f6c1f994f8db904d52dcaf942",
            note=_PROVENANCE_BACKFILL_NOTE
            + " Config access is gated; its contents were not inspected.",
            config_path="config.json",
        ),
        "nvidia-parakeet-tdt-0.6b-v3": ModelArchitectureSource(
            "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
            "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "n3": ModelArchitectureSource(
            "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512",
            "b35d4dfe56c142746f54dbd64f579faab2744308",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "n8": ModelArchitectureSource(
            "https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512",
            "5b26027e7b19eeb4b7352e1fed3926375dd2cb4d",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "n14": ModelArchitectureSource(
            "https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512",
            "29439f81c2be264d8d393273f99e7db9c0961120",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "dv24": ModelArchitectureSource(
            "https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512",
            "55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "dv123": ModelArchitectureSource(
            "https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512",
            "1613bf01adb5e1c6fdc196b46e6b173eae75eb4a",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "zaya1-8b": ModelArchitectureSource(
            "https://huggingface.co/Zyphra/ZAYA1-8B",
            "67d34da515b30409ee8daa67c3605c4402d03f6f",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "laguna-m1": ModelArchitectureSource(
            "https://huggingface.co/poolside/Laguna-M.1",
            "2bf8a4ab6d2e8e1e0d92a41318532a0fee1640d6",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "laguna-xs-2-1": ModelArchitectureSource(
            "https://huggingface.co/Poolside/laguna-xs-2.1",
            "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "laguna-s-2-1": ModelArchitectureSource(
            "https://huggingface.co/Poolside/laguna-s-2.1",
            "0f573140834b11cfac0c2af97a101a7a69a13e22",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "cr13": ModelArchitectureSource(
            "https://huggingface.co/croissantllm/CroissantLLMChat-v0.1",
            "f05665c340c868ce48cebdf997cfafb7dbfef2b3",
            note=_PROVENANCE_BACKFILL_NOTE,
            config_path="config.json",
        ),
        "inkling-asr": ModelArchitectureSource(
            "https://huggingface.co/thinkingmachines/Inkling",
            "828496eeae4c243ff1a22f7f28ff83694f2f7bc9",
            note=_PROVENANCE_BACKFILL_NOTE
            + " ASR variant of inkling; backbone source, not an independent ASR benchmark.",
            config_path="config.json",
        ),
        "inkling-small-asr": ModelArchitectureSource(
            "https://huggingface.co/thinkingmachines/Inkling-Small",
            "8cc5877b44d343f88b92086aa1fb72897950f06a",
            note=_PROVENANCE_BACKFILL_NOTE
            + " ASR variant of inkling-small-preview; backbone source, not an independent ASR benchmark.",
            config_path="config.json",
        ),
        "gemma-4-e2b-asr": ModelArchitectureSource(
            "https://huggingface.co/google/gemma-4-E2B-it",
            "4adf15748de9e1a0097b6274cee9c15836c41c37",
            note=_PROVENANCE_BACKFILL_NOTE
            + " ASR variant of ge2; backbone source, not an independent ASR benchmark.",
            config_path="config.json",
        ),
        "gemma-4-e4b-asr": ModelArchitectureSource(
            "https://huggingface.co/google/gemma-4-E4B-it",
            "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
            note=_PROVENANCE_BACKFILL_NOTE
            + " ASR variant of ge4; backbone source, not an independent ASR benchmark.",
            config_path="config.json",
        ),
        "gemma-4-12b-unified-asr": ModelArchitectureSource(
            "https://huggingface.co/google/gemma-4-12B-it",
            "43c9d014b231474fa7212dbb48a762566c2b255d",
            note=_PROVENANCE_BACKFILL_NOTE
            + " ASR variant of g12; backbone source, not an independent ASR benchmark.",
            config_path="config.json",
        ),
    }
)

# Hosted/proxy entries must explain why immutable open architecture is unavailable.
UNPUBLISHED_MODEL_ARCHITECTURE_SOURCES = {
    "cohere-embed-v4-0": {
        "repository": "https://docs.cohere.com/v1/docs/cohere-embed",
        "revision": None,
        "note": "Captured 2026-11-10: hosted model; no public checkpoint/config is available. Catalog geometry and local capacity are proxies, not a deployable Cohere artifact.",
    },
    "mm31": {
        "repository": "https://docs.mistral.ai/models/mistral-medium-3-1-25-08",
        "revision": None,
        "note": "Captured 2026-11-10: legacy hosted model; no public checkpoint/config is available. Retained catalog geometry is a capacity proxy, not verified architecture.",
    },
}
