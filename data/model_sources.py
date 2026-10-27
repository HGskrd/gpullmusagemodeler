"""Pinned primary-source snapshots for released open-model geometry."""

from dataclasses import dataclass

OPEN_MODEL_ARCHITECTURE_CAPTURED_AT = "2026-10-17"


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
