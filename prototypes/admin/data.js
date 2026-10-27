/* Synthetic sample data for the /admin prototypes.
 *
 * NOT production data. A seeded generator that mimics the shapes seen on
 * gpumodels.gskrd.me/admin (default "open" → session_reset → gpu_add …) and
 * adds the fields the proposals need but the app does not record yet:
 * country/city (Cloudflare CF-IPCountry / visitor-location headers), referrer +
 * UTM on first hit, device class from User-Agent, Accept-Language, model/qty/
 * precision edits, picker searches, fit results, exports and agent-API calls.
 *
 * Exposes window.ADMIN = { visitors, sessions, events, meta, catalog }.
 */
(function () {
  "use strict";

  // ---------- seeded PRNG ----------
  let seed = 20261017;
  function rnd() {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  const pick = (arr) => arr[Math.floor(rnd() * arr.length)];
  const between = (a, b) => a + rnd() * (b - a);
  const int = (a, b) => Math.floor(between(a, b + 1));
  function weighted(pairs) {
    let total = 0;
    for (const [, w] of pairs) total += w;
    let r = rnd() * total;
    for (const [v, w] of pairs) { if ((r -= w) <= 0) return v; }
    return pairs[pairs.length - 1][0];
  }
  const hex = (n) => Array.from({ length: n }, () => Math.floor(rnd() * 16).toString(16)).join("");

  // ---------- catalog (real names from data/) ----------
  // [key, display name, class, vram GB per device, vendor]
  const GPUS = [
    ["H100", "H100 80GB SXM", "datacenter", 80, "NVIDIA"],
    ["H200", "H200 141GB SXM", "datacenter", 141, "NVIDIA"],
    ["B200", "B200 180GB HGX/DGX", "datacenter", 180, "NVIDIA"],
    ["B300", "B300 Blackwell Ultra 288GB HGX/DGX", "datacenter", 288, "NVIDIA"],
    ["GB300", "GB300 NVL72 Blackwell Ultra 288GB/GPU", "datacenter", 288, "NVIDIA"],
    ["A100", "A100 80GB SXM", "datacenter", 80, "NVIDIA"],
    ["L40S", "L40S 48GB", "datacenter", 48, "NVIDIA"],
    ["L4", "L4 24GB", "datacenter", 24, "NVIDIA"],
    ["T4", "T4 16GB", "datacenter", 16, "NVIDIA"],
    ["MI300X", "MI300X 192GB", "datacenter", 192, "AMD"],
    ["MI355X", "MI355X 288GB", "datacenter", 288, "AMD"],
    ["Gaudi3", "Gaudi 3 128GB", "datacenter", 128, "Intel"],
    ["RTXPRO6000_BSE", "RTX PRO 6000 Blackwell Server Edition 96GB", "workstation", 96, "NVIDIA"],
    ["RTXPRO6000_BW_WS", "RTX PRO 6000 Blackwell Workstation 96GB", "workstation", 96, "NVIDIA"],
    ["RTX6000_ADA", "RTX 6000 Ada Generation 48GB", "workstation", 48, "NVIDIA"],
    ["A6000", "RTX A6000 48GB", "workstation", 48, "NVIDIA"],
    ["DGX_STATION_GB300", "DGX Station GB300 Blackwell Ultra 252GB", "workstation", 252, "NVIDIA"],
    ["RadeonAIProR9700", "Radeon AI PRO R9700 32GB", "workstation", 32, "AMD"],
    ["RTX5090", "GeForce RTX 5090 32GB", "consumer", 32, "NVIDIA"],
    ["RTX4090", "GeForce RTX 4090 24GB", "consumer", 24, "NVIDIA"],
    ["RTX3090", "GeForce RTX 3090 24GB", "consumer", 24, "NVIDIA"],
    ["RTX5060TI_16", "GeForce RTX 5060 Ti 16GB", "consumer", 16, "NVIDIA"],
    ["RadeonRX7900XTX", "Radeon RX 7900 XTX 24GB", "consumer", 24, "AMD"],
    ["ArcProB60", "Arc Pro B60 24GB", "consumer", 24, "Intel"],
    ["DGX_SPARK", "DGX Spark GB10 128GB", "unified", 128, "NVIDIA"],
    ["RYZEN_AI_MAX_395_128", "Ryzen AI Max+ 395 128GB system (96GB GPU budget)", "unified", 96, "AMD"],
    ["MAC_STUDIO_M5_MAX", "Mac Studio M5 Max 128GB", "unified", 128, "Apple"],
    ["MAC_STUDIO_M5_ULTRA", "Mac Studio M5 Ultra 512GB", "unified", 512, "Apple"],
    ["MAC_STUDIO_M3_ULTRA", "Mac Studio M3 Ultra 512GB", "unified", 512, "Apple"],
    ["MAC_MINI_M4_PRO", "Mac mini M4 Pro 64GB", "unified", 64, "Apple"],
  ];
  const GPU = Object.fromEntries(GPUS.map(([k, name, cls, vram, vendor]) => [k, { key: k, name, cls, vram, vendor }]));

  // [name, kind, total params B, family]
  const MODELS = [
    ["Llama 3.3 70B", "llm", 70, "Llama"],
    ["Llama 3.1 8B", "llm", 8, "Llama"],
    ["Gemma 4 31B", "llm", 31, "Gemma"],
    ["Gemma 4 26B-A4B", "llm", 26, "Gemma"],
    ["Gemma 4 12B Unified", "llm", 12, "Gemma"],
    ["Gemma 4 E4B", "llm", 8, "Gemma"],
    ["Qwen 3.5 35B-A3B", "llm", 35, "Qwen"],
    ["Qwen 3.8 27B", "llm", 27, "Qwen"],
    ["Qwen 3.5 9B", "llm", 9, "Qwen"],
    ["Qwen 3.8 Flash-Next 125B-A6B +51B N-gram", "llm", 176, "Qwen"],
    ["Qwen 3.8 2.4T-A95B", "llm", 2400, "Qwen"],
    ["Mistral Small 4 119B-A6.5B", "llm", 119, "Mistral"],
    ["Mistral Medium 3.5 128B", "llm", 128, "Mistral"],
    ["Mistral Large 3 675B-A41B", "llm", 675, "Mistral"],
    ["Devstral Small 2 24B", "llm", 24, "Mistral"],
    ["Devstral 2 123B", "llm", 123, "Mistral"],
    ["GLM-4.7-Flash 31B-A3B", "llm", 31, "GLM"],
    ["GLM-5.3-Flash 320B-A18B", "llm", 320, "GLM"],
    ["GLM-5.3 744B-A40B", "llm", 744, "GLM"],
    ["DeepSeek V4 Flash 0731 284B-A13B", "llm", 284, "DeepSeek"],
    ["DeepSeek V4.1 Flash 1017 552B + 196B Engram", "llm", 552, "DeepSeek"],
    ["DeepSeek V4 Pro 0813 1.6T-A49B", "llm", 1600, "DeepSeek"],
    ["Kimi K3 2.78T-A104.2B", "llm", 2780, "Kimi"],
    ["MiniMax M3 428B-A23B", "llm", 428, "MiniMax"],
    ["Nemotron 3 Super 120B-A12B", "llm", 120, "Nemotron"],
    ["Nemotron 3.5 Lightning 30B-A3B", "llm", 30, "Nemotron"],
    ["Nemotron 3 Nano Omni 30B-A3B", "llm", 30, "Nemotron"],
    ["MiMo-V2.5 310B-A15B", "llm", 310, "MiMo"],
    ["Hy4 Preview 770B-A49B", "llm", 770, "Hunyuan"],
    ["Inkling-Small 276B-A12B", "llm", 276, "Inkling"],
    ["Ternary Bonsai 2 27B", "llm", 27, "Bonsai"],
    ["LFM2.5 1.2B Instruct", "llm", 1.2, "LFM"],
    ["Granite 4.2 8B", "llm", 8, "Granite"],
    ["Command A+ 05-2026", "llm", 111, "Cohere"],
    ["Voxtral Mini Realtime 4B", "asr", 4, "Voxtral"],
    ["NVIDIA Parakeet TDT 0.6B v3", "asr", 0.6, "Parakeet"],
    ["NVIDIA Nemotron 3.5 ASR Streaming 0.6B", "asr", 0.6, "Nemotron"],
    ["Microsoft VibeVoice ASR Streaming 7B", "asr", 7, "VibeVoice"],
    ["Kyutai STT 2.6B EN", "asr", 2.6, "Kyutai"],
    ["Cohere Transcribe 03-2026 2B", "asr", 2, "Cohere"],
    ["Jais ASR 5.7B", "asr", 5.7, "Jais"],
    ["BGE-M3", "emb", 0.57, "BGE"],
    ["NVIDIA Nemotron 3 Embed 8B", "emb", 8, "Nemotron"],
    ["PPLX Embed v2 Late 9b", "emb", 9, "PPLX"],
    ["EmbeddingGemma 2 740M", "emb", 0.74, "Gemma"],
    ["DenseOn", "emb", 0.3, "DenseOn"],
  ];
  const MODEL = Object.fromEntries(MODELS.map(([name, kind, b, fam]) => [name, { name, kind, b, fam }]));
  const sizeClass = (b) => (b < 4 ? "tiny <4B" : b < 16 ? "small 4–16B" : b < 80 ? "mid 16–80B" : b < 400 ? "large 80–400B" : "frontier 400B+");
  const SIZE_CLASSES = ["tiny <4B", "small 4–16B", "mid 16–80B", "large 80–400B", "frontier 400B+"];
  const PREC_BYTES = { BF16: 2, FP8: 1, NVFP4: 0.56, MXFP4: 0.56, Q4_K_M: 0.6, INT8: 1 };

  const USE_CASES = ["Customer service agent", "Repository coding agent", "Doc summarization", "Enterprise search assistant",
    "Contact-center QA", "Invoice & claims extraction", "Deep research agent", "Mass classification", "Multilingual translation",
    "Contract review & redlining", "Meeting notes assistant", "Security alert investigation"];

  // Picker searches with no catalog match → catalog-gap signal.
  const MISSING_TERMS = [
    ["gpt-oss-120b", 14], ["Llama 4 Maverick", 9], ["Whisper large-v3", 11], ["RTX 5080", 8], ["Qwen3-Coder 480B", 7],
    ["TPU v6e", 5], ["Trainium2", 4], ["MacBook Pro M5 Max", 9], ["Jetson Orin Nano", 5], ["Llama 3.1 405B", 6],
    ["Phi-5", 4], ["RX 9070", 3], ["Groq LPU", 3], ["A100 40GB", 2], ["Arc B580", 4], ["Strix Halo 64GB", 3],
  ];

  // ---------- geography ----------
  // [iso2, name, iso numeric (world-atlas id), weight, utc offset h, lang, cities]
  const COUNTRIES = [
    ["US", "United States", "840", 22, -6, "en-US", ["San Francisco", "New York", "Seattle", "Austin", "Boston"]],
    ["IN", "India", "356", 9, 5.5, "en-IN", ["Bengaluru", "Hyderabad", "Pune", "Mumbai"]],
    ["DE", "Germany", "276", 7, 2, "de-DE", ["Berlin", "Munich", "Hamburg"]],
    ["GB", "United Kingdom", "826", 6, 1, "en-GB", ["London", "Cambridge", "Manchester"]],
    ["FR", "France", "250", 6, 2, "fr-FR", ["Paris", "Lyon", "Grenoble"]],
    ["CN", "China", "156", 5, 8, "zh-CN", ["Beijing", "Shanghai", "Shenzhen", "Hangzhou"]],
    ["CA", "Canada", "124", 4, -5, "en-CA", ["Toronto", "Montreal", "Vancouver"]],
    ["NL", "Netherlands", "528", 3, 2, "nl-NL", ["Amsterdam", "Eindhoven"]],
    ["JP", "Japan", "392", 3, 9, "ja-JP", ["Tokyo", "Osaka"]],
    ["KR", "South Korea", "410", 3, 9, "ko-KR", ["Seoul", "Pangyo"]],
    ["BR", "Brazil", "076", 3, -3, "pt-BR", ["São Paulo", "Rio de Janeiro"]],
    ["SG", "Singapore", "702", 2, 8, "en-SG", ["Singapore"]],
    ["AU", "Australia", "036", 2, 10, "en-AU", ["Sydney", "Melbourne"]],
    ["PL", "Poland", "616", 2, 2, "pl-PL", ["Warsaw", "Kraków"]],
    ["ES", "Spain", "724", 2, 2, "es-ES", ["Madrid", "Barcelona"]],
    ["IT", "Italy", "380", 2, 2, "it-IT", ["Milan", "Rome"]],
    ["CH", "Switzerland", "756", 1.5, 2, "de-CH", ["Zurich", "Lausanne"]],
    ["SE", "Sweden", "752", 1.5, 2, "sv-SE", ["Stockholm"]],
    ["IL", "Israel", "376", 1.5, 3, "he-IL", ["Tel Aviv"]],
    ["TW", "Taiwan", "158", 1.5, 8, "zh-TW", ["Taipei", "Hsinchu"]],
    ["AE", "United Arab Emirates", "784", 1.2, 4, "en-AE", ["Dubai", "Abu Dhabi"]],
    ["VN", "Vietnam", "704", 1, 7, "vi-VN", ["Hanoi", "Ho Chi Minh City"]],
    ["ID", "Indonesia", "360", 1, 7, "id-ID", ["Jakarta"]],
    ["TR", "Turkey", "792", 1, 3, "tr-TR", ["Istanbul"]],
    ["UA", "Ukraine", "804", 1, 3, "uk-UA", ["Kyiv"]],
    ["MX", "Mexico", "484", 1, -6, "es-MX", ["Mexico City"]],
    ["SA", "Saudi Arabia", "682", 0.8, 3, "ar-SA", ["Riyadh"]],
    ["FI", "Finland", "246", 0.6, 3, "fi-FI", ["Helsinki"]],
    ["NG", "Nigeria", "566", 0.5, 1, "en-NG", ["Lagos"]],
    ["ZA", "South Africa", "710", 0.5, 2, "en-ZA", ["Cape Town"]],
  ];
  const COUNTRY = Object.fromEntries(COUNTRIES.map(([iso, name, num, w, tz, lang, cities]) => [iso, { iso, name, num, tz, lang, cities }]));

  // ---------- personas (generator ground truth; pages re-derive via rules) ----------
  // Weights per persona, then country & channel bias.
  const PERSONA_KEYS = ["homelab", "buyer", "evaluator", "speech", "solutions", "agent", "bouncer"];
  const PERSONA_WEIGHT = { homelab: 26, buyer: 15, evaluator: 13, speech: 6, solutions: 8, agent: 7, bouncer: 25 };
  const COUNTRY_BIAS = {
    homelab: { US: 1.2, DE: 1.4, PL: 1.4, BR: 1.3, UA: 1.5, VN: 1.3, ID: 1.3, TR: 1.3, GB: 1.1 },
    buyer: { US: 1.3, SG: 2.2, AE: 2.6, SA: 2.6, IN: 1.3, JP: 1.4, KR: 1.3, CH: 1.5, GB: 1.2, FR: 1.2 },
    evaluator: { CN: 2.6, TW: 1.7, KR: 1.5, US: 1.1, IL: 1.6, FR: 1.2, CH: 1.3 },
    speech: { IN: 2.0, FR: 1.6, DE: 1.2, AE: 1.6, SA: 1.8, BR: 1.3 },
    solutions: { IN: 1.8, US: 1.2, GB: 1.3, NL: 1.5, CA: 1.2, DE: 1.1 },
    agent: { US: 2.0, IE: 1, NL: 1.2, DE: 1.2 },
    bouncer: { IN: 1.2, BR: 1.2, ID: 1.4, NG: 1.6, VN: 1.2, MX: 1.3 },
  };
  const CHANNELS = ["Direct", "Google", "Hacker News", "r/LocalLLaMA", "X / Twitter", "LinkedIn", "GitHub", "AI assistant referral", "Agent API (no browser)"];
  const CHANNEL_BY_PERSONA = {
    homelab: [["r/LocalLLaMA", 34], ["Google", 22], ["Hacker News", 12], ["X / Twitter", 10], ["GitHub", 8], ["Direct", 10], ["AI assistant referral", 4]],
    buyer: [["Google", 30], ["LinkedIn", 22], ["Direct", 26], ["AI assistant referral", 10], ["Hacker News", 6], ["X / Twitter", 6]],
    evaluator: [["X / Twitter", 24], ["Hacker News", 18], ["GitHub", 16], ["Direct", 18], ["r/LocalLLaMA", 12], ["Google", 12]],
    speech: [["Google", 34], ["GitHub", 18], ["Direct", 20], ["LinkedIn", 10], ["AI assistant referral", 10], ["X / Twitter", 8]],
    solutions: [["LinkedIn", 26], ["Google", 28], ["Direct", 28], ["AI assistant referral", 12], ["Hacker News", 6]],
    agent: [["Agent API (no browser)", 100]],
    bouncer: [["Google", 38], ["r/LocalLLaMA", 16], ["Hacker News", 14], ["X / Twitter", 14], ["Direct", 10], ["LinkedIn", 8]],
  };
  const DEVICE_BY_PERSONA = {
    homelab: [["desktop", 78], ["mobile", 20], ["tablet", 2]],
    buyer: [["desktop", 90], ["mobile", 8], ["tablet", 2]],
    evaluator: [["desktop", 92], ["mobile", 8]],
    speech: [["desktop", 90], ["mobile", 10]],
    solutions: [["desktop", 88], ["mobile", 10], ["tablet", 2]],
    agent: [["agent", 100]],
    bouncer: [["mobile", 55], ["desktop", 42], ["tablet", 3]],
  };
  const AGENT_UAS = [["Claude-User", 38], ["ChatGPT-User", 26], ["python-httpx", 14], ["Perplexity-User", 10], ["curl", 6], ["node-fetch", 6]];

  const HW_BY_PERSONA = {
    homelab: [["RTX4090", 14], ["RTX3090", 14], ["RTX5090", 16], ["MAC_STUDIO_M5_MAX", 9], ["MAC_STUDIO_M3_ULTRA", 6], ["MAC_STUDIO_M5_ULTRA", 6],
      ["DGX_SPARK", 9], ["RYZEN_AI_MAX_395_128", 8], ["RTX5060TI_16", 5], ["RadeonRX7900XTX", 4], ["ArcProB60", 3], ["MAC_MINI_M4_PRO", 5], ["RadeonAIProR9700", 4]],
    buyer: [["H100", 18], ["H200", 16], ["B200", 18], ["B300", 8], ["RTXPRO6000_BSE", 20], ["MI355X", 6], ["MI300X", 5], ["GB300", 4], ["L40S", 5], ["Gaudi3", 2]],
    evaluator: [["H200", 14], ["B200", 18], ["H100", 12], ["RTXPRO6000_BSE", 14], ["MAC_STUDIO_M5_ULTRA", 8], ["DGX_STATION_GB300", 6], ["MI355X", 6], ["GB300", 6]],
    speech: [["L4", 22], ["L40S", 18], ["T4", 14], ["RTX4090", 12], ["RTXPRO6000_BSE", 14], ["H100", 10], ["A100", 10]],
    solutions: [["H100", 20], ["RTXPRO6000_BSE", 22], ["L40S", 14], ["H200", 12], ["A100", 10], ["B200", 10], ["RTX6000_ADA", 6]],
    agent: [["H100", 20], ["RTX4090", 14], ["RTX5090", 12], ["MAC_STUDIO_M5_MAX", 10], ["B200", 10], ["DGX_SPARK", 10], ["H200", 8]],
  };
  const QTY_BY_CLASS = { consumer: [[1, 40], [2, 34], [4, 18], [8, 8]], unified: [[1, 68], [2, 22], [4, 10]], workstation: [[1, 14], [2, 22], [4, 34], [8, 30]], datacenter: [[1, 8], [2, 10], [4, 18], [8, 52], [16, 8], [72, 4]] };

  const MODELS_BY_PERSONA = {
    homelab: [["Qwen 3.5 35B-A3B", 16], ["Gemma 4 31B", 12], ["Qwen 3.8 27B", 12], ["Gemma 4 26B-A4B", 10], ["Llama 3.3 70B", 9], ["GLM-4.7-Flash 31B-A3B", 9],
      ["Devstral Small 2 24B", 8], ["Mistral Small 4 119B-A6.5B", 6], ["Qwen 3.8 Flash-Next 125B-A6B +51B N-gram", 5], ["DeepSeek V4 Flash 0731 284B-A13B", 4],
      ["Ternary Bonsai 2 27B", 4], ["Qwen 3.5 9B", 6], ["Gemma 4 E4B", 3], ["Llama 3.1 8B", 4]],
    buyer: [["Llama 3.3 70B", 10], ["Mistral Medium 3.5 128B", 8], ["DeepSeek V4.1 Flash 1017 552B + 196B Engram", 12], ["GLM-5.3-Flash 320B-A18B", 10], ["Nemotron 3 Super 120B-A12B", 10],
      ["Qwen 3.8 Flash-Next 125B-A6B +51B N-gram", 10], ["Mistral Large 3 675B-A41B", 6], ["Command A+ 05-2026", 6], ["Gemma 4 31B", 6], ["MiniMax M3 428B-A23B", 5], ["DeepSeek V4 Pro 0813 1.6T-A49B", 5]],
    evaluator: [["DeepSeek V4 Pro 0813 1.6T-A49B", 12], ["Kimi K3 2.78T-A104.2B", 10], ["GLM-5.3 744B-A40B", 12], ["Qwen 3.8 2.4T-A95B", 9], ["MiniMax M3 428B-A23B", 9], ["Hy4 Preview 770B-A49B", 7],
      ["MiMo-V2.5 310B-A15B", 7], ["DeepSeek V4.1 Flash 1017 552B + 196B Engram", 10], ["Inkling-Small 276B-A12B", 6], ["GLM-5.3-Flash 320B-A18B", 8], ["Mistral Large 3 675B-A41B", 5]],
    speech: [["NVIDIA Parakeet TDT 0.6B v3", 16], ["Voxtral Mini Realtime 4B", 16], ["NVIDIA Nemotron 3.5 ASR Streaming 0.6B", 14], ["Microsoft VibeVoice ASR Streaming 7B", 10],
      ["Kyutai STT 2.6B EN", 8], ["Cohere Transcribe 03-2026 2B", 8], ["Jais ASR 5.7B", 6], ["BGE-M3", 8], ["NVIDIA Nemotron 3 Embed 8B", 6], ["PPLX Embed v2 Late 9b", 4], ["EmbeddingGemma 2 740M", 4]],
    solutions: [["Llama 3.3 70B", 12], ["Gemma 4 31B", 12], ["Qwen 3.8 27B", 10], ["Mistral Small 4 119B-A6.5B", 12], ["Nemotron 3 Super 120B-A12B", 8], ["Command A+ 05-2026", 8],
      ["Granite 4.2 8B", 6], ["BGE-M3", 8], ["Devstral 2 123B", 6], ["GLM-5.3-Flash 320B-A18B", 6]],
    agent: [["Qwen 3.5 35B-A3B", 12], ["Llama 3.3 70B", 14], ["DeepSeek V4.1 Flash 1017 552B + 196B Engram", 12], ["Gemma 4 31B", 10], ["GLM-5.3-Flash 320B-A18B", 8], ["Kimi K3 2.78T-A104.2B", 6]],
  };
  const PREC_BY_PERSONA = {
    homelab: [["Q4_K_M", 46], ["NVFP4", 18], ["FP8", 18], ["BF16", 18]],
    buyer: [["FP8", 50], ["BF16", 26], ["NVFP4", 16], ["MXFP4", 8]],
    evaluator: [["FP8", 34], ["NVFP4", 26], ["MXFP4", 20], ["BF16", 20]],
    speech: [["BF16", 70], ["FP8", 20], ["INT8", 10]],
    solutions: [["FP8", 44], ["BF16", 40], ["NVFP4", 16]],
    agent: [["FP8", 40], ["BF16", 30], ["Q4_K_M", 20], ["NVFP4", 10]],
  };

  // ---------- time ----------
  const NOW = Date.UTC(2026, 9, 17, 17, 30);
  const DAY = 86400000;
  const START = NOW - 90 * DAY;
  // Traffic spikes: an HN front page and a r/LocalLLaMA thread.
  const SPIKES = [
    { at: Date.UTC(2026, 7, 28, 15), ch: "Hacker News", boost: 70, label: "Hacker News front page" },
    { at: Date.UTC(2026, 8, 17, 9), ch: "r/LocalLLaMA", boost: 45, label: "r/LocalLLaMA thread" },
    { at: Date.UTC(2026, 9, 9, 14), ch: "LinkedIn", boost: 22, label: "LinkedIn post" },
  ];

  function fitBytes(models, gpus) {
    const vram = gpus.reduce((s, g) => s + GPU[g.key].vram * g.qty, 0);
    const need = models.reduce((s, m) => s + MODEL[m.name].b * PREC_BYTES[m.prec] * 1.15, 0);
    return { vram, need, fits: need <= vram * 0.9 };
  }

  // ---------- generate ----------
  const visitors = [];
  const sessions = [];
  const events = [];

  function pickCountry(persona) {
    const bias = COUNTRY_BIAS[persona] || {};
    return weighted(COUNTRIES.map(([iso, , , w]) => [iso, w * (bias[iso] || 1)]));
  }
  function startTime(channel) {
    // Spike-aware arrival time, weekday heavy, recent growth.
    for (const sp of SPIKES) {
      if (sp.ch === channel && rnd() < 0.42) return sp.at + Math.pow(rnd(), 2.2) * 3 * DAY;
    }
    let t;
    do {
      t = START + Math.pow(rnd(), 0.72) * (NOW - START);
      const dow = new Date(t).getUTCDay();
      if ((dow === 0 || dow === 6) && rnd() < 0.35) continue;
      break;
    } while (true);
    return t;
  }
  function localHourBias(t, tz, persona) {
    // Nudge UTC time so local hour matches persona habits.
    const want = persona === "buyer" || persona === "solutions" ? int(9, 17) : persona === "homelab" ? weighted([[int(19, 23), 70], [int(10, 18), 30]]) : int(8, 23);
    const d = new Date(t);
    const local = (d.getUTCHours() + tz + 24) % 24;
    let delta = want - local;
    if (delta > 12) delta -= 24;
    if (delta < -12) delta += 24;
    return Math.min(NOW - 60000, t + delta * 3600000 + int(0, 59) * 60000);
  }

  function makeSession(v, t0, idx) {
    const s = { id: hex(6), visitor: v.id, start: t0, entry: idx === 0 ? v.entry : weighted([["/", 70], ["/use-cases", 10], ["/econ", 10], ["/?s=shared", 10]]), events: [], tab: hex(6) };
    let t = t0;
    const ev = (type, detail, gap) => {
      t += (gap == null ? int(4, 70) : gap) * 1000;
      const e = { t, type, ...detail };
      s.events.push(e);
      return e;
    };
    const p = v.persona;

    if (p === "agent") {
      const n = int(1, 6);
      for (let i = 0; i < n; i++) {
        const gk = weighted(HW_BY_PERSONA.agent);
        const mk = weighted(MODELS_BY_PERSONA.agent);
        const prec = weighted(PREC_BY_PERSONA.agent);
        const qty = weighted(QTY_BY_CLASS[GPU[gk].cls]);
        const endpoint = weighted([["POST /api/agent/plan", 55], ["GET /api/agent/catalog/models", 20], ["GET /api/agent/catalog/gpus", 15], ["GET /llms.txt", 10]]);
        const f = fitBytes([{ name: mk, prec }], [{ key: gk, qty }]);
        ev("api_call", { endpoint, gpu: gk, qty, model: mk, prec, fits: endpoint.startsWith("POST") ? f.fits : null, status: rnd() < 0.06 ? 422 : 200 }, int(1, 20));
      }
      s.end = t;
      return s;
    }

    ev("open", { path: s.entry, default: true }, 0);
    if (p === "bouncer") {
      if (rnd() < 0.35) ev("scroll", { depth: int(20, 60) }, int(5, 25));
      if (rnd() < 0.18) ev("search", { term: weighted(MISSING_TERMS), results: 0 }, int(6, 30));
      s.end = t + int(3, 40) * 1000;
      return s;
    }

    if (idx > 0 && rnd() < 0.45) {
      ev("scenario_restore", { from: "cookie" }, 1);
    } else {
      ev("session_reset", {}, int(3, 20));
    }

    // Hardware
    const gpus = [];
    let searched = false;
    const hwCount = 1;
    for (let i = 0; i < hwCount; i++) {
      if (rnd() < 0.07) { searched = true; ev("search", { term: weighted(MISSING_TERMS.filter(([x]) => /RTX|TPU|Train|Mac|Jetson|RX|Groq|A100|Arc|Strix/.test(x))), results: 0 }); }
      const gk = weighted(HW_BY_PERSONA[p]);
      let qty = weighted(QTY_BY_CLASS[GPU[gk].cls]);
      if (p === "homelab" && GPU[gk].cls === "unified") qty = 1;
      ev("gpu_add", { gpu: gk, qty: 8 > qty ? 8 : qty });
      if (qty !== 8) ev("gpu_qty", { gpu: gk, qty }, int(3, 15));
      gpus.push({ key: gk, qty });
    }
    if (p === "buyer" && rnd() < 0.5) ev("gpu_cost", { gpu: gpus[0].key, usd_hr: +(between(1.6, 6.5)).toFixed(2) });

    // Many people stop right after picking hardware (the dominant prod shape).
    const stopP = { homelab: 0.24, buyer: 0.2, evaluator: 0.08, solutions: 0.16, speech: 0.12 }[p] || 0;
    if (rnd() < stopP) {
      s.end = t + int(10, 90) * 1000;
      s.final = { gpus: gpus.slice(), models: [], fit: null };
      return s;
    }

    // Mode switch for speech/embedding
    if (p === "speech") ev("mode", { mode: rnd() < 0.7 ? "asr" : "embedding" });
    else if (rnd() < 0.02) ev("mode", { mode: "asr" });

    // Models
    const want = p === "evaluator" ? int(3, 8) : p === "buyer" ? int(1, 3) : p === "solutions" ? int(1, 3) : p === "speech" ? int(1, 4) : int(1, 3);
    const chosen = [];
    for (let i = 0; i < want; i++) {
      if (!searched && rnd() < (p === "homelab" ? 0.09 : 0.04)) {
        searched = true;
        ev("search", { term: weighted(MISSING_TERMS.filter(([x]) => !/RTX|TPU|Train|Mac|Jetson|RX|Groq|A100|Arc|Strix/.test(x))), results: 0 });
      }
      let name = weighted(MODELS_BY_PERSONA[p]);
      if (p === "speech") name = weighted(MODELS_BY_PERSONA.speech);
      if (chosen.find((m) => m.name === name)) continue;
      let prec = weighted(PREC_BY_PERSONA[p]);
      if (MODEL[name].kind !== "llm") prec = "BF16";
      const m = { name, prec };
      // Evaluators swap models in and out rather than stacking them.
      if (p === "evaluator" && chosen.length && rnd() < 0.75) {
        const old = chosen.shift();
        ev("model_remove", { model: old.name }, int(4, 20));
      }
      chosen.push(m);
      ev("model_add", { model: name, prec }, int(6, 60));
      const f = fitBytes(chosen, gpus);
      ev("result", { fits: f.fits, need_gb: Math.round(f.need), vram_gb: Math.round(f.vram) }, 1);
      if (!f.fits) {
        // Recovery attempts: quantize down, add hardware, or remove.
        const r = rnd();
        if (r < 0.38 && m.prec !== "NVFP4" && m.prec !== "Q4_K_M" && MODEL[name].kind === "llm") {
          m.prec = p === "homelab" ? "Q4_K_M" : "NVFP4";
          ev("model_prec", { model: name, prec: m.prec }, int(5, 30));
          const f2 = fitBytes(chosen, gpus);
          ev("result", { fits: f2.fits, need_gb: Math.round(f2.need), vram_gb: Math.round(f2.vram) }, 1);
        } else if (r < 0.62) {
          gpus[0].qty = Math.min(gpus[0].qty * 2, 72);
          ev("gpu_qty", { gpu: gpus[0].key, qty: gpus[0].qty }, int(5, 30));
          const f2 = fitBytes(chosen, gpus);
          ev("result", { fits: f2.fits, need_gb: Math.round(f2.need), vram_gb: Math.round(f2.vram) }, 1);
        } else if (r < 0.82) {
          chosen.pop();
          ev("model_remove", { model: name }, int(5, 30));
        } else {
          // Rage-ish: reset and try again
          if (rnd() < 0.5) ev("session_reset", {}, int(4, 12));
          break;
        }
      }
    }

    // Depth actions
    if ((p === "buyer" && rnd() < 0.62) || (p === "evaluator" && rnd() < 0.45) || rnd() < 0.08) {
      ev("compare_duplicate", {});
      const alt = weighted(HW_BY_PERSONA[p === "buyer" || p === "evaluator" ? p : "buyer"]);
      ev("compare_gpu", { side: "B", gpu: alt, qty: weighted(QTY_BY_CLASS[GPU[alt].cls]) });
    }
    if ((p === "buyer" && rnd() < 0.55) || (p === "solutions" && rnd() < 0.35) || rnd() < 0.05) ev("econ_view", { tab: weighted([["cloud vs own", 60], ["TCO", 40]]) });
    if (p === "solutions" || (p === "buyer" && rnd() < 0.1)) {
      const n = p === "solutions" ? int(1, 4) : 1;
      for (let i = 0; i < n; i++) ev("use_case_add", { use_case: pick(USE_CASES) });
      if (rnd() < 0.5) ev("projection_toggle", { on: true });
    }
    if (p === "evaluator" && rnd() < 0.6) ev("dist_slide", { tp: pick([2, 4, 8]), pp: pick([1, 2]), dp: pick([1, 2]) });
    if ((p === "evaluator" || p === "homelab") && rnd() < 0.35) ev("settings_mu", { kv_ctx: pick([32768, 65536, 131072, 262144]) });
    const exportP = { buyer: 0.4, solutions: 0.45, evaluator: 0.22, homelab: 0.1, speech: 0.18 }[p] || 0;
    if (rnd() < exportP) ev(weighted([["copy_link", 50], ["export_report", 35], ["scenario_export", 15]]), {});
    if (rnd() < 0.04) ev("feedback", { text: pick(["please add gpt-oss", "tokens/s for Mac seems high", "need multi-node", "add power cost", "concurrency per user?"]) });

    s.end = t + int(10, 120) * 1000;
    s.final = { gpus: gpus.slice(), models: chosen.slice(), fit: chosen.length ? fitBytes(chosen, gpus).fits : null };
    return s;
  }

  const N = 860;
  for (let i = 0; i < N; i++) {
    const persona = weighted(PERSONA_KEYS.map((k) => [k, PERSONA_WEIGHT[k]]));
    const iso = persona === "agent" ? weighted([["US", 60], ["DE", 10], ["NL", 10], ["IE", 0], ["SG", 8], ["FR", 6], ["GB", 6]]) : pickCountry(persona);
    const c = COUNTRY[iso];
    const channel = weighted(CHANNEL_BY_PERSONA[persona]);
    const device = weighted(DEVICE_BY_PERSONA[persona]);
    const v = {
      id: hex(8),
      persona_truth: persona,
      persona,
      country: iso,
      city: pick(c.cities),
      tz: c.tz,
      lang: rnd() < 0.78 ? c.lang : "en-US",
      device,
      os: device === "agent" ? "—" : device === "mobile" ? weighted([["iOS", 55], ["Android", 45]]) : weighted([["macOS", 46], ["Windows", 34], ["Linux", 20]]),
      ua: device === "agent" ? weighted(AGENT_UAS) : null,
      channel,
      utm: channel === "LinkedIn" && rnd() < 0.5 ? "utm_source=linkedin&utm_campaign=oct-launch" : null,
      entry: channel === "AI assistant referral" ? weighted([["/", 50], ["/use-cases", 25], ["/econ", 25]]) : persona === "solutions" ? weighted([["/", 60], ["/use-cases", 40]]) : persona === "buyer" && rnd() < 0.2 ? "/econ" : "/",
      internal: false,
      sessions: [],
    };
    let t = localHourBias(startTime(channel), v.tz, persona);
    const returnP = { homelab: 0.32, buyer: 0.5, evaluator: 0.55, speech: 0.35, solutions: 0.45, agent: 0.4, bouncer: 0.04 }[persona];
    let idx = 0;
    while (true) {
      const s = makeSession(v, t, idx);
      v.sessions.push(s);
      sessions.push(s);
      idx++;
      if (idx >= 7 || rnd() > returnP) break;
      t = s.end + between(0.3, 14) * DAY;
      if (t > NOW - 3600000) break;
      t = localHourBias(t, v.tz, persona);
    }
    visitors.push(v);
  }

  // The owner (internal) — matches the heavy a45934d3 visitor seen in prod.
  {
    const v = { id: "a45934d3", persona_truth: "internal", persona: "internal", country: "FR", city: "Paris", tz: 2, lang: "fr-FR", device: "desktop", os: "macOS", ua: null, channel: "Direct", utm: null, entry: "/", internal: true, sessions: [] };
    let t = START + 2 * DAY;
    for (let i = 0; i < 26; i++) {
      const s = makeSession({ ...v, persona: pick(["evaluator", "buyer", "speech", "homelab"]) }, t, i);
      s.visitor = v.id;
      v.sessions.push(s); sessions.push(s);
      t += between(1, 4) * DAY;
      if (t > NOW) break;
    }
    visitors.push(v);
  }

  for (const v of visitors) {
    v.first = v.sessions[0].start;
    v.last = v.sessions[v.sessions.length - 1].end;
    for (const s of v.sessions) for (const e of s.events) events.push({ ...e, visitor: v.id, session: s.id });
  }
  events.sort((a, b) => a.t - b.t);

  window.ADMIN = {
    meta: { now: NOW, start: START, spikes: SPIKES, generated: "seed 20261017 · synthetic", channels: CHANNELS },
    catalog: { GPU, MODEL, COUNTRY, COUNTRIES, SIZE_CLASSES, sizeClass, USE_CASES, PREC_BYTES },
    visitors,
    sessions,
    events,
  };
})();
