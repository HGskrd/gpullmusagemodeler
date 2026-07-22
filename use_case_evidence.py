"""Source registry and non-editable evidence for built-in use-case scenarios.

The planner values remain hypotheses to calibrate with deployment telemetry.  Sources
support the existence, workload shape, or economic anchor; they are not presented as
published measurements of every seed value.
"""

USE_CASE_RESEARCH_CAPTURED_AT = "2026-07-22"

USE_CASE_SOURCES = {
    "openai_use_cases": {
        "title": "Identifying and scaling AI use cases",
        "publisher": "OpenAI",
        "published": "2025",
        "url": "https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf",
    },
    "openai_batch": {
        "title": "Batch API reference",
        "publisher": "OpenAI",
        "published": "current documentation",
        "url": "https://platform.openai.com/docs/api-reference/batch/object?api-mode=responses",
    },
    "openai_tokens": {
        "title": "What are tokens and how to count them?",
        "publisher": "OpenAI",
        "published": "current help article",
        "url": "https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them",
    },
    "openai_deep_research": {
        "title": "Introducing deep research",
        "publisher": "OpenAI",
        "published": "2025; updated 2026",
        "url": "https://openai.com/index/introducing-deep-research/",
    },
    "openai_structured_outputs": {
        "title": "Introducing Structured Outputs in the API",
        "publisher": "OpenAI",
        "published": "2024",
        "url": "https://openai.com/index/introducing-structured-outputs-in-the-api/",
    },
    "openai_vector_stores": {
        "title": "Retrieval and vector stores",
        "publisher": "OpenAI",
        "published": "current documentation",
        "url": "https://platform.openai.com/docs/guides/retrieval",
    },
    "github_copilot": {
        "title": "GitHub Copilot licenses",
        "publisher": "GitHub",
        "published": "current documentation",
        "url": "https://docs.github.com/en/billing/concepts/product-billing/github-copilot-licenses",
    },
    "intercom_fin": {
        "title": "Intercom pricing",
        "publisher": "Intercom",
        "published": "current pricing",
        "url": "https://www.intercom.com/pricing",
    },
    "transcript_rate": {
        "title": "Large-scale analysis of conversational speech rate",
        "publisher": "PubMed Central",
        "published": "2025",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12551936/",
    },
    "exchange_limits": {
        "title": "Exchange Online limits",
        "publisher": "Microsoft Learn",
        "published": "current documentation",
        "url": "https://learn.microsoft.com/en-us/office365/servicedescriptions/exchange-online-service-description/exchange-online-limits",
    },
    "lost_middle": {
        "title": "Lost in the Middle: How Language Models Use Long Contexts",
        "publisher": "Transactions of the ACL",
        "published": "2024",
        "url": "https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long",
    },
    "nist_ai_rmf": {
        "title": "AI Risk Management Framework",
        "publisher": "NIST",
        "published": "2023; Generative AI profile 2024",
        "url": "https://www.nist.gov/itl/ai-risk-management-framework",
    },
    "eu_translation": {
        "title": "Translation at the European Commission",
        "publisher": "European Commission",
        "published": "current overview",
        "url": "https://commission.europa.eu/about/departments-and-executive-agencies/translation_en",
    },
    "ironclad": {
        "title": "Ironclad advances contract review with an AI assistant",
        "publisher": "OpenAI",
        "published": "2024",
        "url": "https://openai.com/index/ironclad/",
    },
    "microsoft_security": {
        "title": "Microsoft Digital Defense Report 2025",
        "publisher": "Microsoft",
        "published": "2025",
        "url": "https://www.microsoft.com/en-us/corporate-responsibility/cybersecurity/microsoft-digital-defense-report-2025/",
    },
    "fincen_review": {
        "title": "FinCEN Year in Review",
        "publisher": "FinCEN",
        "published": "FY2025",
        "url": "https://www.fincen.gov/about-fincen/fincen-year-review",
    },
    "anthropic_agents": {
        "title": "How we built our multi-agent research system",
        "publisher": "Anthropic",
        "published": "2025",
        "url": "https://www.anthropic.com/engineering/multi-agent-research-system",
    },
}


ADDITIONAL_USE_CASE_DETAILS = {
    "document_extraction": {
        "summary": "Vision-assisted extraction from invoices, claims, and forms into validated structured records.",
        "examples": ("Invoice line items and three-way-match fields.", "Claims forms, receipts, IDs, and exception queues."),
        "why": ("Pages are the measurable scale driver; the seed includes OCR/vision input, JSON output, and validation retries.", "Compact output makes this prefill-heavy and suitable for asynchronous batches."),
        "routing": ("Require image input only when OCR text is not produced upstream.", "Accuracy should be evaluated by field and exception severity, not generic answer quality alone."),
    },
    "enterprise_search": {
        "summary": "Interactive retrieval-backed answers over company documents and knowledge bases.",
        "examples": ("Policy, product, engineering, and onboarding Q&A.", "Answers with citations to internal source chunks."),
        "why": ("The 8k-token seed approximates a query, roughly eight retrieved chunks, and a concise answer.", "Retrieval is assumed to happen in the application, so model-native tool use is not a hard gate."),
        "routing": ("Daytime concurrency and grounded-answer quality matter more than batch price.", "Replace chunk count and hit rate with retrieval telemetry."),
    },
    "contact_center_qa": {
        "summary": "Post-call scoring, compliance checks, summaries, and voice-of-customer extraction across recorded interactions.",
        "examples": ("Score every call against a QA rubric instead of sampling a small subset.", "Detect escalation, churn, compliance, and coaching signals."),
        "why": ("Recorded hours are more stable than call counts when call duration varies.", "The LLM seed covers transcript analysis; speech-to-text compute is explicitly outside this estimate."),
        "routing": ("Batch after calls and validate high-severity findings with humans.", "Use an audio gate only when the selected LLM itself performs transcription."),
    },
    "translation": {
        "summary": "Bulk multilingual translation and localization with glossary and style constraints.",
        "examples": ("Product documentation, support content, and marketing variants.", "Regulated or brand-sensitive copy sent to human review."),
        "why": ("Source words and target-language count are the defensible scale inputs.", "Three processed tokens per word is an approximate source-plus-target conversion before cache or translation-memory savings."),
        "routing": ("Generic planner quality is only a proxy; benchmark each language pair.", "Translation-memory hits should reduce token demand before sizing."),
    },
    "contract_review": {
        "summary": "Clause extraction, playbook comparison, risk review, and proposed redlines over legal agreements.",
        "examples": ("Procurement and sales contract review.", "Portfolio-wide diligence and change-of-control extraction."),
        "why": ("The seed includes source text, playbook context, rationale, and a revision pass.", "Long context and reasoning are conservative gates for whole-document review."),
        "routing": ("Batch portfolio scans, but reserve interactive capacity for negotiation redlines.", "Use clause-level recall and lawyer acceptance rates for calibration."),
    },
    "security_triage": {
        "summary": "Interactive investigation of alerts using logs, identity context, threat intelligence, and response tools.",
        "examples": ("Phishing, endpoint, identity, and cloud-control-plane alerts.", "Evidence summaries and recommended containment steps."),
        "why": ("Alert count is the business driver; the token seed includes a compact evidence bundle and several tool turns.", "A high quality floor models the asymmetric cost of false negatives."),
        "routing": ("Keep this off night-batch capacity because incident peaks are time sensitive.", "Calibrate by alert family and human escalation outcome."),
    },
    "aml_casework": {
        "summary": "Enhanced AML/KYC investigation over customer, transaction, sanctions, and adverse-media evidence.",
        "examples": ("Case enrichment and investigator narrative drafting.", "Sanctions, beneficial-ownership, and adverse-media review."),
        "why": ("The model assists rather than makes the final regulated decision.", "The 50k-token seed covers retrieved evidence, policy checks, tool turns, and a reviewable narrative."),
        "routing": ("Batch enrichment may run off peak; final review remains human-controlled.", "Track false-negative and citation-support metrics separately from generic quality."),
    },
    "synthetic_generation": {
        "summary": "Decode-heavy generation of test cases, examples, and fixtures followed by critique or validation.",
        "examples": ("Software test cases and structured fixtures.", "Evaluation examples and edge-case corpora."),
        "why": ("The multiplier includes prompt, artifact, critique, and retry tokens.", "Large queues and loose latency make this a natural batch workload."),
        "routing": ("Use spare decode capacity and deduplicate failed or low-diversity outputs.", "Quality requires task-specific validators, not the planner score alone."),
    },
    "catalog_enrichment": {
        "summary": "Generate attributes, categories, and product copy from catalog text and images.",
        "examples": ("Attribute extraction, taxonomy mapping, and missing-field completion.", "Descriptions, locale variants, and seasonal refreshes."),
        "why": ("SKUs, images, locales, and variants drive volume.", "The compact per-SKU seed assumes short source records and outputs, with image support as a hard gate."),
        "routing": ("Shift backfills off peak and route exceptions to stronger models.", "Measure attribute-level precision and merchant acceptance."),
    },
}


USE_CASE_EVIDENCE = {
    "classify": {"confidence": "medium", "assumption": "600 tokens/record assumes records are grouped into larger model requests; validate batch size and schema retries.", "source_ids": ("openai_use_cases", "openai_batch")},
    "summarize": {"confidence": "medium", "assumption": "25k tokens/document is an illustrative document-plus-summary mean; the long tail drives the context gate.", "source_ids": ("openai_use_cases", "openai_tokens", "openai_batch")},
    "chatbot": {"confidence": "medium", "assumption": "25k tokens/ticket assumes five turns with resent history/tool results; $4/M is a conservative infrastructure ceiling, not Fin's product price.", "source_ids": ("intercom_fin",)},
    "email_corrector": {"confidence": "low", "assumption": "10k tokens/employee-day assumes five assisted messages at about 2k tokens; adoption and thread length are organization inputs.", "source_ids": ("openai_use_cases",)},
    "coding": {"confidence": "medium", "assumption": "150k tokens/developer-day represents agentic repository work, not lightweight autocomplete; seat pricing only anchors order-of-magnitude value.", "source_ids": ("github_copilot", "openai_use_cases")},
    "meeting_notes": {"confidence": "medium", "assumption": "20k tokens/hour starts from measured conversational speech rates plus summary/prompt overhead; transcription compute is excluded.", "source_ids": ("transcript_rate", "openai_tokens", "openai_batch")},
    "evals": {"confidence": "low", "assumption": "2k tokens/eval item assumes multiple items may share a larger request; define the judge rubric and agreement target before calibrating SLO.", "source_ids": ("nist_ai_rmf", "openai_batch")},
    "inbox_archive": {"confidence": "low", "assumption": "2.4M tokens/mailbox/day assumes roughly 72M corpus tokens processed over a 30-day backfill; measure actual mailbox bytes and retention.", "source_ids": ("exchange_limits", "openai_batch")},
    "longctx": {"confidence": "low", "assumption": "2M tokens/analysis is approximately sixteen long passes, not a single context window; long-context capacity does not guarantee evidence use.", "source_ids": ("lost_middle",)},
    "research": {"confidence": "low", "assumption": "500k tokens/job assumes about twenty 25k-token agent calls. Agent depth and quality gates are scenario hypotheses.", "source_ids": ("openai_deep_research", "anthropic_agents")},
    "document_extraction": {"confidence": "low", "assumption": "3k tokens/page includes image/OCR input, structured output, and validation; benchmark the actual document mix.", "source_ids": ("openai_structured_outputs", "openai_batch")},
    "enterprise_search": {"confidence": "medium", "assumption": "8k tokens/query approximates eight retrieved chunks plus answer; chunking, reranking, and cache hits can move this sharply.", "source_ids": ("openai_vector_stores",)},
    "contact_center_qa": {"confidence": "medium", "assumption": "20k LLM tokens/recorded hour covers transcript analysis and rubric passes; speech inference is a separate workload.", "source_ids": ("transcript_rate", "openai_batch")},
    "translation": {"confidence": "low", "assumption": "Three processed tokens/source word is a language-dependent source-plus-target approximation; use pair-specific benchmarks and translation-memory hit rates.", "source_ids": ("eu_translation", "openai_tokens", "openai_batch")},
    "contract_review": {"confidence": "low", "assumption": "75k tokens/contract includes playbook, review, and revision passes; agreement length and clause workflow dominate.", "source_ids": ("ironclad", "lost_middle")},
    "security_triage": {"confidence": "low", "assumption": "12k tokens/alert assumes a compact evidence bundle and tool turns; the 100k-alert scale is an illustrative large-enterprise scenario.", "source_ids": ("microsoft_security",)},
    "aml_casework": {"confidence": "low", "assumption": "50k tokens/case and 12k cases/day are scenario inputs; FinCEN supports case volume as a driver, not this institution-level seed.", "source_ids": ("fincen_review",)},
    "synthetic_generation": {"confidence": "low", "assumption": "5k tokens/example includes critique and retries; task-specific validators determine usable yield.", "source_ids": ("openai_batch", "nist_ai_rmf")},
    "catalog_enrichment": {"confidence": "low", "assumption": "2k tokens/SKU assumes compact records and outputs; images, locales, and seasonal reprocessing determine the actual load.", "source_ids": ("openai_structured_outputs", "openai_batch")},
}


def enrich_use_case_details(details: dict) -> dict:
    """Merge evidence and additional copy into the app's existing detail registry."""
    merged = {key: dict(value) for key, value in details.items()}
    for key, value in ADDITIONAL_USE_CASE_DETAILS.items():
        merged[key] = dict(value)
    for key, evidence in USE_CASE_EVIDENCE.items():
        merged.setdefault(key, {}).update(evidence)
    return merged
