# Changelog

All notable changes to this project will be documented in this file.

## 2026-05-22

### Added

- Added an NVIDIA A10 GPU catalog entry with planner specs, picker metadata, TDP data, and regression coverage.

## 2026-05-20

### Added

- Added an RTX A2000 mobile GPU catalog entry.
- Added selectable automatic model-selection strategies for best value per GPU, use-case coverage, quality, lean GPU usage, and throughput.
- Added auto-selection tests covering every declared strategy and fallback behavior.

### Changed

- Updated automatic model selection so the selected strategy is persisted in state, included in reports, and passed through the model panel UI.
- Cleaned up planner navigation with shared calculator, user guide, and use-case tabs.
- Reworked GPU controls to support direct count entry, synced picker state, and faster cost updates.
- Simplified calculator and use-case editor cards by removing duplicate range sliders where numeric inputs already provide direct editing.

## 2026-05-15

### Added

- Added automatic model selection that fits deployed models to configured GPU pools, use-case demand, capability gates, SLOs, and quality floors.
- Added model exclusion and re-allow controls for auto-selected model sets.
- Added a bulk project picker action to add one project for every current use-case definition.
- Added a copyable projection report endpoint and UI action for exporting deployment, routing, economics, supply, and expansion diagnostics.
- Added internal user price readouts for deployed models, split into input and output $/1M token prices.
- Added best-next-GPU recommendations that estimate margin gain, cloud spend avoided, destroyed demand recovered, and served-token uplift.
- Added revenue projection tests for coverage metrics, zero-capacity assignments, and smooth latent-demand activation.

### Changed

- Updated routing quality checks to use confidence-adjusted effective quality plus an explicit per-use-case quality floor.
- Updated cloud and internal effective-price calculations so token efficiency scales output tokens rather than fixed prompt tokens.
- Reworked projection headline metrics around owner revenue, active demand, token coverage, value capture, and revenue multiple.
- Changed latent demand unlocks from a hard threshold to a smooth activation curve around the unlock price.
- Improved per-use-case routing diagnostics for SLO, capability, cloud, price-ceiling, and latent-demand outcomes.

## 2026-05-09

### Added

- Added a dedicated `/use-cases` page for viewing, editing, importing, and exporting reusable use-case definitions.
- Added a use-case library partial with controls for definition metadata, scale model, token multiplier, difficulty, SLO, price ceiling, capability gates, token shape, batch eligibility, and latent-demand economics.
- Added JSON import/export for both the reusable use-case library and the selected calculator use-case set.
- Added organization-scale controls so calculator cards can keep the selected use-case kind separate from the organization's current scale.
- Added scale models for linear, quadratic, network/graph, corpus/backfill, and custom formula demand sizing.
- Added richer built-in use-case definitions, including email correction, meeting notes, and inbox archive workloads.
- Added detailed use-case documentation content with examples, assumptions, routing implications, and token-shape visualizations.
- Added model metadata for mixed attention layouts, including hidden size, local attention, linear attention, CCA-style compressed attention, and attention labels.
- Added planner entries and quality proxies for Kimi Linear 48B, ZAYA1-8B, the legacy ZAYA1 74B proxy, and Laguna XS.2.
- Added GPU catalog entries and ordering updates for Blackwell, AMD MI350/MI355/MI400, Intel Arc Pro B50/B60, and Apple M3 Ultra/M4 Pro profiles.

### Changed

- Refactored project state so selected calculator use cases reference reusable definitions while preserving per-organization scale.
- Simplified calculator use-case cards to show the selected kind summary plus a scale control instead of all definition-editing controls.
- Updated project picker entries to use the editable use-case library rather than the static preset list.
- Moved use-case definition editing out of calculator cards and into the dedicated use-case page.
- Updated routing and state normalization so existing saved projects can infer or preserve their use-case kind.
- Updated aggregate demand syncing to account for use-case scale conversions and imported project sets.
- Updated KV-cache and attention-work calculations to model full attention, local attention windows, linear-attention recurrent state, and MLA tensor-parallel support more accurately.
- Updated model cards to surface attention labels and compressed KV details for MLA and CCA-style models.
- Updated Mistral Medium 3.5 labeling, cloud pricing, and quality assumptions.
- Reworked GPU picker cards into simpler single-action rows with planner-profile notes available as tooltips.
