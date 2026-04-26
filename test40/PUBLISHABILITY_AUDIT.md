# Test30 Publishability Audit

Updated: 2026-04-04

This file tracks the refactor needed to make `test30` the paper-ready branch.
It should be updated during implementation, not reconstructed afterward.

## Status

- [x] Freeze `test27`; continue only in `test30`
- [x] Create persistent audit notes
- [x] Add required LLM-authored bootstrap artifacts
- [x] Add research/debug mode split with fail-closed research mode
- [x] Migrate bootstrap generators to artifact-driven realization
- [~] Remove timeline hard-coding from core generation path
- [~] Move modeling priors out of inline constants and into generated priors
- [x] Rename `v17_runtime` to stable runtime module with compatibility shim
- [x] Reduce whole-file JSON rewrites in steps `40` and `70`
- [ ] Re-run smoke tests and benchmark comparisons on `test30`

## Current Gaps Found During Audit

### Bootstrap content still procedural
- `generate_persons_procedural.py` still depends on `name_banks.py`
- `generate_companies_procedural.py` still uses fixed company prefixes/suffixes and numeric dedupe
- `generate_keywords_procedural.py` still uses fixed keyword pools
- `generate_character_bank.py` still uses fixed title/nickname/moniker pools
- `topup_title_bank.py` still uses fixed decade weights and hard-coded tagline pools
- `generate_company_financial_profiles.py` still uses fixed tier prior tables

### Time-range dependence
- `topup_title_bank.py` still has a debug-mode fallback to `DECADE_WEIGHTS`
- historical anchors have been reduced in `policy_runtime.py`, `financials.py`,
  `generate_edges_hybrid.py`, `scalable_edge_builder.py`, `world_state.py`,
  `secondary_tables.py`, and `big_history_events.py`, but a full audit is still needed
  for remaining late-stage assumptions in movie assembly and post-processing

### Hard-coded priors / coefficients
- `assembly.py` is now much more prior-driven, but still contains some inline weights, boosts, and penalties
- `financials.py` now reads generated financial dictionaries, but still keeps fallback defaults
- `generate_edges_hybrid.py` now reads generated edge priors for key weights and thresholds,
  but still keeps fallback defaults for compatibility
- `secondary_tables.py` now reads generated secondary-table priors for demographics, release timing,
  territory box office, reviews, awards, TV generation, streaming windows, production timelines,
  and person contracts, but still keeps fallback content banks and some authored auxiliary tables
- `contracts.py` still routes most LLM work through flash-lite and does not expose
  stable artifact-tier intent clearly enough

### Performance hotspots
- `enrich_persons_api.py` now uses shard-based batch materialization
- `generate_latent_vars_api.py` now uses shard-based batch materialization
- several debug/raw prompt artifacts are written eagerly even when not needed

## Refactor Notes

### Decisions
- Publication target for first pass is step `100` plus step `130`
- Steps `110` and `120` only need compatibility for now
- Research mode should default to required LLM-authored artifacts and no silent fallback
- `gemini-3.1-pro-preview` should be used only for small, high-leverage artifact generation
- Character naming must remain separate from person naming and support non-literal monikers

### Implementation Progress
- 2026-04-02:
  - started runtime/config rename and shared artifact plumbing
  - started runner refactor to insert artifact generation before procedural bootstrap
  - added `pipeline_runtime.py` and converted `v17_runtime.py` into a compatibility shim
  - added `generate_bootstrap_artifacts_api.py` for:
    - `identity_bank.json`
    - `character_identity_bank.json`
    - `company_lexicon.json`
    - `keyword_seed_bank.json`
    - `title_grammar_bank.json`
    - `temporal_regime_plan.json`
    - `modeling_priors.json`
  - updated `run_pipeline.py` to:
    - default to `--mode research`
    - generate bootstrap artifacts before the old entity steps
    - pass pipeline mode and year bounds through the environment
    - record per-step timing to `pipeline_timing.jsonl`
  - migrated these bootstrap generators to artifact-driven research mode:
    - `generate_persons_procedural.py`
    - `generate_companies_procedural.py`
    - `generate_keywords_procedural.py`
    - `generate_character_bank.py`
    - `topup_title_bank.py`
    - `generate_company_financial_profiles.py`
  - updated early year-sensitive logic in:
    - `policy_runtime.py`
    - `generate_world_policy_api.py`
    - `financials.py`
    - `generate_edges_hybrid.py`
    - `scalable_edge_builder.py`
  - replaced repeated whole-file batch rewrites with shard-based materialization in:
    - `enrich_persons_api.py`
    - `generate_latent_vars_api.py`
  - validation:
    - syntax-check passed for the edited `test30` modules
    - `test30/run_pipeline.py --help` ran successfully

## Remaining Open Work After This Pass

- Deep coefficient migration is still partial.
  - `modeling_priors.json` now exists as a first-class artifact and drives bootstrap generation plus company finance synthesis.
  - Major downstream scoring modules like `assembly.py` and larger parts of `financials.py` / `generate_edges_hybrid.py` still contain inline weights that should move behind the priors artifact next.
- Time-range generalization is improved but not fully complete.
  - Title planning, world-policy defaults, graph window defaults, and several finance anchors now adapt to the active year span.
  - Some historical assumptions still remain in later modules such as `secondary_tables.py`, `world_state.py`, `big_history_events.py`, and parts of `assembly.py`.
- Research-mode artifact quality still needs real smoke runs with the live API.
  - The scaffolding, validation, and runner integration are in place.
  - We still need `50`/`100` movie runs to confirm artifact shapes, dedupe behavior, future-range curves, and benchmark quality.
  - late-stage control logic now ingests generated priors more directly:
    - `pipeline_runtime.py` auto-merges `modeling_priors.json` into runtime config
    - `assembly.py` falls back to generated prior leaves and now reads sectioned selection weights for director/company alignment
    - `financials.py` reads generated financial dictionaries like `country_budget_scale`, `genre_rating_offset`, and tier finance parameters
    - `generate_edges_hybrid.py` reads generated edge priors for thresholds, weights, and cross-genre bridge settings
  - range-agnostic cleanup extended into:
    - `world_state.py`
    - `secondary_tables.py`
    - `big_history_events.py`
  - smoke-run bug found and fixed:
    - `run_pipeline.py` now defaults `--from-step` to `4` so research mode starts at the first required artifact step instead of skipping straight to person generation
    - artifact steps no longer force `gemini-3.1-pro-preview` globally; each bootstrap artifact now uses its own configured default tier unless the user explicitly passes `--model`
  - provider integration bug found and fixed:
    - bootstrap artifact generation no longer forces `thinking_budget=0`, because `gemini-3.1-pro-preview` rejects that and requires its supported/default thinking behavior
    - `llm_provider.py` no longer silently converts an omitted Gemini thinking budget into `0`, which was overriding the artifact fix and still breaking pro-preview
  - research smoke reached the first real artifact-shape issue:
    - `modeling_priors` and `identity_bank` succeeded
    - `company_lexicon` returned JSON but not as a top-level object
    - artifact parsing is now hardened to accept singleton/wrapped object payloads for bootstrap artifacts
    - bootstrap artifact generation now performs one strict JSON repair pass and stores failed raw outputs under `_dev/bootstrap_artifacts/` for inspection
    - `keyword_seed_bank` timed out repeatedly on `gemini-3.1-pro-preview`; its default routing has been lowered to the mid tier for the next smoke
    - `keyword_seed_bank` also exposed a schema alias issue: the model returned `genre_data` instead of `genres`; the artifact normalizer now accepts that alias
    - `character_identity_bank` returned a richer `human_name_mix_by_archetype` structure than the generator expected; the character generator now accepts structured human-name mixes and matches the artifact’s alias placeholder names
    - `temporal_regime_plan` returned `year_weights` as a year→weight object; the artifact normalizer now converts that into the canonical list-of-rows format
    - `title_grammar_bank` exposed richer placeholder names than the title renderer expected; the title/tagline renderer now resolves common alias placeholders dynamically instead of assuming a tiny fixed set
    - future-range smoke (`2027-2075`) exposed one more keyword-bank alias issue: the model returned `genre_metadata` instead of `genres`; the artifact normalizer now accepts that alias too
    - future-range smoke also showed a partial-but-almost-correct keyword artifact (`genres` present but incomplete); bootstrap artifact generation now performs one strict validation-repair call before failing closed
    - step `60` title-bank generation was still using a flat `1 / |genres|` sampler; it now uses non-uniform base genre weights plus temporal-phase modulation, and the modeling-priors prompt now requests optional learned genre prevalence fields for fresh runs
    - LLM usage logging is now run-scoped in `run_pipeline.py`; fresh runs will write to `decision_logs/<run_id>_llm_usage.jsonl` while still mirroring the latest run to `decision_logs/llm_usage.jsonl` for convenience
  - post-smoke quality fixes for the future-range run:
    - `assembly.py` now treats title-bank `genre_hint` as a first-class control signal during `sample_movie_concept`
    - concept-pack selection now prefers exact genre-hint matches and falls back to heuristic generation instead of accepting bucket-collapsed wildcard packs
    - heuristic concept selection now records bucket genre usage and uses that to discourage late-bucket monoculture
    - `generate_plot_summaries_api.py` now treats deterministic placeholder summaries as invalid, regenerates them, and fails research-mode runs if generic plot summaries remain
    - `generate_tv_summaries.py` now validates series/episode text quality, retries incomplete seasons semantically, and fails research-mode runs if TV text coverage is incomplete
    - `topup_title_bank.py` now cleans repeated tokens and malformed punctuation in rendered titles/taglines to prevent artifacts like doubled articles
  - dense-run recovery hardening:
    - `run_pipeline.py` now exports run-scoped decision-log and movie-progress log paths to step subprocesses
    - `world_state.py` uses those run-scoped paths, so dense reruns no longer append blindly into stale selection logs
    - `assembly.py` mirrors selection events into the latest aggregate log for convenience while preserving run-specific logs
    - `generate_movies.py` now emits explicit step-100 progress events (`movie_started`, `movie_completed`, `post_loop_start`, `critic_start`, `critic_done`, `step100_complete`)
    - `generation_critic.py` now persists prompt/report artifacts even on critic failure or invalid output, which removes the previous silent-failure blind spot
  - part-2 control-plane migration pass:
    - `topup_title_bank.py` now sanitizes reused title banks to the requested year window and exact target count before topping up
    - `run_pipeline.py` now validates step `60` against the requested year window instead of accepting any existing `title_bank.csv`
    - `generate_movies.py` now enforces `start_year/end_year` at runtime, filtering the loaded title bank and refusing undersized in-range runs
    - `assembly.py` now reads additional generated priors for:
      - concept-pack scoring weights and penalties
      - writer/director probabilities by tier
      - genre->tier distributions
      - concept latent targets and style vectors
      - release-month seasonality and genre month bumps
    - `secondary_tables.py` TV generation is now range-relative instead of assuming a fixed 2010-2024 worldview
    - `secondary_tables.py` production-timeline and streaming-window fallbacks now anchor to the active range instead of a hardcoded 2010
    - `big_history_events.py` now maps event eras relative to the template registry span instead of a fixed 1970-2035 interpolation
    - `world_state.py` year-cache fallback bounds now use the active pipeline range instead of 1970/2100 sentinels
    - `generate_bootstrap_artifacts_api.py` now asks `modeling_priors` for the new concept-selection and seasonality priors needed by the runtime

## Remaining Hard-Coded Areas That Still Matter

These are the hard-coded parts that are still meaningfully bad for research-mode claims because they shape output behavior rather than only enforcing schema compatibility.

- `financials.py`
  - now substantially improved:
    - `country_budget_scale`
    - `genre_rating_offset`
    - `tier_rating_base`
    - `tier_rating_std`
    - `tier_log_center`
    - `tier_min_votes`
    - market-regime parameters
    - year-quality waveform parameters
    - quality/market latent coefficient mixes
    - performance model
    - vote model
    - runtime model
    - award-campaign weights
    - writer-director bonus
    all now have first-class generated-prior paths via `financial_priors`
  - still remaining:
    - `BUDGET_RANGES` import path still acts as the base budget clip envelope
    - `CERT_DISTS` still defines certification distributions procedurally
    - some latent feature blend formulas inside `compute_financials()` are still manual
  - why it matters: finance is no longer the worst offender, but a few remaining authored envelopes still shape output behavior

- `generate_edges_hybrid.py`
  - now substantially improved:
    - person-person degree-cap distributions can come from generated priors
    - person-person classification thresholds / jitter / weight bands can come from generated priors
    - person-company generation controls can come from generated priors
    - company-company rival / co-production thresholds can come from generated priors
    - serendipitous-edge and triadic-closure behavior can come from generated priors
    - relationship-calibration weights / targets / candidate sizes can come from generated priors
    - research-mode artifact loading no longer silently downgrades to debug for this module
  - still remaining:
    - fallback defaults such as `_CAP_PARAMS`, `_CROSS_GENRE_K`, and auxiliary calibration defaults still exist for compatibility
    - the large-run `scalable_edge_builder.py` path still deserves the same control-plane migration treatment in a later pass
  - why it matters: graph shape is benchmark-relevant, and this module is now much closer to a real prior-driven control plane, but the scalable builder is still a follow-up item

- `assembly.py`
  - now substantially improved:
    - director/company/cast/keyword selection blocks can come from generated priors
    - co-director probabilities, geo boosts, dynamic cast ranges, keyword counts,
      and year-slate family boosts can come from generated priors
    - many concept-selection weights, latent targets, and month-seasonality controls
      can come from generated priors
  - still remaining:
    - `_GENRE_TIER_DIST` and some concept/title fallback heuristics still act as authored defaults
    - some local cast/director/company penalties and late-stage keyword heuristics remain inline
  - why it matters: this module still shapes movie realization directly, so the remaining inline heuristics are still meaningful

- `secondary_tables.py`
  - now substantially improved:
    - demographics, release timing, territory splits, review volume/source mix,
      awards, TV generation, production timelines, streaming windows, and
      person contracts now have generated-prior paths via `secondary_table_priors`
  - still remaining:
    - nationality/city/location banks and some auxiliary vocab lists are still authored content
    - ratings breakdown, movie links, alternate-title transforms, and a few box-office/user-rating
      distributions still lean on inline heuristics
  - why it matters: less central than finance/edges/assembly, but still broad enough that the remaining authored auxiliary behavior should be audited later

- `big_history_events.py`
  - now substantially improved:
    - event families, probabilities, descriptions, durations, and action hints can come from generated `history_event_priors`
    - event windows can now be specified as relative year fractions instead of absolute historical anchors
  - still remaining:
    - the fallback event action templates are still authored
    - `scalable_edge_builder.py` is still the larger graph-path follow-up for big runs
  - why it matters: macro-event timing and ontology are now much closer to the control-plane story, but large-run graph realization still needs its own migration pass

These are acceptable to keep hard-coded:

- benchmark/schema constants and IMDb/JOB compatibility mappings
- canonical entity enums needed by the export contract
- safety clamps, bounds, and debug-only fallback defaults

## 2026-04-04 Modeling Priors Reliability Fix

Problem discovered during clean-run analysis:
- the successful dense run had good outputs, but `modeling_priors.json` was mostly empty
- that meant the broader LLM planning stack was working, but the explicit scalar/table prior-control layer was not reliably surviving generation

Root cause:
- the old `modeling_priors` path asked for one giant object in a single call
- repair/normalization could accept technically valid but semantically empty dict sections
- validation only checked that sections existed as dicts, not that they were actually populated

Fix implemented:
- `generate_bootstrap_artifacts_api.py` now generates `modeling_priors` in grouped subcalls instead of one monolithic prompt:
  - foundation
  - titles
  - selection
  - edges
  - financials
  - secondary
  - history
- added semantic validation for each `modeling_priors` section, not just shape validation
- added grouped fail-closed behavior: requested sections must be non-empty and structurally meaningful
- added cleanup for nested accidental section echoes so the final artifact stays clean
- added richness gating so shallow artifacts fail instead of silently becoming the active research artifact

Verification:
- direct regeneration of `modeling_priors.json` for the dense `2064-2068` setup now yields:
  - populated `title_generation`
  - populated `selection_weights`
  - populated `edge_priors`
  - populated `financial_priors`
  - populated `secondary_table_priors`
  - populated `history_event_priors`
- nested bogus top-level sections inside other sections are now removed

Current implication:
- the `modeling_priors` path now looks strong enough for the next fresh research rerun
- the previous successful dense run remains useful for quality analysis, but it should not be treated as proof that populated modeling priors were active, because that run used the old mostly-empty artifact

## 2026-04-05 Polish And Repair Hardening

What was hardened:
- added a shared `text_polish.py` helper layer for title/tagline/alternate-title/character-name cleanup
- wired that sanitation into:
  - `topup_title_bank.py`
  - `assembly.py`
  - `generate_character_bank.py`
  - `secondary_tables.py`
  - `generation_critic.py`
- strengthened `title_grammar_bank` prompting to demand more market-facing taglines and fewer subtitle-like noun phrases
- strengthened `keyword_seed_bank` prompting to reduce dry procedural/task-style keyword seeds
- added `remove_keyword` as an allowed post-generation critic action for clearly irrelevant keyword assignments
- removed the lingering pandas concat warning in `export_imdb_schema.py`

Why it matters:
- the remaining problems after the clean priors rerun were mostly polish rather than structural modeling
- this pass moves those cleanup rules out of one-off debugging and into the normal generation/export path
- the current dense outputs were also polished in place and re-exported, so the live benchmark bundle now reflects the fixed structural text artifacts rather than only future reruns

## 2026-04-05 Research-Mode Final Rewrite Hardening

What changed:
- added run-scoped research auditing:
  - `run_pipeline.py` now initializes and updates `research_mode_audit.json`
  - per-run audit copies are written under `decision_logs/<run_id>_research_mode_audit.json`
  - each step records status, and research mode now fails if any fallback/default hit was recorded
- strengthened artifact-backed bootstrap/entity generators so research mode no longer silently falls back to authored vocab or defaults in:
  - `generate_persons_procedural.py`
  - `generate_companies_procedural.py`
  - `generate_keywords_procedural.py`
  - `generate_character_bank.py`
  - `topup_title_bank.py`
  - `generate_company_financial_profiles.py`
- hardened runtime prior consumers so research mode refuses missing/defaulted priors in:
  - `assembly.py`
  - `generate_edges_hybrid.py`
  - `financials.py`
  - `scalable_edge_builder.py`
- critic usage is now audit-visible via `generation_critic.py`

Important contract upgrade:
- `generate_bootstrap_artifacts_api.py` now validates `modeling_priors` sections against the actual runtime schema instead of only checking for shallow non-empty dicts
- this especially tightened:
  - `selection_weights`
  - `edge_priors`
  - `scalable_edge_priors`
  - `financial_priors`
- the prompt contract for `selection_weights` and `edge_priors` was also expanded so future artifact generations request the exact nested keys that the runtime consumes

What this means for the paper claim:
- research mode is now much closer to a defensible “LLM-authored control plane + deterministic realization” design
- authored fallbacks still exist, but they are now explicitly debug-mode only in the main bootstrap and modeling-heavy paths
- the remaining hard-coded values are increasingly limited to:
  - benchmark/export enums and schema compatibility mappings
  - numerical safety clamps
  - debug-only fallback defaults that should be unreachable in research mode

What still needs real validation:
- a fresh research-mode rerun is still required to prove that the stricter `modeling_priors` schema and audit manifest work cleanly end to end
- the large-run `scalable_edge_builder.py` path still needs scale validation after this rewrite, even though its policy path is now prior-aware

Auxiliary follow-up completed in the same rewrite pass:
- `secondary_tables.py` now treats `secondary_table_priors` as a fail-closed research artifact at config-merge time instead of silently deep-merging authored defaults
- `big_history_events.py` now disables deterministic fallback execution in research mode and requires `history_event_priors.event_specs`

## 2026-04-06 Final Evidence/Artifact Cleanup

Why this pass was needed:
- the latest clean dense run had good core movie quality, but the evidence artifacts were still weaker than the data itself
- specifically:
  - `research_mode_audit.json` underreported `selection_weights` usage
  - `critic_report.json` did not carry the applied repairs list into audit accounting
  - `movies_analysis.csv` could lag behind improved plot summaries and keep stale boilerplate text
  - `keyword_seed_bank` validation still accepted incomplete genre coverage, which later weakened keyword coherence instead of failing early

What changed:
- `assembly.py`
  - now audits per-key `modeling_priors.json` usage for runtime selection blocks, especially `selection_weights.*`
- `generation_critic.py`
  - now persists `repairs`, `repair_types`, and rewrite-vs-deterministic repair counts in the critic report
  - tagline rewrites now require stronger, non-weak outputs
- `bootstrap_artifacts.py`
  - critic audit ingestion now records sampled/flagged titles, cache hit, and repair breakdowns without silently dropping them
- `generate_plot_summaries_api.py`
  - now rebuilds `movies_analysis.arrow/csv` whenever plot summaries are regenerated
- `export_imdb_schema.py`
  - now rebuilds exported `movies_analysis.csv` from current `movie` + `movies_flat` instead of copying a possibly stale source extra
- `generate_bootstrap_artifacts_api.py`
  - keyword-seed validation now requires rows for every benchmark genre plus non-trivial seeds/qualifiers/contexts
- `generate_keywords_procedural.py`
  - research mode now fail-closes if any benchmark genres are missing from the keyword seed bank
- `assembly.py`
  - keyword cleanup now enforces stronger story-match targets and tighter off-genre/generic caps

Current state after this pass:
- the source code is now aligned with the evidence requirements of the paper claim much better than before
- the currently materialized dense run still has old critic/audit logs, because those files were produced before the new provenance code existed
- the exported `imdb_schema/movies_analysis.csv` was refreshed immediately and is now consistent with the improved plot summaries

Recommended next step:
- run one fresh dense research rerun (`100`, `2064-2068`, through `130`) so the new evidence logging and stricter keyword-bank validation are exercised from scratch on the normal path
