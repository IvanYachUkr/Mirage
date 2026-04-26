# Test30 WSL/Linux Handoff

This file is the authoritative handoff for the next agent that will continue `test30` from WSL/Linux.

Important warning:
- [README.md](README.md) is stale and still describes `test27`, not `test30`.
- [local_ollama_qwen35/README.md](local_ollama_qwen35/README.md) is also stale and still points at `test27`.
- Treat this handoff as the current source of truth until those docs are rewritten.

New Linux-first starter artifacts added during handoff:
- [LINUX_BOOTSTRAP_QUICKSTART.md](LINUX_BOOTSTRAP_QUICKSTART.md)
- [local_ollama_qwen35/run_test30_local_linux.sh](local_ollama_qwen35/run_test30_local_linux.sh)

## 1. Current Status

`test30` is now a research-grade hybrid pipeline with:
- an LLM-authored semantic/control plane,
- deterministic relational realization,
- fail-closed research mode,
- audited runtime prior consumption,
- strict schema/export outputs.

The pipeline is not "pure LLM." It is a hybrid system:
- the LLM controls semantic policy, vocabularies, priors, planning artifacts, and text generation,
- the code still controls mechanics, schema compliance, scaling, and some score decompositions/defaults.

This is an honest and defensible research framing:
- "LLM-authored priors and semantic artifacts control a deterministic relational generator"
- not "all meaningful modeling logic is fully LLM-authored"

## 2. Best Verified Runs On Windows

### Dense fresh sign-off candidate

Run:
- `20260411_212243`

Logs and analysis:
- [_runner_logs/dense100_truefinal_20260411_212241.out.log](_runner_logs/dense100_truefinal_20260411_212241.out.log)
- [_runner_logs/dense100_truefinal_20260411_212241.err.log](_runner_logs/dense100_truefinal_20260411_212241.err.log)
- [_dev/clean_run_analysis_20260411_212243.md](_dev/clean_run_analysis_20260411_212243.md)
- [_dev/clean_run_analysis_20260411_212243.json](_dev/clean_run_analysis_20260411_212243.json)
- [decision_logs/20260411_212243_llm_usage.jsonl](decision_logs/20260411_212243_llm_usage.jsonl)

Audit note:
- there is no dedicated archived `20260411_212243_research_mode_audit.json` in `decision_logs/`,
- the live [decision_logs/research_mode_audit.json](decision_logs/research_mode_audit.json) was later overwritten by the historical run,
- so use the dense run analysis files above as the stable record for dense-run conclusions.

Why it matters:
- this is the best dense run so far,
- it was a true fresh run from scratch,
- it is the strongest candidate for the main paper-quality dense result.

Key quality metrics from the analysis:
- `100` movies
- `100` unique titles
- `100` unique taglines
- year cosine: `0.9976`
- title-bank vs movie genre cosine: `0.9806`
- mean plot length: `57.63`
- generic plots: `0`
- critic flagged titles: `5`
- critic actions: `6`
- mean exact-topic keyword ratio: `0.672`
- mean primary+related keyword ratio: `0.91`
- movies below `0.5` exact-topic ratio: `7`
- zero movies with zero exact-topic support
- zero movies with zero primary+related support
- award campaign vs awards: `0.2745`
- award campaign vs rating: `0.483`

Interpretation:
- structurally strong,
- evidentially strong,
- good enough to treat as the dense sign-off candidate,
- only slightly below the most aggressive keyword exact-topic target.

### Historical validation run

Run:
- `20260412_004507`

Logs and analysis:
- [_runner_logs/historical100_signoff_resume80_20260412_022134.out.log](_runner_logs/historical100_signoff_resume80_20260412_022134.out.log)
- [_runner_logs/historical100_signoff_resume80_20260412_022134.err.log](_runner_logs/historical100_signoff_resume80_20260412_022134.err.log)
- [_dev/clean_run_analysis_20260412_004507.md](_dev/clean_run_analysis_20260412_004507.md)
- [_dev/clean_run_analysis_20260412_004507.json](_dev/clean_run_analysis_20260412_004507.json)
- [decision_logs/20260412_004507_llm_usage.jsonl](decision_logs/20260412_004507_llm_usage.jsonl)
- [decision_logs/20260412_004507_research_mode_audit.json](decision_logs/20260412_004507_research_mode_audit.json)

Important caveat:
- this run resumed from step `80` after a source-level priors normalization fix,
- so it is a good historical validation run,
- but not the pristine sign-off artifact under the strictest protocol.

Key quality metrics from the analysis:
- `100` movies
- `100` unique titles
- `100` unique taglines
- realized year span: `1952-2025`
- title-bank vs movie year cosine: `1.0`
- title-bank vs movie genre cosine: `0.9906`
- mean plot length: `55.71`
- generic plots: `1`
- critic flagged titles: `3`
- critic actions: `5`
- mean exact-topic keyword ratio: `0.623`
- mean primary+related keyword ratio: `0.9`
- zero movies with zero exact-topic support
- zero movies with zero primary+related support
- award campaign vs awards: `0.329`
- award campaign vs rating: `0.4814`

Interpretation:
- this is strong evidence that the pipeline generalizes beyond the dense future-only window,
- but if the paper later requires a pristine historical artifact, do one more fully fresh historical rerun.

## 3. What Is Meaningfully LLM-Driven Now

The LLM controls the parts that are actually worth outsourcing semantically.

### 3.1 Bootstrap / semantic artifact layer

These are LLM-authored artifacts or are now strongly shaped by LLM-authored outputs:
- [identity_bank.json](identity_bank.json)
- [character_identity_bank.json](character_identity_bank.json)
- [company_lexicon.json](company_lexicon.json)
- [keyword_seed_bank.json](keyword_seed_bank.json)
- [title_grammar_bank.json](title_grammar_bank.json)
- [temporal_regime_plan.json](temporal_regime_plan.json)
- [modeling_priors.json](modeling_priors.json)

### 3.2 High-level world planning / semantic policy

These are LLM-backed planning or semantic artifacts:
- [world_policy.json](world_policy.json)
- [year_slate_plan.json](year_slate_plan.json)
- [concept_packs.json](concept_packs.json)
- [franchise_bibles.json](franchise_bibles.json)

### 3.3 LLM-generated text outputs

These are directly LLM-generated or critic-refined:
- movie plot summaries,
- TV series summaries,
- episode descriptions,
- bounded critic rewrites.

### 3.4 Runtime control through priors

The generated priors now matter in real execution. They are not decorative.

Examples:
- keyword slot policy,
- keyword exact-topic minimums,
- related-genre maps,
- title/tagline placeholder and render constraints,
- director/company/cast selection knobs,
- finance shaping,
- award-campaign shaping,
- edge/scalable-edge priors,
- secondary-table priors.

The best evidence of this is the dense sign-off analysis and the runtime audits:
- output quality moved when prior paths were changed,
- research mode rejects missing or malformed contracts instead of silently falling back.

## 4. What Is Still Hard-Coded Or Hybrid

The pipeline still has important human-coded mechanics. This does not invalidate the research claim, but it is important to understand clearly.

### 4.1 Assembly mechanics

[assembly.py](assembly.py) still contains major default structures and score components, including:
- `_DEFAULT_DIRECTOR_SELECTION`
- `_DEFAULT_COMPANY_SELECTION`
- `_DEFAULT_CAST_SELECTION`
- `_DEFAULT_KEYWORD_SELECTION`
- `_GENRE_TONE`
- `_GENRE_TIER_DIST`
- `_GENRE_TO_ARCHETYPES`
- `_GENRE_ARCHETYPE_BONUS`
- `_GENRE_CLUSTERS`
- `_GENRE_ALIASES`

Meaning:
- movie assembly is not free-form LLM behavior,
- it is deterministic realization guided by priors plus human-coded score structure.

### 4.2 Finance and ratings mechanics

[financials.py](financials.py) still contains hybrid/default logic, including:
- `BUDGET_RANGES`
- `CERT_DISTS`
- `COUNTRY_BUDGET_SCALE`
- `GENRE_RATING_OFFSET`
- `TIER_MIN_VOTES`
- `_DEFAULT_MARKET_REGIME`
- `_DEFAULT_YEAR_QUALITY`
- `_DEFAULT_AWARD_CAMPAIGN_WEIGHTS`

Meaning:
- finance is partially controlled by priors,
- but still normalized against human-coded envelopes and fallbacks.

### 4.3 Edge generation and scale switching

[generate_edges_hybrid.py](generate_edges_hybrid.py) still contains:
- `_DEFAULT_PERSON_PERSON_CLASSIFICATION`
- `_DEFAULT_PERSON_COMPANY_GENERATION`
- `_DEFAULT_COMPANY_COMPANY_GENERATION`
- `_DEFAULT_SERENDIPITOUS_EDGES`
- `_DEFAULT_TRIADIC_CLOSURE`
- `_DEFAULT_CALIBRATION`

It also still uses hard-coded scalable graph thresholds:
- `SCALABLE_PERSON_THRESHOLD = 100_000`
- `SCALABLE_COMPANY_THRESHOLD = 10_000`

Meaning:
- the edge path is still materially hybrid,
- and scale behavior for `20k+` / `200k` must be treated as an explicit engineering concern.

### 4.4 Secondary tables and TV generation priors

[secondary_tables.py](secondary_tables.py) still contains default blocks such as:
- `_DEFAULT_AWARD_PRIORS`
- `_DEFAULT_TV_GENERATION_PRIORS`
- `_DEFAULT_STREAMING_WINDOW_PRIORS`
- `_DEFAULT_PERSON_CONTRACT_PRIORS`
- `_DEFAULT_PRODUCTION_TIMELINE_PRIORS`

Meaning:
- awards, TV, contracts, streaming windows, and timelines are not purely LLM-authored,
- they still run through deterministic default structures with prior overlays.

### 4.5 Normalization and clamping layer

[generate_bootstrap_artifacts_api.py](generate_bootstrap_artifacts_api.py) now validates and normalizes:
- `title_generation.allowed_tagline_placeholders`
- `title_generation.tagline_render_constraints`
- `keyword_generation.selection_bucket_targets`
- `selection_weights.keyword_selection.*`
- `edge_priors`
- `scalable_edge_priors`
- `financial_priors`
- `secondary_table_priors`

It also still contains hard-coded floors and caps for research-mode hardening, especially around:
- keyword exact floors,
- primary+related floors,
- generic/off-genre caps,
- slot mix tightening.

Interpretation:
- this normalization/clamping is intentional and acceptable,
- but it means the pipeline is "LLM-controlled within validated envelopes," not raw unconstrained LLM execution.

## 5. What Was Verified On Windows

### 5.1 End-to-end pipeline execution

Verified on Windows:
- dense fresh sign-off run completed through `130`,
- historical validation run completed through `130`,
- root outputs were refreshed successfully:
  - [movie.csv](movie.csv)
  - [tv_series.csv](tv_series.csv)
  - [episodes.csv](episodes.csv)
  - [imdb_schema/export_manifest.json](imdb_schema/export_manifest.json)

### 5.2 Research-mode fail-closed behavior

Verified:
- research mode raises on missing/malformed priors instead of silently falling back,
- keyword capacity constraints now fail loudly,
- title/tagline placeholder leakage is blocked,
- audit tracking is meaningful enough to support the research story.

### 5.3 Title/tagline path

Verified:
- title grammar and title bank are now materially stronger than before,
- title/tagline placeholders are eliminated in the final dense sign-off run,
- duplicate-tagline collapse was fixed enough to reach `100` unique taglines in the fresh dense run.

### 5.4 Keyword path

Verified:
- keyword generation is now bucketed (`exact_anchor`, `related_support`, `story_specific`, `generic`),
- the keyword bank no longer catastrophically misses whole genres,
- movie-time keyword assignment no longer leaves movies with zero exact-topic or zero primary+related support in the final dense sign-off run.

### 5.5 Local provider abstraction

Verified by code inspection and Windows runtime environment:
- [llm_provider.py](llm_provider.py) already supports:
  - `LLM_PROVIDER=gemini`
  - `LLM_PROVIDER=local`
  - OpenAI-compatible servers
  - Ollama-specific fallbacks

Important environment variables:
- `LLM_PROVIDER`
- `LOCAL_LLM_URL`
- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_API_KEY`

### 5.6 Windows-specific bugs already fixed

These classes of Windows issues were fixed during hardening:
- `cp1252` logging/Unicode crashes in progress or warning output,
- stale CSV mirror issues after Arrow/export succeeded,
- multiple schema-contract mismatches discovered late and moved earlier into artifact validation,
- multiple title/tagline and keyword boundary failures moved into fail-early checks.

## 6. Known Hardening Outcomes And Gotchas

These are worth knowing before touching Linux automation or large runs.

### 6.1 Title/tagline path

Current intended behavior:
- [title_grammar_bank.json](title_grammar_bank.json) is the reusable template artifact,
- [entities/title_bank.csv](entities/title_bank.csv) is expected to be fully materialized in research mode,
- unresolved placeholder taglines should not survive into [movie.csv](movie.csv).

Implication for future changes:
- do not reintroduce research-mode runtime tagline synthesis in [assembly.py](assembly.py),
- if Linux changes touch title-bank generation, validate placeholder-free outputs explicitly.

### 6.2 Keyword capacity and bank size

Current intended behavior:
- research mode may generate more keywords than the nominal requested budget if that is required to satisfy exact-topic and primary+related floors,
- the keyword bank now uses bucketed roles such as `exact_anchor`, `related_support`, `story_specific`, and `generic`.

Implication:
- do not assume `n_keywords` is always the final realized keyword count in research mode,
- a larger realized keyword bank is acceptable if it is what the runtime contract needs.

### 6.3 Edge priors shape sensitivity

One real bug discovered on Windows:
- `edge_priors.triadic_closure.extra_cap` must be normalized to a scalar-like value before [generate_edges_hybrid.py](generate_edges_hybrid.py) consumes it.

Implication:
- if Linux runs regenerate priors and step `80` fails, inspect [modeling_priors.json](modeling_priors.json) first,
- especially the `edge_priors` block.

### 6.4 CSV mirror consistency

One important data-integrity issue that was fixed:
- root CSV mirrors could become stale even when in-memory / Arrow / export outputs were current.

Current expectation:
- [generate_movies.py](generate_movies.py) now refreshes root CSV mirrors from final in-memory results,
- future analysis should prefer the final exported/authoritative outputs, not assume an intermediate CSV is the truth by default.

### 6.5 TV summary policy

Current intended behavior:
- TV summaries should stay on the cheaper path,
- no automatic `pro` escalation should be relied on,
- TV generation uses chunked episode batches and resumable logic.

Implication:
- when moving to a local LLM, this path is a strong candidate for cost/runtime gains,
- but it should remain chunked and resumable.

## 7. What Is Not Yet Verified On Linux / WSL

This is the main next step for the new agent.

### 6.1 Basic environment bring-up

Need to verify:
- Python environment setup from [requirements.txt](requirements.txt),
- Arrow/CSV writing behavior under Linux,
- no Windows-only path assumptions remain,
- no lingering encoding assumptions remain,
- audit and log files are produced correctly under Linux paths.

### 6.2 Runtime scripts

Current issue:
- `test30` still has wrappers and local-Ollama notes inherited from `test27`.

Need to verify or build:
- a clean Linux launcher for `test30`,
- a clean WSL-friendly entrypoint,
- automatic setup of venv + dependencies,
- automatic local-provider environment export,
- automatic log capture.

### 6.3 Local LLM path

The next agent should verify the local provider end to end with the actual lab model/server.

At minimum verify:
- `LLM_PROVIDER=local`
- `LOCAL_LLM_URL`
- `LOCAL_LLM_MODEL`
- optional `LOCAL_LLM_API_KEY`
- compatibility with the chosen OpenAI-compatible serving stack

Important:
- [llm_provider.py](llm_provider.py) is already designed for this,
- but the lab runtime still needs real validation with the actual local model backend.

### 6.4 Large-scale behavior

Before any serious `20k+` or `200k` run, the next agent should validate:
- graph scaling path,
- person/company count scaling,
- TV generation explosion,
- long-run local-LLM throughput and stability.

The biggest scale caveat is the graph switch:
- [generate_edges_hybrid.py](generate_edges_hybrid.py) still flips to scalable mode only at:
  - `100,000` persons
  - `10,000` companies

That threshold may be too high for comfortable `20k`-movie experiments, depending on the actual entity counts and machine behavior.

## 8. Runtime And Bottleneck Notes

From the fresh dense run on April 11, 2026:
- total runtime was about `103` minutes for `100` movies

Main bottlenecks:
- step `120` TV summaries: about `43.4` min
- step `4` modeling priors: about `9.3` min
- step `70` latent variables: about `9.3` min
- step `97` concept packs: about `6.5` min
- step `96` keyword motif bank: about `6.3` min
- step `58` title grammar: about `6.1` min
- step `40` enrich persons: about `5.2` min
- step `74` year slate plan: about `4.5` min
- step `100` movie generation: about `4.0` min

Interpretation:
- the dominant bottleneck is still TV summaries,
- a lot of runtime is LLM-bound,
- but large-scale graph/runtime behavior is still a separate risk.

For local-LMM migration:
- local inference may reduce API/network overhead for:
  - person enrichment,
  - plot summaries,
  - TV summaries,
  - some planning steps,
- but it will not solve procedural/graph bottlenecks by itself.

## 9. Recommended Linux/WSL Bring-Up Plan

### Phase 1: Environment sanity

1. Create a clean Linux/WSL venv.
2. Install [requirements.txt](requirements.txt).
3. Verify `python test30/run_pipeline.py --help`.
4. Verify [llm_provider.py](llm_provider.py) can hit the chosen local OpenAI-compatible endpoint.

### Phase 2: Smoke tests

Recommended small checkpoints:
1. `--until-step 30`
2. `--until-step 60`
3. `--until-step 98`

Reason:
- this catches provider wiring, artifact generation, title-bank generation, keyword-bank generation, and graph path issues before doing the expensive text stages.

### Phase 3: Full Linux dense rerun

Do one full dense rerun:
- `100` movies
- `2064-2068`
- `--fresh --mode research --until-step 130`

This is the first experiment that should determine whether the Linux/local-LLM setup is ready.

### Phase 4: Scale validation before massive runs

Before `20k` or `200k` full runs:
- validate through step `98` first,
- consider forcing or lowering the scalable graph threshold,
- inspect TV generation growth,
- inspect local model throughput and error recovery behavior.

## 10. Suggested Commands For The Next Agent

These are starting points, not final lab automation scripts.

### Basic dense research run

```bash
python test30/run_pipeline.py \
  --fresh \
  --mode research \
  --n-movies 100 \
  --start-year 2064 \
  --end-year 2068 \
  --until-step 130
```

### Historical research run

```bash
python test30/run_pipeline.py \
  --fresh \
  --mode research \
  --n-movies 100 \
  --start-year 1950 \
  --end-year 2025 \
  --until-step 130
```

### Local provider environment example

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_URL=http://localhost:8000/v1
export LOCAL_LLM_MODEL=your-local-model-name
```

Then run the same `test30/run_pipeline.py` command.

## 11. Recommended Tasks For The Next Agent

The next agent should do these in order:

1. Treat this handoff as the source of truth, not [README.md](README.md).
2. Inspect and rewrite the `test30` Linux/local-LLM launch path.
3. Port or replace the stale `local_ollama_qwen35` notes/scripts so they target `test30`, not `test27`.
4. Validate `LLM_PROVIDER=local` end to end.
5. Run Linux smoke tests up to step `30`, `60`, and `98`.
6. Run a fresh dense sign-off candidate on Linux.
7. Only after that, think about `20k+` scale validation.

## 12. Final Assessment

The project is not meaningless and it is not "just engineering glue."

The current honest state is:
- semantically meaningful control has been outsourced to the LLM,
- the system is still hybrid,
- the dense Windows run is strong enough to support the core paper claim,
- the historical Windows run is strong enough to support the validation story,
- Linux/local-LLM migration is now the most important next operational step.

If the next agent can make the Linux/local provider path clean and reproducible, the project should be in a very good place for larger-scale experiments in the lab.
