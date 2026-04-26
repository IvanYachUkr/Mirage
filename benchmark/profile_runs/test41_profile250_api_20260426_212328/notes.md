# test41 250-Movie API Profile Notes

Run directory: `/tmp/mirage_profile250_api`

## What This Tested

- Full API-backed setup from fresh cloned code in `/tmp`.
- Step 100 movie generation for `250` movies across `2020-2024`.
- Benchmark profile with core benchmark tables enabled and heavy non-benchmark extras deferred.
- Speed audit and step-100 sanity reporting.

## Important Fix Found During The Run

Step 100 initially failed because `modeling_priors.json` passed artifact generation while missing `financial_priors.genre_rating_offset`. In strict research mode, `financials.py` correctly refused to use a runtime fallback.

Fix applied in `test41/generate_bootstrap_artifacts_api.py`:

- Normalize and validate all runtime-required finance maps before step 100.
- Materialize genre rating offsets, country budget scales, tier rating/vote maps, budget ranges, certification distributions, market/year models, latent/performance/vote/runtime models, and award weights.
- Detect LLM budget ranges written in millions and convert them to absolute USD.

The existing `/tmp` artifact was normalized safely after backing up the original as:

`/tmp/mirage_profile250_api/modeling_priors.pre_finance_normalize.json`

## Runtime Result

- Step 100 completed successfully in `92.0s`.
- Speed audit wall time: `90.29s`.
- Movie-complete events: `250`.
- Movies/sec from progress log: `0.436`.
- Resume manifest status: `complete`, last completed year `2024`.

## Step 100 Bottlenecks

Top measured components:

- `year.plan_llm`: `24.02s` total across `5` yearly boundaries.
- `setup.sample_movie_concepts`: `13.37s`.
- `movie.compute_financials`: `9.55s`.
- `movie.pick_cast`: `6.96s`.
- `main.post_generation_critic`: `4.42s`.
- `movie.pick_keywords`: `3.47s`.
- `movie.pick_director`: `2.60s`.

Interpretation: after the earlier title-bank optimization, step 100 is no longer dominated by title generation. The largest safe optimization target is now yearly planning frequency/caching for huge year ranges, but for a 20k/200k lab run the bigger expected wall-clock cost is still the LLM artifact/entity enrichment pipeline before step 100.

## Quality Snapshot

- Movies: `250`, unique titles: `250`.
- Cast rows: `1446`, unique cast people: `632`, reuse ratio: `2.288`.
- Unique directors: `110`.
- Unique companies: `113`, company Gini: `0.4254`.
- Keyword rows: `2043`, zero exact-topic movies: `0`.
- Awards: `40` rows, movies with awards: `12.0%`, win rate: `7.5%`.
- Duplicate tagline rate: `0.0%`.
- Blank character descriptions: `0.0%`.

Failed sanity gates are expected for a 250-movie smoke/profile run: `movie_count_20k`, `unique_cast_people_ge_14000`, `unique_directors_ge_1000`, and `company_gini_ge_045`.

## Recommendation

No quality-reducing step-100 optimization is justified from this profile. Before the lab run, keep the current benchmark profile, keep resume enabled, and prioritize:

- The finance artifact-contract fix now applied in `test41`.
- Local-LLM startup/resume scripts.
- Avoiding expensive full API/entity reruns on the laptop unless we specifically need another artifact validation pass.
