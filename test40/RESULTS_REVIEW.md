# Test30 Results Review

Run under review:
- future-only research-mode smoke
- `50` movies
- `2027-2075`
- completed through step `130`

Additional run under review:
- dense future top-up regression
- `100` movies
- `2064-2068`
- reused existing early artifacts/entities, restored a `10`-title seed bank, then reran from step `60`

Review policy:
- log findings while inspecting
- separate `OK`, `Issue`, and `Action`
- update this file before and after fixes

## Artifacts

OK:
- All core bootstrap and planning artifacts report `meta.generator_mode = "llm"`.
- Confirmed on:
  - `modeling_priors.json`
  - `identity_bank.json`
  - `company_lexicon.json`
  - `keyword_seed_bank.json`
  - `character_identity_bank.json`
  - `temporal_regime_plan.json`
  - `title_grammar_bank.json`
  - `world_policy.json`
  - `year_slate_plan.json`
  - `keyword_motif_bank.json`
  - `concept_packs.json`
  - `franchise_bibles.json`

Issue:
- `franchise_bibles.json` for this `50`-movie smoke only produced `1` bible, so franchise continuity is not strongly exercised at this size.

## Movies

OK:
- `movie.arrow` has `50` movies over `2027-2075`.
- Final genre distribution is broad: `20` genres, top genre only `6/50`.
- Final country distribution is broad: `29` countries.
- The earlier future-run genre collapse is no longer present.

Issue:
- Plot summaries are still weak in the current finished outputs.
- Measured on `movie.arrow`:
  - mean plot length about `11.8` words
  - `45/50` summaries still match placeholder-like patterns
- Sample bad summaries still look like:
  - `A intimate Sci-Fi film from Iran (2030). Rated PG-13.`
  - `A gripping Comedy film from China (2032). Rated PG-13.`

Action:
- Step `110` was rerun directly on the finished dataset.
- New movie summary quality after rerun:
  - mean plot length about `37.0` words
  - `0/50` placeholder-like summaries
- This was the biggest concrete quality gain in the review pass.

Dense `100` run observations:
- The non-scratch top-up path works correctly.
- Title-bank distribution after top-up reached:
  - `2064: 12`
  - `2065: 28`
  - `2066: 38`
  - `2067: 15`
  - `2068: 7`
- Final movies in the first dense run tracked that shape almost exactly:
  - `2064: 12`
  - `2065: 28`
  - `2066: 38`
  - `2067: 14`
  - `2068: 8`
- This is strong evidence that the top-up logic is usable for longer runs and not just fresh generation.

Issue:
- The first overnight continuation run that resumed from step `100` was not a clean `2064-2068` evaluation.
- Root cause:
  - `topup_title_bank.py` kept legacy out-of-range rows when reusing an existing `title_bank.csv`
  - `generate_movies.py` then sampled from the entire bank instead of enforcing the requested year window
- Measured effect on the finished dense run:
  - `75/100` movies were in `2064-2068`
  - `25/100` leaked outside the target window, across `2028-2073`

Action:
- Fixed `topup_title_bank.py` so step `60` now sanitizes the existing bank to the requested year window and trims surplus rows to the requested target count before topping up.
- Fixed `run_pipeline.py` so step `60` no longer treats any existing `title_bank.csv` as valid unless it has the exact requested count and all years are inside the configured window.
- Fixed `generate_movies.py` so step `100` now enforces the requested `start_year/end_year` at runtime and refuses to proceed if the in-range title bank is undersized.
- Extended the part-2 publishability pass beyond the immediate bug:
  - `assembly.py` now consumes generated priors for concept-pack scoring, genre->tier distributions, writer/director probabilities, concept latent targets, and release-month seasonality
  - `secondary_tables.py` now uses range-relative TV timing and active-range year fallbacks
  - `big_history_events.py` now maps event eras relative to the template registry instead of a hardcoded `1970-2035` interpolation

## Titles And Taglines

OK:
- No adjacent repeated token artifacts remain in titles or taglines.
- Tagline uniqueness is `50/50`.

Issue:
- Some titles/taglines are still semantically awkward, for example:
  - `The Knight and the Knight`
  - `Quest for the Divine Sinner`
- Current cleanup handles obvious doubled tokens but not broader semantic redundancy.

Post-fix state:
- Two critic passes improved several taglines, for example:
  - `The City Who Saved The Abyss` tagline became `Saving the world, one social disaster at a time.`
  - `Trouble in Europa` tagline became `Love is a bureaucratic nightmare.`
- A few awkward title/tagline outliers still remain and should be treated as second-order polish, not a structural blocker.

Dense `100` run issue discovered:
- The first dense run surfaced a real renderer bug, not just weak creativity.
- Placeholder category labels were leaking directly into titles and taglines, for example:
  - `Action Words or Action Words`
  - `Crown of Abstract Nouns`
  - `The Little Mythic Words`

Root cause:
- `title_grammar_bank.json` legitimately uses placeholders like `{abstract_nouns}`, `{action_words}`, and `{mythic_words}`.
- `topup_title_bank.py` only mapped singular aliases like `abstract_noun` and `action_word`.
- When the plural placeholder was encountered, the fallback path inserted the field name literally.

Fix applied:
- `topup_title_bank.py` now maps plural placeholder names correctly.
- It also rejects rendered titles/taglines containing placeholder labels such as:
  - `Abstract Nouns`
  - `Action Words`
  - `Mythic Words`
  - `Technology Words`
- The renderer now also corrects simple `a/an` article mistakes.

Validation:
- On the rebuilt dense run seed-top-up pass, `title_bank.csv` now has `0` rows containing those leaked placeholder labels.
- A full rebuilt dense run from step `60` is in progress to validate the downstream movie outputs after this fix.

## Keywords

OK:
- Keyword diversity is decent in the smoke run: `234` movie-keyword rows.
- Mean aligned keyword fraction per movie is about `0.694`.

Issue:
- `10/50` movies are below `0.5` keyword alignment.
- Selection logs still show some off-genre bundles before reranking, especially around Experimental movies.

Interpretation:
- Keyword quality is no longer a collapse issue.
- It is now a moderate coherence issue and should be improved after the text layer, not before.

Dense `100` run issue discovered:
- Plot summaries were over-literal because step `110` was grounding directly on raw keyword strings such as:
  - `cross-continental-journey`
  - `fluid-dynamics`
  - `dynamic-lighting`
  - `historical-reconstruction`
- This produced unnatural prose that read like copied metadata rather than narrative text.

Fix applied:
- `generate_plot_summaries_api.py` now converts keyword lists into cleaner story cues before prompting.
- It drops obviously non-narrative visual/rendering jargon and placeholder-like cue fragments.
- The prompt now explicitly tells the model to translate cues into natural prose instead of copying awkward hyphenated phrases verbatim.

Pending validation:
- The rebuilt dense run needs to finish through step `110` before this plot-summary fix can be judged on the actual regenerated `100`-movie outputs.

## Franchises

OK:
- Franchise machinery runs end to end and at least one franchise bible was created and used.

Issue:
- This smoke is too small to judge franchise quality strongly.
- Need a larger run or a franchise-targeted smoke for real continuity evaluation.

## TV

OK:
- `tv_series.csv` has `42/42` series summaries.
- `episodes.csv` has `1113/1113` episode descriptions.
- TV summaries are now structurally complete.

Issue:
- Step `120` needed manual rerun after runtime bugs; the underlying quality looks much better than the initial execution stability.

Fix:
- Runtime bugs in step `120` were fixed.
- The TV summary validator now completed successfully on direct rerun and export was refreshed afterward.
- A second dense-run issue was also fixed:
  - one missing series summary (`True Archive Rising`) was caused by the series summary path having no retry-on-invalid logic
  - `generate_tv_summaries.py` now retries invalid series summaries with a tighter repair prompt
  - direct rerun succeeded and reached `60/60` series summaries and `1483/1483` episode descriptions

## Priors And Planning

OK:
- `modeling_priors.json` is structurally rich and includes meaningful sections for:
  - title generation
  - selection weights
  - edge priors
  - financial priors
  - rerank priors
- `year_slate_plan.json` contains broad, non-trivial market/tier conditioning.
- Future temporal shaping works:
  - phase counts in final movies are `6 / 8 / 4 / 12 / 20` across the five defined future phases

Issue:
- At `50` movies over `49` years, the year curve is only lightly visible; this smoke proves support, not realism at scale.
- Selection logs still show some concept-pack choices with weak confidence and genre hints that are not semantically close to the chosen genre.

Question:
- Is `gemini-3.1-pro-preview` worth it here?

Evidence:
- Run-scoped usage log `20260403_004710_llm_usage.jsonl` shows:
  - `gemini-3.1-pro-preview`: `7` calls, about `37.7k` total tokens
  - `gemini-3.1-flash-lite-preview`: `226` calls, about `2.34M` total tokens

Conclusion:
- Yes, keeping `pro-preview` for priors/bootstrap is currently justified.
- The token footprint is tiny compared with the bulk generation cost, while the artifacts are among the strongest parts of the run.

Additional observation:
- Final movie genre distribution now broadly matches the title-bank distribution, which supports the claim that the concept-selection collapse was fixed.

## Fixes

Applied in this pass:
- reran `generate_plot_summaries_api.py` on the finished future dataset
- added quality-aware step checks in `run_pipeline.py` so:
  - step `110` no longer skips if plots are placeholder-like
  - step `120` no longer skips if TV summaries are only non-empty but low-quality
- added `_dev/rerun_post_generation_critic.py` to refresh critic repairs on a finished dataset
- ran the critic twice on the updated dataset
- reran step `130` export after repairs
- strengthened the plot-summary prompt to stay closer to genre/keyword evidence
- fixed title-bank placeholder alias handling in `topup_title_bank.py`
- added title/tagline rejection for leaked placeholder labels in `topup_title_bank.py`
- added simple title article cleanup in `topup_title_bank.py`
- added keyword-to-story-cue cleanup in `generate_plot_summaries_api.py`
- added retry-on-invalid series summaries in `generate_tv_summaries.py`

Current net effect:
- Plot summaries moved from major blocker to acceptable.
- TV coverage is complete and exported.
- The remaining issues are smaller semantic polish issues, not architectural failures.
- The dense top-up path itself is validated.
- The dense rebuilt run is still needed to confirm the downstream movie text and title fixes together on a fully regenerated dataset.

Dense-run recovery status:
- The rebuilt `100`-movie `2064-2068` rerun did not mark step `100` complete cleanly.
- To make the recovery pass debuggable, `test30` now writes run-scoped decision logs and a dedicated movie-progress log during step `100`.
- `generate_movies.py` now emits explicit `movie_started`, `movie_completed`, `post_loop_start`, `critic_start`, `critic_done`, and `step100_complete` events.
- `generation_critic.py` now persists prompt/report artifacts even on critic failure, so future stalls do not leave an empty trail.
- Recovery plan: rerun step `100` with the post-generation critic disabled, finish through `130`, inspect outputs, then re-enable critic once the dense path is stable again.

## Next Tests

Recommended next validation sequence:

1. Finish the rebuilt dense `100`-movie `2064-2068` run through `130`
   - confirm that placeholder-title leakage is gone in final movies, not just in `title_bank.csv`
   - confirm that plot summaries no longer mirror awkward keyword metadata

2. Once the rebuilt dense run finishes:
   - inspect titles/taglines again
   - inspect critic output again
   - inspect low-alignment keyword cases again

3. Fresh `50`-movie historical research run through `130`
   - ensures the same fixes did not regress the historical range

4. After one clean dense run and one clean historical smoke:
   - rerun PostgreSQL JOB and DuckDB timing if the export is stable enough to compare against `test27`

## Dense Run 2026-04-04

Current run:
- fresh research-mode dense run
- `100` movies
- strict year window `2064-2068`
- completed through step `130`

Strong results:
- strict year-range enforcement now works:
  - final movies outside `2064-2068`: `0`
- movie-year distribution is dense and clearly non-uniform:
  - `2064: 12`
  - `2065: 24`
  - `2066: 36`
  - `2067: 16`
  - `2068: 12`
- title-bank top-up stayed aligned with final movies
- title hygiene is much better:
  - unique titles: `100/100`
  - placeholder-title leakage: `0`
  - repeated-word title hits: `0`
- plot summaries are now materially improved:
  - mean plot length: `40.14` words
  - min/max plot length: `32 / 49`
  - short plots under `20` words: `0`
  - placeholder/generic-heuristic hits: `0`
- diversity is strong for a `100`-movie run:
  - distinct genres: `22`
  - distinct countries: `35`
  - distinct languages: `24`
  - distinct production tiers: `5`
- taglines are mostly unique:
  - unique taglines: `99/100`
- TV output is complete:
  - series summaries: `60/60`
  - episode descriptions: `1825/1825`
- export completed successfully and refreshed the JOB/IMDb bundle

Remaining quality issues:
- the biggest remaining weakness is now semantic polish, not structure
- some titles are still awkward or overly synthetic, for example:
  - `Fear and Fear`
  - `Who Killed the Blood?`
- some taglines are still weak, generic, or only loosely matched to the film:
  - `The Sovereign Escape`
  - `Journey to The Abyss`
  - `Inside the Cybernetic`
- the critic still flagged `5/12` sampled titles, but the flags are now mostly soft-quality issues rather than catastrophic placeholders
- tagline punctuation/style is inconsistent:
  - most taglines are phrase fragments rather than finished copy

Operational notes:
- the dense run required several hardening fixes during execution:
  - company-lexicon artifact normalization
  - character alias placeholder compatibility
  - support for vector-shaped concept style tier shifts in generated priors
- these were true compatibility bugs in the new LLM-authored artifact path, and are now fixed in code

Bottom line:
- this run is structurally strong and qualitatively much better than the previous dense attempts
- the remaining work is mainly language/style polish for titles and taglines, plus cleanup of a few warnings

## Clean Dense Run 2026-04-04 Prior Effectiveness Review

Run under review:
- fresh clean research-mode dense run
- `100` movies
- strict year window `2064-2068`
- run id `20260404_200254`

Artifacts inspected:
- `modeling_priors.json`
- `temporal_regime_plan.json`
- `world_policy.json`
- `year_slate_plan.json`
- `concept_packs.json`
- `franchise_bibles.json`
- `movie.csv`
- `movies_flat.csv`
- `critic_report.json`
- `imdb_schema/*`
- `decision_logs/20260404_200254_*`

Quality signals:
- final year counts are:
  - `2064: 12`
  - `2065: 27`
  - `2066: 38`
  - `2067: 15`
  - `2068: 8`
- title-bank year counts were:
  - `2064: 12`
  - `2065: 28`
  - `2066: 40`
  - `2067: 15`
  - `2068: 5`
- genre diversity is broad:
  - `23` genres
  - top genre only `14/100`
- text quality is solid:
  - mean plot length `38.62` words
  - generic plot hits by heuristic `0`
  - unique titles `100/100`
  - unique taglines `99/100`
- export-level correlations are sensible:
  - budget vs box office: `0.7475`
  - rating vs awards: `0.4405`
  - rating vs average critic score: `0.9925`
  - rating vs ratings-breakdown average: `0.9911`

Important finding:
- the explicit `modeling_priors.json` control plane is still mostly empty in this fresh clean run
- non-empty sections:
  - `person_generation`
- empty sections:
  - `company_generation`
  - `keyword_generation`
  - `character_generation`
  - `title_generation`
  - `company_finance_tiers`
  - `selection_weights`
  - `edge_priors`
  - `financial_priors`
  - `secondary_table_priors`
  - `history_event_priors`
  - `rerank_priors`

Interpretation:
- This run **does not yet prove** that the new `modeling_priors.json` layer is strongly steering the system.
- The good quality of the run is currently being driven mainly by the broader LLM planning stack:
  - `temporal_regime_plan.json`
  - `world_policy.json`
  - `year_slate_plan.json`
  - `concept_packs.json`
  - `franchise_bibles.json`
  - `title_grammar_bank.json`
- Strong evidence for that:
  - temporal regime year weights were `0.12 / 0.28 / 0.40 / 0.15 / 0.05`
  - realized movie year shares were `0.12 / 0.27 / 0.38 / 0.15 / 0.08`
  - title-bank vs final movie year cosine similarity: `0.9979`
  - title-bank vs final movie genre cosine similarity: `0.9902`
  - movie-selection logs show `35%` concept-pack usage and `93%` genre-hint preservation

So the current state is:
- **LLM planning works**
- **LLM bootstrap/content banks work**
- **LLM-authored temporal shaping works**
- **explicit LLM-authored scalar/table priors are not validated yet**, because the populated artifact is mostly empty and the run is falling back to code defaults for many of those sections

Remaining concrete issues from this run:
- alternate-title grammar bug still exists for at least:
  - `La Le Cipher Identity`
  - `Los El Algorithm Protocol`
- critic still flagged `2` plot summaries as generic in its sample
- taglines are still short and slogan-like on average (`3.43` words)
- `award_campaign_strength` has only weak observed relation to awards (`0.1581`) and rating (`0.1458`), so that lever is not behaving as strongly as intended

Action implication:
- Before claiming that `modeling_priors.json` solves hard-coded scoring/modeling, we need a fresh run where the prior sections are actually populated and then compare the output against the current mostly-fallback run.
- Until then, the honest paper claim is stronger around:
  - LLM-authored planning artifacts
  - LLM-authored bootstrap banks
  - deterministic realization preserving LLM-imposed structure
- and weaker around:
  - LLM-authored scalar/table priors replacing handcrafted modeling constants

Additional diagnosis:
- The issue is **not** simply that the expensive model cannot generate rich priors.
- Saved artifact debug files show that `gemini-3.1-pro-preview` did produce a detailed multi-section priors payload in:
  - `_dev/bootstrap_artifacts/modeling_priors_failed_raw.txt`
- That raw payload already contained concrete sections for:
  - `title_generation.genre_base_weights`
  - `company_finance_tiers`
  - `selection_weights`
  - `edge_priors`
  - `financial_priors`
  - and many other useful knobs
- So the current blocker is more specific:
  - generation/parsing/validation/persistence of `modeling_priors.json` is not yet robust
  - the successful clean run ended up with a mostly empty normalized artifact, so execution likely fell back to code defaults even though the model is capable of supplying richer priors

Refined conclusion:
- The **idea** of LLM-authored priors looks viable.
- The **current deployment** of those priors is not yet reliable enough to claim success.

## Clean Dense Run 2026-04-04 With Populated Priors

Run under review:
- fresh research-mode dense rerun
- `100` movies
- strict year window `2064-2068`
- run id `20260404_221528`
- this run used the repaired grouped `modeling_priors` generation path

Core outcome:
- the run completed through step `130`
- `modeling_priors.json` is now populated across all major sections instead of being mostly empty
- populated sections include:
  - `title_generation`
  - `selection_weights`
  - `edge_priors`
  - `financial_priors`
  - `secondary_table_priors`
  - `history_event_priors`
  - plus the bootstrap/foundation sections

Quality signals:
- movies: `100`
- year counts:
  - `2064: 12`
  - `2065: 20`
  - `2066: 38`
  - `2067: 17`
  - `2068: 13`
- title-bank year counts:
  - `2064: 12`
  - `2065: 22`
  - `2066: 38`
  - `2067: 18`
  - `2068: 10`
- title-bank vs final movie year cosine: `0.9972`
- title-bank vs final movie genre cosine: `0.9807`
- genres / countries / languages / tiers:
  - `25 / 30 / 22 / 5`
- unique titles / taglines:
  - `100 / 97`
- mean plot length:
  - `37.61` words
- generic plot hits by heuristic:
  - `0`

Observed correlations:
- budget vs box office: `0.6229`
- rating vs awards: `0.4514`
- rating vs average critic score: `0.9859`
- rating vs ratings-breakdown average: `0.9839`
- award campaign vs awards: `0.2041`
- award campaign vs rating: `0.0957`

Interpretation:
- This is the first dense run where the explicit `modeling_priors` control plane is actually populated and active.
- The prior-control story is now much more defensible than before.
- We still do not have a perfect before/after output comparison, because the fresh rerun overwrote the old materialized outputs.
- Even so, this run is now valid evidence that:
  - the populated prior artifact can be generated reliably
  - the pipeline can consume it end to end without collapsing
  - the resulting dataset remains structurally strong and realistic enough for the benchmark objective

Remaining concrete issues:
- critic still flagged `4` sampled titles:
  - `title_id=34`: generic plot summary; tagline lacks biographical focus
  - `title_id=3`: generic plot summary; tagline does not fit Animation genre
  - `title_id=13`: irrelevant keyword `Geological Survey`
  - `title_id=27`: generic plot summary; tagline does not fit Romance genre
- alternate-title grammar bug still exists in the export:
  - `La Le Fallen The Island`
  - `La Le Crimson Blood`
- taglines remain short on average (`3.51` words) and are still one of the weaker creative layers

Current honest verdict:
- The rerun quality is good.
- The explicit LLM-authored priors layer now looks real rather than decorative.
- The strongest remaining issues are creative polish and a few export/auxiliary-text bugs, not structural generation failure.

## 2026-04-05 Structural Polish Pass

Source-side polish implemented:
- added shared text sanitation helpers in `text_polish.py`
- `topup_title_bank.py` now rejects weak/redundant titles and title-like weak taglines during research generation
- `assembly.py` now sanitizes selected titles, taglines, and character names, and it avoids weak title-bank entries at pick time
- `generate_character_bank.py` now strips duplicated articles from moniker-based names such as `The The Oracle`
- `secondary_tables.py` now avoids double-localized alternate-title prefixes like `La Le ...`
- `generate_plot_summaries_api.py` now gates a few over-literal cue phrases such as `geological survey` outside the genres where they plausibly belong
- `generation_critic.py` now supports `remove_keyword` repairs, in addition to rewrite/append actions
- `generate_bootstrap_artifacts_api.py` now asks for stronger title/tagline grammar and a less bureaucratic keyword seed bank
- `export_imdb_schema.py` no longer emits the pandas concat warning after the final `_concat_nonempty` fix

Current dense-run outputs also polished in place:
- ran `polish_outputs.py` on the active `test30` outputs
- structural artifacts removed from current files, including:
  - `The Furious The Core` -> `The Furious Core`
  - `The Rogue The Abyss` -> `The Rogue Abyss`
  - `The Fallen The Island` -> `The Fallen Island`
  - `La Le ...` export variants removed
  - `The The ...` character names removed from character/export tables

Bounded critic refresh on current outputs:
- reran the post-generation critic after the structural polish
- it applied `remove_keyword` repairs for several stray `Geological Survey` usages and rewrote one vague short-format plot summary
- reran the IMDb export afterward so the core JOB/IMDb bundle matches the polished output state

Small caveat:
- the convenience `movies_flat.csv` extra table can lag behind critic-only repairs because the critic refresh does not rebuild the full flat-table export path
- the core benchmark tables in `imdb_schema/` are synchronized and correct

## 2026-04-06 Evidence And Coherence Hardening

What was fixed at the source:
- `generation_critic.py` now persists the actual repair list into `critic_report.json`, including repair types and rewrite-vs-deterministic counts
- `bootstrap_artifacts.py` now records richer critic provenance in `research_mode_audit.json`, including sampled titles, flagged titles, cache hit, and repair breakdowns
- `assembly.py` now audits `modeling_priors.json` section/key usage during runtime selection, so `selection_weights.*` consumption is visible in the research audit instead of being underreported
- `generate_plot_summaries_api.py` now refreshes `movies_analysis.arrow/csv` alongside `movie` and `movies_flat`, so step `110` no longer leaves a stale generic analysis artifact behind
- `export_imdb_schema.py` now rebuilds the exported `movies_analysis.csv` from the current `movie` and `movies_flat` tables instead of blindly copying a possibly stale source file
- `generate_bootstrap_artifacts_api.py` now validates `keyword_seed_bank` against the full benchmark genre list, not just a minimum row count
- `generate_keywords_procedural.py` now fail-closes research mode if the keyword seed bank omits any benchmark genres
- `assembly.py` keyword cleanup is stricter about story-match coverage, generic keyword caps, and off-genre limits
- `generation_critic.py` now rejects weak rewritten taglines instead of accepting any short slogan-like replacement

Immediate effect on the current finished run:
- refreshed `imdb_schema/movies_analysis.csv` now mirrors the real plot summaries instead of the old boilerplate `A [genre] film from ...` descriptions
- the currently saved `critic_report.json` and `research_mode_audit.json` still reflect the older run-time logging path, because that run completed before the new provenance code existed

Implication for the next rerun:
- the next dense research rerun should produce cleaner evidence artifacts, not just cleaner movies
- keyword coherence should also improve earlier in the pipeline because incomplete genre coverage in `keyword_seed_bank` will now fail at the source instead of surfacing as mixed off-genre selections later

## 2026-04-06 Clean Dense Rerun `20260406_124630`

Outcome:
- fresh dense research rerun completed cleanly through `130`
- `research_mode_audit.json` now shows `fallback_hit_count = 0`
- critic provenance is now properly populated:
  - `applied = 10`
  - `repair_types = {"rewrite_plot_summary": 5, "rewrite_tagline": 4, "remove_keyword": 1}`
  - `llm_rewrite_actions = 9`
  - `deterministic_sanitation_actions = 1`
- `modeling_priors.json` usage evidence is much richer now:
  - `61` logged `selection_weights.*` sections in the audit

Strong results:
- strict dense range still holds:
  - movie years: `2064:12, 2065:22, 2066:39, 2067:18, 2068:9`
  - title-bank years: `2064:12, 2065:22, 2066:40, 2067:18, 2068:8`
  - year cosine: `0.9997`
- title-bank vs movie genre cosine: `0.9935`
- diversity is strong:
  - `25` genres
  - `36` countries
  - `21` languages
  - `100` unique titles
  - `99` unique taglines
- main text quality is solid at the dataset level:
  - mean plot length: `40.89` words
  - generic/placeholder plot hits by heuristic: `0`
  - mean tagline length: `7.43` words
  - weak tagline hits by simple heuristic: `0`
- hygiene remains clean:
  - `0` numeric company suffix rows
  - `0` `La Le` / `Los El` / `The The` export artifacts
  - `0` generic rows left in `imdb_schema/movies_analysis.csv`
- internal structure remains believable:
  - budget vs box office: `0.7485`
  - rating vs awards: `0.2958`
  - box office vs votes: `0.9734`
  - rating vs ratings-breakdown average: `0.9875`

Main remaining blocker:
- the keyword layer is still structurally wrong for this dense configuration
- current generated `entities/keyword.csv` only covers:
  - `Action, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, History, Horror`
- current movie output uses additional genres with no direct keyword inventory support, including:
  - `Disaster, Experimental, Martial Arts, Music, Musical, Reality-TV, Romance, Sci-Fi, Short, Superhero, Thriller, War`
- coherence metrics reflect that failure:
  - mean keyword alignment: `0.144`
  - movies below `0.5` keyword alignment: `82/100`
  - movies with off-genre keywords: `91/100`
  - movies with generic keyword-heavy sets: `90/100`

Interpretation:
- this is not just a tuning problem in `pick_keywords()`
- there is a design mismatch between:
  - the benchmark genre set used by movie generation
  - the size target `n_keywords = 220`
  - and the current procedural keyword realization strategy, which greedily fills early genres and truncates later ones
- in other words, the run proves the new audit/evidence layer works, but it also proves the keyword-generation design is still the clearest remaining structural blocker before a final paper-quality run

Creative quality note:
- despite the improved critic and audit path, some title/tagline semantics are still awkward
- examples from the current run:
  - `Survive the Machine. the enemy.`
  - `Pray for Doom. Prepare for the Last end.`
  - `They took his Ocean. Now he will Die them all.`
- these are now polish problems rather than architecture problems, but they are still visible

Current honest verdict:
- evidence path: much better and now paper-defensible
- core movie/table generation: strong
- keyword layer: still not final
- title/tagline semantics: improved, but still need one more polish pass after the keyword fix
