# Structural Hardening Notes

Generated for the lab-facing 20k milestone.

## Current `test38` Baseline

The existing 20k output is useful as evidence, but it is not the final dataset:

- `20000` movie rows were generated successfully.
- Summary enrichment is intentionally deferred for benchmark-first work.
- Remaining structural failures are title uniqueness, exact-topic keyword support on 5 movies, and actor reuse concentration.
- The actor reuse failure is partly a configuration feasibility issue: the run used `32000` people and about `20526` actor-role people for about `320024` cast rows, making a reuse ratio below `15` almost impossible.

## Fixes Landed For The Next Run

- Added exact-topic keyword repair after LLM keyword reranking so the critic should not fail on zero exact-topic support.
- Added canonical cleaned-title uniqueness pressure so near-duplicate/raw duplicate titles are rejected before final assignment.
- Made step-100 sanity reporting benchmark-first by default, so generic plot summaries are reported but do not fail the structural gate before step 110.
- Added a structural preflight that rejects expensive step-100 runs with impossible actor-pool or entity-bank sizing.
- Widened and deepened cast exploration so the selector can use the long-tail actor pool instead of only high-score near-neighbors.
- Added stronger actor saturation penalties and stronger unused-actor boosts while preserving graph/community/style/director/country correlations.
- Added a Linux local-LLM starter script and provider smoke check for vLLM/Ollama/OpenAI-compatible servers.

## Recommended Next 20k Config

For a fresh lab run:

- `N_MOVIES=20000`
- `N_PERSONS=64000`
- `N_COMPANIES=1400`
- `N_KEYWORDS=3200`
- `N_CHARACTERS=340000`
- `N_TITLES=26000`

For a laptop reuse-seed run, prefer older entity sources with more actor-role people, such as `test23`, then regenerate the character bank and top up the title bank before step 100.

## Lab Command Sequence

```bash
./detect_ollama_endpoint.sh
source reports/ollama_detected.env
./run_lab_local_llm.sh check-provider
RUN_PROFILE=tiny10_1y ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-full100
RUN_PROFILE=small100_2y ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-full100
RUN_PROFILE=candidate20k ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-full100
./run_lab_local_llm.sh sanity
./run_lab_local_llm.sh export-only
```

Use `./run_lab_local_llm.sh resume20k-step100` if step 100 is interrupted after the year-boundary resume workspace has been created.

## Rerun Boundary

The latest code changes mainly affect step 100 and entity sizing.  For an
existing folder with valid artifacts and graph inputs, rerun step 100 after
preflight passes.  For local-LLM lab validation, run at least `tiny10_1y`
through step 98/100 from fresh outputs so the local model path itself is tested.
