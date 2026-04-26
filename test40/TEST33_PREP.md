# Test33 Prep

`test33` is the clean candidate-run workspace derived from the current optimized `test32` code.

What is intentionally included:
- current source code and launcher scripts
- local-model helper scripts under `local_ollama_qwen35/`
- small static artifact/config files such as `modeling_priors.json`
- `benchmark_candidate_profile.json` copied from `benchmark/job_exact_v1/profiles/`

What is intentionally not copied from `test32`:
- generated tables like `*.csv`, `*.arrow`, `*.duckdb`
- old `entities/` contents
- `decision_logs/`, `graph/` runtime artifacts, `imdb_schema/`, checkpoints, caches
- signoff reports, q-error outputs, recovery backups, and other run products

The goal is to keep `test32` as the benchmark/debug baseline and use `test33` as the fresh candidate-run workspace.

Linux launcher:
- `./run_benchmark_candidate_50k.sh`

Useful overrides:
- `DATA_SYS_MODEL` to switch the generation model
- `DATA_SYS_N_MOVIES` to change the movie target
- `DATA_SYS_PIPELINE_CONFIG` to point at a different config profile
- `PYTHON_BIN` to select a specific Python interpreter
