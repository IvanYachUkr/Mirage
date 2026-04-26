# Mirage Lab Candidate

This repository contains the Mirage synthetic IMDb/JOB benchmark generator.

For the current lab run, use `test41`. It is the source-first lab candidate with:

- local-LLM run wrappers for smoke, 20k, 50k, 100k, and 200k movie runs
- explicit step-100 year-boundary resume support
- benchmark-oriented runtime profiles
- strict IMDb/JOB export tooling
- JOB, JOB-Complex, PostgreSQL/DuckDB, MSCN, and zero-shot integration scaffolding
- step-100 speed and sanity reporting

`test40` is kept as the previous baseline snapshot. Prefer `test41` unless you specifically need to compare against the older state.

## Quick Start

```bash
cd test41
./lab_smoke_100.sh start
```

Resume the smoke test:

```bash
./lab_smoke_100.sh continue
```

Start fresh lab candidates:

```bash
./lab_20k.sh start
./lab_50k.sh start
./lab_100k.sh start
./lab_200k.sh start
```

Resume lab candidates:

```bash
./lab_20k.sh continue
./lab_50k.sh continue
./lab_100k.sh continue
./lab_200k.sh continue
```

## Local LLM

For vLLM or another OpenAI-compatible local server:

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_MODEL="<lab-model-name-or-path>"
export LOCAL_LLM_API_KEY="not-needed"
```

For Ollama, the wrappers try to detect the endpoint automatically. To allow automatic Ollama model installation:

```bash
LAB_AUTO_INSTALL_OLLAMA=1 ./lab_smoke_100.sh start
```

## Emergency Resume

The wrappers reload the matching `local_run_profiles/*.env` file, so movie/entity counts are restored automatically.

```bash
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=70 UNTIL_STEP=100 ./lab_20k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=80 UNTIL_STEP=100 ./lab_20k.sh continue
LAB_CONTINUE_TARGET=resume-profile-step100 ./lab_20k.sh continue
```

Replace `lab_20k.sh` with `lab_50k.sh`, `lab_100k.sh`, or `lab_200k.sh` for larger runs. Use `FROM_STEP=<step>` if the run stopped earlier.

## Benchmark Tooling

- Exact JOB: `benchmark/job_exact_v1/`
- JOB-Complex: `benchmark/job_complex_v1/`
- PostgreSQL/DuckDB/MSCN/zero-shot CE tooling: `benchmark/cardinality_methods/`
- Original JOB SQL: `imdb_job_dataset/job_queries/`

Generated datasets, Arrow/CSV exports, DuckDB files, logs, virtualenvs, API keys, and checkpoints should stay local and should not be committed.
