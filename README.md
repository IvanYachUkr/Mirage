# Mirage Lab Handoff

This branch is a source-only lab handoff for running the Mirage IMDb/JOB-style generator with a local LLM. It intentionally excludes generated datasets, Arrow/CSV exports, DuckDB files, API keys, logs, and resume checkpoints.

## 1. Install Python Dependencies

```bash
cd test40
./setup_linux_env.sh
source .venv-linux/bin/activate
```

## 2. Start A Local LLM

Preferred lab path with an OpenAI-compatible vLLM server:

```bash
export LOCAL_LLM_MODEL="<lab-model-name-or-path>"
python -m vllm.entrypoints.openai.api_server --model "$LOCAL_LLM_MODEL" --host 0.0.0.0 --port 8000
```

Pipeline environment for that server:

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_MODEL="<lab-model-name-or-path>"
export LOCAL_LLM_API_KEY="not-needed"
```

Ollama fallback/profile installer:

```bash
LOCAL_OLLAMA_PROFILE_ID=qwen36_35b_a3b_mxfp4_moe ./local_ollama_qwen35/install_all.sh
./detect_ollama_endpoint.sh
source reports/ollama_detected.env
```

Check provider wiring:

```bash
./run_lab_local_llm.sh check-provider
```

## 3. Run Smoke And Candidate Jobs

Small local smoke:

```bash
./run_lab_100_fresh.sh ../lab_smoke_100
```

First useful candidate:

```bash
./run_lab_20k_fresh.sh ../lab_candidate_20k
```

Larger candidates, once the smoke and 20k path are stable:

```bash
./run_lab_50k_fresh.sh ../lab_candidate_50k
./run_lab_100k_fresh.sh ../lab_candidate_100k
./run_lab_200k_fresh.sh ../lab_candidate_200k
```

Resume a year-boundary step-100 run:

```bash
cd ../lab_candidate_20k
./run_lab_local_llm.sh resume20k-step100
./run_lab_local_llm.sh resume-progress
```

Validate/export after a run:

```bash
./run_lab_local_llm.sh sanity
./run_lab_local_llm.sh export-only
```

## 4. Benchmark Tooling

- Exact JOB adapter/runner: `benchmark/job_exact_v1/`
- JOB-Complex adapter/runner: `benchmark/job_complex_v1/`
- PostgreSQL/DuckDB/MSCN/zero-shot CE tooling: `benchmark/cardinality_methods/`
- Canonical original JOB SQL: `imdb_job_dataset/job_queries/`

Generated benchmark outputs should go under local `runs/` folders and should not be committed.

