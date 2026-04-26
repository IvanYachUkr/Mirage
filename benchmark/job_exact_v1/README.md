# JOB Exact V1

Stable benchmark workspace for the canonical 113-query JOB corpus.

## Contents

- `exact_job_benchmark.py`
  - copies the canonical JOB SQL files
  - adapts queries by replacing only literal constants
  - enforces exact-structure identity with a normalized structure checksum
  - validates the adapted corpus on Postgres and DuckDB
  - emits `manifest.json`, `coverage_report.md`, and machine-readable result files

- `profile_step100_runtime.py`
  - profiles step-100 runtime from existing run logs
  - summarizes stage totals, year hotspots, and graph build timing

- `profiles/benchmark_candidate_profile.json`
  - benchmark-first runtime/profile defaults for the first larger candidate run
  - disables the heavyweight `reviews` secondary table
  - pushes actor/director reuse and graph closure in a more realistic direction

## Typical Commands

```bash
python3 benchmark/job_exact_v1/exact_job_benchmark.py \
  --source-query-dir imdb_job_dataset/job_queries \
  --dataset-dir test32 \
  --db-name imdb_benchmark \
  --out-dir benchmark/job_exact_v1
```

```bash
python3 benchmark/job_exact_v1/profile_step100_runtime.py \
  --run-dir test32 \
  --out-dir benchmark/job_exact_v1/runs/test32
```
