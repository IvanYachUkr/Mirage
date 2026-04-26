# JOB-Complex V1

This folder stages the official JOB-Complex SQL workload for later evaluation
against the real IMDb database and Mirage-generated IMDb/JOB exports.

## Contents

- `original/JOB-Complex.sql`: official upstream SQL file from
  `DataManagementLab/JOB-Complex`.
- `original_queries/`: the same workload split into 30 one-query SQL files.
- `run_job_complex_postgres.py`: lightweight PostgreSQL runner that records
  actual result rows, root planner estimates, root q-error, timings, and SQL
  errors.
- `run_job_complex_explain_analyze.py`: full PostgreSQL `EXPLAIN ANALYZE`
  runner that stores raw JSON plans plus per-node estimated/actual rows and
  q-error metrics.
- `load_original_imdb_postgres.py`: streaming loader for the local
  `imdb_job_dataset/*.csv` original JOB IMDb files.

## Source

- GitHub: https://github.com/DataManagementLab/JOB-Complex
- Paper/workload description: JOB-Complex contains 30 SQL queries with non-FK
  joins, string joins, and complex predicates.
- Zenodo redistribution reference: https://zenodo.org/records/19205561

## Usage

If the original JOB IMDb CSVs are not already loaded into Postgres, load them
once with:

```bash
python3 benchmark/job_complex_v1/load_original_imdb_postgres.py \
  --dataset-dir imdb_job_dataset \
  --db-name imdb_original_job \
  --pg-container pg_bench \
  --docker-bin "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" \
  --drop-create
```

The loader streams local CSV files into `COPY ... FROM STDIN`; it does not copy
another full dataset into the Docker container. The real local CSV bundle has
about 2.5M `title` rows and 36M `cast_info` rows, so this can still take time
and disk I/O.

Run against a PostgreSQL database that already contains the IMDb/JOB schema:

```bash
python3 benchmark/job_complex_v1/run_job_complex_postgres.py \
  --pg-container pg_bench \
  --db-name imdb \
  --docker-bin "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" \
  --out-dir benchmark/job_complex_v1/runs/real_imdb_pg
```

For a quick focused probe:

```bash
python3 benchmark/job_complex_v1/run_job_complex_postgres.py \
  --pg-container pg_bench \
  --db-name imdb \
  --query-ids 01,02,03 \
  --out-dir benchmark/job_complex_v1/runs/real_imdb_pg_probe
```

For cardinality-estimation metrics, use the `EXPLAIN ANALYZE` runner:

```bash
python3 benchmark/job_complex_v1/run_job_complex_explain_analyze.py \
  --pg-container pg_bench \
  --db-name imdb \
  --docker-bin "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" \
  --timeout-sec 300 \
  --out-dir benchmark/job_complex_v1/runs/real_imdb_explain_analyze
```

The analyzer emits:

- `raw_plans/<query_id>.json`: full PostgreSQL JSON plans.
- `query_summary.csv/json`: one row per JOB-Complex query.
- `plan_nodes.csv`: one row per physical plan node with `Plan Rows`,
  `Actual Rows`, `Actual Loops`, `q_error_per_loop`, and `q_error_total`.
- `summary.json` and `report.md`: corpus-level q-error summaries.

Use the same command shape for Mirage exports after loading the exported
IMDb/JOB schema into Postgres; only `--db-name` and `--out-dir` need to change.

## Notes For Mirage

JOB-Complex is intentionally stricter than canonical JOB. It references columns
such as phonetic codes and uses non-key/string join predicates. If a Mirage
strict export does not yet populate those columns well, JOB-Complex may expose
real schema/value-coverage gaps. That is useful, but it should be treated as the
second benchmark milestone after canonical JOB is stable.
