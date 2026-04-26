# Cardinality / Cost Estimation Methods

This folder stages external learned-estimation code that we may use to evaluate
Mirage beyond PostgreSQL/DuckDB q-error baselines.

The goal for this pass is preparation, not a full integration. Heavy training
artifacts and large benchmark datasets are intentionally not downloaded here.

## Staged Sources

- `sources/learnedcardinalities-master/`: MSCN implementation from
  https://github.com/andreaskipf/learnedcardinalities
- `sources/zero-shot-cost-estimation-main/`: zero-shot cost model code from
  https://github.com/DataManagementLab/zero-shot-cost-estimation
- `archives/`: optional downloaded source archives used to create the staged
  folders. Archives are intentionally not committed in this handoff branch.

## Why These Two First

- MSCN is the historical learned cardinality baseline from "Learned
  Cardinalities: Estimating Correlated Joins with Deep Learning".
- Zero-shot cost estimation is relevant to the lab note about "zero shot" and
  gives us a path toward out-of-the-box learned cost/cardinality experiments.

## Deferred / Optional

- CEB: https://github.com/learnedsystems/CEB
- LCM Eval: https://github.com/DataManagementLab/lcm-eval
- DeepDB: https://github.com/DataManagementLab/deepdb-public

These are useful, but their surrounding datasets/artifacts can become large.
Download them only when we are ready to integrate a specific experiment.

## Current Local Integration

The `scripts/` folder now contains a lightweight, reproducible evaluation
scaffold for the four lab-note baselines:

- `collect_postgres_plan_trace.py`: PostgreSQL `EXPLAIN ANALYZE` cardinality
  trace collector. It emits raw JSON plans, per-query summaries, per-node
  q-errors, CE target-cardinality labels, and a manifest. For aggregate/count
  queries, `target_actual` is taken from the first non-wrapper plan node so the
  learned-method labels are not the aggregate node's single output row.
- `run_duckdb_corpus.py`: DuckDB execution/timing baseline on the same adapted
  SQL corpus.
- `export_mscn_workload.py`: MSCN-compatible workload exporter. It writes
  `.csv`, `.sql`, `.bitmaps`, `column_min_max_vals.csv`, and a manifest.
- `prepare_zero_shot_inputs.py`: packages PostgreSQL plan traces and `pg_stats`
  into a stable directory for zero-shot/lcm-style integration.
- `prepare_zero_shot_parsed_workload.py`: converts Mirage PostgreSQL trace CSVs
  into a minimal DBGen-style parsed workload JSON that the staged zero-shot
  dataloader can consume for smoke/integration tests.
- `run_ce_suite.sh`: orchestration wrapper that runs the above for exact JOB
  and, optionally, JOB-Complex.
- `summarize_ce_run.py`: writes a compact human-readable readiness summary for
  a completed CE run.
- `audit_cleanup_candidates.py`: writes a non-destructive cleanup audit.

Example after a strict IMDb export has been loaded into Postgres:

```bash
bash benchmark/cardinality_methods/scripts/run_ce_suite.sh \
  --pg-db job_test40_structural_probe_v10_pcodes \
  --duckdb-path test40/imdb_schema_structural_probe_v10/test40_v10_core.duckdb \
  --out-dir benchmark/cardinality_methods/runs/test40_v10
```

If DuckDB is not built yet, omit `--duckdb-path` or pass `--skip-duckdb`.
On this WSL setup, prefer the direct snap binary path
`--duckdb-bin /snap/duckdb/9/duckdb`; `/snap/bin/duckdb` may fail inside the
sandbox because Snap tries to create a runtime directory under `/run/user`.

To build a strict-core DuckDB bundle without the Python `duckdb` package:

```bash
python3 benchmark/job_exact_v1/build_duckdb_cli.py \
  --schema-dir test40/imdb_schema_structural_probe_v10 \
  --contract-dir test40 \
  --out test40/imdb_schema_structural_probe_v10/test40_v10_core.duckdb \
  --core-only \
  --overwrite \
  --duckdb-bin /snap/duckdb/9/duckdb
```

## Scientific Status

PostgreSQL and DuckDB are directly runnable baselines.

MSCN is staged and the exporter is operational. The default exporter can still
emit all-ones plumbing bitmaps, but `build_mscn_materialized_bitmaps.py` now
builds real materialized-sample bitmaps from PostgreSQL for exact-JOB style
workloads. For a paper-quality MSCN number, the remaining work is mainly
protocol quality: enough train/test examples, clear split rules, and documenting
the bitmap bridge's predicate handling.

The zero-shot packaging script prepares JSON plans and PostgreSQL statistics.
A minimal parsed-plan bridge can now execute a smoke train/eval, but a
paper-quality zero-shot baseline still needs a fuller PostgreSQL JSON-plan
converter with real output/filter-column features rather than the smoke
converter's conservative dummy features.

## Current Learned-Method Status

Local exploratory runs produced the first reusable learned-method executions,
but generated `runs/` folders are intentionally excluded from this source-only
handoff branch. Regenerate them with `scripts/run_ce_suite.sh` once a strict
IMDb export is available on the lab machine.

Those local runs validated the following integration paths:

- MSCN exact-JOB smoke trains for three CPU epochs over the exported exact-JOB
  workload and writes predictions. This proves the MSCN code path can consume
  Mirage exports, but it is not a scientific baseline yet because the current
  `.bitmaps` file is still synthetic all-ones plumbing.
- MSCN exact-JOB real-bitmap run trains for ten CPU epochs over real
  PostgreSQL-derived materialized sample bitmaps.
- Zero-shot exact-JOB smoke converts 40 Mirage PostgreSQL traces into a minimal
  parsed workload, gathers upstream feature statistics, trains one CPU epoch,
  saves a checkpoint, and writes a test CSV. This proves the upstream model
  stack can ingest Mirage trace structure, but the converter currently uses a
  conservative graph subset with dummy output/filter-column features.
- Zero-shot exact-JOB full parsed-workload CPU run converts all 113 traces,
  trains for five CPU epochs, and writes a reusable checkpoint/test CSV. This is
  still an integration result rather than a pretrained zero-shot baseline.
