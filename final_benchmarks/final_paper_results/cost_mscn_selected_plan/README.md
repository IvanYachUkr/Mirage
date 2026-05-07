# MSCN Selected-Plan Cost Estimation

Status: completed final paper result.

This is an MSCN-style selected-plan runtime estimator trained on Mirage SQL
strings plus sample bitmaps. It is separate from the MSCN cardinality run and
predicts PostgreSQL selected-plan execution runtime in microseconds.

Source run:

```text
benchmark/cardinality_methods/cost_models/runs/test42_100k_selected_plan_mscn_cost_10k_100ep_3seed_20260505
```

Configuration:

- dataset: `imdb_test42_100k_exportpatch`
- training labels: 10,000 non-evaluation Mirage synthetic SQL queries labeled
  with PostgreSQL `EXPLAIN ANALYZE`
- model seeds: `1,2,3`
- epochs: `100`
- sample bitmaps: `1000` deterministic random samples per table
- runtime-label note: PostgreSQL gather workers are disabled around
  `EXPLAIN ANALYZE` to avoid Docker shared-memory failures; benchmark SQL text
  is not rewritten

## Aggregate Result

Recommended reporting policy: median of seed summaries.

| Workload | Queries | Median Q-error | P95 Q-error | Best seed by P95 |
| --- | ---: | ---: | ---: | ---: |
| JOB-Light | 70 | 1.646354 | 6.967269 | 1 |
| JOB | 113 | 1.852447 | 36.927114 | 2 |
| JOB-Complex | 30 | 1.529812 | 4.309624 | 2 |

## Files

- `mscn_cost_aggregate_summary.csv`
- `mscn_cost_aggregate_summary.md`
- `mscn_cost_seed_metrics.csv`
- `selected_plan_mscn_cost_report.json`
- `training_rows_manifest.json`
- `coverage_audit.json`
- `seed_<N>/qerror_summaries.json`
- `seed_<N>/predictions/job_light.csv`
- `seed_<N>/predictions/job.csv`
- `seed_<N>/predictions/job_complex.csv`
