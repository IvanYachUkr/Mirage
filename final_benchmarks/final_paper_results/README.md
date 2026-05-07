# Final Paper Results Index

This folder collects compact paper-facing result artifacts for the
`imdb_test42_100k_exportpatch` dataset. Source runs remain in their original
locations; this directory is an index/copy bundle so the final numbers are easy
to find in one place.

## Cardinality Estimation

### Artifact and Workload Summary

Folder: `artifact_workload_summary/`

Generated database statistics, integrity checks, and JOB-family compatibility
audit. This section also includes the actual result-size distribution for the
final rebound nonzero queries.

| workload | queries | median actual rows | p95 actual rows | max actual rows |
|---|---:|---:|---:|---:|
| JOB-Light | 70 | 1097520.50 | 21561940 | 48779363 |
| JOB | 113 | 678 | 253517 | 911191 |
| JOB-Complex | 30 | 16.50 | 68758.35 | 120906 |

Key files:

- `artifact_workload_summary/generated_database_summary.csv`
- `artifact_workload_summary/generated_database_integrity_summary.csv`
- `artifact_workload_summary/workload_compatibility_summary.csv`
- `artifact_workload_summary/query_result_size_summary.csv`
- `artifact_workload_summary/query_structure_audit.csv`
- `artifact_workload_summary/artifact_workload_audit.json`

### MSCN

Folder: `mscn_cardinality/`

Recommended reporting policy: use the three-seed aggregate
`median_of_seed_summaries`, not the single best seed.

| workload | queries | median Q-error | p95 Q-error |
|---|---:|---:|---:|
| JOB-Light | 70 | 2.38 | 12.24 |
| JOB | 113 | 5.94 | 113.86 |
| JOB-Complex | 30 | 5.27 | 3193.26 |

Key files:

- `mscn_cardinality/mscn_aggregate_table_policy.csv`
- `mscn_cardinality/mscn_seed_metrics.csv`
- `mscn_cardinality/mscn_seed_qerror_handlebars.svg`
- `mscn_cardinality/mscn_signed_qerror_bias_by_seed.svg`
- `mscn_cardinality/artifact_audit.json`

### Base-Selection Table 1

Folder: `base_selection_table1/`

This is the Table-1-style base-selection cardinality result for PostgreSQL and
DuckDB. All reported base selections have nonzero actual cardinality.

| estimator | JOB-Light median | JOB median | JOB-Complex median |
|---|---:|---:|---:|
| PostgreSQL | 1.005 | 1.000 | 1.000 |
| DuckDB | 4.127 | 2.289 | 2.119 |

Key files:

- `paper_current_manifest.json`
- `base_selection_table1/README.md`
- `base_selection_table1/table1_base_selection_summary.json`
- `base_selection_table1/base_selection_qerrors.csv`

## Cost Estimation

### PostgreSQL Selected-Plan Cost

Folder: `cost_postgres_selected_plan/`

`scaled_q_*` maps PostgreSQL planner cost units to runtime with one log-mean
multiplicative scale per workload.

| workload | median Q-error | p95 Q-error |
|---|---:|---:|
| JOB-Light | 1.50 | 3.23 |
| JOB | 2.40 | 63.18 |
| JOB-Complex | 2.06 | 10.11 |

Key files:

- `cost_postgres_selected_plan/postgres_cost_summary.md`
- `cost_postgres_selected_plan/postgres_cost_summary.csv`
- `cost_postgres_selected_plan/postgres_cost_per_query.csv`

### Pretrained ZeroShot Cost

Folder: `cost_zeroshot_pretrained/`

Pooled three-seed pretrained ZeroShot result.

| workload | median Q-error | p95 Q-error |
|---|---:|---:|
| JOB-Light | 2.14 | 5.64 |
| JOB | 3.18 | 48.57 |
| JOB-Complex | 2.67 | 48.48 |

Key files:

- `cost_zeroshot_pretrained/summary_qerrors.md`
- `cost_zeroshot_pretrained/summary_qerrors.csv`

### MSCN Selected-Plan Cost

Folder: `cost_mscn_selected_plan/`

Completed MSCN-style selected-plan runtime estimator trained on 10K Mirage
synthetic SQL queries and evaluated on JOB-Light, JOB, and JOB-Complex.

| workload | median Q-error | p95 Q-error |
|---|---:|---:|
| JOB-Light | 1.65 | 6.97 |
| JOB | 1.85 | 36.93 |
| JOB-Complex | 1.53 | 4.31 |

Key files:

- `cost_mscn_selected_plan/mscn_cost_aggregate_summary.csv`
- `cost_mscn_selected_plan/mscn_cost_seed_metrics.csv`
- `cost_mscn_selected_plan/selected_plan_mscn_cost_report.json`

## Paper Plots

### Signed Q-Error Plots

Folder: `signed_qerror_plots/`

Signed log-scale Q-error plots with overestimates above zero and
underestimates below zero. The `paper_jobstyle_*` figures are the preferred
paper-facing versions because they follow the original JOB paper's vertical
boxplot style. The `paper_clean_*` horizontal interval plots and older
`signed_*` point-cloud plots are kept as alternatives/diagnostics.

Key files:

- `signed_qerror_plots/paper_jobstyle_base_selection_cardinality_qerror.svg`
- `signed_qerror_plots/paper_jobstyle_mscn_full_query_cardinality_qerror.svg`
- `signed_qerror_plots/paper_jobstyle_selected_plan_cost_runtime_qerror.svg`
- `signed_qerror_plots/paper_clean_signed_qerror_interval_summary.csv`
- `signed_qerror_plots/signed_qerror_plot_points.csv`
- `signed_qerror_plots/signed_qerror_plot_summary.csv`

## Raw Data For Paper Plots

Folder: `paper_requested_raw_data/`

Normalized per-query/per-selection CSVs for external plotting. These are the
paper-GPT handoff files: actual versus estimated cardinalities, selected-plan
cost/runtime predictions, result sizes, query metadata, runtime breakdowns, and
basic PostgreSQL plan metadata.

Key files:

- `paper_requested_raw_data/full_query_cardinality.csv`
- `paper_requested_raw_data/selected_plan_cost.csv`
- `paper_requested_raw_data/base_selection_cardinality.csv`
- `paper_requested_raw_data/query_result_sizes.csv`
- `paper_requested_raw_data/query_metadata.csv`
- `paper_requested_raw_data/runtime_breakdowns.csv`
- `paper_requested_raw_data/plan_metadata.csv`
- `paper_requested_raw_data/raw_data_manifest.json`

## Excluded From Paper-Current Results

- Custom-trained ZeroShot cardinality runs are legacy exploratory artifacts, not
  paper-current results. The only current ZeroShot entry is the pretrained
  lcm-eval selected-plan cost model in `cost_zeroshot_pretrained/`.
- PostgreSQL alternative-plan / candidate-plan cost experiments are archived in
  `../_legacy_not_for_paper/` and should not be mixed with the selected-plan
  cost table.
- The superseded base-selection v1 run is archived in
  `../_legacy_not_for_paper/`; use `base_selection_table1/`, which is copied
  from the corrected v2 run.
