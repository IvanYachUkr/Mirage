# Table 1 Base-Selection Q-Errors

Final paper-facing run for the `imdb_test42_100k_exportpatch` dataset.

Protocol:

- CSV export source: PostgreSQL database `imdb_test42_100k_exportpatch`
- DuckDB source: freshly rebuilt from the CSV export in this workspace
- Query source: copied final query set under
  `test44/final_benchmarks/test42_100k_exportpatch/queries`
- Duplicate string values inside `IN (...)` are canonicalized before estimation
  because they are semantically redundant adaptation artifacts.
- All reported rows have nonzero actual cardinality; this run has zero
  excluded rows for all workloads.

## PostgreSQL

| workload | base selections | median | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| JOB-Light | 157 | 1.005 | 1.048 | 1.071 | 10.000 | 10.000 |
| JOB | 606 | 1.000 | 5.000 | 19.469 | 91.000 | 138.099 |
| JOB-Complex | 186 | 1.000 | 9.773 | 36.173 | 88.715 | 196.875 |

## DuckDB

| workload | base selections | median | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| JOB-Light | 157 | 4.127 | 94.909 | 167.457 | 25,158.500 | 25,158.500 |
| JOB | 606 | 2.289 | 845.000 | 14,713.143 | 120,000.000 | 248,217.000 |
| JOB-Complex | 186 | 2.119 | 1,470.750 | 48,000.000 | 202,654.450 | 496,434.000 |

Raw rows: `base_selection_qerrors.csv`

Machine-readable summary: `table1_base_selection_summary.json`
