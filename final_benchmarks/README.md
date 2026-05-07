# Final Benchmark Workspace

This folder is for paper-facing benchmark artifacts built from the final
`imdb_test42_100k_exportpatch` PostgreSQL dataset.

The goal is to keep all reported numbers tied to one consistent export:

1. Export the final PostgreSQL IMDb core tables to CSV.
2. Build a fresh DuckDB database from those CSVs.
3. Run Table-1-style base-selection q-error extraction for PostgreSQL and
   DuckDB over JOB, JOB-Light, and JOB-Complex adapted queries.

Do not use older DuckDB files copied from VM backups for paper numbers; those
can predate later export patches.

Compact paper-facing result copies are collected under
`final_paper_results/`.

## Paper-Current Results

Use these folders for paper tables/plots:

- `final_paper_results/base_selection_table1/`
  - PostgreSQL and DuckDB Table-1-style base-selection cardinality Q-errors.
- `final_paper_results/mscn_cardinality/`
  - Three-seed MSCN cardinality Q-errors.
- `final_paper_results/cost_postgres_selected_plan/`
  - PostgreSQL selected-plan cost/runtime Q-errors.
- `final_paper_results/cost_zeroshot_pretrained/`
  - Pretrained lcm-eval ZeroShot selected-plan cost/runtime Q-errors.
- `final_paper_results/cost_mscn_selected_plan/`
  - MSCN selected-plan cost/runtime Q-errors.

## Paper Sources

These are supporting inputs/provenance for the current paper results:

- `test42_100k_exportpatch/`
  - Final 100K export, adapted JOB/JOB-Light/JOB-Complex queries, DuckDB file,
    and cost-model input traces.
- `runs/table1_base_selection_test42_100k_exportpatch_v2/`
  - Source run for the copied base-selection result. This is the corrected
    version with zero-actual selections fixed.
- `scripts/`
  - Scripts used to export the final PostgreSQL tables and rerun Table 1.

## Legacy

Artifacts that are preserved but should not be used directly in the paper are
under `_legacy_not_for_paper/`.
