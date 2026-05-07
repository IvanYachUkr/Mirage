# Paper Requested Raw Data

Normalized raw CSVs for paper plotting and external GPT analysis.
These files contain per-query/per-selection values, not only summary percentiles.

## Files

- `full_query_cardinality.csv`: 852 data rows.
- `selected_plan_cost.csv`: 1491 data rows.
- `base_selection_cardinality.csv`: 1898 data rows.
- `query_result_sizes.csv`: 213 data rows.
- `query_metadata.csv`: 213 data rows.
- `runtime_breakdowns.csv`: 213 data rows.
- `plan_metadata.csv`: 213 data rows.

## Notes

- `full_query_cardinality.csv` contains PostgreSQL selected target-node cardinalities and MSCN per-seed full-query predictions.
- DuckDB is included for base-selection cardinality only; no DuckDB full-query join cardinality run was finalized.
- `selected_plan_cost.csv` contains selected-plan cost/runtime rows for PostgreSQL scaled planner cost, pretrained ZeroShot, and MSCN selected-plan runtime.
- Pretrained ZeroShot rows use the lcm-eval runtime/cost label scale from the downloaded artifacts.
- PostgreSQL `plan_metadata.csv` root rows are aggregate roots for `COUNT(*)`; `full_query_cardinality.csv` uses the target node/result size instead.
- Query metadata counts joins/predicates with a lightweight SQL parser intended for plotting, not as a formal SQL AST.
