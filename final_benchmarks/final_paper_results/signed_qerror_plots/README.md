# Signed Q-Error Plots

This folder contains paper-facing signed Q-error plots for the final Test42
100K benchmark instance.

Signed Q-error is plotted as `sign * log10(Q-error)`, where positive values are
overestimates, negative values are underestimates, and zero is exact. Each plot
uses boxplots with overlaid individual query points and an orange diamond for
the mean.

Plots:

- `signed_base_selection_cardinality_qerror.svg/.png`: PostgreSQL and DuckDB
  base-selection cardinality Q-errors.
- `signed_mscn_full_query_cardinality_qerror.svg/.png`: MSCN full-query
  cardinality Q-errors, pooled across the three trained seeds.
- `signed_selected_plan_cost_runtime_qerror.svg/.png`: selected-plan
  cost/runtime Q-errors for PostgreSQL, pretrained ZeroShot, and MSCN.

Data files:

- `signed_qerror_plot_points.csv`: one row per plotted estimate.
- `signed_qerror_plot_summary.csv`: compact summary by plot/model/workload.

## Clean Horizontal Interval Figures

The `paper_clean_*` figures are alternative compact versions. They avoid
the diagnostic point cloud and instead show one horizontal interval per
model/workload:

- thin line: 5th-95th percentile signed log10 Q-error;
- thick line: interquartile range;
- hollow marker: median.

Horizontal files:

- `paper_clean_base_selection_cardinality_qerror.svg/.png`
- `paper_clean_mscn_full_query_cardinality_qerror.svg/.png`
- `paper_clean_selected_plan_cost_runtime_qerror.svg/.png`
- `paper_clean_signed_qerror_interval_summary.csv`

## Preferred Vertical JOB-Style Figures

The `paper_jobstyle_*` figures follow the visual style of the original JOB
paper more closely: vertical signed log-scale boxplots, panels by estimator,
and colors by workload. Boxes show the IQR, whiskers show the 5th-95th
percentile, and only a small deterministic sample of tail points is drawn.

Preferred files:

- `paper_jobstyle_base_selection_cardinality_qerror.svg/.png`
- `paper_jobstyle_mscn_full_query_cardinality_qerror.svg/.png`
- `paper_jobstyle_selected_plan_cost_runtime_qerror.svg/.png`

