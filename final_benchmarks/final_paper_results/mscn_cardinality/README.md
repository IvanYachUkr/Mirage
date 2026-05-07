# Final MSCN Cardinality Bundle

Generated at: 2026-05-05T19:03:23
Source run: `C:\Users\vanya\Documents\DATA_SYS_LAB\benchmark\cardinality_methods\runs\test42_100k_paper_mscn_cardinality_100k_100ep_3seed_20260504_exact3600s_quotefix`

Recommended reporting policy: `median_of_seed_summaries`.

Reason: choosing the single best seed is optimistic and materially changes JOB/JOB-Complex tail results.

Files:
- `artifact_audit.json`: verifies training/eval artifacts are present.
- `mscn_aggregate_table_policy.csv`: paper table candidates and seed-selection deltas.
- `mscn_seed_metrics.csv`: full seed-level Q-error metrics.
- `mscn_seed_qerror_handlebars.svg`: per-seed Q-error distributions.
- `mscn_signed_qerror_bias_by_seed.svg`: over/under-estimation bias by seed.
