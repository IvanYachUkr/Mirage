# PostgreSQL Cost Summary

Correlation columns compare PostgreSQL root `Total Cost` with actual execution runtime.
`scaled_q_*` uses one log-mean multiplicative scale per workload to map planner cost units to milliseconds.

| workload | n | cost_runtime_pearson | cost_runtime_spearman | log_cost_runtime_pearson | median_cost_to_ms_scale | scaled_q_median | scaled_q_p90 | scaled_q_p95 | scaled_q_p99 | scaled_q_max | raw_cost_vs_ms_q_median | raw_cost_vs_ms_q_p95 | raw_cost_vs_ms_q_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| job_light | 70 | 0.933437 | 0.971786 | 0.966597 | 0.0115323 | 1.49621 | 2.99744 | 3.2284 | 3.95266 | 5.3546 | 70.4586 | 269.389 | 464.315 |
| job | 113 | 0.0894643 | 0.486064 | 0.359749 | 0.0284705 | 2.40097 | 16.7148 | 63.1841 | 185.976 | 997.888 | 53.2592 | 169.269 | 1015.58 |
| job_complex | 30 | 0.30102 | 0.681874 | 0.651654 | 0.0144842 | 2.05966 | 9.83904 | 10.1144 | 16.3197 | 18.7921 | 66.9585 | 682.374 | 708.82 |
