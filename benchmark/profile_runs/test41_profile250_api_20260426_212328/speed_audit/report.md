# Step 100 Speed Audit

- Experiment: `profile250_api_step100_resume`
- Updated: `2026-04-26T19:24:58+00:00`
- Wall seconds: `90.290602`
- Accounted component seconds: `167.39859`

## Top Components

| Component | Calls | Total s | Avg s | Max s |
|---|---:|---:|---:|---:|
| `main.assemble_movies` | 1 | 84.305732 | 84.305732 | 84.305732 |
| `year.plan_llm` | 5 | 24.021129 | 4.804226 | 4.917604 |
| `setup.sample_movie_concepts` | 1 | 13.369353 | 13.369353 | 13.369353 |
| `movie.compute_financials` | 250 | 9.545093 | 0.03818 | 0.179508 |
| `movie.pick_cast` | 250 | 6.957716 | 0.027831 | 0.176868 |
| `main.post_generation_critic` | 1 | 4.42356 | 4.42356 | 4.42356 |
| `movie.pick_keywords` | 250 | 3.472494 | 0.01389 | 0.053297 |
| `movie.pick_crew` | 250 | 2.916195 | 0.011665 | 0.08645 |
| `movie.pick_director` | 250 | 2.598603 | 0.010394 | 0.067385 |
| `year.program_compile` | 5 | 2.287702 | 0.45754 | 0.521329 |
| `movie.pick_companies` | 250 | 1.945031 | 0.00778 | 0.042922 |
| `secondary.release_dates` | 250 | 1.541389 | 0.006166 | 0.015012 |
| `main.world_load` | 1 | 1.117867 | 1.117867 | 1.117867 |
| `year.program_targeted` | 5 | 1.079753 | 0.215951 | 0.240931 |
| `secondary.box_office_daily` | 250 | 0.85902 | 0.003436 | 0.010162 |

## Slowest Samples

| Component | Seconds | Metadata |
|---|---:|---|
| `main.assemble_movies` | 84.305732 | `{"category": "main", "metadata": null, "note": "", "units": 250}` |
| `setup.sample_movie_concepts` | 13.369353 | `{"category": "setup", "metadata": null, "note": "", "units": 250}` |
| `year.plan_llm` | 4.917604 | `{"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2024, "model": "gemini-3.1-flash-lite-preview", "to_year": 2025}, "note": "", "units": 1}` |
| `year.plan_llm` | 4.862188 | `{"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2023, "model": "gemini-3.1-flash-lite-preview", "to_year": 2024}, "note": "", "units": 1}` |
| `year.plan_llm` | 4.851222 | `{"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2020, "model": "gemini-3.1-flash-lite-preview", "to_year": 2021}, "note": "", "units": 1}` |
| `year.plan_llm` | 4.826895 | `{"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2021, "model": "gemini-3.1-flash-lite-preview", "to_year": 2022}, "note": "", "units": 1}` |
| `year.plan_llm` | 4.56322 | `{"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2022, "model": "gemini-3.1-flash-lite-preview", "to_year": 2023}, "note": "", "units": 1}` |
| `main.post_generation_critic` | 4.42356 | `{"category": "main", "metadata": null, "note": "", "units": 0}` |
| `main.world_load` | 1.117867 | `{"category": "main", "metadata": null, "note": "", "units": 0}` |
| `post_loop.merge_preloop_outputs` | 0.82923 | `{"category": "post_loop", "metadata": null, "note": "", "units": 0}` |
| `global.tv_series_bundle` | 0.622541 | `{"category": "global_tables", "metadata": null, "note": "", "units": 150}` |
| `year.program_compile` | 0.521329 | `{"category": "year_boundary", "metadata": {"from_year": 2023, "planner_source": "llm", "to_year": 2024}, "note": "", "units": 11}` |
| `year.program_compile` | 0.469531 | `{"category": "year_boundary", "metadata": {"from_year": 2024, "planner_source": "llm", "to_year": 2025}, "note": "", "units": 11}` |
| `global.episode_cast` | 0.458724 | `{"category": "global_tables", "metadata": null, "note": "", "units": 17120}` |
| `year.program_compile` | 0.450904 | `{"category": "year_boundary", "metadata": {"from_year": 2020, "planner_source": "llm", "to_year": 2021}, "note": "", "units": 10}` |
