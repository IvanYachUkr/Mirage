# Step 100 Runtime Summary

- Generated: `2026-04-26T19:25:08+00:00`
- Base dir: `/tmp/mirage_profile250_api`
- Resume status: `complete`; last completed year: `2024`
- Progress events: `9515`; movie-complete events: `250`
- Movies/sec from log: `0.436088`

## Runtime Profile

- Config found: `True`
- Disabled secondary tables: `['reviews']`
- Disabled global tables: `[]`
- Disabled post-loop tables: `['user_ratings', 'production_timeline', 'streaming_windows', 'person_contracts', 'movie_sequence', 'person_collaborations', 'media_links', 'world_events']`

## Speed Audit

- Audit dir: `/tmp/mirage_profile250_api/reports/speed_audit/20260426_212328`
- Wall seconds: `90.288987`
- Accounted component seconds: `167.39859`

### Top Components

| component | calls | total_seconds | avg_seconds | max_seconds |
|---|---|---|---|---|
| main.assemble_movies | 1 | 84.305732 | 84.305732 | 84.305732 |
| year.plan_llm | 5 | 24.021129 | 4.804226 | 4.917604 |
| setup.sample_movie_concepts | 1 | 13.369353 | 13.369353 | 13.369353 |
| movie.compute_financials | 250 | 9.545093 | 0.03818 | 0.179508 |
| movie.pick_cast | 250 | 6.957716 | 0.027831 | 0.176868 |
| main.post_generation_critic | 1 | 4.42356 | 4.42356 | 4.42356 |
| movie.pick_keywords | 250 | 3.472494 | 0.01389 | 0.053297 |
| movie.pick_crew | 250 | 2.916195 | 0.011665 | 0.08645 |
| movie.pick_director | 250 | 2.598603 | 0.010394 | 0.067385 |
| year.program_compile | 5 | 2.287702 | 0.45754 | 0.521329 |
| movie.pick_companies | 250 | 1.945031 | 0.00778 | 0.042922 |
| secondary.release_dates | 250 | 1.541389 | 0.006166 | 0.015012 |
| main.world_load | 1 | 1.117867 | 1.117867 | 1.117867 |
| year.program_targeted | 5 | 1.079753 | 0.215951 | 0.240931 |
| secondary.box_office_daily | 250 | 0.85902 | 0.003436 | 0.010162 |

### Slow Samples

| component | elapsed_seconds | metadata_json |
|---|---|---|
| main.assemble_movies | 84.305732 | {"category": "main", "metadata": null, "note": "", "units": 250} |
| setup.sample_movie_concepts | 13.369353 | {"category": "setup", "metadata": null, "note": "", "units": 250} |
| year.plan_llm | 4.917604 | {"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2024, "model": "gemini-3.1-flash-lite-preview", "to_year": 2025}, "note": "", "units": 1} |
| year.plan_llm | 4.862188 | {"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2023, "model": "gemini-3.1-flash-lite-preview", "to_year": 2024}, "note": "", "units": 1} |
| year.plan_llm | 4.851222 | {"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2020, "model": "gemini-3.1-flash-lite-preview", "to_year": 2021}, "note": "", "units": 1} |
| year.plan_llm | 4.826895 | {"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2021, "model": "gemini-3.1-flash-lite-preview", "to_year": 2022}, "note": "", "units": 1} |
| year.plan_llm | 4.56322 | {"category": "year_boundary", "metadata": {"enable_llm": true, "from_year": 2022, "model": "gemini-3.1-flash-lite-preview", "to_year": 2023}, "note": "", "units": 1} |
| main.post_generation_critic | 4.42356 | {"category": "main", "metadata": null, "note": "", "units": 0} |
| main.world_load | 1.117867 | {"category": "main", "metadata": null, "note": "", "units": 0} |
| post_loop.merge_preloop_outputs | 0.82923 | {"category": "post_loop", "metadata": null, "note": "", "units": 0} |

## Quality Guardrails

- Sanity report: `/tmp/mirage_profile250_api/reports/step100_sanity_report.json`
- Failed gates: `['company_gini_ge_045', 'movie_count_20k', 'unique_cast_people_ge_14000', 'unique_directors_ge_1000']`
- movie_rows: `250`
- unique_titles: `250`
- duplicate_titles: `{}`
- duplicated_tagline_rate_pct: `0.0`
- cast_reuse_ratio: `2.288`
- unique_cast_people: `632`
- blank_character_description_rate_pct: `0.0`
- unique_directors: `110`
- company_gini: `0.4254`
- unique_companies: `113`
- keyword_zero_exact_topic_rate_pct: `0.0`
- awards_movie_share_pct: `12.0`
- awards_win_share_pct: `7.5`
