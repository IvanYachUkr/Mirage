# Step 100 Sanity Report

- Dataset: `/tmp/mirage_profile250_api`
- Generated: `2026-04-26T21:25:08`
- Summary policy: `benchmark_first_deferred`
- Overall: NEEDS WORK
- Failed gates: `movie_count_20k, unique_cast_people_ge_14000, unique_directors_ge_1000, company_gini_ge_045`

## Core Counts
- Movies: `250` rows, `250` unique titles
- Cast: `1446` rows, `632` unique people, reuse ratio `2.288`
- Directors: `110` unique, top director `[{'value': '966', 'count': 21}]`
- Companies: `113` unique, Gini `0.4254`
- Keywords: `2043` rows, zero exact-topic movies `0`
- Awards: `40` rows, movies with awards `12.0%`, win rate `7.5%`

## Notable Details
- Generic step-100 plot summaries: `243`
- Duplicate titles: `{}`
- Top actors: `[{'value': '1374', 'count': 27}, {'value': '256', 'count': 13}, {'value': '458', 'count': 12}, {'value': '1343', 'count': 11}, {'value': '1691', 'count': 9}]`
- Top repeated taglines: `[{'value': 'Witness the steel that defied the ages.', 'count': 1}, {'value': 'When cosmos forces collide, the world trembles.', 'count': 1}, {'value': 'A last portrait of a dust defined by memory.', 'count': 1}, {'value': 'Beyond the Kingdom lies a fable secret.', 'count': 1}, {'value': 'Experience the broken intensity of a focused vision.', 'count': 1}]`
- Zero exact-topic title ids: `[]`
