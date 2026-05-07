# Mirage

This is the source/static workspace for the Mirage generator, not a generated
dataset bundle.

Mirage generates scalable synthetic IMDb-like databases for cardinality
estimation experiments. The workflow supports both fresh generation and
continuation from a completed Mirage workspace.

## What Is Included

- Python source for entity generation, LLM artifact generation, graph/runtime
  construction, movie assembly, Step100 resume, and IMDb/JOB export.
- Static priors and reusable seed artifacts such as title grammar, keyword seed
  banks, world policy templates, and profile files.
- Shell/PowerShell wrappers for smoke, candidate, resume, and API runs.

Generated data is intentionally not included in this workspace. Runtime outputs
such as `entities/`, `graph/`, `_step100_resume/`, Arrow/CSV exports,
`imdb_schema/`, logs, reports, and caches are ignored by `.gitignore`.

## Main Entry Points

- `run_pipeline.py`: end-to-end pipeline runner.
- `generate_movies.py`: Step100 movie/table assembly.
- `prepare_continuation_entities.py`: entity top-up for continuation.
- `continuation_lifecycle.py`: balanced person/company carryover lifecycle.
- `export_imdb_schema.py`: strict IMDb/JOB CSV export.
- `build_duckdb_from_imdb_schema.py`: DuckDB builder for exported JOB CSVs.
- `run_job_queries_duckdb.py`: quick JOB query execution against DuckDB.

## Fresh Candidate Run

From this directory:

```bash
python run_pipeline.py --profile candidate100k_2000_2050 --fresh --force-scalable-graph --benchmark-mode --from-step 4 --until-step 100
```

The profile loader reads `local_run_profiles/<name>.env` and fills missing
cardinality/year/runtime settings while preserving explicit CLI overrides.

The shell wrappers are still available for local-lab runs:

```bash
./lab_20k.sh start
./lab_50k.sh start
./lab_100k.sh start
./lab_200k.sh start
```

Resume an interrupted candidate:

```bash
./lab_100k.sh continue
```

The `candidate100k_2000_2050` profile is available under
`local_run_profiles/candidate100k_2000_2050.env`.

## Continuation Run

Continuation requires a full completed Mirage workspace, including
`_step100_resume`, entities, graph/runtime artifacts, and generated static
artifacts. It does not currently support continuation from only SQL, DuckDB, or
Postgres exports.

Example cumulative 100K to 300K extension:

```bash
python run_pipeline.py \
  --n-movies 300000 \
  --start-year 2051 \
  --end-year 2100 \
  --n-persons 720000 \
  --n-companies 27000 \
  --n-keywords 36000 \
  --n-characters 5100000 \
  --n-titles 390000 \
  --mode research \
  --model gemini-3.1-flash-lite-preview \
  --bootstrap-model gemini-3.1-flash-lite-preview \
  --planning-model gemini-3.1-flash-lite-preview \
  --bulk-artifact-model gemini-3.1-flash-lite-preview \
  --force-scalable-graph \
  --skip-diagnostic-cold-edges \
  --benchmark-mode \
  --extend-step100 \
  --from-step 10 \
  --until-step 100
```

`--n-movies` is cumulative. In this example a completed 100K source workspace
receives 200K additional movies across 2051-2100.

## Local LLM Overrides

For any OpenAI-compatible local server:

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_MODEL="<model-name-or-path>"
export LOCAL_LLM_API_KEY="not-needed"
```

For long automated API runs, prefer `gemini-3.1-flash-lite-preview` unless a
short planning step explicitly needs a stronger model.

Model defaults are centralized in `model_defaults.py`. You can override a role
without editing source code, for example:

```bash
export MIRAGE_MODEL_ARTIFACT_BULK=gemini-3.1-flash-lite-preview
export MIRAGE_MODEL_ARTIFACT_PRO=gemini-3.1-pro-preview
```

## Large Files

Some source files are still large. They are intentionally left in place for now
because a broad refactor could risk changing generation behavior before the
paper deadline.

- `assembly.py`: movie assembly and research title fallback logic.
- `generate_movies.py`: Step100 orchestration and resume/extend entry point.
- `graph_runtime.py`: graph loading, caches, and runtime selectors.
- `secondary_tables.py`: derived IMDb/JOB-compatible secondary tables.
- `year_planner.py`: year slate and planning helpers.

For publication, the safer short-term approach is to document these boundaries
and keep behavior stable. A later cleanup can split them into packages after the
100K/300K evidence path is secure.
