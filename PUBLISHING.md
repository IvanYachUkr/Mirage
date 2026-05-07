# Publishing Notes

This workspace is prepared as a code/static distribution candidate.

## Include

- Python source files.
- Shell and PowerShell wrappers.
- Static JSON priors and reusable generated templates needed to bootstrap runs.
- `local_run_profiles/`.
- `README.md`, `CONTINUATION.md`, and this file.

## Exclude

- `.env` or any API key material.
- Generated entity folders, graph/runtime folders, `_step100_resume`, logs,
  reports, Arrow/CSV exports, DuckDB/Postgres exports, and benchmark outputs.
- Local model caches and machine-specific scratch folders.

The `.gitignore` in this folder is intentionally conservative. It ignores CSVs
and Arrow files because those are generated data products in this project.

## Before Publishing

1. Run `python -m py_compile *.py`.
2. Run a tiny smoke pipeline or at least the Step100 structural preflight.
3. Confirm no `.env` file exists in the publish folder.
4. Confirm generated folders such as `entities/`, `graph/`, and
   `_step100_resume/` are absent.
5. For a paper artifact release, include a separate generated dataset archive or
   a script that recreates one, not both mixed into this code workspace.

## Refactor Boundary

The largest files are still monolithic. That is acceptable for a deadline-driven
artifact if the public interface is documented and tested. The next safe cleanup
would be:

1. Extract Step100 resume/extend orchestration from `generate_movies.py`.
2. Extract title generation and fallback logic from `assembly.py`.
3. Split graph cache/selector code out of `graph_runtime.py`.
4. Keep compatibility wrappers so old run scripts still work.

Do this after the production 100K run and continuation evidence are secured.
