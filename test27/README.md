# Test27

`test27` is the current standalone IMDb-generation pipeline folder.

It is designed to be cloneable on another machine and runnable from scratch with:
- procedural base generation
- LLM-enriched planning and movie assembly
- optional local-Ollama execution
- final JOB/IMDb schema export

## Included

- Core pipeline runners:
  - `run_pipeline.py`
  - `run_full_api_pipeline.py`
  - `run_full_api_pipeline.ps1`
  - `run_full_api_pipeline.cmd`
- Movie/world generation modules:
  - all `*.py` files in this folder
- IMDb export:
  - `export_imdb_schema.py`
  - `imdb_job_contract.py`
  - `validate_imdb_schema.py`
- Static taxonomy/runtime assets:
  - `entities/genres.json`
  - `local_ollama_qwen35/model_profiles.json`
- Local-Ollama helper scripts:
  - everything under `local_ollama_qwen35/` except generated runtime files

Generated outputs such as `.arrow`, planning JSONs, logs, checkpoints, and exported CSV bundles are intentionally excluded.

## Fresh Setup

Install Python dependencies from the repo root:

```bash
pip install -r test27/requirements.txt
```

API mode expects credentials in your environment or `.env`.

## Fresh Full Run

Example API smoke run from the repo root:

```bash
python test27/run_pipeline.py --fresh --n-movies 100
```

Default full helper run from the repo root:

```bash
python test27/run_full_api_pipeline.py
```

Equivalent commands from inside `test27/`:

```bash
python run_pipeline.py --fresh --n-movies 100
python run_full_api_pipeline.py
```

Prepared wrapper scripts:

```powershell
.\test27\run_full_api_pipeline.cmd
.\test27\run_full_api_pipeline.ps1
```

Those wrappers call `run_full_api_pipeline.py`.
By default they run:
- `200000` movies
- through step `130`
- with model `gemini-3.1-flash-lite-preview`

Smoke-size override:

```powershell
.\test27\run_full_api_pipeline.ps1 -Movies 100 -UntilStep 130 -Model gemini-3.1-flash-lite-preview
```

Example custom full run:

```powershell
.\test27\run_full_api_pipeline.ps1 -Movies 1000 -UntilStep 130 -Model gemini-3.1-flash-lite-preview
```

Important steps:
- `100` = movie generation
- `110` = plot summaries
- `120` = TV summaries
- `130` = strict JOB/IMDb export bundle

The export bundle is written to:

```text
test27/imdb_schema/
```

## Local Ollama

Local-model scripts live in:

```text
test27/local_ollama_qwen35/
```

They support profile-based selection, including larger multi-GPU Qwen profiles.

Useful command on Windows to inspect what a machine would choose before downloading anything:

```powershell
.\test27\local_ollama_qwen35\inspect_model_profiles_windows.ps1
```

## Notes

- This folder is meant to be source-only.
- Run artifacts are ignored by `test27/.gitignore`.
- If a run stops, resume with `--from-step`.
