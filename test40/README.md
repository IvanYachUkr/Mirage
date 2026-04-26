# Test27

`test27` is the current standalone IMDb-generation pipeline folder.

It is designed to be runnable from scratch with:
- procedural base generation
- LLM-enriched planning and movie assembly
- optional local-Ollama execution
- final JOB/IMDb schema export

## Fresh Setup

Install Python dependencies from the repo root:

```bash
pip install -r test27/requirements.txt
```

API mode expects credentials in your environment or `.env`.

## Main Entry Points

From the repo root:

```bash
python test27/run_pipeline.py --fresh --n-movies 100
python test27/run_full_api_pipeline.py
```

From inside `test27/`:

```bash
python run_pipeline.py --fresh --n-movies 100
python run_full_api_pipeline.py
```

Prepared Windows wrappers:

```powershell
.\test27\run_full_api_pipeline.cmd
.\test27\run_full_api_pipeline.ps1
```

## Default Full Helper Behavior

`run_full_api_pipeline.py` is the main high-level helper. By default it runs:

- `200000` movies
- `200000` titles
- `450000` persons
- `30000` companies
- `38000` keywords
- `786000` characters
- year span `1950-2025`
- through step `130`
- model `gemini-3.1-flash-lite-preview`

Example `200k` run to movie generation only:

```bash
python run_full_api_pipeline.py --n-movies 200000 --until-step 100 --skip-compare
```

That command automatically resolves the matching entity counts above unless you explicitly override them.

## Important Steps

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

The prepared local `200k` scripts default to:

- `200000` movies
- `200000` titles
- `450000` persons
- `30000` companies
- `38000` keywords
- `786000` characters
- year span `1950-2025`
- through step `130`

One-command local bootstrap runners:

Windows:

```powershell
.\test27\local_ollama_qwen35\run_pipeline_200k_local_windows.ps1
```

Linux:

```bash
./test27/local_ollama_qwen35/run_pipeline_200k_local.sh
```

Those scripts are intended to:
- detect hardware and select a model profile automatically
- create a local Python environment from the system Python
- install local pipeline requirements
- install Ollama if missing
- download/import the selected GGUF model if missing
- export the local LLM environment variables
- and run the full local pipeline

Useful command on Windows to inspect what a machine would choose before downloading anything:

```powershell
.\test27\local_ollama_qwen35\inspect_model_profiles_windows.ps1
```

## Notes

- This folder is meant to be source-first; run artifacts are generated locally.
- If a run stops, resume with `--from-step`.
- If you only need the benchmarkable database, step `100` plus step `130` export is usually enough.
