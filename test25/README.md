# Test25 Manual

`test25` is the minimal standalone folder for running the fresh movie-generation pipeline, including a local-Ollama path for large runs.

## What Is In Here

- [run_pipeline.py](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/run_pipeline.py)
  - Main end-to-end runner.
- [requirements.txt](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/requirements.txt)
  - Core Python dependencies for the pipeline.
- [entities/genres.json](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/entities/genres.json)
  - Shared genre and subgenre taxonomy.
- [local_ollama_qwen35](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/local_ollama_qwen35)
  - Linux and Windows local-Ollama launch scripts for the `200k` run.

## Fresh-Run Pipeline Steps

The runner currently does:

1. procedural persons
2. procedural companies
3. procedural keywords
4. LLM person enrichment
5. procedural character bank
6. procedural title bank with year-span distribution
7. LLM person/company latents
8. graph build
9. entity CSV conversion
10. company financial profile synthesis
11. LLM keyword genre enrichment
12. movie generation
13. movie plot summaries
14. TV summaries

For serious large runs, step `100` is the practical stop point. The TV/post-summary steps are optional.

## Python Setup

Install the core Python dependencies:

```bash
pip install -r test25/requirements.txt
```

Current core packages:
- `numpy`
- `pandas`
- `pyarrow`
- `polars`
- `python-dotenv`
- `requests`
- `google-genai`
- `psutil`
- `pympler`

## LLM Modes

Two intended modes:

- API mode
  - requires an external `.env` or exported environment variables
- local mode
  - uses Ollama through the scripts in [local_ollama_qwen35](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/local_ollama_qwen35)
  - no API key required

For local mode, the scripts export:
- `LLM_PROVIDER=local`
- `LOCAL_LLM_URL=http://127.0.0.1:11434/v1`
- `LOCAL_LLM_MODEL=qwen35-35b-a3b`

## Small Smoke Run

Example:

```bash
python test25/run_pipeline.py \
  --fresh \
  --n-movies 100 \
  --start-year 2024 \
  --end-year 2025
```

Notes:
- this is a real end-to-end run
- if a step fails, resume with `--from-step`
- checkpoint file: [`.pipeline_checkpoint.json`](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/.pipeline_checkpoint.json)

Useful resume examples:

```bash
python test25/run_pipeline.py --n-movies 100 --start-year 2024 --end-year 2025 --from-step 70
```

```bash
python test25/run_pipeline.py --n-movies 100 --start-year 2024 --end-year 2025 --only 80 --force
```

## Local Ollama Setup

Folder:
- [local_ollama_qwen35](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/local_ollama_qwen35)

Main scripts:
- [install_all.sh](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/local_ollama_qwen35/install_all.sh)
- [start_qwen35_ollama.sh](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/local_ollama_qwen35/start_qwen35_ollama.sh)
- [run_pipeline_200k_local.sh](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/local_ollama_qwen35/run_pipeline_200k_local.sh)
- [install_all_windows.ps1](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/local_ollama_qwen35/install_all_windows.ps1)
- [start_qwen35_ollama_windows.ps1](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/local_ollama_qwen35/start_qwen35_ollama_windows.ps1)
- [run_pipeline_200k_local_windows.ps1](/C:/Users/vanya/Documents/DATA_SYS_LAB/test25/local_ollama_qwen35/run_pipeline_200k_local_windows.ps1)

Linux install and run:

```bash
chmod +x test25/local_ollama_qwen35/*.sh
./test25/local_ollama_qwen35/install_all.sh
./test25/local_ollama_qwen35/run_pipeline_200k_local.sh
```

Windows install and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\test25\local_ollama_qwen35\install_all_windows.ps1
.\test25\local_ollama_qwen35\run_pipeline_200k_local_windows.ps1
```

This is the intended `200k` entrypoint for the shared `test25` folder.

Behavior:
- installs Ollama
- downloads `Qwen3.5-35B-A3B-Q4_K_M.gguf`
- imports the model into Ollama
- checks for a GPU with at least `24 GB` VRAM
- uses GPU if eligible, otherwise forces CPU
- runs the `200k` pipeline through step `100`

The underlying pipeline command is:

```bash
python test25/run_pipeline.py \
  --fresh \
  --n-movies 200000 \
  --start-year 1950 \
  --end-year 2025 \
  --n-titles 200000 \
  --n-persons 450000 \
  --n-companies 30000 \
  --n-keywords 38000 \
  --n-characters 786000 \
  --enable-llm-evolution \
  --until-step 100
```

## Outputs

Main outputs are written inside `test25` itself:
- Arrow tables such as `movie.arrow`, `movies_flat.arrow`, `movies_analysis.arrow`
- graph outputs under `graph/`
- entity files under `entities/`

The runner is intended to stay inside `test25` by default.

## Notes

- `test25` is intentionally minimal. It is not a full utility dump.
- Some scripts still read a parent `.env` if you use API mode.
- For local-Ollama mode, the launcher scripts set the needed environment directly.
