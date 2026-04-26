# Current Lab Status

Generated on 2026-04-24 for the lab-facing local-LLM handoff.

## What Is Verified In `test40`

- `candidate20k_reuse_test23` structural preflight passes.
- Person career stages/timelines are now repaired and validated before graph generation.
- Step 80 graph-only validation passes on the 20k reuse-seed setup.
- Step 80 timing improved from `1777.6s` to `195.7s` after skipping precomputed P-C/C-C diagnostic cold edges.
- The retained runtime graph has about `605k` person-person edges and still uses on-demand person-company/company-company scoring during assembly.

## Important Defaults

- `run_lab_local_llm.sh` defaults to `SKIP_DIAGNOSTIC_COLD_EDGES=1`.
- Candidate profiles set `FAST_TITLE_TAGLINES=1` to avoid spending step-100 CPU on expensive grammar taglines; this preserves title uniqueness/metadata completeness but defers archival tagline polish.
- Set `SKIP_DIAGNOSTIC_COLD_EDGES=0` only if you deliberately want huge diagnostic cold-edge artifacts.
- `candidate20k_reuse_test23` is the older deterministic laptop/debug profile.
- `candidate20k_reuse_test23_gemini` is the active laptop/API profile and has `STEP100_DISABLE_LLM=0`.
- For a real lab run, use `candidate20k`, `candidate50k`, or larger profiles with `STEP100_DISABLE_LLM=0` and an explicit `LOCAL_LLM_MODEL`.

## Recommended Lab Ladder

```bash
./setup_linux_env.sh
export LLM_PROVIDER=local
export LOCAL_LLM_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_MODEL="<model-served-by-the-lab>"
./run_lab_local_llm.sh profile-list
./run_lab_local_llm.sh show-config
./run_lab_local_llm.sh check-provider
RUN_PROFILE=tiny10_1y ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-full100
RUN_PROFILE=small100_2y ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-full100
RUN_PROFILE=candidate20k ./run_lab_local_llm.sh preflight
RUN_PROFILE=candidate20k ./run_lab_local_llm.sh profile-graph80
RUN_PROFILE=candidate20k ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-full100
```

The cleaner handoff wrappers create fresh code-only run directories and then
execute the same canonical runner:

```bash
./run_lab_100_fresh.sh ../lab_smoke_100
./run_lab_1000_fresh.sh ../lab_check_1000
./run_lab_20k_fresh.sh ../lab_candidate_20k
./run_lab_50k_fresh.sh ../lab_candidate_50k
./run_lab_100k_fresh.sh ../lab_candidate_100k
./run_lab_200k_fresh.sh ../lab_candidate_200k
```

Windows Ollama status on this laptop: Windows itself can see `qwen3.5:4b`, but
WSL cannot reach the server until Ollama is restarted with a WSL-visible host
binding. Use:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_windows_ollama_for_wsl.ps1 -Model qwen3.5:4b -ForceRestart
```

After the restart, WSL reaches Ollama through the route-table gateway, currently
`http://172.22.192.1:11434/v1`; `detect_ollama_endpoint.sh` now discovers this.
The provider check passes for both installed models. The true research
from-scratch smoke still fails at step 4 on this laptop because the installed
4B/9B models cannot satisfy the strict `modeling_priors` artifact schema. Use
`run_lab_10_plumbing.sh` for laptop plumbing validation, and use a stronger lab
model for `run_lab_10_fresh.sh` / `run_lab_100_fresh.sh`.

Use resume only after a step-100 run has created `_step100_resume/`:

```bash
RUN_PROFILE=candidate20k ./run_lab_local_llm.sh resume-profile-step100
```

For normal interrupted lab runs, prefer the broader continuation target:

```bash
cd ../lab_candidate_20k
./run_lab_local_llm.sh profile-continue
```

The runner now persists `reports/active_local_provider.env` for local-provider
runs and reloads it before profile resolution on continue/resume targets. This
prevents a stale shell environment from accidentally switching a continued lab
run back to Gemini/API or to the wrong model alias.

Inspect an in-progress step-100 run from committed resume shards:

```bash
./run_lab_local_llm.sh resume-progress
```

For the current `test40` laptop folder, use the borrowed venv explicitly:

```bash
PYTHON_BIN=../test39_local_smoke/.venv-linux/bin/python RUN_PROFILE=candidate20k_reuse_test23_gemini ./run_lab_local_llm.sh resume-progress
```

## API Provider Status

The active fallback path is Gemini API with `gemini-3.1-flash-lite-preview`.

## What To Send To The Lab

Send a fresh code copy based on this directory, not old `test27` scripts. The lab should run the local LLM endpoint first, then `check-provider`, then the tiny/small ladder before a `20k` candidate. Step 110/120 summary enrichment can wait until step 100 and benchmark/export checks are structurally healthy.

The Ollama profile ladder now prioritizes `unsloth/Qwen3.6-35B-A3B-GGUF` over
the old Qwen3.5 122B profiles. For a 96 GB GPU, the expected first choice is
`qwen36_35b_a3b_q8_0`; the faster practical fallback is
`qwen36_35b_a3b_mxfp4_moe`.
