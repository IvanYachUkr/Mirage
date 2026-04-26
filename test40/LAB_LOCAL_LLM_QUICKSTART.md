# Lab Local-LLM Quickstart

> Current 2026-04-24 lab status and verified timings are summarized in [CURRENT_LAB_STATUS.md](CURRENT_LAB_STATUS.md).

This is the intended handoff path for running the generator on a Linux GPU
machine with a local LLM server.  The goal is to validate the structural movie
pipeline first; full movie/TV summary enrichment can wait until the benchmark
tables look healthy.

## 1. Python Environment

From the run directory:

```bash
./setup_linux_env.sh
```

The pipeline expects `pandas`, `pyarrow`, `numpy`, `requests`, and
`python-dotenv`.  `requirements.txt` contains the current minimal set.
The runner automatically uses `.venv-linux/bin/python` when it exists.

For local testing, avoid overwriting a real dataset directory:

```bash
./clone_code_only_run.sh ../test39_local_smoke
cd ../test39_local_smoke
./setup_linux_env.sh
```

## 2. Start A Local LLM Server

Any OpenAI-compatible server should work. Set `LOCAL_LLM_MODEL` explicitly; the runner intentionally does not guess a default model name. For vLLM, the shape is:

```bash
export LOCAL_LLM_MODEL="<model-served-by-the-lab>"
python -m vllm.entrypoints.openai.api_server \
  --model "$LOCAL_LLM_MODEL" \
  --host 0.0.0.0 \
  --port 8000
```

Then in the pipeline shell:

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_MODEL="<model-served-by-the-lab>"
export LOCAL_LLM_API_KEY="not-needed"
```

Ollama can also work if it exposes either `/v1/chat/completions` or its native
API.  Use `LOCAL_LLM_URL=http://127.0.0.1:11434/v1` for the OpenAI-compatible
path.

The bundled Ollama installer/profile ladder lives in `local_ollama_qwen35/`
(`qwen35` is a legacy folder name). The current preferred lab family is
`unsloth/Qwen3.6-35B-A3B-GGUF`, with `qwen36_35b_a3b_q8_0` selected first on
large single-GPU machines and `qwen36_35b_a3b_mxfp4_moe` as the practical
default/fallback. On a 96 GB GPU, start by trying:

```bash
LOCAL_OLLAMA_PROFILE_ID=qwen36_35b_a3b_q8_0 ./local_ollama_qwen35/install_all.sh
```

For a faster/lower-memory first attempt:

```bash
LOCAL_OLLAMA_PROFILE_ID=qwen36_35b_a3b_mxfp4_moe ./local_ollama_qwen35/install_all.sh
```

## 3. Smoke Test The Provider

If the local backend is Ollama on Windows, first start Ollama and then probe it
from WSL:

```bash
./detect_ollama_endpoint.sh
source reports/ollama_detected.env
```

If the detector says Windows Ollama is running but WSL cannot reach it, restart
Ollama from PowerShell with a WSL-visible host binding:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_windows_ollama_for_wsl.ps1 -Model qwen3.5:4b -ForceRestart
```

Then rerun:

```bash
./detect_ollama_endpoint.sh
source reports/ollama_detected.env
./run_lab_local_llm.sh check-provider
```

For the known Windows Ollama profile from the old local setup, this shortcut is
usually enough:

```bash
./run_local_ollama_test.sh check-provider
```

For vLLM or another OpenAI-compatible server:

```bash
./run_lab_local_llm.sh check-provider
```

This makes one JSON-mode request and writes `reports/local_llm_check.json`.

## 4. Profile Ladder

Profiles live in `local_run_profiles/` and can be listed with:

```bash
./run_lab_local_llm.sh profile-list
export LOCAL_LLM_MODEL="<model-served-by-the-lab>"
RUN_PROFILE=tiny10_1y ./run_lab_local_llm.sh show-config
```

Current profiles:

- `tiny10_1y`: 10 movies in one year; cheapest provider/pipeline smoke.
- `small100_2y`: 100 movies across two years; good local sanity test.
- `check1000_5y`: medium local structural shakeout.
- `candidate20k`: first paper-candidate scale.
- `candidate20k_reuse_test23`: deterministic laptop/debug reuse profile; not the final lab protocol.
- `candidate20k_reuse_test23_gemini`: laptop/API reuse profile with Gemini enabled; useful for this local 20k validation.
- `candidate50k`, `candidate100k`, `candidate200k`: lab-only larger candidates.

## 4.1 Fresh Run Convenience Scripts

For lab handoff, the simplest path is to create a fresh code-only run directory
and execute one of the scale wrappers from this source directory:

```bash
./run_lab_10_fresh.sh ../lab_smoke_10
./run_lab_100_fresh.sh ../lab_smoke_100
./run_lab_1000_fresh.sh ../lab_check_1000
./run_lab_20k_fresh.sh ../lab_candidate_20k
./run_lab_50k_fresh.sh ../lab_candidate_50k
./run_lab_100k_fresh.sh ../lab_candidate_100k
./run_lab_200k_fresh.sh ../lab_candidate_200k
```

The wrappers clone only code/configs into the destination, install the Linux
venv unless `SKIP_SETUP=1`, autodetect Ollama when possible, run
`check-provider`, and then call the canonical `run_lab_local_llm.sh` target.
They refuse non-empty destination folders unless `ALLOW_EXISTING_RUN_DIR=1` is
set deliberately.

The 10 and 100 movie wrappers run `profile-full100-export`, so they also produce
a sanity report and strict JOB export after step 100. Larger wrappers stop at
step 100 by default; run export after structural acceptance:

```bash
cd ../lab_candidate_20k
./run_lab_local_llm.sh sanity
./run_lab_local_llm.sh export-only
```

For very small laptop models that pass `check-provider` but cannot author the
full research control-plane artifacts, use the explicit plumbing smoke. This
still verifies the local LLM endpoint first, then runs the deterministic debug
pipeline through step 100/export without pretending the weak model produced
paper-grade artifacts:

```bash
./run_lab_10_plumbing.sh ../lab_plumbing_10
./run_lab_100_plumbing.sh ../lab_plumbing_100
```

On this laptop, `qwen3.5:4b` and `qwen35-9b-ud-q4-k-xl:latest` both reached
Ollama successfully, but failed the real from-scratch `modeling_priors` research
schema at step 4. Treat that as a model-capability limit, not a runner failure.

## 5. Small End-To-End Smoke

Use a disposable copy of the run directory for smoke tests, or set
`ALLOW_OVERWRITE=1` deliberately.

```bash
RUN_PROFILE=tiny10_1y ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-full100
RUN_PROFILE=small100_2y ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-full100
./run_lab_local_llm.sh sanity
```

Or use the one-command fresh wrappers:

```bash
./run_lab_10_fresh.sh ../lab_smoke_10
./run_lab_100_fresh.sh ../lab_smoke_100
```

This proves the local LLM can support the artifact/enrichment path through
movie generation without requiring a 20k run.

Candidate profiles also set `FAST_TITLE_TAGLINES=1`. This keeps benchmark-first step 100 focused on relational structure by generating cheap unique taglines when the title bank has blanks; full archival tagline polish can be rerun later if needed.

## 6. 20k Benchmark-First Candidate

For a full fresh local-LLM run, keep the default scale unless the lab hardware
suggests otherwise:

```bash
export N_MOVIES=20000
export N_PERSONS=64000
export N_COMPANIES=1400
export N_KEYWORDS=3200
export N_CHARACTERS=340000
export N_TITLES=26000
```

Before starting step 100, run:

```bash
RUN_PROFILE=candidate20k ./run_lab_local_llm.sh preflight
RUN_PROFILE=candidate20k ./run_lab_local_llm.sh profile-graph80
```

The preflight exists because the old 20k candidate had only about 20.5k
actor-role people for about 320k cast rows, making the target reuse gate
mathematically impossible.  The default lab profile uses a larger person pool
so the actor reuse gate is feasible before the expensive run starts.

Start or resume the benchmark-first movie generation path:

```bash
RUN_PROFILE=candidate20k ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-full100
RUN_PROFILE=candidate20k ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh profile-step100
RUN_PROFILE=candidate20k ./run_lab_local_llm.sh resume-profile-step100
```

`candidate20k-full100` is the clean lab path from a fresh directory.  It runs
through step 98, executes the structural preflight, then starts step 100.
`candidate20k-step100` is for a directory that already has valid entities,
latents, graph inputs, and generated artifacts.  Both paths intentionally stop
at step 100 and do not spend lab time on full movie/TV summary enrichment.

Continuation is explicit. The runner writes
`reports/active_local_provider.env` whenever a local provider target starts, so
an interrupted run can recover the same profile, endpoint, and model even if the
shell later contains stale API/Gemini variables:

```bash
cd ../lab_candidate_20k
./run_lab_local_llm.sh profile-continue
./run_lab_local_llm.sh profile-continue-export
```

`profile-continue` resumes from the normal pipeline checkpoint and automatically
adds `--resume-step100` when `_step100_resume/manifest.json` is incomplete. Use
`resume-profile-step100` only when you specifically want to restart step 100
from the last committed year boundary.

During a long step-100 run, inspect committed-year progress without touching the generator:

```bash
./run_lab_local_llm.sh resume-progress
```

After step 100:

```bash
./run_lab_local_llm.sh sanity
./run_lab_local_llm.sh export-only
```

## 7. What Must Be Rerun?

The recent structural fixes mostly affect step 100:

- Re-run step 100 if entities, latents, graph, concept packs, keyword motifs,
  title bank, and franchise bibles already exist and pass preflight.
- Re-run steps before 100 for a fresh local-LLM validation, because steps 4-98
  are exactly what prove the local model can produce the control-plane
  artifacts.
- Rebuild/regenerate entities before step 100 if preflight says the actor pool,
  title bank, or character bank is too small.
- Step 110 and step 120 are intentionally deferred until the structural movie
  dataset is accepted.

## 8. Optional Reuse Seed Path

For laptop debugging, not final reproducibility, you can seed from an older
entity folder:

```bash
export ENTITY_SOURCE_DIR="../test23"
export ARTIFACT_SOURCE_DIR="."
export EXTRA_TITLE_SOURCE_DIR="../test35"
export N_PERSONS=64000
./run_lab_local_llm.sh prepare-reuse-seed
```

This cleans generated outputs in the current directory and copies a selected
entity subset, so use it only in an intentionally disposable run directory.
The helper also regenerates the character bank and tops up the title bank to
the configured 20k profile before running the structural preflight.

## 9. Summary Steps Are Deferred

Step 110 and step 120 should be treated as optional after structural signoff.
For the lab meeting, the important proof is:

- local provider check passes
- small smoke reaches step 100
- 20k step 100 can run/resume
- sanity report is structurally acceptable
- strict JOB/IMDb export works
