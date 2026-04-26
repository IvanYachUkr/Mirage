# Mirage Lab Handoff

This branch is a source-only lab handoff for running Mirage with a local LLM. It excludes generated datasets, Arrow/CSV exports, DuckDB files, API keys, logs, and resume checkpoints.

## One-Command Lab Path

From the repository root:

```bash
cd test40
```

Run a fresh 100-movie smoke test:

```bash
./lab_smoke_100.sh start
```

Continue or resume that smoke test:

```bash
./lab_smoke_100.sh continue
```

Start real fresh lab candidates:

```bash
./lab_20k.sh start
./lab_50k.sh start
./lab_100k.sh start
./lab_200k.sh start
```

Continue/resume real lab candidates:

```bash
./lab_20k.sh continue
./lab_50k.sh continue
./lab_100k.sh continue
./lab_200k.sh continue
```

Each wrapper has exactly one optional argument: `start` or `continue`. With no argument, it defaults to `start`.

## Emergency Resume Commands

The wrappers below reload the matching `local_run_profiles/*.env` file, so the movie/entity counts are restored automatically. Use these if the server shuts down and you need an explicit restart point.

20k candidate: 20,000 movies, 64,000 persons, 1,400 companies, 3,200 keywords, 340,000 characters.

```bash
cd test40
./lab_20k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=70 UNTIL_STEP=100 ./lab_20k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=80 UNTIL_STEP=100 ./lab_20k.sh continue
LAB_CONTINUE_TARGET=resume-profile-step100 ./lab_20k.sh continue
```

50k candidate: 50,000 movies, 120,000 persons, 3,500 companies, 7,000 keywords, 850,000 characters.

```bash
cd test40
./lab_50k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=70 UNTIL_STEP=100 ./lab_50k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=80 UNTIL_STEP=100 ./lab_50k.sh continue
LAB_CONTINUE_TARGET=resume-profile-step100 ./lab_50k.sh continue
```

100k candidate: 100,000 movies, 240,000 persons, 7,000 companies, 12,000 keywords, 1,700,000 characters.

```bash
cd test40
./lab_100k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=70 UNTIL_STEP=100 ./lab_100k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=80 UNTIL_STEP=100 ./lab_100k.sh continue
LAB_CONTINUE_TARGET=resume-profile-step100 ./lab_100k.sh continue
```

200k candidate: 200,000 movies, 480,000 persons, 14,000 companies, 22,000 keywords, 3,400,000 characters.

```bash
cd test40
./lab_200k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=70 UNTIL_STEP=100 ./lab_200k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=80 UNTIL_STEP=100 ./lab_200k.sh continue
LAB_CONTINUE_TARGET=resume-profile-step100 ./lab_200k.sh continue
```

Meaning of the commands:

- `./lab_<scale>.sh continue` auto-detects the safest continuation path.
- `FROM_STEP=70` or `FROM_STEP=80` resumes the general pipeline from that step while preserving the profile counts.
- Replace `FROM_STEP=70` with another completed pipeline step, such as `FROM_STEP=7`, if the run stopped earlier.
- `resume-profile-step100` resumes movie generation from the last completed year-boundary checkpoint.
- If you are already inside a run directory such as `lab_candidate_20k`, the equivalent low-level command is `RUN_PROFILE=candidate20k FROM_STEP=70 UNTIL_STEP=100 ./run_lab_local_llm.sh profile-continue`.

## Local LLM

If the lab uses vLLM or another OpenAI-compatible server, start it first and export:

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_MODEL="<lab-model-name-or-path>"
export LOCAL_LLM_API_KEY="not-needed"
```

If the lab uses Ollama, the wrappers try to detect it automatically. To allow automatic Ollama model installation, run:

```bash
LAB_AUTO_INSTALL_OLLAMA=1 ./lab_smoke_100.sh start
```

The default Ollama install profile is `qwen36_35b_a3b_mxfp4_moe`; override it with `LOCAL_OLLAMA_PROFILE_ID=...`.

## Scale Profiles

Fresh-run entity counts live in `test40/local_run_profiles/`:

- `small100_2y.env`: 100 movies, 800 persons, 80 companies, 320 keywords, 1,800 characters.
- `candidate20k.env`: 20,000 movies, 64,000 persons, 1,400 companies, 3,200 keywords, 340,000 characters.
- `candidate50k.env`: 50,000 movies, 120,000 persons, 3,500 companies, 7,000 keywords, 850,000 characters.
- `candidate100k.env`: 100,000 movies, 240,000 persons, 7,000 companies, 12,000 keywords, 1,700,000 characters.
- `candidate200k.env`: 200,000 movies, 480,000 persons, 14,000 companies, 22,000 keywords, 3,400,000 characters.

These candidate profiles are fresh-from-scratch profiles, not legacy entity reuse profiles.

## Benchmark Tooling

- Exact JOB adapter/runner: `benchmark/job_exact_v1/`
- JOB-Complex adapter/runner: `benchmark/job_complex_v1/`
- PostgreSQL/DuckDB/MSCN/zero-shot CE tooling: `benchmark/cardinality_methods/`
- Canonical original JOB SQL: `imdb_job_dataset/job_queries/`

Generated benchmark outputs should go under local `runs/` folders and should not be committed.
