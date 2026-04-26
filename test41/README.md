# Test41 Clean Lab Workspace

`test41` is a cleaned source-first workspace derived from `test40`. It keeps the current generator, local-LLM runners, step-100 resume support, and benchmark-oriented profiles, while dropping old generated exports and historical notes from earlier test folders.

## Recommended Lab Flow

Start from this directory:

```bash
cd test41
```

Run a fresh 100-movie smoke:

```bash
./lab_smoke_100.sh start
```

Continue or resume the smoke:

```bash
./lab_smoke_100.sh continue
```

Run fresh lab candidates. These wrapper commands run generation, sanity/runtime
reporting, and the strict IMDb/JOB export by default:

```bash
./lab_20k.sh start
./lab_50k.sh start
./lab_100k.sh start
./lab_200k.sh start
```

Resume lab candidates. If a resumed run reaches completion, these wrappers also
run the strict export by default:

```bash
./lab_20k.sh continue
./lab_50k.sh continue
./lab_100k.sh continue
./lab_200k.sh continue
```

## Local LLM

For vLLM or any OpenAI-compatible local server:

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_MODEL="<lab-model-name-or-path>"
export LOCAL_LLM_API_KEY="not-needed"
```

For Ollama, the scripts try to detect the endpoint automatically. To let the script install the bundled Ollama model profile:

```bash
LAB_AUTO_INSTALL_OLLAMA=1 ./lab_smoke_100.sh start
```

## Emergency Resume

The wrappers reload the correct profile counts automatically. If a server dies and you need an explicit restart point:

```bash
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=70 UNTIL_STEP=100 ./lab_20k.sh continue
LAB_CONTINUE_TARGET=profile-continue FROM_STEP=80 UNTIL_STEP=100 ./lab_20k.sh continue
LAB_CONTINUE_TARGET=resume-profile-step100 ./lab_20k.sh continue
```

Replace `lab_20k.sh` with `lab_50k.sh`, `lab_100k.sh`, or `lab_200k.sh` for larger runs. Replace `FROM_STEP=70` with another completed step if the run stopped earlier.

If export needs to be rerun manually after a completed candidate:

```bash
cd ../lab_candidate_20k
./run_lab_local_llm.sh export-only
```

## Fresh Candidate Profiles

- `candidate20k`: 20,000 movies, 64,000 persons, 1,400 companies, 3,200 keywords, 340,000 characters.
- `candidate50k`: 50,000 movies, 120,000 persons, 3,500 companies, 7,000 keywords, 850,000 characters.
- `candidate100k`: 100,000 movies, 240,000 persons, 7,000 companies, 12,000 keywords, 1,700,000 characters.
- `candidate200k`: 200,000 movies, 480,000 persons, 14,000 companies, 22,000 keywords, 3,400,000 characters.

These are fresh-from-scratch profiles, not legacy entity-reuse profiles.
