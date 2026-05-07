# Mirage Test44 Lab Handoff

`test44` is the current local code and benchmark handoff folder.

## Local Ollama Setup

The project includes a local Ollama installer bundle:

```bash
./local_ollama_qwen35/install_all.sh
```

The folder name is historical. The current default model family is Qwen3.6.

By default, the installer auto-selects the strongest built-in profile that fits
the detected hardware. For a large lab GPU this should choose a higher-quality
Qwen3.6 quant, such as `qwen36_35b_a3b_q8_0` or
`qwen36_35b_a3b_ud_q6_k_xl`, rather than forcing the smaller practical profile.

To force the faster practical profile explicitly:

```bash
LOCAL_OLLAMA_PROFILE_ID=qwen36_35b_a3b_mxfp4_moe ./local_ollama_qwen35/install_all.sh
```

To force the highest-quality built-in Qwen3.6 profile:

```bash
LOCAL_OLLAMA_PROFILE_ID=qwen36_35b_a3b_q8_0 ./local_ollama_qwen35/install_all.sh
```

## Main Run Entrypoints

Use the higher-level lab wrappers unless debugging the installer directly:

```bash
./lab_smoke_100.sh start
./lab_20k.sh start
./lab_50k.sh start
./lab_100k.sh start
./lab_200k.sh start
```

Continue an interrupted run:

```bash
./lab_20k.sh continue
```

Use a custom profile/run directory:

```bash
LAB_PROFILE=candidate100k LAB_RUN_DIR=../lab_candidate_100k ./lab.sh start
```

## Local Endpoint Override

If an OpenAI-compatible local server is already running:

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_URL=http://127.0.0.1:11434/v1
export LOCAL_LLM_MODEL=<model-name>
export LOCAL_LLM_API_KEY=not-needed
./run_local_ollama_test.sh check-provider
```

## Notes

- `clone_code_only_run.sh` copies this code into a clean run directory while
  excluding generated datasets, checkpoints, logs, and large exports.
- `final_benchmarks/` contains the current paper-facing benchmark outputs and
  raw CSV handoff bundle.
- Do not commit or copy a `.env` file into a shared repository or archive.
