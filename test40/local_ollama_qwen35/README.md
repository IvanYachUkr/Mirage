# Local Ollama Lab Launcher

This folder contains the Linux/Windows helper scripts for running the pipeline
against a local Ollama model. The folder name is historical; the current model
ladder is no longer Qwen3.5-only.

## Main Entrypoints

Linux:

```bash
./local_ollama_qwen35/install_all.sh
./local_ollama_qwen35/run_pipeline_20k_local.sh
./local_ollama_qwen35/run_pipeline_200k_local.sh
```

Continue an interrupted run without `--fresh`:

```bash
./local_ollama_qwen35/run_pipeline_20k_local.sh continue
./local_ollama_qwen35/run_pipeline_200k_local.sh continue
```

Windows:

```powershell
.\local_ollama_qwen35\install_all_windows.ps1
.\local_ollama_qwen35\run_pipeline_200k_local_windows.ps1
```

The launchers:

- detect GPU hardware
- choose a model profile automatically unless overridden
- create a local Python virtual environment
- install pipeline requirements plus `huggingface_hub[cli]`
- install/start Ollama if needed
- download/import the selected GGUF
- export `LLM_PROVIDER=local`, `LOCAL_LLM_URL`, and `LOCAL_LLM_MODEL`
- launch the pipeline with the active Ollama alias passed to every model slot
  (`--model`, `--bootstrap-model`, `--planning-model`, and
  `--bulk-artifact-model`)
- add `--resume-step100` on continue when the step-100 resume manifest is not
  complete

## Recommended Model Order

The current first-choice family is `unsloth/Qwen3.6-35B-A3B-GGUF`, because it is
a recent 35B-total / 3B-active MoE model and should be a better lab default than
older, much larger Qwen3.5 122B profiles.

Built-in Qwen3.6 profiles:

- `qwen36_35b_a3b_q8_0`: preferred for 48-96 GB GPUs, highest quality built-in Qwen3.6 profile.
- `qwen36_35b_a3b_ud_q6_k_xl`: quality fallback when Q8 is too tight.
- `qwen36_35b_a3b_mxfp4_moe`: practical default and auto-disabled fallback.
- `qwen36_35b_a3b_ud_q4_k_m`: conservative 4-bit fallback.

Older Qwen3.5 profiles remain available as fallbacks, including the small
`qwen35_9b_ud_q4_k_xl` smoke profile and the previous 35B/122B options.

## Selection Rules

- Manual model overrides win if all of `LOCAL_OLLAMA_MODEL_REPO`,
  `LOCAL_OLLAMA_MODEL_FILE`, and `LOCAL_OLLAMA_MODEL_NAME` are set.
- `LOCAL_OLLAMA_PROFILE_ID` wins next and forces a built-in profile.
- Otherwise the launcher auto-selects the highest-priority profile that fits the
  detected GPU inventory.
- If auto-selection is disabled, the default profile is
  `qwen36_35b_a3b_mxfp4_moe`.

For a 96 GB GPU lab machine, the expected auto choice is usually:

```bash
qwen36_35b_a3b_q8_0
```

To force the faster practical profile:

```bash
LOCAL_OLLAMA_PROFILE_ID=qwen36_35b_a3b_mxfp4_moe \
./local_ollama_qwen35/run_pipeline_20k_local.sh
```

To force the high-quality Q8 profile:

```bash
LOCAL_OLLAMA_PROFILE_ID=qwen36_35b_a3b_q8_0 \
./local_ollama_qwen35/run_pipeline_20k_local.sh
```

## Inspect Before Download

Windows:

```powershell
.\local_ollama_qwen35\inspect_model_profiles_windows.ps1
.\local_ollama_qwen35\inspect_model_profiles_windows.ps1 -ProfileId 'qwen36_35b_a3b_q8_0'
```

Linux, without downloading:

```bash
LOCAL_OLLAMA_PROFILE_ID=qwen36_35b_a3b_q8_0 \
bash -lc 'source local_ollama_qwen35/common.sh; detect_accelerator; print_accelerator_summary'
```

## Caveats

- `published_size_gb` is a GGUF file-size estimate, not an exact runtime VRAM
  guarantee. Context length, KV cache, Ollama overhead, and GPU offload behavior
  still matter.
- The old 122B profiles are still present but no longer prioritized. Some are
  sharded GGUF downloads and require `gguf-split` before Ollama import.
- For laptop plumbing tests, use the project-level `run_lab_10_plumbing.sh`.
  For real research artifacts, use a stronger lab model and the fresh run
  wrappers.
