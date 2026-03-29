# Local Ollama Helpers

This folder contains the local-model launch scripts for `test27`.

## Main Scripts

- `install_all.sh`
- `install_all_windows.ps1`
- `start_qwen35_ollama.sh`
- `start_qwen35_ollama_windows.ps1`
- `run_pipeline_200k_local.sh`
- `run_pipeline_200k_local_windows.ps1`
- `inspect_model_profiles_windows.ps1`
- `common.sh`
- `common_windows.ps1`
- `model_profiles.json`

## What They Do

- install/start Ollama
- choose or override a Qwen GGUF profile
- expose the model through an OpenAI-compatible local endpoint
- launch `test27/run_pipeline.py` with the local provider settings

## Model Selection

Built-in profiles are stored in `model_profiles.json`.

The Windows helper can inspect available GPU VRAM and choose the highest profile that should fit, without downloading every model first.

Inspect only:

```powershell
.\test27\local_ollama_qwen35\inspect_model_profiles_windows.ps1
```

Force a profile:

```powershell
.\test27\local_ollama_qwen35\run_pipeline_200k_local_windows.ps1 -ProfileId 'qwen35_35b_a3b_q4_k_m'
```

## Notes

- Some very large profiles are sharded and may require `gguf-split` for merge/import.
- Generated runtime files and logs are intentionally ignored by `test27/.gitignore`.
