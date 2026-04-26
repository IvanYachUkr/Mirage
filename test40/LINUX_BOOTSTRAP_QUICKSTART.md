# Test30 Linux Bootstrap Quickstart

This is the short operational companion to [WSL_LINUX_HANDOFF.md](WSL_LINUX_HANDOFF.md).

Use this when bringing `test30` up on WSL/Linux with a local Ollama-backed model.

## 1. Read First

Before doing anything else, read:
- [WSL_LINUX_HANDOFF.md](WSL_LINUX_HANDOFF.md)

Do not trust these files as the current source of truth:
- [README.md](README.md)
- [local_ollama_qwen35/README.md](local_ollama_qwen35/README.md)

They still describe `test27`.

## 2. New Linux Starter Script

Use:
- [local_ollama_qwen35/run_test30_local_linux.sh](local_ollama_qwen35/run_test30_local_linux.sh)

After cloning or copying onto Linux, make it executable:

```bash
chmod +x test30/local_ollama_qwen35/run_test30_local_linux.sh
```

## 3. Supported Targets

The script supports these targets:
- `smoke30`
- `smoke60`
- `smoke98`
- `dense100`
- `historical100`
- `scale20k98`

Examples:

```bash
./test30/local_ollama_qwen35/run_test30_local_linux.sh smoke30
./test30/local_ollama_qwen35/run_test30_local_linux.sh smoke98
./test30/local_ollama_qwen35/run_test30_local_linux.sh dense100
./test30/local_ollama_qwen35/run_test30_local_linux.sh historical100
./test30/local_ollama_qwen35/run_test30_local_linux.sh scale20k98
```

You can append extra `run_pipeline.py` args after the target:

```bash
./test30/local_ollama_qwen35/run_test30_local_linux.sh dense100 --disable-llm-critic
```

## 4. Recommended Validation Order

Use this order:
1. `smoke30`
2. `smoke60`
3. `smoke98`
4. `dense100`
5. `historical100`
6. only then consider `scale20k98`

This order is intentional:
- early artifact/provider issues fail fast by step `30`,
- title/keyword/artifact issues are mostly exposed by step `60`,
- graph and high-level planning issues are mostly exposed by step `98`,
- the expensive text stages are saved until after the setup is trusted.

## 5. Local Provider Notes

The starter script uses the existing Ollama/bootstrap machinery in:
- [local_ollama_qwen35/common.sh](local_ollama_qwen35/common.sh)

The actual pipeline provider abstraction remains in:
- [llm_provider.py](llm_provider.py)

Important environment variables if you later move away from Ollama to another local OpenAI-compatible server:
- `LLM_PROVIDER`
- `LOCAL_LLM_URL`
- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_API_KEY`

If you switch from Ollama to a different server stack, the next agent should likely keep this script as a template but replace the Ollama bootstrap section.

## 6. What This Script Is For

This script is meant to:
- give the next agent a working Linux-first starting point,
- reduce manual setup friction,
- and keep the first Linux experiments aligned with the validated Windows workflow.

It is not meant to be the final lab-scale automation layer yet.
