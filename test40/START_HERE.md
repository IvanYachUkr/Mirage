# Start Here

If you are coming back to `test30` after a break, read this first.

## Do Not Start With

- [README.md](README.md)
- [local_ollama_qwen35/README.md](local_ollama_qwen35/README.md)

They are stale and still describe `test27`.

## Start With

1. [WSL_LINUX_HANDOFF.md](WSL_LINUX_HANDOFF.md)
2. [LINUX_BOOTSTRAP_QUICKSTART.md](LINUX_BOOTSTRAP_QUICKSTART.md)

## Best Windows Evidence Runs

- Dense fresh sign-off candidate:
  - run id `20260411_212243`
  - [_dev/clean_run_analysis_20260411_212243.md](_dev/clean_run_analysis_20260411_212243.md)
- Historical validation:
  - run id `20260412_004507`
  - [_dev/clean_run_analysis_20260412_004507.md](_dev/clean_run_analysis_20260412_004507.md)

## Current Honest State

- `test30` is a research-grade hybrid pipeline.
- LLMs control semantic artifacts, priors, planning, and text generation.
- The code still controls mechanics, schema constraints, scaling, and some scoring defaults.
- The dense run is strong enough to support the main paper story.
- The next important step is Linux/WSL + local-LLM bring-up.

## Practical Next Move

If you are on Linux/WSL, use:

- [local_ollama_qwen35/run_test30_local_linux.sh](local_ollama_qwen35/run_test30_local_linux.sh)

Recommended order:
- `smoke30`
- `smoke60`
- `smoke98`
- `dense100`
- `historical100`

That is the shortest safe path back into the project.
