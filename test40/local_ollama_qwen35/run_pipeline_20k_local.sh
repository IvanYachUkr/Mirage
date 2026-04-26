#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

require_linux
ensure_apt_packages
ensure_python_venv
ensure_ollama_installed
detect_accelerator
export_runtime_env
write_runtime_env
print_accelerator_summary
warn_if_cpu_only_memory_is_tight

ensure_ollama_server
ensure_model_file
msg "About to import/register the Ollama model alias if needed..."
ensure_ollama_model
msg "Finished model import/register step."
msg "About to warm the Ollama model..."
warm_model
msg "Finished warmup step."
show_runtime_status

if [[ "${1:-}" == "continue" || "${1:-}" == "resume" ]]; then
  shift
  msg "Continuing 20k pipeline with local Ollama model..."
  run_pipeline_continue_20k "$@"
else
  msg "Starting fresh 20k pipeline with local Ollama model..."
  run_pipeline_20k "$@"
fi
