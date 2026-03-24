#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

require_linux
detect_accelerator
export_runtime_env
write_runtime_env
print_accelerator_summary

ensure_ollama_server
ensure_model_file
ensure_ollama_model
warm_model
show_runtime_status

msg "Starting test25 200k pipeline with local Ollama model..."
run_pipeline_200k "$@"
