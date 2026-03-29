#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

require_linux
msg "Installing local Ollama bundle..."

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
ensure_ollama_model
warm_model
show_runtime_status

msg "Install complete."
msg "Next command:"
msg "  ${SCRIPT_DIR}/run_pipeline_200k_local.sh"
