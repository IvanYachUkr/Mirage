#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LLM_PROVIDER="${LLM_PROVIDER:-local}"
if [[ -f "$BASE_DIR/reports/ollama_detected.env" && -z "${LOCAL_LLM_MODEL:-}" && -z "${LOCAL_LLM_URL:-}" ]]; then
  # shellcheck disable=SC1091
  source "$BASE_DIR/reports/ollama_detected.env"
fi
export LOCAL_LLM_URL="${LOCAL_LLM_URL:-http://127.0.0.1:11434/v1}"
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-${LOCAL_OLLAMA_MODEL_NAME:-${OLLAMA_MODEL:-}}}"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-not-needed}"
export RUN_PROFILE="${RUN_PROFILE:-tiny10_1y}"

target="${1:-check-provider}"
shift || true

exec "$BASE_DIR/run_lab_local_llm.sh" "$target" "$@"
