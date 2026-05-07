#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./lab_run_fresh_profile.sh <profile> <dest_dir> [target]

Examples:
  ./lab_run_fresh_profile.sh small100_2y ../lab_smoke_100 profile-full100-export
  ./lab_run_fresh_profile.sh candidate20k ../lab_candidate_20k profile-full100

Environment:
  LOCAL_LLM_MODEL      Model exposed by the local server, for example qwen36-35b-a3b-mxfp4-moe.
  LOCAL_LLM_URL        OpenAI-compatible base URL, for example http://127.0.0.1:11434/v1.
  LLM_PROVIDER         Defaults to local.
  SKIP_SETUP=1         Do not create/install .venv-linux in the destination.
  SKIP_PROVIDER_CHECK=1 Do not run the one-request local LLM JSON check.
  ALLOW_EXISTING_RUN_DIR=1 Permit using a non-empty destination directory.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -lt 2 ]]; then
  usage
  exit 2
fi

PROFILE="$1"
DEST_DIR="$2"
TARGET="${3:-profile-full100}"

target_needs_local_model() {
  case "$1" in
    plumbing-full100-export)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

case "$TARGET" in
  resume-*|*continue*)
    echo "Fresh-run helper refuses resume/continue targets. Use run_lab_local_llm.sh inside the run directory instead." >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$DEST_DIR")"
if [[ -e "$DEST_DIR" ]] && [[ -n "$(find "$DEST_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  if [[ "${ALLOW_EXISTING_RUN_DIR:-0}" != "1" ]]; then
    echo "Destination already exists and is not empty: $DEST_DIR" >&2
    echo "Choose a new run directory, or set ALLOW_EXISTING_RUN_DIR=1 deliberately." >&2
    exit 3
  fi
fi

"$SRC_DIR/clone_code_only_run.sh" "$DEST_DIR"
DEST_DIR="$(cd "$DEST_DIR" && pwd)"
cd "$DEST_DIR"

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  ./setup_linux_env.sh
fi

export RUN_PROFILE="$PROFILE"
export LLM_PROVIDER="${LLM_PROVIDER:-local}"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-not-needed}"
export LAB_ENABLE_SPEED_AUDIT="${LAB_ENABLE_SPEED_AUDIT:-1}"
export LAB_ENABLE_MEMORY_AUDIT="${LAB_ENABLE_MEMORY_AUDIT:-0}"

if [[ -z "${LOCAL_LLM_URL:-}" || -z "${LOCAL_LLM_MODEL:-}" ]]; then
  if ./detect_ollama_endpoint.sh; then
    # shellcheck disable=SC1091
    source "$DEST_DIR/reports/ollama_detected.env"
  else
    status=$?
    if [[ "$status" -eq 2 && -f "$DEST_DIR/reports/ollama_windows_seen.env" ]]; then
      # shellcheck disable=SC1091
      source "$DEST_DIR/reports/ollama_windows_seen.env"
    fi
  fi
fi

export LOCAL_LLM_URL="${LOCAL_LLM_URL:-http://127.0.0.1:11434/v1}"
if target_needs_local_model "$TARGET" && [[ -z "${LOCAL_LLM_MODEL:-}" ]]; then
  echo "LOCAL_LLM_MODEL was not set and no Ollama model was detected." >&2
  echo "Start/install a local model first, set LOCAL_LLM_MODEL explicitly, or rerun through lab.sh with LAB_AUTO_INSTALL_OLLAMA=1." >&2
  exit 4
fi

./run_lab_local_llm.sh show-config

if target_needs_local_model "$TARGET" && [[ "${SKIP_PROVIDER_CHECK:-0}" != "1" ]]; then
  ./run_lab_local_llm.sh check-provider
fi

ALLOW_OVERWRITE=1 ./run_lab_local_llm.sh "$TARGET"
