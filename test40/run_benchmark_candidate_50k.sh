#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${DATA_SYS_MODEL:-gemini-3.1-flash-lite-preview}"
N_MOVIES="${DATA_SYS_N_MOVIES:-50000}"

export DATA_SYS_PIPELINE_CONFIG="${DATA_SYS_PIPELINE_CONFIG:-$BASE_DIR/benchmark_candidate_profile.json}"
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" \
  "$BASE_DIR/run_full_api_pipeline.py" \
  --base-dir "$BASE_DIR" \
  --n-movies "$N_MOVIES" \
  --profile standard \
  --model "$MODEL" \
  "$@"
