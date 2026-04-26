#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$BASE_DIR/_runner_logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/gemini20k_reuse_${STAMP}.log"
PID_FILE="$LOG_DIR/gemini20k_reuse_latest.pid"
LOG_POINTER="$LOG_DIR/gemini20k_reuse_latest.logpath"

nohup bash -c '
set -euo pipefail
cd "$1"
export RUN_PROFILE=candidate20k_reuse_test23_gemini
export PYTHON_BIN="${PYTHON_BIN:-../test39_local_smoke/.venv-linux/bin/python}"
export ALLOW_OVERWRITE=1
export LLM_PROVIDER=gemini
export LOCAL_LLM_MODEL=gemini-3.1-flash-lite-preview
exec bash run_lab_local_llm.sh candidate20k-step100
' bash "$BASE_DIR" > "$LOG_FILE" 2>&1 &

PID="$!"
printf '%s\n' "$PID" > "$PID_FILE"
printf '%s\n' "$LOG_FILE" > "$LOG_POINTER"

echo "Started Gemini 20k reuse run"
echo "PID: $PID"
echo "Log: $LOG_FILE"
