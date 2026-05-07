#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

set -a
source "$BASE_DIR/local_run_profiles/candidate100k_2000_2050.env"
set +a

export RUN_PROFILE=candidate100k_2000_2050
export LLM_PROVIDER=gemini
export LOCAL_LLM_MODEL=gemini-3.1-flash-lite-preview
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DATA_SYS_LLM_USAGE_LOG="${DATA_SYS_LLM_USAGE_LOG:-$BASE_DIR/reports/gemini_100k_2000_2050_usage.jsonl}"
export DATA_SYS_LLM_USAGE_LOG_LATEST="${DATA_SYS_LLM_USAGE_LOG_LATEST:-$BASE_DIR/reports/gemini_100k_2000_2050_usage_latest.jsonl}"
export DATA_SYS_LLM_MAX_ATTEMPTS="${DATA_SYS_LLM_MAX_ATTEMPTS:-5}"
export DATA_SYS_LLM_TIMEOUT_SEC="${DATA_SYS_LLM_TIMEOUT_SEC:-180}"
export DATA_SYS_LLM_BASE_DELAY_SEC="${DATA_SYS_LLM_BASE_DELAY_SEC:-2}"
export DATA_SYS_LLM_MAX_DELAY_SEC="${DATA_SYS_LLM_MAX_DELAY_SEC:-60}"
export DATA_SYS_SKIP_DIAGNOSTIC_COLD_EDGES="${DATA_SYS_SKIP_DIAGNOSTIC_COLD_EDGES:-1}"
export DATA_SYS_FAST_TITLE_TAGLINES="${DATA_SYS_FAST_TITLE_TAGLINES:-1}"

LOG_DIR="$BASE_DIR/_runner_logs/candidate100k_2000_2050"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nohup_step80_scalable_to98.log"
PID_FILE="$LOG_DIR/nohup_step80_scalable_to98.pid"

nohup "$BASE_DIR/.venv-linux/bin/python" "$BASE_DIR/run_pipeline.py" \
  --n-movies "$N_MOVIES" \
  --start-year "$START_YEAR" \
  --end-year "$END_YEAR" \
  --n-persons "$N_PERSONS" \
  --n-companies "$N_COMPANIES" \
  --n-keywords "$N_KEYWORDS" \
  --n-characters "$N_CHARACTERS" \
  --n-titles "$N_TITLES" \
  --seed 42 \
  --mode research \
  --model "$LOCAL_LLM_MODEL" \
  --bootstrap-model "$LOCAL_LLM_MODEL" \
  --planning-model "$LOCAL_LLM_MODEL" \
  --bulk-artifact-model "$LOCAL_LLM_MODEL" \
  --force-scalable-graph \
  --benchmark-mode \
  --rerank-budget-movies "$RERANK_BUDGET_MOVIES" \
  --keyword-rerank-budget-movies "$KEYWORD_RERANK_BUDGET_MOVIES" \
  --from-step 80 \
  --until-step 98 \
  --force \
  > "$LOG_FILE" 2>&1 < /dev/null &

pid="$!"
printf "%s\n" "$pid" > "$PID_FILE"
printf "started %s\nlog %s\n" "$pid" "$LOG_FILE"
