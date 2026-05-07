#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

LOG_DIR="$BASE_DIR/_runner_logs/candidate100k_2000_2050"
mkdir -p "$LOG_DIR"

export FROM_STEP="${FROM_STEP:-60}"
export UNTIL_STEP="${UNTIL_STEP:-98}"
export DATA_SYS_DETERMINISTIC_ENRICH="${DATA_SYS_DETERMINISTIC_ENRICH:-1}"
export DATA_SYS_DETERMINISTIC_LATENTS="${DATA_SYS_DETERMINISTIC_LATENTS:-1}"
export DATA_SYS_FAST_TITLE_TAGLINES="${DATA_SYS_FAST_TITLE_TAGLINES:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

LOG_FILE="$LOG_DIR/nohup_continue_step60_fast_taglines.log"
PID_FILE="$LOG_DIR/nohup_continue_step60_fast_taglines.pid"

nohup ./continue_100k_2000_2050_api_from_step30.sh > "$LOG_FILE" 2>&1 < /dev/null &
pid="$!"
printf "%s\n" "$pid" > "$PID_FILE"
printf "started %s\nlog %s\n" "$pid" "$LOG_FILE"
