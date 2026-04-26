#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$BASE_DIR/.." && pwd)"
SOURCE_DIR="${SOURCE_DIR:-$REPO_DIR/test32}"
MODEL="${DATA_SYS_MODEL:-gemini-3.1-flash-lite-preview}"
N_MOVIES="${DATA_SYS_N_MOVIES:-1000}"
N_PERSONS="${DATA_SYS_N_PERSONS:-6000}"
N_COMPANIES="${DATA_SYS_N_COMPANIES:-650}"
N_KEYWORDS="${DATA_SYS_N_KEYWORDS:-1400}"
N_CHARACTERS="${DATA_SYS_N_CHARACTERS:-40000}"
N_TITLES="${DATA_SYS_N_TITLES:-10000}"
HOST_SCRIPT_WIN="$(wslpath -w "$BASE_DIR/run_pipeline_host.ps1")"

python3 "$BASE_DIR/prepare_seeded_from_test32.py" \
  --source-dir "$SOURCE_DIR" \
  --dest-dir "$BASE_DIR" \
  --target-persons "$N_PERSONS"

/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \
  -ExecutionPolicy Bypass \
  -File "$HOST_SCRIPT_WIN" \
  --n-movies "$N_MOVIES" \
  --n-persons "$N_PERSONS" \
  --n-companies "$N_COMPANIES" \
  --n-keywords "$N_KEYWORDS" \
  --n-characters "$N_CHARACTERS" \
  --n-titles "$N_TITLES" \
  --from-step 80 \
  --until-step 100 \
  --disable-llm-evolution \
  --disable-llm-critic \
  --disable-llm-rerank \
  --disable-llm-keyword-rerank \
  --force-legacy-graph \
  --model "$MODEL"

/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \
  -ExecutionPolicy Bypass \
  -File "$HOST_SCRIPT_WIN" \
  --n-movies "$N_MOVIES" \
  --n-persons "$N_PERSONS" \
  --n-companies "$N_COMPANIES" \
  --n-keywords "$N_KEYWORDS" \
  --n-characters "$N_CHARACTERS" \
  --n-titles "$N_TITLES" \
  --only 130 \
  --model "$MODEL"
