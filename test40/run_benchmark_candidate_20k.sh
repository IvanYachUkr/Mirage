#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$BASE_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ARTIFACT_SOURCE_DIR="${ARTIFACT_SOURCE_DIR:-$BASE_DIR}"
ENTITY_SOURCE_DIR="${ENTITY_SOURCE_DIR:-$REPO_DIR/test32}"
EXTRA_TITLE_SOURCE_DIR="${EXTRA_TITLE_SOURCE_DIR:-$REPO_DIR/test23}"
MODEL="${DATA_SYS_MODEL:-gemini-3.1-flash-lite-preview}"
N_MOVIES="${DATA_SYS_N_MOVIES:-20000}"
N_PERSONS="${DATA_SYS_N_PERSONS:-32000}"
N_COMPANIES="${DATA_SYS_N_COMPANIES:-650}"
N_KEYWORDS="${DATA_SYS_N_KEYWORDS:-1400}"
N_CHARACTERS="${DATA_SYS_N_CHARACTERS:-40000}"
N_TITLES="${DATA_SYS_N_TITLES:-29161}"
HOST_SCRIPT_WIN="$(wslpath -w "$BASE_DIR/run_pipeline_host.ps1")"

"$PYTHON_BIN" "$BASE_DIR/prepare_seeded_from_test32.py" \
  --artifact-source-dir "$ARTIFACT_SOURCE_DIR" \
  --entity-source-dir "$ENTITY_SOURCE_DIR" \
  --extra-title-source-dir "$EXTRA_TITLE_SOURCE_DIR" \
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
  --benchmark-mode \
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
