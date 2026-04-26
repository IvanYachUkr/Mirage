#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LAB_PROFILE=candidate100k
export LAB_RUN_DIR="${LAB_RUN_DIR:-../lab_candidate_100k}"
exec "$SCRIPT_DIR/lab.sh" "${1:-start}"

