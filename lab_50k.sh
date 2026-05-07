#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LAB_PROFILE=candidate50k
export LAB_RUN_DIR="${LAB_RUN_DIR:-../lab_candidate_50k}"
export LAB_START_TARGET="${LAB_START_TARGET:-profile-full100-export}"
export LAB_EXPORT_AFTER_CONTINUE="${LAB_EXPORT_AFTER_CONTINUE:-1}"
exec "$SCRIPT_DIR/lab.sh" "${1:-start}"
