#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/lab_run_fresh_profile.sh" tiny10_1y "${1:-../lab_plumbing_10}" "${2:-plumbing-full100-export}"
