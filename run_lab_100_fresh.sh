#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/lab_run_fresh_profile.sh" small100_2y "${1:-../lab_smoke_100}" "${2:-profile-full100-export}"
