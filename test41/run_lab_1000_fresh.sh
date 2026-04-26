#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/lab_run_fresh_profile.sh" check1000_5y "${1:-../lab_check_1000}" "${2:-profile-full100}"
