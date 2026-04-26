#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/lab_run_fresh_profile.sh" candidate20k "${1:-../lab_candidate_20k}" "${2:-profile-full100}"
