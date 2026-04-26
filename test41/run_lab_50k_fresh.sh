#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/lab_run_fresh_profile.sh" candidate50k "${1:-../lab_candidate_50k}" "${2:-profile-full100}"
