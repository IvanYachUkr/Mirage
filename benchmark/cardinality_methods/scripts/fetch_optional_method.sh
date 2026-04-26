#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./benchmark/cardinality_methods/scripts/fetch_optional_method.sh <method>

Methods:
  ceb        Download learnedsystems/CEB source archive.
  lcm-eval   Download DataManagementLab/lcm-eval source archive.
  deepdb     Download DataManagementLab/deepdb-public source archive.

This script downloads source archives only. It does not download benchmark
datasets, model checkpoints, OSF artifacts, or IMDb dumps.
EOF
}

method="${1:-}"
if [[ -z "$method" || "$method" == "-h" || "$method" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$BASE_DIR/archives" "$BASE_DIR/sources"

case "$method" in
  ceb)
    url="https://github.com/learnedsystems/CEB/archive/refs/heads/main.zip"
    archive="$BASE_DIR/archives/CEB-main.zip"
    ;;
  lcm-eval)
    url="https://github.com/DataManagementLab/lcm-eval/archive/refs/heads/main.zip"
    archive="$BASE_DIR/archives/lcm-eval-main.zip"
    ;;
  deepdb)
    url="https://github.com/DataManagementLab/deepdb-public/archive/refs/heads/master.zip"
    archive="$BASE_DIR/archives/deepdb-public-master.zip"
    ;;
  *)
    echo "Unknown method: $method" >&2
    usage >&2
    exit 2
    ;;
esac

echo "Downloading source archive: $url"
curl -L --fail --show-error -o "$archive" "$url"
echo "Extracting into $BASE_DIR/sources"
unzip -q -o "$archive" -d "$BASE_DIR/sources"
sha256sum "$archive"
