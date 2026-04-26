#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="$ROOT_DIR/benchmark/cardinality_methods/runs/$(date +%Y%m%d_%H%M%S)"
PG_CONTAINER="pg_bench"
PG_USER="postgres"
PG_DB=""
DUCKDB_PATH=""
DOCKER_BIN="/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
DUCKDB_BIN="/snap/duckdb/9/duckdb"
EXACT_QUERY_DIR="$ROOT_DIR/benchmark/job_exact_v1/runs/test40_structural_probe_v9_full_witness_real_negatives/adapted_queries/test40"
EXACT_ORIGINAL_DIR="$ROOT_DIR/benchmark/job_exact_v1/runs/test40_structural_probe_v9_full_witness_real_negatives/original_queries"
COMPLEX_QUERY_DIR="$ROOT_DIR/benchmark/job_complex_v1/adapted_v2_test40_v10_pcodes/adapted_queries/job_test40_structural_probe_v10_pcodes"
COMPLEX_ORIGINAL_DIR="$ROOT_DIR/benchmark/job_complex_v1/original_queries"
RUN_DUCKDB=1
RUN_COMPLEX=1

usage() {
  cat <<'EOF'
Usage:
  benchmark/cardinality_methods/scripts/run_ce_suite.sh --pg-db DB [options]

Runs the lightweight CE baseline scaffold:
  1. PostgreSQL EXPLAIN ANALYZE q-error traces.
  2. DuckDB execution/timing baseline when --duckdb-path is provided.
  3. MSCN-compatible workload export files.
  4. Zero-shot input packages from PostgreSQL traces and pg_stats.

Options:
  --pg-db DB                  PostgreSQL database name, required.
  --duckdb-path PATH          Optional DuckDB database file.
  --out-dir DIR               Output directory.
  --pg-container NAME         Default: pg_bench.
  --pg-user USER              Default: postgres.
  --docker-bin PATH           Docker CLI path.
  --duckdb-bin PATH           DuckDB CLI path for Python environments without duckdb.
  --exact-query-dir DIR       Adapted exact-JOB SQL corpus.
  --exact-original-dir DIR    Original exact-JOB SQL corpus for structure checks.
  --complex-query-dir DIR     Adapted JOB-Complex SQL corpus.
  --complex-original-dir DIR  Original JOB-Complex SQL corpus for structure checks.
  --skip-duckdb               Do not run DuckDB baseline.
  --skip-complex              Do not run JOB-Complex path.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pg-db) PG_DB="$2"; shift 2 ;;
    --duckdb-path) DUCKDB_PATH="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --pg-container) PG_CONTAINER="$2"; shift 2 ;;
    --pg-user) PG_USER="$2"; shift 2 ;;
    --docker-bin) DOCKER_BIN="$2"; shift 2 ;;
    --duckdb-bin) DUCKDB_BIN="$2"; shift 2 ;;
    --exact-query-dir) EXACT_QUERY_DIR="$2"; shift 2 ;;
    --exact-original-dir) EXACT_ORIGINAL_DIR="$2"; shift 2 ;;
    --complex-query-dir) COMPLEX_QUERY_DIR="$2"; shift 2 ;;
    --complex-original-dir) COMPLEX_ORIGINAL_DIR="$2"; shift 2 ;;
    --skip-duckdb) RUN_DUCKDB=0; shift ;;
    --skip-complex) RUN_COMPLEX=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PG_DB" ]]; then
  echo "--pg-db is required" >&2
  usage >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "[ce-suite] output: $OUT_DIR"
echo "[ce-suite] postgres exact JOB"
python3 "$ROOT_DIR/benchmark/cardinality_methods/scripts/collect_postgres_plan_trace.py" \
  --query-dir "$EXACT_QUERY_DIR" \
  --original-query-dir "$EXACT_ORIGINAL_DIR" \
  --db-name "$PG_DB" \
  --pg-container "$PG_CONTAINER" \
  --pg-user "$PG_USER" \
  --docker-bin "$DOCKER_BIN" \
  --mode count \
  --out-dir "$OUT_DIR/postgres/job_exact"

echo "[ce-suite] mscn export exact JOB"
python3 "$ROOT_DIR/benchmark/cardinality_methods/scripts/export_mscn_workload.py" \
  --query-dir "$EXACT_QUERY_DIR" \
  --labels "$OUT_DIR/postgres/job_exact/query_summary.csv" \
  --out-prefix "$OUT_DIR/mscn/job_exact"

echo "[ce-suite] zero-shot package exact JOB"
python3 "$ROOT_DIR/benchmark/cardinality_methods/scripts/prepare_zero_shot_inputs.py" \
  --plan-trace-dir "$OUT_DIR/postgres/job_exact" \
  --db-name "$PG_DB" \
  --pg-container "$PG_CONTAINER" \
  --pg-user "$PG_USER" \
  --docker-bin "$DOCKER_BIN" \
  --out-dir "$OUT_DIR/zero_shot/job_exact"

if [[ "$RUN_DUCKDB" -eq 1 && -n "$DUCKDB_PATH" ]]; then
  echo "[ce-suite] duckdb exact JOB"
  python3 "$ROOT_DIR/benchmark/cardinality_methods/scripts/run_duckdb_corpus.py" \
    --db "$DUCKDB_PATH" \
    --duckdb-bin "$DUCKDB_BIN" \
    --query-dir "$EXACT_QUERY_DIR" \
    --mode count \
    --timeout-sec 300 \
    --out-dir "$OUT_DIR/duckdb/job_exact"
fi

if [[ "$RUN_COMPLEX" -eq 1 ]]; then
  echo "[ce-suite] postgres JOB-Complex"
  python3 "$ROOT_DIR/benchmark/cardinality_methods/scripts/collect_postgres_plan_trace.py" \
    --query-dir "$COMPLEX_QUERY_DIR" \
    --original-query-dir "$COMPLEX_ORIGINAL_DIR" \
    --db-name "$PG_DB" \
    --pg-container "$PG_CONTAINER" \
    --pg-user "$PG_USER" \
    --docker-bin "$DOCKER_BIN" \
    --mode original \
    --timeout-sec 300 \
    --out-dir "$OUT_DIR/postgres/job_complex"

  echo "[ce-suite] mscn export JOB-Complex"
  python3 "$ROOT_DIR/benchmark/cardinality_methods/scripts/export_mscn_workload.py" \
    --query-dir "$COMPLEX_QUERY_DIR" \
    --labels "$OUT_DIR/postgres/job_complex/query_summary.csv" \
    --out-prefix "$OUT_DIR/mscn/job_complex"

  echo "[ce-suite] zero-shot package JOB-Complex"
  python3 "$ROOT_DIR/benchmark/cardinality_methods/scripts/prepare_zero_shot_inputs.py" \
    --plan-trace-dir "$OUT_DIR/postgres/job_complex" \
    --db-name "$PG_DB" \
    --pg-container "$PG_CONTAINER" \
    --pg-user "$PG_USER" \
    --docker-bin "$DOCKER_BIN" \
    --out-dir "$OUT_DIR/zero_shot/job_complex"

  if [[ "$RUN_DUCKDB" -eq 1 && -n "$DUCKDB_PATH" ]]; then
    echo "[ce-suite] duckdb JOB-Complex"
    python3 "$ROOT_DIR/benchmark/cardinality_methods/scripts/run_duckdb_corpus.py" \
      --db "$DUCKDB_PATH" \
      --duckdb-bin "$DUCKDB_BIN" \
      --query-dir "$COMPLEX_QUERY_DIR" \
      --mode original \
      --timeout-sec 300 \
      --out-dir "$OUT_DIR/duckdb/job_complex"
  fi
fi

cat > "$OUT_DIR/README.md" <<EOF
# CE Suite Run

- PostgreSQL DB: \`$PG_DB\`
- Exact JOB corpus: \`$EXACT_QUERY_DIR\`
- JOB-Complex corpus: \`$COMPLEX_QUERY_DIR\`
- DuckDB path: \`${DUCKDB_PATH:-not provided}\`

Outputs:
- \`postgres/\`: PostgreSQL EXPLAIN ANALYZE plan/q-error traces.
- \`duckdb/\`: DuckDB execution/timing baseline when enabled.
- \`mscn/\`: MSCN-compatible workload CSV/SQL/bitmap files.
- \`zero_shot/\`: JSON-plan/statistics packages for zero-shot model integration.
EOF

echo "[ce-suite] done: $OUT_DIR"
