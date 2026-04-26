from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DOCKER_BIN_CANDIDATES = (
    Path(r"/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"),
    Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
)


def detect_docker_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("docker") or shutil.which("docker.exe")
    if found:
        return found
    for candidate in DEFAULT_DOCKER_BIN_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "docker"


def read_sql(path: Path) -> str:
    sql = path.read_text(encoding="utf-8").strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def docker_psql(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    sql: str,
    timeout_sec: int,
) -> tuple[int, str, str, float]:
    cmd = [
        docker_bin,
        "exec",
        container,
        "psql",
        "-X",
        "-U",
        user,
        "-d",
        db_name,
        "-v",
        "ON_ERROR_STOP=1",
        "-P",
        "pager=off",
        "-tAq",
        "-c",
        sql,
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip(), elapsed_ms


def root_plan_rows(explain_json: str) -> float | None:
    data = json.loads(explain_json)
    if not data:
        return None
    plan = data[0].get("Plan", {})
    rows = plan.get("Plan Rows")
    return float(rows) if rows is not None else None


def q_error(actual: int | None, estimate: float | None) -> float | None:
    if actual is None or estimate is None:
        return None
    if actual < 0 or estimate < 0:
        return None
    if actual == 0 and estimate == 0:
        return 1.0
    if actual == 0 or estimate == 0:
        return math.inf
    return max(float(actual) / float(estimate), float(estimate) / float(actual))


def run_query(
    query_path: Path,
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    timeout_sec: int,
) -> dict[str, Any]:
    sql = read_sql(query_path)
    row: dict[str, Any] = {
        "query_id": query_path.stem,
        "file": str(query_path),
        "status": "ok",
        "actual_rows": None,
        "estimated_rows": None,
        "root_q_error": None,
        "count_ms": None,
        "explain_ms": None,
        "error": "",
    }

    count_sql = f"SELECT COUNT(*) FROM ({sql}) AS job_complex_q;"
    try:
        rc, stdout, stderr, elapsed_ms = docker_psql(
            docker_bin=docker_bin,
            container=container,
            db_name=db_name,
            user=user,
            sql=count_sql,
            timeout_sec=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        row["error"] = f"COUNT timed out after {timeout_sec}s"
        return row
    row["count_ms"] = round(elapsed_ms, 3)
    if rc != 0:
        row["status"] = "error"
        row["error"] = stderr or stdout
        return row
    row["actual_rows"] = int(float(stdout or "0"))

    explain_sql = f"EXPLAIN (FORMAT JSON) {sql};"
    try:
        rc, stdout, stderr, elapsed_ms = docker_psql(
            docker_bin=docker_bin,
            container=container,
            db_name=db_name,
            user=user,
            sql=explain_sql,
            timeout_sec=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        row["error"] = f"EXPLAIN timed out after {timeout_sec}s"
        return row
    row["explain_ms"] = round(elapsed_ms, 3)
    if rc != 0:
        row["status"] = "error"
        row["error"] = stderr or stdout
        return row
    try:
        row["estimated_rows"] = root_plan_rows(stdout)
        row["root_q_error"] = q_error(row["actual_rows"], row["estimated_rows"])
    except Exception as exc:  # noqa: BLE001 - keep report generation robust.
        row["status"] = "error"
        row["error"] = f"Could not parse EXPLAIN JSON: {exc}"
    return row


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "job_complex_postgres_results.json"
    csv_path = out_dir / "job_complex_postgres_results.csv"
    summary_path = out_dir / "job_complex_postgres_summary.json"

    json_path.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    fieldnames = [
        "query_id",
        "status",
        "actual_rows",
        "estimated_rows",
        "root_q_error",
        "count_ms",
        "explain_ms",
        "error",
        "file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    q_errors = [
        float(row["root_q_error"])
        for row in rows
        if row.get("status") == "ok" and row.get("root_q_error") not in (None, math.inf)
    ]
    summary = {
        "query_count": len(rows),
        "ok": sum(1 for row in rows if row.get("status") == "ok"),
        "errors": sum(1 for row in rows if row.get("status") == "error"),
        "timeouts": sum(1 for row in rows if row.get("status") == "timeout"),
        "zero_actual": sum(1 for row in rows if row.get("actual_rows") == 0),
        "non_zero_actual": sum(1 for row in rows if (row.get("actual_rows") or 0) > 0),
        "mean_root_q_error": (sum(q_errors) / len(q_errors)) if q_errors else None,
        "median_root_q_error": sorted(q_errors)[len(q_errors) // 2] if q_errors else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JOB-Complex SQL files on a PostgreSQL Docker container.")
    parser.add_argument("--query-dir", default=str(Path(__file__).resolve().parent / "original_queries"))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "runs"))
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", default=None, help="Optional docker/docker.exe path.")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--query-ids", default=None, help="Optional comma-separated query IDs, e.g. 01,02,30.")
    args = parser.parse_args()

    query_dir = Path(args.query_dir)
    selected = None
    if args.query_ids:
        selected = {item.strip() for item in args.query_ids.split(",") if item.strip()}
    query_paths = sorted(query_dir.glob("*.sql"))
    if selected is not None:
        query_paths = [path for path in query_paths if path.stem in selected]
    if not query_paths:
        raise FileNotFoundError(f"No JOB-Complex query files found in {query_dir}")
    docker_bin = detect_docker_bin(args.docker_bin)

    rows = [
        run_query(
            query_path,
            docker_bin=docker_bin,
            container=args.pg_container,
            db_name=args.db_name,
            user=args.pg_user,
            timeout_sec=args.timeout_sec,
        )
        for query_path in query_paths
    ]
    write_outputs(rows, Path(args.out_dir))
    print(json.dumps({"query_count": len(rows), "out_dir": str(Path(args.out_dir))}, indent=2))


if __name__ == "__main__":
    main()
