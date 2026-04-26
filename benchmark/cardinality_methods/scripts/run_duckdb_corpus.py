#!/usr/bin/env python3
"""Run a SQL corpus against DuckDB for portability and timing baselines."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import shutil
import subprocess
from pathlib import Path

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover - exercised when only the CLI is installed.
    duckdb = None


ROOT = Path(__file__).resolve().parents[3]
EXACT_JOB_DIR = ROOT / "benchmark" / "job_exact_v1"
if str(EXACT_JOB_DIR) not in sys.path:
    sys.path.insert(0, str(EXACT_JOB_DIR))

from exact_job_benchmark import sort_key, strip_to_count  # noqa: E402


DEFAULT_DUCKDB_BIN_CANDIDATES = (
    Path("/snap/bin/duckdb"),
    Path("/snap/duckdb/9/duckdb"),
)


def normalize_sql_for_duckdb(sql: str) -> str:
    normalized = str(sql or "")
    normalized = re.sub(r"\baka_title\s+AS\s+AT\b", "aka_title AS aka_t", normalized, flags=re.I)
    normalized = re.sub(r"\bat\.", "aka_t.", normalized, flags=re.I)
    return normalized


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def detect_duckdb_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("duckdb") or shutil.which("duckdb.exe")
    if found:
        return found
    for candidate in DEFAULT_DUCKDB_BIN_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("DuckDB Python package is missing and no DuckDB CLI was found.")


def run_cli_query(
    *,
    duckdb_bin: str,
    db_path: Path,
    sql: str,
    timeout_sec: int,
) -> tuple[int | None, int, str | None]:
    try:
        completed = subprocess.run(
            [duckdb_bin, str(db_path), "-csv", "-c", sql],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return None, 0, f"timeout after {timeout_sec}s"
    if completed.returncode != 0:
        return None, 0, (completed.stderr or completed.stdout or "").strip().splitlines()[0][:240]
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        return 0, 0, None
    # In -csv mode DuckDB prints a header followed by result rows.
    row_count = max(0, len(lines) - 1)
    value = None
    if row_count >= 1:
        first_data = lines[1].split(",", 1)[0].strip().strip('"')
        try:
            value = int(float(first_data))
        except Exception:
            value = row_count
    return value, row_count, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="DuckDB database path.")
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--query-ids", default=None)
    parser.add_argument("--mode", choices=["original", "count"], default="count")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--duckdb-bin", default=None, help="Optional DuckDB CLI path used when Python duckdb is unavailable.")
    parser.add_argument("--timeout-sec", type=int, default=300)
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    query_dir = Path(args.query_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {item.strip() for item in args.query_ids.split(",") if item.strip()} if args.query_ids else None
    query_paths = [
        path for path in sorted(query_dir.glob("*.sql"), key=sort_key)
        if selected is None or path.stem in selected
    ]
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")
    if not query_paths:
        raise FileNotFoundError(f"No SQL files found in {query_dir}")

    duckdb_bin = None if duckdb is not None else detect_duckdb_bin(args.duckdb_bin)
    if duckdb is None:
        rows: list[dict[str, object]] = []
        for idx, path in enumerate(query_paths, start=1):
            query_id = path.stem
            sql = path.read_text(encoding="utf-8").strip().rstrip(";")
            sql = strip_to_count(sql) if args.mode == "count" else sql
            sql = normalize_sql_for_duckdb(sql)
            start = time.perf_counter()
            value, row_count, error = run_cli_query(
                duckdb_bin=duckdb_bin,
                db_path=db_path,
                sql=sql,
                timeout_sec=int(args.timeout_sec),
            )
            row = {
                "query_id": query_id,
                "status": "error" if error else "ok",
                "mode": args.mode,
                "result_value": value,
                "row_count": row_count,
                "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
                "error": error or "",
            }
            rows.append(row)
            print(json.dumps({"idx": idx, "total": len(query_paths), **row}), flush=True)

        csv_path = out_dir / "duckdb_results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        timings = [float(row["elapsed_ms"]) for row in rows if row["status"] == "ok" and row["elapsed_ms"] is not None]
        manifest = {
            "method": "duckdb",
            "execution_backend": "cli",
            "duckdb_bin": duckdb_bin,
            "db_path": str(db_path),
            "query_dir": str(query_dir),
            "mode": args.mode,
            "analyzed_tables": 0,
            "summary": {
                "query_count": len(rows),
                "ok": sum(1 for row in rows if row["status"] == "ok"),
                "errors": sum(1 for row in rows if row["status"] == "error"),
                "timing_ms": summarize(timings),
            },
            "queries": rows,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest["summary"], indent=2), flush=True)
        return

    con = duckdb.connect(str(db_path), read_only=False)
    try:
        con.execute(f"PRAGMA threads={int(args.threads)}")
        analyzed_tables = 0
        if not args.skip_analyze:
            tables = con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
            ).fetchall()
            for (table_name,) in tables:
                con.execute(f'ANALYZE "{table_name}"')
                analyzed_tables += 1

        rows: list[dict[str, object]] = []
        for idx, path in enumerate(query_paths, start=1):
            query_id = path.stem
            sql = path.read_text(encoding="utf-8").strip().rstrip(";")
            sql = strip_to_count(sql) if args.mode == "count" else sql
            sql = normalize_sql_for_duckdb(sql)
            start = time.perf_counter()
            row: dict[str, object] = {
                "query_id": query_id,
                "status": "ok",
                "mode": args.mode,
                "result_value": None,
                "row_count": None,
                "elapsed_ms": None,
                "error": "",
            }
            try:
                result = con.execute(sql).fetchall()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                row["elapsed_ms"] = round(elapsed_ms, 3)
                row["row_count"] = len(result)
                if args.mode == "count":
                    row["result_value"] = int(result[0][0]) if result else 0
                else:
                    row["result_value"] = len(result)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "error"
                row["elapsed_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
                row["error"] = str(exc).splitlines()[0][:240]
            rows.append(row)
            print(json.dumps({"idx": idx, "total": len(query_paths), **row}), flush=True)

        csv_path = out_dir / "duckdb_results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        timings = [float(row["elapsed_ms"]) for row in rows if row["status"] == "ok" and row["elapsed_ms"] is not None]
        manifest = {
            "method": "duckdb",
            "db_path": str(db_path),
            "query_dir": str(query_dir),
            "mode": args.mode,
            "analyzed_tables": analyzed_tables,
            "summary": {
                "query_count": len(rows),
                "ok": sum(1 for row in rows if row["status"] == "ok"),
                "errors": sum(1 for row in rows if row["status"] == "error"),
                "timing_ms": summarize(timings),
            },
            "queries": rows,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest["summary"], indent=2), flush=True)
    finally:
        con.close()


if __name__ == "__main__":
    main()
