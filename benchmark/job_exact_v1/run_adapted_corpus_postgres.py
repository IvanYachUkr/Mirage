#!/usr/bin/env python3
"""Run Postgres EXPLAIN/COUNT for an already-adapted JOB SQL corpus."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from exact_job_benchmark import (
    DEFAULT_PG_CONTAINER,
    DEFAULT_PG_USER,
    PostgresRunner,
    explain_postgres_query,
    sort_key,
    structure_fingerprint,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    query_dir = Path(args.query_dir).resolve()
    original_query_dir = Path(args.original_query_dir).resolve() if args.original_query_dir else None
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        [path for path in query_dir.glob("*.sql") if path.name not in {"schema.sql", "fkindexes.sql"}],
        key=sort_key,
    )
    if args.query_ids:
        wanted = {item.strip() for item in args.query_ids.split(",") if item.strip()}
        paths = [path for path in paths if path.stem in wanted]

    runner = PostgresRunner(
        container=args.pg_container,
        db_name=args.db_name,
        user=args.pg_user,
        persistent=bool(args.persistent_pg),
    )
    rows: list[dict[str, Any]] = []
    manifest_queries: list[dict[str, Any]] = []
    try:
        for index, path in enumerate(paths, start=1):
            qid = path.stem
            print(f"[{index}/{len(paths)}] explaining {qid}", flush=True)
            sql = path.read_text(encoding="utf-8")
            original_sql = None
            structure_guard_passed = None
            if original_query_dir is not None:
                original_path = original_query_dir / path.name
                if original_path.exists():
                    original_sql = original_path.read_text(encoding="utf-8")
                    structure_guard_passed = structure_fingerprint(sql) == structure_fingerprint(original_sql)
            status = "OK"
            error = None
            stats = None
            try:
                stats = explain_postgres_query(runner=runner, query_sql=sql)
            except Exception as exc:
                status = "ERROR"
                error = str(exc).splitlines()[0][:240]
            row = {
                "query": qid,
                "status": status,
                "actual_count": None if stats is None else stats.actual_count,
                "structure_guard_passed": structure_guard_passed,
                "sub_agg_estimate": None if stats is None else round(stats.sub_agg_estimate, 4),
                "sub_agg_actual": None if stats is None else round(stats.sub_agg_actual, 4),
                "sub_agg_qerror": None if stats is None else round(stats.sub_agg_qerror, 4),
                "sub_agg_bias": None if stats is None else stats.sub_agg_bias,
                "sub_agg_node": None if stats is None else stats.sub_agg_node,
                "planning_ms": None if stats is None else round(stats.planning_ms, 4),
                "execution_ms": None if stats is None else round(stats.execution_ms, 4),
                "total_ms": None if stats is None else round(stats.total_ms, 4),
                "error": error,
            }
            rows.append(row)
            manifest_queries.append({"query_id": qid, "path": str(path), **row})
            write_csv(out_dir / "postgres_qerror.csv", rows)
            (out_dir / "postgres_qerror.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(
                f"[{index}/{len(paths)}] finished {qid}: "
                f"actual_count={row['actual_count']}, qerror={row['sub_agg_qerror']}, status={status}",
                flush=True,
            )
    finally:
        runner.close()

    qerrors = [float(row["sub_agg_qerror"]) for row in rows if row["status"] == "OK" and row["sub_agg_qerror"] is not None]
    summary = {
        "query_count": len(rows),
        "ok": sum(1 for row in rows if row["status"] == "OK"),
        "errors": sum(1 for row in rows if row["status"] != "OK"),
        "non_zero": sum(1 for row in rows if row["status"] == "OK" and int(row["actual_count"] or 0) > 0),
        "zero": sum(1 for row in rows if row["status"] == "OK" and int(row["actual_count"] or 0) == 0),
        "structure_failures": sum(1 for row in rows if row["structure_guard_passed"] is False),
        "qerror_mean": None if not qerrors else statistics.mean(qerrors),
        "qerror_median": None if not qerrors else statistics.median(qerrors),
        "qerror_max": None if not qerrors else max(qerrors),
    }
    manifest = {
        "db_name": args.db_name,
        "query_dir": str(query_dir),
        "original_query_dir": None if original_query_dir is None else str(original_query_dir),
        "summary": summary,
        "queries": manifest_queries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True, help="Directory containing adapted SQL files.")
    parser.add_argument("--original-query-dir", default=None, help="Optional original SQL directory for structure checks.")
    parser.add_argument("--db-name", required=True, help="Postgres database name.")
    parser.add_argument("--out-dir", required=True, help="Output directory for q-error CSV/JSON/manifest.")
    parser.add_argument("--pg-container", default=DEFAULT_PG_CONTAINER, help="Docker container name for Postgres.")
    parser.add_argument("--pg-user", default=DEFAULT_PG_USER, help="Postgres user.")
    parser.add_argument("--query-ids", default=None, help="Optional comma-separated subset.")
    parser.add_argument("--persistent-pg", action="store_true", help="Use persistent psql session.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
