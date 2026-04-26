#!/usr/bin/env python3
"""Package PostgreSQL plan traces and dataset stats for zero-shot estimators."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


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


def run_psql(*, docker_bin: str, container: str, db_name: str, user: str, sql: str) -> str:
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
        "-c",
        sql,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    return completed.stdout


def write_pg_stats(*, out_dir: Path, docker_bin: str, container: str, db_name: str, user: str) -> None:
    table_sql = """
    COPY (
    SELECT relname AS table_name, n_live_tup::bigint AS estimated_rows
    FROM pg_stat_user_tables
    ORDER BY relname
    ) TO STDOUT WITH CSV HEADER;
    """
    column_sql = """
    COPY (
    SELECT schemaname, tablename, attname, null_frac, n_distinct,
           COALESCE(array_to_string(most_common_vals, '|'), '') AS most_common_vals,
           COALESCE(array_to_string(histogram_bounds, '|'), '') AS histogram_bounds
    FROM pg_stats
    WHERE schemaname = 'public'
    ORDER BY tablename, attname
    ) TO STDOUT WITH CSV HEADER;
    """
    (out_dir / "postgres_table_stats.csv").write_text(
        run_psql(docker_bin=docker_bin, container=container, db_name=db_name, user=user, sql=table_sql),
        encoding="utf-8",
    )
    (out_dir / "postgres_column_stats.csv").write_text(
        run_psql(docker_bin=docker_bin, container=container, db_name=db_name, user=user, sql=column_sql),
        encoding="utf-8",
    )


def copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-trace-dir", required=True, help="Output dir from collect_postgres_plan_trace.py.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--db-name", default=None, help="Optional Postgres DB for pg_stats export.")
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", default=None)
    parser.add_argument("--zero-shot-source", default="benchmark/cardinality_methods/sources/zero-shot-cost-estimation-main")
    parser.add_argument("--copy-raw-plans", action="store_true")
    args = parser.parse_args()

    plan_trace_dir = Path(args.plan_trace_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = {
        "manifest": copy_if_exists(plan_trace_dir / "manifest.json", out_dir / "postgres_trace_manifest.json"),
        "query_summary": copy_if_exists(plan_trace_dir / "query_summary.csv", out_dir / "query_summary.csv"),
        "plan_nodes": copy_if_exists(plan_trace_dir / "plan_nodes.csv", out_dir / "plan_nodes.csv"),
    }
    raw_plan_target = None
    if args.copy_raw_plans and (plan_trace_dir / "raw_plans").exists():
        raw_plan_target = out_dir / "raw_plans"
        if raw_plan_target.exists():
            shutil.rmtree(raw_plan_target)
        shutil.copytree(plan_trace_dir / "raw_plans", raw_plan_target)

    pg_stats_exported = False
    if args.db_name:
        write_pg_stats(
            out_dir=out_dir,
            docker_bin=detect_docker_bin(args.docker_bin),
            container=args.pg_container,
            db_name=args.db_name,
            user=args.pg_user,
        )
        pg_stats_exported = True

    manifest = {
        "method": "zero_shot_input_package",
        "plan_trace_dir": str(plan_trace_dir),
        "zero_shot_source": str(Path(args.zero_shot_source).resolve()),
        "db_name": args.db_name,
        "copied": copied,
        "raw_plans": None if raw_plan_target is None else str(raw_plan_target),
        "postgres_stats_exported": pg_stats_exported,
        "notes": [
            "This is an input package for zero-shot/lcm-style estimators, not a trained-model result.",
            "The staged zero-shot repository expects additional dataset registration before model execution.",
            "The package preserves PostgreSQL JSON-plan traces plus pg_stats so integration can start from stable artifacts.",
        ],
    }
    (out_dir / "zero_shot_input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = [
        "# Zero-Shot Input Package",
        "",
        "This directory contains PostgreSQL plan traces and statistics prepared for zero-shot cost/cardinality model integration.",
        "",
        f"- source trace: `{plan_trace_dir}`",
        f"- zero-shot source checkout: `{manifest['zero_shot_source']}`",
        f"- postgres stats exported: `{pg_stats_exported}`",
        "",
        "This does not claim a zero-shot learned result yet; it makes the lab run reproducible and ready for model-specific registration.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
