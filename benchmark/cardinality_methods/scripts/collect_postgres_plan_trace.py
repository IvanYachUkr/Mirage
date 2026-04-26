#!/usr/bin/env python3
"""Collect PostgreSQL EXPLAIN ANALYZE cardinality traces for a SQL corpus."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXACT_JOB_DIR = ROOT / "benchmark" / "job_exact_v1"
if str(EXACT_JOB_DIR) not in sys.path:
    sys.path.insert(0, str(EXACT_JOB_DIR))

from exact_job_benchmark import sort_key, strip_to_count, structure_fingerprint  # noqa: E402


DEFAULT_DOCKER_BIN_CANDIDATES = (
    Path(r"/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"),
    Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
)

WRAPPER_NODE_TYPES = {
    "Aggregate",
    "Finalize Aggregate",
    "Partial Aggregate",
    "GroupAggregate",
    "HashAggregate",
    "Result",
    "Limit",
    "Sort",
    "Incremental Sort",
    "Unique",
    "Gather",
    "Gather Merge",
    "Materialize",
}


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
    return path.read_text(encoding="utf-8").strip().rstrip(";")


def q_error(estimate: Any, actual: Any) -> float | None:
    if estimate is None or actual is None:
        return None
    estimate = float(estimate)
    actual = float(actual)
    if estimate == 0 and actual == 0:
        return 1.0
    if estimate == 0 or actual == 0:
        return math.inf
    return max(estimate / actual, actual / estimate)


def finite(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def summarize(values: list[float | None]) -> dict[str, Any]:
    finite_values = finite(values)
    inf_count = sum(1 for value in values if value is not None and math.isinf(float(value)))
    return {
        "count": len([value for value in values if value is not None]),
        "finite_count": len(finite_values),
        "infinite_count": inf_count,
        "mean": statistics.fmean(finite_values) if finite_values else None,
        "median": statistics.median(finite_values) if finite_values else None,
        "p90": percentile(finite_values, 0.90),
        "p95": percentile(finite_values, 0.95),
        "max": max(finite_values) if finite_values else (math.inf if inf_count else None),
    }


def bias_direction(estimate: Any, actual: Any) -> str:
    if estimate is None or actual is None:
        return ""
    estimate = float(estimate)
    actual = float(actual)
    if estimate == actual:
        return "exact"
    if actual == 0:
        return "over" if estimate > 0 else "exact"
    return "over" if estimate > actual else "under"


def extract_json_array(stdout: str) -> Any:
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Could not find EXPLAIN JSON array in output: {stdout[:400]}")
    return json.loads(stdout[start:end + 1])


def run_explain(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    sql: str,
    timeout_sec: int,
    buffers: bool,
) -> tuple[int, str, str, float]:
    options = ["ANALYZE", "VERBOSE", "FORMAT JSON", "TIMING TRUE", "SUMMARY TRUE"]
    if buffers:
        options.append("BUFFERS")
    wrapped = (
        f"SET statement_timeout = '{int(timeout_sec * 1000)}ms'; "
        f"EXPLAIN ({', '.join(options)}) {sql};"
    )
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
        wrapped,
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec + 20,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip(), (time.perf_counter() - start) * 1000.0


def flatten_plan(
    plan: dict[str, Any],
    *,
    query_id: str,
    parent_id: str = "",
    depth: int = 0,
    counter: list[int] | None = None,
) -> list[dict[str, Any]]:
    if counter is None:
        counter = [0]
    counter[0] += 1
    node_id = f"{query_id}:{counter[0]:04d}"
    plan_rows = plan.get("Plan Rows")
    actual_rows = plan.get("Actual Rows")
    actual_loops = plan.get("Actual Loops")
    actual_total_rows = None
    if actual_rows is not None and actual_loops is not None:
        actual_total_rows = float(actual_rows) * float(actual_loops)
    row = {
        "query_id": query_id,
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "node_type": plan.get("Node Type", ""),
        "join_type": plan.get("Join Type", ""),
        "relation_name": plan.get("Relation Name", ""),
        "alias": plan.get("Alias", ""),
        "plan_rows": plan_rows,
        "actual_rows_per_loop": actual_rows,
        "actual_loops": actual_loops,
        "actual_total_rows": actual_total_rows,
        "q_error_per_loop": q_error(plan_rows, actual_rows),
        "q_error_total": q_error(plan_rows, actual_total_rows),
        "startup_cost": plan.get("Startup Cost"),
        "total_cost": plan.get("Total Cost"),
        "actual_startup_ms": plan.get("Actual Startup Time"),
        "actual_total_ms": plan.get("Actual Total Time"),
        "filter": plan.get("Filter", ""),
        "join_filter": plan.get("Join Filter", ""),
        "hash_cond": plan.get("Hash Cond", ""),
        "merge_cond": plan.get("Merge Cond", ""),
        "index_cond": plan.get("Index Cond", ""),
    }
    rows = [row]
    for child in plan.get("Plans", []) or []:
        rows.extend(flatten_plan(child, query_id=query_id, parent_id=node_id, depth=depth + 1, counter=counter))
    return rows


def choose_target_node(nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the CE label node, skipping aggregate/result wrappers when possible.

    PostgreSQL EXPLAIN for COUNT(*) and aggregate JOB-Complex queries usually
    has a root node with exactly one output row. The cardinality target for CE
    methods is the first real relational node underneath that wrapper.
    """
    for node in nodes:
        if str(node.get("node_type") or "") not in WRAPPER_NODE_TYPES:
            return node
    return nodes[0] if nodes else None


def node_actual_label(node: dict[str, Any] | None) -> float | None:
    if node is None:
        return None
    actual_total = node.get("actual_total_rows")
    if actual_total is not None:
        return float(actual_total)
    actual_per_loop = node.get("actual_rows_per_loop")
    if actual_per_loop is not None:
        return float(actual_per_loop)
    return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", default=None)
    parser.add_argument("--original-query-dir", default=None)
    parser.add_argument("--query-ids", default=None)
    parser.add_argument("--mode", choices=["original", "count"], default="count")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--buffers", action="store_true")
    args = parser.parse_args()

    query_dir = Path(args.query_dir).resolve()
    original_query_dir = Path(args.original_query_dir).resolve() if args.original_query_dir else None
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_plan_dir = out_dir / "raw_plans"
    raw_plan_dir.mkdir(parents=True, exist_ok=True)

    selected = {item.strip() for item in args.query_ids.split(",") if item.strip()} if args.query_ids else None
    query_paths = [
        path for path in sorted(query_dir.glob("*.sql"), key=sort_key)
        if selected is None or path.stem in selected
    ]
    docker_bin = detect_docker_bin(args.docker_bin)

    query_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    for idx, path in enumerate(query_paths, start=1):
        query_id = path.stem
        base_sql = read_sql(path)
        executed_sql = strip_to_count(base_sql) if args.mode == "count" else base_sql
        row: dict[str, Any] = {
            "query_id": query_id,
            "status": "ok",
            "mode": args.mode,
            "elapsed_ms": None,
            "planning_ms": None,
            "execution_ms": None,
            "node_count": 0,
            "root_node_type": "",
            "root_plan_rows": None,
            "root_actual_rows": None,
            "root_q_error": None,
            "target_node_id": "",
            "target_node_type": "",
            "target_plan_rows": None,
            "target_actual": None,
            "target_q_error": None,
            "target_bias": "",
            "node_q_error_median": None,
            "node_q_error_p90": None,
            "node_q_error_max": None,
            "structure_guard_passed": None,
            "error": "",
        }
        if original_query_dir is not None and (original_query_dir / path.name).exists():
            row["structure_guard_passed"] = (
                structure_fingerprint(base_sql) == structure_fingerprint(read_sql(original_query_dir / path.name))
            )
        try:
            rc, stdout, stderr, elapsed_ms = run_explain(
                docker_bin=docker_bin,
                container=args.pg_container,
                db_name=args.db_name,
                user=args.pg_user,
                sql=executed_sql,
                timeout_sec=args.timeout_sec,
                buffers=bool(args.buffers),
            )
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
            row["error"] = f"timeout after {args.timeout_sec}s"
            query_rows.append(row)
            print(json.dumps({"query_id": query_id, "status": row["status"], "error": row["error"]}), flush=True)
            continue
        row["elapsed_ms"] = round(elapsed_ms, 3)
        if rc != 0:
            row["status"] = "error"
            row["error"] = stderr or stdout
            query_rows.append(row)
            print(json.dumps({"query_id": query_id, "status": row["status"], "error": row["error"][:160]}), flush=True)
            continue
        try:
            explain_doc = extract_json_array(stdout)
            (raw_plan_dir / f"{query_id}.json").write_text(json.dumps(explain_doc, indent=2, allow_nan=True), encoding="utf-8")
            plan_doc = explain_doc[0]
            nodes = flatten_plan(plan_doc["Plan"], query_id=query_id)
            root = nodes[0]
            target = choose_target_node(nodes)
            target_actual = node_actual_label(target)
            target_plan_rows = None if target is None else target.get("plan_rows")
            node_qerrors = [node.get("q_error_per_loop") for node in nodes]
            node_summary = summarize(node_qerrors)
            row.update(
                {
                    "planning_ms": plan_doc.get("Planning Time"),
                    "execution_ms": plan_doc.get("Execution Time"),
                    "node_count": len(nodes),
                    "root_node_type": root.get("node_type", ""),
                    "root_plan_rows": root.get("plan_rows"),
                    "root_actual_rows": root.get("actual_rows_per_loop"),
                    "root_q_error": root.get("q_error_per_loop"),
                    "target_node_id": "" if target is None else target.get("node_id", ""),
                    "target_node_type": "" if target is None else target.get("node_type", ""),
                    "target_plan_rows": target_plan_rows,
                    "target_actual": target_actual,
                    "target_q_error": q_error(target_plan_rows, target_actual),
                    "target_bias": bias_direction(target_plan_rows, target_actual),
                    "node_q_error_median": node_summary["median"],
                    "node_q_error_p90": node_summary["p90"],
                    "node_q_error_max": node_summary["max"],
                }
            )
            node_rows.extend(nodes)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"could not parse plan: {exc}"
        query_rows.append(row)
        print(
            json.dumps(
                {
                    "idx": idx,
                    "total": len(query_paths),
                    "query_id": query_id,
                    "status": row["status"],
                    "target_actual": row["target_actual"],
                    "target_q_error": row["target_q_error"],
                    "node_q_error_median": row["node_q_error_median"],
                    "execution_ms": row["execution_ms"],
                },
                allow_nan=True,
            ),
            flush=True,
        )

    query_fields = list(query_rows[0].keys()) if query_rows else []
    node_fields = [
        "query_id", "node_id", "parent_id", "depth", "node_type", "join_type", "relation_name", "alias",
        "plan_rows", "actual_rows_per_loop", "actual_loops", "actual_total_rows", "q_error_per_loop",
        "q_error_total", "startup_cost", "total_cost", "actual_startup_ms", "actual_total_ms",
        "filter", "join_filter", "hash_cond", "merge_cond", "index_cond",
    ]
    if query_fields:
        write_csv(out_dir / "query_summary.csv", query_rows, query_fields)
    write_csv(out_dir / "plan_nodes.csv", node_rows, node_fields)
    summary = {
        "query_count": len(query_rows),
        "ok": sum(1 for row in query_rows if row["status"] == "ok"),
        "errors": sum(1 for row in query_rows if row["status"] == "error"),
        "timeouts": sum(1 for row in query_rows if row["status"] == "timeout"),
        "mode": args.mode,
        "target_nonzero": sum(
            1 for row in query_rows
            if row["status"] == "ok" and row.get("target_actual") not in (None, "") and float(row["target_actual"]) > 0
        ),
        "target_zero": sum(
            1 for row in query_rows
            if row["status"] == "ok" and row.get("target_actual") not in (None, "") and float(row["target_actual"]) == 0
        ),
        "structure_guard_failures": sum(1 for row in query_rows if row.get("structure_guard_passed") is False),
        "target_q_error_summary": summarize([row.get("target_q_error") for row in query_rows]),
        "node_q_error_summary": summarize([row.get("q_error_per_loop") for row in node_rows]),
        "query_median_q_error_summary": summarize([row.get("node_q_error_median") for row in query_rows]),
    }
    manifest = {
        "method": "postgres",
        "db_name": args.db_name,
        "query_dir": str(query_dir),
        "original_query_dir": None if original_query_dir is None else str(original_query_dir),
        "summary": summary,
        "queries": query_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
