from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
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


def extract_json_array(stdout: str) -> Any:
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Could not find JSON array in psql output: {stdout[:400]}")
    return json.loads(stdout[start : end + 1])


def q_error(estimate: float | None, actual: float | None) -> float | None:
    if estimate is None or actual is None:
        return None
    estimate = float(estimate)
    actual = float(actual)
    if estimate < 0 or actual < 0:
        return None
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
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summarize_qerrors(values: list[float | None]) -> dict[str, Any]:
    finite_values = finite(values)
    infinite_count = sum(1 for value in values if value is not None and math.isinf(float(value)))
    return {
        "count": len([value for value in values if value is not None]),
        "finite_count": len(finite_values),
        "infinite_count": infinite_count,
        "mean": statistics.fmean(finite_values) if finite_values else None,
        "median": statistics.median(finite_values) if finite_values else None,
        "p90": percentile(finite_values, 0.90),
        "p95": percentile(finite_values, 0.95),
        "max": max(finite_values) if finite_values else (math.inf if infinite_count else None),
    }


def explain_analyze(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    sql: str,
    timeout_sec: int,
    buffers: bool,
) -> tuple[int, str, str, float]:
    explain_options = ["ANALYZE", "VERBOSE", "FORMAT JSON", "TIMING TRUE", "SUMMARY TRUE"]
    if buffers:
        explain_options.append("BUFFERS")
    wrapped_sql = (
        f"SET statement_timeout = '{int(timeout_sec * 1000)}ms'; "
        f"EXPLAIN ({', '.join(explain_options)}) {sql};"
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
        wrapped_sql,
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec + 20,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip(), elapsed_ms


def node_field(plan: dict[str, Any], field: str) -> Any:
    value = plan.get(field)
    if value is None:
        return ""
    return value


def flatten_plan_nodes(
    plan: dict[str, Any],
    *,
    query_id: str,
    parent_id: str | None = None,
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
        "parent_id": parent_id or "",
        "depth": depth,
        "node_type": node_field(plan, "Node Type"),
        "join_type": node_field(plan, "Join Type"),
        "relation_name": node_field(plan, "Relation Name"),
        "schema": node_field(plan, "Schema"),
        "alias": node_field(plan, "Alias"),
        "plan_rows": plan_rows,
        "plan_width": plan.get("Plan Width"),
        "startup_cost": plan.get("Startup Cost"),
        "total_cost": plan.get("Total Cost"),
        "actual_startup_ms": plan.get("Actual Startup Time"),
        "actual_total_ms": plan.get("Actual Total Time"),
        "actual_rows_per_loop": actual_rows,
        "actual_loops": actual_loops,
        "actual_total_rows": actual_total_rows,
        "q_error_per_loop": q_error(plan_rows, actual_rows),
        "q_error_total": q_error(plan_rows, actual_total_rows),
        "filter": node_field(plan, "Filter"),
        "join_filter": node_field(plan, "Join Filter"),
        "hash_cond": node_field(plan, "Hash Cond"),
        "merge_cond": node_field(plan, "Merge Cond"),
        "index_cond": node_field(plan, "Index Cond"),
        "recheck_cond": node_field(plan, "Recheck Cond"),
    }
    rows = [row]
    for child in plan.get("Plans", []) or []:
        rows.extend(
            flatten_plan_nodes(
                child,
                query_id=query_id,
                parent_id=node_id,
                depth=depth + 1,
                counter=counter,
            )
        )
    return rows


def run_one(
    query_path: Path,
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    timeout_sec: int,
    buffers: bool,
    raw_plan_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_id = query_path.stem
    row: dict[str, Any] = {
        "query_id": query_id,
        "status": "ok",
        "file": str(query_path),
        "elapsed_ms": None,
        "planning_ms": None,
        "execution_ms": None,
        "node_count": 0,
        "root_node_type": "",
        "root_plan_rows": None,
        "root_actual_rows_per_loop": None,
        "root_actual_loops": None,
        "root_q_error_per_loop": None,
        "root_q_error_total": None,
        "node_q_error_mean": None,
        "node_q_error_median": None,
        "node_q_error_p90": None,
        "node_q_error_max": None,
        "error": "",
    }
    sql = read_sql(query_path)
    try:
        rc, stdout, stderr, elapsed_ms = explain_analyze(
            docker_bin=docker_bin,
            container=container,
            db_name=db_name,
            user=user,
            sql=sql,
            timeout_sec=timeout_sec,
            buffers=buffers,
        )
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        row["error"] = f"EXPLAIN ANALYZE timed out after {timeout_sec}s"
        return row, []
    row["elapsed_ms"] = round(elapsed_ms, 3)
    if rc != 0:
        row["status"] = "error"
        row["error"] = stderr or stdout
        return row, []

    try:
        explain_doc = extract_json_array(stdout)
        raw_plan_dir.mkdir(parents=True, exist_ok=True)
        (raw_plan_dir / f"{query_id}.json").write_text(
            json.dumps(explain_doc, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        plan_doc = explain_doc[0]
        root_plan = plan_doc["Plan"]
        nodes = flatten_plan_nodes(root_plan, query_id=query_id)
    except Exception as exc:  # noqa: BLE001 - keep the corpus run going.
        row["status"] = "error"
        row["error"] = f"Could not parse plan JSON: {exc}"
        return row, []

    root = nodes[0]
    node_summary = summarize_qerrors([node.get("q_error_per_loop") for node in nodes])
    row.update(
        {
            "planning_ms": plan_doc.get("Planning Time"),
            "execution_ms": plan_doc.get("Execution Time"),
            "node_count": len(nodes),
            "root_node_type": root.get("node_type", ""),
            "root_plan_rows": root.get("plan_rows"),
            "root_actual_rows_per_loop": root.get("actual_rows_per_loop"),
            "root_actual_loops": root.get("actual_loops"),
            "root_q_error_per_loop": root.get("q_error_per_loop"),
            "root_q_error_total": root.get("q_error_total"),
            "node_q_error_mean": node_summary["mean"],
            "node_q_error_median": node_summary["median"],
            "node_q_error_p90": node_summary["p90"],
            "node_q_error_max": node_summary["max"],
        }
    )
    return row, nodes


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_report(out_dir: Path, query_rows: list[dict[str, Any]], node_rows: list[dict[str, Any]]) -> None:
    query_qerrors = [row.get("node_q_error_median") for row in query_rows if row.get("status") == "ok"]
    node_qerrors = [row.get("q_error_per_loop") for row in node_rows]
    summary = {
        "query_count": len(query_rows),
        "ok": sum(1 for row in query_rows if row.get("status") == "ok"),
        "errors": sum(1 for row in query_rows if row.get("status") == "error"),
        "timeouts": sum(1 for row in query_rows if row.get("status") == "timeout"),
        "total_plan_nodes": len(node_rows),
        "query_median_q_error_summary": summarize_qerrors(query_qerrors),
        "node_q_error_summary": summarize_qerrors(node_qerrors),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# JOB-Complex EXPLAIN ANALYZE Report",
        "",
        f"- queries: {summary['query_count']}",
        f"- ok: {summary['ok']}",
        f"- errors: {summary['errors']}",
        f"- timeouts: {summary['timeouts']}",
        f"- plan nodes: {summary['total_plan_nodes']}",
        f"- node median q-error: {summary['node_q_error_summary']['median']}",
        f"- node p90 q-error: {summary['node_q_error_summary']['p90']}",
        f"- node max q-error: {summary['node_q_error_summary']['max']}",
        "",
        "Note: `q_error_per_loop` compares PostgreSQL `Plan Rows` with `Actual Rows`.",
        "`q_error_total` is also emitted for cases where loop-expanded actual rows are useful.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXPLAIN ANALYZE JSON for JOB-Complex and export node-level q-error metrics."
    )
    parser.add_argument("--query-dir", default=str(Path(__file__).resolve().parent / "original_queries"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", default=None)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--query-ids", default=None, help="Optional comma-separated query IDs, e.g. 01,02,30.")
    parser.add_argument("--buffers", action="store_true", help="Include BUFFERS in EXPLAIN output.")
    args = parser.parse_args()

    query_dir = Path(args.query_dir)
    selected = None
    if args.query_ids:
        selected = {item.strip() for item in args.query_ids.split(",") if item.strip()}
    query_paths = sorted(query_dir.glob("*.sql"))
    if selected is not None:
        query_paths = [path for path in query_paths if path.stem in selected]
    if not query_paths:
        raise FileNotFoundError(f"No query files found in {query_dir}")

    docker_bin = detect_docker_bin(args.docker_bin)
    out_dir = Path(args.out_dir)
    raw_plan_dir = out_dir / "raw_plans"
    query_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []

    for query_path in query_paths:
        query_row, query_nodes = run_one(
            query_path,
            docker_bin=docker_bin,
            container=args.pg_container,
            db_name=args.db_name,
            user=args.pg_user,
            timeout_sec=args.timeout_sec,
            buffers=args.buffers,
            raw_plan_dir=raw_plan_dir,
        )
        query_rows.append(query_row)
        node_rows.extend(query_nodes)
        print(
            json.dumps(
                {
                    "query_id": query_row["query_id"],
                    "status": query_row["status"],
                    "node_count": query_row["node_count"],
                    "median_q_error": query_row["node_q_error_median"],
                    "execution_ms": query_row["execution_ms"],
                },
                allow_nan=True,
            ),
            flush=True,
        )

    query_fields = [
        "query_id",
        "status",
        "elapsed_ms",
        "planning_ms",
        "execution_ms",
        "node_count",
        "root_node_type",
        "root_plan_rows",
        "root_actual_rows_per_loop",
        "root_actual_loops",
        "root_q_error_per_loop",
        "root_q_error_total",
        "node_q_error_mean",
        "node_q_error_median",
        "node_q_error_p90",
        "node_q_error_max",
        "error",
        "file",
    ]
    node_fields = [
        "query_id",
        "node_id",
        "parent_id",
        "depth",
        "node_type",
        "join_type",
        "relation_name",
        "schema",
        "alias",
        "plan_rows",
        "plan_width",
        "startup_cost",
        "total_cost",
        "actual_startup_ms",
        "actual_total_ms",
        "actual_rows_per_loop",
        "actual_loops",
        "actual_total_rows",
        "q_error_per_loop",
        "q_error_total",
        "filter",
        "join_filter",
        "hash_cond",
        "merge_cond",
        "index_cond",
        "recheck_cond",
    ]
    write_csv(out_dir / "query_summary.csv", query_rows, query_fields)
    write_csv(out_dir / "plan_nodes.csv", node_rows, node_fields)
    (out_dir / "query_summary.json").write_text(
        json.dumps(query_rows, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    write_report(out_dir, query_rows, node_rows)
    print(json.dumps({"query_count": len(query_rows), "node_count": len(node_rows), "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
