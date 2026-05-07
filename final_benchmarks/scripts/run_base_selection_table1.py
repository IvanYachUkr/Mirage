from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import duckdb


TEST44_DIR = Path(__file__).resolve().parents[2]
ROOT = TEST44_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST44_DIR) not in sys.path:
    sys.path.insert(0, str(TEST44_DIR))
JOB_EXACT_DIR = ROOT / "benchmark" / "job_exact_v1"
if str(JOB_EXACT_DIR) not in sys.path:
    sys.path.insert(0, str(JOB_EXACT_DIR))

from exact_job_benchmark import extract_from_where, parse_alias_map, sort_key  # noqa: E402
from benchmark.cardinality_methods.scripts.run_rich_mscn_experiment import local_conditions  # noqa: E402
from run_job_queries_duckdb import _normalize_sql_for_duckdb  # noqa: E402


DEFAULT_DOCKER = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
DEFAULT_RUN = TEST44_DIR / "final_benchmarks" / "runs" / "table1_base_selection_test42_100k_exportpatch"
DEFAULT_DUCKDB = TEST44_DIR / "final_benchmarks" / "test42_100k_exportpatch" / "duckdb" / "test42_100k_exportpatch.duckdb"
DEFAULT_QUERY_ROOT = TEST44_DIR / "final_benchmarks" / "test42_100k_exportpatch" / "queries"

WORKLOAD_DIRS = {
    "job_exact": DEFAULT_QUERY_ROOT / "job_exact",
    "job_complex": DEFAULT_QUERY_ROOT / "job_complex",
    "job_light": DEFAULT_QUERY_ROOT / "job_light",
}

EST_RE = re.compile(r"~([0-9,]+) rows?")
IN_RE = re.compile(
    r"(?P<expr>\b[A-Za-z_][\w$]*\.[A-Za-z_][\w$]*\b)\s+IN\s*\((?P<body>(?:[^']|'(?:''|[^'])*')*)\)",
    re.I | re.S,
)
SINGLE_QUOTED_RE = re.compile(r"'((?:''|[^'])*)'")


def qerror(estimate: float, actual: float) -> float:
    estimate = max(float(estimate), 1.0)
    actual = max(float(actual), 1.0)
    return estimate / actual if estimate >= actual else actual / estimate


def percentile(values: list[float], p: float) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return None
    k = (len(clean) - 1) * p / 100.0
    floor = math.floor(k)
    ceil = math.ceil(k)
    if floor == ceil:
        return clean[int(k)]
    return clean[floor] * (ceil - k) + clean[ceil] * (k - floor)


def quote_sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def canonicalize_duplicate_in(condition: str) -> tuple[str, bool]:
    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        values = [m.group(1).replace("''", "'") for m in SINGLE_QUOTED_RE.finditer(match.group("body"))]
        if not values:
            return match.group(0)
        unique: list[str] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        if len(unique) == len(values):
            return match.group(0)
        changed = True
        expr = match.group("expr")
        if len(unique) == 1:
            return f"{expr} = {quote_sql_text(unique[0])}"
        return f"{expr} IN ({', '.join(quote_sql_text(value) for value in unique)})"

    return IN_RE.sub(repl, condition), changed


def sanitize_alias(alias: str, *, engine: str) -> str:
    if engine == "duckdb" and alias.lower() == "at":
        return "aka_t"
    return alias


def make_base_sql(table: str, alias: str, conditions: list[str], *, engine: str) -> tuple[str, str, bool]:
    rewritten: list[str] = []
    canonicalized = False
    alias_out = sanitize_alias(alias, engine=engine)
    for condition in conditions:
        if alias_out != alias:
            condition = re.sub(rf"\b{re.escape(alias)}\.", alias_out + ".", condition, flags=re.I)
        condition, changed = canonicalize_duplicate_in(condition)
        canonicalized = canonicalized or changed
        rewritten.append(condition)
    where = " AND ".join(f"({condition})" for condition in rewritten)
    table_sql = '"' + table.replace('"', '""') + '"'
    base = f"FROM {table_sql} AS {alias_out}"
    if where:
        base += f" WHERE {where}"
    return f"SELECT COUNT(*) {base}", f"SELECT * {base}", canonicalized


class PsqlSession:
    def __init__(self, *, docker_bin: str, container: str, db_name: str, user: str = "postgres") -> None:
        self.seq = 0
        self.proc = subprocess.Popen(
            [
                docker_bin,
                "exec",
                "-i",
                container,
                "psql",
                "-X",
                "-U",
                user,
                "-d",
                db_name,
                "-P",
                "pager=off",
                "-qAt",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def run(self, sql: str, *, timeout_s: float = 60.0) -> str:
        if self.proc.poll() is not None:
            raise RuntimeError("psql session is closed")
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.seq += 1
        sentinel = f"__MIRAGE_TABLE1_END_{self.seq}__"
        self.proc.stdin.write(sql.strip().rstrip(";") + ";\n")
        self.proc.stdin.write(f"\\echo {sentinel}\n")
        self.proc.stdin.flush()
        lines: list[str] = []
        started = time.perf_counter()
        while True:
            if time.perf_counter() - started > timeout_s:
                raise TimeoutError(f"psql timeout after {timeout_s}s")
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("psql session ended unexpectedly")
            text = line.rstrip("\n")
            if text == sentinel:
                break
            lines.append(text)
        out = "\n".join(lines).strip()
        for line in lines:
            if line.startswith(("ERROR:", "FATAL:", "PANIC:")):
                raise RuntimeError(out[:1000])
        return out

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write("\\q\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def postgres_estimate(session: PsqlSession, explain_sql: str) -> int:
    raw = session.run("EXPLAIN (FORMAT JSON) " + explain_sql, timeout_s=60)
    payload = json.loads(raw)
    return int(payload[0]["Plan"]["Plan Rows"])


def duckdb_estimate(con: duckdb.DuckDBPyConnection, explain_sql: str) -> int:
    plan = "\n".join(str(cell) for row in con.execute("EXPLAIN " + explain_sql).fetchall() for cell in row if cell is not None)
    matches = EST_RE.findall(plan)
    if not matches:
        raise RuntimeError("DuckDB EXPLAIN did not contain an estimated row count")
    return int(matches[0].replace(",", ""))


def iter_base_selections(workload_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for workload, query_dir in workload_dirs.items():
        if not query_dir.exists():
            raise FileNotFoundError(f"query directory not found for {workload}: {query_dir}")
        for sql_path in sorted(query_dir.glob("*.sql"), key=sort_key):
            query_id = sql_path.stem
            sql = _normalize_sql_for_duckdb(sql_path.read_text(encoding="utf-8").strip().rstrip(";"))
            from_clause, _ = extract_from_where(sql)
            alias_map = parse_alias_map(from_clause)
            cond_map = local_conditions(sql, alias_map)
            for alias, conditions in cond_map.items():
                if not conditions:
                    continue
                selections.append(
                    {
                        "workload": workload,
                        "query_id": query_id,
                        "table": alias_map[alias],
                        "alias": alias,
                        "predicate_count": len(conditions),
                        "conditions": conditions,
                        "source_sql_path": str(sql_path),
                    }
                )
    return selections


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for estimator in sorted({row["estimator"] for row in rows}):
        summary[estimator] = {}
        for workload in sorted({row["workload"] for row in rows}):
            vals = [
                float(row["q_error"])
                for row in rows
                if row["estimator"] == estimator and row["workload"] == workload and not row["actual_zero"]
            ]
            summary[estimator][workload] = {
                "rows": len(vals),
                "zero_actual_excluded": sum(
                    1
                    for row in rows
                    if row["estimator"] == estimator and row["workload"] == workload and row["actual_zero"]
                ),
                "canonicalized_duplicate_in_rows": sum(
                    1
                    for row in rows
                    if row["estimator"] == estimator and row["workload"] == workload and row["canonicalized_duplicate_in"]
                ),
                "median": percentile(vals, 50),
                "p90": percentile(vals, 90),
                "p95": percentile(vals, 95),
                "p99": percentile(vals, 99),
                "max": max(vals) if vals else None,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run original-JOB-style base-selection q-error extraction.")
    parser.add_argument("--docker-bin", default=DEFAULT_DOCKER)
    parser.add_argument("--container", default="pg_bench")
    parser.add_argument("--pg-db", default="imdb_test42_100k_exportpatch")
    parser.add_argument("--duckdb", type=Path, default=DEFAULT_DUCKDB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--engines", default="postgres,duckdb")
    parser.add_argument("--job-exact-dir", type=Path, default=WORKLOAD_DIRS["job_exact"])
    parser.add_argument("--job-complex-dir", type=Path, default=WORKLOAD_DIRS["job_complex"])
    parser.add_argument("--job-light-dir", type=Path, default=WORKLOAD_DIRS["job_light"])
    args = parser.parse_args()

    engines = {part.strip().lower() for part in args.engines.split(",") if part.strip()}
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    workload_dirs = {
        "job_exact": args.job_exact_dir.resolve(),
        "job_complex": args.job_complex_dir.resolve(),
        "job_light": args.job_light_dir.resolve(),
    }
    selections = iter_base_selections(workload_dirs)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    started = time.perf_counter()
    pg: PsqlSession | None = None
    duck: duckdb.DuckDBPyConnection | None = None
    try:
        if "postgres" in engines:
            pg = PsqlSession(docker_bin=args.docker_bin, container=args.container, db_name=args.pg_db)
        if "duckdb" in engines:
            duck = duckdb.connect(str(args.duckdb.resolve()), read_only=True)
            duck.execute("PRAGMA threads=4")

        for i, selection in enumerate(selections, start=1):
            if i % 100 == 0:
                print(f"processed {i}/{len(selections)} base selections", flush=True)
            for engine in sorted(engines):
                try:
                    count_sql, explain_sql, canonicalized = make_base_sql(
                        selection["table"],
                        selection["alias"],
                        list(selection["conditions"]),
                        engine=engine,
                    )
                    if engine == "postgres":
                        assert pg is not None
                        actual = int(pg.run(count_sql, timeout_s=60) or "0")
                        estimate = postgres_estimate(pg, explain_sql)
                    elif engine == "duckdb":
                        assert duck is not None
                        actual = int(duck.execute(count_sql).fetchone()[0])
                        estimate = duckdb_estimate(duck, explain_sql)
                    else:
                        raise ValueError(f"unknown engine: {engine}")
                    rows.append(
                        {
                            "estimator": engine,
                            "workload": selection["workload"],
                            "query_id": selection["query_id"],
                            "table": selection["table"],
                            "alias": selection["alias"],
                            "predicate_count": selection["predicate_count"],
                            "actual": actual,
                            "estimate": estimate,
                            "q_error": qerror(estimate, actual),
                            "bias": "over" if estimate > max(actual, 1) else ("under" if estimate < max(actual, 1) else "exact"),
                            "actual_zero": actual == 0,
                            "canonicalized_duplicate_in": canonicalized,
                            "count_sql": count_sql,
                            "explain_sql": explain_sql,
                            "source_sql_path": selection["source_sql_path"],
                        }
                    )
                except Exception as exc:
                    errors.append({**selection, "engine": engine, "error": str(exc)[:1000]})
    finally:
        if pg is not None:
            pg.close()
        if duck is not None:
            duck.close()

    csv_path = out_dir / "base_selection_qerrors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["estimator"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "run_dir": str(out_dir),
        "pg_db": args.pg_db,
        "duckdb": str(args.duckdb.resolve()),
        "workload_dirs": {name: str(path) for name, path in workload_dirs.items()},
        "elapsed_s": time.perf_counter() - started,
        "selection_count": len(selections),
        "row_count": len(rows),
        "error_count": len(errors),
        "errors": errors[:50],
        "summary": summarize(rows),
    }
    report_path = out_dir / "table1_base_selection_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
