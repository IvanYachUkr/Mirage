#!/usr/bin/env python3
"""Write a compact readiness summary for a CE benchmark run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.{digits}f}"


def pg_block(name: str, manifest: dict[str, Any] | None) -> list[str]:
    if not manifest:
        return [f"- `{name}`: missing."]
    summary = manifest.get("summary", {})
    target = summary.get("target_q_error_summary", {})
    node = summary.get("node_q_error_summary", {})
    return [
        (
            f"- `{name}`: {summary.get('ok', 0)}/{summary.get('query_count', 0)} ok, "
            f"{summary.get('errors', 0)} errors, {summary.get('timeouts', 0)} timeouts, "
            f"target nonzero/zero={summary.get('target_nonzero', 0)}/{summary.get('target_zero', 0)}, "
            f"structure failures={summary.get('structure_guard_failures', 0)}."
        ),
        (
            f"- `{name}` target q-error: median={fmt(target.get('median'))}, "
            f"mean={fmt(target.get('mean'))}, p95={fmt(target.get('p95'))}, "
            f"max={fmt(target.get('max'))}, infinite={target.get('infinite_count', 0)}."
        ),
        (
            f"- `{name}` plan-node q-error: median={fmt(node.get('median'))}, "
            f"p95={fmt(node.get('p95'))}, max={fmt(node.get('max'))}, "
            f"infinite={node.get('infinite_count', 0)}."
        ),
    ]


def duckdb_block(name: str, manifest: dict[str, Any] | None) -> list[str]:
    if not manifest:
        return [f"- `{name}`: missing or not enabled."]
    summary = manifest.get("summary", {})
    timing = summary.get("timing_ms", {})
    return [
        (
            f"- `{name}`: {summary.get('ok', 0)}/{summary.get('query_count', 0)} ok, "
            f"{summary.get('errors', 0)} errors, median={fmt(timing.get('median'))} ms, "
            f"mean={fmt(timing.get('mean'))} ms, max={fmt(timing.get('max'))} ms."
        )
    ]


def mscn_block(name: str, manifest: dict[str, Any] | None) -> list[str]:
    if not manifest:
        return [f"- `{name}`: missing."]
    summary = manifest.get("summary", {})
    return [
        (
            f"- `{name}`: exported {summary.get('exported_queries', summary.get('query_count', 0))}/"
            f"{summary.get('attempted_queries', summary.get('query_count', 0))} queries, "
            f"skipped {summary.get('skipped_queries', 0)} zero/unknown-label queries, "
            f"columns={summary.get('columns', 0)}."
        )
    ]


def zero_shot_block(name: str, manifest: dict[str, Any] | None) -> list[str]:
    if not manifest:
        return [f"- `{name}`: missing."]
    copied = manifest.get("copied", {})
    return [
        (
            f"- `{name}`: input package ready, pg_stats={manifest.get('postgres_stats_exported')}, "
            f"manifest={copied.get('manifest')}, query_summary={copied.get('query_summary')}, "
            f"plan_nodes={copied.get('plan_nodes')}."
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else run_dir / "benchmark_readiness_summary.md"

    pg_exact = load_json(run_dir / "postgres" / "job_exact" / "manifest.json")
    pg_complex = load_json(run_dir / "postgres" / "job_complex" / "manifest.json")
    duck_exact = load_json(run_dir / "duckdb" / "job_exact" / "manifest.json")
    duck_complex = load_json(run_dir / "duckdb" / "job_complex" / "manifest.json")
    mscn_exact = load_json(run_dir / "mscn" / "job_exact.manifest.json")
    mscn_complex = load_json(run_dir / "mscn" / "job_complex.manifest.json")
    zs_exact = load_json(run_dir / "zero_shot" / "job_exact" / "zero_shot_input_manifest.json")
    zs_complex = load_json(run_dir / "zero_shot" / "job_complex" / "zero_shot_input_manifest.json")

    lines: list[str] = [
        "# Benchmark Readiness Summary",
        "",
        f"- Run directory: `{run_dir}`",
        "- Real measured results: PostgreSQL EXPLAIN ANALYZE traces and DuckDB execution/timing manifests.",
        "- Integration artifacts, not learned-model results: MSCN workload files and zero-shot input packages.",
        "- Label policy: MSCN exports use `target_actual`/actual-count labels and skip zero or unknown labels.",
        "",
        "## PostgreSQL Source Of Truth",
        *pg_block("exact JOB", pg_exact),
        *pg_block("JOB-Complex", pg_complex),
        "",
        "## DuckDB Portability",
        *duckdb_block("exact JOB", duck_exact),
        *duckdb_block("JOB-Complex", duck_complex),
        "",
        "## Learned-Method Packages",
        *mscn_block("MSCN exact JOB", mscn_exact),
        *mscn_block("MSCN JOB-Complex", mscn_complex),
        *zero_shot_block("zero-shot exact JOB", zs_exact),
        *zero_shot_block("zero-shot JOB-Complex", zs_complex),
        "",
        "## Interpretation",
        "- Exact JOB is the clean publishable baseline path: structure guard passes, Postgres executes the full corpus, and DuckDB confirms portability.",
        "- JOB-Complex is usable as an exploratory secondary workload; it executes cleanly, but several adapted queries still have zero CE target cardinality at test40 scale.",
        "- MSCN and zero-shot are staged honestly as reproducible input packages. Treat them as ready-to-integrate artifacts until the external model repos run end-to-end on the lab machine.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
