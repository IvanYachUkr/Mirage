#!/usr/bin/env python3
"""Convert Mirage PostgreSQL plan traces into zero-shot parsed-workload JSON.

The upstream zero-shot repository expects DBGen-style parsed plan objects. This
bridge intentionally starts with the conservative feature subset that its
PostgresTrueCardDetail model requires: plan-node features, simple table nodes,
and measured runtimes. Filter/output-column parsing can be added later.

For cardinality-estimation experiments, pass ``--label-source target_actual``.
The upstream code always reads labels from ``plan_runtime / 1000``; in that
mode this bridge stores the CE target cardinality in that slot, scaled by 1000.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "Infinity", "inf", "nan", "NaN"):
        return default
    try:
        numeric = float(value)
    except Exception:
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_table_stats(table_stats_csv: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = read_csv(table_stats_csv)
    names = sorted({row["table_name"] for row in rows if row.get("table_name")})
    table_id = {name: idx for idx, name in enumerate(names)}
    reltuples = {row["table_name"]: as_float(row.get("estimated_rows"), 1.0) for row in rows}
    stats = [
        {
            "relname": name,
            "reltuples": max(1.0, reltuples.get(name, 1.0)),
            "relpages": max(1.0, reltuples.get(name, 1.0) / 100.0),
        }
        for name in names
    ]
    return stats, table_id


def node_to_plan_operator(
    node: dict[str, str],
    children: list[dict[str, Any]],
    *,
    table_id: dict[str, int],
    is_root: bool = False,
) -> dict[str, Any]:
    est_card = max(1.0, as_float(node.get("plan_rows"), 1.0))
    act_card = max(0.0, as_float(node.get("actual_total_rows"), as_float(node.get("actual_rows_per_loop"), 0.0)))
    est_children = 1.0
    act_children = 1.0
    for child in children:
        params = child["plan_parameters"]
        est_children *= max(1.0, as_float(params.get("est_card"), 1.0))
        act_children *= max(1.0, as_float(params.get("act_card"), 1.0))

    params: dict[str, Any] = {
        "op_name": node.get("node_type") or "Unknown",
        "est_card": est_card,
        "act_card": act_card,
        "est_width": 1.0,
        "workers_planned": 0.0,
        "est_children_card": est_children,
        "act_children_card": act_children,
    }
    relation = node.get("relation_name") or ""
    if relation in table_id:
        params["table"] = table_id[relation]
    if is_root:
        params["output_columns"] = [{"aggregation": "COUNT", "columns": [0]}]
        params["filter_columns"] = {
            "operator": "=",
            "column": 0,
            "literal_feature": 1.0,
            "children": [],
        }

    return {
        "plain_content": [],
        "plan_parameters": params,
        "children": children,
        "plan_runtime": 0.0,
    }


def build_plan(
    rows_by_parent: dict[str, list[dict[str, str]]],
    node: dict[str, str],
    *,
    table_id: dict[str, int],
    is_root: bool = False,
) -> dict[str, Any]:
    children = [
        build_plan(rows_by_parent, child, table_id=table_id)
        for child in rows_by_parent.get(node["node_id"], [])
    ]
    return node_to_plan_operator(node, children, table_id=table_id, is_root=is_root)


def label_from_summary(summary: dict[str, str], label_source: str) -> float:
    if label_source == "execution_ms":
        return max(0.001, as_float(summary.get("execution_ms"), 0.001))
    if label_source == "target_actual":
        return as_float(summary.get("target_actual"), 0.0) * 1000.0
    if label_source == "actual_count":
        value = summary.get("actual_count") or summary.get("count") or summary.get("root_actual_rows")
        return as_float(value, 0.0) * 1000.0
    raise ValueError(f"Unsupported label source: {label_source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, help="Directory containing query_summary.csv and plan_nodes.csv.")
    parser.add_argument("--table-stats-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--label-source",
        choices=["execution_ms", "target_actual", "actual_count"],
        default="execution_ms",
        help=(
            "Upstream zero-shot labels are read from plan_runtime / 1000. "
            "Use execution_ms for original cost/runtime experiments, or "
            "target_actual for cardinality-estimation experiments."
        ),
    )
    parser.add_argument(
        "--skip-nonpositive-label",
        action="store_true",
        help="Skip rows whose selected label is <= 0, which is required for q-error training.",
    )
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir).resolve()
    out_path = Path(args.out).resolve()
    table_stats, table_id = build_table_stats(Path(args.table_stats_csv).resolve())
    summaries = {
        row["query_id"]: row
        for row in read_csv(trace_dir / "query_summary.csv")
        if row.get("status") == "ok"
    }
    plan_rows = [
        row for row in read_csv(trace_dir / "plan_nodes.csv")
        if row.get("query_id") in summaries
    ]
    by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in plan_rows:
        by_query[row["query_id"]].append(row)

    parsed_plans: list[dict[str, Any]] = []
    for query_id in sorted(by_query):
        rows = by_query[query_id]
        rows_by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
        roots: list[dict[str, str]] = []
        for row in rows:
            parent_id = row.get("parent_id") or ""
            if parent_id:
                rows_by_parent[parent_id].append(row)
            else:
                roots.append(row)
        if len(roots) != 1:
            continue
        label = label_from_summary(summaries[query_id], args.label_source)
        if args.skip_nonpositive_label and label <= 0:
            continue
        plan = build_plan(rows_by_parent, roots[0], table_id=table_id, is_root=True)
        plan["plan_runtime"] = label
        parsed_plans.append(plan)
        if args.max_queries is not None and len(parsed_plans) >= args.max_queries:
            break

    payload = {
        "parsed_plans": parsed_plans,
        "database_stats": {
            "table_stats": table_stats,
            "column_stats": [
                {
                    "avg_width": 8.0,
                    "correlation": 0.0,
                    "data_type": "integer",
                    "n_distinct": 1000.0,
                    "null_frac": 0.0,
                }
            ],
            "run_kwars": {},
        },
        "run_kwargs": {
            "source": str(trace_dir),
            "format": "mirage_minimal_postgres_trace_v1",
            "label_source": args.label_source,
            "skip_nonpositive_label": bool(args.skip_nonpositive_label),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"parsed_plans": len(parsed_plans), "tables": len(table_stats), "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
