#!/usr/bin/env python3
"""Export adapted JOB/JOB-Complex SQL into MSCN workload files.

This produces the CSV/bitmap shape expected by the historical
`learnedcardinalities` implementation. String predicates are encoded as stable
numeric category IDs per column so the original MSCN numeric-only loader can
ingest Mirage/JOB predicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXACT_JOB_DIR = ROOT / "benchmark" / "job_exact_v1"
if str(EXACT_JOB_DIR) not in sys.path:
    sys.path.insert(0, str(EXACT_JOB_DIR))

from exact_job_benchmark import (  # noqa: E402
    Slot,
    extract_from_where,
    extract_slots,
    parse_alias_map,
    sort_key,
    sql_unquote,
)


JOIN_RE = re.compile(
    r"\b(?P<a>[A-Za-z_][\w$]*)\.(?P<ac>[A-Za-z_][\w$]*)\s*=\s*"
    r"(?P<b>[A-Za-z_][\w$]*)\.(?P<bc>[A-Za-z_][\w$]*)\b"
)


def stable_category_id(column: str, value: str) -> int:
    digest = hashlib.sha1(f"{column}\0{value}".encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16)


def literal_to_number(slot: Slot, literal: str) -> float:
    if not literal.startswith("'"):
        try:
            return float(literal)
        except Exception:
            return float(stable_category_id(slot.expr, literal))
    text = sql_unquote(literal)
    try:
        return float(text)
    except Exception:
        return float(stable_category_id(slot.expr, text))


def operator_for_mscn(slot: Slot) -> str:
    op = slot.operator.upper().replace("<>", "!=")
    if slot.kind == "like":
        return "LIKE" if "NOT" not in op else "NOT LIKE"
    if slot.kind == "in":
        return "!=" if "NOT" in op else "="
    return op


def predicates_for_slot(slot: Slot) -> list[tuple[str, str, float]]:
    if slot.kind == "between" and len(slot.literals) == 2:
        return [
            (slot.expr, ">=", literal_to_number(slot, slot.literals[0].sql_text)),
            (slot.expr, "<=", literal_to_number(slot, slot.literals[1].sql_text)),
        ]
    op = operator_for_mscn(slot)
    return [(slot.expr, op, literal_to_number(slot, literal.sql_text)) for literal in slot.literals]


def extract_query_features(sql: str) -> tuple[list[str], list[str], list[tuple[str, str, float]]]:
    from_clause, where_clause = extract_from_where(sql)
    alias_map = parse_alias_map(from_clause)
    tables = [
        f"{table} {alias}"
        for alias, table in sorted(alias_map.items())
    ]

    joins: list[str] = []
    seen_joins: set[str] = set()
    for match in JOIN_RE.finditer(where_clause):
        a = match.group("a")
        b = match.group("b")
        if a.lower() == b.lower():
            continue
        if a.lower() not in alias_map or b.lower() not in alias_map:
            continue
        join = f"{a}.{match.group('ac')}={b}.{match.group('bc')}"
        reverse = f"{b}.{match.group('bc')}={a}.{match.group('ac')}"
        if join in seen_joins or reverse in seen_joins:
            continue
        seen_joins.add(join)
        joins.append(join)

    predicates: list[tuple[str, str, float]] = []
    for slot in extract_slots(sql):
        predicates.extend(predicates_for_slot(slot))
    return tables, joins, predicates


LABEL_COLUMNS = ("target_actual", "actual_count", "actual", "result_value", "root_actual_rows")


def parse_positive_label(row: dict[str, Any]) -> tuple[int | None, str]:
    for column in LABEL_COLUMNS:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            numeric = float(value)
        except Exception:
            return None, f"non_numeric_{column}"
        if numeric <= 0:
            return None, f"non_positive_{column}"
        return max(1, int(round(numeric))), column
    return None, "missing_label"


def load_labels(path: Path | None) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    if path is None:
        return {}, {}, {}
    labels: dict[str, int] = {}
    skipped: dict[str, str] = {}
    sources: dict[str, str] = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("queries", payload if isinstance(payload, list) else [])
        for row in rows:
            qid = str(row.get("query_id") or row.get("query") or "")
            label, reason = parse_positive_label(row)
            if qid and label is not None:
                labels[qid] = label
                sources[qid] = reason
            elif qid:
                skipped[qid] = reason
        return labels, skipped, sources
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = str(row.get("query_id") or row.get("query") or "")
            label, reason = parse_positive_label(row)
            if qid and label is not None:
                labels[qid] = label
                sources[qid] = reason
            elif qid:
                skipped[qid] = reason
    return labels, skipped, sources


def write_bitmaps(path: Path, table_counts: list[int], num_samples: int, mode: str) -> None:
    num_bytes = int((num_samples + 7) >> 3)
    if mode == "ones":
        bits = np.ones(num_bytes * 8, dtype=np.uint8)
    else:
        bits = np.zeros(num_bytes * 8, dtype=np.uint8)
    packed = np.packbits(bits[: num_bytes * 8]).tobytes()
    with path.open("wb") as handle:
        for count in table_counts:
            handle.write(struct.pack("<I", int(count)))
            for _ in range(int(count)):
                handle.write(packed[:num_bytes])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--labels", default=None, help="CSV/JSON with query_id/query and actual_count columns.")
    parser.add_argument("--out-prefix", required=True, help="Output path without extension, e.g. out/job_exact.")
    parser.add_argument("--query-ids", default=None)
    parser.add_argument("--num-materialized-samples", type=int, default=1000)
    parser.add_argument("--bitmap-mode", choices=["ones", "zeros"], default="ones")
    args = parser.parse_args()

    query_dir = Path(args.query_dir).resolve()
    out_prefix = Path(args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    label_path = Path(args.labels).resolve() if args.labels else None
    labels, skipped_labels, label_sources = load_labels(label_path)
    labels_are_required = label_path is not None
    selected = {item.strip() for item in args.query_ids.split(",") if item.strip()} if args.query_ids else None
    query_paths = [
        path for path in sorted(query_dir.glob("*.sql"), key=sort_key)
        if selected is None or path.stem in selected
    ]
    if not query_paths:
        raise FileNotFoundError(f"No SQL files found in {query_dir}")

    rows: list[list[str]] = []
    sql_lines: list[str] = []
    table_counts: list[int] = []
    column_values: dict[str, list[float]] = {}
    metadata: list[dict[str, Any]] = []
    skipped_metadata: list[dict[str, Any]] = []
    for path in query_paths:
        qid = path.stem
        sql = path.read_text(encoding="utf-8").strip().rstrip(";")
        tables, joins, predicates = extract_query_features(sql)
        label = labels.get(qid)
        if labels_are_required and label is None:
            skipped_metadata.append(
                {
                    "query_id": qid,
                    "reason": skipped_labels.get(qid, "missing_label"),
                    "sql_path": str(path),
                }
            )
            continue
        if label is None:
            label = 1
        predicate_parts: list[str] = []
        for column, op, value in predicates:
            predicate_parts.extend([column, op, str(float(value))])
            column_values.setdefault(column, []).append(float(value))
        rows.append([
            ",".join(tables),
            ",".join(joins),
            ",".join(predicate_parts),
            str(max(1, int(label))),
        ])
        sql_lines.append(sql + ";")
        table_counts.append(len(tables))
        metadata.append(
            {
                "query_id": qid,
                "table_count": len(tables),
                "join_count": len(joins),
                "predicate_count": len(predicates),
                "label": max(1, int(label)),
                "label_source": label_sources.get(qid, "synthetic_default"),
                "sql_path": str(path),
            }
        )

    if not metadata:
        raise RuntimeError(
            "No MSCN workload rows were exported. All labels were zero/unknown, "
            "or no queries matched the selected corpus."
        )

    with out_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="#", lineterminator="\n")
        writer.writerows(rows)
    out_prefix.with_suffix(".sql").write_text("\n".join(sql_lines) + "\n", encoding="utf-8")
    write_bitmaps(
        out_prefix.with_suffix(".bitmaps"),
        table_counts,
        num_samples=int(args.num_materialized_samples),
        mode=args.bitmap_mode,
    )

    minmax_path = out_prefix.with_suffix(".column_min_max_vals.csv")
    with minmax_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["column", "min", "max"])
        for column, values in sorted(column_values.items()):
            low = min(values)
            high = max(values)
            if low == high:
                high = low + 1.0
            writer.writerow([column, low, high])
    # Historical MSCN scripts expect this filename in their working `data/`
    # directory. Keep emitting it for single-workload runs, but manifests point
    # at the workload-specific file so exact JOB and JOB-Complex cannot clobber
    # each other's column ranges in shared output folders.
    legacy_minmax_path = out_prefix.parent / "column_min_max_vals.csv"
    if legacy_minmax_path != minmax_path:
        legacy_minmax_path.write_text(minmax_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "method": "mscn_export",
        "query_dir": str(query_dir),
        "labels": None if label_path is None else str(label_path),
        "out_prefix": str(out_prefix),
        "column_min_max_vals": str(minmax_path),
        "num_materialized_samples": int(args.num_materialized_samples),
        "bitmap_mode": args.bitmap_mode,
        "warning": (
            "Bitmap mode is a plumbing aid. Use real materialized-sample bitmaps "
            "before treating MSCN numbers as a scientific baseline."
        ),
        "summary": {
            "query_count": len(metadata),
            "attempted_queries": len(query_paths),
            "exported_queries": len(metadata),
            "skipped_queries": len(skipped_metadata),
            "label_source_count": len(label_sources),
            "columns": len(column_values),
            "median_tables": float(np.median(table_counts)) if table_counts else None,
        },
        "queries": metadata,
        "skipped": skipped_metadata,
    }
    out_prefix.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
