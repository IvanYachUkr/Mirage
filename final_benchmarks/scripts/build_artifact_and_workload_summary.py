"""Build paper-facing artifact and workload compatibility summaries.

The query-structure audit intentionally normalizes only literal constants
(quoted strings and numeric constants). If the adapted query removed a join,
changed an operator, changed table/alias structure, or added/removed a
predicate, the canonical structure should no longer match.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_BENCH = SCRIPT_DIR.parent
WORKSPACE = FINAL_BENCH.parents[1]
EXPORT_ROOT = FINAL_BENCH / "test42_100k_exportpatch"
SCHEMA_DIR = EXPORT_ROOT / "imdb_schema"
QUERY_ROOT = EXPORT_ROOT / "queries"
OUT_DIR = FINAL_BENCH / "final_paper_results" / "artifact_workload_summary"

SOURCE_QUERIES = {
    "job_exact": WORKSPACE / "benchmark" / "job_exact_v1" / "original_queries",
    "job_complex": WORKSPACE / "benchmark" / "job_complex_v1" / "original_queries",
    "job_light": (
        WORKSPACE
        / "benchmark"
        / "cardinality_methods"
        / "sources"
        / "deepdb-public"
        / "benchmarks"
        / "job-light"
        / "sql"
        / "job_light_queries.sql"
    ),
}

ADAPTED_QUERIES = {
    "job_exact": QUERY_ROOT / "job_exact",
    "job_complex": QUERY_ROOT / "job_complex",
    "job_light": QUERY_ROOT / "job_light",
}

CE_SUMMARY_RUN = (
    WORKSPACE
    / "benchmark"
    / "cardinality_methods"
    / "runs"
    / "test42_100k_ce_all3_exportpatch_v3_20260504"
    / "postgres"
)

WORKLOAD_LABELS = {
    "job_light": "JOB-Light",
    "job_exact": "JOB",
    "job_complex": "JOB-Complex",
}

REQUIRED_FIELDS = {
    "title": ("id", "title", "kind_id", "production_year"),
    "name": ("id", "name"),
    "company_name": ("id", "name"),
    "char_name": ("id", "name"),
    "keyword": ("id", "keyword"),
    "cast_info": ("id", "person_id", "movie_id", "role_id"),
    "movie_companies": ("id", "movie_id", "company_id", "company_type_id"),
    "movie_keyword": ("id", "movie_id", "keyword_id"),
    "movie_info": ("id", "movie_id", "info_type_id", "info"),
    "movie_info_idx": ("id", "movie_id", "info_type_id", "info"),
    "person_info": ("id", "person_id", "info_type_id", "info"),
}


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        yield from csv.DictReader(fh)


def load_export_manifest() -> dict:
    with (SCHEMA_DIR / "export_manifest.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def table_counts_from_manifest(manifest: dict) -> dict[str, int]:
    return {entry["table"]: int(entry["rows"]) for entry in manifest["tables"]}


def load_id_set(table: str) -> set[int]:
    ids: set[int] = set()
    for row in read_csv_rows(SCHEMA_DIR / f"{table}.csv"):
        value = row.get("id", "")
        if value:
            ids.add(int(value))
    return ids


def duplicate_pk_count(table: str) -> int:
    seen: set[str] = set()
    dupes = 0
    for row in read_csv_rows(SCHEMA_DIR / f"{table}.csv"):
        pk = row.get("id", "")
        if not pk:
            continue
        if pk in seen:
            dupes += 1
        else:
            seen.add(pk)
    return dupes


def empty_required_field_count(table: str) -> int:
    required = REQUIRED_FIELDS.get(table, ())
    if not required:
        return 0
    empties = 0
    for row in read_csv_rows(SCHEMA_DIR / f"{table}.csv"):
        for field in required:
            if not (row.get(field) or "").strip():
                empties += 1
    return empties


def title_stats() -> dict:
    kind_counts: Counter[str] = Counter()
    movie_years: list[int] = []
    all_years: list[int] = []
    for row in read_csv_rows(SCHEMA_DIR / "title.csv"):
        kind_id = row.get("kind_id") or ""
        kind_counts[kind_id] += 1
        year_raw = row.get("production_year") or ""
        if year_raw:
            year = int(year_raw)
            all_years.append(year)
            if kind_id == "1":
                movie_years.append(year)
    return {
        "kind_counts": dict(kind_counts),
        "movie_count": kind_counts.get("1", 0),
        "movie_year_min": min(movie_years) if movie_years else None,
        "movie_year_max": max(movie_years) if movie_years else None,
        "all_title_year_min": min(all_years) if all_years else None,
        "all_title_year_max": max(all_years) if all_years else None,
    }


def count_fk_violations(table: str, checks: list[tuple[str, set[int], bool]]) -> int:
    violations = 0
    for row in read_csv_rows(SCHEMA_DIR / f"{table}.csv"):
        for field, target_ids, optional in checks:
            raw = row.get(field, "")
            if optional and not raw:
                continue
            if not raw or int(raw) not in target_ids:
                violations += 1
    return violations


def integrity_stats(counts: dict[str, int]) -> dict:
    id_sets = {
        "title": load_id_set("title"),
        "name": load_id_set("name"),
        "char_name": load_id_set("char_name"),
        "company_name": load_id_set("company_name"),
        "keyword": load_id_set("keyword"),
        "info_type": load_id_set("info_type"),
        "role_type": load_id_set("role_type"),
        "company_type": load_id_set("company_type"),
        "link_type": load_id_set("link_type"),
        "kind_type": load_id_set("kind_type"),
        "comp_cast_type": load_id_set("comp_cast_type"),
    }

    fk_checks = {
        "cast_info": [
            ("person_id", id_sets["name"], False),
            ("movie_id", id_sets["title"], False),
            ("person_role_id", id_sets["char_name"], True),
            ("role_id", id_sets["role_type"], False),
        ],
        "movie_companies": [
            ("movie_id", id_sets["title"], False),
            ("company_id", id_sets["company_name"], False),
            ("company_type_id", id_sets["company_type"], False),
        ],
        "movie_keyword": [
            ("movie_id", id_sets["title"], False),
            ("keyword_id", id_sets["keyword"], False),
        ],
        "movie_info": [
            ("movie_id", id_sets["title"], False),
            ("info_type_id", id_sets["info_type"], False),
        ],
        "movie_info_idx": [
            ("movie_id", id_sets["title"], False),
            ("info_type_id", id_sets["info_type"], False),
        ],
        "movie_link": [
            ("movie_id", id_sets["title"], False),
            ("linked_movie_id", id_sets["title"], False),
            ("link_type_id", id_sets["link_type"], False),
        ],
        "aka_title": [
            ("movie_id", id_sets["title"], False),
            ("kind_id", id_sets["kind_type"], False),
            ("episode_of_id", id_sets["title"], True),
        ],
        "aka_name": [("person_id", id_sets["name"], False)],
        "person_info": [
            ("person_id", id_sets["name"], False),
            ("info_type_id", id_sets["info_type"], False),
        ],
        "complete_cast": [
            ("movie_id", id_sets["title"], False),
            ("subject_id", id_sets["comp_cast_type"], False),
            ("status_id", id_sets["comp_cast_type"], False),
        ],
    }

    duplicate_pks = sum(duplicate_pk_count(table) for table in counts)
    empty_required = sum(empty_required_field_count(table) for table in counts)
    fk_violations_by_table = {
        table: count_fk_violations(table, checks) for table, checks in fk_checks.items()
    }
    return {
        "duplicate_primary_keys": duplicate_pks,
        "empty_required_fields_checked": empty_required,
        "foreign_key_violations_checked": sum(fk_violations_by_table.values()),
        "foreign_key_violations_by_table": fk_violations_by_table,
    }


def strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n\r]*", " ", sql)
    return sql


def split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
    in_string = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        buf.append(ch)
        if ch == "'":
            if in_string and i + 1 < len(sql) and sql[i + 1] == "'":
                buf.append(sql[i + 1])
                i += 2
                continue
            in_string = not in_string
        elif ch == ";" and not in_string:
            statement = "".join(buf).strip()
            if statement:
                statements.append(statement)
            buf = []
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def normalize_query_structure(sql: str) -> str:
    sql = strip_comments(sql).strip().rstrip(";")
    sql = re.sub(r"'(?:''|[^'])*'", "__STR__", sql)
    sql = re.sub(r"\b\d+(?:\.\d+)?\b", "__NUM__", sql)
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower().strip()


def load_source_queries(workload: str) -> dict[str, str]:
    source = SOURCE_QUERIES[workload]
    if workload == "job_light":
        statements = split_sql_statements(source.read_text(encoding="utf-8", errors="replace"))
        return {f"{idx:02d}": statement for idx, statement in enumerate(statements, start=1)}
    return {
        path.stem: path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(source.glob("*.sql"))
    }


def load_adapted_queries(workload: str) -> dict[str, str]:
    source = ADAPTED_QUERIES[workload]
    return {
        path.stem: path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(source.glob("*.sql"))
    }


def load_query_summary(workload: str) -> dict[str, dict]:
    path = CE_SUMMARY_RUN / workload / "query_summary.csv"
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row["query_id"]] = row
    return rows


def audit_queries() -> tuple[list[dict], list[dict]]:
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for workload, label in WORKLOAD_LABELS.items():
        source = load_source_queries(workload)
        adapted = load_adapted_queries(workload)
        query_summary = load_query_summary(workload)
        all_ids = sorted(set(source) | set(adapted))

        for query_id in all_ids:
            source_sql = source.get(query_id)
            adapted_sql = adapted.get(query_id)
            summary = query_summary.get(query_id, {})
            source_norm = normalize_query_structure(source_sql) if source_sql else ""
            adapted_norm = normalize_query_structure(adapted_sql) if adapted_sql else ""
            structure_match = bool(source_sql and adapted_sql and source_norm == adapted_norm)
            status_ok = (summary.get("status") == "ok") if summary else False
            try:
                target_actual = float(summary.get("target_actual", "nan"))
            except ValueError:
                target_actual = float("nan")
            nonzero = bool(status_ok and target_actual > 0)
            detail_rows.append(
                {
                    "workload": label,
                    "workload_key": workload,
                    "query_id": query_id,
                    "source_present": source_sql is not None,
                    "adapted_present": adapted_sql is not None,
                    "structure_match_after_literal_normalization": structure_match,
                    "valid_sql_status_ok": status_ok,
                    "target_actual": target_actual if target_actual == target_actual else "",
                    "nonzero_result": nonzero,
                    "source_structure": source_norm,
                    "adapted_structure": adapted_norm,
                }
            )

        workload_rows = [row for row in detail_rows if row["workload_key"] == workload]
        source_count = len(source)
        adapted_count = len(adapted)
        structure_count = sum(
            1 for row in workload_rows if row["structure_match_after_literal_normalization"]
        )
        valid_count = sum(1 for row in workload_rows if row["valid_sql_status_ok"])
        nonzero_count = sum(1 for row in workload_rows if row["nonzero_result"])
        included = sum(
            1
            for row in workload_rows
            if row["adapted_present"]
            and row["structure_match_after_literal_normalization"]
            and row["valid_sql_status_ok"]
            and row["nonzero_result"]
        )
        summary_rows.append(
            {
                "workload": label,
                "original_query_templates": source_count,
                "instantiated_queries": adapted_count,
                "structure_preserved": structure_count,
                "valid_sql": valid_count,
                "nonzero_result": nonzero_count,
                "included_in_benchmark": included,
            }
        )

    return detail_rows, summary_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    lines = []
    lines.append("| " + " | ".join(header for header, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for _, key in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def format_count_stat(value: float) -> str:
    if value != value:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def build_query_result_size_rows(detail_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for workload in ("JOB-Light", "JOB", "JOB-Complex"):
        actuals = [
            float(row["target_actual"])
            for row in detail_rows
            if row["workload"] == workload and row["nonzero_result"]
        ]
        rows.append(
            {
                "workload": workload,
                "queries": len(actuals),
                "min_actual_cardinality": format_count_stat(min(actuals) if actuals else float("nan")),
                "median_actual_cardinality": format_count_stat(percentile(actuals, 0.50)),
                "p90_actual_cardinality": format_count_stat(percentile(actuals, 0.90)),
                "p95_actual_cardinality": format_count_stat(percentile(actuals, 0.95)),
                "p99_actual_cardinality": format_count_stat(percentile(actuals, 0.99)),
                "max_actual_cardinality": format_count_stat(max(actuals) if actuals else float("nan")),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_export_manifest()
    counts = table_counts_from_manifest(manifest)
    title = title_stats()
    integrity = integrity_stats(counts)

    total_exported_rows = sum(counts.values())
    db_summary_rows = [
        {"component": "Primary movies (title.kind = movie)", "count": title["movie_count"]},
        {"component": "Title rows", "count": counts["title"]},
        {"component": "People / names", "count": counts["name"]},
        {"component": "Companies", "count": counts["company_name"]},
        {"component": "Characters", "count": counts["char_name"]},
        {"component": "Cast / role rows", "count": counts["cast_info"]},
        {"component": "Keywords", "count": counts["keyword"]},
        {"component": "Movie-keyword links", "count": counts["movie_keyword"]},
        {"component": "Movie-company links", "count": counts["movie_companies"]},
        {"component": "Movie info rows", "count": counts["movie_info"]},
        {"component": "Movie info index rows", "count": counts["movie_info_idx"]},
        {"component": "Person info rows", "count": counts["person_info"]},
        {"component": "AKA title rows", "count": counts["aka_title"]},
        {"component": "AKA name rows", "count": counts["aka_name"]},
        {"component": "Movie links", "count": counts["movie_link"]},
        {"component": "Complete-cast rows", "count": counts["complete_cast"]},
        {"component": "Total exported rows", "count": total_exported_rows},
        {
            "component": "Movie years covered",
            "count": f"{title['movie_year_min']}-{title['movie_year_max']}",
        },
    ]

    integrity_rows = [
        {"property": "Duplicate primary keys", "value": integrity["duplicate_primary_keys"]},
        {
            "property": "Foreign-key violations (checked core links)",
            "value": integrity["foreign_key_violations_checked"],
        },
        {
            "property": "Empty required fields (checked core columns)",
            "value": integrity["empty_required_fields_checked"],
        },
    ]

    detail_rows, workload_rows = audit_queries()
    result_size_rows = build_query_result_size_rows(detail_rows)

    write_csv(OUT_DIR / "generated_database_summary.csv", db_summary_rows)
    write_csv(OUT_DIR / "generated_database_integrity_summary.csv", integrity_rows)
    write_csv(
        OUT_DIR / "workload_compatibility_summary.csv",
        workload_rows,
        [
            "workload",
            "original_query_templates",
            "instantiated_queries",
            "structure_preserved",
            "valid_sql",
            "nonzero_result",
            "included_in_benchmark",
        ],
    )
    write_csv(
        OUT_DIR / "query_structure_audit.csv",
        detail_rows,
        [
            "workload",
            "query_id",
            "source_present",
            "adapted_present",
            "structure_match_after_literal_normalization",
            "valid_sql_status_ok",
            "target_actual",
            "nonzero_result",
            "source_structure",
            "adapted_structure",
        ],
    )
    write_csv(
        OUT_DIR / "query_result_size_summary.csv",
        result_size_rows,
        [
            "workload",
            "queries",
            "min_actual_cardinality",
            "median_actual_cardinality",
            "p90_actual_cardinality",
            "p95_actual_cardinality",
            "p99_actual_cardinality",
            "max_actual_cardinality",
        ],
    )

    write_markdown_table(
        OUT_DIR / "generated_database_summary.md",
        db_summary_rows,
        [("Component", "component"), ("Count / value", "count")],
    )
    write_markdown_table(
        OUT_DIR / "generated_database_integrity_summary.md",
        integrity_rows,
        [("Property", "property"), ("Value", "value")],
    )
    write_markdown_table(
        OUT_DIR / "workload_compatibility_summary.md",
        workload_rows,
        [
            ("Workload", "workload"),
            ("Original templates", "original_query_templates"),
            ("Instantiated", "instantiated_queries"),
            ("Structure preserved", "structure_preserved"),
            ("Valid SQL", "valid_sql"),
            ("Nonzero result", "nonzero_result"),
            ("Included", "included_in_benchmark"),
        ],
    )
    write_markdown_table(
        OUT_DIR / "query_result_size_summary.md",
        result_size_rows,
        [
            ("Workload", "workload"),
            ("Queries", "queries"),
            ("Min actual", "min_actual_cardinality"),
            ("Median actual", "median_actual_cardinality"),
            ("P90 actual", "p90_actual_cardinality"),
            ("P95 actual", "p95_actual_cardinality"),
            ("P99 actual", "p99_actual_cardinality"),
            ("Max actual", "max_actual_cardinality"),
        ],
    )

    structure_failures = [
        row for row in detail_rows if not row["structure_match_after_literal_normalization"]
    ]
    nonzero_failures = [row for row in detail_rows if not row["nonzero_result"]]
    audit = {
        "dataset": manifest["source_db"],
        "generated_database": {
            "summary": db_summary_rows,
            "integrity": integrity_rows,
            "title_kind_counts": title["kind_counts"],
            "foreign_key_violations_by_table": integrity["foreign_key_violations_by_table"],
        },
        "workload_compatibility": workload_rows,
        "query_result_size_summary": result_size_rows,
        "query_structure_policy": (
            "SQL structures are compared after normalizing quoted string "
            "literals and numeric constants only."
        ),
        "structure_failure_count": len(structure_failures),
        "nonzero_failure_count": len(nonzero_failures),
        "structure_failures": [
            {
                "workload": row["workload"],
                "query_id": row["query_id"],
                "source_present": row["source_present"],
                "adapted_present": row["adapted_present"],
            }
            for row in structure_failures
        ],
        "nonzero_failures": [
            {
                "workload": row["workload"],
                "query_id": row["query_id"],
                "target_actual": row["target_actual"],
            }
            for row in nonzero_failures
        ],
    }
    (OUT_DIR / "artifact_workload_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    readme = """# Artifact and Workload Summary

This folder contains paper-facing summary tables for the final Test42 100K
benchmark instance and its JOB-family query instantiations.

The query-structure audit compares original templates to final adapted SQL after
normalizing only quoted string literals and numeric constants. Tables, aliases,
joins, operators, predicate positions, aggregate shape, and SQL skeleton must
therefore match exactly.

`query_result_size_summary.csv/.md` reports the distribution of actual
cardinalities for the final nonzero adapted queries. It is intended to show that
the JOB-family instantiations are not merely valid, but span nontrivial result
sizes.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
