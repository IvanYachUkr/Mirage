"""Build raw per-query CSVs requested for paper plotting.

The output of this script is intentionally plot-oriented and compact. It does
not copy full generated database tables, run histories, virtual environments, or
environment files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable


SCRIPT = Path(__file__).resolve()
BENCH = SCRIPT.parents[1]
TEST44 = SCRIPT.parents[2]
WORKSPACE = SCRIPT.parents[3]
FINAL = BENCH / "final_paper_results"
OUT = FINAL / "paper_requested_raw_data"
HANDOFF = BENCH / "paper_handoff_20260505"
HANDOFF_ZIP = BENCH / "paper_handoff_20260505.zip"

POSTGRES_CE = (
    WORKSPACE
    / "benchmark"
    / "cardinality_methods"
    / "runs"
    / "test42_100k_ce_all3_exportpatch_v3_20260504"
    / "postgres"
)
MSCN_CARD = (
    WORKSPACE
    / "benchmark"
    / "cardinality_methods"
    / "runs"
    / "test42_100k_paper_mscn_cardinality_100k_100ep_3seed_20260504_exact3600s_quotefix"
    / "model_seeds"
)
ZEROSHOT_COST = (
    WORKSPACE
    / "benchmark"
    / "cardinality_methods"
    / "cost_models"
    / "runs"
    / "pretrained_zeroshot_mirage_test42_100k"
)
LCM_INPUTS = (
    WORKSPACE
    / "benchmark"
    / "cardinality_methods"
    / "cost_models"
    / "lcm_eval_inputs"
    / "mirage_test42_100k_20260504"
    / "parsed_plans"
)
QUERY_ROOT = BENCH / "test42_100k_exportpatch" / "queries"

WORKLOADS = {
    "job_light": "JOB-Light",
    "job_exact": "JOB",
    "job": "JOB",
    "job_complex": "JOB-Complex",
}
POSTGRES_CE_DIRS = {
    "job_light": "JOB-Light",
    "job_exact": "JOB",
    "job_complex": "JOB-Complex",
}
MSCN_PRED_FILES = {
    "job_light.csv": "JOB-Light",
    "job_exact.csv": "JOB",
    "job_complex.csv": "JOB-Complex",
}
ZEROSHOT_FILES = {
    "mirage_job_light_{seed}_test_pred.csv": ("job_light", "JOB-Light"),
    "mirage_job_{seed}_test_pred.csv": ("job", "JOB"),
    "mirage_job_complex_{seed}_test_pred.csv": ("job_complex", "JOB-Complex"),
}
COST_MSCN_FILES = {
    "job_light.csv": "JOB-Light",
    "job.csv": "JOB",
    "job_complex.csv": "JOB-Complex",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
            count += 1
    return count


def as_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def canonical_sql(sql: str) -> str:
    sql = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"\s+", " ", sql.strip().rstrip(";").lower())
    return sql


def natural_key(text: str) -> list[object]:
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p for p in parts]


def workload_query_dirs() -> dict[str, Path]:
    return {
        "JOB-Light": QUERY_ROOT / "job_light",
        "JOB": QUERY_ROOT / "job_exact",
        "JOB-Complex": QUERY_ROOT / "job_complex",
    }


def sql_by_workload() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for workload, folder in workload_query_dirs().items():
        by_id: dict[str, str] = {}
        if folder.exists():
            for path in sorted(folder.glob("*.sql"), key=lambda p: natural_key(p.stem)):
                by_id[path.stem] = path.read_text(encoding="utf-8")
        result[workload] = by_id
    return result


def count_top_level_tables(from_clause: str) -> int:
    depth = 0
    tables = 1 if from_clause.strip() else 0
    for ch in from_clause:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            tables += 1
    joins = len(re.findall(r"\bjoin\b", from_clause, flags=re.IGNORECASE))
    return max(tables, joins + 1 if joins else tables)


def count_where_predicates(sql: str) -> int:
    normalized = re.sub(r"\s+", " ", sql)
    match = re.search(
        r"\bwhere\b(.*?)(?:\bgroup\s+by\b|\border\s+by\b|\blimit\b|;|$)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return 0
    where = match.group(1)
    pieces = re.split(r"\s+and\s+", where, flags=re.IGNORECASE)
    return len([piece for piece in pieces if piece.strip()])


def query_metadata_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for workload, queries in sql_by_workload().items():
        for query_id, sql in queries.items():
            normalized = re.sub(r"\s+", " ", sql.strip())
            from_match = re.search(
                r"\bfrom\b(.*?)(?:\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\blimit\b|;|$)",
                normalized,
                flags=re.IGNORECASE | re.DOTALL,
            )
            table_count = count_top_level_tables(from_match.group(1)) if from_match else 0
            rows.append(
                {
                    "workload": workload,
                    "query_id": query_id,
                    "num_joins": max(0, table_count - 1),
                    "num_predicates": count_where_predicates(sql),
                    "template_family": workload,
                    "sql_text": sql.strip(),
                }
            )
    return rows


def lcm_query_id_maps(sql_lookup: dict[str, dict[str, str]]) -> dict[str, dict[int, str]]:
    maps: dict[str, dict[int, str]] = {}
    lcm_files = {
        "job_light": ("JOB-Light", LCM_INPUTS / "mirage_job_light.json"),
        "job": ("JOB", LCM_INPUTS / "mirage_job.json"),
        "job_complex": ("JOB-Complex", LCM_INPUTS / "mirage_job_complex.json"),
    }
    for lcm_name, (workload, path) in lcm_files.items():
        query_ids_by_sql = {
            canonical_sql(sql): query_id for query_id, sql in sql_lookup.get(workload, {}).items()
        }
        index_map: dict[int, str] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for idx, plan in enumerate(data.get("parsed_plans", [])):
                query_id = query_ids_by_sql.get(canonical_sql(str(plan.get("sql", ""))))
                index_map[idx] = query_id if query_id else f"idx_{idx}"
        maps[lcm_name] = index_map
    return maps


def build_full_query_cardinality() -> int:
    rows: list[dict[str, object]] = []

    for dirname, workload in POSTGRES_CE_DIRS.items():
        path = POSTGRES_CE / dirname / "query_summary.csv"
        for row in read_csv(path):
            if row.get("status") != "ok":
                continue
            rows.append(
                {
                    "workload": workload,
                    "query_id": row["query_id"],
                    "method": "PostgreSQL",
                    "seed": "",
                    "actual_cardinality": row["target_actual"],
                    "estimated_cardinality": row["target_plan_rows"],
                }
            )

    for seed_dir in sorted(MSCN_CARD.glob("seed_*"), key=lambda p: natural_key(p.name)):
        seed = seed_dir.name.removeprefix("seed_")
        pred_dir = seed_dir / "predictions"
        for filename, workload in MSCN_PRED_FILES.items():
            path = pred_dir / filename
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "workload": workload,
                        "query_id": row["query_id"],
                        "method": "MSCN",
                        "seed": seed,
                        "actual_cardinality": row["actual"],
                        "estimated_cardinality": row["prediction"],
                    }
                )

    return write_csv(
        OUT / "full_query_cardinality.csv",
        rows,
        [
            "workload",
            "query_id",
            "method",
            "seed",
            "actual_cardinality",
            "estimated_cardinality",
        ],
    )


def build_selected_plan_cost(sql_lookup: dict[str, dict[str, str]]) -> int:
    rows: list[dict[str, object]] = []

    scale_by_workload: dict[str, float] = {}
    summary_path = FINAL / "cost_postgres_selected_plan" / "postgres_cost_summary.csv"
    for row in read_csv(summary_path):
        scale_by_workload[row["workload"]] = as_float(row["median_cost_to_ms_scale"])

    pg_path = FINAL / "cost_postgres_selected_plan" / "postgres_cost_per_query.csv"
    for row in read_csv(pg_path):
        workload = WORKLOADS[row["workload"]]
        scale = scale_by_workload[row["workload"]]
        rows.append(
            {
                "workload": workload,
                "query_id": row["query_id"],
                "method": "PostgreSQL scaled planner cost",
                "seed": "",
                "actual_runtime": row["execution_ms"],
                "predicted_value": as_float(row["total_cost"]) * scale,
            }
        )

    index_maps = lcm_query_id_maps(sql_lookup)
    for seed in [0, 1, 2]:
        seed_dir = ZEROSHOT_COST / f"seed_{seed}"
        for pattern, (lcm_name, workload) in ZEROSHOT_FILES.items():
            path = seed_dir / pattern.format(seed=seed)
            if not path.exists():
                continue
            query_map = index_maps.get(lcm_name, {})
            for row in read_csv(path):
                idx = int(row["query_index"])
                rows.append(
                    {
                        "workload": workload,
                        "query_id": query_map.get(idx, f"idx_{idx}"),
                        "method": "Pretrained ZeroShot",
                        "seed": seed,
                        "actual_runtime": row["label"],
                        "predicted_value": row["prediction"],
                    }
                )

    cost_mscn = FINAL / "cost_mscn_selected_plan"
    for seed_dir in sorted(cost_mscn.glob("seed_*"), key=lambda p: natural_key(p.name)):
        seed = seed_dir.name.removeprefix("seed_")
        pred_dir = seed_dir / "predictions"
        for filename, workload in COST_MSCN_FILES.items():
            path = pred_dir / filename
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "workload": workload,
                        "query_id": row["query_id"],
                        "method": "MSCN selected-plan runtime",
                        "seed": seed,
                        "actual_runtime": row["actual_ms"],
                        "predicted_value": row["prediction_ms"],
                    }
                )

    return write_csv(
        OUT / "selected_plan_cost.csv",
        rows,
        ["workload", "query_id", "method", "seed", "actual_runtime", "predicted_value"],
    )


def build_base_selection_cardinality() -> int:
    rows: list[dict[str, object]] = []
    path = FINAL / "base_selection_table1" / "base_selection_qerrors.csv"
    for row in read_csv(path):
        estimator = {
            "postgres": "PostgreSQL",
            "duckdb": "DuckDB",
        }.get(row["estimator"], row["estimator"])
        workload = WORKLOADS[row["workload"]]
        selection_id = f"{workload}:{row['query_id']}:{row['alias']}"
        rows.append(
            {
                "workload": workload,
                "selection_id": selection_id,
                "estimator": estimator,
                "actual_cardinality": row["actual"],
                "estimated_cardinality": row["estimate"],
            }
        )
    return write_csv(
        OUT / "base_selection_cardinality.csv",
        rows,
        [
            "workload",
            "selection_id",
            "estimator",
            "actual_cardinality",
            "estimated_cardinality",
        ],
    )


def build_query_result_sizes() -> int:
    rows: list[dict[str, object]] = []
    path = FINAL / "artifact_workload_summary" / "query_structure_audit.csv"
    for row in read_csv(path):
        rows.append(
            {
                "workload": row["workload"],
                "query_id": row["query_id"],
                "actual_result_size": row["target_actual"],
            }
        )
    return write_csv(
        OUT / "query_result_sizes.csv",
        rows,
        ["workload", "query_id", "actual_result_size"],
    )


def build_runtime_breakdowns() -> int:
    rows: list[dict[str, object]] = []
    for dirname, workload in POSTGRES_CE_DIRS.items():
        path = POSTGRES_CE / dirname / "query_summary.csv"
        for row in read_csv(path):
            if row.get("status") != "ok":
                continue
            rows.append(
                {
                    "workload": workload,
                    "query_id": row["query_id"],
                    "dbms": "PostgreSQL",
                    "planning_time": row["planning_ms"],
                    "execution_time": row["execution_ms"],
                }
            )
    return write_csv(
        OUT / "runtime_breakdowns.csv",
        rows,
        ["workload", "query_id", "dbms", "planning_time", "execution_time"],
    )


def build_plan_metadata() -> int:
    rows: list[dict[str, object]] = []
    for dirname, workload in POSTGRES_CE_DIRS.items():
        path = POSTGRES_CE / dirname / "query_summary.csv"
        for row in read_csv(path):
            if row.get("status") != "ok":
                continue
            shape = "|".join(
                [
                    workload,
                    row["query_id"],
                    row.get("root_node_type", ""),
                    row.get("target_node_type", ""),
                    row.get("node_count", ""),
                ]
            )
            rows.append(
                {
                    "workload": workload,
                    "query_id": row["query_id"],
                    "dbms": "PostgreSQL",
                    "estimated_rows_root": row["root_plan_rows"],
                    "actual_rows_root": row["root_actual_rows"],
                    "chosen_plan_hash": hashlib.sha1(shape.encode("utf-8")).hexdigest()[:16],
                }
            )
    return write_csv(
        OUT / "plan_metadata.csv",
        rows,
        [
            "workload",
            "query_id",
            "dbms",
            "estimated_rows_root",
            "actual_rows_root",
            "chosen_plan_hash",
        ],
    )


def build_query_metadata() -> int:
    return write_csv(
        OUT / "query_metadata.csv",
        query_metadata_rows(),
        ["workload", "query_id", "num_joins", "num_predicates", "template_family", "sql_text"],
    )


def write_readme(row_counts: dict[str, int]) -> None:
    readme = OUT / "README.md"
    lines = [
        "# Paper Requested Raw Data",
        "",
        "Normalized raw CSVs for paper plotting and external GPT analysis.",
        "These files contain per-query/per-selection values, not only summary percentiles.",
        "",
        "## Files",
        "",
    ]
    for name, count in row_counts.items():
        lines.append(f"- `{name}`: {count} data rows.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `full_query_cardinality.csv` contains PostgreSQL selected target-node cardinalities and MSCN per-seed full-query predictions.",
            "- DuckDB is included for base-selection cardinality only; no DuckDB full-query join cardinality run was finalized.",
            "- `selected_plan_cost.csv` contains selected-plan cost/runtime rows for PostgreSQL scaled planner cost, pretrained ZeroShot, and MSCN selected-plan runtime.",
            "- Pretrained ZeroShot rows use the lcm-eval runtime/cost label scale from the downloaded artifacts.",
            "- PostgreSQL `plan_metadata.csv` root rows are aggregate roots for `COUNT(*)`; `full_query_cardinality.csv` uses the target node/result size instead.",
            "- Query metadata counts joins/predicates with a lightweight SQL parser intended for plotting, not as a formal SQL AST.",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(row_counts: dict[str, int]) -> None:
    manifest = {
        "dataset": "imdb_test42_100k_exportpatch",
        "created_by": "test44/final_benchmarks/scripts/build_paper_requested_raw_data.py",
        "purpose": "Raw per-query/per-selection CSVs requested for paper plots.",
        "row_counts": row_counts,
        "files": {
            "full_query_cardinality.csv": {
                "schema": [
                    "workload",
                    "query_id",
                    "method",
                    "seed",
                    "actual_cardinality",
                    "estimated_cardinality",
                ],
                "methods": ["PostgreSQL", "MSCN"],
                "source_note": "PostgreSQL uses selected target-node row estimate/actual from EXPLAIN ANALYZE; MSCN uses three seed prediction files.",
            },
            "selected_plan_cost.csv": {
                "schema": [
                    "workload",
                    "query_id",
                    "method",
                    "seed",
                    "actual_runtime",
                    "predicted_value",
                ],
                "methods": [
                    "PostgreSQL scaled planner cost",
                    "Pretrained ZeroShot",
                    "MSCN selected-plan runtime",
                ],
            },
            "base_selection_cardinality.csv": {
                "schema": [
                    "workload",
                    "selection_id",
                    "estimator",
                    "actual_cardinality",
                    "estimated_cardinality",
                ],
                "estimators": ["PostgreSQL", "DuckDB"],
            },
            "query_result_sizes.csv": {
                "schema": ["workload", "query_id", "actual_result_size"],
            },
            "query_metadata.csv": {
                "schema": [
                    "workload",
                    "query_id",
                    "num_joins",
                    "num_predicates",
                    "template_family",
                    "sql_text",
                ],
            },
            "runtime_breakdowns.csv": {
                "schema": ["workload", "query_id", "dbms", "planning_time", "execution_time"],
            },
            "plan_metadata.csv": {
                "schema": [
                    "workload",
                    "query_id",
                    "dbms",
                    "estimated_rows_root",
                    "actual_rows_root",
                    "chosen_plan_hash",
                ],
            },
        },
        "explicitly_not_included": [
            "Full generated database table CSVs",
            ".env or API keys",
            "Python virtual environments",
            "Historical run directories",
        ],
    }
    (OUT / "raw_data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def copy_to_handoff() -> None:
    if not HANDOFF.exists():
        return

    handoff_results = HANDOFF / "final_paper_results"
    handoff_results.mkdir(parents=True, exist_ok=True)
    for filename in ["README.md", "paper_current_manifest.json"]:
        src = FINAL / filename
        if src.exists():
            shutil.copy2(src, handoff_results / filename)

    script_dst = HANDOFF / "final_benchmark_scripts"
    script_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, script_dst / SCRIPT.name)

    handoff_final = handoff_results / OUT.name
    if handoff_final.exists():
        shutil.rmtree(handoff_final)
    shutil.copytree(OUT, handoff_final)

    requested = HANDOFF / "requested_csv_files"
    requested.mkdir(parents=True, exist_ok=True)
    for csv_path in OUT.glob("*.csv"):
        shutil.copy2(csv_path, requested / f"{OUT.name}_{csv_path.name}")

    remove_packaging_noise()
    update_handoff_manifest()
    rebuild_handoff_zip()


def is_package_noise(path: Path) -> bool:
    parts = set(path.parts)
    name = path.name
    return (
        "__pycache__" in parts
        or name.endswith(".pyc")
        or name == ".env"
        or name.startswith(".env.")
        or "site-packages" in parts
        or "runs" in parts
        or "envs" in parts
    )


def remove_packaging_noise() -> None:
    for pycache in sorted(HANDOFF.rglob("__pycache__"), reverse=True):
        if pycache.is_dir():
            shutil.rmtree(pycache)
    for path in HANDOFF.rglob("*.pyc"):
        if path.is_file():
            path.unlink()


def update_handoff_manifest() -> None:
    if not HANDOFF.exists():
        return
    rows: list[dict[str, object]] = []
    for path in sorted(HANDOFF.rglob("*")):
        if path.is_file() and not is_package_noise(path):
            rel = path.relative_to(HANDOFF).as_posix()
            rows.append({"path": rel, "bytes": path.stat().st_size})
    write_csv(HANDOFF / "file_manifest.csv", rows, ["path", "bytes"])


def rebuild_handoff_zip() -> None:
    if not HANDOFF.exists():
        return
    with zipfile.ZipFile(HANDOFF_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(HANDOFF.rglob("*")):
            if path.is_file() and not is_package_noise(path):
                zf.write(path, path.relative_to(HANDOFF.parent).as_posix())


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    sql_lookup = sql_by_workload()
    row_counts = {
        "full_query_cardinality.csv": build_full_query_cardinality(),
        "selected_plan_cost.csv": build_selected_plan_cost(sql_lookup),
        "base_selection_cardinality.csv": build_base_selection_cardinality(),
        "query_result_sizes.csv": build_query_result_sizes(),
        "query_metadata.csv": build_query_metadata(),
        "runtime_breakdowns.csv": build_runtime_breakdowns(),
        "plan_metadata.csv": build_plan_metadata(),
    }
    write_readme(row_counts)
    write_manifest(row_counts)
    copy_to_handoff()

    print(json.dumps({"output": str(OUT), "row_counts": row_counts}, indent=2))
    if HANDOFF_ZIP.exists():
        print(json.dumps({"handoff_zip": str(HANDOFF_ZIP), "bytes": HANDOFF_ZIP.stat().st_size}))


if __name__ == "__main__":
    main()
