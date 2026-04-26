from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXACT_JOB_DIR = ROOT / "benchmark" / "job_exact_v1"
if str(EXACT_JOB_DIR) not in sys.path:
    sys.path.insert(0, str(EXACT_JOB_DIR))

from exact_job_benchmark import (  # noqa: E402
    Slot,
    extract_from_where,
    extract_slots,
    flexible_like_patterns,
    mine_global_values_for_slot,
    original_slot_values,
    render_sql,
    slot_manifest_entry,
    slot_values_from_witness,
    sort_key,
    sql_quote,
    strip_to_count,
    strip_to_exists,
    structure_fingerprint,
    with_statement_timeout,
)


DEFAULT_DOCKER_BIN_CANDIDATES = (
    Path(r"/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"),
    Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
)


class Postgres:
    def __init__(self, *, docker_bin: str, container: str, db_name: str, user: str) -> None:
        self.docker_bin = docker_bin
        self.container = container
        self.db_name = db_name
        self.user = user

    def sql(self, sql: str, *, timeout_sec: int = 120) -> str:
        cmd = [
            self.docker_bin,
            "exec",
            self.container,
            "psql",
            "-X",
            "-U",
            self.user,
            "-d",
            self.db_name,
            "-v",
            "ON_ERROR_STOP=1",
            "-P",
            "pager=off",
            "-tAq",
            "-c",
            sql,
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, timeout_sec),
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(message or f"psql exited with code {completed.returncode}")
        return (completed.stdout or "").strip()

    def scalar_int(self, sql: str, *, timeout_sec: int = 120) -> int:
        raw = self.sql(sql, timeout_sec=timeout_sec)
        return int(float(raw or "0"))

    def json_scalar(self, sql: str, *, timeout_sec: int = 120) -> Any:
        raw = self.sql(sql, timeout_sec=timeout_sec)
        return json.loads(raw or "null")

    def analyze(self) -> None:
        self.sql("ANALYZE;", timeout_sec=300)


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


def sql_literal(value: Any, slot: Slot) -> str:
    literal = slot.literals[0] if slot.literals else None
    if literal is not None and literal.kind == "number":
        try:
            if float(value).is_integer():
                return str(int(float(value)))
            return str(float(value))
        except Exception:
            return "0"
    return sql_quote(str(value))


def build_fallback_candidates(slot: Slot, pg: Postgres, limit: int) -> list[tuple[str, ...]]:
    candidates = [original_slot_values(slot)]
    try:
        global_rows = mine_global_values_for_slot(slot=slot, runner=pg, limit=limit)
    except Exception:
        global_rows = []

    if slot.kind == "like":
        original = slot.literals[0].sql_text.strip("'") if slot.literals else ""
        for row in global_rows:
            value = row.get("value")
            if value is None:
                continue
            for pattern in flexible_like_patterns(original, str(value)):
                candidates.append((sql_quote(pattern),))
    elif slot.kind == "in":
        values: list[str] = []
        for row in global_rows:
            value = row.get("value")
            if value is not None:
                values.append(sql_literal(value, slot))
        if values:
            arity = slot.arity
            for start in range(0, min(4, max(1, len(values) - arity + 1))):
                chunk = values[start:start + arity]
                if len(chunk) == arity:
                    candidates.append(tuple(chunk))
    elif slot.kind == "between":
        nums = []
        for row in global_rows:
            try:
                nums.append(int(round(float(row.get("value")))))
            except Exception:
                pass
        if nums:
            nums = sorted(set(nums))
            try:
                width = max(0, int(float(slot.literals[1].sql_text) - float(slot.literals[0].sql_text)))
            except Exception:
                width = 0
            for value in nums[:5]:
                candidates.append((str(value), str(value + width)))
    else:
        for row in global_rows:
            value = row.get("value")
            if value is not None:
                candidates.append((sql_literal(value, slot),))

    deduped: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def mine_witness_rows(
    *,
    original_sql: str,
    slots: list[Slot],
    pg: Postgres,
    limit: int,
    timeout_ms: int,
) -> list[dict[str, Any]]:
    if not slots:
        return []
    relaxed_sql = render_sql(
        original_sql,
        slots,
        {},
        slot_overrides={slot.index: "TRUE" for slot in slots},
    )
    from_clause, where_clause = extract_from_where(relaxed_sql)
    select_list = ",\n            ".join(f"{slot.expr} AS v{slot.index}" for slot in slots)
    non_null = "\n          AND ".join(f"{slot.expr} IS NOT NULL" for slot in slots)
    query = f"""
    SELECT COALESCE(json_agg(row_to_json(s))::text, '[]')
    FROM (
        SELECT {select_list}
        {from_clause}
        WHERE {where_clause}
          AND {non_null}
        LIMIT {int(limit)}
    ) s;
    """
    payload = pg.json_scalar(with_statement_timeout(query, timeout_ms), timeout_sec=max(10, timeout_ms // 1000 + 10))
    return payload if isinstance(payload, list) else []


def state_from_witness(
    *,
    slots: list[Slot],
    row: dict[str, Any],
    slot_candidates: dict[int, list[tuple[str, ...]]],
) -> dict[int, tuple[str, ...]]:
    state: dict[int, tuple[str, ...]] = {}
    for slot in slots:
        values = slot_values_from_witness(
            slot,
            row.get(f"v{slot.index}"),
            slot_candidates.get(slot.index),
        )
        if values and values != original_slot_values(slot):
            state[slot.index] = values
    return state


def evaluate_state(
    *,
    original_sql: str,
    slots: list[Slot],
    state: dict[int, tuple[str, ...]],
    pg: Postgres,
    count_timeout_ms: int,
) -> tuple[bool, int | None, str | None]:
    adapted_sql = render_sql(original_sql, slots, state)
    try:
        exists_sql = with_statement_timeout(strip_to_exists(adapted_sql), count_timeout_ms)
        exists = pg.scalar_int(exists_sql, timeout_sec=max(10, count_timeout_ms // 1000 + 10))
        if not exists:
            return False, 0, None
        count_sql = with_statement_timeout(strip_to_count(adapted_sql), count_timeout_ms)
        actual = pg.scalar_int(count_sql, timeout_sec=max(10, count_timeout_ms // 1000 + 10))
        return True, actual, None
    except Exception as exc:
        return False, None, str(exc).splitlines()[0][:240]


def state_exists(
    *,
    original_sql: str,
    slots: list[Slot],
    state: dict[int, tuple[str, ...]],
    pg: Postgres,
    timeout_ms: int,
) -> tuple[bool, str | None]:
    adapted_sql = render_sql(original_sql, slots, state)
    try:
        exists_sql = with_statement_timeout(strip_to_exists(adapted_sql), timeout_ms)
        exists = pg.scalar_int(exists_sql, timeout_sec=max(10, timeout_ms // 1000 + 10))
        return bool(exists), None
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def count_state(
    *,
    original_sql: str,
    slots: list[Slot],
    state: dict[int, tuple[str, ...]],
    pg: Postgres,
    timeout_ms: int,
) -> tuple[int | None, str | None]:
    adapted_sql = render_sql(original_sql, slots, state)
    try:
        count_sql = with_statement_timeout(strip_to_count(adapted_sql), timeout_ms)
        actual = pg.scalar_int(count_sql, timeout_sec=max(10, timeout_ms // 1000 + 10))
        return actual, None
    except Exception as exc:
        return None, str(exc).splitlines()[0][:240]


def adapt_one(
    *,
    query_path: Path,
    pg: Postgres,
    witness_limit: int,
    candidate_limit: int,
    witness_timeout_ms: int,
    count_timeout_ms: int,
) -> tuple[str, dict[str, Any]]:
    query_id = query_path.stem
    original_sql = query_path.read_text(encoding="utf-8").strip().rstrip(";")
    slots = extract_slots(original_sql)
    notes: list[str] = []

    slot_candidates = {
        slot.index: build_fallback_candidates(slot, pg, candidate_limit)
        for slot in slots
    }

    states: list[dict[int, tuple[str, ...]]] = [{}]
    try:
        witness_rows = mine_witness_rows(
            original_sql=original_sql,
            slots=slots,
            pg=pg,
            limit=witness_limit,
            timeout_ms=witness_timeout_ms,
        )
        for row in witness_rows:
            if isinstance(row, dict):
                state = state_from_witness(slots=slots, row=row, slot_candidates=slot_candidates)
                if state:
                    states.append(state)
    except Exception as exc:
        witness_rows = []
        notes.append(f"witness mining failed: {str(exc).splitlines()[0][:240]}")

    best_state: dict[int, tuple[str, ...]] = {}
    best_actual: int | None = 0
    best_error: str | None = None
    best_exists = False
    best_score = (-1, 0)
    seen: set[tuple[tuple[int, tuple[str, ...]], ...]] = set()
    for state in states:
        key = tuple(sorted(state.items()))
        if key in seen:
            continue
        seen.add(key)
        exists, error = state_exists(
            original_sql=original_sql,
            slots=slots,
            state=state,
            pg=pg,
            timeout_ms=min(count_timeout_ms, 15000),
        )
        changed = sum(1 for slot in slots if state.get(slot.index) and state[slot.index] != original_slot_values(slot))
        score = (
            1 if exists and not error else 0,
            -changed,
        )
        if score > best_score:
            best_score = score
            best_state = state
            best_error = error
            best_exists = exists
        if exists:
            break

    if best_exists:
        best_actual, count_error = count_state(
            original_sql=original_sql,
            slots=slots,
            state=best_state,
            pg=pg,
            timeout_ms=count_timeout_ms,
        )
        if count_error:
            best_error = count_error if best_error is None else best_error
            notes.append(f"final count failed or timed out: {count_error}")

    adapted_sql = render_sql(original_sql, slots, best_state)
    structure_hash = structure_fingerprint(original_sql)
    structure_ok = structure_fingerprint(adapted_sql) == structure_hash
    changed_slots = sum(
        1 for slot in slots
        if best_state.get(slot.index) and best_state[slot.index] != original_slot_values(slot)
    )
    if not witness_rows:
        notes.append("no coherent all-literals-relaxed witness row found")
    if not best_actual:
        notes.append("adapted query has zero rows or count timed out")
    if best_error:
        notes.append(f"best-state evaluation error: {best_error}")

    return adapted_sql, {
        "query_id": query_id,
        "source_path": str(query_path),
        "structure_checksum": structure_hash,
        "structure_guard_passed": structure_ok,
        "slot_count": len(slots),
        "changed_slots": changed_slots,
        "has_rows": best_exists,
        "actual_count": best_actual,
        "status": "OK" if structure_ok and (best_error is None or best_exists) else "ERROR",
        "error": best_error,
        "notes": notes,
        "original_literals": [slot_manifest_entry(slot, original_slot_values(slot)) for slot in slots],
        "adapted_literals": [
            slot_manifest_entry(slot, best_state.get(slot.index, original_slot_values(slot)))
            for slot in slots
        ],
    }


def write_report(out_dir: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# JOB-Complex Literal Adaptation Report",
        "",
        f"- database: `{manifest['db_name']}`",
        f"- queries: `{summary['query_count']}`",
        f"- structure guard passed: `{summary['structure_guard_passed']}/{summary['query_count']}`",
        f"- witness-positive adapted queries: `{summary['exists_queries']}/{summary['query_count']}`",
        f"- nonzero adapted queries: `{summary['nonzero_queries']}/{summary['query_count']}`",
        f"- zero/count-timeout adapted queries: `{summary['zero_or_unknown_queries']}/{summary['query_count']}`",
        f"- errors: `{summary['errors']}`",
        f"- changed slots median: `{summary['changed_slots_median']}`",
        "",
        "## Zero Or Blocked Queries",
        "",
    ]
    blocked = [q for q in manifest["queries"] if not q.get("actual_count")]
    if not blocked:
        lines.append("- none")
    else:
        for query in blocked:
            notes = "; ".join(query.get("notes") or []) or "no note"
            lines.append(f"- `{query['query_id']}`: {notes}")
    lines.append("")
    out_dir.joinpath("coverage_report.md").write_text("\n".join(lines), encoding="utf-8")


def median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt JOB-Complex literals using coherent witness rows.")
    parser.add_argument("--source-query-dir", default=str(Path(__file__).resolve().parent / "original_queries"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", default=None)
    parser.add_argument("--query-ids", default=None, help="Comma-separated query IDs to adapt.")
    parser.add_argument("--witness-limit", type=int, default=24)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--witness-timeout-ms", type=int, default=30000)
    parser.add_argument("--count-timeout-ms", type=int, default=60000)
    args = parser.parse_args()

    source_query_dir = Path(args.source_query_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    adapted_dir = out_dir / "adapted_queries" / args.db_name
    original_dir = out_dir / "original_queries"
    adapted_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    selected = None
    if args.query_ids:
        selected = {item.strip() for item in args.query_ids.split(",") if item.strip()}
    query_paths = [
        path for path in sorted(source_query_dir.glob("*.sql"), key=sort_key)
        if selected is None or path.stem in selected
    ]
    if not query_paths:
        raise FileNotFoundError(f"No SQL files found in {source_query_dir}")

    pg = Postgres(
        docker_bin=detect_docker_bin(args.docker_bin),
        container=args.pg_container,
        db_name=args.db_name,
        user=args.pg_user,
    )
    pg.analyze()

    queries: list[dict[str, Any]] = []
    for idx, query_path in enumerate(query_paths, start=1):
        print(json.dumps({"event": "adapt_start", "idx": idx, "total": len(query_paths), "query_id": query_path.stem}), flush=True)
        original_dir.joinpath(query_path.name).write_text(query_path.read_text(encoding="utf-8"), encoding="utf-8")
        adapted_sql, record = adapt_one(
            query_path=query_path,
            pg=pg,
            witness_limit=args.witness_limit,
            candidate_limit=args.candidate_limit,
            witness_timeout_ms=args.witness_timeout_ms,
            count_timeout_ms=args.count_timeout_ms,
        )
        adapted_dir.joinpath(query_path.name).write_text(adapted_sql + "\n", encoding="utf-8")
        record["adapted_path"] = str(adapted_dir / query_path.name)
        queries.append(record)
        print(
            json.dumps(
                {
                    "event": "adapt_done",
                    "query_id": record["query_id"],
                    "actual_count": record["actual_count"],
                    "changed_slots": record["changed_slots"],
                    "structure_guard_passed": record["structure_guard_passed"],
                    "notes": record["notes"],
                },
                allow_nan=True,
            ),
            flush=True,
        )

    changed = [int(q["changed_slots"]) for q in queries]
    manifest = {
        "benchmark_id": "job_complex_adapted_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_query_dir": str(source_query_dir),
        "db_name": args.db_name,
        "adapted_query_dir": str(adapted_dir),
        "summary": {
            "query_count": len(queries),
            "structure_guard_passed": sum(1 for q in queries if q["structure_guard_passed"]),
            "exists_queries": sum(1 for q in queries if q.get("has_rows")),
            "nonzero_queries": sum(1 for q in queries if q.get("actual_count") and int(q["actual_count"]) > 0),
            "zero_or_unknown_queries": sum(1 for q in queries if not q.get("actual_count")),
            "errors": sum(1 for q in queries if q.get("status") == "ERROR"),
            "changed_slots_median": median(changed),
            "changed_slots_mean": sum(changed) / len(changed) if changed else None,
        },
        "queries": queries,
    }
    out_dir.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    write_report(out_dir, manifest)
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
