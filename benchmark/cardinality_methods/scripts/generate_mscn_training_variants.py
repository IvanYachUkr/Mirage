#!/usr/bin/env python3
"""Generate a separate MSCN training workload from adapted JOB templates.

The generated SQL preserves each source query's structure and changes only
literal constants. Labels are PostgreSQL COUNT(*) cardinalities. The intent is
to train MSCN on synthetic variants and evaluate on a held-out exact-JOB corpus
instead of leaking the evaluation queries into the training set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXACT_JOB_DIR = ROOT / "benchmark" / "job_exact_v1"
if str(EXACT_JOB_DIR) not in sys.path:
    sys.path.insert(0, str(EXACT_JOB_DIR))

from exact_job_benchmark import (  # noqa: E402
    build_slot_candidates,
    extract_slots,
    mine_context_values_for_slot,
    mine_global_values_for_slot,
    mine_joint_witness_states,
    mine_token_match_values_for_slot,
    mine_values_for_slot,
    original_slot_values,
    render_sql,
    sort_key,
    strip_to_count,
    strip_to_exists,
    structure_fingerprint,
    with_statement_timeout,
)


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


class PsqlRunner:
    """Small persistent psql runner compatible with exact_job_benchmark helpers."""

    def __init__(self, *, docker_bin: str, container: str, db_name: str, user: str) -> None:
        self._seq = 0
        self._cache: dict[str, str] = {}
        self._docker_bin = docker_bin
        self._container = container
        self._db_name = db_name
        self._user = user
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
            bufsize=1,
        )

    def _run_once(self, sql: str) -> str:
        completed = subprocess.run(
            [
                self._docker_bin,
                "exec",
                self._container,
                "psql",
                "-X",
                "-U",
                self._user,
                "-d",
                self._db_name,
                "-v",
                "ON_ERROR_STOP=1",
                "-P",
                "pager=off",
                "-tAq",
                "-c",
                sql.strip().rstrip(";") + ";",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(message[:240] or f"psql exited with code {completed.returncode}")
        return (completed.stdout or "").strip()

    def run(self, sql: str) -> str:
        key = sql.strip()
        if key in self._cache:
            return self._cache[key]
        if self.proc.poll() is not None:
            output = self._run_once(sql)
            self._cache[key] = output
            return output
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("psql session is not available")
        self._seq += 1
        sentinel = f"__MIRAGE_MSCN_VARIANT_END_{self._seq}__"
        self.proc.stdin.write(sql.strip().rstrip(";") + ";\n")
        self.proc.stdin.write(f"\\echo {sentinel}\n")
        self.proc.stdin.flush()
        lines: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                output = self._run_once(sql)
                self._cache[key] = output
                return output
            stripped = line.rstrip("\n")
            if stripped == sentinel:
                break
            lines.append(stripped)
        output = "\n".join(lines).strip()
        for line in lines:
            if line.startswith(("ERROR:", "FATAL:", "PANIC:")):
                raise RuntimeError(line[:240])
        self._cache[key] = output
        return output

    def scalar_int(self, sql: str) -> int:
        raw = self.run(sql)
        return int(float(raw or "0"))

    def json_scalar(self, sql: str) -> Any:
        raw = self.run(sql)
        return json.loads(raw or "null")

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write("\\q\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def sql_hash(sql: str) -> str:
    return hashlib.sha256(" ".join(sql.split()).encode("utf-8")).hexdigest()


def count_query(sql: str, runner: PsqlRunner, timeout_ms: int) -> tuple[int, str | None, float]:
    start = time.perf_counter()
    try:
        exists_sql = with_statement_timeout(strip_to_exists(sql), timeout_ms)
        if runner.scalar_int(exists_sql) <= 0:
            return 0, None, (time.perf_counter() - start) * 1000.0
        count_sql = with_statement_timeout(strip_to_count(sql), timeout_ms)
        return runner.scalar_int(count_sql), None, (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return 0, str(exc).splitlines()[0][:240], (time.perf_counter() - start) * 1000.0


def load_eval_hashes(paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        sql = path.read_text(encoding="utf-8").strip().rstrip(";")
        hashes.add(sql_hash(sql))
    return hashes


def mine_candidates_for_query(
    *,
    query_id: str,
    sql: str,
    runner: PsqlRunner,
    candidate_limit: int,
    witness_limit: int,
) -> tuple[dict[int, list[tuple[str, ...]]], list[dict[int, tuple[str, ...]]], list[str]]:
    slots = extract_slots(sql)
    slot_candidates: dict[int, list[tuple[str, ...]]] = {}
    notes: list[str] = []
    for slot in slots:
        local_values = mine_values_for_slot(
            slot=slot,
            base_sql=sql,
            base_state={},
            runner=runner,
            limit=candidate_limit,
        )
        context_values = mine_context_values_for_slot(
            slot=slot,
            base_sql=sql,
            runner=runner,
            limit=candidate_limit,
        )
        token_values = mine_token_match_values_for_slot(slot=slot, runner=runner, limit=candidate_limit)
        global_values = mine_global_values_for_slot(slot=slot, runner=runner, limit=candidate_limit)
        candidates = build_slot_candidates(
            slot=slot,
            query_local_values=local_values,
            historical_local_values=[],
            query_context_values=context_values,
            token_match_values=token_values,
            global_values=global_values,
            historical_state={},
        )
        slot_candidates[slot.index] = candidates
        if len(candidates) <= 1:
            notes.append(f"{query_id}:{slot.expr} has no mined alternatives")
    witnesses = mine_joint_witness_states(
        base_sql=sql,
        slots=slots,
        runner=runner,
        limit=witness_limit,
        slot_candidates=slot_candidates,
    )
    for witness in witnesses:
        for slot_index, values in witness.items():
            existing = slot_candidates.setdefault(slot_index, [])
            if values not in existing:
                existing.append(values)
    return slot_candidates, witnesses, notes


def random_state(
    *,
    slots: list[Any],
    slot_candidates: dict[int, list[tuple[str, ...]]],
    rng: random.Random,
    max_changed_slots: int,
) -> dict[int, tuple[str, ...]]:
    changeable = [
        slot
        for slot in slots
        if any(candidate != original_slot_values(slot) for candidate in slot_candidates.get(slot.index, []))
    ]
    if not changeable:
        return {}
    changed_count = rng.randint(1, min(max_changed_slots, len(changeable)))
    selected = rng.sample(changeable, changed_count)
    state: dict[int, tuple[str, ...]] = {}
    for slot in selected:
        alternatives = [
            candidate
            for candidate in slot_candidates.get(slot.index, [])
            if candidate != original_slot_values(slot)
        ]
        if alternatives:
            # Bias slightly toward frequent/mined candidates but keep diversity.
            limit = min(len(alternatives), max(3, int(len(alternatives) ** 0.5) + 2))
            state[slot.index] = rng.choice(alternatives[:limit])
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-query-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", default=None)
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--target-per-template", type=int, default=8)
    parser.add_argument("--max-attempts-per-template", type=int, default=80)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--witness-limit", type=int, default=48)
    parser.add_argument("--max-changed-slots", type=int, default=4)
    parser.add_argument("--statement-timeout-ms", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source_dir = Path(args.source_query_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    query_out = out_dir / "training_queries"
    query_out.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(source_dir.glob("*.sql"), key=sort_key)
    eval_hashes = load_eval_hashes(source_paths)
    runner = PsqlRunner(
        docker_bin=detect_docker_bin(args.docker_bin),
        container=args.pg_container,
        db_name=args.db_name,
        user=args.pg_user,
    )
    rng = random.Random(args.seed)
    start = time.perf_counter()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set(eval_hashes)
    labels_path = out_dir / "labels.csv"
    manifest_path = out_dir / "manifest.json"

    try:
        for template_idx, path in enumerate(source_paths, start=1):
            if len(accepted) >= int(args.target_total):
                break
            source_id = path.stem
            source_sql = path.read_text(encoding="utf-8").strip().rstrip(";")
            slots = extract_slots(source_sql)
            structure = structure_fingerprint(source_sql)
            slot_candidates, witnesses, notes = mine_candidates_for_query(
                query_id=source_id,
                sql=source_sql,
                runner=runner,
                candidate_limit=int(args.candidate_limit),
                witness_limit=int(args.witness_limit),
            )
            candidate_states: list[dict[int, tuple[str, ...]]] = [dict(item) for item in witnesses]
            attempts = 0
            accepted_for_template = 0
            while (
                attempts < int(args.max_attempts_per_template)
                and accepted_for_template < int(args.target_per_template)
                and len(accepted) < int(args.target_total)
            ):
                attempts += 1
                if candidate_states:
                    state = candidate_states.pop(0)
                else:
                    state = random_state(
                        slots=slots,
                        slot_candidates=slot_candidates,
                        rng=rng,
                        max_changed_slots=int(args.max_changed_slots),
                    )
                if not state:
                    rejected.append({"source_query_id": source_id, "reason": "empty_state"})
                    continue
                sql = render_sql(source_sql, slots, state)
                if structure_fingerprint(sql) != structure:
                    rejected.append({"source_query_id": source_id, "reason": "structure_changed"})
                    continue
                digest = sql_hash(sql)
                if digest in seen_hashes:
                    rejected.append({"source_query_id": source_id, "reason": "duplicate_or_eval_sql"})
                    continue
                actual, error, elapsed_ms = count_query(sql, runner, int(args.statement_timeout_ms))
                if error:
                    rejected.append({"source_query_id": source_id, "reason": "postgres_error", "error": error})
                    continue
                if actual <= 0:
                    rejected.append({"source_query_id": source_id, "reason": "zero_actual"})
                    continue
                seen_hashes.add(digest)
                query_id = f"train_{len(accepted) + 1:05d}_{source_id}"
                query_path = query_out / f"{query_id}.sql"
                query_path.write_text(sql.strip().rstrip(";") + ";\n", encoding="utf-8")
                changed_slots = sum(
                    1
                    for slot in slots
                    if state.get(slot.index) and state[slot.index] != original_slot_values(slot)
                )
                row = {
                    "query_id": query_id,
                    "source_query_id": source_id,
                    "actual_count": int(actual),
                    "changed_slots": changed_slots,
                    "slot_count": len(slots),
                    "structure_checksum": structure,
                    "sql_hash": digest,
                    "count_ms": round(elapsed_ms, 3),
                    "sql_path": str(query_path),
                }
                accepted.append(row)
                accepted_for_template += 1
                print(
                    json.dumps(
                        {
                            "template": f"{template_idx}/{len(source_paths)}",
                            "accepted_total": len(accepted),
                            **row,
                        }
                    ),
                    flush=True,
                )
            if accepted_for_template == 0:
                notes.append(f"{source_id}: no accepted variants")
            (out_dir / "template_notes.jsonl").open("a", encoding="utf-8").write(
                json.dumps(
                    {
                        "source_query_id": source_id,
                        "accepted": accepted_for_template,
                        "attempts": attempts,
                        "slot_count": len(slots),
                        "candidate_counts": {
                            str(slot.index): len(slot_candidates.get(slot.index, []))
                            for slot in slots
                        },
                        "witnesses": len(witnesses),
                        "notes": notes,
                    }
                )
                + "\n"
            )
    finally:
        runner.close()

    import csv

    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "query_id",
            "source_query_id",
            "actual_count",
            "changed_slots",
            "slot_count",
            "structure_checksum",
            "sql_hash",
            "count_ms",
            "sql_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accepted)

    summary = {
        "method": "mscn_exact_job_literal_variant_training_v1",
        "source_query_dir": str(source_dir),
        "training_query_dir": str(query_out),
        "labels_csv": str(labels_path),
        "db_name": args.db_name,
        "seed": int(args.seed),
        "target_total": int(args.target_total),
        "target_per_template": int(args.target_per_template),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "templates_with_training_rows": len({row["source_query_id"] for row in accepted}),
        "source_templates": len(source_paths),
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "rejection_reasons": {},
    }
    for item in rejected:
        reason = str(item.get("reason", "unknown"))
        summary["rejection_reasons"][reason] = summary["rejection_reasons"].get(reason, 0) + 1
    manifest_path.write_text(
        json.dumps({**summary, "accepted_queries": accepted}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
