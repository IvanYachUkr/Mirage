#!/usr/bin/env python3
"""Build real MSCN materialized-sample bitmaps from a PostgreSQL database."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXACT_JOB_DIR = ROOT / "benchmark" / "job_exact_v1"
if str(EXACT_JOB_DIR) not in sys.path:
    sys.path.insert(0, str(EXACT_JOB_DIR))

from exact_job_benchmark import extract_from_where, extract_slots, parse_alias_map, sort_key  # noqa: E402


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


class PsqlSession:
    def __init__(self, *, docker_bin: str, container: str, db_name: str, user: str) -> None:
        self._seq = 0
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
        if self.proc.poll() is not None:
            return self._run_once(sql)
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("psql session is not available")
        self._seq += 1
        sentinel = f"__MIRAGE_BITMAP_END_{self._seq}__"
        self.proc.stdin.write(sql.strip().rstrip(";") + ";\n")
        self.proc.stdin.write(f"\\echo {sentinel}\n")
        self.proc.stdin.flush()
        lines: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                return self._run_once(sql)
            stripped = line.rstrip("\n")
            if stripped == sentinel:
                break
            lines.append(stripped)
        output = "\n".join(lines).strip()
        for line in lines:
            if line.startswith(("ERROR:", "FATAL:", "PANIC:")):
                raise RuntimeError(line)
        return output

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


def load_manifest_query_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(row["query_id"]) for row in payload.get("queries", [])]


def pack_bits(bit_text: str, num_samples: int) -> bytes:
    bits = [1 if char == "1" else 0 for char in bit_text[:num_samples]]
    if len(bits) < num_samples:
        bits.extend([0] * (num_samples - len(bits)))
    packed = bytearray()
    for offset in range(0, num_samples, 8):
        byte = 0
        for bit in bits[offset:offset + 8]:
            byte = (byte << 1) | bit
        if len(bits[offset:offset + 8]) < 8:
            byte <<= 8 - len(bits[offset:offset + 8])
        packed.append(byte)
    return bytes(packed)


def local_predicate_fragments(sql: str, alias: str) -> list[str]:
    slots = extract_slots(sql)
    return [
        sql[slot.span_start:slot.span_end].strip()
        for slot in slots
        if slot.alias.lower() == alias.lower()
    ]


def bitmap_sql(*, table: str, alias: str, predicates: list[str], num_samples: int) -> str:
    condition = " AND ".join(f"({predicate})" for predicate in predicates) if predicates else "TRUE"
    return f"""
    SELECT COALESCE(
        string_agg(
            CASE WHEN {condition} THEN '1' ELSE '0' END,
            '' ORDER BY {alias}.id
        ),
        ''
    )
    FROM (
        SELECT *
        FROM {table}
        ORDER BY id
        LIMIT {int(num_samples)}
    ) AS {alias}
    """


def write_bitmaps(
    *,
    query_dir: Path,
    query_ids: list[str],
    out_path: Path,
    manifest_path: Path,
    session: PsqlSession,
    num_samples: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    query_meta: list[dict[str, Any]] = []
    table_cache: dict[tuple[str, str, tuple[str, ...]], str] = {}
    start = time.perf_counter()
    with out_path.open("wb") as handle:
        for idx, query_id in enumerate(query_ids, start=1):
            sql_path = query_dir / f"{query_id}.sql"
            sql = sql_path.read_text(encoding="utf-8").strip().rstrip(";")
            from_clause, _ = extract_from_where(sql)
            alias_map = parse_alias_map(from_clause)
            alias_items = sorted(alias_map.items())
            densities: list[float] = []
            zero_tables = 0
            handle.write(struct.pack("<I", len(alias_items)))
            for alias, table in alias_items:
                predicates = local_predicate_fragments(sql, alias)
                cache_key = (table, alias, tuple(predicates))
                bit_text = table_cache.get(cache_key)
                if bit_text is None:
                    bit_text = session.run(bitmap_sql(table=table, alias=alias, predicates=predicates, num_samples=num_samples))
                    table_cache[cache_key] = bit_text
                one_count = bit_text.count("1")
                density = one_count / float(num_samples)
                densities.append(density)
                if one_count == 0:
                    zero_tables += 1
                handle.write(pack_bits(bit_text, num_samples))
            row = {
                "query_id": query_id,
                "table_count": len(alias_items),
                "zero_table_bitmaps": zero_tables,
                "mean_bitmap_density": sum(densities) / len(densities) if densities else None,
                "min_bitmap_density": min(densities) if densities else None,
                "max_bitmap_density": max(densities) if densities else None,
            }
            query_meta.append(row)
            print(json.dumps({"idx": idx, "total": len(query_ids), **row}), flush=True)

    manifest = {
        "method": "materialized_mscn_bitmaps",
        "query_dir": str(query_dir),
        "out_path": str(out_path),
        "num_materialized_samples": int(num_samples),
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "summary": {
            "query_count": len(query_meta),
            "table_bitmaps": sum(row["table_count"] for row in query_meta),
            "zero_table_bitmaps": sum(row["zero_table_bitmaps"] for row in query_meta),
            "mean_query_bitmap_density": (
                sum(float(row["mean_bitmap_density"]) for row in query_meta) / len(query_meta)
                if query_meta else None
            ),
        },
        "queries": query_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--mscn-manifest", default=None, help="Use query order from an existing MSCN export manifest.")
    parser.add_argument("--out-bitmaps", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", default=None)
    parser.add_argument("--num-materialized-samples", type=int, default=1000)
    args = parser.parse_args()

    query_dir = Path(args.query_dir).resolve()
    query_ids = load_manifest_query_ids(Path(args.mscn_manifest).resolve() if args.mscn_manifest else None)
    if query_ids is None:
        query_ids = [path.stem for path in sorted(query_dir.glob("*.sql"), key=sort_key)]
    session = PsqlSession(
        docker_bin=detect_docker_bin(args.docker_bin),
        container=args.pg_container,
        db_name=args.db_name,
        user=args.pg_user,
    )
    try:
        write_bitmaps(
            query_dir=query_dir,
            query_ids=query_ids,
            out_path=Path(args.out_bitmaps).resolve(),
            manifest_path=Path(args.out_manifest).resolve(),
            session=session,
            num_samples=int(args.num_materialized_samples),
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
