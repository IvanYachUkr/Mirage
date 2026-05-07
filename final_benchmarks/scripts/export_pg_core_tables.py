from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


TEST44_DIR = Path(__file__).resolve().parents[2]
ROOT = TEST44_DIR.parent
if str(TEST44_DIR) not in sys.path:
    sys.path.insert(0, str(TEST44_DIR))

from imdb_job_contract import JOB_TABLE_COLUMNS  # noqa: E402


DEFAULT_DOCKER = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
DEFAULT_OUT = TEST44_DIR / "final_benchmarks" / "test42_100k_exportpatch" / "imdb_schema"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def run_psql_scalar(docker_bin: str, container: str, db_name: str, sql: str) -> str:
    proc = subprocess.run(
        [
            docker_bin,
            "exec",
            container,
            "psql",
            "-X",
            "-U",
            "postgres",
            "-d",
            db_name,
            "-qAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            sql,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def export_table(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    table: str,
    columns: list[str],
    out_path: Path,
) -> dict[str, object]:
    col_sql = ", ".join(quote_ident(col) for col in columns)
    table_sql = quote_ident(table)
    copy_sql = (
        f"COPY (SELECT {col_sql} FROM {table_sql} ORDER BY {quote_ident(columns[0])}) "
        "TO STDOUT WITH CSV HEADER"
    )
    count_sql = f"SELECT COUNT(*) FROM {table_sql};"
    expected_count = int(run_psql_scalar(docker_bin, container, db_name, count_sql))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with out_path.open("wb") as handle:
        proc = subprocess.run(
            [
                docker_bin,
                "exec",
                container,
                "psql",
                "-X",
                "-U",
                "postgres",
                "-d",
                db_name,
                "-q",
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                copy_sql,
            ],
            stdout=handle,
            stderr=subprocess.PIPE,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip())

    return {
        "table": table,
        "path": str(out_path),
        "rows": expected_count,
        "bytes": out_path.stat().st_size,
        "elapsed_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final PostgreSQL IMDb core tables to CSV.")
    parser.add_argument("--docker-bin", default=DEFAULT_DOCKER)
    parser.add_argument("--container", default="pg_bench")
    parser.add_argument("--db-name", default="imdb_test42_100k_exportpatch")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tables: list[dict[str, object]] = []
    for table, columns in JOB_TABLE_COLUMNS.items():
        print(f"export {table} ...", flush=True)
        tables.append(
            export_table(
                docker_bin=args.docker_bin,
                container=args.container,
                db_name=args.db_name,
                table=table,
                columns=list(columns),
                out_path=out_dir / f"{table}.csv",
            )
        )

    sanity = {
        "title_count": int(run_psql_scalar(args.docker_bin, args.container, args.db_name, "SELECT COUNT(*) FROM title;")),
        "movie_info_internet_note_count": int(
            run_psql_scalar(
                args.docker_bin,
                args.container,
                args.db_name,
                "SELECT COUNT(*) FROM movie_info WHERE note LIKE '%internet%';",
            )
        ),
    }
    manifest = {
        "source_db": args.db_name,
        "container": args.container,
        "exported_at_unix": time.time(),
        "elapsed_s": time.perf_counter() - started,
        "out_dir": str(out_dir),
        "tables": tables,
        "sanity": sanity,
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

