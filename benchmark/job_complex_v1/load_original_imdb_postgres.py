from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_DOCKER_BIN_CANDIDATES = (
    Path(r"/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"),
    Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
)

JOB_TABLES = [
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "comp_cast_type",
    "company_name",
    "company_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
]


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


def run_psql(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    sql: str,
    capture: bool = True,
) -> str:
    cmd = [
        docker_bin,
        "exec",
        container,
        "psql",
        "-X",
        "-U",
        user,
        "-d",
        db_name,
        "-v",
        "ON_ERROR_STOP=1",
        "-P",
        "pager=off",
        "-tAq",
        "-c",
        sql,
    ]
    completed = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(message or f"psql failed with exit code {completed.returncode}")
    return (completed.stdout or "").strip() if capture else ""


def create_database(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    drop_create: bool,
) -> None:
    exists = run_psql(
        docker_bin=docker_bin,
        container=container,
        db_name="postgres",
        user=user,
        sql=f"SELECT 1 FROM pg_database WHERE datname = '{db_name}';",
    )
    if exists and not drop_create:
        raise RuntimeError(f"Database already exists: {db_name}. Pass --drop-create to replace it.")
    if exists:
        run_psql(
            docker_bin=docker_bin,
            container=container,
            db_name="postgres",
            user=user,
            sql=(
                "SELECT pg_terminate_backend(pid) "
                f"FROM pg_stat_activity WHERE datname = '{db_name}' AND pid <> pg_backend_pid();"
            ),
        )
        run_psql(
            docker_bin=docker_bin,
            container=container,
            db_name="postgres",
            user=user,
            sql=f'DROP DATABASE "{db_name}";',
        )
    run_psql(
        docker_bin=docker_bin,
        container=container,
        db_name="postgres",
        user=user,
        sql=f'CREATE DATABASE "{db_name}";',
    )


def execute_sql_file(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    sql_path: Path,
) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    run_psql(
        docker_bin=docker_bin,
        container=container,
        db_name=db_name,
        user=user,
        sql=sql,
        capture=True,
    )


def copy_table(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    table: str,
    csv_path: Path,
) -> dict[str, object]:
    cmd = [
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
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        f"COPY {table} FROM STDIN WITH (FORMAT CSV, ESCAPE E'\\\\', NULL '');",
    ]
    start = time.perf_counter()
    with csv_path.open("rb") as handle:
        completed = subprocess.run(cmd, stdin=handle, capture_output=True, text=True, check=False)
    elapsed_s = time.perf_counter() - start
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Failed loading {table}: {message}")
    count_raw = run_psql(
        docker_bin=docker_bin,
        container=container,
        db_name=db_name,
        user=user,
        sql=f"SELECT COUNT(*) FROM {table};",
    )
    return {
        "table": table,
        "csv": str(csv_path),
        "rows": int(count_raw or "0"),
        "elapsed_s": round(elapsed_s, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-load the local original JOB IMDb CSV files into a PostgreSQL Docker container."
    )
    parser.add_argument("--dataset-dir", default="imdb_job_dataset")
    parser.add_argument("--db-name", default="imdb_original_job")
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", default=None)
    parser.add_argument("--drop-create", action="store_true")
    parser.add_argument("--skip-indexes", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    schema_path = dataset_dir / "job_queries" / "schema.sql"
    index_path = dataset_dir / "job_queries" / "fkindexes.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema file: {schema_path}")
    missing = [table for table in JOB_TABLES if not (dataset_dir / f"{table}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {', '.join(missing)}")

    docker_bin = detect_docker_bin(args.docker_bin)
    plan = {
        "dataset_dir": str(dataset_dir),
        "db_name": args.db_name,
        "container": args.pg_container,
        "tables": JOB_TABLES,
        "skip_indexes": bool(args.skip_indexes),
        "skip_analyze": bool(args.skip_analyze),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    create_database(
        docker_bin=docker_bin,
        container=args.pg_container,
        db_name=args.db_name,
        user=args.pg_user,
        drop_create=args.drop_create,
    )
    execute_sql_file(
        docker_bin=docker_bin,
        container=args.pg_container,
        db_name=args.db_name,
        user=args.pg_user,
        sql_path=schema_path,
    )

    results = []
    for table in JOB_TABLES:
        csv_path = dataset_dir / f"{table}.csv"
        print(json.dumps({"event": "copy_start", "table": table, "csv": str(csv_path)}), flush=True)
        result = copy_table(
            docker_bin=docker_bin,
            container=args.pg_container,
            db_name=args.db_name,
            user=args.pg_user,
            table=table,
            csv_path=csv_path,
        )
        results.append(result)
        print(json.dumps({"event": "copy_done", **result}), flush=True)

    if not args.skip_indexes:
        print(json.dumps({"event": "create_indexes_start"}), flush=True)
        execute_sql_file(
            docker_bin=docker_bin,
            container=args.pg_container,
            db_name=args.db_name,
            user=args.pg_user,
            sql_path=index_path,
        )
        print(json.dumps({"event": "create_indexes_done"}), flush=True)

    if not args.skip_analyze:
        print(json.dumps({"event": "analyze_start"}), flush=True)
        run_psql(
            docker_bin=docker_bin,
            container=args.pg_container,
            db_name=args.db_name,
            user=args.pg_user,
            sql="ANALYZE;",
        )
        print(json.dumps({"event": "analyze_done"}), flush=True)

    print(json.dumps({"event": "load_complete", "db_name": args.db_name, "tables": results}, indent=2))


if __name__ == "__main__":
    main()
