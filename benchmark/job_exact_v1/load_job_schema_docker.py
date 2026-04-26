from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PG_CONTAINER = "pg_bench"
DEFAULT_PG_USER = "postgres"
DEFAULT_DOCKER_BIN_CANDIDATES = (
    Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
)


@dataclass(frozen=True)
class JobContract:
    core_tables: list[str]
    table_columns: dict[str, list[str]]
    table_types: dict[str, dict[str, str]]
    index_ddl: list[str]


def _detect_docker_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("docker") or shutil.which("docker.exe")
    if found:
        return found
    for candidate in DEFAULT_DOCKER_BIN_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Could not find docker CLI. Pass --docker-bin explicitly.")


def _docker_cmd(docker_bin: str, *args: str) -> list[str]:
    return [docker_bin, *args]


def _docker_host_path(docker_bin: str, path: Path) -> str:
    """Return a host path format accepted by docker/docker.exe."""
    raw = str(path)
    if docker_bin.lower().endswith("docker.exe") and raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5]
        if raw[6] == "/":
            suffix = raw[7:].replace("/", "\\")
            return f"{drive.upper()}:\\{suffix}"
    return raw


def _run(cmd: list[str], *, capture: bool = True) -> str:
    import subprocess

    completed = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        message = stderr or stdout or f"command failed with exit code {completed.returncode}"
        raise RuntimeError(message)
    return (completed.stdout or "").strip() if capture else ""


def _load_contract(contract_dir: Path) -> JobContract:
    contract_path = contract_dir / "imdb_job_contract.py"
    if not contract_path.exists():
        raise FileNotFoundError(f"Could not find imdb_job_contract.py in {contract_dir}")
    spec = importlib.util.spec_from_file_location("job_contract_module", contract_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load contract module from {contract_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return JobContract(
        core_tables=list(module.JOB_CORE_TABLES),
        table_columns=dict(module.JOB_TABLE_COLUMNS),
        table_types=dict(module.JOB_TABLE_TYPES),
        index_ddl=list(module.JOB_INDEX_DDL),
    )


def _validate_csv_contract(schema_dir: Path, contract: JobContract) -> None:
    missing: list[str] = []
    mismatched: list[tuple[str, list[str], list[str]]] = []
    for table, expected in contract.table_columns.items():
        csv_path = schema_dir / f"{table}.csv"
        if not csv_path.exists():
            missing.append(table)
            continue
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != expected:
            mismatched.append((table, expected, header))
    if missing or mismatched:
        parts: list[str] = []
        if missing:
            parts.append("Missing required JOB CSVs: " + ", ".join(sorted(missing)))
        for table, expected, header in mismatched:
            parts.append(f"Header mismatch for {table}: expected {expected}, got {header}")
        raise RuntimeError("\n".join(parts))


def _psql_exec(
    docker_bin: str,
    *,
    container: str,
    db_name: str,
    user: str,
    sql: str,
) -> str:
    return _run(
        _docker_cmd(
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
        )
    )


def _create_database(
    docker_bin: str,
    *,
    container: str,
    db_name: str,
    user: str,
    drop_create: bool,
) -> None:
    exists = _psql_exec(
        docker_bin,
        container=container,
        db_name="postgres",
        user=user,
        sql=f"SELECT 1 FROM pg_database WHERE datname = '{db_name}';",
    ).strip()
    if exists and drop_create:
        _psql_exec(
            docker_bin,
            container=container,
            db_name="postgres",
            user=user,
            sql=(
                "SELECT pg_terminate_backend(pid) "
                f"FROM pg_stat_activity WHERE datname = '{db_name}' AND pid <> pg_backend_pid();"
            ),
        )
        _psql_exec(
            docker_bin,
            container=container,
            db_name="postgres",
            user=user,
            sql=f'DROP DATABASE "{db_name}";',
        )
        exists = ""
    if not exists:
        _psql_exec(
            docker_bin,
            container=container,
            db_name="postgres",
            user=user,
            sql=f'CREATE DATABASE "{db_name}";',
        )


def _create_table(
    docker_bin: str,
    *,
    container: str,
    db_name: str,
    user: str,
    table: str,
    column_order: list[str],
    column_types: dict[str, str],
) -> None:
    type_map = {"INT": "integer", "TEXT": "text"}
    col_defs = ", ".join(f'"{col}" {type_map[column_types[col]]}' for col in column_order)
    sql = f'DROP TABLE IF EXISTS "{table}" CASCADE; CREATE TABLE "{table}" ({col_defs});'
    _psql_exec(docker_bin, container=container, db_name=db_name, user=user, sql=sql)


def _copy_table(
    docker_bin: str,
    *,
    container: str,
    db_name: str,
    user: str,
    table: str,
    columns: list[str],
    csv_path_in_container: str,
) -> int:
    col_list = ", ".join(f'"{column}"' for column in columns)
    sql = (
        f'COPY "{table}" ({col_list}) '
        f"FROM '{csv_path_in_container}' "
        "WITH (FORMAT CSV, HEADER TRUE, NULL '');"
    )
    _psql_exec(docker_bin, container=container, db_name=db_name, user=user, sql=sql)
    count = _psql_exec(
        docker_bin,
        container=container,
        db_name=db_name,
        user=user,
        sql=f'SELECT COUNT(*) FROM "{table}";',
    )
    return int(count or "0")


def _container_mkdir(docker_bin: str, *, container: str, path: str) -> None:
    _run(_docker_cmd(docker_bin, "exec", container, "mkdir", "-p", path), capture=False)


def _container_rm_rf(docker_bin: str, *, container: str, path: str) -> None:
    _run(_docker_cmd(docker_bin, "exec", container, "rm", "-rf", path), capture=False)


def load_core(
    *,
    schema_dir: Path,
    contract_dir: Path,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    drop_create: bool,
    analyze: bool,
    keep_container_files: bool,
) -> None:
    contract = _load_contract(contract_dir)
    _validate_csv_contract(schema_dir, contract)

    scratch_dir = f"/tmp/job_exact_loader_{db_name}_{int(time.time())}"
    _container_rm_rf(docker_bin, container=container, path=scratch_dir)
    _container_mkdir(docker_bin, container=container, path=scratch_dir)
    _run(_docker_cmd(docker_bin, "cp", _docker_host_path(docker_bin, schema_dir), f"{container}:{scratch_dir}/schema"), capture=False)
    container_schema_dir = f"{scratch_dir}/schema"

    _create_database(
        docker_bin,
        container=container,
        db_name=db_name,
        user=user,
        drop_create=drop_create,
    )

    print(f"Schema dir: {schema_dir}")
    print(f"Contract dir: {contract_dir}")
    print(f"Container: {container}")
    print(f"Database: {db_name}")

    for table in contract.core_tables:
        csv_path = schema_dir / f"{table}.csv"
        _create_table(
            docker_bin,
            container=container,
            db_name=db_name,
            user=user,
            table=table,
            column_order=contract.table_columns[table],
            column_types=contract.table_types[table],
        )
        count = _copy_table(
            docker_bin,
            container=container,
            db_name=db_name,
            user=user,
            table=table,
            columns=contract.table_columns[table],
            csv_path_in_container=f"{container_schema_dir}/{csv_path.name}",
        )
        print(f"  OK  {table:<16} {count:>10,} rows")

    print("Creating JOB indexes...")
    for ddl in contract.index_ddl:
        _psql_exec(docker_bin, container=container, db_name=db_name, user=user, sql=ddl)

    if analyze:
        print("Running ANALYZE...")
        _psql_exec(docker_bin, container=container, db_name=db_name, user=user, sql="ANALYZE;")

    if not keep_container_files:
        _container_rm_rf(docker_bin, container=container, path=scratch_dir)

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load strict JOB/IMDb CSVs into Postgres using docker exec and COPY.")
    parser.add_argument("--schema-dir", required=True, help="Directory containing strict JOB CSV exports.")
    parser.add_argument(
        "--contract-dir",
        default=None,
        help="Directory containing imdb_job_contract.py. Defaults to the parent of --schema-dir.",
    )
    parser.add_argument("--db-name", required=True, help="Target Postgres database name.")
    parser.add_argument("--pg-container", default=DEFAULT_PG_CONTAINER, help="Docker container name for Postgres.")
    parser.add_argument("--pg-user", default=DEFAULT_PG_USER, help="Postgres user inside the container.")
    parser.add_argument("--docker-bin", default=None, help="Optional explicit docker CLI path.")
    parser.add_argument("--drop-create", action="store_true", help="Drop/recreate the target database if it exists.")
    parser.add_argument("--analyze", action="store_true", help="Run ANALYZE after loading the core JOB tables.")
    parser.add_argument("--keep-container-files", action="store_true", help="Keep copied CSV files inside the container scratch directory.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    schema_dir = Path(args.schema_dir).resolve()
    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")
    contract_dir = Path(args.contract_dir).resolve() if args.contract_dir else schema_dir.parent
    docker_bin = _detect_docker_bin(args.docker_bin)

    load_core(
        schema_dir=schema_dir,
        contract_dir=contract_dir,
        docker_bin=docker_bin,
        container=args.pg_container,
        db_name=args.db_name,
        user=args.pg_user,
        drop_create=bool(args.drop_create),
        analyze=bool(args.analyze),
        keep_container_files=bool(args.keep_container_files),
    )


if __name__ == "__main__":
    main()
