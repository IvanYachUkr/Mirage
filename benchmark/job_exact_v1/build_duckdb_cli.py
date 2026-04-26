from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
from pathlib import Path


DEFAULT_DUCKDB_BIN_CANDIDATES = (
    Path(r"C:\Users\vanya\AppData\Local\Microsoft\WinGet\Packages\DuckDB.cli_Microsoft.Winget.Source_8wekyb3d8bbwe\duckdb.exe"),
)


def _detect_duckdb_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("duckdb") or shutil.which("duckdb.exe")
    if found:
        return found
    for candidate in DEFAULT_DUCKDB_BIN_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Could not find DuckDB CLI. Pass --duckdb-bin explicitly.")


def _load_job_core_tables(contract_dir: Path) -> list[str]:
    contract_path = contract_dir / "imdb_job_contract.py"
    if not contract_path.exists():
        raise FileNotFoundError(f"Could not find imdb_job_contract.py in {contract_dir}")
    spec = importlib.util.spec_from_file_location("job_contract_module", contract_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load contract module from {contract_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.JOB_CORE_TABLES)


def _csv_files(schema_dir: Path, *, core_only: bool, contract_dir: Path | None) -> list[Path]:
    files = sorted(schema_dir.glob("*.csv"))
    if not core_only:
        return files
    if contract_dir is None:
        raise ValueError("--contract-dir is required when --core-only is set")
    allowed = {f"{table}.csv" for table in _load_job_core_tables(contract_dir)}
    return [path for path in files if path.name in allowed]


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_duckdb(
    *,
    schema_dir: Path,
    out_path: Path,
    duckdb_bin: str,
    core_only: bool,
    contract_dir: Path | None,
    overwrite: bool,
) -> Path:
    schema_dir = schema_dir.resolve()
    out_path = out_path.resolve()
    files = _csv_files(schema_dir, core_only=core_only, contract_dir=contract_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files selected from {schema_dir}")
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file already exists: {out_path}")
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    statements: list[str] = [
        "PRAGMA threads=4",
    ]
    for csv_path in files:
        table_name = csv_path.stem.replace('"', '""')
        csv_literal = _sql_literal(str(csv_path))
        statements.append(f'DROP TABLE IF EXISTS "{table_name}"')
        statements.append(
            f'''
            CREATE TABLE "{table_name}" AS
            SELECT *
            FROM read_csv_auto(
                {csv_literal},
                header=true,
                sample_size=-1,
                all_varchar=false,
                ignore_errors=false
            )
            '''.strip()
        )
    statements.append(
        "CREATE OR REPLACE TABLE _import_manifest AS "
        f"SELECT current_timestamp AS imported_at, {_sql_literal(str(schema_dir))} AS source_dir, {len(files)} AS csv_count"
    )
    sql = ";\n".join(statements) + ";\n"
    completed = subprocess.run(
        [duckdb_bin, str(out_path), "-c", sql],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        message = stderr or stdout or f"duckdb exited with code {completed.returncode}"
        raise RuntimeError(message)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DuckDB file from an IMDb-schema CSV export using the DuckDB CLI.")
    parser.add_argument("--schema-dir", required=True, help="Directory containing the CSV export.")
    parser.add_argument("--out", required=True, help="Output DuckDB database path.")
    parser.add_argument("--duckdb-bin", default=None, help="Optional explicit DuckDB CLI path.")
    parser.add_argument("--core-only", action="store_true", help="Import only the strict JOB core tables.")
    parser.add_argument("--contract-dir", default=None, help="Directory containing imdb_job_contract.py for --core-only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    schema_dir = Path(args.schema_dir).resolve()
    contract_dir = Path(args.contract_dir).resolve() if args.contract_dir else None
    out_path = build_duckdb(
        schema_dir=schema_dir,
        out_path=Path(args.out),
        duckdb_bin=_detect_duckdb_bin(args.duckdb_bin),
        core_only=bool(args.core_only),
        contract_dir=contract_dir,
        overwrite=bool(args.overwrite),
    )
    print(out_path)


if __name__ == "__main__":
    main()
