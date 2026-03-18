"""Load strict JOB/IMDB CSVs into Postgres with typed DDL and schema checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import psycopg2

from imdb_job_contract import JOB_CORE_TABLES, JOB_INDEX_DDL, JOB_TABLE_COLUMNS, JOB_TABLE_TYPES


def _validate_csv_contract(schema_dir: Path):
    missing = []
    mismatched = []
    for table, expected in JOB_TABLE_COLUMNS.items():
        p = schema_dir / f"{table}.csv"
        if not p.exists():
            missing.append(table)
            continue
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        if header != expected:
            mismatched.append((table, expected, header))
    if missing or mismatched:
        parts = []
        if missing:
            parts.append("Missing required JOB CSVs: " + ", ".join(sorted(missing)))
        for table, exp, got in mismatched:
            parts.append(f"Header mismatch for {table}: expected {exp}, got {got}")
        raise RuntimeError("\n".join(parts))


def _create_database(host: str, port: int, user: str, password: str, db_name: str, drop_create: bool):
    conn = psycopg2.connect(host=host, port=port, dbname="postgres", user=user, password=password)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    exists = cur.fetchone() is not None

    if exists and drop_create:
        print(f"Dropping existing database {db_name}...")
        cur.execute(f'DROP DATABASE "{db_name}"')
        exists = False

    if not exists:
        print(f"Creating database {db_name}...")
        cur.execute(f'CREATE DATABASE "{db_name}"')
    else:
        print(f"Using existing database {db_name}")

    cur.close()
    conn.close()


def _db_connect(host: str, port: int, user: str, password: str, db_name: str):
    conn = psycopg2.connect(host=host, port=port, dbname=db_name, user=user, password=password)
    conn.autocommit = True
    return conn


def _create_table(cur, table: str):
    col_types = JOB_TABLE_TYPES[table]
    cols = JOB_TABLE_COLUMNS[table]
    col_defs = ", ".join([f'"{c}" {col_types[c]}' for c in cols])
    cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
    cur.execute(f'CREATE TABLE "{table}" ({col_defs})')


def _load_csv(cur, table: str, csv_path: Path):
    cols = JOB_TABLE_COLUMNS[table]
    col_list = ", ".join([f'"{c}"' for c in cols])
    sql = f'COPY "{table}" ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL \'\')'
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        cur.copy_expert(sql, f)


def _assert_db_contract(cur):
    expected_tables = set(JOB_CORE_TABLES)
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    got_tables = {r[0] for r in cur.fetchall()}
    missing = sorted(expected_tables - got_tables)
    if missing:
        raise RuntimeError("Missing loaded JOB tables: " + ", ".join(missing))

    for table in JOB_CORE_TABLES:
        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        rows = cur.fetchall()
        got_cols = [r[0] for r in rows]
        exp_cols = JOB_TABLE_COLUMNS[table]
        if got_cols != exp_cols:
            raise RuntimeError(f"DB header mismatch for {table}: expected {exp_cols}, got {got_cols}")

        got_types = {col: dtype for col, dtype in rows}
        for col, t in JOB_TABLE_TYPES[table].items():
            exp_db_type = "integer" if t == "INT" else "text"
            if got_types.get(col) != exp_db_type:
                raise RuntimeError(
                    f"DB type mismatch for {table}.{col}: expected {exp_db_type}, got {got_types.get(col)}"
                )


def load_core(schema_dir: Path, host: str, port: int, user: str, password: str, db_name: str, drop_create: bool, analyze: bool):
    _validate_csv_contract(schema_dir)

    _create_database(host=host, port=port, user=user, password=password, db_name=db_name, drop_create=drop_create)
    conn = _db_connect(host=host, port=port, user=user, password=password, db_name=db_name)
    cur = conn.cursor()

    for table in JOB_CORE_TABLES:
        p = schema_dir / f"{table}.csv"
        _create_table(cur, table)
        _load_csv(cur, table, p)
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        n = cur.fetchone()[0]
        print(f"  OK  {table:<16} {n:>10,} rows")

    print("Creating JOB indexes...")
    for ddl in JOB_INDEX_DDL:
        cur.execute(ddl)

    if analyze:
        print("Running ANALYZE...")
        cur.execute("ANALYZE")

    _assert_db_contract(cur)
    print("DB schema contract: OK")

    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load JOB/IMDB schema CSVs into Postgres")
    parser.add_argument("--schema-dir", default=str((Path(__file__).resolve().parent / "imdb_schema").resolve()),
                        help="Directory containing converted JOB CSVs")
    parser.add_argument("--db-name", default="imdb_benchmark", help="Target Postgres database name")
    parser.add_argument("--drop-create", action="store_true", help="Drop database if it exists and recreate it")
    parser.add_argument("--analyze", action="store_true", help="Run ANALYZE after loading")

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="bench")
    args = parser.parse_args()

    schema_dir = Path(args.schema_dir).resolve()
    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")

    print(f"Schema dir: {schema_dir}")
    print(f"Database:   {args.db_name} ({args.host}:{args.port})")

    load_core(
        schema_dir=schema_dir,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        db_name=args.db_name,
        drop_create=bool(args.drop_create),
        analyze=bool(args.analyze),
    )
    print("Done.")


if __name__ == "__main__":
    main()
