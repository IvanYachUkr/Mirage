"""Fix missing INT casts in benchmark_v15 that caused type errors in 3 queries."""
import psycopg2
conn = psycopg2.connect(host="localhost", port=5433, dbname="benchmark_v15",
                        user="postgres", password="bench")
conn.autocommit = True
cur = conn.cursor()

FIXES = [
    ("awards",        "person_id",    "INT"),
    ("company_links", "company_id_1", "INT"),
    ("company_links", "company_id_2", "INT"),
    ("media_links",   "source_id",    "INT"),
    ("media_links",   "target_id",    "INT"),
]

for table, col, typ in FIXES:
    sql = (f'ALTER TABLE {table} ALTER COLUMN "{col}" TYPE {typ} '
           f"USING NULLIF(\"{col}\",'')::{typ}")
    try:
        cur.execute(sql)
        print(f"  OK  {table}.{col} -> {typ}")
    except Exception as e:
        conn.rollback()
        conn.autocommit = True
        print(f"  ERR {table}.{col}: {e}")

cur.execute("ANALYZE")
print("Done.")
cur.close()
conn.close()
