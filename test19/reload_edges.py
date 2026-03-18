"""Reload edges from edges_temporal.csv (10.3M rows including expired)."""
import psycopg2, csv, sys, io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')
conn.autocommit = True
cur = conn.cursor()

EDGES_CSV = r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\edges_temporal.csv"

cur.execute("DROP TABLE IF EXISTS edges CASCADE")
print("Dropped old edges table")

with open(EDGES_CSV, 'r', encoding='utf-8', errors='replace') as f:
    header = next(csv.reader(f))
print(f"Columns: {header}")

col_defs = ', '.join([f'"{c}" TEXT' for c in header])
cur.execute(f"CREATE TABLE edges ({col_defs})")
print("Created edges table")

print("Loading edges_temporal.csv (10.3M rows)...", flush=True)
col_list = ', '.join([f'"{c}"' for c in header])
with open(EDGES_CSV, 'r', encoding='utf-8', errors='replace') as f:
    cur.copy_expert(f"COPY edges ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')", f)

cur.execute("SELECT COUNT(*) FROM edges")
print(f"Loaded: {cur.fetchone()[0]:,} rows")

print("Converting column types...", flush=True)
for col in ['src_id', 'dst_id']:
    cur.execute(f'ALTER TABLE edges ALTER COLUMN "{col}" TYPE INT USING "{col}"::INT')
cur.execute("""ALTER TABLE edges ALTER COLUMN "weight" TYPE FLOAT USING NULLIF("weight",'')::FLOAT""")
for col in ['valid_from', 'valid_to']:
    cur.execute(f"""ALTER TABLE edges ALTER COLUMN "{col}" TYPE FLOAT USING NULLIF("{col}",'')::FLOAT""")

print("Creating indexes...", flush=True)
cur.execute("CREATE INDEX idx_edges_src ON edges(src_id)")
cur.execute("CREATE INDEX idx_edges_dst ON edges(dst_id)")
cur.execute("CREATE INDEX idx_edges_type ON edges(edge_type)")
cur.execute("CREATE INDEX idx_edges_src_dst ON edges(src_id, dst_id)")
cur.execute("CREATE INDEX idx_edges_valid ON edges(valid_from, valid_to)")
cur.execute("ANALYZE edges")

# Stats
cur.execute("SELECT edge_type, COUNT(*) c FROM edges GROUP BY edge_type ORDER BY c DESC")
print("\nEdge types:")
for r in cur.fetchall():
    print(f"  {r}")

cur.execute("SELECT MIN(valid_from), MAX(valid_from), MIN(valid_to), MAX(valid_to) FROM edges WHERE valid_from IS NOT NULL")
r = cur.fetchone()
print(f"\nTemporal range: valid_from {r[0]:.0f}-{r[1]:.0f}, valid_to {r[2]:.0f}-{r[3]:.0f}")

cur.execute("SELECT COUNT(*) FROM edges WHERE valid_to < 2025")
print(f"Expired (valid_to < 2025): {cur.fetchone()[0]:,}")
cur.execute("SELECT COUNT(*) FROM edges WHERE valid_to >= 2025")
print(f"Active (valid_to >= 2025): {cur.fetchone()[0]:,}")

print("\nDone!")
cur.close()
conn.close()
