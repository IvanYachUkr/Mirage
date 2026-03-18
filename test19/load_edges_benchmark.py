"""Load edges_final.csv into the imdb_benchmark database and explore temporal distribution."""
import psycopg2, csv, sys, io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB = dict(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')
conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

EDGES_CSV = r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\edges_final.csv"

# Check if edges already exists
cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='edges')")
exists = cur.fetchone()[0]
print(f"edges table exists: {exists}")

if exists:
    cur.execute("SELECT COUNT(*) FROM edges")
    print(f"  Current row count: {cur.fetchone()[0]:,}")
else:
    # Read header
    with open(EDGES_CSV, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
    print(f"CSV columns: {header}")

    # Create table
    cur.execute("DROP TABLE IF EXISTS edges CASCADE")
    col_defs = ', '.join([f'"{c}" TEXT' for c in header])
    cur.execute(f"CREATE TABLE edges ({col_defs})")
    print("Created edges table")

    # Load
    print("Loading edges_final.csv...", flush=True)
    col_list = ', '.join([f'"{c}"' for c in header])
    with open(EDGES_CSV, 'r', encoding='utf-8', errors='replace') as f:
        cur.copy_expert(f"COPY edges ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')", f)

    cur.execute("SELECT COUNT(*) FROM edges")
    n = cur.fetchone()[0]
    print(f"Loaded: {n:,} rows")

    # Cast types
    print("Converting column types...", flush=True)
    for col in ['src_id', 'dst_id']:
        cur.execute(f'ALTER TABLE edges ALTER COLUMN "{col}" TYPE INT USING "{col}"::INT')
    cur.execute('ALTER TABLE edges ALTER COLUMN "weight" TYPE FLOAT USING NULLIF("weight",\'\')::FLOAT')
    for col in ['valid_from', 'valid_to']:
        cur.execute(f'ALTER TABLE edges ALTER COLUMN "{col}" TYPE FLOAT USING NULLIF("{col}",\'\')::FLOAT')

    # Indexes
    print("Creating indexes...", flush=True)
    cur.execute("CREATE INDEX idx_edges_src ON edges(src_id)")
    cur.execute("CREATE INDEX idx_edges_dst ON edges(dst_id)")
    cur.execute("CREATE INDEX idx_edges_type ON edges(edge_type)")
    cur.execute("CREATE INDEX idx_edges_src_dst ON edges(src_id, dst_id)")
    cur.execute("CREATE INDEX idx_edges_valid ON edges(valid_from, valid_to)")
    cur.execute("ANALYZE edges")
    print("Done loading edges!")

# ── Explore temporal data ──
print("\n=== Temporal Distribution ===")
cur.execute("SELECT edge_type, COUNT(*) c FROM edges GROUP BY edge_type ORDER BY c DESC")
for r in cur.fetchall(): print(f"  {r}")

cur.execute("SELECT src_type, dst_type, COUNT(*) FROM edges GROUP BY src_type, dst_type ORDER BY COUNT(*) DESC LIMIT 10")
print("\n=== Entity type combos ===")
for r in cur.fetchall(): print(f"  {r}")

cur.execute("SELECT sign, COUNT(*) FROM edges GROUP BY sign")
print("\n=== Sign ===")
for r in cur.fetchall(): print(f"  {r}")

cur.execute("SELECT source_kind, COUNT(*) c FROM edges GROUP BY source_kind ORDER BY c DESC LIMIT 10")
print("\n=== Source kind ===")
for r in cur.fetchall(): print(f"  {r}")

cur.execute("""SELECT 
    MIN(valid_from::float), MAX(valid_from::float), AVG(valid_from::float),
    MIN(valid_to::float), MAX(valid_to::float), AVG(valid_to::float)
FROM edges WHERE valid_from IS NOT NULL""")
r = cur.fetchone()
print(f"\n=== Temporal range ===")
print(f"  valid_from: {r[0]:.0f} - {r[1]:.0f} (avg {r[2]:.0f})")
print(f"  valid_to:   {r[3]:.0f} - {r[4]:.0f} (avg {r[5]:.0f})")

# Distribution by decade
cur.execute("""SELECT (FLOOR(valid_from / 10) * 10)::int AS decade, COUNT(*) 
              FROM edges WHERE valid_from IS NOT NULL 
              GROUP BY decade ORDER BY decade""")
print("\n=== Edges by decade (valid_from) ===")
for r in cur.fetchall(): print(f"  {r[0]}s: {r[1]:,}")

# How many edges connect to title.id via person_id?
cur.execute("""SELECT COUNT(*) FROM edges e 
              JOIN name n ON n.id = e.src_id 
              WHERE e.src_type = 'person' LIMIT 1""")
print(f"\n=== person edges joinable to name: {cur.fetchone()[0]:,} ===")

# Edges where valid_from <= year <= valid_to (active in given year)
cur.execute("""SELECT COUNT(*) FROM edges 
              WHERE valid_from <= 2015 AND valid_to >= 2015""")
print(f"\nEdges active in 2015: {cur.fetchone()[0]:,}")

cur.execute("""SELECT COUNT(*) FROM edges 
              WHERE valid_from <= 2000 AND valid_to >= 2000""")
print(f"Edges active in 2000: {cur.fetchone()[0]:,}")

cur.close()
conn.close()
