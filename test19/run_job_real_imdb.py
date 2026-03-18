"""
Load real IMDB JOB dataset into PostgreSQL and run original 113 JOB queries
with EXPLAIN ANALYZE to measure q-error for direct comparison.
"""
import psycopg2, csv, sys, io, time, math, re
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB_ADMIN = dict(host='localhost', port=5433, dbname='postgres', user='postgres', password='bench')
DB_NAME = 'imdb_real'
DB = dict(host='localhost', port=5433, dbname=DB_NAME, user='postgres', password='bench')

DATA_DIR = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\imdb_job_dataset")
SCHEMA_SQL = DATA_DIR / "schematext.sql"
QUERY_DIR = DATA_DIR / "job_queries"

TABLES = [
    "comp_cast_type", "company_type", "info_type", "kind_type", "link_type", "role_type",
    "company_name", "keyword", "name", "char_name",
    "title", "aka_name", "aka_title",
    "movie_info", "movie_info_idx", "movie_keyword", "movie_link",
    "movie_companies", "complete_cast",
    "person_info", "cast_info",
]

# ── Step 1: Create database ──────────────────────────────────
print("=" * 60)
print("Step 1: Creating database...")
print("=" * 60)
conn = psycopg2.connect(**DB_ADMIN)
conn.autocommit = True
cur = conn.cursor()
cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
if cur.fetchone():
    print(f"  Database '{DB_NAME}' already exists")
else:
    cur.execute(f"CREATE DATABASE {DB_NAME}")
    print(f"  Created database '{DB_NAME}'")
cur.close()
conn.close()

# ── Step 2: Create schema ────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Creating schema...")
print("=" * 60)
conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

# Check if tables exist
cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'")
n_tables = cur.fetchone()[0]
if n_tables > 0:
    print(f"  {n_tables} tables already exist, checking data...")
    cur.execute("SELECT COUNT(*) FROM title")
    n_title = cur.fetchone()[0]
    print(f"  title has {n_title:,} rows")
    if n_title > 0:
        print("  Data already loaded, skipping to queries...")
        SKIP_LOAD = True
    else:
        SKIP_LOAD = False
else:
    schema_sql = SCHEMA_SQL.read_text(encoding="utf-8")
    cur.execute(schema_sql)
    print("  Schema created")
    SKIP_LOAD = False

# ── Step 3: Load data ────────────────────────────────────────
if not SKIP_LOAD:
    print("\n" + "=" * 60)
    print("Step 3: Loading CSV data (this will take several minutes)...")
    print("=" * 60)

    for table in TABLES:
        csv_path = DATA_DIR / f"{table}.csv"
        if not csv_path.exists():
            print(f"  SKIP {table} (no CSV)")
            continue

        t0 = time.time()
        with open(csv_path, 'rb') as f:
            cur.copy_expert(
                f"COPY {table} FROM STDIN WITH (FORMAT CSV, NULL '', QUOTE '\"', ESCAPE '\\')",
                f
            )

        cur.execute(f"SELECT COUNT(*) FROM {table}")
        cnt = cur.fetchone()[0]
        elapsed = time.time() - t0
        print(f"  {table:<20} {cnt:>12,} rows  ({elapsed:.1f}s)", flush=True)

    # ── Step 4: Create FK indexes ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Creating indexes...")
    print("=" * 60)
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_ci_movie ON cast_info(movie_id)",
        "CREATE INDEX IF NOT EXISTS idx_ci_person ON cast_info(person_id)",
        "CREATE INDEX IF NOT EXISTS idx_ci_role ON cast_info(role_id)",
        "CREATE INDEX IF NOT EXISTS idx_mi_movie ON movie_info(movie_id)",
        "CREATE INDEX IF NOT EXISTS idx_mi_infotype ON movie_info(info_type_id)",
        "CREATE INDEX IF NOT EXISTS idx_mii_movie ON movie_info_idx(movie_id)",
        "CREATE INDEX IF NOT EXISTS idx_mk_movie ON movie_keyword(movie_id)",
        "CREATE INDEX IF NOT EXISTS idx_mk_keyword ON movie_keyword(keyword_id)",
        "CREATE INDEX IF NOT EXISTS idx_mc_movie ON movie_companies(movie_id)",
        "CREATE INDEX IF NOT EXISTS idx_mc_company ON movie_companies(company_id)",
        "CREATE INDEX IF NOT EXISTS idx_ml_movie ON movie_link(movie_id)",
        "CREATE INDEX IF NOT EXISTS idx_cc_movie ON complete_cast(movie_id)",
        "CREATE INDEX IF NOT EXISTS idx_an_person ON aka_name(person_id)",
        "CREATE INDEX IF NOT EXISTS idx_at_movie ON aka_title(movie_id)",
        "CREATE INDEX IF NOT EXISTS idx_pi_person ON person_info(person_id)",
        "CREATE INDEX IF NOT EXISTS idx_pi_infotype ON person_info(info_type_id)",
        "CREATE INDEX IF NOT EXISTS idx_t_kind ON title(kind_id)",
        "CREATE INDEX IF NOT EXISTS idx_t_year ON title(production_year)",
    ]
    for idx_sql in indexes:
        cur.execute(idx_sql)
    print(f"  Created {len(indexes)} indexes")

    print("\nRunning ANALYZE...", flush=True)
    cur.execute("ANALYZE")
    print("  Done")

# ── Step 5: Run original JOB queries ─────────────────────────
print("\n" + "=" * 60)
print("Step 5: Running 113 JOB queries with EXPLAIN ANALYZE...")
print("=" * 60)

def q_error(est, act):
    e = max(est, 0.5)
    a = max(act, 0.5)
    return max(e / a, a / e)

query_files = sorted(QUERY_DIR.glob("*.sql"),
                     key=lambda p: (re.match(r"(\d+)", p.stem).group().zfill(3) if re.match(r"(\d+)", p.stem) else "999") + p.stem)

print(f"\n  {'Q':<8} {'Actual':>10} {'Est':>10} {'SubAgg-QE':>12} {'Node':>16} {'Bias':>6} {'Joins':>6} {'ms':>7}")
print("  " + "-" * 85)

results = []
for qf in query_files:
    qid = qf.stem
    sql_text = qf.read_text(encoding="utf-8").strip().rstrip(";")

    # Wrap in COUNT(*) for EXPLAIN ANALYZE
    try:
        t0 = time.time()
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql_text}")
        plan_json = cur.fetchone()[0]
        elapsed = (time.time() - t0) * 1000
        plan = plan_json[0]["Plan"]

        def walk(node, depth=0):
            res = []
            nt = node.get("Node Type", "")
            est = float(node.get("Plan Rows", 1))
            loops = max(int(node.get("Actual Loops", 1)), 1)
            act = float(node.get("Actual Rows", 0)) * loops
            res.append((depth, nt, est, max(act, 0.5), q_error(est, act), math.log2(max(est,0.5)/max(act,0.5))))
            for sub in node.get("Plans", []):
                res.extend(walk(sub, depth+1))
            return res

        all_nodes = walk(plan)
        n_joins = sum(1 for _, nt, _, _, _, _ in all_nodes if 'Join' in nt or 'Nested' in nt)

        # Sub-aggregate
        root = plan
        skip = {"Aggregate", "Result", "Sort", "Limit", "Unique", "Gather", "Gather Merge"}
        while root.get("Node Type", "") in skip:
            children = root.get("Plans", [])
            if not children: break
            root = children[0]
        sub_est = float(root.get("Plan Rows", 1))
        sub_loops = max(int(root.get("Actual Loops", 1)), 1)
        sub_act = float(root.get("Actual Rows", 0)) * sub_loops
        sub_qe = q_error(sub_est, sub_act)
        sub_bias = "UNDER" if sub_est < sub_act else ("OVER" if sub_est > sub_act else "EXACT")

        # Worst node
        real_nodes = [n for n in all_nodes if n[3] > 0.5]
        worst = max(real_nodes, key=lambda x: x[4]) if real_nodes else max(all_nodes, key=lambda x: x[4])

        print(f"  {qid:<8} {sub_act:>10,.0f} {sub_est:>10,.0f} {sub_qe:>10.1f}x  {worst[1]:>16} {sub_bias:>6} {n_joins:>6} {elapsed:>6.0f}", flush=True)
        results.append({
            "query": qid,
            "sub_agg_actual": sub_act,
            "sub_agg_est": sub_est,
            "sub_agg_qe": round(sub_qe, 2),
            "sub_agg_bias": sub_bias,
            "worst_qe": round(worst[4], 2),
            "worst_node": worst[1],
            "n_joins": n_joins,
            "ms": round(elapsed, 1),
        })
    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        print(f"  {qid:<8}  ERROR ({elapsed:.0f}ms): {str(ex)[:70]}", flush=True)

# ── Summary ──────────────────────────────────────────────────
ok = [r for r in results if "sub_agg_qe" in r]
sub_qe = [r["sub_agg_qe"] for r in ok]
worst_qe = [r["worst_qe"] for r in ok]
under = [r for r in ok if r["sub_agg_bias"] == "UNDER"]

print(f"\n{'='*80}")
print(f"REAL IMDB JOB BENCHMARK SUMMARY ({len(ok)} queries)")
print(f"{'='*80}")
print(f"  UNDER-estimates:         {len(under)}/{len(ok)}")
print(f"  Sub-agg q-error > 2x:    {sum(1 for q in sub_qe if q > 2)}/{len(sub_qe)}")
print(f"  Sub-agg q-error > 10x:   {sum(1 for q in sub_qe if q > 10)}/{len(sub_qe)}")
print(f"  Sub-agg q-error > 100x:  {sum(1 for q in sub_qe if q > 100)}/{len(sub_qe)}")
print(f"  Sub-agg q-error > 1000x: {sum(1 for q in sub_qe if q > 1000)}/{len(sub_qe)}")
print(f"  Sub-agg median:          {sorted(sub_qe)[len(sub_qe)//2]:,.1f}x")
print(f"  Sub-agg mean:            {sum(sub_qe)/len(sub_qe):,.1f}x")
print(f"  Sub-agg max:             {max(sub_qe):,.1f}x")
print(f"  Worst-node median:       {sorted(worst_qe)[len(worst_qe)//2]:,.1f}x")
print(f"  Worst-node max:          {max(worst_qe):,.1f}x")

out = Path(__file__).parent / "job_real_imdb_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ok[0].keys()))
    w.writeheader()
    w.writerows(ok)
print(f"  CSV -> {out.name}")

cur.close()
conn.close()
