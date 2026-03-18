"""
Proper analysis: compare ACTUAL data correlations and q-error by join complexity
between real IMDB and synthetic dataset.

1. Direct correlation checks on both DBs
2. Q-error grouped by join count (the only fair comparison axis)
"""
import psycopg2, sys, io, math, csv, statistics

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REAL = dict(host='localhost', port=5433, dbname='imdb_real', user='postgres', password='bench')
SYNTH = dict(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')

def query(db, sql):
    conn = psycopg2.connect(**db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    cur.close()
    conn.close()
    return cols, rows

# ══════════════════════════════════════════════════════════════
# PART 1: CORRELATION CHECKS
# ══════════════════════════════════════════════════════════════
print("=" * 90)
print("PART 1: DATA CORRELATION COMPARISON")
print("     Checking if key column pairs have similar correlation patterns")
print("=" * 90)

CORR_QUERIES = [
    # Genre-country correlation: are certain genres more common in certain countries?
    ("Genre × Country distribution",
     """SELECT mi1.info AS genre, mi2.info AS country, COUNT(*) AS cnt
        FROM movie_info mi1
        JOIN movie_info mi2 ON mi1.movie_id = mi2.movie_id
        JOIN info_type it1 ON it1.id = mi1.info_type_id
        JOIN info_type it2 ON it2.id = mi2.info_type_id
        WHERE it1.info = 'genres' AND it2.info = 'countries'
        GROUP BY mi1.info, mi2.info
        ORDER BY cnt DESC
        LIMIT 15"""),

    # Actor-genre correlation: do actors cluster in genres?
    ("Top actor-genre combos (role_id for actor)",
     """SELECT n.name, mi.info AS genre, COUNT(*) AS cnt
        FROM cast_info ci
        JOIN name n ON n.id = ci.person_id
        JOIN movie_info mi ON mi.movie_id = ci.movie_id
        JOIN info_type it ON it.id = mi.info_type_id
        JOIN role_type rt ON rt.id = ci.role_id
        WHERE it.info = 'genres' AND rt.role = 'actor'
        GROUP BY n.name, mi.info
        ORDER BY cnt DESC
        LIMIT 15"""),

    # Company-genre: do studios specialize?
    ("Company × Genre specialization",
     """SELECT cn.name, mi.info AS genre, COUNT(*) AS cnt
        FROM movie_companies mc
        JOIN company_name cn ON cn.id = mc.company_id
        JOIN movie_info mi ON mi.movie_id = mc.movie_id
        JOIN info_type it ON it.id = mi.info_type_id
        WHERE it.info = 'genres'
        GROUP BY cn.name, mi.info
        ORDER BY cnt DESC
        LIMIT 15"""),

    # Genre × certificate (rating) correlation
    ("Genre × Certificate",
     """SELECT mi1.info AS genre, mi2.info AS cert, COUNT(*) AS cnt
        FROM movie_info mi1
        JOIN movie_info mi2 ON mi1.movie_id = mi2.movie_id
        JOIN info_type it1 ON it1.id = mi1.info_type_id
        JOIN info_type it2 ON it2.id = mi2.info_type_id
        WHERE it1.info = 'genres' AND it2.info = 'certificates'
        GROUP BY mi1.info, mi2.info
        ORDER BY cnt DESC
        LIMIT 10"""),

    # Year × genre correlation  
    ("Year × Genre trend (2010+)",
     """SELECT t.production_year, mi.info AS genre, COUNT(*) AS cnt
        FROM title t
        JOIN movie_info mi ON mi.movie_id = t.id
        JOIN info_type it ON it.id = mi.info_type_id
        WHERE it.info = 'genres' AND t.production_year >= 2010
        GROUP BY t.production_year, mi.info
        ORDER BY cnt DESC
        LIMIT 15"""),
]

for title, sql in CORR_QUERIES:
    print(f"\n  ── {title} ──")
    print(f"  {'REAL IMDB':<50} {'SYNTHETIC':<50}")
    print(f"  {'-'*50} {'-'*50}")
    try:
        _, r_rows = query(REAL, sql)
        _, s_rows = query(SYNTH, sql)
        for i in range(max(len(r_rows), len(s_rows))):
            r_str = ""
            s_str = ""
            if i < len(r_rows):
                parts = [str(x)[:18] for x in r_rows[i]]
                r_str = " | ".join(parts)
            if i < len(s_rows):
                parts = [str(x)[:18] for x in s_rows[i]]
                s_str = " | ".join(parts)
            print(f"  {r_str:<50} {s_str:<50}")
    except Exception as e:
        print(f"  ERROR: {e}")

# ══════════════════════════════════════════════════════════════
# PART 2: SELECTIVITY FRACTIONS
# ══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 90}")
print("PART 2: SELECTIVITY FRACTIONS")
print("     Same predicate, what % of total rows does it select?")
print("=" * 90)

SEL_QUERIES = [
    ("Genre=Drama fraction of movies",
     "SELECT COUNT(DISTINCT mi.movie_id)::float / (SELECT COUNT(*) FROM title) FROM movie_info mi JOIN info_type it ON it.id = mi.info_type_id WHERE it.info = 'genres' AND mi.info = 'Drama'"),
    
    ("Genre=Action fraction",
     "SELECT COUNT(DISTINCT mi.movie_id)::float / (SELECT COUNT(*) FROM title) FROM movie_info mi JOIN info_type it ON it.id = mi.info_type_id WHERE it.info = 'genres' AND mi.info = 'Action'"),

    ("Country=USA fraction",
     "SELECT COUNT(DISTINCT mi.movie_id)::float / (SELECT COUNT(*) FROM title) FROM movie_info mi JOIN info_type it ON it.id = mi.info_type_id WHERE it.info = 'countries' AND mi.info = 'USA'"),

    ("Cert=R fraction",
     "SELECT COUNT(DISTINCT mi.movie_id)::float / (SELECT COUNT(*) FROM title) FROM movie_info mi JOIN info_type it ON it.id = mi.info_type_id WHERE it.info = 'certificates' AND mi.info = 'R'"),

    ("Actor role fraction of cast_info",
     "SELECT COUNT(*)::float / (SELECT COUNT(*) FROM cast_info) FROM cast_info ci JOIN role_type rt ON rt.id = ci.role_id WHERE rt.role = 'actor'"),

    ("Avg genres per movie",
     "SELECT AVG(c) FROM (SELECT mi.movie_id, COUNT(*) c FROM movie_info mi JOIN info_type it ON it.id = mi.info_type_id WHERE it.info = 'genres' GROUP BY mi.movie_id) x"),

    ("Avg cast per movie",
     "SELECT AVG(c) FROM (SELECT movie_id, COUNT(*) c FROM cast_info GROUP BY movie_id) x"),

    ("Avg companies per movie",
     "SELECT AVG(c) FROM (SELECT movie_id, COUNT(*) c FROM movie_companies GROUP BY movie_id) x"),

    ("Avg keywords per movie",
     "SELECT AVG(c) FROM (SELECT movie_id, COUNT(*) c FROM movie_keyword GROUP BY movie_id) x"),

    ("Movies with year > 2010 fraction",
     "SELECT COUNT(*)::float / (SELECT COUNT(*) FROM title) FROM title WHERE production_year > 2010"),
]

print(f"\n  {'Metric':<40} {'Real IMDB':>12} {'Synthetic':>12} {'Ratio':>8}")
print(f"  {'-'*75}")
for title, sql in SEL_QUERIES:
    try:
        _, r = query(REAL, sql)
        _, s = query(SYNTH, sql)
        rv = float(r[0][0]) if r[0][0] else 0
        sv = float(s[0][0]) if s[0][0] else 0
        ratio = rv / sv if sv > 0 else float('inf')
        print(f"  {title:<40} {rv:>12.4f} {sv:>12.4f} {ratio:>7.2f}x")
    except Exception as e:
        print(f"  {title:<40} ERROR: {e}")

# ══════════════════════════════════════════════════════════════
# PART 3: CONDITIONAL SELECTIVITIES (actual correlations)
# ══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 90}")
print("PART 3: CONDITIONAL SELECTIVITIES (correlation strength)")
print("     P(Country=USA | Genre=Action) vs P(Country=USA) — if different = correlation")
print("=" * 90)

COND_QUERIES = [
    ("P(USA | Action) vs P(USA)",
     "SELECT COUNT(DISTINCT mi2.movie_id)::float / NULLIF(COUNT(DISTINCT mi1.movie_id), 0) FROM movie_info mi1 JOIN info_type it1 ON it1.id = mi1.info_type_id JOIN movie_info mi2 ON mi2.movie_id = mi1.movie_id JOIN info_type it2 ON it2.id = mi2.info_type_id WHERE it1.info = 'genres' AND mi1.info = 'Action' AND it2.info = 'countries' AND mi2.info = 'USA'",
     "SELECT COUNT(DISTINCT mi.movie_id)::float / (SELECT COUNT(DISTINCT id) FROM title) FROM movie_info mi JOIN info_type it ON it.id = mi.info_type_id WHERE it.info = 'countries' AND mi.info = 'USA'"),

    ("P(USA | Drama) vs P(USA)",
     "SELECT COUNT(DISTINCT mi2.movie_id)::float / NULLIF(COUNT(DISTINCT mi1.movie_id), 0) FROM movie_info mi1 JOIN info_type it1 ON it1.id = mi1.info_type_id JOIN movie_info mi2 ON mi2.movie_id = mi1.movie_id JOIN info_type it2 ON it2.id = mi2.info_type_id WHERE it1.info = 'genres' AND mi1.info = 'Drama' AND it2.info = 'countries' AND mi2.info = 'USA'",
     "SELECT COUNT(DISTINCT mi.movie_id)::float / (SELECT COUNT(DISTINCT id) FROM title) FROM movie_info mi JOIN info_type it ON it.id = mi.info_type_id WHERE it.info = 'countries' AND mi.info = 'USA'"),

    ("P(R-rated | Horror) vs P(R-rated)",
     "SELECT COUNT(DISTINCT mi2.movie_id)::float / NULLIF(COUNT(DISTINCT mi1.movie_id), 0) FROM movie_info mi1 JOIN info_type it1 ON it1.id = mi1.info_type_id JOIN movie_info mi2 ON mi2.movie_id = mi1.movie_id JOIN info_type it2 ON it2.id = mi2.info_type_id WHERE it1.info = 'genres' AND mi1.info = 'Horror' AND it2.info = 'certificates' AND mi2.info = 'R'",
     "SELECT COUNT(DISTINCT mi.movie_id)::float / (SELECT COUNT(DISTINCT id) FROM title) FROM movie_info mi JOIN info_type it ON it.id = mi.info_type_id WHERE it.info = 'certificates' AND mi.info = 'R'"),

    ("P(Actor | year>2010) vs P(Actor)",
     "SELECT COUNT(DISTINCT ci.id)::float / NULLIF((SELECT COUNT(*) FROM cast_info ci2 JOIN title t2 ON t2.id = ci2.movie_id WHERE t2.production_year > 2010), 0) FROM cast_info ci JOIN title t ON t.id = ci.movie_id JOIN role_type rt ON rt.id = ci.role_id WHERE rt.role = 'actor' AND t.production_year > 2010",
     "SELECT COUNT(*)::float / (SELECT COUNT(*) FROM cast_info) FROM cast_info ci JOIN role_type rt ON rt.id = ci.role_id WHERE rt.role = 'actor'"),
]

print(f"\n  {'Correlation':<30} {'Real Cond':>10} {'Real Base':>10} {'R Lift':>8} {'Syn Cond':>10} {'Syn Base':>10} {'S Lift':>8}")
print(f"  {'-'*88}")
for title, cond_sql, base_sql in COND_QUERIES:
    try:
        _, rc = query(REAL, cond_sql)
        _, rb = query(REAL, base_sql)
        _, sc = query(SYNTH, cond_sql)
        _, sb = query(SYNTH, base_sql)
        rcv = float(rc[0][0]) if rc[0][0] else 0
        rbv = float(rb[0][0]) if rb[0][0] else 0
        scv = float(sc[0][0]) if sc[0][0] else 0
        sbv = float(sb[0][0]) if sb[0][0] else 0
        r_lift = rcv / rbv if rbv > 0 else 0
        s_lift = scv / sbv if sbv > 0 else 0
        print(f"  {title:<30} {rcv:>10.4f} {rbv:>10.4f} {r_lift:>7.2f}x {scv:>10.4f} {sbv:>10.4f} {s_lift:>7.2f}x")
    except Exception as e:
        print(f"  {title:<30} ERROR: {e}")

# ══════════════════════════════════════════════════════════════
# PART 4: Q-ERROR BY JOIN COUNT (fair comparison)
# ══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 90}")
print("PART 4: Q-ERROR BY JOIN COUNT (fair comparison)")
print("     Group queries by #joins, compare distributions")
print("=" * 90)

real_d, synth_d = {}, {}
with open("job_real_imdb_results.csv") as f:
    for r in csv.DictReader(f): real_d[r["query"]] = r
with open("job_adapted_qerror_per_query.csv") as f:
    for r in csv.DictReader(f): synth_d[r["query"]] = r

queries = sorted(set(real_d.keys()) & set(synth_d.keys()))

# Group by join count (use real's join count as structure proxy)
by_joins = {}
for q in queries:
    nj = int(real_d[q].get("n_joins", 0))
    if nj not in by_joins:
        by_joins[nj] = []
    by_joins[nj].append(q)

print(f"\n  {'Joins':>5} {'Count':>6} {'Real Med QE':>12} {'Synth Med QE':>13} {'Real Mean':>10} {'Synth Mean':>11} {'R 90pct':>10} {'S 90pct':>10}")
print(f"  {'-'*80}")
for nj in sorted(by_joins.keys()):
    qs = by_joins[nj]
    r_qe = sorted([float(real_d[q]["sub_agg_qe"]) for q in qs])
    s_qe = sorted([float(synth_d[q]["sub_agg_qerror"]) for q in qs])
    r_med = statistics.median(r_qe)
    s_med = statistics.median(s_qe)
    r_mean = statistics.mean(r_qe)
    s_mean = statistics.mean(s_qe)
    r_90 = r_qe[int(len(r_qe)*0.9)] if len(r_qe) > 1 else r_qe[0]
    s_90 = s_qe[int(len(s_qe)*0.9)] if len(s_qe) > 1 else s_qe[0]
    print(f"  {nj:>5} {len(qs):>6} {r_med:>12,.1f}x {s_med:>13,.1f}x {r_mean:>10,.1f}x {s_mean:>11,.1f}x {r_90:>10,.1f}x {s_90:>10,.1f}x")

print("\nDone.")
