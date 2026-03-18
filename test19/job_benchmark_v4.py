"""
JOB-Style Benchmark v4 FINAL: Converted from test17 to proper JOB schema.

Focus on FK-path fan-out queries that produce UNDER-estimation.
The key patterns:
1. Multiple roles hitting same movie (actor+writer, director+producer)
2. cast_info joining to both person_info AND movie_info paths
3. movie_info self-joins with broader IN() predicates
4. aka_title + aka_name producing unexpected fan-out
5. complete_cast / movie_info_idx creating hidden correlations

All queries use proper JOB-style lookup table joins (role_type, info_type, etc.)
instead of raw numeric IDs.
"""
import sys, io, time, math, psycopg2, csv, re
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB = dict(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')
conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

QUERY_DIR = Path(__file__).parent / "job_v4"
QUERY_DIR.mkdir(exist_ok=True)

# ─── Define queries ──────────────────────────────────────────
# Each: (file_stem, n_joins, description, SQL)
QUERIES = [

# === Multi-role: same movie has actor AND writer/director ===
("j1a", "4J", "actor+writer same movie, US company, year>2005",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci1,
     cast_info ci2,
     movie_companies mc,
     company_name cn,
     role_type rt1,
     role_type rt2
WHERE ci1.movie_id = t.id
  AND ci2.movie_id = t.id
  AND mc.movie_id = t.id
  AND mc.company_id = cn.id
  AND rt1.id = ci1.role_id
  AND rt2.id = ci2.role_id
  AND rt1.role = 'actor'
  AND rt2.role = 'writer'
  AND cn.country_code = '[us]'
  AND t.production_year > 2005;"""),

("j1b", "4J", "actor+director same movie + keyword",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci1,
     cast_info ci2,
     movie_keyword mk,
     role_type rt1,
     role_type rt2
WHERE ci1.movie_id = t.id
  AND ci2.movie_id = t.id
  AND mk.movie_id = t.id
  AND rt1.id = ci1.role_id
  AND rt2.id = ci2.role_id
  AND rt1.role = 'actor'
  AND rt2.role = 'director'
  AND t.production_year > 2000;"""),

("j1c", "5J", "actress+producer+composer same movie + company",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci1,
     cast_info ci2,
     cast_info ci3,
     movie_companies mc,
     role_type rt1,
     role_type rt2,
     role_type rt3
WHERE ci1.movie_id = t.id
  AND ci2.movie_id = t.id
  AND ci3.movie_id = t.id
  AND mc.movie_id = t.id
  AND rt1.id = ci1.role_id
  AND rt2.id = ci2.role_id
  AND rt3.id = ci3.role_id
  AND rt1.role = 'actress'
  AND rt2.role = 'producer'
  AND rt3.role = 'composer';"""),

# === FK fan-out: cast + movie_info + keyword + company (star join) ===
("j2a", "4J", "actor + keyword + genre(Drama,Thriller,Crime)",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci,
     movie_keyword mk,
     movie_info mi,
     role_type rt,
     info_type it
WHERE ci.movie_id = t.id
  AND mk.movie_id = t.id
  AND mi.movie_id = t.id
  AND rt.id = ci.role_id
  AND it.id = mi.info_type_id
  AND rt.role = 'actor'
  AND it.info = 'genres'
  AND mi.info IN ('Drama', 'Thriller', 'Crime')
  AND t.production_year > 2005;"""),

("j2b", "5J", "cast + keyword + genre + company + year>2010",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci,
     movie_keyword mk,
     movie_info mi,
     movie_companies mc,
     info_type it
WHERE ci.movie_id = t.id
  AND mk.movie_id = t.id
  AND mi.movie_id = t.id
  AND mc.movie_id = t.id
  AND it.id = mi.info_type_id
  AND it.info = 'genres'
  AND mi.info IN ('Action', 'Sci-Fi')
  AND t.production_year > 2010;"""),

# === EAV paths via movie_info joining to cast ===
("j4a", "5J", "genre(Drama) + country(USA) + actor + keyword",
"""SELECT COUNT(*)
FROM movie_info mi1,
     movie_info mi2,
     cast_info ci,
     movie_keyword mk,
     title t,
     info_type it1,
     info_type it2,
     role_type rt
WHERE mi1.movie_id = t.id
  AND mi2.movie_id = t.id
  AND ci.movie_id = t.id
  AND mk.movie_id = t.id
  AND it1.id = mi1.info_type_id
  AND it2.id = mi2.info_type_id
  AND rt.id = ci.role_id
  AND it1.info = 'genres'
  AND mi1.info = 'Drama'
  AND it2.info = 'countries'
  AND mi2.info = 'USA'
  AND rt.role = 'actor';"""),

("j4b", "6J", "genre(Action) + country(USA) + cert(R) + actor + keyword + company",
"""SELECT COUNT(*)
FROM title t,
     movie_info mi1,
     movie_info mi2,
     movie_info mi3,
     cast_info ci,
     movie_keyword mk,
     movie_companies mc,
     info_type it1,
     info_type it2,
     info_type it3,
     role_type rt
WHERE mi1.movie_id = t.id
  AND mi2.movie_id = t.id
  AND mi3.movie_id = t.id
  AND ci.movie_id = t.id
  AND mk.movie_id = t.id
  AND mc.movie_id = t.id
  AND it1.id = mi1.info_type_id
  AND it2.id = mi2.info_type_id
  AND it3.id = mi3.info_type_id
  AND rt.id = ci.role_id
  AND it1.info = 'genres'
  AND mi1.info = 'Action'
  AND it2.info = 'countries'
  AND mi2.info = 'USA'
  AND it3.info = 'certificates'
  AND mi3.info = 'R'
  AND rt.role = 'actor';"""),

# === Person-centric paths: person_info -> cast -> movie_info ===
("j6a", "5J", "person(nat=American) + actor + genre(Drama) + company + keyword",
"""SELECT COUNT(*)
FROM person_info pi,
     cast_info ci,
     movie_info mi,
     movie_companies mc,
     movie_keyword mk,
     info_type it1,
     info_type it2,
     role_type rt
WHERE pi.person_id = ci.person_id
  AND mi.movie_id = ci.movie_id
  AND mc.movie_id = ci.movie_id
  AND mk.movie_id = ci.movie_id
  AND it1.id = pi.info_type_id
  AND it2.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND it1.info = 'nationality'
  AND pi.info = 'American'
  AND rt.role = 'actor'
  AND it2.info = 'genres'
  AND mi.info = 'Drama';"""),

("j6b", "5J", "person(nat IN) + actress + genre(Horror,Thriller) + keyword",
"""SELECT COUNT(*)
FROM person_info pi,
     cast_info ci,
     movie_info mi,
     movie_keyword mk,
     title t,
     info_type it1,
     info_type it2,
     role_type rt
WHERE pi.person_id = ci.person_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND mk.movie_id = t.id
  AND it1.id = pi.info_type_id
  AND it2.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND it1.info = 'nationality'
  AND pi.info IN ('Japanese', 'South Korean')
  AND rt.role = 'actress'
  AND it2.info = 'genres'
  AND mi.info IN ('Horror', 'Thriller');"""),

# === aka fan-out: aka_title + aka_name amplify rows ===
("j8a", "5J", "cast + aka_name + aka_title + genre(Drama) + year>2010",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci,
     aka_name an,
     aka_title at,
     movie_info mi,
     info_type it
WHERE ci.movie_id = t.id
  AND an.person_id = ci.person_id
  AND at.movie_id = t.id
  AND mi.movie_id = t.id
  AND it.id = mi.info_type_id
  AND it.info = 'genres'
  AND mi.info = 'Drama'
  AND t.production_year > 2010;"""),

("j8b", "6J", "cast + aka_name + keyword + company + genre + country",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci,
     aka_name an,
     movie_keyword mk,
     movie_companies mc,
     movie_info mi1,
     movie_info mi2,
     info_type it1,
     info_type it2
WHERE ci.movie_id = t.id
  AND an.person_id = ci.person_id
  AND mk.movie_id = t.id
  AND mc.movie_id = t.id
  AND mi1.movie_id = t.id
  AND mi2.movie_id = t.id
  AND it1.id = mi1.info_type_id
  AND it2.id = mi2.info_type_id
  AND it1.info = 'genres'
  AND mi1.info IN ('Action', 'Thriller')
  AND it2.info = 'countries'
  AND mi2.info IN ('USA', 'UK');"""),

# === complete_cast / movie_info_idx hidden correlations ===
("j9a", "5J", "complete_cast + genre(Drama) + actor + company + year>2000",
"""SELECT COUNT(*)
FROM title t,
     complete_cast cc,
     cast_info ci,
     movie_info mi,
     movie_companies mc,
     info_type it,
     role_type rt
WHERE cc.movie_id = t.id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND mc.movie_id = t.id
  AND it.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND it.info = 'genres'
  AND mi.info = 'Drama'
  AND rt.role = 'actor'
  AND t.production_year > 2000;"""),

("j9b", "5J", "movie_info_idx(votes) + genre + cast + keyword + company",
"""SELECT COUNT(*)
FROM title t,
     movie_info_idx mii,
     cast_info ci,
     movie_info mi,
     movie_keyword mk,
     movie_companies mc,
     info_type it1,
     info_type it2,
     role_type rt
WHERE mii.movie_id = t.id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND mk.movie_id = t.id
  AND mc.movie_id = t.id
  AND it1.id = mii.info_type_id
  AND it2.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND it1.info = 'votes'
  AND it2.info = 'genres'
  AND mi.info IN ('Action', 'Drama')
  AND rt.role IN ('actor', 'actress');"""),

# === Multi-path: two independent paths merge on title ===
("j10a", "6J", "cast-path + link-path merge: cast+keyword+link+aka_title+genre",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci,
     movie_keyword mk,
     movie_link ml,
     aka_title at,
     movie_info mi,
     info_type it,
     role_type rt
WHERE ci.movie_id = t.id
  AND mk.movie_id = t.id
  AND ml.movie_id = t.id
  AND at.movie_id = t.id
  AND mi.movie_id = t.id
  AND it.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND it.info = 'genres'
  AND mi.info IN ('Action', 'Comedy', 'Drama')
  AND rt.role = 'actor';"""),

("j10b", "7J", "cast-path + company-path + EAV: actor+keyword+company+genre+country+cert",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci,
     movie_keyword mk,
     movie_companies mc,
     movie_info mi1,
     movie_info mi2,
     movie_info mi3,
     info_type it1,
     info_type it2,
     info_type it3,
     role_type rt
WHERE ci.movie_id = t.id
  AND mk.movie_id = t.id
  AND mc.movie_id = t.id
  AND mi1.movie_id = t.id
  AND mi2.movie_id = t.id
  AND mi3.movie_id = t.id
  AND it1.id = mi1.info_type_id
  AND it2.id = mi2.info_type_id
  AND it3.id = mi3.info_type_id
  AND rt.id = ci.role_id
  AND rt.role = 'actor'
  AND it1.info = 'genres'
  AND mi1.info = 'Drama'
  AND it2.info = 'countries'
  AND mi2.info = 'USA'
  AND it3.info = 'certificates'
  AND mi3.info = 'R';"""),
]

# ─── Write individual SQL files ──────────────────────────────
for stem, _, desc, sql in QUERIES:
    path = QUERY_DIR / f"{stem}.sql"
    path.write_text(f"-- {desc}\n{sql.strip()}\n", encoding="utf-8")
print(f"Wrote {len(QUERIES)} SQL files to {QUERY_DIR}/")

# ─── Run benchmark ──────────────────────────────────────────
cur.execute("ANALYZE")

def q_error(est, act):
    e = max(est, 0.5)
    a = max(act, 0.5)
    return max(e / a, a / e)

print(f"\nRunning {len(QUERIES)} JOB-style v4 queries on {DB['dbname']}...")
print(f"  {'Q':<6} {'#J':<6} {'Description':<46} {'Q-Error':>12} {'Est':>10} {'Act':>10} {'ms':>7}  Bias")
print("  " + "-" * 106)

results = []
for qid, nj, desc, sql in QUERIES:
    try:
        t0 = time.time()
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
        plan_json = cur.fetchone()[0]
        elapsed = (time.time() - t0) * 1000
        plan = plan_json[0]["Plan"]

        def walk(node, depth=0):
            res = []
            nt = node.get("Node Type", "")
            est = float(node.get("Plan Rows", 1))
            loops = max(int(node.get("Actual Loops", 1)), 1)
            act = float(node.get("Actual Rows", 0)) * loops
            act = max(act, 0.5)
            if est <= 0: est = 0.5
            qe = max(est/act, act/est)
            bias = math.log2(est/act)
            res.append((depth, nt, est, act, qe, bias))
            for sub in node.get("Plans", []):
                res.extend(walk(sub, depth+1))
            return res

        all_nodes = walk(plan)
        real_nodes = [n for n in all_nodes if n[3] > 0.5]
        worst = max(real_nodes, key=lambda x: x[4]) if real_nodes else max(all_nodes, key=lambda x: x[4])
        _, _, w_est, w_act, w_qe, w_bias = worst

        # Sub-aggregate node
        root = plan
        skip = {"Aggregate", "Result", "Sort", "Limit", "Unique"}
        while root.get("Node Type", "") in skip:
            children = root.get("Plans", [])
            if not children: break
            root = children[0]
        sub_est = float(root.get("Plan Rows", 1))
        sub_loops = max(int(root.get("Actual Loops", 1)), 1)
        sub_act = float(root.get("Actual Rows", 0)) * sub_loops
        sub_qe = q_error(sub_est, sub_act)

        bias_dir = "UNDER" if w_bias < 0 else "OVER"

        print(f"  {qid:<6} {nj:<6} {desc[:44]:<46} {w_qe:>10.1f}x  {w_est:>9.0f} {w_act:>9.0f} {elapsed:>6.0f}  {bias_dir}", flush=True)
        results.append({
            "query": qid,
            "joins": nj,
            "desc": desc,
            "worst_qe": round(w_qe, 2),
            "worst_est": w_est,
            "worst_act": w_act,
            "worst_bias": round(w_bias, 2),
            "bias_dir": bias_dir,
            "sub_agg_qe": round(sub_qe, 2),
            "sub_agg_est": sub_est,
            "sub_agg_act": sub_act,
            "ms": round(elapsed, 1),
        })
    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        print(f"  {qid:<6} {nj:<6} {desc[:44]:<46}  ERROR ({elapsed:.0f}ms): {str(ex)[:60]}", flush=True)

# ─── Summary ──────────────────────────────────────────────
real_qe = [r["worst_qe"] for r in results]
sub_qe = [r["sub_agg_qe"] for r in results]
under = [r for r in results if r["bias_dir"] == 'UNDER']

print(f"\n{'='*80}")
print(f"SUMMARY ({len(results)} queries)")
print(f"{'='*80}")
print(f"  UNDER-estimates: {len(under)}/{len(results)}")
print(f"  Worst-node q-error > 100x:  {sum(1 for q in real_qe if q > 100)}/{len(real_qe)}")
print(f"  Worst-node q-error > 1000x: {sum(1 for q in real_qe if q > 1000)}/{len(real_qe)}")
print(f"  Max worst-node q-error:     {max(real_qe):,.1f}x")
print(f"  Median worst-node q-error:  {sorted(real_qe)[len(real_qe)//2]:,.1f}x")
print(f"\n  Sub-agg q-error median:     {sorted(sub_qe)[len(sub_qe)//2]:,.1f}x")
print(f"  Sub-agg q-error max:        {max(sub_qe):,.1f}x")

out = Path(__file__).parent / "job_v4_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print(f"  CSV -> {out.name}")

cur.close()
conn.close()
