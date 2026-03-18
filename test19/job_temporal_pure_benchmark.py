"""
Pure Temporal Q-Error Benchmark: Queries testing how PostgreSQL estimates
temporal predicates and temporal correlations within the IMDB schema.

NOT edge queries — these test temporal reasoning on production_year,
birth dates, release dates, series years, and temporal correlations
between person age/career-span and movie production timelines.

Key patterns that challenge estimation:
1. Narrow year ranges on skewed production_year distributions
2. Cross-table temporal correlations (birth year → production year)
3. Temporal self-joins (movies from same year/decade)
4. Decade-boundary effects (movies before vs after a year threshold)
5. Temporal range predicates combined with genre/keyword/company selectivity
6. Series temporal spans (series_years) correlated with episode counts
"""
import sys, io, time, math, psycopg2, csv
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB = dict(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')
conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

QUERY_DIR = Path(__file__).parent / "job_temporal_pure"
QUERY_DIR.mkdir(exist_ok=True)

QUERIES = [

# ═══════════════════════════════════════════════════════════════
# TP1: Production year range selectivity
# (Skew: 1970s have ~100/year, 2010s have ~2800/year)
# ═══════════════════════════════════════════════════════════════
("tp1a", "3J", "movies 1970-1980 + actor + genre(Drama)",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci,
     movie_info mi,
     info_type it,
     role_type rt
WHERE ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND it.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND t.production_year BETWEEN 1970 AND 1980
  AND it.info = 'genres'
  AND mi.info = 'Drama'
  AND rt.role = 'actor';"""),

("tp1b", "3J", "movies 2018-2020 + actor + genre(Drama) — same structure, dense era",
"""SELECT COUNT(*)
FROM title t,
     cast_info ci,
     movie_info mi,
     info_type it,
     role_type rt
WHERE ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND it.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND t.production_year BETWEEN 2018 AND 2020
  AND it.info = 'genres'
  AND mi.info = 'Drama'
  AND rt.role = 'actor';"""),

("tp1c", "4J", "narrow year 2019 + genre + keyword + company",
"""SELECT COUNT(*)
FROM title t,
     movie_info mi,
     movie_keyword mk,
     movie_companies mc,
     info_type it
WHERE mi.movie_id = t.id
  AND mk.movie_id = t.id
  AND mc.movie_id = t.id
  AND it.id = mi.info_type_id
  AND t.production_year = 2019
  AND it.info = 'genres'
  AND mi.info IN ('Action', 'Thriller');"""),

("tp1d", "4J", "narrow year 1985 + genre + keyword + company — sparse era",
"""SELECT COUNT(*)
FROM title t,
     movie_info mi,
     movie_keyword mk,
     movie_companies mc,
     info_type it
WHERE mi.movie_id = t.id
  AND mk.movie_id = t.id
  AND mc.movie_id = t.id
  AND it.id = mi.info_type_id
  AND t.production_year = 1985
  AND it.info = 'genres'
  AND mi.info IN ('Action', 'Thriller');"""),

# ═══════════════════════════════════════════════════════════════
# TP2: Cross-table temporal correlation — person birth year
# vs movie production year (career span queries)
# ═══════════════════════════════════════════════════════════════
("tp2a", "4J", "actors born before 1950 in movies after 2010",
"""SELECT COUNT(*)
FROM cast_info ci,
     person_info pi_bd,
     title t,
     info_type it_bd,
     role_type rt
WHERE ci.person_id = pi_bd.person_id
  AND ci.movie_id = t.id
  AND it_bd.id = pi_bd.info_type_id
  AND rt.id = ci.role_id
  AND it_bd.info = 'birth date'
  AND pi_bd.info < '1950'
  AND t.production_year > 2010
  AND rt.role = 'actor';"""),

("tp2b", "4J", "actors born after 1990 in movies before 2015",
"""SELECT COUNT(*)
FROM cast_info ci,
     person_info pi_bd,
     title t,
     info_type it_bd,
     role_type rt
WHERE ci.person_id = pi_bd.person_id
  AND ci.movie_id = t.id
  AND it_bd.id = pi_bd.info_type_id
  AND rt.id = ci.role_id
  AND it_bd.info = 'birth date'
  AND pi_bd.info > '1990'
  AND t.production_year < 2015
  AND rt.role = 'actor';"""),

("tp2c", "5J", "young actors (born>1985) in old movies (<2005) + genre(Horror)",
"""SELECT COUNT(*)
FROM cast_info ci,
     person_info pi_bd,
     title t,
     movie_info mi,
     info_type it_bd,
     info_type it_g,
     role_type rt
WHERE ci.person_id = pi_bd.person_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND it_bd.id = pi_bd.info_type_id
  AND it_g.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND it_bd.info = 'birth date'
  AND pi_bd.info > '1985'
  AND t.production_year < 2005
  AND it_g.info = 'genres'
  AND mi.info = 'Horror'
  AND rt.role IN ('actor', 'actress');"""),

("tp2d", "5J", "veteran actors (born<1960) in recent movies (>2020) + genre + keyword",
"""SELECT COUNT(*)
FROM cast_info ci,
     person_info pi_bd,
     title t,
     movie_info mi,
     movie_keyword mk,
     info_type it_bd,
     info_type it_g,
     role_type rt
WHERE ci.person_id = pi_bd.person_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND mk.movie_id = t.id
  AND it_bd.id = pi_bd.info_type_id
  AND it_g.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND it_bd.info = 'birth date'
  AND pi_bd.info < '1960'
  AND t.production_year > 2020
  AND it_g.info = 'genres'
  AND mi.info = 'Drama'
  AND rt.role = 'actor';"""),

# ═══════════════════════════════════════════════════════════════
# TP3: Temporal self-joins — two movies from same/different eras
# ═══════════════════════════════════════════════════════════════
("tp3a", "4J", "same actor in 2 movies from different decades",
"""SELECT COUNT(*)
FROM cast_info ci1,
     cast_info ci2,
     title t1,
     title t2,
     role_type rt
WHERE ci1.person_id = ci2.person_id
  AND ci1.movie_id = t1.id
  AND ci2.movie_id = t2.id
  AND rt.id = ci1.role_id
  AND ci1.movie_id != ci2.movie_id
  AND t1.production_year BETWEEN 1990 AND 2000
  AND t2.production_year BETWEEN 2015 AND 2025
  AND rt.role = 'actor';"""),

("tp3b", "5J", "same actor in 2 movies, year gap > 20, + genre(Drama)",
"""SELECT COUNT(*)
FROM cast_info ci1,
     cast_info ci2,
     title t1,
     title t2,
     movie_info mi,
     info_type it,
     role_type rt
WHERE ci1.person_id = ci2.person_id
  AND ci1.movie_id = t1.id
  AND ci2.movie_id = t2.id
  AND mi.movie_id = t1.id
  AND it.id = mi.info_type_id
  AND rt.id = ci1.role_id
  AND ci1.movie_id != ci2.movie_id
  AND t2.production_year - t1.production_year > 20
  AND t1.production_year >= 1980
  AND it.info = 'genres'
  AND mi.info = 'Drama'
  AND rt.role = 'actor';"""),

# ═══════════════════════════════════════════════════════════════
# TP4: Release date patterns combined with production year
# ═══════════════════════════════════════════════════════════════
("tp4a", "4J", "theatrical release + year>2015 + genre(Action) + company",
"""SELECT COUNT(*)
FROM title t,
     movie_info mi_rd,
     movie_info mi_g,
     movie_companies mc,
     info_type it_rd,
     info_type it_g
WHERE mi_rd.movie_id = t.id
  AND mi_g.movie_id = t.id
  AND mc.movie_id = t.id
  AND it_rd.id = mi_rd.info_type_id
  AND it_g.id = mi_g.info_type_id
  AND it_rd.info = 'release dates'
  AND mi_rd.note = '(Theatrical)'
  AND it_g.info = 'genres'
  AND mi_g.info = 'Action'
  AND t.production_year > 2015;"""),

("tp4b", "5J", "streaming release + year 2018-2022 + genre + keyword + US company",
"""SELECT COUNT(*)
FROM title t,
     movie_info mi_rd,
     movie_info mi_g,
     movie_keyword mk,
     movie_companies mc,
     company_name cn,
     info_type it_rd,
     info_type it_g
WHERE mi_rd.movie_id = t.id
  AND mi_g.movie_id = t.id
  AND mk.movie_id = t.id
  AND mc.movie_id = t.id
  AND mc.company_id = cn.id
  AND it_rd.id = mi_rd.info_type_id
  AND it_g.id = mi_g.info_type_id
  AND it_rd.info = 'release dates'
  AND mi_rd.note = '(Streaming)'
  AND it_g.info = 'genres'
  AND mi_g.info IN ('Drama', 'Thriller')
  AND cn.country_code = '[us]'
  AND t.production_year BETWEEN 2018 AND 2022;"""),

# ═══════════════════════════════════════════════════════════════
# TP5: Episode temporal patterns — episodes + series production years
# ═══════════════════════════════════════════════════════════════
("tp5a", "3J", "episodes of series from 2015-2020 + cast",
"""SELECT COUNT(*)
FROM title t_ep,
     title t_series,
     cast_info ci,
     kind_type kt
WHERE t_ep.episode_of_id = t_series.id
  AND ci.movie_id = t_ep.id
  AND kt.id = t_series.kind_id
  AND kt.kind = 'tv series'
  AND t_series.production_year BETWEEN 2015 AND 2020;"""),

("tp5b", "4J", "episodes + series year + genre + keyword",
"""SELECT COUNT(*)
FROM title t_ep,
     title t_series,
     movie_info mi,
     movie_keyword mk,
     kind_type kt,
     info_type it
WHERE t_ep.episode_of_id = t_series.id
  AND mi.movie_id = t_series.id
  AND mk.movie_id = t_ep.id
  AND kt.id = t_series.kind_id
  AND it.id = mi.info_type_id
  AND kt.kind = 'tv series'
  AND t_series.production_year BETWEEN 2015 AND 2020
  AND it.info = 'genres'
  AND mi.info IN ('Drama', 'Crime');"""),

# ═══════════════════════════════════════════════════════════════
# TP6: Birth-death temporal window (career span)
# ═══════════════════════════════════════════════════════════════
("tp6a", "5J", "actors with birth date + nationality + movies in birth decade",
"""SELECT COUNT(*)
FROM cast_info ci,
     person_info pi_bd,
     person_info pi_nat,
     title t,
     info_type it_bd,
     info_type it_nat,
     role_type rt
WHERE ci.person_id = pi_bd.person_id
  AND ci.person_id = pi_nat.person_id
  AND ci.movie_id = t.id
  AND it_bd.id = pi_bd.info_type_id
  AND it_nat.id = pi_nat.info_type_id
  AND rt.id = ci.role_id
  AND it_bd.info = 'birth date'
  AND pi_bd.info BETWEEN '1970' AND '1980'
  AND it_nat.info = 'nationality'
  AND pi_nat.info = 'American'
  AND rt.role = 'actor'
  AND t.production_year > 2010;"""),

# ═══════════════════════════════════════════════════════════════
# TP7: Complex temporal multi-table joins
# ═══════════════════════════════════════════════════════════════
("tp7a", "6J", "actor born 1970s + Drama + US + recent movies + keyword",
"""SELECT COUNT(*)
FROM cast_info ci,
     person_info pi_bd,
     title t,
     movie_info mi_g,
     movie_info mi_c,
     movie_keyword mk,
     info_type it_bd,
     info_type it_g,
     info_type it_c,
     role_type rt
WHERE ci.person_id = pi_bd.person_id
  AND ci.movie_id = t.id
  AND mi_g.movie_id = t.id
  AND mi_c.movie_id = t.id
  AND mk.movie_id = t.id
  AND it_bd.id = pi_bd.info_type_id
  AND it_g.id = mi_g.info_type_id
  AND it_c.id = mi_c.info_type_id
  AND rt.id = ci.role_id
  AND it_bd.info = 'birth date'
  AND pi_bd.info BETWEEN '1970' AND '1980'
  AND it_g.info = 'genres'
  AND mi_g.info = 'Drama'
  AND it_c.info = 'countries'
  AND mi_c.info = 'USA'
  AND rt.role = 'actor'
  AND t.production_year > 2015;"""),

("tp7b", "7J", "actor born 1980s + budget + genre + company + year range + keyword",
"""SELECT COUNT(*)
FROM cast_info ci,
     person_info pi_bd,
     title t,
     movie_info mi_g,
     movie_info mi_b,
     movie_companies mc,
     movie_keyword mk,
     info_type it_bd,
     info_type it_g,
     info_type it_b,
     role_type rt
WHERE ci.person_id = pi_bd.person_id
  AND ci.movie_id = t.id
  AND mi_g.movie_id = t.id
  AND mi_b.movie_id = t.id
  AND mc.movie_id = t.id
  AND mk.movie_id = t.id
  AND it_bd.id = pi_bd.info_type_id
  AND it_g.id = mi_g.info_type_id
  AND it_b.id = mi_b.info_type_id
  AND rt.id = ci.role_id
  AND it_bd.info = 'birth date'
  AND pi_bd.info BETWEEN '1980' AND '1990'
  AND it_g.info = 'genres'
  AND mi_g.info IN ('Action', 'Sci-Fi')
  AND it_b.info = 'budget'
  AND rt.role IN ('actor', 'actress')
  AND t.production_year BETWEEN 2015 AND 2025;"""),

# ═══════════════════════════════════════════════════════════════
# TP8: Production year threshold + rating correlation
# ═══════════════════════════════════════════════════════════════
("tp8a", "4J", "movies after 2010 + rating>7.0 + genre + company",
"""SELECT COUNT(*)
FROM title t,
     movie_info_idx mii,
     movie_info mi,
     movie_companies mc,
     info_type it_r,
     info_type it_g
WHERE mii.movie_id = t.id
  AND mi.movie_id = t.id
  AND mc.movie_id = t.id
  AND it_r.id = mii.info_type_id
  AND it_g.id = mi.info_type_id
  AND it_r.info = 'rating'
  AND mii.info > '7.0'
  AND it_g.info = 'genres'
  AND mi.info IN ('Drama', 'Thriller')
  AND t.production_year > 2010;"""),

("tp8b", "5J", "movies 1980-2000 + top 250 + genre + cast + keyword",
"""SELECT COUNT(*)
FROM title t,
     movie_info_idx mii,
     movie_info mi,
     cast_info ci,
     movie_keyword mk,
     info_type it_r,
     info_type it_g,
     role_type rt
WHERE mii.movie_id = t.id
  AND mi.movie_id = t.id
  AND ci.movie_id = t.id
  AND mk.movie_id = t.id
  AND it_r.id = mii.info_type_id
  AND it_g.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND it_r.info = 'top 250 rank'
  AND it_g.info = 'genres'
  AND mi.info = 'Drama'
  AND rt.role = 'actor'
  AND t.production_year BETWEEN 1980 AND 2000;"""),
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

print(f"\nRunning {len(QUERIES)} pure temporal q-error queries on {DB['dbname']}...")
print(f"  {'Q':<6} {'#J':<6} {'Description':<52} {'SubAgg-QE':>12} {'Est':>12} {'Act':>12} {'ms':>7}  Bias")
print("  " + "-" * 115)

results = []
for qid, nj, desc, sql in QUERIES:
    try:
        t0 = time.time()
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
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

        # Sub-aggregate node
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
        _, wnt, w_est, w_act, w_qe, w_bias = worst

        print(f"  {qid:<6} {nj:<6} {desc[:50]:<52} {sub_qe:>10.1f}x  {sub_est:>11.0f} {sub_act:>11.0f} {elapsed:>6.0f}  {sub_bias}", flush=True)
        results.append({
            "query": qid,
            "joins": nj,
            "desc": desc,
            "sub_agg_qe": round(sub_qe, 2),
            "sub_agg_est": sub_est,
            "sub_agg_act": sub_act,
            "sub_agg_bias": sub_bias,
            "worst_qe": round(w_qe, 2),
            "worst_est": w_est,
            "worst_act": w_act,
            "worst_node": wnt,
            "ms": round(elapsed, 1),
        })
    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        print(f"  {qid:<6} {nj:<6} {desc[:50]:<52}  ERROR ({elapsed:.0f}ms): {str(ex)[:60]}", flush=True)
        results.append({
            "query": qid, "joins": nj, "desc": desc,
            "sub_agg_qe": -1, "sub_agg_est": 0, "sub_agg_act": 0,
            "sub_agg_bias": "ERROR", "worst_qe": -1, "worst_est": 0, "worst_act": 0,
            "worst_node": "ERROR", "ms": round(elapsed, 1),
        })

# ─── Summary ──────────────────────────────────────────────
ok_results = [r for r in results if r["sub_agg_qe"] > 0]
sub_qe = [r["sub_agg_qe"] for r in ok_results]
worst_qe = [r["worst_qe"] for r in ok_results]
under = [r for r in ok_results if r["sub_agg_bias"] == "UNDER"]

print(f"\n{'='*90}")
print(f"PURE TEMPORAL BENCHMARK SUMMARY ({len(ok_results)} queries)")
print(f"{'='*90}")
print(f"  UNDER-estimates:             {len(under)}/{len(ok_results)}")
print(f"  Sub-agg q-error > 2x:        {sum(1 for q in sub_qe if q > 2)}/{len(sub_qe)}")
print(f"  Sub-agg q-error > 10x:       {sum(1 for q in sub_qe if q > 10)}/{len(sub_qe)}")
print(f"  Sub-agg q-error > 100x:      {sum(1 for q in sub_qe if q > 100)}/{len(sub_qe)}")
print(f"  Sub-agg q-error > 1000x:     {sum(1 for q in sub_qe if q > 1000)}/{len(sub_qe)}")
print(f"  Sub-agg median:              {sorted(sub_qe)[len(sub_qe)//2]:,.1f}x")
print(f"  Sub-agg max:                 {max(sub_qe):,.1f}x")
print(f"  Worst-node median:           {sorted(worst_qe)[len(worst_qe)//2]:,.1f}x")
print(f"  Worst-node max:              {max(worst_qe):,.1f}x")

out = Path(__file__).parent / "job_temporal_pure_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print(f"  CSV -> {out.name}")

cur.close()
conn.close()
