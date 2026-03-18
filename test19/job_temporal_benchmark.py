"""
Temporal Q-Error Benchmark: Queries combining IMDB schema with the edges table.

These queries exercise temporal predicates (valid_from, valid_to, production_year)
and edge-table joins that create complex estimation challenges for PostgreSQL:

1. **Temporal range overlap**: edges active during a movie's production year
2. **Temporal fan-out**: many edges per person × many movies per person
3. **Edge-type × temporal selectivity**: specific edge types in specific eras
4. **Multi-hop temporal**: person→edge→person→cast→movie→info chains
5. **Temporal aggregation skew**: counting relationships in narrow vs wide windows
"""
import sys, io, time, math, psycopg2, csv
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB = dict(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')
conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

QUERY_DIR = Path(__file__).parent / "job_temporal"
QUERY_DIR.mkdir(exist_ok=True)

QUERIES = [

# ═══════════════════════════════════════════════════════════════
# T1: Basic temporal range — edges active during a specific year
# ═══════════════════════════════════════════════════════════════
("t1a", "2J", "friendships active in 2015",
"""SELECT COUNT(*)
FROM edges e,
     name n
WHERE e.src_id = n.id
  AND e.src_type = 'person'
  AND e.edge_type = 'friendship'
  AND e.valid_from <= 2015
  AND e.valid_to >= 2015;"""),

("t1b", "2J", "collaborations active 2000-2010",
"""SELECT COUNT(*)
FROM edges e,
     name n
WHERE e.src_id = n.id
  AND e.src_type = 'person'
  AND e.edge_type = 'friendship'
  AND e.valid_from <= 2010
  AND e.valid_to >= 2000;"""),

("t1c", "2J", "mentorships started after 2005, still active 2020",
"""SELECT COUNT(*)
FROM edges e,
     name n
WHERE e.src_id = n.id
  AND e.src_type = 'person'
  AND e.edge_type = 'mentorship'
  AND e.valid_from >= 2005
  AND e.valid_to >= 2020;"""),

# ═══════════════════════════════════════════════════════════════
# T2: Edge → Person → Cast → Movie temporal chains
# ═══════════════════════════════════════════════════════════════
("t2a", "3J", "friendship edge + actor cast + movie year>2010",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci,
     title t
WHERE e.src_id = ci.person_id
  AND ci.movie_id = t.id
  AND e.src_type = 'person'
  AND e.edge_type = 'friendship'
  AND e.sign = '+'
  AND t.production_year > 2010;"""),

("t2b", "4J", "friendship active during movie year + genre(Drama)",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci,
     title t,
     movie_info mi,
     info_type it
WHERE e.src_id = ci.person_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND it.id = mi.info_type_id
  AND e.src_type = 'person'
  AND e.edge_type = 'friendship'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND it.info = 'genres'
  AND mi.info = 'Drama';"""),

("t2c", "4J", "mentorship edge active during movie + keyword",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci,
     title t,
     movie_keyword mk
WHERE e.src_id = ci.person_id
  AND ci.movie_id = t.id
  AND mk.movie_id = t.id
  AND e.src_type = 'person'
  AND e.edge_type = 'mentorship'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND t.production_year > 2005;"""),

# ═══════════════════════════════════════════════════════════════
# T3: Two-person temporal — both endpoints of edge join to cast
# ═══════════════════════════════════════════════════════════════
("t3a", "4J", "both friends in same movie",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci1,
     cast_info ci2,
     title t
WHERE e.src_id = ci1.person_id
  AND e.dst_id = ci2.person_id
  AND ci1.movie_id = t.id
  AND ci2.movie_id = ci1.movie_id
  AND e.src_type = 'person'
  AND e.dst_type = 'person'
  AND e.edge_type = 'friendship'
  AND t.production_year > 2010;"""),

("t3b", "5J", "both friends in same movie during edge validity + genre",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci1,
     cast_info ci2,
     title t,
     movie_info mi,
     info_type it
WHERE e.src_id = ci1.person_id
  AND e.dst_id = ci2.person_id
  AND ci1.movie_id = t.id
  AND ci2.movie_id = ci1.movie_id
  AND mi.movie_id = t.id
  AND it.id = mi.info_type_id
  AND e.src_type = 'person'
  AND e.dst_type = 'person'
  AND e.edge_type = 'friendship'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND it.info = 'genres'
  AND mi.info IN ('Action', 'Thriller');"""),

("t3c", "5J", "rivals in same movie + company + temporal overlap",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci1,
     cast_info ci2,
     title t,
     movie_companies mc
WHERE e.src_id = ci1.person_id
  AND e.dst_id = ci2.person_id
  AND ci1.movie_id = t.id
  AND ci2.movie_id = ci1.movie_id
  AND mc.movie_id = t.id
  AND e.src_type = 'person'
  AND e.dst_type = 'person'
  AND e.edge_type = 'rivalry'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year;"""),

# ═══════════════════════════════════════════════════════════════
# T4: Temporal window counting — narrow vs wide
# ═══════════════════════════════════════════════════════════════
("t4a", "3J", "edges starting in narrow window 2018-2020 + actor cast",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci,
     title t,
     role_type rt
WHERE e.src_id = ci.person_id
  AND ci.movie_id = t.id
  AND rt.id = ci.role_id
  AND e.src_type = 'person'
  AND e.edge_type = 'friendship'
  AND e.valid_from BETWEEN 2018 AND 2020
  AND rt.role = 'actor';"""),

("t4b", "3J", "brand_fit edges after 2010 + company × person",
"""SELECT COUNT(*)
FROM edges e,
     name n,
     company_name cn
WHERE e.src_id = cn.id
  AND e.dst_id = n.id
  AND e.src_type = 'company'
  AND e.dst_type = 'person'
  AND e.edge_type = 'brand_fit'
  AND e.valid_from > 2010
  AND e.weight > 0.7;"""),

# ═══════════════════════════════════════════════════════════════
# T5: Edge × movie_info EAV temporal chains
# ═══════════════════════════════════════════════════════════════
("t5a", "5J", "friendship + actor + genre(Action) + country(USA) + temporal",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci,
     title t,
     movie_info mi1,
     movie_info mi2,
     info_type it1,
     info_type it2,
     role_type rt
WHERE e.src_id = ci.person_id
  AND ci.movie_id = t.id
  AND mi1.movie_id = t.id
  AND mi2.movie_id = t.id
  AND it1.id = mi1.info_type_id
  AND it2.id = mi2.info_type_id
  AND rt.id = ci.role_id
  AND e.src_type = 'person'
  AND e.edge_type = 'friendship'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND rt.role = 'actor'
  AND it1.info = 'genres'
  AND mi1.info = 'Action'
  AND it2.info = 'countries'
  AND mi2.info = 'USA';"""),

("t5b", "6J", "mentorship + cast + genre + company + keyword + temporal",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci,
     title t,
     movie_info mi,
     movie_companies mc,
     movie_keyword mk,
     info_type it
WHERE e.src_id = ci.person_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND mc.movie_id = t.id
  AND mk.movie_id = t.id
  AND it.id = mi.info_type_id
  AND e.src_type = 'person'
  AND e.edge_type = 'mentorship'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND it.info = 'genres'
  AND mi.info IN ('Drama', 'Crime')
  AND t.production_year > 2005;"""),

# ═══════════════════════════════════════════════════════════════
# T6: Multi-edge-type temporal — combining edge types with sign
# ═══════════════════════════════════════════════════════════════
("t6a", "3J", "negative edges (blacklist/rivalry) + actor cast + recent",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci,
     title t
WHERE e.src_id = ci.person_id
  AND ci.movie_id = t.id
  AND e.src_type = 'person'
  AND e.sign = '-'
  AND e.edge_type IN ('blacklist', 'rivalry', 'avoid')
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND t.production_year > 2010;"""),

("t6b", "4J", "chemistry edge + both actors in same movie + temporal",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci1,
     cast_info ci2,
     title t,
     role_type rt1,
     role_type rt2
WHERE e.src_id = ci1.person_id
  AND e.dst_id = ci2.person_id
  AND ci1.movie_id = t.id
  AND ci2.movie_id = ci1.movie_id
  AND rt1.id = ci1.role_id
  AND rt2.id = ci2.role_id
  AND e.src_type = 'person'
  AND e.dst_type = 'person'
  AND e.edge_type = 'chemistry'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND rt1.role IN ('actor', 'actress')
  AND rt2.role IN ('actor', 'actress');"""),

# ═══════════════════════════════════════════════════════════════
# T7: Company-person edge temporal chains
# ═══════════════════════════════════════════════════════════════
("t7a", "4J", "co_production edge + movie_companies + genre + temporal",
"""SELECT COUNT(*)
FROM edges e,
     movie_companies mc,
     title t,
     movie_info mi,
     info_type it
WHERE e.src_id = mc.company_id
  AND mc.movie_id = t.id
  AND mi.movie_id = t.id
  AND it.id = mi.info_type_id
  AND e.src_type = 'company'
  AND e.dst_type = 'company'
  AND e.edge_type = 'co_production'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND it.info = 'genres'
  AND mi.info IN ('Action', 'Sci-Fi')
  AND t.production_year > 2010;"""),

("t7b", "5J", "employment edge + person cast + company + movie + temporal",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci,
     movie_companies mc,
     title t,
     company_name cn
WHERE e.src_id = cn.id
  AND e.dst_id = ci.person_id
  AND e.src_type = 'company'
  AND e.dst_type = 'person'
  AND ci.movie_id = t.id
  AND mc.movie_id = t.id
  AND mc.company_id = cn.id
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND t.production_year > 2010;"""),

# ═══════════════════════════════════════════════════════════════
# T8: Complex multi-table temporal joins
# ═══════════════════════════════════════════════════════════════
("t8a", "6J", "friendship + actor + person_info(nat) + genre + temporal",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci,
     person_info pi,
     title t,
     movie_info mi,
     info_type it1,
     info_type it2,
     role_type rt
WHERE e.src_id = ci.person_id
  AND pi.person_id = ci.person_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND it1.id = pi.info_type_id
  AND it2.id = mi.info_type_id
  AND rt.id = ci.role_id
  AND e.src_type = 'person'
  AND e.edge_type = 'friendship'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND it1.info = 'nationality'
  AND pi.info = 'American'
  AND rt.role = 'actor'
  AND it2.info = 'genres'
  AND mi.info = 'Drama'
  AND t.production_year > 2010;"""),

("t8b", "7J", "friendship + both persons + cast + movie + genre + keyword + temporal",
"""SELECT COUNT(*)
FROM edges e,
     cast_info ci1,
     cast_info ci2,
     title t,
     movie_info mi,
     movie_keyword mk,
     info_type it,
     role_type rt
WHERE e.src_id = ci1.person_id
  AND e.dst_id = ci2.person_id
  AND ci1.movie_id = t.id
  AND ci2.movie_id = ci1.movie_id
  AND mi.movie_id = t.id
  AND mk.movie_id = t.id
  AND it.id = mi.info_type_id
  AND rt.id = ci1.role_id
  AND e.src_type = 'person'
  AND e.dst_type = 'person'
  AND e.edge_type = 'friendship'
  AND e.valid_from <= t.production_year
  AND e.valid_to >= t.production_year
  AND rt.role = 'actor'
  AND it.info = 'genres'
  AND mi.info IN ('Action', 'Drama')
  AND t.production_year > 2005;"""),
]

# ─── Write individual SQL files ──────────────────────────────
for stem, _, desc, sql in QUERIES:
    path = QUERY_DIR / f"{stem}.sql"
    path.write_text(f"-- {desc}\n{sql.strip()}\n", encoding="utf-8")
print(f"Wrote {len(QUERIES)} SQL files to {QUERY_DIR}/")

# ─── Run benchmark ──────────────────────────────────────────
cur.execute("ANALYZE")
cur.execute("ANALYZE edges")

def q_error(est, act):
    e = max(est, 0.5)
    a = max(act, 0.5)
    return max(e / a, a / e)

print(f"\nRunning {len(QUERIES)} temporal q-error queries on {DB['dbname']}...")
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
        skip = {"Aggregate", "Result", "Sort", "Limit", "Unique"}
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
        _, _, w_est, w_act, w_qe, w_bias = worst

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
            "ms": round(elapsed, 1),
        })
    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        print(f"  {qid:<6} {nj:<6} {desc[:50]:<52}  ERROR ({elapsed:.0f}ms): {str(ex)[:60]}", flush=True)
        results.append({
            "query": qid, "joins": nj, "desc": desc,
            "sub_agg_qe": -1, "sub_agg_est": 0, "sub_agg_act": 0,
            "sub_agg_bias": "ERROR", "worst_qe": -1, "worst_est": 0, "worst_act": 0,
            "ms": round(elapsed, 1),
        })

# ─── Summary ──────────────────────────────────────────────
ok_results = [r for r in results if r["sub_agg_qe"] > 0]
sub_qe = [r["sub_agg_qe"] for r in ok_results]
worst_qe = [r["worst_qe"] for r in ok_results]
under = [r for r in ok_results if r["sub_agg_bias"] == "UNDER"]

print(f"\n{'='*90}")
print(f"TEMPORAL BENCHMARK SUMMARY ({len(ok_results)} queries)")
print(f"{'='*90}")
print(f"  UNDER-estimates: {len(under)}/{len(ok_results)}")
print(f"  Sub-agg q-error > 10x:   {sum(1 for q in sub_qe if q > 10)}/{len(sub_qe)}")
print(f"  Sub-agg q-error > 100x:  {sum(1 for q in sub_qe if q > 100)}/{len(sub_qe)}")
print(f"  Sub-agg q-error > 1000x: {sum(1 for q in sub_qe if q > 1000)}/{len(sub_qe)}")
print(f"  Sub-agg median:          {sorted(sub_qe)[len(sub_qe)//2]:,.1f}x")
print(f"  Sub-agg max:             {max(sub_qe):,.1f}x")
print(f"  Worst-node median:       {sorted(worst_qe)[len(worst_qe)//2]:,.1f}x")
print(f"  Worst-node max:          {max(worst_qe):,.1f}x")

out = Path(__file__).parent / "job_temporal_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print(f"  CSV -> {out.name}")

cur.close()
conn.close()
