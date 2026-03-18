"""
Explosive edge-relational queries: exploit multiplicative q-error across join layers.
Pattern: edge_pattern × per-vertex relational fan-out = massive underestimate.

T6 formula: triangle(3M×) × cast_fanout³ × genre_filter³ = 814M×
New queries try different combinations of edge patterns × relational paths.
"""
import sys, io, time, math, psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_v17', user='postgres', password='bench')
conn.autocommit = True
cur = conn.cursor()
cur.execute("SET statement_timeout = '600000'")  # 10 min max

QUERIES = [

# --- PATTERN A: Triangle + different relational joins per vertex ---

("X1", "Triangle + person_info(nationality) for all 3 vertices",
"""SELECT COUNT(*) FROM edges e1, edges e2, edges e3,
        cast_info ci1, cast_info ci2, cast_info ci3,
        person_info pi1, person_info pi2, person_info pi3
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'friendship' AND e3.edge_type = 'friendship'
     AND e1.dst_id = e2.src_id AND e2.dst_id = e3.src_id AND e3.dst_id = e1.src_id
     AND ci1.person_id = e1.src_id AND ci2.person_id = e2.src_id AND ci3.person_id = e3.src_id
     AND pi1.person_id = e1.src_id AND pi2.person_id = e2.src_id AND pi3.person_id = e3.src_id
     AND pi1.info_type_id = 18 AND pi2.info_type_id = 18 AND pi3.info_type_id = 18
     AND e1.src_id < e1.dst_id"""),

("X2", "Triangle + all 3 linked to companies via cast",
"""SELECT COUNT(*) FROM edges e1, edges e2, edges e3,
        cast_info ci1, cast_info ci2, cast_info ci3,
        movie_companies mc1, movie_companies mc2, movie_companies mc3
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'friendship' AND e3.edge_type = 'friendship'
     AND e1.dst_id = e2.src_id AND e2.dst_id = e3.src_id AND e3.dst_id = e1.src_id
     AND ci1.person_id = e1.src_id AND ci2.person_id = e2.src_id AND ci3.person_id = e3.src_id
     AND mc1.movie_id = ci1.movie_id AND mc2.movie_id = ci2.movie_id AND mc3.movie_id = ci3.movie_id
     AND e1.src_id < e1.dst_id AND e2.src_id < e3.dst_id"""),

("X3", "Triangle + all 3 have keywords on their movies",
"""SELECT COUNT(*) FROM edges e1, edges e2, edges e3,
        cast_info ci1, cast_info ci2, cast_info ci3,
        movie_keyword mk1, movie_keyword mk2, movie_keyword mk3
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'friendship' AND e3.edge_type = 'friendship'
     AND e1.dst_id = e2.src_id AND e2.dst_id = e3.src_id AND e3.dst_id = e1.src_id
     AND ci1.person_id = e1.src_id AND ci2.person_id = e2.src_id AND ci3.person_id = e3.src_id
     AND mk1.movie_id = ci1.movie_id AND mk2.movie_id = ci2.movie_id AND mk3.movie_id = ci3.movie_id
     AND e1.src_id < e1.dst_id AND e2.src_id < e3.dst_id"""),

# --- PATTERN B: 2-way edge + heavier per-endpoint relational paths ---

("X4", "Friendship pair + both in Drama + both have keywords + both have companies",
"""SELECT COUNT(*) FROM edges e,
        cast_info ci1, cast_info ci2,
        movie_info mi1, movie_info mi2,
        movie_keyword mk1, movie_keyword mk2,
        movie_companies mc1, movie_companies mc2
   WHERE e.edge_type = 'friendship'
     AND ci1.person_id = e.src_id AND ci2.person_id = e.dst_id
     AND mi1.movie_id = ci1.movie_id AND mi2.movie_id = ci2.movie_id
     AND mk1.movie_id = ci1.movie_id AND mk2.movie_id = ci2.movie_id
     AND mc1.movie_id = ci1.movie_id AND mc2.movie_id = ci2.movie_id
     AND mi1.info_type_id = 3 AND mi1.info = 'Drama'
     AND mi2.info_type_id = 3 AND mi2.info = 'Drama'"""),

("X5", "Rivalry pair + both in Action + same US company",
"""SELECT COUNT(*) FROM edges e,
        cast_info ci1, cast_info ci2,
        movie_info mi1, movie_info mi2,
        movie_companies mc1, movie_companies mc2,
        company_name cn1, company_name cn2
   WHERE e.edge_type = 'rivalry'
     AND ci1.person_id = e.src_id AND ci2.person_id = e.dst_id
     AND mi1.movie_id = ci1.movie_id AND mi2.movie_id = ci2.movie_id
     AND mc1.movie_id = ci1.movie_id AND mc2.movie_id = ci2.movie_id
     AND mc1.company_id = cn1.id AND mc2.company_id = cn2.id
     AND mi1.info_type_id = 3 AND mi1.info = 'Action'
     AND mi2.info_type_id = 3 AND mi2.info = 'Action'
     AND cn1.country_code = 'us' AND cn2.country_code = 'us'"""),

# --- PATTERN C: Mentorship chain + relational fan-out at every step ---

("X6", "Mentorship chain A→B→C, all 3 in Drama movies",
"""SELECT COUNT(*) FROM edges e1, edges e2,
        cast_info ci1, cast_info ci2, cast_info ci3,
        movie_info mi1, movie_info mi2, movie_info mi3
   WHERE e1.edge_type = 'mentorship' AND e2.edge_type = 'mentorship'
     AND e1.dst_id = e2.src_id AND e1.src_id <> e2.dst_id
     AND ci1.person_id = e1.src_id AND ci2.person_id = e1.dst_id AND ci3.person_id = e2.dst_id
     AND mi1.movie_id = ci1.movie_id AND mi2.movie_id = ci2.movie_id AND mi3.movie_id = ci3.movie_id
     AND mi1.info_type_id = 3 AND mi1.info = 'Drama'
     AND mi2.info_type_id = 3 AND mi2.info = 'Drama'
     AND mi3.info_type_id = 3 AND mi3.info = 'Drama'"""),

("X7", "Mentorship chain + person_info(nationality) at each step",
"""SELECT COUNT(*) FROM edges e1, edges e2,
        person_info pi1, person_info pi2, person_info pi3
   WHERE e1.edge_type = 'mentorship' AND e2.edge_type = 'mentorship'
     AND e1.dst_id = e2.src_id AND e1.src_id <> e2.dst_id
     AND pi1.person_id = e1.src_id AND pi2.person_id = e1.dst_id AND pi3.person_id = e2.dst_id
     AND pi1.info_type_id = 18 AND pi2.info_type_id = 18 AND pi3.info_type_id = 18"""),

# --- PATTERN D: Mixed edge types forming a path ---

("X8", "A friends B, B mentors C (mixed edge path) + person_info × 3",
"""SELECT COUNT(*) FROM edges e1, edges e2,
        person_info pi1, person_info pi2, person_info pi3
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'mentorship'
     AND e1.dst_id = e2.src_id
     AND pi1.person_id = e1.src_id AND pi2.person_id = e1.dst_id AND pi3.person_id = e2.dst_id
     AND pi1.info_type_id = 19 AND pi2.info_type_id = 19 AND pi3.info_type_id = 19"""),

("X9", "A friends B, B rivals C, all 3 cast in Action movies",
"""SELECT COUNT(*) FROM edges e1, edges e2,
        cast_info ci1, cast_info ci2, cast_info ci3,
        movie_info mi1, movie_info mi2, movie_info mi3
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'rivalry'
     AND e1.dst_id = e2.src_id
     AND ci1.person_id = e1.src_id AND ci2.person_id = e1.dst_id AND ci3.person_id = e2.dst_id
     AND mi1.movie_id = ci1.movie_id AND mi2.movie_id = ci2.movie_id AND mi3.movie_id = ci3.movie_id
     AND mi1.info_type_id = 3 AND mi1.info = 'Action'
     AND mi2.info_type_id = 3 AND mi2.info = 'Action'
     AND mi3.info_type_id = 3 AND mi3.info = 'Action'"""),

# --- PATTERN E: brand_fit edge + relational amplification ---

("X10", "brand_fit + person has bio + company has movies with keywords",
"""SELECT COUNT(*) FROM edges e, person_info pi, cast_info ci,
        movie_companies mc, movie_keyword mk
   WHERE e.edge_type = 'brand_fit' AND e.sign = '+'
     AND pi.person_id = e.src_id AND pi.info_type_id = 19
     AND ci.person_id = e.src_id AND mc.company_id = e.dst_id
     AND mk.movie_id = ci.movie_id
     AND ci.movie_id = mc.movie_id"""),

("X11", "brand_fit + actor in Drama + company in same movie + keyword",
"""SELECT COUNT(*) FROM edges e, cast_info ci, movie_info mi,
        movie_companies mc, movie_keyword mk
   WHERE e.edge_type = 'brand_fit' AND e.sign = '+'
     AND ci.person_id = e.src_id AND mc.company_id = e.dst_id
     AND ci.movie_id = mc.movie_id
     AND mi.movie_id = ci.movie_id AND mk.movie_id = ci.movie_id
     AND mi.info_type_id = 3 AND mi.info = 'Drama'
     AND ci.role_id = 1"""),
]

print(f"Running {len(QUERIES)} explosive queries on imdb_v17...")
print(f"  {'Q':<5} {'Description':<58} {'Q-Error':>14} {'Est':>12} {'Act':>12} {'Time':>8}")
print("  " + "-" * 112)

for qid, desc, sql in QUERIES:
    try:
        t0 = time.time()
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
        plan_json = cur.fetchone()[0]
        elapsed = (time.time() - t0)
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
        worst = max(all_nodes, key=lambda x: x[4])
        _, _, w_est, w_act, w_qe, w_bias = worst
        bias_dir = "UNDER" if w_bias < 0 else "OVER"
        time_str = f"{elapsed:.1f}s" if elapsed >= 1 else f"{elapsed*1000:.0f}ms"

        print(f"  {qid:<5} {desc[:56]:<58} {w_qe:>12,.1f}x  {w_est:>11,.0f} {w_act:>11,.0f}  {time_str:>7}  {bias_dir}", flush=True)
    except Exception as ex:
        elapsed = (time.time() - t0)
        print(f"  {qid:<5} {desc[:56]:<58}  ERROR ({elapsed:.0f}s): {str(ex)[:50]}", flush=True)

print("\nDone.")
cur.close(); conn.close()
