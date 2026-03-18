"""Fast 2-way edge queries only (no triangle joins)."""
import sys, io, json, time, math, psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_v17', user='postgres', password='bench')
conn.autocommit = True
cur = conn.cursor()

QUERIES = [

("T4", "Mentorship chain (A mentors B, B mentors C)",
"""SELECT COUNT(*) FROM edges e1, edges e2
   WHERE e1.edge_type = 'mentorship' AND e2.edge_type = 'mentorship'
     AND e1.dst_id = e2.src_id
     AND e1.src_id <> e2.dst_id"""),

("T5", "Friends who became rivals (temporal evolution)",
"""SELECT COUNT(*) FROM edges e1, edges e2
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'rivalry'
     AND e1.src_id = e2.src_id AND e1.dst_id = e2.dst_id"""),

("G1", "brand_fit edges with cast in same movie",
"""SELECT COUNT(*) FROM edges e, cast_info ci, movie_companies mc
   WHERE e.edge_type = 'brand_fit' AND e.sign = '+'
     AND ci.person_id = e.src_id
     AND mc.company_id = e.dst_id
     AND ci.movie_id = mc.movie_id"""),

("G2", "Friendship edge + both actors in Drama",
"""SELECT COUNT(*) FROM edges e, cast_info ci1, cast_info ci2,
        movie_info mi1, movie_info mi2
   WHERE e.edge_type = 'friendship'
     AND ci1.person_id = e.src_id AND ci2.person_id = e.dst_id
     AND mi1.movie_id = ci1.movie_id AND mi1.info = 'Drama'
     AND mi1.info_type_id = 3
     AND mi2.movie_id = ci2.movie_id AND mi2.info = 'Drama'
     AND mi2.info_type_id = 3"""),

("G3", "Rivalry + both in same company",
"""SELECT COUNT(*) FROM edges e, cast_info ci1, cast_info ci2,
        movie_companies mc1, movie_companies mc2
   WHERE e.edge_type = 'rivalry'
     AND ci1.person_id = e.src_id AND ci2.person_id = e.dst_id
     AND mc1.movie_id = ci1.movie_id AND mc2.movie_id = ci2.movie_id
     AND mc1.company_id = mc2.company_id"""),

("G4", "blacklist edges where person still cast",
"""SELECT COUNT(*) FROM edges e, cast_info ci
   WHERE e.edge_type = 'blacklist'
     AND ci.person_id = e.src_id"""),

("G5", "co_production company pairs sharing a movie",
"""SELECT COUNT(*) FROM edges e, movie_companies mc1, movie_companies mc2
   WHERE e.edge_type = 'co_production'
     AND mc1.company_id = e.src_id AND mc2.company_id = e.dst_id
     AND mc1.movie_id = mc2.movie_id"""),

]

print(f"Running {len(QUERIES)} edge queries on imdb_v17...")
print(f"  {'Query':<6} {'Description':<55} {'Q-Error':>10} {'Est':>10} {'Act':>10} {'Time':>8}")
print("  " + "-" * 105)

for qid, desc, sql in QUERIES:
    try:
        t0 = time.time()
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
        plan_json = cur.fetchone()[0]
        elapsed = (time.time() - t0) * 1000
        plan = plan_json[0]["Plan"]

        def walk(node, depth=0):
            results = []
            nt = node.get("Node Type", "")
            est = float(node.get("Plan Rows", 1))
            loops = max(int(node.get("Actual Loops", 1)), 1)
            act = float(node.get("Actual Rows", 0)) * loops
            act = max(act, 0.5)
            if est <= 0: est = 0.5
            qe = max(est/act, act/est)
            bias = math.log2(est/act)
            results.append((depth, nt, est, act, qe, bias))
            for sub in node.get("Plans", []):
                results.extend(walk(sub, depth+1))
            return results

        all_nodes = walk(plan)
        worst = max(all_nodes, key=lambda x: x[4])
        _, _, worst_est, worst_act, worst_qe, worst_bias = worst
        bias_dir = "UNDER" if worst_bias < 0 else "OVER"

        print(f"  {qid:<6} {desc[:53]:<55} {worst_qe:>8.1f}x  {worst_est:>9.0f} {worst_act:>9.0f}  {elapsed:>6.0f}ms  {bias_dir}")

        high_qe = [n for n in all_nodes if n[4] > 5 and ("Join" in n[1] or "Loop" in n[1])]
        for d, nt, e, a, q, b in sorted(high_qe, key=lambda x: -x[4])[:2]:
            print(f"           d={d} {nt:<20} est={e:>10.0f}  act={a:>10.0f}  qe={q:>8.1f}x")
    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        print(f"  {qid:<6} {desc[:53]:<55}  ERROR: {str(ex)[:60]}  ({elapsed:.0f}ms)")

print()
cur.close()
conn.close()
