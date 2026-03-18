"""Triangle and self-join benchmark queries on the edges table."""
import sys, io, json, time, math, psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_v16', user='postgres', password='bench')
conn.autocommit = True
cur = conn.cursor()

QUERIES = [

("T1", "Friendship triangle (3 mutual friends)",
"""SELECT COUNT(*) FROM edges e1, edges e2, edges e3
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'friendship' AND e3.edge_type = 'friendship'
     AND e1.dst_id = e2.src_id
     AND e2.dst_id = e3.src_id
     AND e3.dst_id = e1.src_id
     AND e1.src_id < e1.dst_id"""),

("T2", "Friend triangles who co-star in same movie",
"""SELECT COUNT(*) FROM edges e1, edges e2, edges e3,
        cast_info ci1, cast_info ci2, cast_info ci3, title t
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'friendship' AND e3.edge_type = 'friendship'
     AND e1.dst_id = e2.src_id
     AND e2.dst_id = e3.src_id
     AND e3.dst_id = e1.src_id
     AND ci1.person_id = e1.src_id AND ci1.movie_id = t.id
     AND ci2.person_id = e2.src_id AND ci2.movie_id = t.id
     AND ci3.person_id = e3.src_id AND ci3.movie_id = t.id
     AND e1.src_id < e1.dst_id AND e2.src_id < e3.dst_id"""),

("T3", "Rivalry pair who both friend a third person",
"""SELECT COUNT(*) FROM edges e_rival, edges e_friend1, edges e_friend2
   WHERE e_rival.edge_type = 'rivalry'
     AND e_friend1.edge_type = 'friendship'
     AND e_friend2.edge_type = 'friendship'
     AND e_friend1.src_id = e_rival.src_id
     AND e_friend2.src_id = e_rival.dst_id
     AND e_friend1.dst_id = e_friend2.dst_id"""),

("T4", "Mentorship chain (A mentors B, B mentors C)",
"""SELECT COUNT(*) FROM edges e1, edges e2
   WHERE e1.edge_type = 'mentorship' AND e2.edge_type = 'mentorship'
     AND e1.dst_id = e2.src_id
     AND e1.src_id <> e2.dst_id"""),

("T5", "Friends who became rivals (temporal edge evolution)",
"""SELECT COUNT(*) FROM edges e1, edges e2
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'rivalry'
     AND e1.src_id = e2.src_id AND e1.dst_id = e2.dst_id"""),

("T6", "Friendship triangle + all 3 in Drama movies",
"""SELECT COUNT(*) FROM edges e1, edges e2, edges e3,
        cast_info ci1, cast_info ci2, cast_info ci3,
        title t1, title t2, title t3, 
        movie_info mi1, movie_info mi2, movie_info mi3
   WHERE e1.edge_type = 'friendship' AND e2.edge_type = 'friendship' AND e3.edge_type = 'friendship'
     AND e1.dst_id = e2.src_id AND e2.dst_id = e3.src_id AND e3.dst_id = e1.src_id
     AND ci1.person_id = e1.src_id AND ci2.person_id = e2.src_id AND ci3.person_id = e3.src_id
     AND ci1.movie_id = t1.id AND ci2.movie_id = t2.id AND ci3.movie_id = t3.id
     AND mi1.movie_id = t1.id AND mi1.info_type_id = 3 AND mi1.info = 'Drama'
     AND mi2.movie_id = t2.id AND mi2.info_type_id = 3 AND mi2.info = 'Drama'
     AND mi3.movie_id = t3.id AND mi3.info_type_id = 3 AND mi3.info = 'Drama'
     AND e1.src_id < e1.dst_id AND e2.src_id < e3.dst_id"""),
]

print(f"Running {len(QUERIES)} triangle/self-join queries on edges table...")
print(f"{'Query':<6} {'Description':<55} {'Q-Error':>10} {'Est':>10} {'Act':>10} {'Time':>8}")
print("-" * 105)

for qid, desc, sql in QUERIES:
    try:
        # Get EXPLAIN ANALYZE
        t0 = time.time()
        explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}"
        cur.execute(explain_sql)
        plan_json = cur.fetchone()[0]
        elapsed = (time.time() - t0) * 1000
        
        plan = plan_json[0]["Plan"]
        
        # Walk plan tree to find worst q-error
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
        
        # Find worst q-error node
        worst = max(all_nodes, key=lambda x: x[4])
        worst_depth, worst_type, worst_est, worst_act, worst_qe, worst_bias = worst
        
        # Get top-level result
        top_est = plan.get("Plan Rows", 1)
        top_act = max(float(plan.get("Actual Rows", 0)) * max(int(plan.get("Actual Loops", 1)), 1), 0.5)
        top_qe = max(float(top_est)/top_act, top_act/float(top_est))
        
        print(f"  {qid:<4} {desc[:53]:<55} {worst_qe:>8.1f}x  {worst_est:>9.0f} {worst_act:>9.0f}  {elapsed:>6.0f}ms")
        
        # Print all join nodes with high q-error
        high_qe = [n for n in all_nodes if n[4] > 5 and ("Join" in n[1] or "Loop" in n[1])]
        for d, nt, e, a, q, b in sorted(high_qe, key=lambda x: -x[4])[:3]:
            print(f"         d={d} {nt:<20} est={e:>10.0f}  act={a:>10.0f}  qe={q:>8.1f}x")
        
    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        print(f"  {qid:<4} {desc[:53]:<55}  ERROR: {str(ex)[:60]}  ({elapsed:.0f}ms)")

print()
cur.close()
conn.close()
