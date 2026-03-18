"""Check which benchmark queries have REAL q-errors (actual > 0 at worst node)."""
import sys, io, json, time, math, psycopg2

# encoding handled by PYTHONIOENCODING

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_v16', user='postgres', password='bench')
conn.autocommit = True
cur = conn.cursor()

# Import the queries from our benchmark
sys.path.insert(0, 'test16')
from job_benchmark_v2 import JOB_QUERIES

JOIN_NODE_TYPES = {"Hash Join", "Merge Join", "Nested Loop", "Hash Anti Join", "Merge Anti Join", "Hash Semi Join", "Merge Semi Join"}

def collect_real_nodes(node, depth=0):
    results = []
    nt = node.get("Node Type", "")
    est = float(node.get("Plan Rows", 1))
    loops = max(int(node.get("Actual Loops", 1)), 1)
    raw_act = float(node.get("Actual Rows", 0)) * loops
    if nt in JOIN_NODE_TYPES or "Join" in nt:
        results.append((depth, nt, est, raw_act))
    for sub in node.get("Plans", []):
        results.extend(collect_real_nodes(sub, depth+1))
    return results

print(f"\n{'='*100}")
print("HONEST Q-ERROR AUDIT: only counting join nodes where actual > 0")
print(f"{'='*100}\n")
print(f"  {'Query':<8s}  {'Worst Real QE':>13s}  {'Est':>8s}  {'Actual':>10s}  {'Joins':>5s}  {'Verdict':<10s}  Description")
print("-" * 100)

results = []
for qid, n_joins, desc, sql in JOB_QUERIES:
    try:
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
        plan_json = cur.fetchone()[0]
        plan = plan_json[0]["Plan"]
        nodes = collect_real_nodes(plan)
        
        # Filter: only nodes where actual > 0
        real_nodes = [(d, nt, e, a) for d, nt, e, a in nodes if a > 0]
        
        if not real_nodes:
            print(f"  {qid:<8s}  {'ALL ZERO':>13s}  {'':>8s}  {'':>10s}  {n_joins:>5d}  {'TRASH':<10s}  {desc[:45]}")
            continue
        
        # Find worst real q-error
        worst_real = max(real_nodes, key=lambda x: max(x[2]/max(x[3],0.5), x[3]/max(x[2],0.5)))
        d, nt, e, a = worst_real
        qe = max(e/max(a,0.5), a/max(e,0.5))
        
        verdict = "REAL" if qe >= 5 else "MILD"
        if qe >= 50: verdict = "STRONG"
        if qe >= 100: verdict = "MASSIVE"
        
        print(f"  {qid:<8s}  {qe:>11.1f}x  {e:>8.0f}  {a:>10.0f}  {n_joins:>5d}  {verdict:<10s}  {desc[:45]}")
        results.append((qid, qe, e, a, n_joins, desc))
        
    except Exception as ex:
        print(f"  {qid:<8s}  ERROR: {str(ex)[:60]}")

print(f"\n{'='*100}")
print("SUMMARY: Queries with REAL q-errors > 5x (actual > 0, no inflation)")
print(f"{'='*100}")
real_strong = [(q, qe, e, a, j, d) for q, qe, e, a, j, d in results if qe >= 5]
real_strong.sort(key=lambda x: -x[1])
for q, qe, e, a, j, d in real_strong:
    bias = "UNDER" if e < a else "OVER"
    print(f"  {q:<8s}  {qe:>8.1f}x  est={e:>8.0f}  act={a:>10.0f}  {bias}estimate  {d[:50]}")

print(f"\nTotal queries with real q-error > 5x: {len(real_strong)}")
print(f"Total queries with real q-error > 10x: {len([x for x in real_strong if x[1] >= 10])}")
print(f"Total queries with real q-error > 50x: {len([x for x in real_strong if x[1] >= 50])}")
print(f"Max REAL q-error: {max(x[1] for x in real_strong):.1f}x" if real_strong else "No real errors found")

cur.close()
conn.close()
