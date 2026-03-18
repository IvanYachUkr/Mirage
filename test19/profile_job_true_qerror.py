"""Correct JOB q-error measurement using EXPLAIN ANALYZE on COUNT(*).

For each query:
1. Strip SELECT min(...) -> SELECT COUNT(*)
2. Run EXPLAIN (ANALYZE, FORMAT JSON) on the COUNT(*) query
3. Walk the plan tree to find the node just below the final Aggregate
4. Extract BOTH Plan Rows (estimate) AND Actual Rows*Loops (actual) from that SAME node
5. q-error = max(est/act, act/est)

This ensures estimate and actual come from the exact same plan execution.
"""
import sys, io, time, re, csv, math, json
from pathlib import Path
import psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB = dict(host="localhost", port=5433, dbname="imdb_benchmark", user="postgres", password="bench")
QUERY_DIR = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\job_adapted")

conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

# Fresh stats
cur.execute("ANALYZE")

def _sort_key(p):
    m = re.match(r"(\d+)([a-z]?)", p.stem)
    return (int(m.group(1)), m.group(2)) if m else (999, p.stem)

sql_files = sorted(QUERY_DIR.glob("*.sql"), key=_sort_key)


def strip_to_count(sql: str) -> str:
    """Replace SELECT min(...) ... FROM with SELECT COUNT(*) FROM ..."""
    m = re.search(r'\bFROM\b', sql, re.IGNORECASE)
    if m:
        return "SELECT COUNT(*) " + sql[m.start():]
    return f"SELECT COUNT(*) FROM ({sql}) AS _q"


def find_below_aggregate(node):
    """Walk past Aggregate/Result/Sort/Limit to find the top join or scan node."""
    skip = {"Aggregate", "Result", "Sort", "Limit", "Unique"}
    while node.get("Node Type", "") in skip:
        children = node.get("Plans", [])
        if not children:
            break
        node = children[0]
    return node


def walk_all_nodes(node, depth=0):
    """Recursively collect all nodes with their estimated and actual rows."""
    nt = node.get("Node Type", "")
    est = float(node.get("Plan Rows", 1))
    loops = max(int(node.get("Actual Loops", 1)), 1)
    act_per_loop = float(node.get("Actual Rows", 0))
    act_total = act_per_loop * loops

    rows = [{
        "depth": depth,
        "node_type": nt,
        "est": est,
        "act": act_total,
        "loops": loops,
    }]
    for child in node.get("Plans", []):
        rows.extend(walk_all_nodes(child, depth + 1))
    return rows


def q_error(est, act):
    """Compute q-error with floor of 0.5 to avoid div-by-zero."""
    e = max(est, 0.5)
    a = max(act, 0.5)
    return max(e / a, a / e)


print(f"JOB Q-Error (Corrected): {len(sql_files)} queries")
print(f"Method: EXPLAIN ANALYZE on COUNT(*), extract est/act from same plan node below Aggregate")
print(f"Database: {DB['dbname']} @ {DB['host']}:{DB['port']}")
print()
print(f"  {'Q':<8} {'Actual':>10} {'Est':>10} {'QE(sub-agg)':>12} {'Node':>22} {'Bias':>6} {'#nodes':>6} {'worst_qe':>10} {'ms':>7}")
print(f"  {'-'*100}")

results = []
all_nodes = []

for sf in sql_files:
    qid = sf.stem
    original_sql = sf.read_text(encoding="utf-8").strip().rstrip(";")
    count_sql = strip_to_count(original_sql)

    t0 = time.time()
    try:
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {count_sql}")
        plan_json = cur.fetchone()[0]
        elapsed = (time.time() - t0) * 1000

        root = plan_json[0]["Plan"]
        planning_ms = plan_json[0].get("Planning Time", 0)
        execution_ms = plan_json[0].get("Execution Time", 0)

        # Find the node below the Aggregate
        sub_agg = find_below_aggregate(root)
        sub_est = float(sub_agg.get("Plan Rows", 1))
        sub_loops = max(int(sub_agg.get("Actual Loops", 1)), 1)
        sub_act = float(sub_agg.get("Actual Rows", 0)) * sub_loops
        sub_qe = q_error(sub_est, sub_act)
        sub_node = sub_agg.get("Node Type", "")

        # Also get the actual COUNT(*) result from root
        root_act = float(root.get("Actual Rows", 0)) * max(int(root.get("Actual Loops", 1)), 1)

        # Determine bias
        bias = "OVER" if sub_est > sub_act else ("UNDER" if sub_est < sub_act else "EXACT")
        if sub_act == 0 and sub_est == 0:
            bias = "EXACT"
        elif sub_act == 0 and sub_est > 0:
            bias = "OVER"

        # Walk all nodes for per-node stats
        nodes = walk_all_nodes(root)
        for n in nodes:
            n["query"] = qid
            n["qe"] = q_error(n["est"], n["act"])
            all_nodes.append(n)

        # Find worst q-error across all non-aggregate nodes
        non_agg = [n for n in nodes if n["node_type"] not in ("Aggregate", "Result")]
        worst_qe = max((n["qe"] for n in non_agg), default=1.0)

        print(
            f"  {qid:<8} {sub_act:>10,.0f} {sub_est:>10,.0f} {sub_qe:>11.1f}x "
            f"{sub_node:>22} {bias:>6} {len(nodes):>6} {worst_qe:>9.1f}x {elapsed:>6.0f}",
            flush=True,
        )

        results.append({
            "query": qid,
            "sub_agg_actual": sub_act,
            "sub_agg_estimate": sub_est,
            "sub_agg_qerror": round(sub_qe, 2),
            "sub_agg_node": sub_node,
            "sub_agg_bias": bias,
            "worst_node_qerror": round(worst_qe, 2),
            "n_nodes": len(nodes),
            "planning_ms": round(planning_ms, 2),
            "execution_ms": round(execution_ms, 2),
            "total_ms": round(elapsed, 1),
        })

    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        err_msg = str(ex).split("\n")[0][:60]
        print(f"  {qid:<8}  ERROR: {err_msg} ({elapsed:.0f}ms)", flush=True)
        conn.rollback()

# ── Summary ──
sub_qe_vals = [r["sub_agg_qerror"] for r in results]
worst_qe_vals = [r["worst_node_qerror"] for r in results]
actuals = [r["sub_agg_actual"] for r in results]

# Separate empty vs non-empty
non_empty = [r for r in results if r["sub_agg_actual"] > 0]
non_empty_qe = [r["sub_agg_qerror"] for r in non_empty]

print(f"\n{'='*100}")
print(f"SUMMARY: {len(results)} queries profiled")
print(f"{'='*100}")

print(f"\n  Actual Row Counts:")
print(f"    Zero rows:    {sum(1 for a in actuals if a == 0):>4}")
print(f"    1-10 rows:    {sum(1 for a in actuals if 1 <= a <= 10):>4}")
print(f"    11-1K rows:   {sum(1 for a in actuals if 11 <= a <= 1000):>4}")
print(f"    1K-100K rows: {sum(1 for a in actuals if 1001 <= a <= 100000):>4}")
print(f"    100K+ rows:   {sum(1 for a in actuals if a > 100000):>4}")

s = sorted(sub_qe_vals)
print(f"\n  Sub-Aggregate Q-Error (ALL {len(s)} queries):")
print(f"    Min:      {s[0]:>12.1f}x")
print(f"    Median:   {s[len(s)//2]:>12.1f}x")
print(f"    Mean:     {sum(s)/len(s):>12.1f}x")
print(f"    90th pct: {s[int(len(s)*0.9)]:>12.1f}x")
print(f"    95th pct: {s[int(len(s)*0.95)]:>12.1f}x")
print(f"    Max:      {s[-1]:>12.1f}x")

if non_empty_qe:
    sne = sorted(non_empty_qe)
    print(f"\n  Sub-Aggregate Q-Error (NON-EMPTY {len(sne)} queries only):")
    print(f"    Min:      {sne[0]:>12.1f}x")
    print(f"    Median:   {sne[len(sne)//2]:>12.1f}x")
    print(f"    Mean:     {sum(sne)/len(sne):>12.1f}x")
    print(f"    90th pct: {sne[int(len(sne)*0.9)]:>12.1f}x")
    print(f"    95th pct: {sne[min(int(len(sne)*0.95), len(sne)-1)]:>12.1f}x")
    print(f"    Max:      {sne[-1]:>12.1f}x")

print(f"\n  Bias Distribution (sub-aggregate):")
under = sum(1 for r in results if r["sub_agg_bias"] == "UNDER")
over = sum(1 for r in results if r["sub_agg_bias"] == "OVER")
exact = sum(1 for r in results if r["sub_agg_bias"] == "EXACT")
print(f"    UNDER: {under:>4} ({100*under/len(results):.0f}%)")
print(f"    OVER:  {over:>4} ({100*over/len(results):.0f}%)")
print(f"    EXACT: {exact:>4} ({100*exact/len(results):.0f}%)")

print(f"\n  Q-Error > thresholds (sub-aggregate):")
for threshold in [2, 5, 10, 50, 100, 1000]:
    n = sum(1 for q in sub_qe_vals if q > threshold)
    print(f"    > {threshold:>6}x: {n:>4} ({100*n/len(sub_qe_vals):.0f}%)")

# Per-node stats
all_qe = [n["qe"] for n in all_nodes if n["node_type"] not in ("Aggregate", "Result")]
print(f"\n  All Non-Aggregate Nodes ({len(all_qe)} total):")
print(f"    q-error > 10x:    {sum(1 for q in all_qe if q > 10):>5} ({100*sum(1 for q in all_qe if q > 10)/len(all_qe):.1f}%)")
print(f"    q-error > 100x:   {sum(1 for q in all_qe if q > 100):>5} ({100*sum(1 for q in all_qe if q > 100)/len(all_qe):.1f}%)")
print(f"    q-error > 1000x:  {sum(1 for q in all_qe if q > 1000):>5} ({100*sum(1 for q in all_qe if q > 1000)/len(all_qe):.1f}%)")

print(f"\n  Top 15 Worst Q-Errors (sub-aggregate):")
for r in sorted(results, key=lambda x: -x["sub_agg_qerror"])[:15]:
    print(f"    {r['query']:<8} actual={r['sub_agg_actual']:>10,.0f}  est={r['sub_agg_estimate']:>10,.0f}  qe={r['sub_agg_qerror']:>10.1f}x  {r['sub_agg_bias']:<6} via {r['sub_agg_node']}")

# ── Save CSVs ──
out_q = Path(__file__).parent / "job_adapted_qerror_per_query.csv"
with open(out_q, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

out_n = Path(__file__).parent / "job_adapted_qerror_all_nodes.csv"
with open(out_n, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["query","depth","node_type","est","act","loops","qe"])
    w.writeheader()
    w.writerows(all_nodes)

print(f"\n  Saved: {out_q.name} ({len(results)} rows)")
print(f"  Saved: {out_n.name} ({len(all_nodes)} rows)")

cur.close()
conn.close()
