"""Full Postgres EXPLAIN ANALYZE profiling for all original JOB queries.

For each query, walks the entire plan tree and computes q-error at every
operator node.  Reports per-query worst q-error, overall statistics,
and saves detailed per-node results to CSV.
"""
import sys, io, os, time, re, csv, math, json
from pathlib import Path

import psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB = dict(host="localhost", port=5433, dbname="imdb_benchmark", user="postgres", password="bench")
QUERY_DIR = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\imdb_job_dataset\job_queries")
OUT_DIR = Path(__file__).parent

conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()

# Ensure fresh stats
cur.execute("ANALYZE")

def _sort_key(p):
    m = re.match(r"(\d+)([a-z]?)", p.stem)
    return (int(m.group(1)), m.group(2)) if m else (999, p.stem)

sql_files = sorted(QUERY_DIR.glob("*.sql"), key=_sort_key)
# Filter out DDL files
sql_files = [f for f in sql_files if f.stem not in ("schema", "fkindexes")]

def walk_plan(node, depth=0):
    """Recursively walk a JSON plan tree, yielding (depth, node_type, est, act, loops, qe, bias)."""
    rows = []
    nt = node.get("Node Type", "")
    est = float(node.get("Plan Rows", 1))
    loops = max(int(node.get("Actual Loops", 1)), 1)
    act_raw = float(node.get("Actual Rows", 0))
    act_total = act_raw * loops  # total rows across all loops

    # Avoid division by zero
    est_safe = max(est, 0.5)
    act_safe = max(act_total, 0.5)

    qe = max(est_safe / act_safe, act_safe / est_safe)
    bias = math.log2(est_safe / act_safe)  # positive = over-estimate, negative = under-estimate

    rows.append({
        "depth": depth,
        "node_type": nt,
        "est_rows": est,
        "act_rows": act_total,
        "loops": loops,
        "q_error": round(qe, 2),
        "bias_log2": round(bias, 2),
        "bias_dir": "UNDER" if bias < -0.1 else ("OVER" if bias > 0.1 else "EXACT"),
    })

    for sub in node.get("Plans", []):
        rows.extend(walk_plan(sub, depth + 1))
    return rows


print(f"JOB Q-Error Profiling: {len(sql_files)} queries")
print(f"Database: {DB['dbname']} @ {DB['host']}:{DB['port']}")
print()
print(f"  {'Query':<8} {'Joins':>5} {'Nodes':>5} {'Worst QE':>10} {'At Node':<24} {'Est':>10} {'Act':>10} {'Bias':<6} {'ms':>7}")
print(f"  {'-'*95}")

query_results = []
all_node_rows = []

for sf in sql_files:
    qid = sf.stem
    sql = sf.read_text(encoding="utf-8").strip().rstrip(";")

    t0 = time.time()
    try:
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
        plan_json = cur.fetchone()[0]
        elapsed = (time.time() - t0) * 1000

        plan = plan_json[0]["Plan"]
        planning_time = plan_json[0].get("Planning Time", 0)
        execution_time = plan_json[0].get("Execution Time", 0)
        nodes = walk_plan(plan)

        # Count joins (number of tables referenced)
        n_joins = sql.lower().count(" join ") + max(0, sql.count(",") - sql.count("("))

        # Tag each node with query id
        for n in nodes:
            n["query"] = qid
            all_node_rows.append(n)

        # Find worst q-error node (skip nodes with act_rows == 0 and est_rows == 0)
        real_nodes = [n for n in nodes if n["act_rows"] > 0.5 or n["est_rows"] > 0.5]
        if not real_nodes:
            real_nodes = nodes

        worst = max(real_nodes, key=lambda x: x["q_error"])

        # Root node stats (final result)
        root = nodes[0]

        query_results.append({
            "query": qid,
            "n_joins": n_joins,
            "n_nodes": len(nodes),
            "worst_qe": worst["q_error"],
            "worst_node": worst["node_type"],
            "worst_est": worst["est_rows"],
            "worst_act": worst["act_rows"],
            "worst_bias": worst["bias_dir"],
            "root_est": root["est_rows"],
            "root_act": root["act_rows"],
            "root_qe": root["q_error"],
            "planning_ms": round(planning_time, 2),
            "execution_ms": round(execution_time, 2),
            "total_ms": round(elapsed, 1),
        })

        print(
            f"  {qid:<8} {n_joins:>5} {len(nodes):>5} {worst['q_error']:>10.1f}x "
            f"{worst['node_type'][:22]:<24} {worst['est_rows']:>10.0f} {worst['act_rows']:>10.0f} "
            f"{worst['bias_dir']:<6} {elapsed:>6.0f}",
            flush=True,
        )

    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        err_msg = str(ex).split("\n")[0][:60]
        print(f"  {qid:<8}  ERROR: {err_msg} ({elapsed:.0f}ms)", flush=True)
        conn.rollback()

# ── Summary ──
qe_values = [r["worst_qe"] for r in query_results]
root_qe_values = [r["root_qe"] for r in query_results]
all_qe = [n["q_error"] for n in all_node_rows]

print(f"\n{'='*95}")
print(f"SUMMARY: {len(query_results)} queries profiled")
print(f"{'='*95}")

print(f"\n  Per-Query Worst Q-Error:")
print(f"    Min:      {min(qe_values):>10.1f}x")
print(f"    Median:   {sorted(qe_values)[len(qe_values)//2]:>10.1f}x")
print(f"    Mean:     {sum(qe_values)/len(qe_values):>10.1f}x")
print(f"    90th pct: {sorted(qe_values)[int(len(qe_values)*0.9)]:>10.1f}x")
print(f"    95th pct: {sorted(qe_values)[int(len(qe_values)*0.95)]:>10.1f}x")
print(f"    Max:      {max(qe_values):>10.1f}x")

print(f"\n  Root-Node Q-Error (final cardinality):")
print(f"    Min:      {min(root_qe_values):>10.1f}x")
print(f"    Median:   {sorted(root_qe_values)[len(root_qe_values)//2]:>10.1f}x")
print(f"    Mean:     {sum(root_qe_values)/len(root_qe_values):>10.1f}x")
print(f"    Max:      {max(root_qe_values):>10.1f}x")

print(f"\n  All Nodes ({len(all_qe)} total):")
print(f"    q-error > 10x:    {sum(1 for q in all_qe if q > 10):>5} ({100*sum(1 for q in all_qe if q > 10)/len(all_qe):.1f}%)")
print(f"    q-error > 100x:   {sum(1 for q in all_qe if q > 100):>5} ({100*sum(1 for q in all_qe if q > 100)/len(all_qe):.1f}%)")
print(f"    q-error > 1000x:  {sum(1 for q in all_qe if q > 1000):>5} ({100*sum(1 for q in all_qe if q > 1000)/len(all_qe):.1f}%)")

print(f"\n  Bias Distribution (worst node per query):")
under = sum(1 for r in query_results if r["worst_bias"] == "UNDER")
over = sum(1 for r in query_results if r["worst_bias"] == "OVER")
exact = sum(1 for r in query_results if r["worst_bias"] == "EXACT")
print(f"    UNDER-estimates: {under:>4} ({100*under/len(query_results):.0f}%)")
print(f"    OVER-estimates:  {over:>4} ({100*over/len(query_results):.0f}%)")
print(f"    EXACT:           {exact:>4} ({100*exact/len(query_results):.0f}%)")

# Top 10 worst
print(f"\n  Top 10 Worst Q-Errors:")
for r in sorted(query_results, key=lambda x: -x["worst_qe"])[:10]:
    print(f"    {r['query']:<8} {r['worst_qe']:>10.1f}x  {r['worst_node']:<22} est={r['worst_est']:.0f} act={r['worst_act']:.0f}  {r['worst_bias']}")

# ── Save CSVs ──
with open(OUT_DIR / "job_qerror_per_query.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(query_results[0].keys()))
    w.writeheader()
    w.writerows(query_results)

with open(OUT_DIR / "job_qerror_all_nodes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_node_rows[0].keys()))
    w.writeheader()
    w.writerows(all_node_rows)

print(f"\n  Saved: job_qerror_per_query.csv ({len(query_results)} rows)")
print(f"  Saved: job_qerror_all_nodes.csv ({len(all_node_rows)} rows)")

cur.close()
conn.close()
