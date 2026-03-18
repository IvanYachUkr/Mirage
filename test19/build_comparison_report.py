"""Build side-by-side comparison report: Real IMDB vs Synthetic dataset."""
import csv, sys, io, statistics
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Load results ──
real = {}
with open("job_real_imdb_results.csv") as f:
    for r in csv.DictReader(f):
        real[r["query"]] = r

synth = {}
with open("job_adapted_qerror_per_query.csv") as f:
    for r in csv.DictReader(f):
        synth[r["query"]] = r

# Normalize query names (both should have 1a, 1b, ... 33c)
all_queries = sorted(set(real.keys()) & set(synth.keys()),
                     key=lambda q: (int(''.join(c for c in q if c.isdigit())), q))

# ── Build markdown ──
lines = []
lines.append("# JOB Benchmark: Real IMDB vs Synthetic Dataset — Side-by-Side Comparison\n")
lines.append(f"> Generated from {len(all_queries)} matching queries\n")

# Summary table
lines.append("## Summary Statistics\n")
r_qe = [float(real[q]["sub_agg_qe"]) for q in all_queries]
s_qe = [float(synth[q]["sub_agg_qerror"]) for q in all_queries]

r_under = sum(1 for q in all_queries if real[q]["sub_agg_bias"] == "UNDER")
r_over = sum(1 for q in all_queries if real[q]["sub_agg_bias"] == "OVER")
r_exact = sum(1 for q in all_queries if real[q]["sub_agg_bias"] == "EXACT")
s_under = sum(1 for q in all_queries if synth[q]["sub_agg_bias"] == "UNDER")
s_over = sum(1 for q in all_queries if synth[q]["sub_agg_bias"] == "OVER")
s_exact = sum(1 for q in all_queries if synth[q]["sub_agg_bias"] == "EXACT")

r_sorted = sorted(r_qe)
s_sorted = sorted(s_qe)
n = len(all_queries)

lines.append("| Metric | Real IMDB | Synthetic | Ratio |")
lines.append("|--------|----------:|----------:|------:|")
lines.append(f"| **Titles** | **2,528,312** | **19,161** | 132x |")
lines.append(f"| **Cast rows** | **36,244,344** | **362,759** | 100x |")
lines.append(f"| **movie_info rows** | **14,835,720** | **341,913** | 43x |")
lines.append(f"| **Queries analyzed** | {n} | {n} | — |")
lines.append(f"| | | | |")
lines.append(f"| **Sub-agg median QE** | **{statistics.median(r_qe):,.1f}x** | **{statistics.median(s_qe):,.1f}x** | {statistics.median(r_qe)/max(statistics.median(s_qe),0.1):.1f}x |")
lines.append(f"| Sub-agg mean QE | {statistics.mean(r_qe):,.1f}x | {statistics.mean(s_qe):,.1f}x | {statistics.mean(r_qe)/max(statistics.mean(s_qe),0.1):.1f}x |")
lines.append(f"| Sub-agg max QE | {max(r_qe):,.1f}x | {max(s_qe):,.1f}x | {max(r_qe)/max(max(s_qe),0.1):.1f}x |")
lines.append(f"| Sub-agg 90th pct | {r_sorted[int(n*0.9)]:,.1f}x | {s_sorted[int(n*0.9)]:,.1f}x | {r_sorted[int(n*0.9)]/max(s_sorted[int(n*0.9)],0.1):.1f}x |")
lines.append(f"| Sub-agg 95th pct | {r_sorted[int(n*0.95)]:,.1f}x | {s_sorted[int(n*0.95)]:,.1f}x | {r_sorted[int(n*0.95)]/max(s_sorted[int(n*0.95)],0.1):.1f}x |")
lines.append(f"| | | | |")
lines.append(f"| UNDER-estimated | {r_under}/{n} ({100*r_under//n}%) | {s_under}/{n} ({100*s_under//n}%) | — |")
lines.append(f"| OVER-estimated | {r_over}/{n} ({100*r_over//n}%) | {s_over}/{n} ({100*s_over//n}%) | — |")
lines.append(f"| EXACT | {r_exact}/{n} ({100*r_exact//n}%) | {s_exact}/{n} ({100*s_exact//n}%) | — |")
lines.append(f"| | | | |")
lines.append(f"| QE > 2x | {sum(1 for q in r_qe if q>2)}/{n} ({100*sum(1 for q in r_qe if q>2)//n}%) | {sum(1 for q in s_qe if q>2)}/{n} ({100*sum(1 for q in s_qe if q>2)//n}%) | — |")
lines.append(f"| QE > 10x | {sum(1 for q in r_qe if q>10)}/{n} ({100*sum(1 for q in r_qe if q>10)//n}%) | {sum(1 for q in s_qe if q>10)}/{n} ({100*sum(1 for q in s_qe if q>10)//n}%) | — |")
lines.append(f"| QE > 100x | {sum(1 for q in r_qe if q>100)}/{n} ({100*sum(1 for q in r_qe if q>100)//n}%) | {sum(1 for q in s_qe if q>100)}/{n} ({100*sum(1 for q in s_qe if q>100)//n}%) | — |")
lines.append(f"| QE > 1000x | {sum(1 for q in r_qe if q>1000)}/{n} ({100*sum(1 for q in r_qe if q>1000)//n}%) | {sum(1 for q in s_qe if q>1000)}/{n} ({100*sum(1 for q in s_qe if q>1000)//n}%) | — |")
lines.append(f"| QE > 10000x | {sum(1 for q in r_qe if q>10000)}/{n} ({100*sum(1 for q in r_qe if q>10000)//n}%) | {sum(1 for q in s_qe if q>10000)}/{n} ({100*sum(1 for q in s_qe if q>10000)//n}%) | — |")
lines.append("")

# Full per-query table
lines.append("## Per-Query Comparison (Sub-Aggregate Q-Error)\n")
lines.append("| Query | Real Actual | Real Est | Real QE | Real Bias | Synth Actual | Synth Est | Synth QE | Synth Bias | Winner |")
lines.append("|-------|----------:|--------:|--------:|:---------:|----------:|--------:|--------:|:---------:|:------:|")

for q in all_queries:
    r = real[q]
    s = synth[q]
    r_act = float(r["sub_agg_actual"])
    r_est = float(r["sub_agg_est"])
    r_q = float(r["sub_agg_qe"])
    r_b = r["sub_agg_bias"]
    s_act = float(s["sub_agg_actual"])
    s_est = float(s["sub_agg_estimate"])
    s_q = float(s["sub_agg_qerror"])
    s_b = s["sub_agg_bias"]
    
    # Who has lower q-error?
    if abs(r_q - s_q) < 0.5:
        winner = "≈"
    elif r_q < s_q:
        winner = "Real"
    else:
        winner = "Synth"
    
    lines.append(f"| **{q}** | {r_act:,.0f} | {r_est:,.0f} | {r_q:,.1f}x | {r_b} | {s_act:,.0f} | {s_est:,.0f} | {s_q:,.1f}x | {s_b} | {winner} |")

lines.append("")

# Bias agreement analysis
lines.append("## Bias Direction Agreement\n")
agree = sum(1 for q in all_queries if real[q]["sub_agg_bias"] == synth[q]["sub_agg_bias"])
disagree_list = [(q, real[q]["sub_agg_bias"], synth[q]["sub_agg_bias"]) 
                 for q in all_queries if real[q]["sub_agg_bias"] != synth[q]["sub_agg_bias"]]
lines.append(f"**{agree}/{n} queries** ({100*agree//n}%) have the same bias direction.\n")
if disagree_list:
    lines.append("| Query | Real Bias | Synth Bias | Real QE | Synth QE |")
    lines.append("|-------|:---------:|:----------:|--------:|--------:|")
    for q, rb, sb in disagree_list:
        lines.append(f"| {q} | {rb} | {sb} | {float(real[q]['sub_agg_qe']):,.1f}x | {float(synth[q]['sub_agg_qerror']):,.1f}x |")
    lines.append("")

# Query family analysis
lines.append("## Query Family Analysis\n")
families = {}
for q in all_queries:
    fam = ''.join(c for c in q if c.isdigit())
    if fam not in families:
        families[fam] = []
    families[fam].append(q)

lines.append("| Family | Queries | Real Median QE | Synth Median QE | Real Max QE | Synth Max QE |")
lines.append("|--------|:-------:|--------------:|---------------:|-----------:|------------:|")
for fam in sorted(families.keys(), key=int):
    qs = families[fam]
    r_fam = [float(real[q]["sub_agg_qe"]) for q in qs]
    s_fam = [float(synth[q]["sub_agg_qerror"]) for q in qs]
    lines.append(f"| Q{fam} | {len(qs)} | {statistics.median(r_fam):,.1f}x | {statistics.median(s_fam):,.1f}x | {max(r_fam):,.1f}x | {max(s_fam):,.1f}x |")
lines.append("")

# Worst queries comparison
lines.append("## Top 15 Worst Queries — Real IMDB\n")
worst_real = sorted(all_queries, key=lambda q: float(real[q]["sub_agg_qe"]), reverse=True)[:15]
lines.append("| Rank | Query | Real QE | Real Actual | Real Bias | Synth QE | Synth Actual | Synth Bias |")
lines.append("|-----:|-------|--------:|-----------:|:---------:|--------:|-----------:|:---------:|")
for i, q in enumerate(worst_real, 1):
    r = real[q]
    s = synth[q]
    lines.append(f"| {i} | **{q}** | **{float(r['sub_agg_qe']):,.1f}x** | {float(r['sub_agg_actual']):,.0f} | {r['sub_agg_bias']} | {float(s['sub_agg_qerror']):,.1f}x | {float(s['sub_agg_actual']):,.0f} | {s['sub_agg_bias']} |")
lines.append("")

lines.append("## Top 15 Worst Queries — Synthetic\n")
worst_synth = sorted(all_queries, key=lambda q: float(synth[q]["sub_agg_qerror"]), reverse=True)[:15]
lines.append("| Rank | Query | Synth QE | Synth Actual | Synth Bias | Real QE | Real Actual | Real Bias |")
lines.append("|-----:|-------|--------:|-----------:|:---------:|--------:|-----------:|:---------:|")
for i, q in enumerate(worst_synth, 1):
    r = real[q]
    s = synth[q]
    lines.append(f"| {i} | **{q}** | **{float(s['sub_agg_qerror']):,.1f}x** | {float(s['sub_agg_actual']):,.0f} | {s['sub_agg_bias']} | {float(r['sub_agg_qe']):,.1f}x | {float(r['sub_agg_actual']):,.0f} | {r['sub_agg_bias']} |")
lines.append("")

# Where synthetic BEATS real
lines.append("## Where Synthetic Has HIGHER Q-Error Than Real\n")
synth_worse = [(q, float(synth[q]["sub_agg_qerror"]), float(real[q]["sub_agg_qe"]))
               for q in all_queries if float(synth[q]["sub_agg_qerror"]) > float(real[q]["sub_agg_qe"])]
synth_worse.sort(key=lambda x: x[1]/max(x[2], 0.1), reverse=True)
lines.append(f"**{len(synth_worse)}/{n} queries** have higher q-error on synthetic than real.\n")
if synth_worse:
    lines.append("| Query | Synth QE | Real QE | Synth/Real Ratio |")
    lines.append("|-------|--------:|--------:|----------------:|")
    for q, sq, rq in synth_worse[:20]:
        lines.append(f"| {q} | {sq:,.1f}x | {rq:,.1f}x | {sq/max(rq,0.1):,.1f}x |")
lines.append("")

# Execution time comparison
lines.append("## Execution Time Comparison\n")
r_ms = [float(real[q]["ms"]) for q in all_queries]
s_ms = [float(synth[q]["total_ms"]) for q in all_queries]
lines.append("| Metric | Real IMDB | Synthetic |")
lines.append("|--------|----------:|----------:|")
lines.append(f"| Median query time | {statistics.median(r_ms):,.0f} ms | {statistics.median(s_ms):,.0f} ms |")
lines.append(f"| Mean query time | {statistics.mean(r_ms):,.0f} ms | {statistics.mean(s_ms):,.0f} ms |")
lines.append(f"| Max query time | {max(r_ms):,.0f} ms | {max(s_ms):,.0f} ms |")
lines.append(f"| Total benchmark time | {sum(r_ms)/1000:,.1f} s | {sum(s_ms)/1000:,.1f} s |")
lines.append("")

report = "\n".join(lines)

out = Path(r"C:\Users\vanya\.gemini\antigravity\brain\a6c4321f-31a7-4dbc-89fc-0b2fa7e01449\job_comparison_report.md")
out.write_text(report, encoding="utf-8")
print(f"Report written to {out}")
print(f"\nQuick stats:")
print(f"  Bias agreement: {agree}/{n} ({100*agree//n}%)")
print(f"  Synth worse: {len(synth_worse)}/{n}")
print(f"  Real median QE: {statistics.median(r_qe):.1f}x")
print(f"  Synth median QE: {statistics.median(s_qe):.1f}x")
