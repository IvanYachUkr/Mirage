"""
Analyze whether the q-error gap is driven by SCALE or STRUCTURE.

Key tests:
1. Correlation between actual_rows ratio and q-error ratio
   → If high correlation: scaling will help
   → If low correlation: structural issue

2. Normalized q-error: QE / actual_rows
   → If similar across datasets: pure scale effect
   → If different: structural differences in data correlations

3. Postgres estimation accuracy: est/actual ratio comparison
   → Same bias magnitude = same correlation structure
   → Different = different correlations
"""
import csv, sys, io, math, statistics

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Load
real, synth = {}, {}
with open("job_real_imdb_results.csv") as f:
    for r in csv.DictReader(f): real[r["query"]] = r
with open("job_adapted_qerror_per_query.csv") as f:
    for r in csv.DictReader(f): synth[r["query"]] = r

queries = sorted(set(real.keys()) & set(synth.keys()),
                 key=lambda q: (int(''.join(c for c in q if c.isdigit())), q))

print("=" * 80)
print("TEST 1: Does the actual-row ratio predict the q-error ratio?")
print("=" * 80)
print("If scaling helps, queries where real returns N× more rows should have ~N× more q-error.\n")

pairs = []
for q in queries:
    r_act = max(float(real[q]["sub_agg_actual"]), 1)
    s_act = max(float(synth[q]["sub_agg_actual"]), 1)
    r_qe = float(real[q]["sub_agg_qe"])
    s_qe = float(synth[q]["sub_agg_qerror"])
    row_ratio = r_act / s_act  # how many more rows real has
    qe_ratio = r_qe / max(s_qe, 0.1)  # how much worse real QE is
    pairs.append((q, r_act, s_act, row_ratio, r_qe, s_qe, qe_ratio))

# Correlation between log(row_ratio) and log(qe_ratio)
log_rr = [math.log(max(p[3], 0.01)) for p in pairs]
log_qr = [math.log(max(p[6], 0.01)) for p in pairs]
n = len(log_rr)
mean_rr = sum(log_rr) / n
mean_qr = sum(log_qr) / n
cov = sum((a - mean_rr) * (b - mean_qr) for a, b in zip(log_rr, log_qr)) / n
std_rr = (sum((a - mean_rr)**2 for a in log_rr) / n) ** 0.5
std_qr = (sum((a - mean_qr)**2 for a in log_qr) / n) ** 0.5
corr = cov / (std_rr * std_qr) if std_rr * std_qr > 0 else 0

print(f"  Pearson correlation(log row_ratio, log qe_ratio) = {corr:.3f}")
if corr > 0.5:
    print(f"  → STRONG positive correlation: scaling WOULD likely help")
elif corr > 0.2:
    print(f"  → MODERATE correlation: scaling would partially help")
else:
    print(f"  → WEAK correlation: q-error gap is NOT primarily about scale")

print(f"\n  Top 10 queries sorted by row ratio (real has most more rows):")
by_rr = sorted(pairs, key=lambda p: p[3], reverse=True)[:10]
print(f"  {'Q':<6} {'Real Act':>10} {'Synth Act':>10} {'Row Ratio':>10} {'Real QE':>10} {'Synth QE':>10} {'QE Ratio':>10}")
for q, r_a, s_a, rr, r_q, s_q, qr in by_rr:
    print(f"  {q:<6} {r_a:>10,.0f} {s_a:>10,.0f} {rr:>10,.1f}x {r_q:>10,.1f}x {s_q:>10,.1f}x {qr:>10,.1f}x")

print(f"\n" + "=" * 80)
print("TEST 2: Estimation accuracy — how far is est from actual, PROPORTIONALLY?")
print("=" * 80)
print("log2(est/actual) = 0 means perfect. Negative = UNDER.\n")

r_log_ratios = []
s_log_ratios = []
for q in queries:
    r_act = max(float(real[q]["sub_agg_actual"]), 1)
    r_est = max(float(real[q]["sub_agg_est"]), 1)
    s_act = max(float(synth[q]["sub_agg_actual"]), 1)
    s_est = max(float(synth[q]["sub_agg_estimate"]), 1)
    r_log_ratios.append(math.log2(r_est / r_act))
    s_log_ratios.append(math.log2(s_est / s_act))

print(f"  {'Metric':<30} {'Real IMDB':>12} {'Synthetic':>12}")
print(f"  {'-'*54}")
print(f"  {'Mean log2(est/act)':<30} {statistics.mean(r_log_ratios):>12.2f} {statistics.mean(s_log_ratios):>12.2f}")
print(f"  {'Median log2(est/act)':<30} {statistics.median(r_log_ratios):>12.2f} {statistics.median(s_log_ratios):>12.2f}")
print(f"  {'Std log2(est/act)':<30} {statistics.stdev(r_log_ratios):>12.2f} {statistics.stdev(s_log_ratios):>12.2f}")
print(f"  {'Min log2(est/act)':<30} {min(r_log_ratios):>12.2f} {min(s_log_ratios):>12.2f}")
print(f"  {'Max log2(est/act)':<30} {max(r_log_ratios):>12.2f} {max(s_log_ratios):>12.2f}")

print(f"\n  Interpretation:")
print(f"  - Mean for Real={statistics.mean(r_log_ratios):.2f} vs Synth={statistics.mean(s_log_ratios):.2f}")
if abs(statistics.mean(r_log_ratios) - statistics.mean(s_log_ratios)) < 2:
    print(f"  → SIMILAR estimation bias = similar correlation structure")
    print(f"  → Scaling would bring q-errors closer")
else:
    print(f"  → DIFFERENT estimation bias = different correlation structure")
    print(f"  → Scaling alone may not be enough")

print(f"\n" + "=" * 80)
print("TEST 3: Queries where synthetic returns FEWER rows — is QE proportional?")
print("=" * 80)

# Bucket by how many fewer rows synthetic has
buckets = {"1-10x fewer": [], "10-100x fewer": [], "100-1000x fewer": [], "1000x+ fewer": [], "synth has MORE": []}
for q, r_a, s_a, rr, r_q, s_q, qr in pairs:
    if rr < 1:
        buckets["synth has MORE"].append((q, rr, r_q, s_q, qr))
    elif rr < 10:
        buckets["1-10x fewer"].append((q, rr, r_q, s_q, qr))
    elif rr < 100:
        buckets["10-100x fewer"].append((q, rr, r_q, s_q, qr))
    elif rr < 1000:
        buckets["100-1000x fewer"].append((q, rr, r_q, s_q, qr))
    else:
        buckets["1000x+ fewer"].append((q, rr, r_q, s_q, qr))

print(f"\n  {'Bucket':<20} {'Count':>6} {'Avg Row Ratio':>14} {'Avg QE Ratio':>14} {'Scale predicts?'}")
print(f"  {'-'*70}")
for bname, items in buckets.items():
    if not items:
        print(f"  {bname:<20} {0:>6}")
        continue
    avg_rr = statistics.mean([i[1] for i in items])
    avg_qr = statistics.mean([i[4] for i in items])
    scale_match = "YES" if 0.2 < avg_qr / max(avg_rr, 0.01) < 5 else "NO"
    print(f"  {bname:<20} {len(items):>6} {avg_rr:>14.1f}x {avg_qr:>14.1f}x {scale_match}")

print(f"\n" + "=" * 80)
print("TEST 4: Per-query 'difficulty ratio' — QE / sqrt(actual_rows)")  
print("=" * 80)
print("If this is SIMILAR across datasets, q-error scales with sqrt(rows) → scaling helps.\n")

r_diff = [float(real[q]["sub_agg_qe"]) / max(float(real[q]["sub_agg_actual"]) ** 0.5, 1) for q in queries]
s_diff = [float(synth[q]["sub_agg_qerror"]) / max(float(synth[q]["sub_agg_actual"]) ** 0.5, 1) for q in queries]

print(f"  {'Metric':<30} {'Real IMDB':>12} {'Synthetic':>12}")
print(f"  {'-'*54}")
print(f"  {'Median QE/sqrt(act)':<30} {statistics.median(r_diff):>12.2f} {statistics.median(s_diff):>12.2f}")
print(f"  {'Mean QE/sqrt(act)':<30} {statistics.mean(r_diff):>12.2f} {statistics.mean(s_diff):>12.2f}")

print(f"\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
# Final analysis
synth_better = sum(1 for _, _, _, _, rq, sq, _ in pairs if rq > sq)
synth_worse = sum(1 for _, _, _, _, rq, sq, _ in pairs if sq > rq)
# Queries where synthetic has est=1 and actual=1 (trivially exact)
trivial = sum(1 for q in queries 
              if float(synth[q]["sub_agg_actual"]) <= 2 and float(synth[q]["sub_agg_qerror"]) <= 2)
print(f"\n  Real has higher QE: {synth_better}/113")
print(f"  Synth has higher QE: {synth_worse}/113")
print(f"  Synth queries with actual ≤ 2 rows (trivially easy): {trivial}/113")
print(f"  Correlation(row_ratio, qe_ratio): {corr:.3f}")
