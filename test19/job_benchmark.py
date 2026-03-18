#!/usr/bin/env python3
"""
job_benchmark.py
================
Runs a representative subset of the official Join Order Benchmark (JOB)
queries against the imdb_benchmark database, measuring cardinality estimation
q-error using EXPLAIN ANALYZE.

JOB queries are adapted from: https://github.com/gregrahn/join-order-benchmark
The 113 JOB queries span Q1a through Q33 with lettered variants.
We run a curated set of ~30 representative queries across all complexity tiers.

Usage:
    python job_benchmark.py [--host HOST] [--port PORT] [--csv FILE]
"""
import sys, io, argparse, time, math, csv, statistics
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import psycopg2

# ── Job queries ────────────────────────────────────────────────────────────
# Format: (query_id, n_joins, description, sql)
# Adapted for our schema: title.id, name.id, cast_info.movie_id/person_id
# movie_info uses (movie_id, info_type_id, info) structure
# We skip queries requiring complete_cast / comp_cast_type (not populated)

JOB_QUERIES = [

# ── Q1: 2-way joins ─────────────────────────────────────────────────────
("JOB-1a", 2, "Movie + company info",
"""
SELECT MIN(mc.note) AS production_note,
       MIN(t.title) AS movie_title,
       MIN(t.production_year) AS movie_year
FROM company_name cn, movie_companies mc, title t
WHERE cn.country_code = '[us]'
  AND t.production_year > 1990
  AND t.id = mc.movie_id
  AND mc.company_id = cn.id
"""),

("JOB-1b", 2, "Movie + info (genres)",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(mi.info) AS movie_genre
FROM title t, movie_info mi
WHERE t.production_year > 2000
  AND mi.info_type_id = 3
  AND t.id = mi.movie_id
"""),

# ── Q2: Cast + name ─────────────────────────────────────────────────────
("JOB-2a", 2, "Cast info join name",
"""
SELECT MIN(n.name) AS person_name,
       MIN(t.title) AS movie_title
FROM cast_info ci, name n, title t
WHERE ci.role_id = 1
  AND n.id = ci.person_id
  AND t.id = ci.movie_id
  AND t.production_year > 2005
"""),

("JOB-2b", 2, "Actress in drama",
"""
SELECT MIN(n.name) AS actress_name,
       MIN(t.title) AS movie_title
FROM cast_info ci, name n, title t,
     movie_info mi
WHERE ci.role_id = 2
  AND mi.info = 'Drama'
  AND mi.info_type_id = 3
  AND n.id = ci.person_id
  AND t.id = ci.movie_id
  AND mi.movie_id = t.id
"""),

# ── Q3: Movie info + keyword ─────────────────────────────────────────────
("JOB-3a", 3, "Genre + keyword + title",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(k.keyword) AS movie_keyword
FROM title t, movie_info mi, keyword k, movie_keyword mk
WHERE mi.info = 'Action'
  AND mi.info_type_id = 3
  AND k.id = mk.keyword_id
  AND t.id = mk.movie_id
  AND mi.movie_id = t.id
"""),

("JOB-3b", 3, "Country + keyword + title",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(k.keyword) AS kw
FROM title t, movie_info mi, keyword k, movie_keyword mk
WHERE mi.info = 'USA'
  AND mi.info_type_id = 8
  AND t.production_year >= 2000
  AND k.id = mk.keyword_id
  AND t.id = mk.movie_id
  AND mi.movie_id = t.id
"""),

# ── Q4: Company + genre ──────────────────────────────────────────────────
("JOB-4a", 4, "Company type + genre + keyword",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(cn.name) AS company_name
FROM title t, movie_info mi, company_name cn,
     movie_companies mc, keyword k, movie_keyword mk
WHERE mi.info = 'Horror'
  AND mi.info_type_id = 3
  AND mc.company_type_id = 1
  AND cn.id = mc.company_id
  AND t.id = mc.movie_id
  AND mi.movie_id = t.id
  AND k.id = mk.keyword_id
  AND mk.movie_id = t.id
"""),

("JOB-4b", 4, "US company + recent + keyword",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(cn.name) AS company_name
FROM title t, movie_info mi, company_name cn,
     movie_companies mc, keyword k, movie_keyword mk
WHERE cn.country_code = 'us'
  AND t.production_year > 2010
  AND mi.info_type_id = 3
  AND cn.id = mc.company_id
  AND t.id = mc.movie_id
  AND mi.movie_id = t.id
  AND k.id = mk.keyword_id
  AND mk.movie_id = t.id
"""),

# ── Q5: aka_name ─────────────────────────────────────────────────────────
("JOB-5a", 4, "Alternate name + cast + movie",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(an.name) AS alt_name
FROM title t, cast_info ci, name n,
     aka_name an, movie_info mi
WHERE mi.info = 'Drama'
  AND mi.info_type_id = 3
  AND n.id = ci.person_id
  AND an.person_id = n.id
  AND t.id = ci.movie_id
  AND mi.movie_id = t.id
"""),

# ── Q6: 4-way join ──────────────────────────────────────────────────────
("JOB-6a", 4, "Cast + role + movie + info",
"""
SELECT MIN(n.name) AS person_name,
       MIN(t.title) AS movie_title
FROM cast_info ci, name n, title t,
     role_type rt, movie_info mi
WHERE rt.role = 'director'
  AND mi.info = 'Drama'
  AND mi.info_type_id = 3
  AND t.production_year >= 1990
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND t.id = ci.movie_id
  AND mi.movie_id = t.id
"""),

# ── Q7: 5-way join ──────────────────────────────────────────────────────
("JOB-7a", 5, "5-way: title+cast+co+kw+info",
"""
SELECT MIN(n.name) AS person_name,
       MIN(t.title) AS movie_title,
       MIN(cn.name)  AS company_name
FROM title t, name n, cast_info ci,
     company_name cn, movie_companies mc,
     movie_info mi
WHERE t.production_year BETWEEN 2000 AND 2015
  AND mi.info = 'Action'
  AND mi.info_type_id = 3
  AND ci.role_id = 1
  AND mc.company_type_id = 1
  AND n.id = ci.person_id
  AND t.id = ci.movie_id
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND mi.movie_id = t.id
"""),

("JOB-7b", 5, "5-way: director in dramedy",
"""
SELECT MIN(n.name) AS director_name,
       MIN(t.title) AS movie_title
FROM title t, name n, cast_info ci,
     role_type rt, movie_info mi, keyword k, movie_keyword mk
WHERE rt.role IN ('director', 'writer')
  AND mi.info IN ('Drama', 'Comedy')
  AND mi.info_type_id = 3
  AND t.production_year > 1990
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND t.id = ci.movie_id
  AND mi.movie_id = t.id
  AND k.id = mk.keyword_id
  AND mk.movie_id = t.id
"""),

# ── Q8: 6-way join ──────────────────────────────────────────────────────
("JOB-8a", 6, "6-way: cast+co+kw+info+link",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(t2.title) AS linked_title
FROM title t, title t2, movie_link ml, link_type lt,
     movie_info mi, company_name cn, movie_companies mc
WHERE lt.link = 'sequel'
  AND mi.info = 'Action'
  AND mi.info_type_id = 3
  AND cn.country_code = 'us'
  AND t.id = ml.movie_id
  AND t2.id = ml.linked_movie_id
  AND lt.id = ml.link_type_id
  AND mi.movie_id = t.id
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
"""),

("JOB-8b", 6, "6-way: cast+name+role+info+kw+co",
"""
SELECT MIN(n.name) AS person_name,
       MIN(t.title) AS movie_title,
       MIN(cn.name) AS company_name
FROM title t, name n, cast_info ci, role_type rt,
     movie_info mi, company_name cn, movie_companies mc
WHERE rt.role = 'actor'
  AND mi.info = 'USA'
  AND mi.info_type_id = 8
  AND mc.company_type_id IN (1, 2)
  AND t.production_year >= 2000
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND t.id = ci.movie_id
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND mi.movie_id = t.id
"""),

# ── Q9: Company + movie_link ─────────────────────────────────────────────
("JOB-9a", 6, "Links between movies + companies",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(t2.title) AS sequel_title,
       MIN(cn.name)  AS studio
FROM title t, title t2, movie_link ml, link_type lt,
     company_name cn, movie_companies mc1, movie_companies mc2
WHERE lt.link IN ('sequel', 'followed by')
  AND cn.country_code = 'us'
  AND t.id = ml.movie_id
  AND t2.id = ml.linked_movie_id
  AND lt.id = ml.link_type_id
  AND cn.id = mc1.company_id
  AND mc1.movie_id = t.id
  AND mc2.movie_id = t2.id
"""),

# ── Q10: aka_title ───────────────────────────────────────────────────────
("JOB-10a", 5, "Alternate titles + info + cast",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(at.title) AS alt_title,
       MIN(n.name) AS actor_name
FROM title t, aka_title at, cast_info ci, name n, movie_info mi
WHERE mi.info = 'Drama'
  AND mi.info_type_id = 3
  AND ci.role_id = 1
  AND at.movie_id = t.id
  AND n.id = ci.person_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
"""),

# ── Q11: person_info ─────────────────────────────────────────────────────
("JOB-11a", 5, "Person info + cast + movie",
"""
SELECT MIN(n.name) AS person_name,
       MIN(pi.info) AS birth_info,
       MIN(t.title) AS movie_title
FROM title t, name n, cast_info ci,
     person_info pi, info_type it
WHERE it.info = 'birth date'
  AND n.id = ci.person_id
  AND pi.person_id = n.id
  AND it.id = pi.info_type_id
  AND t.id = ci.movie_id
  AND t.production_year > 2000
"""),

("JOB-11b", 5, "Person nationality + drama",
"""
SELECT MIN(n.name) AS person_name,
       MIN(pi.info) AS nationality,
       MIN(t.title) AS movie_title
FROM title t, name n, cast_info ci,
     person_info pi, info_type it,
     movie_info mi
WHERE it.info = 'nationality'
  AND mi.info = 'Drama'
  AND mi.info_type_id = 3
  AND n.id = ci.person_id
  AND pi.person_id = n.id
  AND it.id = pi.info_type_id
  AND t.id = ci.movie_id
  AND mi.movie_id = t.id
"""),

# ── Q12: 7-way join ──────────────────────────────────────────────────────
("JOB-12a", 7, "7-way: full cast+co+info+kw+person_info",
"""
SELECT MIN(n.name) AS person_name,
       MIN(t.title) AS movie_title,
       MIN(k.keyword) AS kw
FROM title t, name n, cast_info ci, role_type rt,
     movie_info mi, keyword k, movie_keyword mk,
     company_name cn, movie_companies mc
WHERE rt.role = 'actor'
  AND mi.info IN ('Action', 'Thriller')
  AND mi.info_type_id = 3
  AND t.production_year >= 2005
  AND cn.country_code = 'us'
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND k.id = mk.keyword_id
  AND mk.movie_id = t.id
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
"""),

("JOB-12b", 7, "7-way: actor+company+link+info",
"""
SELECT MIN(n.name) AS person_name,
       MIN(t.title) AS movie_title,
       MIN(t2.title) AS related_title
FROM title t, title t2, name n, cast_info ci,
     movie_link ml, link_type lt,
     movie_info mi, company_name cn, movie_companies mc
WHERE lt.link IN ('sequel', 'remake of')
  AND mi.info = 'USA'
  AND mi.info_type_id = 8
  AND n.id = ci.person_id
  AND ci.movie_id = t.id
  AND t.id = ml.movie_id
  AND t2.id = ml.linked_movie_id
  AND lt.id = ml.link_type_id
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND mi.movie_id = t.id
"""),

# ── Q13: 8-way join ──────────────────────────────────────────────────────
("JOB-13a", 8, "8-way: all core tables",
"""
SELECT MIN(n.name) AS person_name,
       MIN(t.title) AS movie_title,
       MIN(k.keyword) AS kw,
       MIN(cn.name) AS company_name
FROM title t, name n, cast_info ci, role_type rt,
     movie_info mi, keyword k, movie_keyword mk,
     company_name cn, movie_companies mc,
     person_info pi, info_type it
WHERE rt.role = 'actor'
  AND mi.info = 'Drama'
  AND mi.info_type_id = 3
  AND it.info = 'nationality'
  AND cn.country_code = 'us'
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND k.id = mk.keyword_id
  AND mk.movie_id = t.id
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND pi.person_id = n.id
  AND it.id = pi.info_type_id
"""),

("JOB-13b", 8, "8-way: director + company + genre + link",
"""
SELECT MIN(n.name) AS director_name,
       MIN(t.title) AS movie_title,
       MIN(t2.title) AS sequel_title
FROM title t, title t2, name n, cast_info ci, role_type rt,
     movie_link ml, link_type lt,
     movie_info mi, company_name cn, movie_companies mc
WHERE rt.role = 'director'
  AND lt.link = 'followed by'
  AND mi.info IN ('Action', 'Sci-Fi')
  AND mi.info_type_id = 3
  AND cn.country_code = 'us'
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND ci.movie_id = t.id
  AND t.id = ml.movie_id
  AND t2.id = ml.linked_movie_id
  AND lt.id = ml.link_type_id
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND mi.movie_id = t.id
"""),

# ── Q14: 9-way join ──────────────────────────────────────────────────────
("JOB-14a", 9, "9-way: cast+co+info+kw+link+aka+person",
"""
SELECT MIN(t.title) AS movie_title,
       MIN(n.name) AS actor_name
FROM title t, name n, cast_info ci,
     movie_info mi, company_name cn, movie_companies mc,
     keyword k, movie_keyword mk,
     aka_title at, movie_link ml
WHERE mi.info = 'USA'
  AND mi.info_type_id = 8
  AND cn.country_code = 'us'
  AND ci.role_id = 1
  AND n.id = ci.person_id
  AND ci.movie_id = t.id
  AND mi.movie_id = t.id
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND k.id = mk.keyword_id
  AND mk.movie_id = t.id
  AND at.movie_id = t.id
  AND ml.movie_id = t.id
"""),

# ── Bonus: V14-specific extras (not in JOB — our differentiator) ─────────
("V14-edges-1", 4, "Graph: actors in same movie via edge",
"""
SELECT MIN(n1.name) AS actor1, MIN(n2.name) AS actor2,
       MIN(t.title) AS movie_title
FROM name n1, name n2, edges e, cast_info ci1, cast_info ci2, title t
WHERE e.src_id = n1.id
  AND e.dst_id = n2.id
  AND e.edge_type = 'friendship'
  AND ci1.person_id = n1.id
  AND ci2.person_id = n2.id
  AND ci1.movie_id = t.id
  AND ci2.movie_id = t.id
  AND n1.id <> n2.id
"""),

("V14-edges-2", 5, "Graph: friends + company + genre",
"""
SELECT MIN(n1.name) AS actor1, MIN(n2.name) AS actor2,
       MIN(t.title) AS movie_title, MIN(cn.name) AS company
FROM name n1, name n2, edges e, cast_info ci1, cast_info ci2,
     title t, movie_info mi, company_name cn, movie_companies mc
WHERE e.src_id = n1.id
  AND e.dst_id = n2.id
  AND e.edge_type = 'friendship'
  AND ci1.person_id = n1.id
  AND ci2.person_id = n2.id
  AND ci1.movie_id = t.id
  AND ci2.movie_id = t.id
  AND mi.info = 'Drama'
  AND mi.info_type_id = 3
  AND mi.movie_id = t.id
  AND cn.id = mc.company_id
  AND mc.movie_id = t.id
  AND n1.id <> n2.id
"""),
]

# ── Benchmark runner ───────────────────────────────────────────────────────

def q_error(est: float, act: float) -> float:
    if est <= 0 and act <= 0: return 1.0
    if act == 0: return float('inf')
    if est == 0: return float('inf')
    return max(est / act, act / est)


def extract_top_estimate(plan: dict) -> float | None:
    """Pull the estimated rows from the top plan node (below any Aggregate)."""
    node = plan
    for _ in range(3):
        if node.get("Node Type", "") in ("Aggregate", "Limit"):
            node = (node.get("Plans") or [{}])[0]
        else:
            break
    return node.get("Plan Rows")


def run_benchmark(conn, queries, timeout_ms=30_000):
    results = []
    cur = conn.cursor()
    cur.execute(f"SET statement_timeout = {timeout_ms}")

    for qid, n_joins, desc, sql in queries:
        try:
            t0 = time.perf_counter()
            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS OFF) {sql}")
            plan_json = cur.fetchone()[0][0]
            wall_ms = (time.perf_counter() - t0) * 1000

            plan = plan_json.get("Plan", {})
            est = extract_top_estimate(plan)

            # Actual count
            cur.execute(f"SELECT COUNT(*) FROM ({sql}) _q")
            actual = cur.fetchone()[0]

            qe = q_error(est, actual) if est is not None else None
            bias = math.log2(est / actual) if est and actual > 0 else None

            results.append({
                "id": qid, "joins": n_joins, "desc": desc,
                "est": est, "actual": actual, "qe": qe,
                "bias": bias, "wall_ms": wall_ms, "error": None,
            })
            status = f"{'inf' if qe == float('inf') else f'{qe:.1f}x'}"
            print(f"  {qid:<14} est={est:>8} act={actual:>8} qe={status:>8} "
                  f"({wall_ms:.0f}ms)", flush=True)

        except Exception as ex:
            conn.rollback()
            cur.execute(f"SET statement_timeout = {timeout_ms}")
            msg = str(ex).split('\n')[0][:80]
            print(f"  {qid:<14} ERROR: {msg}", flush=True)
            results.append({
                "id": qid, "joins": n_joins, "desc": desc,
                "est": None, "actual": None, "qe": None,
                "bias": None, "wall_ms": None, "error": msg,
            })

    cur.close()
    return results


def print_summary(results):
    good = [r for r in results if r["qe"] is not None and r["qe"] != float('inf')]
    if not good:
        print("\nNo successful queries to summarize.")
        return

    qes = sorted(r["qe"] for r in good)
    biases = [r["bias"] for r in good if r["bias"] is not None]

    def pct(lst, p):
        i = int(len(lst) * p / 100)
        return lst[min(i, len(lst)-1)]

    print(f"\n{'='*65}")
    print(f"  JOB Benchmark Summary  ({len(results)} queries, {len(good)} successful)")
    print(f"{'='*65}")
    print(f"  q-error  median={statistics.median(qes):.2f}x  "
          f"p75={pct(qes,75):.2f}x  p90={pct(qes,90):.2f}x  "
          f"max={max(qes):.2f}x  mean={statistics.mean(qes):.2f}x")
    if biases:
        mean_b = statistics.mean(biases)
        over = sum(1 for b in biases if b > 0.1)
        under = sum(1 for b in biases if b < -0.1)
        print(f"  bias     mean log2={mean_b:.3f}  "
              f"over={over} ({100*over//len(biases)}%)  "
              f"under={under} ({100*under//len(biases)}%)")
    print()

    # By join count
    by_joins = {}
    for r in good:
        by_joins.setdefault(r["joins"], []).append(r["qe"])
    print(f"  {'Joins':>5}  {'N':>4}  {'Median qe':>10}  {'Max qe':>10}")
    for j in sorted(by_joins):
        qs = sorted(by_joins[j])
        print(f"  {j:>5}  {len(qs):>4}  {statistics.median(qs):>10.2f}x  {max(qs):>10.2f}x")
    print()

    # Top 10 worst
    print("  Top offenders:")
    worst = sorted(good, key=lambda r: r["qe"], reverse=True)[:10]
    for r in worst:
        b = f"{r['bias']:+.2f}" if r["bias"] else "   ---"
        print(f"    {r['id']:<16} qe={r['qe']:>8.1f}x  est={r['est']:>8}  "
              f"act={r['actual']:>8}  bias={b}  {r['desc']}")
    print(f"{'='*65}\n")


def write_csv(path: str, results: list):
    fn = ["id","joins","desc","est","actual","qe","bias","wall_ms","error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  CSV written: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default="localhost")
    parser.add_argument("--port",   type=int, default=5433)
    parser.add_argument("--dbname", default="imdb_benchmark")
    parser.add_argument("--user",   default="postgres")
    parser.add_argument("--password", default="bench")
    parser.add_argument("--csv",    default="job_results.csv")
    parser.add_argument("--timeout",type=int, default=30000,
                        help="Per-query timeout ms (default 30000)")
    args = parser.parse_args()

    print(f"\nConnecting to {args.dbname} on {args.host}:{args.port}...", flush=True)
    conn = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname,
                            user=args.user, password=args.password)
    conn.autocommit = True

    # Make sure planner uses real statistics
    cur = conn.cursor()
    cur.execute("ANALYZE")
    cur.close()

    print(f"Running {len(JOB_QUERIES)} JOB queries...\n", flush=True)
    print(f"  {'Query':<14} {'est':>8} {'act':>8} {'q-err':>8}  time", flush=True)
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}  ----", flush=True)

    results = run_benchmark(conn, JOB_QUERIES, timeout_ms=args.timeout)
    conn.close()

    print_summary(results)
    write_csv(args.csv, results)


if __name__ == "__main__":
    main()
