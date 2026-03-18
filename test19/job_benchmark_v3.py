#!/usr/bin/env python3
"""
job_benchmark_v3.py — Comprehensive q-error benchmark on v16 synthetic IMDB
===========================================================================
All predicates use ACTUAL values from the v16 dataset (no edges table).
Only counts q-errors at join nodes where actual > 0.

Query categories:
  R*  = Relational multi-table joins (JOB-style)
  E*  = EAV correlation queries (movie_info self-joins)
  P*  = Person-info correlation queries
  S*  = Selectivity spectrum (same structure, varying filter tightness)
  M*  = Multi-path join queries (multiple join paths to same table)
"""
import sys, io, argparse, time, math, csv, json
from dataclasses import dataclass, field
from typing import Optional

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import psycopg2

JOIN_NODE_TYPES = {
    "Hash Join", "Merge Join", "Nested Loop",
    "Hash Anti Join", "Merge Anti Join",
    "Hash Semi Join", "Merge Semi Join",
}

def collect_join_nodes(node, depth=0):
    nodes = []
    nt = node.get("Node Type", "")
    if nt in JOIN_NODE_TYPES or "Join" in nt:
        est = float(node.get("Plan Rows", 1))
        loops = max(int(node.get("Actual Loops", 1)), 1)
        raw_act = float(node.get("Actual Rows", 0)) * loops
        if est <= 0: est = 0.5
        cond = node.get("Hash Cond") or node.get("Merge Cond") or node.get("Join Filter") or ""
        nodes.append({
            "depth": depth, "type": nt, "est": est, "act": raw_act,
            "cond": str(cond)[:60]
        })
    for sub in node.get("Plans", []):
        nodes.extend(collect_join_nodes(sub, depth + 1))
    return nodes


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY SUITE — all predicates verified against v16 data
# ═══════════════════════════════════════════════════════════════════════════════

QUERIES = [

# ────────── R: RELATIONAL MULTI-TABLE (2-5 joins) ──────────────────────────

("R1", 2, "cast(actor) + title (year>2000)",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, title t
   WHERE ci.role_id = 1 AND t.production_year > 2000
     AND n.id = ci.person_id AND t.id = ci.movie_id"""),

("R2", 2, "cast(actress) + company(us)",
"""SELECT MIN(n.name), MIN(cn.name)
   FROM name n, cast_info ci, title t, company_name cn, movie_companies mc
   WHERE ci.role_id = 2 AND cn.country_code = 'us'
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

("R3", 3, "writer + Drama + title(2010+)",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, role_type rt, title t, movie_info mi
   WHERE rt.role = 'writer' AND mi.info = 'Drama'
     AND mi.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND t.production_year >= 2010
     AND n.id = ci.person_id AND rt.id = ci.role_id
     AND t.id = ci.movie_id AND mi.movie_id = t.id"""),

("R4", 3, "producer + Horror + company(in)",
"""SELECT MIN(n.name), MIN(t.title), MIN(cn.name)
   FROM name n, cast_info ci, role_type rt, title t, movie_info mi,
        company_name cn, movie_companies mc
   WHERE rt.role = 'producer' AND mi.info = 'Horror'
     AND mi.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND cn.country_code = 'in'
     AND n.id = ci.person_id AND rt.id = ci.role_id
     AND t.id = ci.movie_id AND mi.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

("R5", 4, "actor + keyword + company + title(1990-2005)",
"""SELECT MIN(n.name), MIN(t.title), MIN(k.keyword)
   FROM name n, cast_info ci, title t, keyword k, movie_keyword mk,
        company_name cn, movie_companies mc
   WHERE ci.role_id = 1
     AND t.production_year BETWEEN 1990 AND 2005
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND k.id = mk.keyword_id AND mk.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

("R6", 4, "actress + aka_title + keyword + Comedy",
"""SELECT MIN(n.name), MIN(at2.title), MIN(k.keyword)
   FROM name n, cast_info ci, title t, aka_title at2,
        keyword k, movie_keyword mk, movie_info mi
   WHERE ci.role_id = 2 AND mi.info = 'Comedy'
     AND mi.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND at2.movie_id = t.id
     AND k.id = mk.keyword_id AND mk.movie_id = t.id
     AND mi.movie_id = t.id"""),

("R7", 5, "director + Action + USA + company + keyword",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, role_type rt, title t,
        movie_info mi_g, movie_info mi_c,
        company_name cn, movie_companies mc,
        keyword k, movie_keyword mk
   WHERE rt.role = 'director' AND mi_g.info = 'Action'
     AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'USA'
     AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND cn.country_code = 'us'
     AND n.id = ci.person_id AND rt.id = ci.role_id
     AND t.id = ci.movie_id AND mi_g.movie_id = t.id
     AND mi_c.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id
     AND k.id = mk.keyword_id AND mk.movie_id = t.id"""),

("R8", 5, "actor + aka_name + person_info(bio) + Thriller + company",
"""SELECT MIN(n.name), MIN(an.name), MIN(t.title)
   FROM name n, aka_name an, cast_info ci, title t,
        person_info pi, movie_info mi,
        company_name cn, movie_companies mc
   WHERE ci.role_id = 1 AND mi.info = 'Thriller'
     AND mi.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND pi.info_type_id = (SELECT id FROM info_type WHERE info='biography')
     AND n.id = ci.person_id AND an.person_id = n.id
     AND t.id = ci.movie_id AND mi.movie_id = t.id
     AND pi.person_id = n.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

# ────────── E: EAV CORRELATION (movie_info self-joins) ──────────────────────
# These exploit the EAV anti-pattern: Postgres assumes genre and country are
# independent, but they're correlated (e.g. Bollywood = India + Drama/Action)

("E1", 2, "EAV: genre(Drama) + country(USA) — common combo",
"""SELECT MIN(t.title)
   FROM title t, movie_info mi_g, movie_info mi_c
   WHERE mi_g.info = 'Drama' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'USA' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id"""),

("E2", 2, "EAV: genre(Horror) + country(Japan) — rare combo",
"""SELECT MIN(t.title)
   FROM title t, movie_info mi_g, movie_info mi_c
   WHERE mi_g.info = 'Horror' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'Japan' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id"""),

("E3", 3, "EAV: genre + country + cert + company",
"""SELECT MIN(t.title), MIN(cn.name)
   FROM title t, movie_info mi_g, movie_info mi_c, movie_info mi_cert,
        company_name cn, movie_companies mc
   WHERE mi_g.info = 'Action' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'USA' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND mi_cert.info = 'R' AND mi_cert.info_type_id = (SELECT id FROM info_type WHERE info='certificates')
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id AND mi_cert.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

("E4", 3, "EAV: genre + country + language",
"""SELECT MIN(t.title)
   FROM title t, movie_info mi_g, movie_info mi_c, movie_info mi_l
   WHERE mi_g.info = 'Drama' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'India' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND mi_l.info = 'Hindi' AND mi_l.info_type_id = (SELECT id FROM info_type WHERE info='languages')
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id AND mi_l.movie_id = t.id"""),

("E5", 3, "EAV: genre(Sci-Fi) + country(UK) + language(English)",
"""SELECT MIN(t.title)
   FROM title t, movie_info mi_g, movie_info mi_c, movie_info mi_l
   WHERE mi_g.info = 'Sci-Fi' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'UK' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND mi_l.info = 'English' AND mi_l.info_type_id = (SELECT id FROM info_type WHERE info='languages')
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id AND mi_l.movie_id = t.id"""),

("E6", 4, "EAV: genre + country + cert + color (4-way EAV)",
"""SELECT MIN(t.title)
   FROM title t, movie_info mi_g, movie_info mi_c, movie_info mi_cert, movie_info mi_col
   WHERE mi_g.info IN ('Action','Thriller') AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info IN ('USA','UK') AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND mi_cert.info = 'R' AND mi_cert.info_type_id = (SELECT id FROM info_type WHERE info='certificates')
     AND mi_col.info = 'Color' AND mi_col.info_type_id = (SELECT id FROM info_type WHERE info='color info')
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id
     AND mi_cert.movie_id = t.id AND mi_col.movie_id = t.id"""),

("E7", 5, "EAV: genre + country + cast + company (EAV + relational)",
"""SELECT MIN(n.name), MIN(t.title)
   FROM title t, name n, cast_info ci,
        movie_info mi_g, movie_info mi_c,
        company_name cn, movie_companies mc
   WHERE mi_g.info IN ('Comedy','Romance') AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info IN ('France','Germany') AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND ci.role_id IN (1,2)
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

# ────────── P: PERSON-INFO CORRELATION ──────────────────────────────────────
# person_info EAV joins: nationality + biography + cast

("P1", 3, "person nationality + actor + Drama",
"""SELECT MIN(n.name), MIN(pi.info), MIN(t.title)
   FROM name n, cast_info ci, person_info pi, title t, movie_info mi,
        info_type it_pi
   WHERE it_pi.info = 'nationality' AND ci.role_id = 1
     AND mi.info = 'Drama' AND mi.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND n.id = ci.person_id AND pi.person_id = n.id
     AND it_pi.id = pi.info_type_id
     AND t.id = ci.movie_id AND mi.movie_id = t.id"""),

("P2", 4, "person nationality + birth_date + cast + company",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, person_info pi_nat, person_info pi_birth,
        title t, company_name cn, movie_companies mc
   WHERE pi_nat.info_type_id = (SELECT id FROM info_type WHERE info='nationality')
     AND pi_birth.info_type_id = (SELECT id FROM info_type WHERE info='birth date')
     AND ci.role_id IN (1,2)
     AND n.id = ci.person_id AND pi_nat.person_id = n.id AND pi_birth.person_id = n.id
     AND t.id = ci.movie_id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

("P3", 5, "person bio + nationality + cast + genre + country",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, person_info pi_nat, person_info pi_bio,
        title t, movie_info mi_g, movie_info mi_c
   WHERE pi_nat.info_type_id = (SELECT id FROM info_type WHERE info='nationality')
     AND pi_bio.info_type_id = (SELECT id FROM info_type WHERE info='biography')
     AND mi_g.info = 'Action' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'USA' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND ci.role_id = 1
     AND n.id = ci.person_id AND pi_nat.person_id = n.id AND pi_bio.person_id = n.id
     AND t.id = ci.movie_id AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id"""),

# ────────── S: SELECTIVITY SPECTRUM (same structure, varying filters) ───────
# Tight filters → few rows → Postgres overestimates → q-error
# Broad filters → many rows → Postgres underestimates → q-error

("S1", 3, "Tight: Action + Japan + 2015-2020",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, title t, movie_info mi_g, movie_info mi_c
   WHERE mi_g.info = 'Action' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'Japan' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND t.production_year BETWEEN 2015 AND 2020 AND ci.role_id = 1
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id"""),

("S2", 3, "Medium: Drama + USA + 2000-2020",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, title t, movie_info mi_g, movie_info mi_c
   WHERE mi_g.info = 'Drama' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'USA' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND t.production_year BETWEEN 2000 AND 2020 AND ci.role_id IN (1,2)
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id"""),

("S3", 3, "Broad: any genre IN(5) + any country IN(3) + all years",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, title t, movie_info mi_g, movie_info mi_c
   WHERE mi_g.info IN ('Drama','Action','Comedy','Horror','Thriller')
     AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info IN ('USA','UK','India')
     AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id"""),

# ────────── M: MULTI-PATH JOINS (multiple FK paths converge) ───────────────

("M1", 4, "Multi-path: cast + keyword + company all → same title",
"""SELECT MIN(n.name), MIN(t.title), MIN(k.keyword), MIN(cn.name)
   FROM name n, cast_info ci, title t,
        keyword k, movie_keyword mk,
        company_name cn, movie_companies mc
   WHERE ci.role_id = 1 AND t.production_year > 2010
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND k.id = mk.keyword_id AND mk.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

("M2", 5, "Multi-path: cast + link + aka_title + keyword → title",
"""SELECT MIN(n.name), MIN(t.title), MIN(t2.title)
   FROM name n, cast_info ci, title t,
        movie_link ml, title t2,
        aka_title at2, keyword k, movie_keyword mk
   WHERE ci.role_id IN (1,2)
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND t.id = ml.movie_id AND t2.id = ml.linked_movie_id
     AND at2.movie_id = t.id
     AND k.id = mk.keyword_id AND mk.movie_id = t.id"""),

("M3", 6, "Multi-path: cast+aka_name+person_info+genre+cert+company",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, aka_name an, person_info pi, title t,
        movie_info mi_g, movie_info mi_cert,
        company_name cn, movie_companies mc
   WHERE ci.role_id = 1
     AND mi_g.info IN ('Drama','Thriller') AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_cert.info = 'R' AND mi_cert.info_type_id = (SELECT id FROM info_type WHERE info='certificates')
     AND pi.info_type_id = (SELECT id FROM info_type WHERE info='nationality')
     AND n.id = ci.person_id AND an.person_id = n.id AND pi.person_id = n.id
     AND t.id = ci.movie_id AND mi_g.movie_id = t.id AND mi_cert.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

# ────────── X: CROSS-DOMAIN (unusual table combos) ─────────────────────────

("X1", 3, "complete_cast + genre + company",
"""SELECT MIN(t.title), MIN(cn.name)
   FROM title t, complete_cast cc, movie_info mi,
        company_name cn, movie_companies mc
   WHERE mi.info = 'Sci-Fi' AND mi.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND t.id = cc.movie_id AND mi.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

("X2", 4, "movie_info_idx(votes) + genre + cast + company",
"""SELECT MIN(t.title), MIN(n.name)
   FROM title t, movie_info_idx mi_idx, movie_info mi,
        cast_info ci, name n, company_name cn, movie_companies mc
   WHERE mi_idx.info_type_id = (SELECT id FROM info_type WHERE info='votes')
     AND mi.info = 'Action' AND mi.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND ci.role_id = 1
     AND t.id = mi_idx.movie_id AND mi.movie_id = t.id
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND cn.id = mc.company_id AND mc.movie_id = t.id"""),

("X3", 4, "aka_title + aka_name + cast + genre",
"""SELECT MIN(n.name), MIN(an.name), MIN(at2.title), MIN(t.title)
   FROM name n, aka_name an, cast_info ci, title t, aka_title at2, movie_info mi
   WHERE mi.info = 'Comedy' AND mi.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND ci.role_id IN (1,2)
     AND n.id = ci.person_id AND an.person_id = n.id
     AND t.id = ci.movie_id AND at2.movie_id = t.id
     AND mi.movie_id = t.id"""),

("X4", 5, "8-way: all core tables (actor+genre+country+kw+company+link+pinfo)",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, title t,
        movie_info mi_g, movie_info mi_c,
        keyword k, movie_keyword mk,
        company_name cn, movie_companies mc,
        person_info pi, movie_link ml
   WHERE ci.role_id = 1
     AND mi_g.info = 'Drama' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'USA' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND pi.info_type_id = (SELECT id FROM info_type WHERE info='nationality')
     AND cn.country_code = 'us'
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id
     AND k.id = mk.keyword_id AND mk.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id
     AND pi.person_id = n.id AND ml.movie_id = t.id"""),

("X5", 5, "8-way mega: actress+Horror+Japan+keyword+company+pinfo+link+aka",
"""SELECT MIN(n.name), MIN(t.title)
   FROM name n, cast_info ci, title t,
        movie_info mi_g, movie_info mi_c,
        keyword k, movie_keyword mk,
        company_name cn, movie_companies mc,
        person_info pi, aka_title at2
   WHERE ci.role_id = 2
     AND mi_g.info = 'Horror' AND mi_g.info_type_id = (SELECT id FROM info_type WHERE info='genres')
     AND mi_c.info = 'Japan' AND mi_c.info_type_id = (SELECT id FROM info_type WHERE info='countries')
     AND pi.info_type_id = (SELECT id FROM info_type WHERE info='nationality')
     AND n.id = ci.person_id AND t.id = ci.movie_id
     AND mi_g.movie_id = t.id AND mi_c.movie_id = t.id
     AND k.id = mk.keyword_id AND mk.movie_id = t.id
     AND cn.id = mc.company_id AND mc.movie_id = t.id
     AND pi.person_id = n.id AND at2.movie_id = t.id"""),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--db", default="imdb_v16")
    parser.add_argument("--csv", default="test16/benchmark_v3_results.csv")
    args = parser.parse_args()

    conn = psycopg2.connect(host='localhost', port=args.port, dbname=args.db,
                            user='postgres', password='bench')
    conn.autocommit = True
    cur = conn.cursor()

    print(f"\nRunning {len(QUERIES)} queries on {args.db}...")
    print(f"\n  {'Query':<6} {'J':>2} {'Nodes':>5} {'Worst(real)':>11} {'Est':>8} {'Act':>10} {'Bias':>6} {'ms':>6}  Description")
    print("  " + "-" * 95)

    csv_rows = []
    for qid, n_joins, desc, sql in QUERIES:
        try:
            t0 = time.time()
            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
            plan_json = cur.fetchone()[0]
            elapsed = (time.time() - t0) * 1000
            plan = plan_json[0]["Plan"]
            nodes = collect_join_nodes(plan)

            # Only real nodes (actual > 0)
            real_nodes = [n for n in nodes if n["act"] > 0]

            if not real_nodes:
                print(f"  {qid:<6} {n_joins:>2} {len(nodes):>5} {'ZERO-ACT':>11} {'':>8} {'':>10} {'':>6} {elapsed:>6.0f}  {desc[:50]}")
                csv_rows.append({"qid": qid, "joins": n_joins, "nodes": len(nodes),
                                 "worst_real_qe": 0, "est": 0, "act": 0, "bias": 0,
                                 "wall_ms": elapsed, "desc": desc, "verdict": "ZERO"})
                continue

            # Compute q-error for real nodes
            for n in real_nodes:
                e, a = n["est"], n["act"]
                n["qe"] = max(e / max(a, 0.5), a / max(e, 0.5))
                n["bias"] = math.log2(e / max(a, 0.5))

            worst = max(real_nodes, key=lambda x: x["qe"])
            e, a, qe, bias = worst["est"], worst["act"], worst["qe"], worst["bias"]
            bias_str = f"{bias:+.1f}" if abs(bias) < 20 else f"{bias:+.0f}"

            verdict = "mild" if qe < 5 else "REAL" if qe < 50 else "STRONG" if qe < 100 else "MASSIVE"

            print(f"  {qid:<6} {n_joins:>2} {len(real_nodes):>5} {qe:>9.1f}x  {e:>8.0f} {a:>10.0f} {bias_str:>6} {elapsed:>6.0f}  {desc[:50]}")

            csv_rows.append({"qid": qid, "joins": n_joins, "nodes": len(real_nodes),
                             "worst_real_qe": round(qe, 2), "est": e, "act": a,
                             "bias": round(bias, 2), "wall_ms": round(elapsed),
                             "desc": desc, "verdict": verdict})

        except Exception as ex:
            print(f"  {qid:<6} ERROR: {str(ex)[:70]}")

    # Summary
    real = [r for r in csv_rows if r["worst_real_qe"] > 0 and r["verdict"] != "ZERO"]
    strong = [r for r in real if r["worst_real_qe"] >= 5]
    massive = [r for r in real if r["worst_real_qe"] >= 50]

    print(f"\n{'='*80}")
    print("SUMMARY (only counting join nodes with actual > 0)")
    print(f"{'='*80}")
    print(f"  Total queries:           {len(QUERIES)}")
    print(f"  Returned rows (non-zero): {len(real)}")
    print(f"  Real q-error > 5x:       {len(strong)}")
    print(f"  Real q-error > 50x:      {len(massive)}")
    if real:
        qes = sorted([r["worst_real_qe"] for r in real], reverse=True)
        print(f"  Max real q-error:        {qes[0]:.1f}x")
        print(f"  Median real q-error:     {qes[len(qes)//2]:.1f}x")
        print(f"  p90 real q-error:        {qes[max(0,len(qes)//10)]:.1f}x")

    print(f"\n  Top 10 worst REAL q-errors:")
    for r in sorted(real, key=lambda x: -x["worst_real_qe"])[:10]:
        bias_dir = "UNDER" if r["bias"] < 0 else "OVER"
        print(f"    {r['qid']:<6} {r['worst_real_qe']:>8.1f}x  est={r['est']:>8.0f}  act={r['act']:>10.0f}  {bias_dir}  {r['desc'][:50]}")

    # Save CSV
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["qid","joins","nodes","worst_real_qe","est","act","bias","wall_ms","desc","verdict"])
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n  CSV -> {args.csv}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
