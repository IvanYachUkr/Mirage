#!/usr/bin/env python3
"""
job_benchmark_v2.py
===================
Methodologically correct JOB-style cardinality estimation benchmark
on the imdb_benchmark (IMDB-compatible) schema.

CORRECT METHODOLOGY
-------------------
JOB q-error is measured at EVERY INTERNAL JOIN NODE in the EXPLAIN ANALYZE
plan tree — NOT at the query result level. The MIN() aggregates in JOB
queries always return 1 row; that number is irrelevant.

For each query:
  1. Run EXPLAIN (ANALYZE, FORMAT JSON) to get plan + actual row counts
  2. Walk the plan tree recursively; collect every Hash/Merge/Nested-Loop join
  3. At each join node:
       actual_rows = node["Actual Rows"] * node["Actual Loops"]
       est_rows    = node["Plan Rows"]
       q_error     = max(est/act, act/est)   (if both > 0)
  4. Report: worst_qe, deepest_qe, mean_bias, std_bias

PRE-VERIFIED PREDICATES
------------------------
These are grounded in actual data from imdb_benchmark:
- movie_info.info_type_id 3=genres  8=countries  1=runtimes  16=certificates
  4=languages  7=color info  99=taglines
- Genres: Drama(516) Action(438) Comedy(431) Thriller(319) Horror(311)
- Countries: USA(968) UK(290) India(274) Japan(216) South Korea(215)
- cast_info.role_id: 1=actor(13562) 2=actress(12288)  [no directors here]
- company_name.country_code: us(61) in(42) cn(42) br(42) jp(42)
- movie_link.link_type_id: 16=unknown(369) 10=spin-off(62)
- person_info: birth_date(21) birth_notes(22) death_date(26,only 19) height(34)
  nationality(500) biography(501) — all present for every person
- title.production_year: 1970-2027
- info_type.id 3=genres 8=countries 1=runtimes 500=nationality 501=biography

Usage:
    python job_benchmark_v2.py [--csv FILE] [--port PORT]
"""

import sys, io, argparse, time, math, csv, json, statistics
from dataclasses import dataclass, field
from typing import Optional

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import psycopg2

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class JoinNode:
    depth: int
    node_type: str
    est_rows: float
    act_rows: float      # actual_rows * loops
    loops: int
    qe: float            # q-error
    bias: float          # log2(est/act)  positive=over, negative=under
    condition: str = ""


@dataclass
class QueryResult:
    qid: str
    n_joins: int
    description: str
    joins: list[JoinNode] = field(default_factory=list)
    worst_qe: float = 1.0
    deepest_qe: float = 1.0   # q-error at deepest join
    mean_bias: float = 0.0
    std_bias: float = 0.0
    n_join_nodes: int = 0
    wall_ms: float = 0.0
    error: Optional[str] = None


# ─── Plan tree walker ─────────────────────────────────────────────────────────

JOIN_NODE_TYPES = {
    "Hash Join", "Merge Join", "Nested Loop",
    "Hash Anti Join", "Merge Anti Join",
    "Hash Semi Join", "Merge Semi Join",
}


def collect_join_nodes(node: dict, depth: int = 0) -> list[JoinNode]:
    """Recursively collect all join nodes from an EXPLAIN ANALYZE JSON plan."""
    nodes = []
    node_type = node.get("Node Type", "")

    if node_type in JOIN_NODE_TYPES or "Join" in node_type:
        est   = float(node.get("Plan Rows", 1))
        loops = max(int(node.get("Actual Loops", 1)), 1)
        act   = float(node.get("Actual Rows", 0)) * loops
        act   = max(act, 0.5)   # prevent division by zero; treat 0 rows as 0.5

        if est <= 0:
            est = 0.5

        qe   = max(est / act, act / est)
        bias = math.log2(est / act)

        cond = (node.get("Hash Cond") or node.get("Merge Cond") or
                node.get("Join Filter") or node.get("Index Cond") or "")

        nodes.append(JoinNode(
            depth=depth, node_type=node_type,
            est_rows=est, act_rows=act, loops=loops,
            qe=qe, bias=bias, condition=str(cond)[:80]
        ))

    for sub in node.get("Plans", []):
        nodes.extend(collect_join_nodes(sub, depth + 1))

    return nodes


# ─── JOB query suite ─────────────────────────────────────────────────────────
# Format: (id, n_joins, description, sql)
#
# All queries preserve the original JOB MIN() aggregate structure — this is
# intentional. Q-error is measured at internal join nodes, not at result rows.
#
# Queries are ordered by join count (2 → 9) to reveal how error compounds.

JOB_QUERIES = [

# ══════════ 2-JOIN QUERIES ═══════════════════════════════════════════════════

("J1a", 2, "company_type + movie_companies + title",
"""SELECT MIN(mc.note)        AS production_note,
          MIN(t.title)         AS movie_title,
          MIN(t.production_year) AS movie_year
   FROM   company_type ct,
          movie_companies mc,
          title t
   WHERE  ct.kind = 'production companies'
     AND  t.production_year > 2000
     AND  ct.id   = mc.company_type_id
     AND  t.id    = mc.movie_id"""),

("J1b", 2, "movie_info (genre) + title",
"""SELECT MIN(t.title)  AS movie_title,
          MIN(mi.info)  AS genre
   FROM   title t,
          movie_info mi
   WHERE  mi.info_type_id = 3
     AND  mi.info IN ('Drama', 'Action')
     AND  t.production_year > 1990
     AND  t.id = mi.movie_id"""),

("J1c", 2, "name + cast_info (actor)",
"""SELECT MIN(n.name)  AS actor_name,
          MIN(t.title) AS movie_title
   FROM   name n,
          cast_info ci,
          title t
   WHERE  ci.role_id = 1
     AND  n.id    = ci.person_id
     AND  t.id    = ci.movie_id"""),

("J1d", 2, "keyword + movie_keyword",
"""SELECT MIN(t.title)  AS movie_title,
          MIN(k.keyword) AS kw
   FROM   title t,
          keyword k,
          movie_keyword mk
   WHERE  k.id  = mk.keyword_id
     AND  t.id  = mk.movie_id
     AND  t.production_year BETWEEN 2000 AND 2020"""),

("J1e", 2, "company_name (us) + movie_companies",
"""SELECT MIN(cn.name)  AS company_name,
          MIN(t.title)  AS movie_title
   FROM   company_name cn,
          movie_companies mc,
          title t
   WHERE  cn.country_code = 'us'
     AND  cn.id  = mc.company_id
     AND  t.id   = mc.movie_id"""),

# ══════════ 3-JOIN QUERIES ════════════════════════════════════════════════════

("J2a", 3, "genre + keyword + title",
"""SELECT MIN(t.title)  AS movie_title,
          MIN(k.keyword) AS keyword
   FROM   title t,
          movie_info mi,
          keyword k,
          movie_keyword mk
   WHERE  mi.info = 'Action'
     AND  mi.info_type_id = 3
     AND  k.id   = mk.keyword_id
     AND  t.id   = mk.movie_id
     AND  mi.movie_id = t.id"""),

("J2b", 3, "country + company + title",
"""SELECT MIN(t.title)  AS movie_title,
          MIN(cn.name)  AS company
   FROM   title t,
          movie_info mi,
          company_name cn,
          movie_companies mc
   WHERE  mi.info = 'USA'
     AND  mi.info_type_id = 8
     AND  cn.id   = mc.company_id
     AND  t.id    = mc.movie_id
     AND  mi.movie_id = t.id"""),

("J2c", 3, "actress + genre + title",
"""SELECT MIN(n.name)  AS actress_name,
          MIN(t.title) AS movie_title
   FROM   name n,
          cast_info ci,
          title t,
          movie_info mi
   WHERE  ci.role_id = 2
     AND  mi.info = 'Drama'
     AND  mi.info_type_id = 3
     AND  n.id   = ci.person_id
     AND  t.id   = ci.movie_id
     AND  mi.movie_id = t.id"""),

("J2d", 3, "role_type + cast + title",
"""SELECT MIN(n.name)  AS person_name,
          MIN(t.title) AS movie_title
   FROM   name n,
          cast_info ci,
          role_type rt,
          title t
   WHERE  rt.role = 'actress'
     AND  t.production_year BETWEEN 1995 AND 2015
     AND  n.id   = ci.person_id
     AND  rt.id  = ci.role_id
     AND  t.id   = ci.movie_id"""),

("J2e", 3, "certificate + cast + title",
"""SELECT MIN(t.title)  AS movie_title,
          MIN(mi.info)  AS cert
   FROM   title t,
          movie_info mi,
          cast_info ci,
          name n
   WHERE  mi.info = 'R'
     AND  mi.info_type_id = 16
     AND  ci.role_id = 1
     AND  t.id   = mi.movie_id
     AND  t.id   = ci.movie_id
     AND  n.id   = ci.person_id"""),

# ══════════ 4-JOIN QUERIES ════════════════════════════════════════════════════

("J3a", 4, "company_type + genre + keyword + title",
"""SELECT MIN(cn.name)  AS company_name,
          MIN(t.title)  AS movie_title
   FROM   company_type ct,
          company_name cn,
          movie_companies mc,
          movie_info mi,
          title t
   WHERE  ct.kind = 'production companies'
     AND  mi.info = 'Horror'
     AND  mi.info_type_id = 3
     AND  ct.id   = mc.company_type_id
     AND  cn.id   = mc.company_id
     AND  t.id    = mc.movie_id
     AND  mi.movie_id = t.id"""),

("J3b", 4, "us-company + film year + keyword + cast",
"""SELECT MIN(n.name)  AS person_name,
          MIN(t.title) AS movie_title
   FROM   name n,
          cast_info ci,
          title t,
          company_name cn,
          movie_companies mc
   WHERE  cn.country_code = 'us'
     AND  t.production_year > 2010
     AND  ci.role_id = 1
     AND  n.id   = ci.person_id
     AND  t.id   = ci.movie_id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id"""),

("J3c", 4, "aka_title + genre + cast + title",
"""SELECT MIN(t.title)   AS movie_title,
          MIN(at2.title) AS alt_title,
          MIN(n.name)    AS actor_name
   FROM   title t,
          aka_title at2,
          cast_info ci,
          name n,
          movie_info mi
   WHERE  mi.info = 'Drama'
     AND  mi.info_type_id = 3
     AND  ci.role_id = 1
     AND  at2.movie_id = t.id
     AND  n.id   = ci.person_id
     AND  ci.movie_id = t.id
     AND  mi.movie_id = t.id"""),

("J3d", 4, "color + country + company + title",
"""SELECT MIN(t.title)   AS movie_title,
          MIN(cn.name)   AS company
   FROM   title t,
          movie_info mi_color,
          movie_info mi_country,
          company_name cn,
          movie_companies mc
   WHERE  mi_color.info = 'Color'
     AND  mi_color.info_type_id = 7
     AND  mi_country.info = 'USA'
     AND  mi_country.info_type_id = 8
     AND  cn.id   = mc.company_id
     AND  t.id    = mc.movie_id
     AND  mi_color.movie_id = t.id
     AND  mi_country.movie_id = t.id"""),

# ══════════ 5-JOIN QUERIES ════════════════════════════════════════════════════

("J4a", 5, "5-way: cast+co+genre+keyword+title",
"""SELECT MIN(n.name)   AS person_name,
          MIN(t.title)  AS movie_title,
          MIN(cn.name)  AS company_name
   FROM   title t,
          name n,
          cast_info ci,
          company_name cn,
          movie_companies mc,
          movie_info mi
   WHERE  t.production_year BETWEEN 2000 AND 2015
     AND  mi.info = 'Action'
     AND  mi.info_type_id = 3
     AND  ci.role_id = 1
     AND  mc.company_type_id = 1
     AND  n.id   = ci.person_id
     AND  t.id   = ci.movie_id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id
     AND  mi.movie_id = t.id"""),

("J4b", 5, "5-way: aka_name + person_info + cast + genre",
"""SELECT MIN(n.name)   AS person_name,
          MIN(an.name)  AS aka_name,
          MIN(t.title)  AS movie_title
   FROM   name n,
          aka_name an,
          cast_info ci,
          title t,
          movie_info mi
   WHERE  mi.info = 'Drama'
     AND  mi.info_type_id = 3
     AND  ci.role_id IN (1, 2)
     AND  an.person_id = n.id
     AND  n.id   = ci.person_id
     AND  t.id   = ci.movie_id
     AND  mi.movie_id = t.id"""),

("J4c", 5, "5-way: keyword + company_type(prod) + company + movie + year",
"""SELECT MIN(t.title)  AS movie_title,
          MIN(k.keyword) AS keyword,
          MIN(cn.name)  AS company
   FROM   title t,
          keyword k,
          movie_keyword mk,
          company_type ct,
          company_name cn,
          movie_companies mc
   WHERE  ct.kind = 'production companies'
     AND  t.production_year >= 2005
     AND  k.id   = mk.keyword_id
     AND  mk.movie_id = t.id
     AND  ct.id   = mc.company_type_id
     AND  cn.id   = mc.company_id
     AND  mc.movie_id = t.id"""),

("J4d", 5, "5-way: role_type + country + runtime + cast",
"""SELECT MIN(n.name)   AS person_name,
          MIN(t.title)  AS movie_title
   FROM   title t,
          name n,
          cast_info ci,
          role_type rt,
          movie_info mi_rt,
          movie_info mi_ct
   WHERE  rt.role = 'actor'
     AND  mi_ct.info = 'USA'
     AND  mi_ct.info_type_id = 8
     AND  mi_rt.info_type_id = 1
     AND  n.id   = ci.person_id
     AND  rt.id  = ci.role_id
     AND  t.id   = ci.movie_id
     AND  mi_ct.movie_id = t.id
     AND  mi_rt.movie_id = t.id"""),

# ══════════ 6-JOIN QUERIES ════════════════════════════════════════════════════

("J5a", 6, "6-way: cast+co+kw+genre+link+title",
"""SELECT MIN(t.title)   AS movie_title,
          MIN(t2.title)  AS linked_title
   FROM   title t,
          title t2,
          movie_link ml,
          movie_info mi,
          company_name cn,
          movie_companies mc,
          keyword k,
          movie_keyword mk
   WHERE  mi.info = 'Action'
     AND  mi.info_type_id = 3
     AND  cn.country_code = 'us'
     AND  t.id   = ml.movie_id
     AND  t2.id  = ml.linked_movie_id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id
     AND  mi.movie_id = t.id
     AND  k.id   = mk.keyword_id
     AND  mk.movie_id = t.id"""),

("J5b", 6, "6-way: person_info (nationality via info_type join) + cast + genre",
"""SELECT MIN(n.name)   AS person_name,
          MIN(pi.info)  AS nationality,
          MIN(t.title)  AS movie_title
   FROM   name n,
          cast_info ci,
          person_info pi,
          info_type it,
          title t,
          movie_info mi
   WHERE  it.info = 'nationality'
     AND  mi.info = 'Drama'
     AND  mi.info_type_id = 3
     AND  n.id   = ci.person_id
     AND  pi.person_id = n.id
     AND  it.id  = pi.info_type_id
     AND  t.id   = ci.movie_id
     AND  mi.movie_id = t.id"""),

# height maps to info_type_id=34 which we don't have in info_type table
# use nationality instead (id=500)
("J5c", 6, "6-way: nationality + cast + company + genre",
"""SELECT MIN(n.name)   AS person_name,
          MIN(pi.info)  AS nationality,
          MIN(t.title)  AS movie_title,
          MIN(cn.name)  AS company
   FROM   name n,
          cast_info ci,
          person_info pi,
          title t,
          movie_info mi,
          company_name cn,
          movie_companies mc
   WHERE  pi.info_type_id = 500
     AND  mi.info = 'Drama'
     AND  mi.info_type_id = 3
     AND  n.id   = ci.person_id
     AND  pi.person_id = n.id
     AND  t.id   = ci.movie_id
     AND  mi.movie_id = t.id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id"""),

("J5d", 6, "6-way: aka_title + company_type + genre + kw",
"""SELECT MIN(t.title)   AS movie_title,
          MIN(at2.title) AS alt_title,
          MIN(cn.name)   AS company
   FROM   title t,
          aka_title at2,
          company_type ct,
          company_name cn,
          movie_companies mc,
          movie_info mi,
          keyword k,
          movie_keyword mk
   WHERE  ct.kind = 'production companies'
     AND  mi.info IN ('Drama', 'Thriller')
     AND  mi.info_type_id = 3
     AND  t.production_year BETWEEN 2000 AND 2020
     AND  at2.movie_id = t.id
     AND  ct.id   = mc.company_type_id
     AND  cn.id   = mc.company_id
     AND  mc.movie_id = t.id
     AND  mi.movie_id = t.id
     AND  k.id   = mk.keyword_id
     AND  mk.movie_id = t.id"""),

# ══════════ 7-JOIN QUERIES ════════════════════════════════════════════════════

("J6a", 7, "7-way: cast+co+kw+genre+person_info+link",
"""SELECT MIN(n.name)   AS person_name,
          MIN(t.title)  AS movie_title,
          MIN(k.keyword) AS keyword
   FROM   title t,
          name n,
          cast_info ci,
          movie_info mi,
          keyword k,
          movie_keyword mk,
          company_name cn,
          movie_companies mc,
          person_info pi
   WHERE  mi.info IN ('Action', 'Thriller')
     AND  mi.info_type_id = 3
     AND  ci.role_id = 1
     AND  pi.info_type_id = 500
     AND  cn.country_code = 'us'
     AND  n.id   = ci.person_id
     AND  t.id   = ci.movie_id
     AND  mi.movie_id = t.id
     AND  k.id   = mk.keyword_id
     AND  mk.movie_id = t.id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id
     AND  pi.person_id = n.id"""),

("J6b", 7, "7-way: role_type + company + genre + keyword + year",
"""SELECT MIN(n.name)   AS person_name,
          MIN(t.title)  AS movie_title,
          MIN(cn.name)  AS company
   FROM   title t,
          name n,
          cast_info ci,
          role_type rt,
          company_name cn,
          movie_companies mc,
          movie_info mi,
          keyword k,
          movie_keyword mk
   WHERE  rt.role IN ('actor', 'actress')
     AND  mi.info IN ('Comedy', 'Drama')
     AND  mi.info_type_id = 3
     AND  t.production_year > 2000
     AND  n.id   = ci.person_id
     AND  rt.id  = ci.role_id
     AND  t.id   = ci.movie_id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id
     AND  mi.movie_id = t.id
     AND  k.id   = mk.keyword_id
     AND  mk.movie_id = t.id"""),

# ══════════ 8-JOIN QUERIES ════════════════════════════════════════════════════

("J7a", 8, "8-way: all core JOB tables",
"""SELECT MIN(n.name)   AS person_name,
          MIN(t.title)  AS movie_title,
          MIN(k.keyword) AS keyword,
          MIN(cn.name)  AS company_name
   FROM   title t,
          name n,
          cast_info ci,
          role_type rt,
          movie_info mi,
          keyword k,
          movie_keyword mk,
          company_name cn,
          movie_companies mc,
          person_info pi
   WHERE  rt.role = 'actor'
     AND  mi.info = 'Drama'
     AND  mi.info_type_id = 3
     AND  pi.info_type_id = 500
     AND  cn.country_code = 'us'
     AND  n.id   = ci.person_id
     AND  rt.id  = ci.role_id
     AND  t.id   = ci.movie_id
     AND  mi.movie_id = t.id
     AND  k.id   = mk.keyword_id
     AND  mk.movie_id = t.id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id
     AND  pi.person_id = n.id"""),

("J7b", 8, "8-way: actress + dual-genre + company + link + keyword",
"""SELECT MIN(n.name)   AS actress_name,
          MIN(t.title)  AS movie_title,
          MIN(t2.title) AS linked_title
   FROM   title t,
          title t2,
          name n,
          cast_info ci,
          movie_link ml,
          movie_info mi,
          company_name cn,
          movie_companies mc,
          keyword k,
          movie_keyword mk
   WHERE  ci.role_id = 2
     AND  mi.info IN ('Drama', 'Thriller')
     AND  mi.info_type_id = 3
     AND  n.id   = ci.person_id
     AND  t.id   = ci.movie_id
     AND  t.id   = ml.movie_id
     AND  t2.id  = ml.linked_movie_id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id
     AND  mi.movie_id = t.id
     AND  k.id   = mk.keyword_id
     AND  mk.movie_id = t.id"""),

# ══════════ 9-JOIN QUERIES ════════════════════════════════════════════════════

("J8a", 9, "9-way: cast+co+kw+info+link+aka+person_info",
"""SELECT MIN(t.title)  AS movie_title,
          MIN(n.name)   AS actor_name
   FROM   title t,
          name n,
          cast_info ci,
          movie_info mi,
          company_name cn,
          movie_companies mc,
          keyword k,
          movie_keyword mk,
          aka_title at2,
          movie_link ml
   WHERE  mi.info = 'USA'
     AND  mi.info_type_id = 8
     AND  cn.country_code = 'us'
     AND  ci.role_id = 1
     AND  n.id   = ci.person_id
     AND  t.id   = ci.movie_id
     AND  mi.movie_id = t.id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id
     AND  k.id   = mk.keyword_id
     AND  mk.movie_id = t.id
     AND  at2.movie_id = t.id
     AND  ml.movie_id  = t.id"""),

("J8b", 9, "9-way: person_info + role + company_type + genre + kw + year",
"""SELECT MIN(n.name)   AS person_name,
          MIN(pi.info)  AS bio_snippet,
          MIN(t.title)  AS movie_title
   FROM   name n,
          cast_info ci,
          person_info pi,
          role_type rt,
          title t,
          movie_info mi,
          company_type ct,
          company_name cn,
          movie_companies mc,
          keyword k,
          movie_keyword mk
   WHERE  rt.role = 'actress'
     AND  pi.info_type_id = 501
     AND  ct.kind = 'production companies'
     AND  mi.info = 'Drama'
     AND  mi.info_type_id = 3
     AND  t.production_year >= 2000
     AND  n.id   = ci.person_id
     AND  rt.id  = ci.role_id
     AND  pi.person_id = n.id
     AND  t.id   = ci.movie_id
     AND  mi.movie_id = t.id
     AND  ct.id   = mc.company_type_id
     AND  cn.id   = mc.company_id
     AND  mc.movie_id = t.id
     AND  k.id   = mk.keyword_id
     AND  mk.movie_id = t.id"""),

# ══════════ V14-UNIQUE: GRAPH LAYER EXTRAS ════════════════════════════════════
# These don't exist in JOB — our differentiator.

("G1", 3, "Graph friends + co-appear in same movie",
"""SELECT MIN(n1.name) AS actor1,
          MIN(n2.name) AS actor2,
          MIN(t.title) AS movie_title
   FROM   name n1,
          name n2,
          edges e,
          cast_info ci1,
          cast_info ci2,
          title t
   WHERE  e.edge_type = 'friendship'
     AND  e.sign      = '+'
     AND  e.src_id    = n1.id
     AND  e.dst_id    = n2.id
     AND  ci1.person_id = n1.id
     AND  ci2.person_id = n2.id
     AND  ci1.movie_id  = t.id
     AND  ci2.movie_id  = t.id
     AND  n1.id <> n2.id"""),

("G2", 4, "Graph friends + co-star + genre + company",
"""SELECT MIN(n1.name) AS actor1,
          MIN(n2.name) AS actor2,
          MIN(t.title) AS movie_title
   FROM   name n1,
          name n2,
          edges e,
          cast_info ci1,
          cast_info ci2,
          title t,
          movie_info mi,
          company_name cn,
          movie_companies mc
   WHERE  e.edge_type = 'friendship'
     AND  e.src_id    = n1.id
     AND  e.dst_id    = n2.id
     AND  ci1.person_id = n1.id
     AND  ci2.person_id = n2.id
     AND  ci1.movie_id  = t.id
     AND  ci2.movie_id  = t.id
     AND  mi.info = 'Drama'
     AND  mi.info_type_id = 3
     AND  mi.movie_id = t.id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id
     AND  n1.id <> n2.id"""),

("G3", 5, "Rivals cast in same movie + genre + company + year",
"""SELECT MIN(n1.name) AS rival1,
          MIN(n2.name) AS rival2,
          MIN(t.title) AS movie_title
   FROM   name n1,
          name n2,
          edges e,
          cast_info ci1,
          cast_info ci2,
          title t,
          movie_info mi,
          company_name cn,
          movie_companies mc
   WHERE  e.edge_type = 'rivalry'
     AND  e.dst_id    = n2.id
     AND  e.src_id    = n1.id
     AND  ci1.person_id = n1.id
     AND  ci2.person_id = n2.id
     AND  ci1.movie_id  = t.id
     AND  ci2.movie_id  = t.id
     AND  mi.info IN ('Action', 'Thriller')
     AND  mi.info_type_id = 3
     AND  mi.movie_id = t.id
     AND  cn.id  = mc.company_id
     AND  mc.movie_id = t.id
     AND  n1.id <> n2.id
     AND  t.production_year >= 2000"""),

("G4", 5, "High-weight friendship + person_info + cast + movie",
"""SELECT MIN(n1.name)  AS actor1,
          MIN(n2.name)  AS actor2,
          MIN(pi1.info) AS nationality1,
          MIN(t.title)  AS movie_title
   FROM   name n1,
          name n2,
          edges e,
          cast_info ci1,
          cast_info ci2,
          person_info pi1,
          title t,
          movie_info mi
   WHERE  e.edge_type = 'friendship'
     AND  e.weight::float > 0.7
     AND  e.src_id    = n1.id
     AND  e.dst_id    = n2.id
     AND  ci1.person_id = n1.id
     AND  ci2.person_id = n2.id
     AND  ci1.movie_id  = t.id
     AND  ci2.movie_id  = t.id
     AND  pi1.person_id = n1.id
     AND  pi1.info_type_id = 500
     AND  mi.info = 'Drama'
     AND  mi.info_type_id = 3
     AND  mi.movie_id = t.id
     AND  n1.id <> n2.id"""),

# ══════════ HIGH-CARDINALITY VARIANTS ════════════════════════════════════════
# Same structural patterns as J* queries, but predicates use IN lists /
# wide year ranges so join nodes produce non-trivial row counts.
# Goal: confirm that q-errors are real and not just data-sparsity artifacts.

# ── HC-2: 2-join, broad filters ───────────────────────────────────────────────
("HC2a", 2, "HC: top-5 genres IN list + title",
"""SELECT MIN(t.title) AS movie_title, MIN(mi.info) AS genre
   FROM   title t, movie_info mi
   WHERE  mi.info_type_id = 3
     AND  mi.info IN ('Drama','Action','Comedy','Thriller','Horror')
     AND  t.id = mi.movie_id"""),

("HC2b", 2, "HC: all actors + any movie (no filter)",
"""SELECT MIN(n.name) AS actor_name, MIN(t.title) AS movie_title
   FROM   name n, cast_info ci, title t
   WHERE  n.id = ci.person_id AND t.id = ci.movie_id"""),

# ── HC-3: 3-join, wide predicates ─────────────────────────────────────────────
("HC3a", 3, "HC: cast + multi-genre + title (1990+)",
"""SELECT MIN(n.name) AS person_name, MIN(t.title) AS movie_title
   FROM   name n, cast_info ci, title t, movie_info mi
   WHERE  ci.role_id IN (1, 2)
     AND  mi.info IN ('Drama','Action','Comedy')
     AND  mi.info_type_id = 3
     AND  t.production_year >= 1990
     AND  n.id = ci.person_id AND t.id = ci.movie_id
     AND  mi.movie_id = t.id"""),

("HC3b", 3, "HC: person_info birth_date (all persons) + cast + movie",
"""SELECT MIN(n.name) AS person_name, MIN(pi.info) AS birth_date
   FROM   name n, cast_info ci, person_info pi, title t
   WHERE  pi.info_type_id = 21
     AND  n.id = ci.person_id AND pi.person_id = n.id
     AND  t.id = ci.movie_id"""),

# ── HC-4: 4-join, dual EAV — the main pattern causing compounding error ───────
("HC4a", 4, "HC: dual EAV (genre IN + country IN) + movie + company",
"""SELECT MIN(t.title) AS movie_title, MIN(cn.name) AS company
   FROM   title t, movie_info mi_g, movie_info mi_c,
          company_name cn, movie_companies mc
   WHERE  mi_g.info IN ('Drama','Action','Comedy')
     AND  mi_g.info_type_id = 3
     AND  mi_c.info IN ('USA','UK','India')
     AND  mi_c.info_type_id = 8
     AND  cn.id = mc.company_id AND t.id = mc.movie_id
     AND  mi_g.movie_id = t.id AND mi_c.movie_id = t.id"""),

("HC4b", 4, "HC: nationality (all persons) + cast + multi-genre + title",
"""SELECT MIN(n.name) AS person_name, MIN(pi.info) AS nationality
   FROM   name n, cast_info ci, person_info pi, title t, movie_info mi
   WHERE  pi.info_type_id = 500
     AND  mi.info IN ('Drama','Action','Comedy','Thriller')
     AND  mi.info_type_id = 3
     AND  n.id = ci.person_id AND pi.person_id = n.id
     AND  t.id = ci.movie_id AND mi.movie_id = t.id"""),

# ── HC-5: 5-join, broad selects ───────────────────────────────────────────────
("HC5a", 5, "HC: birth_date + cast + dual-EAV genre+country + company",
"""SELECT MIN(n.name) AS person_name, MIN(t.title) AS movie_title
   FROM   name n, cast_info ci, person_info pi, title t,
          movie_info mi_g, movie_info mi_c,
          company_name cn, movie_companies mc
   WHERE  pi.info_type_id = 21
     AND  mi_g.info IN ('Drama','Action','Comedy')
     AND  mi_g.info_type_id = 3
     AND  mi_c.info IN ('USA','UK','India','Japan')
     AND  mi_c.info_type_id = 8
     AND  n.id = ci.person_id AND pi.person_id = n.id
     AND  t.id = ci.movie_id
     AND  mi_g.movie_id = t.id AND mi_c.movie_id = t.id
     AND  cn.id = mc.company_id AND mc.movie_id = t.id"""),

# ── HC-6: 6-join, maximum realistic cardinality ───────────────────────────────
("HC6a", 6, "HC: 6-way broad (cast+kw+multi-genre+company+year)",
"""SELECT MIN(n.name) AS person_name, MIN(t.title) AS movie_title,
          MIN(k.keyword) AS kw, MIN(cn.name) AS company
   FROM   name n, cast_info ci, title t, movie_info mi,
          keyword k, movie_keyword mk, company_name cn, movie_companies mc
   WHERE  ci.role_id IN (1, 2)
     AND  mi.info IN ('Drama','Action','Comedy','Thriller')
     AND  mi.info_type_id = 3
     AND  t.production_year >= 1990
     AND  n.id = ci.person_id AND t.id = ci.movie_id
     AND  mi.movie_id = t.id
     AND  k.id = mk.keyword_id AND mk.movie_id = t.id
     AND  cn.id = mc.company_id AND mc.movie_id = t.id"""),

# ── HC-Graph: broad graph queries ─────────────────────────────────────────────
("HCG1", 3, "HC-Graph: all positive edges + co-cast in any movie",
"""SELECT MIN(n1.name) AS actor1, MIN(n2.name) AS actor2, MIN(t.title) AS movie
   FROM   name n1, name n2, edges e, cast_info ci1, cast_info ci2, title t
   WHERE  e.sign = '+'
     AND  e.src_id = n1.id AND e.dst_id = n2.id
     AND  ci1.person_id = n1.id AND ci2.person_id = n2.id
     AND  ci1.movie_id = t.id AND ci2.movie_id = t.id
     AND  n1.id != n2.id"""),

("HCG2", 4, "HC-Graph: pos edges + co-cast + multi-genre",
"""SELECT MIN(n1.name) AS actor1, MIN(n2.name) AS actor2, MIN(t.title) AS movie
   FROM   name n1, name n2, edges e, cast_info ci1, cast_info ci2,
          title t, movie_info mi
   WHERE  e.sign = '+'
     AND  e.src_id = n1.id AND e.dst_id = n2.id
     AND  ci1.person_id = n1.id AND ci2.person_id = n2.id
     AND  ci1.movie_id = t.id AND ci2.movie_id = t.id
     AND  mi.info IN ('Drama','Action','Comedy')
     AND  mi.info_type_id = 3 AND mi.movie_id = t.id
     AND  n1.id != n2.id"""),

("HCG3", 5, "HC-Graph: pos edges + co-cast + nationality + genre + company",
"""SELECT MIN(n1.name) AS actor1, MIN(pi.info) AS nat, MIN(t.title) AS movie
   FROM   name n1, name n2, edges e, cast_info ci1, cast_info ci2,
          person_info pi, title t, movie_info mi,
          company_name cn, movie_companies mc
   WHERE  e.sign = '+'
     AND  e.src_id = n1.id AND e.dst_id = n2.id
     AND  ci1.person_id = n1.id AND ci2.person_id = n2.id
     AND  ci1.movie_id = t.id AND ci2.movie_id = t.id
     AND  pi.person_id = n1.id AND pi.info_type_id = 500
     AND  mi.info IN ('Drama','Action','Comedy')
     AND  mi.info_type_id = 3 AND mi.movie_id = t.id
     AND  cn.id = mc.company_id AND mc.movie_id = t.id
     AND  n1.id != n2.id"""),

]  # end JOB_QUERIES


# ─── q-error ─────────────────────────────────────────────────────────────────

def qerr(est: float, act: float) -> float:
    e = max(est, 0.5)
    a = max(act, 0.5)
    return max(e / a, a / e)


# ─── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(conn, queries: list, timeout_ms: int = 60_000) -> list[QueryResult]:
    results = []
    cur = conn.cursor()
    cur.execute(f"SET statement_timeout = {timeout_ms}")

    for qid, n_joins, desc, sql in queries:
        t0 = time.perf_counter()
        try:
            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS OFF)\n{sql}")
            plan_json = cur.fetchone()[0][0]
            wall_ms = (time.perf_counter() - t0) * 1000

            joins = collect_join_nodes(plan_json.get("Plan", {}))

            if not joins:
                # No join nodes found → e.g. tiny result that got seq-scanned
                # Fall back to top-level estimate vs actual
                top = plan_json.get("Plan", {})
                est = float(top.get("Plan Rows", 1))
                act = float(top.get("Actual Rows", 1))
                loops = int(top.get("Actual Loops", 1))
                act = max(act * loops, 0.5)
                qe = qerr(est, act)
                bias = math.log2(max(est, 0.5) / act)
                joins = [JoinNode(0, top.get("Node Type","Seq Scan"),
                                  est, act, loops, qe, bias)]

            # Sort by depth for deepest_qe
            joins.sort(key=lambda j: j.depth, reverse=True)

            worst_qe   = max(j.qe for j in joins)
            deepest_qe = joins[0].qe  # deepest (highest depth number)
            biases     = [j.bias for j in joins]
            mean_bias  = statistics.mean(biases) if biases else 0.0
            std_bias   = statistics.stdev(biases) if len(biases) > 1 else 0.0

            r = QueryResult(
                qid=qid, n_joins=n_joins, description=desc,
                joins=joins, worst_qe=worst_qe, deepest_qe=deepest_qe,
                mean_bias=mean_bias, std_bias=std_bias,
                n_join_nodes=len(joins), wall_ms=wall_ms,
            )
            results.append(r)

            # Format line
            wq_str = f"{worst_qe:.2f}x" if worst_qe != float('inf') else "inf"
            bias_str = f"{mean_bias:+.2f}"
            print(f"  {qid:<8} joins={n_joins}  nodes={len(joins)}  "
                  f"worst_qe={wq_str:>9}  deepest_qe={deepest_qe:.2f}x  "
                  f"bias={bias_str}  ({wall_ms:.0f}ms)", flush=True)

        except Exception as ex:
            wall_ms = (time.perf_counter() - t0) * 1000
            conn.rollback()
            cur.execute(f"SET statement_timeout = {timeout_ms}")
            msg = str(ex).split('\n')[0][:100]
            print(f"  {qid:<8} ERROR: {msg}", flush=True)
            results.append(QueryResult(qid=qid, n_joins=n_joins,
                                       description=desc, wall_ms=wall_ms,
                                       error=msg))
    cur.close()
    return results


# ─── Reporting ────────────────────────────────────────────────────────────────

def _pct(lst, p):
    lst = sorted(lst)
    i = int(len(lst) * p / 100)
    return lst[min(i, len(lst)-1)]

def fmt(v: float) -> str:
    if v == float('inf'): return "     inf"
    return f"{v:>8.2f}x"

def summarise(results: list[QueryResult]):
    good = [r for r in results if r.error is None]
    errs = [r for r in results if r.error is not None]
    if not good:
        print("No successful queries.")
        return

    W = [r.worst_qe for r in good if r.worst_qe != float('inf')]
    D = [r.deepest_qe for r in good if r.deepest_qe != float('inf')]
    B = [r.mean_bias for r in good]

    BAR = "=" * 72
    print(f"\n{BAR}")
    print(f"  JOB Benchmark v2  —  {len(good)} queries OK, {len(errs)} errors")
    print(BAR)

    def stats_line(label, vals):
        if not vals: return
        print(f"  {label:<14}  med={_pct(vals,50):.2f}x  "
              f"p75={_pct(vals,75):.2f}x  p90={_pct(vals,90):.2f}x  "
              f"max={max(vals):.2f}x  mean={statistics.mean(vals):.2f}x")

    stats_line("worst_qe", W)
    stats_line("deepest_qe", D)

    over  = sum(1 for b in B if b >  0.1)
    under = sum(1 for b in B if b < -0.1)
    exact = len(B) - over - under
    sign  = "OVER" if statistics.mean(B) > 0 else "UNDER"
    print(f"  bias           mean log2={statistics.mean(B):+.3f}  ({sign}-estimator)  "
          f"over={over} under={under} exact={exact}")

    # By number of input joins (query complexity)
    print(f"\n  {'Joins':>5}  {'N':>3}  {'med worst_qe':>12}  "
          f"{'max worst_qe':>12}  {'med bias':>9}  {'family'}")
    print(f"  {'-'*5}  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*9}  -------")

    by_joins: dict[int, list[QueryResult]] = {}
    for r in good:
        by_joins.setdefault(r.n_joins, []).append(r)
    for j in sorted(by_joins):
        rs = by_joins[j]
        wqs = [r.worst_qe for r in rs if r.worst_qe != float('inf')]
        bs  = [r.mean_bias for r in rs]
        fam = "JOB" if j <= 9 and not rs[0].qid.startswith("G") else "Graph"
        if rs[0].qid.startswith("G"): fam = "Graph(V14)"
        print(f"  {j:>5}  {len(rs):>3}  "
              f"{_pct(wqs,50) if wqs else float('inf'):>12.2f}x  "
              f"{max(wqs) if wqs else float('inf'):>12.2f}x  "
              f"{statistics.mean(bs) if bs else 0:>+9.2f}  {fam}")

    # q-error histogram
    print(f"\n  q-error histogram (worst_qe):")
    buckets = [(1,2,"1–2x"), (2,5,"2–5x"), (5,10,"5–10x"),
               (10,20,"10–20x"), (20,100,"20–100x"), (100,1e9,"≥100x")]
    for lo, hi, label in buckets:
        n = sum(1 for r in good if lo <= r.worst_qe < hi)
        bar = "#" * n
        print(f"    {label:>8}  {bar:<30} {n}")

    # Top offenders
    print(f"\n  Top 10 worst q-error queries:")
    worst = sorted([r for r in good if r.worst_qe != float('inf')],
                   key=lambda r: r.worst_qe, reverse=True)[:10]
    for r in worst:
        b = f"{r.mean_bias:+.2f}" if r.mean_bias else " ---"
        print(f"    {r.qid:<8}  worst_qe={fmt(r.worst_qe)}  "
              f"deepest={fmt(r.deepest_qe)}  bias={b}  {r.description}")

    # Deepest-join node analysis for top 5 worst
    print(f"\n  Deepest join node details (top 5 worst queries):")
    for r in worst[:5]:
        print(f"    [{r.qid}] {r.description}")
        for j in sorted(r.joins, key=lambda jj: jj.depth, reverse=True)[:3]:
            print(f"       depth={j.depth}  {j.node_type:<18}  "
                  f"est={j.est_rows:>8.0f}  act={j.act_rows:>8.1f}  "
                  f"qe={j.qe:>8.2f}x  [{j.condition[:50]}]")

    print(BAR)


def write_csv(path: str, results: list[QueryResult]):
    rows = []
    for r in results:
        rows.append({
            "qid": r.qid, "n_joins": r.n_joins,
            "description": r.description,
            "worst_qe":    round(r.worst_qe, 4) if r.worst_qe != float('inf') else "inf",
            "deepest_qe":  round(r.deepest_qe, 4) if r.deepest_qe != float('inf') else "inf",
            "mean_bias":   round(r.mean_bias, 4),
            "std_bias":    round(r.std_bias, 4),
            "n_join_nodes":r.n_join_nodes,
            "wall_ms":     round(r.wall_ms, 1) if r.wall_ms else "",
            "error":       r.error or "",
        })
    fn = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV -> {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Correct JOB benchmark on IMDB schema")
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=5433)
    parser.add_argument("--dbname",   default="imdb_benchmark")
    parser.add_argument("--user",     default="postgres")
    parser.add_argument("--password", default="bench")
    parser.add_argument("--csv",      default="job_v2_results.csv")
    parser.add_argument("--timeout",  type=int, default=60000,
                        help="statement_timeout per query (ms)")
    args = parser.parse_args()

    print(f"\nConnecting {args.dbname} @ {args.host}:{args.port}")
    conn = psycopg2.connect(
        host=args.host, port=args.port, dbname=args.dbname,
        user=args.user, password=args.password)
    conn.autocommit = True

    cur = conn.cursor()
    cur.execute("ANALYZE")
    cur.close()

    print(f"Running {len(JOB_QUERIES)} queries...\n")
    print(f"  {'Query':<8}  {'joins':>5}  {'nodes':>5}  "
          f"{'worst_qe':>9}  {'deepest_qe':>10}  {'bias':>6}  ms")
    print(f"  {'-'*8}  {'-'*5}  {'-'*5}  "
          f"{'-'*9}  {'-'*10}  {'-'*6}  ----")

    results = run_benchmark(conn, JOB_QUERIES, timeout_ms=args.timeout)
    conn.close()

    summarise(results)
    write_csv(args.csv, results)


if __name__ == "__main__":
    main()
