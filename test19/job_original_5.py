"""Run 5 original JOB benchmark queries (q7a, q8a, q16a, q17a, q25a) adapted for our dataset."""
import sys, io, json, time, math, psycopg2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_v16', user='postgres', password='bench')
conn.autocommit = True
cur = conn.cursor()

# These are the 5 hardest original JOB queries, adapted for our schema.
# Changes from original:
#   - country_code: '[us]' -> 'us', '[jp]' -> 'jp'
#   - info_type.info: 'mini biography' -> 'biography' (our mapping)
#   - pi.note: removed 'Volker Boehm' filter (not in our dataset)
#   - k.keyword: adapted to keywords in our dataset
#   - ci.note: adapted to notes in our dataset

QUERIES = [

# Q7a: 7 joins — person_info + aka_name + cast + link + title
# HARDEST in JOB: person_info × cast_info × aka_name creates massive underestimates
("JOB-7a", 7, "ORIGINAL JOB q7a: aka_name + cast + person_info + movie_link",
"""SELECT MIN(n.name) AS of_person,
       MIN(t.title) AS biography_movie
FROM aka_name AS an,
     cast_info AS ci,
     info_type AS it,
     link_type AS lt,
     movie_link AS ml,
     name AS n,
     person_info AS pi,
     title AS t
WHERE an.name LIKE '%a%'
  AND it.info = 'biography'
  AND lt.link = 'features'
  AND t.production_year BETWEEN 1980 AND 2010
  AND n.id = an.person_id
  AND n.id = pi.person_id
  AND ci.person_id = n.id
  AND t.id = ci.movie_id
  AND ml.linked_movie_id = t.id
  AND lt.id = ml.link_type_id
  AND it.id = pi.info_type_id
  AND pi.person_id = an.person_id
  AND pi.person_id = ci.person_id
  AND an.person_id = ci.person_id
  AND ci.movie_id = ml.linked_movie_id"""),

# Q8a: 6 joins — company(jp) + cast + aka_name + role_type
# Known for catastrophic underestimates on note-filtered joins
("JOB-8a", 6, "ORIGINAL JOB q8a: aka_name + company(jp) + cast + role_type",
"""SELECT MIN(an1.name) AS actress_pseudonym,
       MIN(t.title) AS movie_dubbed
FROM aka_name AS an1,
     cast_info AS ci,
     company_name AS cn,
     movie_companies AS mc,
     name AS n1,
     role_type AS rt,
     title AS t
WHERE cn.country_code = 'jp'
  AND rt.role = 'actress'
  AND an1.person_id = n1.id
  AND n1.id = ci.person_id
  AND ci.movie_id = t.id
  AND t.id = mc.movie_id
  AND mc.company_id = cn.id
  AND ci.role_id = rt.id
  AND an1.person_id = ci.person_id
  AND ci.movie_id = mc.movie_id"""),

# Q16a: 7 joins — keyword + company + cast + aka_name + episode range
# The episode_nr predicate is the classic JOB killer
("JOB-16a", 7, "ORIGINAL JOB q16a: aka_name + keyword + company + episode_nr",
"""SELECT MIN(an.name) AS cool_actor_pseudonym,
       MIN(t.title) AS series_named_after_char
FROM aka_name AS an,
     cast_info AS ci,
     company_name AS cn,
     keyword AS k,
     movie_companies AS mc,
     movie_keyword AS mk,
     name AS n,
     title AS t
WHERE cn.country_code = 'us'
  AND t.episode_nr >= 5
  AND t.episode_nr < 50
  AND an.person_id = n.id
  AND n.id = ci.person_id
  AND ci.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND t.id = mc.movie_id
  AND mc.company_id = cn.id
  AND an.person_id = ci.person_id
  AND ci.movie_id = mc.movie_id
  AND ci.movie_id = mk.movie_id
  AND mc.movie_id = mk.movie_id"""),

# Q17a: 6 joins — keyword + company + cast + name LIKE
# Multi-table equality joins with selective predicates
("JOB-17a", 6, "ORIGINAL JOB q17a: keyword + company(us) + cast + name LIKE",
"""SELECT MIN(n.name) AS member_in_charnamed_american_movie,
       MIN(n.name) AS a1
FROM cast_info AS ci,
     company_name AS cn,
     keyword AS k,
     movie_companies AS mc,
     movie_keyword AS mk,
     name AS n,
     title AS t
WHERE cn.country_code = 'us'
  AND n.name LIKE 'B%'
  AND n.id = ci.person_id
  AND ci.movie_id = t.id
  AND t.id = mk.movie_id
  AND mk.keyword_id = k.id
  AND t.id = mc.movie_id
  AND mc.company_id = cn.id
  AND ci.movie_id = mc.movie_id
  AND ci.movie_id = mk.movie_id
  AND mc.movie_id = mk.movie_id"""),

# Q25a: 8 joins — writer + genre(Horror) + votes + violent keywords
# The dual info_type + keyword combination is devastating
("JOB-25a", 8, "ORIGINAL JOB q25a: writer + genre(Horror) + votes + violent keywords",
"""SELECT MIN(mi.info) AS movie_budget,
       MIN(mi_idx.info) AS movie_votes,
       MIN(n.name) AS male_writer,
       MIN(t.title) AS violent_movie_title
FROM cast_info AS ci,
     info_type AS it1,
     info_type AS it2,
     keyword AS k,
     movie_info AS mi,
     movie_info_idx AS mi_idx,
     movie_keyword AS mk,
     name AS n,
     title AS t
WHERE it1.info = 'genres'
  AND it2.info = 'votes'
  AND mi.info = 'Horror'
  AND n.gender = 'm'
  AND t.id = mi.movie_id
  AND t.id = mi_idx.movie_id
  AND t.id = ci.movie_id
  AND t.id = mk.movie_id
  AND ci.movie_id = mi.movie_id
  AND ci.movie_id = mi_idx.movie_id
  AND ci.movie_id = mk.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND mi.movie_id = mk.movie_id
  AND mi_idx.movie_id = mk.movie_id
  AND n.id = ci.person_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id
  AND k.id = mk.keyword_id"""),
]

JOIN_NODE_TYPES = {
    "Hash Join", "Merge Join", "Nested Loop",
    "Hash Anti Join", "Merge Anti Join",
    "Hash Semi Join", "Merge Semi Join",
}

def collect_nodes(node, depth=0):
    results = []
    nt = node.get("Node Type", "")
    est = float(node.get("Plan Rows", 1))
    loops = max(int(node.get("Actual Loops", 1)), 1)
    act = float(node.get("Actual Rows", 0)) * loops
    act = max(act, 0.5)
    if est <= 0: est = 0.5
    qe = max(est/act, act/est)
    bias = math.log2(est/act)
    cond = node.get("Hash Cond") or node.get("Merge Cond") or node.get("Join Filter") or ""
    if nt in JOIN_NODE_TYPES or "Join" in nt:
        results.append((depth, nt, est, act, qe, bias, str(cond)[:60]))
    for sub in node.get("Plans", []):
        results.extend(collect_nodes(sub, depth+1))
    return results

print(f"\n{'='*90}")
print("ORIGINAL JOB QUERIES ON SYNTHETIC DATASET (v16)")
print(f"{'='*90}\n")

for qid, n_joins, desc, sql in QUERIES:
    try:
        t0 = time.time()
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
        plan_json = cur.fetchone()[0]
        elapsed_ms = (time.time() - t0) * 1000
        
        plan = plan_json[0]["Plan"]
        nodes = collect_nodes(plan)
        
        if not nodes:
            print(f"  {qid:<10s} {desc[:60]:<62s} NO JOIN NODES  ({elapsed_ms:.0f}ms)")
            continue
        
        worst = max(nodes, key=lambda x: x[4])
        deepest = max(nodes, key=lambda x: x[0])
        
        print(f"  {qid:<10s} joins={n_joins}  nodes={len(nodes)}  worst_qe={worst[4]:>10.1f}x  deepest_qe={deepest[4]:>6.1f}x  ({elapsed_ms:.0f}ms)")
        print(f"  {'':10s} {desc}")
        
        # Show top 3 worst join nodes
        for d, nt, e, a, q, b, c in sorted(nodes, key=lambda x: -x[4])[:3]:
            over_under = "UNDER" if b < 0 else "OVER"
            print(f"  {'':10s}   d={d} {nt:<18s} est={e:>10.0f}  act={a:>10.0f}  qe={q:>8.1f}x  {over_under}  {c}")
        print()
        
    except Exception as ex:
        elapsed_ms = (time.time() - t0) * 1000
        print(f"  {qid:<10s} ERROR: {str(ex)[:80]}  ({elapsed_ms:.0f}ms)\n")

print(f"{'='*90}")
cur.close()
conn.close()
