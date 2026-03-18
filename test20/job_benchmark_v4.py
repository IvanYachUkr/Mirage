"""
JOB-Style Benchmark v4 FINAL: Focus on FK-path fan-out queries that produce UNDER-estimation.
The key patterns that produce massive q-errors:
1. Multiple roles hitting same movie (actor+writer, director+producer)
2. cast_info joining to both person_info AND movie_info paths
3. movie_info self-joins with broader IN() predicates
4. aka_title + aka_name producing unexpected fan-out
5. complete_cast / movie_info_idx creating hidden correlations
"""
import sys, io, time, math, psycopg2, csv

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_v17', user='postgres', password='bench')
conn.autocommit = True
cur = conn.cursor()

QUERIES = [

# === Multi-role: same movie has actor AND writer/director ===
("J1a", "4J", "actor+writer same movie, US company, year>2005",
"""SELECT COUNT(*) FROM title t, cast_info ci1, cast_info ci2,
        movie_companies mc, company_name cn
   WHERE ci1.movie_id = t.id AND ci2.movie_id = t.id
     AND mc.movie_id = t.id AND mc.company_id = cn.id
     AND ci1.role_id = 1 AND ci2.role_id = 5
     AND cn.country_code = 'us' AND t.production_year > 2005"""),

("J1b", "4J", "actor+director same movie + keyword",
"""SELECT COUNT(*) FROM title t, cast_info ci1, cast_info ci2, movie_keyword mk
   WHERE ci1.movie_id = t.id AND ci2.movie_id = t.id AND mk.movie_id = t.id
     AND ci1.role_id = 1 AND ci2.role_id = 6
     AND t.production_year > 2000"""),

("J1c", "5J", "actress+producer+composer same movie + company",
"""SELECT COUNT(*) FROM title t, cast_info ci1, cast_info ci2, cast_info ci3,
        movie_companies mc
   WHERE ci1.movie_id = t.id AND ci2.movie_id = t.id AND ci3.movie_id = t.id
     AND mc.movie_id = t.id
     AND ci1.role_id = 2 AND ci2.role_id = 4 AND ci3.role_id = 8"""),

# === FK fan-out: cast + movie_info + keyword + company (star join) ===
("J2a", "4J", "actor + keyword + genre(Drama,Thriller,Crime)",
"""SELECT COUNT(*) FROM title t, cast_info ci, movie_keyword mk, movie_info mi
   WHERE ci.movie_id = t.id AND mk.movie_id = t.id AND mi.movie_id = t.id
     AND ci.role_id = 1 AND mi.info_type_id = 3
     AND mi.info IN ('Drama','Thriller','Crime')
     AND t.production_year > 2005"""),

("J2b", "5J", "cast + keyword + genre + company + year>2010",
"""SELECT COUNT(*) FROM title t, cast_info ci, movie_keyword mk,
        movie_info mi, movie_companies mc
   WHERE ci.movie_id = t.id AND mk.movie_id = t.id AND mi.movie_id = t.id
     AND mc.movie_id = t.id
     AND mi.info_type_id = 3 AND mi.info IN ('Action','Sci-Fi')
     AND t.production_year > 2010"""),

# === EAV paths via movie_info joining to cast ===
("J4a", "5J", "genre(Drama) + country(USA) + actor + keyword",
"""SELECT COUNT(*) FROM movie_info mi1, movie_info mi2, cast_info ci,
        movie_keyword mk, title t
   WHERE mi1.movie_id = t.id AND mi2.movie_id = t.id
     AND ci.movie_id = t.id AND mk.movie_id = t.id
     AND mi1.info_type_id = 3 AND mi1.info = 'Drama'
     AND mi2.info_type_id = 5 AND mi2.info = 'USA'
     AND ci.role_id = 1"""),

("J4b", "6J", "genre(Action) + country(USA) + cert(R) + actor + keyword + company",
"""SELECT COUNT(*) FROM title t, movie_info mi1, movie_info mi2, movie_info mi3,
        cast_info ci, movie_keyword mk, movie_companies mc
   WHERE mi1.movie_id = t.id AND mi2.movie_id = t.id AND mi3.movie_id = t.id
     AND ci.movie_id = t.id AND mk.movie_id = t.id AND mc.movie_id = t.id
     AND mi1.info_type_id = 3 AND mi1.info = 'Action'
     AND mi2.info_type_id = 5 AND mi2.info = 'USA'
     AND mi3.info_type_id = 6 AND mi3.info = 'R'
     AND ci.role_id = 1"""),

# === Person-centric paths: person_info → cast → movie_info ===
("J6a", "5J", "person(nat=USA) + actor + genre(Drama) + company + keyword",
"""SELECT COUNT(*) FROM person_info pi, cast_info ci,
        movie_info mi, movie_companies mc, movie_keyword mk
   WHERE pi.person_id = ci.person_id AND mi.movie_id = ci.movie_id
     AND mc.movie_id = ci.movie_id AND mk.movie_id = ci.movie_id
     AND pi.info_type_id = 18 AND pi.info = 'USA'
     AND ci.role_id = 1
     AND mi.info_type_id = 3 AND mi.info = 'Drama'"""),

("J6b", "5J", "person(nat IN) + actress + genre(Horror,Thriller) + keyword",
"""SELECT COUNT(*) FROM person_info pi, cast_info ci,
        movie_info mi, movie_keyword mk, title t
   WHERE pi.person_id = ci.person_id AND ci.movie_id = t.id
     AND mi.movie_id = t.id AND mk.movie_id = t.id
     AND pi.info_type_id = 18 AND pi.info IN ('Japan','South Korea')
     AND ci.role_id = 2
     AND mi.info_type_id = 3 AND mi.info IN ('Horror','Thriller')"""),

# === aka fan-out: aka_title + aka_name amplify rows ===
("J8a", "5J", "cast + aka_name + aka_title + genre(Drama) + year>2010",
"""SELECT COUNT(*) FROM title t, cast_info ci, aka_name an, aka_title at, movie_info mi
   WHERE ci.movie_id = t.id AND an.person_id = ci.person_id AND at.movie_id = t.id
     AND mi.movie_id = t.id
     AND mi.info_type_id = 3 AND mi.info = 'Drama'
     AND t.production_year > 2010"""),

("J8b", "6J", "cast + aka_name + keyword + company + genre + country",
"""SELECT COUNT(*) FROM title t, cast_info ci, aka_name an,
        movie_keyword mk, movie_companies mc, movie_info mi1, movie_info mi2
   WHERE ci.movie_id = t.id AND an.person_id = ci.person_id
     AND mk.movie_id = t.id AND mc.movie_id = t.id
     AND mi1.movie_id = t.id AND mi2.movie_id = t.id
     AND mi1.info_type_id = 3 AND mi1.info IN ('Action','Thriller')
     AND mi2.info_type_id = 5 AND mi2.info IN ('USA','UK')"""),

# === complete_cast / movie_info_idx hidden correlations ===
("J9a", "5J", "complete_cast + genre(Drama) + actor + company + year>2000",
"""SELECT COUNT(*) FROM title t, complete_cast cc, cast_info ci,
        movie_info mi, movie_companies mc
   WHERE cc.movie_id = t.id AND ci.movie_id = t.id AND mi.movie_id = t.id
     AND mc.movie_id = t.id
     AND mi.info_type_id = 3 AND mi.info = 'Drama'
     AND ci.role_id = 1
     AND t.production_year > 2000"""),

("J9b", "5J", "movie_info_idx(votes) + genre + cast + keyword + company",
"""SELECT COUNT(*) FROM title t, movie_info_idx mii, cast_info ci,
        movie_info mi, movie_keyword mk, movie_companies mc
   WHERE mii.movie_id = t.id AND ci.movie_id = t.id AND mi.movie_id = t.id
     AND mk.movie_id = t.id AND mc.movie_id = t.id
     AND mii.info_type_id = 13
     AND mi.info_type_id = 3 AND mi.info IN ('Action','Drama')
     AND ci.role_id IN (1,2)"""),

# === Multi-path: two independent paths merge on title ===
("J10a", "6J", "cast-path + link-path merge: cast+keyword+link+aka_title+genre",
"""SELECT COUNT(*) FROM title t, cast_info ci, movie_keyword mk,
        movie_link ml, aka_title at, movie_info mi
   WHERE ci.movie_id = t.id AND mk.movie_id = t.id AND ml.movie_id = t.id
     AND at.movie_id = t.id AND mi.movie_id = t.id
     AND mi.info_type_id = 3 AND mi.info IN ('Action','Comedy','Drama')
     AND ci.role_id = 1"""),

("J10b", "7J", "cast-path + company-path + EAV: actor+keyword+company+genre+country+cert",
"""SELECT COUNT(*) FROM title t, cast_info ci, movie_keyword mk,
        movie_companies mc, movie_info mi1, movie_info mi2, movie_info mi3
   WHERE ci.movie_id = t.id AND mk.movie_id = t.id AND mc.movie_id = t.id
     AND mi1.movie_id = t.id AND mi2.movie_id = t.id AND mi3.movie_id = t.id
     AND ci.role_id = 1
     AND mi1.info_type_id = 3 AND mi1.info = 'Drama'
     AND mi2.info_type_id = 5 AND mi2.info = 'USA'
     AND mi3.info_type_id = 6 AND mi3.info = 'R'"""),
]

print(f"Running {len(QUERIES)} JOB-style v4 queries on imdb_v17...")
print(f"  {'Q':<6} {'#J':<6} {'Description':<46} {'Q-Error':>12} {'Est':>10} {'Act':>10} {'ms':>7}  Bias")
print("  " + "-" * 106)

results = []
for qid, nj, desc, sql in QUERIES:
    try:
        t0 = time.time()
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
        plan_json = cur.fetchone()[0]
        elapsed = (time.time() - t0) * 1000
        plan = plan_json[0]["Plan"]

        def walk(node, depth=0):
            res = []
            nt = node.get("Node Type", "")
            est = float(node.get("Plan Rows", 1))
            loops = max(int(node.get("Actual Loops", 1)), 1)
            act = float(node.get("Actual Rows", 0)) * loops
            act = max(act, 0.5)
            if est <= 0: est = 0.5
            qe = max(est/act, act/est)
            bias = math.log2(est/act)
            res.append((depth, nt, est, act, qe, bias))
            for sub in node.get("Plans", []):
                res.extend(walk(sub, depth+1))
            return res

        all_nodes = walk(plan)
        real_nodes = [n for n in all_nodes if n[3] > 0.5]
        worst = max(real_nodes, key=lambda x: x[4]) if real_nodes else max(all_nodes, key=lambda x: x[4])
        _, _, w_est, w_act, w_qe, w_bias = worst
        bias_dir = "UNDER" if w_bias < 0 else "OVER"

        print(f"  {qid:<6} {nj:<6} {desc[:44]:<46} {w_qe:>10.1f}x  {w_est:>9.0f} {w_act:>9.0f} {elapsed:>6.0f}  {bias_dir}", flush=True)
        results.append((qid, nj, desc, w_qe, w_est, w_act, w_bias, elapsed, bias_dir))
    except Exception as ex:
        elapsed = (time.time() - t0) * 1000
        print(f"  {qid:<6} {nj:<6} {desc[:44]:<46}  ERROR ({elapsed:.0f}ms): {str(ex)[:50]}", flush=True)

real_qe = [r[3] for r in results]
under = [r for r in results if r[8] == 'UNDER']
print(f"\n{'='*80}")
print(f"SUMMARY ({len(results)} queries)")
print(f"{'='*80}")
print(f"  UNDER-estimates: {len(under)}/{len(results)}")
print(f"  q-error > 100x:  {sum(1 for q in real_qe if q > 100)}/{len(real_qe)}")
print(f"  q-error > 1000x: {sum(1 for q in real_qe if q > 1000)}/{len(real_qe)}")
print(f"  Max q-error:     {max(real_qe):,.1f}x")
print(f"  Median q-error:  {sorted(real_qe)[len(real_qe)//2]:,.1f}x")

with open('benchmark_v17_job_v4.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['qid','joins','desc','worst_real_qe','est','act','bias','ms','bias_dir'])
    for r in results:
        w.writerow(r)
print(f"  CSV -> benchmark_v17_job_v4.csv")

cur.close(); conn.close()
