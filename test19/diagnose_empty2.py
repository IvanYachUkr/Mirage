"""Deep diagnosis of remaining 30 empty queries.
For each, check the exact adapted SQL and run sub-queries to find blockers.
"""
import psycopg2, re, sys, io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')
conn.autocommit = True
cur = conn.cursor()

ADAPTED = Path(r"c:\Users\vanya\Documents\DATA_SYS_LAB\test19\job_adapted")

EMPTY = ["2b","2c","6c","7a","7b","11a","11b","11c","12b","14b",
         "15a","15b","16a","16c","16d","21a","21b","21c","24b","25b",
         "27a","27b","27c","29a","29b","29c","31b","33a","33b","33c"]

# For each empty query, let's check specific blocking combos
for qid in EMPTY:
    sql = (ADAPTED / f"{qid}.sql").read_text(encoding="utf-8").strip()
    print(f"\n{'='*60}")
    print(f"  {qid}")
    print(f"{'='*60}")
    # Print the actual adapted SQL (truncated)
    for line in sql.split('\n'):
        line = line.strip()
        if line and not line.startswith('SELECT') and 'AS ' not in line and '.' in line:
            print(f"  {line[:90]}")

# Now run specific diagnostic queries for the blockers
print("\n\n=== SPECIFIC DIAGNOSTICS ===\n")

diagnostics = {
    # 2b/2c: These have link_type + movie_link. How many movie_link entries exist?
    "movie_link_count": "SELECT COUNT(*) FROM movie_link",
    "movie_link_by_type": "SELECT lt.link, COUNT(ml.id) FROM movie_link ml JOIN link_type lt ON lt.id=ml.link_type_id GROUP BY lt.link ORDER BY COUNT(ml.id) DESC",
    
    # 11a-c, 21a-c, 27a-c: All use movie_link + mc.note IS NOT NULL
    # The combo of movie_link + movie_keyword + movie_companies might be empty
    "ml_mk_overlap": """SELECT COUNT(DISTINCT ml.movie_id) FROM movie_link ml 
        JOIN movie_keyword mk ON mk.movie_id=ml.movie_id""",
    "ml_mc_overlap": """SELECT COUNT(DISTINCT ml.movie_id) FROM movie_link ml 
        JOIN movie_companies mc ON mc.movie_id=ml.movie_id""",
    "ml_mk_mc_overlap": """SELECT COUNT(DISTINCT ml.movie_id) FROM movie_link ml 
        JOIN movie_keyword mk ON mk.movie_id=ml.movie_id
        JOIN movie_companies mc ON mc.movie_id=ml.movie_id""",
    "ml_mk_mc_mi_overlap": """SELECT COUNT(DISTINCT ml.movie_id) FROM movie_link ml 
        JOIN movie_keyword mk ON mk.movie_id=ml.movie_id
        JOIN movie_companies mc ON mc.movie_id=ml.movie_id
        JOIN movie_info mi ON mi.movie_id=ml.movie_id""",
    
    # 7a, 7b: person_info + movie_link
    "pi_count": "SELECT COUNT(*) FROM person_info",
    "pi_with_ml": """SELECT COUNT(DISTINCT pi.person_id) FROM person_info pi 
        JOIN cast_info ci ON ci.person_id=pi.person_id 
        JOIN movie_link ml ON ml.movie_id=ci.movie_id""",
    
    # 12b: it2.info='bottom 10 rank' — do we have bottom 10 rank entries?
    "bottom_10_rank": """SELECT COUNT(*) FROM movie_info_idx mi JOIN info_type it ON it.id=mi.info_type_id 
        WHERE it.info='bottom 10 rank'""",
    "idx_info_types": """SELECT it.info, COUNT(mi.id) FROM movie_info_idx mi 
        JOIN info_type it ON it.id=mi.info_type_id GROUP BY it.info ORDER BY COUNT(mi.id) DESC""",
    
    # 15a,15b: mi.note like '%Streaming%' with cn=[us] + it1='release dates'
    "streaming_count": "SELECT COUNT(*) FROM movie_info WHERE note LIKE '%Streaming%'",
    "streaming_with_cn": """SELECT COUNT(*) FROM movie_info mi 
        JOIN info_type it ON it.id=mi.info_type_id 
        JOIN title t ON t.id=mi.movie_id
        JOIN movie_companies mc ON mc.movie_id=t.id
        JOIN company_name cn ON cn.id=mc.company_id
        WHERE mi.note LIKE '%Streaming%' AND it.info='release dates' AND cn.country_code='[us]'""",
    "streaming_with_aka": """SELECT COUNT(*) FROM movie_info mi 
        JOIN info_type it ON it.id=mi.info_type_id 
        JOIN aka_title at ON at.movie_id=mi.movie_id
        WHERE mi.note LIKE '%Streaming%' AND it.info='release dates'""",
    
    # 16a,c,d: episode_nr ranges — max is 24
    "episode_keywords": """SELECT COUNT(*) FROM title t 
        JOIN movie_keyword mk ON mk.movie_id=t.id 
        WHERE t.episode_nr IS NOT NULL AND t.episode_nr >= 5""",
    "episode_keyword_murder": """SELECT COUNT(*) FROM title t 
        JOIN movie_keyword mk ON mk.movie_id=t.id 
        JOIN keyword k ON k.id=mk.keyword_id
        WHERE t.episode_nr IS NOT NULL AND t.episode_nr >= 5 AND k.keyword='murder'""",
    "episode_keyword_any": """SELECT k.keyword, COUNT(*) FROM title t 
        JOIN movie_keyword mk ON mk.movie_id=t.id 
        JOIN keyword k ON k.id=mk.keyword_id
        WHERE t.episode_nr IS NOT NULL AND t.episode_nr >= 1 AND t.episode_nr <= 24
        GROUP BY k.keyword ORDER BY COUNT(*) DESC LIMIT 10""",
    
    # 29a-c: complete_cast. How many complete_cast entries?
    "complete_cast_count": "SELECT COUNT(*) FROM complete_cast",
    "cc_types": """SELECT cct.kind, COUNT(cc.id) FROM complete_cast cc 
        JOIN comp_cast_type cct ON cct.id=cc.subject_id GROUP BY cct.kind""",
    "cc_status": """SELECT cct.kind, COUNT(cc.id) FROM complete_cast cc 
        JOIN comp_cast_type cct ON cct.id=cc.status_id GROUP BY cct.kind""",
    "cc_with_keyword": """SELECT COUNT(DISTINCT cc.movie_id) FROM complete_cast cc 
        JOIN movie_keyword mk ON mk.movie_id=cc.movie_id""",
    "cc_with_mi": """SELECT COUNT(DISTINCT cc.movie_id) FROM complete_cast cc 
        JOIN movie_info mi ON mi.movie_id=cc.movie_id""",
    "cc_with_ci": """SELECT COUNT(DISTINCT cc.movie_id) FROM complete_cast cc 
        JOIN cast_info ci ON ci.movie_id=cc.movie_id""",
    
    # 33a-c: movie_link + movie_info_idx for BOTH linked titles
    "ml_with_idx": """SELECT COUNT(*) FROM movie_link ml
        JOIN movie_info_idx mi1 ON mi1.movie_id=ml.movie_id
        JOIN movie_info_idx mi2 ON mi2.movie_id=ml.linked_movie_id""",
    
    # 6c: keyword violence + name + year combo - what was adapted?
    "violence_name_year": """SELECT COUNT(*) FROM movie_keyword mk
        JOIN keyword k ON k.id=mk.keyword_id
        JOIN cast_info ci ON ci.movie_id=mk.movie_id
        JOIN name n ON n.id=ci.person_id
        WHERE k.keyword='violence' AND n.name LIKE '%Sofia%Uribe%'""",
    
    # 14b: What title LIKE patterns exist with murder keyword?
    "murder_titles": """SELECT t.title FROM title t 
        JOIN movie_keyword mk ON mk.movie_id=t.id 
        JOIN keyword k ON k.id=mk.keyword_id
        WHERE k.keyword IN ('murder','violence') AND (t.title LIKE '%Broken%' OR t.title LIKE '%Storm%')
        LIMIT 10""",
    
    # Check aka_title count
    "aka_title_count": "SELECT COUNT(*) FROM aka_title",
}

for name, sql in diagnostics.items():
    try:
        cur.execute(sql)
        results = cur.fetchall()
        print(f"\n{name}:")
        for r in results:
            print(f"  {r}")
    except Exception as e:
        conn.rollback()
        print(f"\n{name}: ERROR - {str(e)[:80]}")

cur.close()
conn.close()
