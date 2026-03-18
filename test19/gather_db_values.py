"""Gather predicate values from our synthetic IMDB database for JOB query adaptation."""
import psycopg2

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')
cur = conn.cursor()

sections = [
    ("COMPANY_TYPE", "SELECT id, kind FROM company_type ORDER BY id"),
    ("KIND_TYPE", "SELECT id, kind FROM kind_type ORDER BY id"),
    ("ROLE_TYPE", "SELECT id, role FROM role_type ORDER BY id"),
    ("INFO_TYPE", "SELECT id, info FROM info_type ORDER BY id"),
    ("LINK_TYPE", "SELECT id, link FROM link_type ORDER BY id"),
    ("COMP_CAST_TYPE", "SELECT id, kind FROM comp_cast_type ORDER BY id"),
    ("TOP 30 KEYWORDS (by movie count)",
     "SELECT k.keyword, COUNT(mk.id) c FROM keyword k JOIN movie_keyword mk ON k.id=mk.keyword_id GROUP BY k.keyword ORDER BY c DESC LIMIT 30"),
    ("GENRES (movie_info info_type_id=3)",
     "SELECT info, COUNT(id) c FROM movie_info WHERE info_type_id=3 GROUP BY info ORDER BY c DESC LIMIT 20"),
    ("COUNTRIES (movie_info info_type_id=8)",
     "SELECT info, COUNT(id) c FROM movie_info WHERE info_type_id=8 GROUP BY info ORDER BY c DESC LIMIT 15"),
    ("LANGUAGES (movie_info info_type_id=4)",
     "SELECT info, COUNT(id) c FROM movie_info WHERE info_type_id=4 GROUP BY info ORDER BY c DESC LIMIT 10"),
    ("CERTIFICATES (movie_info info_type_id=5)",
     "SELECT info, COUNT(id) c FROM movie_info WHERE info_type_id=5 GROUP BY info ORDER BY c DESC LIMIT 10"),
    ("COLOR_INFO (movie_info info_type_id=2)",
     "SELECT info, COUNT(id) c FROM movie_info WHERE info_type_id=2 GROUP BY info ORDER BY c DESC"),
    ("PRODUCTION_TIER (movie_info info_type_id=200)",
     "SELECT info, COUNT(id) c FROM movie_info WHERE info_type_id=200 GROUP BY info ORDER BY c DESC"),
    ("COMPANY country_code distribution",
     "SELECT country_code, COUNT(id) c FROM company_name GROUP BY country_code ORDER BY c DESC LIMIT 15"),
    ("MOVIE_COMPANIES notes (sample)",
     "SELECT note, COUNT(id) c FROM movie_companies WHERE note IS NOT NULL AND note != '' GROUP BY note ORDER BY c DESC LIMIT 20"),
    ("CAST_INFO role_id distribution",
     "SELECT r.role, ci.role_id, COUNT(ci.id) c FROM cast_info ci JOIN role_type r ON r.id=ci.role_id GROUP BY r.role, ci.role_id ORDER BY c DESC"),
    ("CAST_INFO notes (top patterns)",
     "SELECT note, COUNT(id) c FROM cast_info WHERE note IS NOT NULL AND note != '' GROUP BY note ORDER BY c DESC LIMIT 15"),
    ("PERSON_INFO info_type_id distribution",
     "SELECT it.info, pi.info_type_id, COUNT(pi.id) c FROM person_info pi JOIN info_type it ON it.id=pi.info_type_id GROUP BY it.info, pi.info_type_id ORDER BY c DESC"),
    ("TITLE production_year distribution",
     "SELECT production_year, COUNT(id) c FROM title WHERE kind_id=1 GROUP BY production_year ORDER BY production_year DESC LIMIT 20"),
    ("TITLE kind_id distribution",
     "SELECT kt.kind, t.kind_id, COUNT(t.id) c FROM title t JOIN kind_type kt ON kt.id=t.kind_id GROUP BY kt.kind, t.kind_id ORDER BY c DESC"),
    ("SAMPLE person names (top 15 actors by cast count)",
     "SELECT n.name, COUNT(ci.id) c FROM name n JOIN cast_info ci ON n.id=ci.person_id WHERE ci.role_id IN (1,2) GROUP BY n.name ORDER BY c DESC LIMIT 15"),
    ("SAMPLE company names (top 15 by movie count)",
     "SELECT cn.name, COUNT(mc.id) c FROM company_name cn JOIN movie_companies mc ON cn.id=mc.company_id GROUP BY cn.name ORDER BY c DESC LIMIT 15"),
    ("SAMPLE char_name (top 15 by cast count)",
     "SELECT cn.name, COUNT(ci.id) c FROM char_name cn JOIN cast_info ci ON cn.id=ci.person_role_id GROUP BY cn.name ORDER BY c DESC LIMIT 15"),
    ("SAMPLE movie titles (top 15 by cast count)",
     "SELECT t.title, COUNT(ci.id) c FROM title t JOIN cast_info ci ON t.id=ci.movie_id WHERE t.kind_id=1 GROUP BY t.title ORDER BY c DESC LIMIT 15"),
    ("MOVIE_INFO_IDX info_type_id distribution",
     "SELECT it.info, mii.info_type_id, COUNT(mii.id) c FROM movie_info_idx mii JOIN info_type it ON it.id=mii.info_type_id GROUP BY it.info, mii.info_type_id ORDER BY c DESC"),
    ("AKA_TITLE count", "SELECT COUNT(id) FROM aka_title"),
    ("AKA_NAME count", "SELECT COUNT(id) FROM aka_name"),
    ("MOVIE_LINK link_type distribution",
     "SELECT lt.link, ml.link_type_id, COUNT(ml.id) c FROM movie_link ml JOIN link_type lt ON lt.id=ml.link_type_id GROUP BY lt.link, ml.link_type_id ORDER BY c DESC"),
    ("COMPLETE_CAST subject_id and status_id distribution",
     "SELECT cct1.kind AS subject, cct2.kind AS status, COUNT(cc.id) c FROM complete_cast cc JOIN comp_cast_type cct1 ON cct1.id=cc.subject_id JOIN comp_cast_type cct2 ON cct2.id=cc.status_id GROUP BY cct1.kind, cct2.kind ORDER BY c DESC"),
    ("PERSON_INFO nationality values (info_type_id=201)",
     "SELECT info, COUNT(id) c FROM person_info WHERE info_type_id=201 GROUP BY info ORDER BY c DESC LIMIT 15"),
    ("PERSON_INFO bio contributor notes (info_type_id=19)",
     "SELECT note, COUNT(id) c FROM person_info WHERE info_type_id=19 GROUP BY note ORDER BY c DESC LIMIT 10"),
    ("MOVIE_INFO taglines sample (info_type_id=9)",
     "SELECT LEFT(info, 60), COUNT(id) c FROM movie_info WHERE info_type_id=9 GROUP BY LEFT(info, 60) ORDER BY c DESC LIMIT 10"),
    ("MOVIE_INFO release_dates sample (info_type_id=16)",
     "SELECT LEFT(info, 30), COUNT(id) c FROM movie_info WHERE info_type_id=16 GROUP BY LEFT(info, 30) ORDER BY c DESC LIMIT 15"),
]

for title, sql in sections:
    print(f"\n=== {title} ===")
    cur.execute(sql)
    for r in cur.fetchall():
        print(f"  {r}")

cur.close()
conn.close()
