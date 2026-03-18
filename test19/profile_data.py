"""Profile the v16 dataset to understand data distributions for query design."""
import psycopg2

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_v16', user='postgres', password='bench')
cur = conn.cursor()

def show(label, sql):
    cur.execute(sql)
    print(f'\n=== {label} ===')
    for r in cur.fetchall():
        if len(r) == 2:
            print(f'  {str(r[0]):30s} {r[1]:>8,}')
        else:
            print(f'  {r}')

show('GENRES', "SELECT mi.info, COUNT(*) FROM movie_info mi JOIN info_type it ON mi.info_type_id=it.id WHERE it.info='genres' GROUP BY mi.info ORDER BY COUNT(*) DESC LIMIT 15")
show('COUNTRIES', "SELECT mi.info, COUNT(*) FROM movie_info mi JOIN info_type it ON mi.info_type_id=it.id WHERE it.info='countries' GROUP BY mi.info ORDER BY COUNT(*) DESC LIMIT 10")
show('COMPANY COUNTRY CODES', "SELECT country_code, COUNT(*) FROM company_name GROUP BY country_code ORDER BY COUNT(*) DESC LIMIT 10")
show('ROLE TYPES', "SELECT rt.role, COUNT(*) FROM cast_info ci JOIN role_type rt ON ci.role_id=rt.id GROUP BY rt.role ORDER BY COUNT(*) DESC")
show('LINK TYPES', "SELECT lt.link, COUNT(*) FROM movie_link ml JOIN link_type lt ON ml.link_type_id=lt.id GROUP BY lt.link ORDER BY COUNT(*) DESC LIMIT 10")
show('CERTIFICATIONS', "SELECT mi.info, COUNT(*) FROM movie_info mi JOIN info_type it ON mi.info_type_id=it.id WHERE it.info='certificates' GROUP BY mi.info ORDER BY COUNT(*) DESC LIMIT 8")
show('LANGUAGES', "SELECT mi.info, COUNT(*) FROM movie_info mi JOIN info_type it ON mi.info_type_id=it.id WHERE it.info='languages' GROUP BY mi.info ORDER BY COUNT(*) DESC LIMIT 8")
show('COLOR INFO', "SELECT mi.info, COUNT(*) FROM movie_info mi JOIN info_type it ON mi.info_type_id=it.id WHERE it.info='color info' GROUP BY mi.info ORDER BY COUNT(*) DESC LIMIT 5")
show('COMPANY TYPES', "SELECT ct.kind, COUNT(*) FROM movie_companies mc JOIN company_type ct ON mc.company_type_id=ct.id GROUP BY ct.kind ORDER BY COUNT(*) DESC")
show('KIND TYPES', "SELECT kt.kind, COUNT(*) FROM title t JOIN kind_type kt ON t.kind_id=kt.id GROUP BY kt.kind ORDER BY COUNT(*) DESC")

# Year buckets
cur.execute("SELECT CASE WHEN production_year < 1980 THEN '1970-1979' WHEN production_year < 1990 THEN '1980-1989' WHEN production_year < 2000 THEN '1990-1999' WHEN production_year < 2010 THEN '2000-2009' WHEN production_year < 2020 THEN '2010-2019' ELSE '2020+' END AS decade, COUNT(*) FROM title WHERE kind_id=1 GROUP BY 1 ORDER BY 1")
print('\n=== YEAR BUCKETS (movies only) ===')
for r in cur.fetchall(): print(f'  {r[0]:15s} {r[1]:>8,}')

# Table sizes
print('\n=== TABLE SIZES ===')
for t in ['title','name','cast_info','movie_info','movie_info_idx','movie_keyword','movie_companies','person_info','aka_name','aka_title','movie_link','complete_cast','keyword','company_name','char_name']:
    cur.execute(f'SELECT COUNT(*) FROM "{t}"')
    print(f'  {t:20s} {cur.fetchone()[0]:>10,}')

# person_info types
show('PERSON_INFO TYPES', "SELECT it.info, COUNT(*) FROM person_info pi JOIN info_type it ON pi.info_type_id=it.id GROUP BY it.info ORDER BY COUNT(*) DESC LIMIT 10")

# aka_name stats
cur.execute("SELECT COUNT(DISTINCT person_id) FROM aka_name")
print(f'\n=== AKA stats ===')
print(f'  aka_name distinct persons: {cur.fetchone()[0]:,}')
cur.execute("SELECT COUNT(DISTINCT movie_id) FROM aka_title")
print(f'  aka_title distinct movies: {cur.fetchone()[0]:,}')

# Cast per movie stats
cur.execute("SELECT MIN(cnt), AVG(cnt)::int, MAX(cnt) FROM (SELECT movie_id, COUNT(*) cnt FROM cast_info GROUP BY movie_id) sub")
r = cur.fetchone()
print(f'\n=== CAST PER MOVIE ===')
print(f'  min={r[0]}, avg={r[1]}, max={r[2]}')

# Keywords per movie
cur.execute("SELECT AVG(cnt)::int, MAX(cnt) FROM (SELECT movie_id, COUNT(*) cnt FROM movie_keyword GROUP BY movie_id) sub")
r = cur.fetchone()
print(f'  keywords per movie: avg={r[0]}, max={r[1]}')

cur.close(); conn.close()
