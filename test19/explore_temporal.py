"""Explore temporal data across IMDB tables for designing temporal queries."""
import psycopg2, sys, io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')
cur = conn.cursor()

# production_year distribution
cur.execute("SELECT MIN(production_year), MAX(production_year), AVG(production_year)::int FROM title WHERE production_year IS NOT NULL")
r = cur.fetchone()
print(f"title.production_year: {r[0]} - {r[1]} (avg {r[2]})")

cur.execute("SELECT production_year, COUNT(*) FROM title WHERE production_year IS NOT NULL GROUP BY production_year ORDER BY production_year")
print("\nYear distribution:")
for y, c in cur.fetchall():
    print(f"  {y}: {c}")

# release dates info
cur.execute("""SELECT mi.info, COUNT(*) FROM movie_info mi 
              JOIN info_type it ON it.id = mi.info_type_id 
              WHERE it.info = 'release dates' 
              GROUP BY mi.info ORDER BY COUNT(*) DESC LIMIT 10""")
print("\nRelease date values (top 10):")
for r in cur.fetchall(): print(f"  {r}")

# release date notes
cur.execute("""SELECT mi.note, COUNT(*) FROM movie_info mi 
              JOIN info_type it ON it.id = mi.info_type_id 
              WHERE it.info = 'release dates' 
              GROUP BY mi.note ORDER BY COUNT(*) DESC LIMIT 10""")
print("\nRelease date notes (top 10):")
for r in cur.fetchall(): print(f"  {r}")

# birth date info
cur.execute("""SELECT pi.info, COUNT(*) FROM person_info pi 
              JOIN info_type it ON it.id = pi.info_type_id 
              WHERE it.info = 'birth date' 
              GROUP BY pi.info ORDER BY COUNT(*) DESC LIMIT 10""")
print("\nBirth date values (top 10):")
for r in cur.fetchall(): print(f"  {r}")

# How many have birth date?
cur.execute("""SELECT COUNT(*) FROM person_info pi 
              JOIN info_type it ON it.id = pi.info_type_id 
              WHERE it.info = 'birth date'""")
print(f"Total people with birth date: {cur.fetchone()[0]}")

# death date
cur.execute("""SELECT COUNT(*) FROM person_info pi 
              JOIN info_type it ON it.id = pi.info_type_id 
              WHERE it.info = 'death date'""")
print(f"Total people with death date: {cur.fetchone()[0]}")

# Extract year from birth date sample 
cur.execute("""SELECT pi.info FROM person_info pi 
              JOIN info_type it ON it.id = pi.info_type_id
              WHERE it.info = 'birth date' LIMIT 10""")
print("\nBirth date samples:")
for r in cur.fetchall(): print(f"  {r}")

# aka_title.production_year
cur.execute("SELECT MIN(production_year), MAX(production_year) FROM aka_title WHERE production_year IS NOT NULL")
print(f"\naka_title.production_year range: {cur.fetchone()}")

# title.series_years
cur.execute("SELECT series_years, COUNT(*) FROM title WHERE series_years IS NOT NULL GROUP BY series_years ORDER BY COUNT(*) DESC LIMIT 10")
print("\nSeries years (top 10):")
for r in cur.fetchall(): print(f"  {r}")

# Runtimes distribution
cur.execute("""SELECT mi.info, COUNT(*) FROM movie_info mi 
              JOIN info_type it ON it.id = mi.info_type_id 
              WHERE it.info = 'runtimes' 
              GROUP BY mi.info ORDER BY COUNT(*) DESC LIMIT 10""")
print("\nRuntimes values (top 10):")
for r in cur.fetchall(): print(f"  {r}")

# How many titles per kind
cur.execute("""SELECT kt.kind, COUNT(*) FROM title t 
              JOIN kind_type kt ON kt.id = t.kind_id 
              GROUP BY kt.kind ORDER BY COUNT(*) DESC""")
print("\nTitle kinds:")
for r in cur.fetchall(): print(f"  {r}")

# episode_of_id distribution
cur.execute("SELECT COUNT(DISTINCT episode_of_id) FROM title WHERE episode_of_id IS NOT NULL")
print(f"\nDistinct episode_of_id parents: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM title WHERE episode_of_id IS NOT NULL")
print(f"Total episodes: {cur.fetchone()[0]}")

conn.close()
