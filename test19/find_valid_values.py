"""Find valid value combinations for fixing empty JOB queries."""
import psycopg2, sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

conn = psycopg2.connect(host='localhost', port=5433, dbname='imdb_benchmark', user='postgres', password='bench')
cur = conn.cursor()

queries = {
    # What link types exist?
    "link_types": "SELECT link, COUNT(id) FROM link_type GROUP BY link ORDER BY COUNT(id) DESC",
    # Any mc.note IS NULL?
    "mc_null_notes": "SELECT COUNT(id) FROM movie_companies WHERE note IS NULL OR note = ''",
    # mc.note distinct values (top 30)
    "mc_notes_sample": "SELECT note, COUNT(id) c FROM movie_companies GROUP BY note ORDER BY c DESC LIMIT 30",
    # mi.note values for release dates
    "mi_notes_release": "SELECT note, COUNT(id) c FROM movie_info WHERE info_type_id=16 AND note IS NOT NULL GROUP BY note ORDER BY c DESC LIMIT 10",
    # mi.note values for all types
    "mi_notes_all": "SELECT note, COUNT(id) c FROM movie_info WHERE note IS NOT NULL AND note != '' GROUP BY note ORDER BY c DESC LIMIT 20",
    # Any mi with 'internet' in note?
    "mi_internet": "SELECT COUNT(id) FROM movie_info WHERE note LIKE '%internet%'",
    # What movie_info notes look like
    "mi_notes_sample": "SELECT DISTINCT note FROM movie_info WHERE note IS NOT NULL LIMIT 20",
    # Keywords we have
    "all_keywords": "SELECT k.keyword, COUNT(mk.id) c FROM keyword k JOIN movie_keyword mk ON k.id=mk.keyword_id GROUP BY k.keyword ORDER BY c DESC LIMIT 50",
    # Do we have 'sequel' keyword?
    "sequel_keyword": "SELECT k.keyword, COUNT(mk.id) FROM keyword k JOIN movie_keyword mk ON k.id=mk.keyword_id WHERE k.keyword IN ('sequel','revenge','based-on-novel','character-name-in-title','superhero','computer-animation','hero','martial-arts','hand-to-hand-combat','nerd','loner','alienation','gore') GROUP BY k.keyword",
    # n.name_pcode_cf values
    "name_pcode_cf": "SELECT name_pcode_cf, COUNT(id) c FROM name WHERE name_pcode_cf IS NOT NULL GROUP BY name_pcode_cf ORDER BY c DESC LIMIT 20",
    # What company has 'Studios' or 'Pictures' in name + [us] code?
    "companies_with_studios": "SELECT cn.name, cn.country_code, COUNT(mc.id) c FROM company_name cn JOIN movie_companies mc ON cn.id=mc.company_id WHERE cn.name LIKE '%Studios%' OR cn.name LIKE '%Pictures%' GROUP BY cn.name, cn.country_code ORDER BY c DESC LIMIT 15",
    # Titles with common words
    "titles_broken": "SELECT title FROM title WHERE title LIKE '%Broken%' AND kind_id=1 LIMIT 10",
    "titles_storm": "SELECT title FROM title WHERE title LIKE '%Storm%' AND kind_id=1 LIMIT 10",
    "titles_silent": "SELECT title FROM title WHERE title LIKE '%Silent%' AND kind_id=1 LIMIT 10",
    # Actresses with voice roles
    "voice_actresses": "SELECT n.name, COUNT(ci.id) c FROM name n JOIN cast_info ci ON n.id=ci.person_id WHERE ci.note='(voice)' AND n.gender='f' GROUP BY n.name ORDER BY c DESC LIMIT 15",
    # Char names with common words
    "chars_shadow": "SELECT name FROM char_name WHERE name LIKE '%Shadow%' LIMIT 10",
    "chars_colonel": "SELECT name FROM char_name WHERE name LIKE '%Colonel%' LIMIT 10",
    # ci.note values
    "ci_notes": "SELECT note, COUNT(id) c FROM cast_info WHERE note IS NOT NULL GROUP BY note ORDER BY c DESC LIMIT 20",
    # episode_nr max
    "episode_nr_max": "SELECT MAX(episode_nr), MIN(episode_nr), AVG(episode_nr) FROM title WHERE episode_nr IS NOT NULL",
    "episode_nr_dist": "SELECT episode_nr, COUNT(id) FROM title WHERE episode_nr IS NOT NULL GROUP BY episode_nr ORDER BY episode_nr LIMIT 30",
    # Budget info
    "budget_info": "SELECT info, COUNT(id) c FROM movie_info WHERE info_type_id=105 GROUP BY info ORDER BY c DESC LIMIT 10",
    # trivia info
    "trivia_info": "SELECT COUNT(id) FROM movie_info WHERE info_type_id=17",
    # Vampire title?
    "vampire_titles": "SELECT title FROM title WHERE title LIKE '%Vampire%' OR title LIKE '%Dragon%' OR title LIKE '%Monster%' LIMIT 10",
    # Person names with common patterns
    "persons_y": "SELECT name FROM name WHERE name LIKE '%Yo%' AND gender='f' LIMIT 5",
    # Country codes in company
    "country_codes": "SELECT country_code, COUNT(id) c FROM company_name GROUP BY country_code ORDER BY c DESC LIMIT 20",
    # Movies with [jp] companies
    "jp_companies": "SELECT cn.name, COUNT(mc.id) c FROM company_name cn JOIN movie_companies mc ON cn.id=mc.company_id WHERE cn.country_code='[jp]' GROUP BY cn.name ORDER BY c DESC LIMIT 10",
}

for name, sql in queries.items():
    print(f"\n=== {name} ===")
    cur.execute(sql)
    for r in cur.fetchall():
        print(f"  {r}")

cur.close()
conn.close()
