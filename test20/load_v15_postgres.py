"""Load V15 data into Postgres benchmark_v15 database."""
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

import psycopg2

from pathlib import Path
DATA_DIR = str(Path(__file__).resolve().parent)

# â”€â”€ Create / replace database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Connecting to postgres...", flush=True)
conn0 = psycopg2.connect(host="localhost", port=5433, dbname="postgres",
                         user="postgres", password="bench")
conn0.autocommit = True
cur0 = conn0.cursor()
cur0.execute("SELECT 1 FROM pg_database WHERE datname='benchmark_v15'")
if cur0.fetchone():
    print("  Dropping existing benchmark_v15...", flush=True)
    cur0.execute("DROP DATABASE benchmark_v15")
cur0.execute("CREATE DATABASE benchmark_v15")
print("  Created benchmark_v15.", flush=True)
cur0.close(); conn0.close()

# â”€â”€ Connect to new database â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
conn = psycopg2.connect(host="localhost", port=5433, dbname="benchmark_v15",
                        user="postgres", password="bench")
conn.autocommit = True
cur = conn.cursor()

# CSV files to load (table_name, path_relative_to_DATA_DIR)
FILES = [
    # Core movie / person tables
    ("movie",                    "movie.csv"),
    ("cast_info",                "cast_info.csv"),
    ("movie_directors",          "movie_directors.csv"),
    ("movie_companies",          "movie_companies.csv"),
    ("movie_keyword",            "movie_keyword.csv"),
    ("movie_crew",               "movie_crew.csv"),
    ("awards",                   "awards.csv"),
    ("release_dates",            "release_dates.csv"),
    ("box_office_weekly",        "box_office_weekly.csv"),
    ("box_office_by_territory",  "box_office_by_territory.csv"),
    ("box_office_daily",         "box_office_daily.csv"),
    ("reviews",                  "reviews.csv"),
    ("locations",                "locations.csv"),
    ("alternate_titles",         "alternate_titles.csv"),
    ("ratings_breakdown",        "ratings_breakdown.csv"),
    ("movie_links",              "movie_links.csv"),
    ("person_demographics",      "person_demographics.csv"),
    ("movie_sequence",           "movie_sequence.csv"),
    ("person_collaborations",    "person_collaborations.csv"),
    ("media_links",              "media_links.csv"),
    # TV
    ("tv_series",                "tv_series.csv"),
    ("seasons",                  "seasons.csv"),
    ("episodes",                 "episodes.csv"),
    ("episode_cast",             "episode_cast.csv"),
    # Company / graph / extras
    ("company_links",            "company_links.csv"),
    ("world_events",             "world_events.csv"),
    ("production_timeline",      "production_timeline.csv"),
    ("streaming_windows",        "streaming_windows.csv"),
    ("person_contracts",         "person_contracts.csv"),
    # Persons / entities
    ("persons",                  "persons_enriched.csv"),
    ("keyword",                  "entities/keyword.csv"),
    ("company",                  "entities/company.csv"),
    # Social graph
    ("edges",                    "graph/edge_graph.csv"),
]

ok = 0
for table_name, csv_rel in FILES:
    csv_path = os.path.join(DATA_DIR, csv_rel)
    if not os.path.exists(csv_path):
        print(f"  SKIP {table_name}: not found ({csv_rel})", flush=True)
        continue

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().strip()
    if not header_line:
        print(f"  SKIP {table_name}: empty", flush=True)
        continue

    cols = [h.strip().strip('"') for h in header_line.split(",")]

    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
    col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
    cur.execute(f"CREATE TABLE {table_name} ({col_defs})")

    col_list = ", ".join(f'"{c}"' for c in cols)
    copy_sql = f"COPY {table_name} ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER, NULL '')"
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            cur.copy_expert(copy_sql, f)
        cur.execute(f"SELECT count(*) FROM {table_name}")
        n = cur.fetchone()[0]
        print(f"  OK  {table_name:<30} {n:>10,} rows", flush=True)
        ok += 1
    except Exception as ex:
        conn.rollback()
        conn.autocommit = True
        print(f"  ERR {table_name}: {str(ex)[:120]}", flush=True)

print(f"\nLoaded {ok}/{len(FILES)} tables.", flush=True)

# â”€â”€ Cast numeric columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nCasting numeric columns...", flush=True)
CASTS = [
    ("movie",          [("title_id","INT"), ("year","INT"), ("budget_usd","BIGINT"),
                        ("box_office_usd","BIGINT"), ("runtime_minutes","INT"),
                        ("rating","FLOAT"), ("num_votes","INT"), ("seed","INT"),
                        ("award_campaign_strength","FLOAT")]),
    ("cast_info",      [("title_id","INT"), ("person_id","INT"), ("billing_order","INT")]),
    ("movie_directors",[("title_id","INT"), ("director_id","INT")]),
    ("movie_companies",[("title_id","INT"), ("company_id","INT")]),
    ("movie_keyword",  [("title_id","INT"), ("keyword_id","INT")]),
    ("awards",         [("award_id","INT"), ("title_id","INT"), ("award_year","INT"), ("person_id","INT")]),
    ("edges",          [("src_id","INT"), ("dst_id","INT"), ("weight","FLOAT"),
                        ("valid_from","INT"), ("valid_to","INT")]),
    ("persons",        [("person_id","INT"), ("debut_year","INT"), ("peak_start","INT"),
                        ("peak_end","INT"), ("retirement_year","INT"),
                        ("yearly_max","INT"), ("pop_weight","FLOAT")]),
    ("box_office_daily",[("daily_id","INT"), ("title_id","INT"), ("day_number","INT")]),
    ("reviews",        [("review_id","INT"), ("title_id","INT"), ("rating_10","FLOAT")]),
    ("ratings_breakdown",[("breakdown_id","INT"), ("title_id","INT"),
                          ("vote_count","INT"), ("avg_rating","FLOAT")]),
    ("tv_series",      [("series_id","INT"), ("year_start","INT"), ("year_end","INT"),
                        ("total_seasons","INT"), ("overall_rating","FLOAT")]),
    ("seasons",        [("season_id","INT"), ("series_id","INT"), ("season_number","INT"),
                        ("year","INT"), ("num_episodes","INT"), ("avg_rating","FLOAT")]),
    ("episodes",       [("episode_id","INT"), ("season_id","INT"), ("series_id","INT"),
                        ("episode_number","INT"), ("runtime_minutes","INT"),
                        ("rating","FLOAT"), ("viewership_millions","FLOAT")]),
    ("episode_cast",   [("episode_cast_id","INT"), ("episode_id","INT"), ("series_id","INT"),
                        ("person_id","INT"), ("credit_order","INT")]),
    ("world_events",   [("event_id","INT"), ("year","INT"), ("duration_years","INT")]),
    ("production_timeline",[("timeline_id","INT"), ("movie_id","INT")]),
    ("streaming_windows",[("window_id","INT"), ("movie_id","INT")]),
    ("person_contracts",[("contract_id","INT"), ("person_id","INT"), ("company_id","INT")]),
    ("person_collaborations",[("person_a_id","INT"), ("person_b_id","INT"),
                              ("collaboration_count","INT"), ("first_year","INT"), ("last_year","INT")]),
    ("keyword",        [("keyword_id","INT")]),
    ("company",        [("company_id","INT")]),
    ("locations",      [("location_id","INT"), ("title_id","INT"), ("location_order","INT")]),
    ("release_dates",  [("release_id","INT"), ("title_id","INT")]),
    ("movie_crew",     [("crew_id","INT"), ("title_id","INT"), ("person_id","INT")]),
]

for table, cols in CASTS:
    for col, typ in cols:
        try:
            cur.execute(
                f'ALTER TABLE {table} ALTER COLUMN "{col}" TYPE {typ} '
                f"USING NULLIF(\"{col}\",'')::{typ}"
            )
        except Exception:
            conn.rollback()
            conn.autocommit = True
    print(f"  cast {table}", flush=True)

# â”€â”€ Indexes for benchmark query patterns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nCreating indexes...", flush=True)
INDEXES = [
    "CREATE INDEX ON movie(year)",
    "CREATE INDEX ON movie(genre)",
    "CREATE INDEX ON movie(rating)",
    "CREATE INDEX ON cast_info(title_id)",
    "CREATE INDEX ON cast_info(person_id)",
    "CREATE INDEX ON movie_directors(title_id)",
    "CREATE INDEX ON movie_directors(director_id)",
    "CREATE INDEX ON movie_companies(title_id)",
    "CREATE INDEX ON movie_companies(company_id)",
    "CREATE INDEX ON movie_keyword(title_id)",
    "CREATE INDEX ON movie_keyword(keyword_id)",
    "CREATE INDEX ON awards(title_id)",
    "CREATE INDEX ON awards(person_id)",
    "CREATE INDEX ON reviews(title_id)",
    "CREATE INDEX ON box_office_daily(title_id)",
    "CREATE INDEX ON edges(src_id)",
    "CREATE INDEX ON edges(dst_id)",
    "CREATE INDEX ON persons(person_id)",
    "CREATE INDEX ON episode_cast(person_id)",
    "CREATE INDEX ON episode_cast(series_id)",
]
for idx in INDEXES:
    try:
        cur.execute(idx)
        print(f"  {idx[:70]}", flush=True)
    except Exception as ex:
        print(f"  ERR idx: {ex}", flush=True)
        conn.rollback(); conn.autocommit = True

print("\nANALYZE...", flush=True)
cur.execute("ANALYZE")
print("Done!", flush=True)
cur.close()
conn.close()


