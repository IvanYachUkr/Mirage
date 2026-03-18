"""
Regenerate secondary tables (demographics, awards, box_office_daily)
from existing movie/cast/crew data using the FIXED secondary_tables.py.
Run this instead of a full re-run to apply code fixes without a 1-hour wait.
"""
import os, sys, pandas as pd, numpy as np

from pathlib import Path
BASE = str(Path(__file__).parent)
sys.path.insert(0, BASE)

from secondary_tables import (
    generate_person_demographics, generate_awards,
    generate_box_office_daily, generate_box_office_weekly,
)
from world_state import WorldState
from contracts import SNAPSHOT_CONFIG

print("Loading world state...")
world = WorldState(BASE, seed=SNAPSHOT_CONFIG["seed"])
world.load()
rng = world.rng

# ─── 1. Regenerate person_demographics.csv ───────────────────────────
print("\n[B05/B06] Regenerating person_demographics.csv...")
persons_df = pd.read_csv(os.path.join(BASE, "persons_enriched.csv"))
movies = pd.read_csv(os.path.join(BASE, "movie.csv"))
cast_info = pd.read_csv(os.path.join(BASE, "cast_info.csv"))
movie_crew = pd.read_csv(os.path.join(BASE, "movie_crew.csv"))
movie_directors = pd.read_csv(os.path.join(BASE, "movie_directors.csv"))

# B05 fix: compute earliest movie year per person so birth_year is anchored
# to their actual first film appearance, not the LLM-assigned debut_year.
# Without this, a "rising" actor with debut_year=2017 who directed a 1970 film
# gets birth_year=1995 -> age = 1970-1995 = -25 -> impossible.
year_by_movie = movies.set_index("title_id")["year"].to_dict()

earliest_year = {}  # person_id -> earliest movie year they appeared in
latest_year = {}   # person_id -> latest movie year they appeared in (for age>105 guard)

def _update_year(pid, yr):
    if yr is not None:
        if pid not in earliest_year or yr < earliest_year[pid]:
            earliest_year[pid] = int(yr)
        if pid not in latest_year or yr > latest_year[pid]:
            latest_year[pid] = int(yr)

# From cast_info
for _, row in cast_info.iterrows():
    _update_year(int(row["person_id"]), year_by_movie.get(int(row["title_id"])))

# From movie_directors
for _, row in movie_directors.iterrows():
    _update_year(int(row["director_id"]), year_by_movie.get(int(row["title_id"])))

# From movie_crew
if "person_id" in movie_crew.columns:
    for _, row in movie_crew.iterrows():
        _update_year(int(row["person_id"]), year_by_movie.get(int(row["title_id"])))

print(f"  Earliest movie year computed for {len(earliest_year)} persons.")

# Add earliest_year_in_film to persons_df so the generator can use it
persons_df["_earliest_film_year"] = persons_df["person_id"].map(earliest_year)

rng_demo = np.random.RandomState(SNAPSHOT_CONFIG["seed"] + 1)
demo_rows = []

for _, p in persons_df.iterrows():
    pid = int(p["person_id"])
    nationality = str(p.get("nationality", "American"))
    career_stage = str(p.get("career_stage", "prime"))
    gender = str(p.get("gender", "M"))

    # Determine debut year: use earliest actual film year OR LLM debut_year
    earliest_film = p.get("_earliest_film_year")
    lm_debut = None
    for col in ("debut_year", "peak_start", "career_start_year"):
        v = p.get(col)
        if v is not None and str(v) not in ("", "nan", "None"):
            try:
                lm_debut = int(float(v))
                break
            except (ValueError, TypeError):
                pass

    # Use whichever is EARLIER to ensure the person is old enough at first appearance
    earliest_film_valid = earliest_film if (earliest_film is not None and not (isinstance(earliest_film, float) and np.isnan(earliest_film))) else None
    if earliest_film_valid is not None and lm_debut is not None:
        debut_year = min(int(earliest_film_valid), int(lm_debut))
    elif earliest_film_valid is not None:
        debut_year = int(earliest_film_valid)
    elif lm_debut is not None:
        debut_year = int(lm_debut)
    else:
        # Fallback: estimate from career stage
        stage_offsets = {"rising": -2, "prime": -8, "veteran": -20, "legend": -30, "retired": -35}
        debut_year = 2018 + stage_offsets.get(career_stage, -5)

    # Birth year: debut - age at debut for this career stage
    _CAREER_STAGE_AGE_RANGES = {
        "rising": (22, 32), "prime": (28, 45),
        "veteran": (45, 65), "legend": (55, 80), "retired": (60, 85),
    }
    age_lo, age_hi = _CAREER_STAGE_AGE_RANGES.get(career_stage, (28, 45))
    age = int(rng_demo.randint(age_lo, age_hi + 1))
    birth_year = debut_year - age

    # Hard guard: age must be 16-104 at BOTH earliest and latest film appearance
    # Lower bound: birth_year >= latest_film_year - 104 (not impossibly old at last film)
    # Upper bound: birth_year <= earliest_film_year - 16 (old enough at first film)
    eff_latest = latest_year.get(pid, debut_year) or debut_year
    min_birth_year = max(debut_year - 90, int(eff_latest) - 104)
    max_birth_year = debut_year - 16
    if min_birth_year > max_birth_year:
        min_birth_year = max_birth_year - 10
    birth_year = int(np.clip(birth_year, min_birth_year, max_birth_year))

    birth_month = int(rng_demo.randint(1, 13))
    birth_day = int(rng_demo.randint(1, 29))
    birth_date = f"{birth_year:04d}-{birth_month:02d}-{birth_day:02d}"

    # Death date (rare -- ~3% of legends/retired, must be 2020+)
    death_date = None
    if career_stage in ("legend", "retired") and rng_demo.random() < 0.03:
        death_age = int(rng_demo.randint(max(age, 65), 95))
        death_year = birth_year + death_age
        death_year = max(death_year, debut_year + 40, 2020)
        if death_year <= 2025:
            dm = int(rng_demo.randint(1, 13))
            dd_day = int(rng_demo.randint(1, 29))
            death_date = f"{death_year:04d}-{dm:02d}-{dd_day:02d}"

    # Birth place
    from secondary_tables import _NATIONALITY_BIRTH_COUNTRIES, _BIRTH_CITY_BANKS
    birth_countries = _NATIONALITY_BIRTH_COUNTRIES.get(nationality, ["USA"])
    birth_country = str(rng_demo.choice(birth_countries))
    cities = _BIRTH_CITY_BANKS.get(birth_country, ["Unknown"])
    birth_city = str(rng_demo.choice(cities))

    if gender == "F":
        height = int(np.clip(rng_demo.normal(167, 6), 150, 188))
    elif gender == "NB":
        height = int(np.clip(rng_demo.normal(172, 7), 152, 195))
    else:
        height = int(np.clip(rng_demo.normal(178, 7), 158, 200))

    demo_rows.append({
        "person_id": pid, "nationality": nationality,
        "birth_date": birth_date, "death_date": death_date,
        "birth_city": birth_city, "birth_country": birth_country,
        "height_cm": height,
    })

demo_df = pd.DataFrame(demo_rows)
demo_df.to_csv(os.path.join(BASE, "person_demographics.csv"), index=False)
print(f"  Saved {len(demo_df)} rows.")
print(f"  Death dates assigned: {demo_df['death_date'].notna().sum()}")
demo_df['birth_year'] = pd.to_datetime(demo_df['birth_date'], errors='coerce').dt.year
print(f"  Birth year range: {demo_df['birth_year'].min()} - {demo_df['birth_year'].max()}")


# ─── 2. Regenerate awards.csv ────────────────────────────────────────
print("\n[B02] Regenerating awards.csv...")
movies = pd.read_csv(os.path.join(BASE, "movie.csv"))
cast_info = pd.read_csv(os.path.join(BASE, "cast_info.csv"))
movie_crew = pd.read_csv(os.path.join(BASE, "movie_crew.csv"))
movie_directors = pd.read_csv(os.path.join(BASE, "movie_directors.csv"))

# Build cast and crew per movie
cast_by_movie = cast_info.groupby("title_id").apply(
    lambda g: g[["person_id", "billing_order", "archetype", "character_name"]].to_dict("records")
).to_dict()
crew_by_movie = movie_crew.groupby("title_id").apply(
    lambda g: g[["person_id", "crew_role"]].to_dict("records")
).to_dict()
directors_by_movie = movie_directors.groupby("title_id")["director_id"].first().to_dict()

award_rng = np.random.RandomState(SNAPSHOT_CONFIG["seed"] + 42)
award_rows = []
for _, row in movies.iterrows():
    tid = int(row["title_id"])
    year = int(row.get("year", 2000))
    rating = float(row.get("rating", 6.0))
    tier = str(row.get("production_tier", "Mid"))
    director_id = directors_by_movie.get(tid)
    cast = cast_by_movie.get(tid, [])
    crew = crew_by_movie.get(tid, [])
    rows = generate_awards(
        title_id=tid, year=year, rating=rating, tier=tier,
        director_id=director_id, cast=cast, crew_rows=crew,
        rng=award_rng, world=world,
    )
    award_rows.extend(rows)

awards_df = pd.DataFrame(award_rows)
if not awards_df.empty:
    awards_df.insert(0, "award_id", range(1, len(awards_df) + 1))
awards_df.to_csv(os.path.join(BASE, "awards.csv"), index=False)
print(f"  Saved {len(awards_df)} award rows.")

# Quick validation
if not awards_df.empty:
    total_with_pid = awards_df["person_id"].notna().sum()
    # Check none of those are in crew vs in cast
    if total_with_pid > 0:
        aw_cast = awards_df.dropna(subset=["person_id"]).merge(
            cast_info[["title_id", "person_id"]], on=["title_id", "person_id"], how="left", indicator=True
        )
        in_cast = (aw_cast["_merge"] == "both").sum()
        total = len(aw_cast)
        print(f"  Nominees in cast: {in_cast}/{total} ({100*in_cast/total:.1f}%)")

# ─── 3. Regenerate box_office_daily.csv (B03: column order fix) ──────
print("\n[B03] Regenerating box_office_daily.csv...")
release_dates = pd.read_csv(os.path.join(BASE, "release_dates.csv"))
theatrical = release_dates[release_dates["release_type"] == "Theatrical"][["title_id", "release_date"]].drop_duplicates("title_id")
rd_by_movie = theatrical.set_index("title_id")["release_date"].to_dict()

bo_rng = np.random.RandomState(SNAPSHOT_CONFIG["seed"] + 7)
daily_rows = []
weekly_rows = []
for _, row in movies.iterrows():
    tid = int(row["title_id"])
    bo = float(row.get("box_office_usd", 0) or 0)
    base_date = rd_by_movie.get(tid, f"{int(row.get('year', 2000))}-06-01")
    d_rows = generate_box_office_daily(
        title_id=tid, total_box_office_usd=bo,
        base_release_date=base_date, rng=bo_rng
    )
    daily_rows.extend(d_rows)
    w_rows = generate_box_office_weekly(
        title_id=tid, total_box_office_usd=bo,
        base_release_date=base_date, rng=bo_rng,
        daily_rows=d_rows,  # D29: derive weekly from daily
    )
    weekly_rows.extend(w_rows)

daily_df = pd.DataFrame(daily_rows)
daily_df.to_csv(os.path.join(BASE, "box_office_daily.csv"), index=False)
print(f"  Saved {len(daily_df)} daily rows.")
if not daily_df.empty:
    print(f"  Daily columns: {daily_df.columns.tolist()[:6]} (gross_usd_total should be first)")

weekly_df = pd.DataFrame(weekly_rows)
weekly_df.to_csv(os.path.join(BASE, "box_office_weekly.csv"), index=False)
print(f"  Saved {len(weekly_df)} weekly rows.")

print("\nDone. Run tests now to verify fixes.")
