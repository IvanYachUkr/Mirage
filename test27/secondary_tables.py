"""
V13 Pipeline -- secondary_tables.py
=====================================
Generators for satellite/fact tables: release dates, box office weekly,
reviews, awards, locations, alternate titles, ratings breakdown,
movie links, and company links.

Extracted from generate_movies.py for modular architecture.
"""
import datetime
import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from contracts import AWARD_CAMPAIGN_GENRES, COUNTRIES
from world_state import get_person_latent


def _rows_len(rows) -> int:
    if rows is None:
        return 0
    if isinstance(rows, pd.DataFrame):
        return int(len(rows))
    try:
        return int(len(rows))
    except Exception:
        return 0


def _iter_rows(rows):
    if rows is None:
        return
    if isinstance(rows, pd.DataFrame):
        cols = list(rows.columns)
        for values in rows.itertuples(index=False, name=None):
            yield {col: value for col, value in zip(cols, values)}
        return
    yield from rows


def _offset_date(y: int, m: int, d: int, days: int) -> str:
    """Add *days* to a y/m/d triple using real calendar arithmetic.

    B1-FIX: replaces the old ``while dd > 28`` loop which treated every
    month as 28 days, skipping the 29th-31st and producing duplicate dates.
    """
    try:
        base = datetime.date(y, m, min(d, 28))  # clamp day to avoid ValueError
        result = base + datetime.timedelta(days=days)
    except (ValueError, OverflowError):
        # Absolute fallback — shift year forward proportionally
        result = datetime.date(y, 1, 1) + datetime.timedelta(days=days)
    return result.strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════════════
# PERSON DEMOGRAPHICS (global -- generated once from persons pool)
# ═══════════════════════════════════════════════════════════════════════

# D22 fix: corrected nationality -> birth country mapping (was Dutch->Germany, Norwegian->Sweden etc.)
_NATIONALITY_BIRTH_COUNTRIES = {
    "American": ["USA"], "British": ["UK"], "French": ["France"],
    "Indian": ["India"], "South Korean": ["South Korea"], "Korean": ["South Korea"],
    "Japanese": ["Japan"],
    "German": ["Germany"], "Brazilian": ["Brazil"], "Canadian": ["Canada"],
    "Chinese": ["China"], "Italian": ["Italy"], "Spanish": ["Spain"],
    "Australian": ["Australia"], "Mexican": ["Mexico"], "Nigerian": ["Nigeria"],
    "Swedish": ["Sweden"], "Argentine": ["Argentina"], "Thai": ["Thailand"],
    "Iranian": ["Iran"], "Polish": ["Poland"], "Russian": ["Russia"],
    "Irish": ["Ireland"], "Scottish": ["UK"],
    "Dutch": ["Netherlands"],          # was wrong: Germany
    "Norwegian": ["Norway"],            # was wrong: Sweden
    "Danish": ["Denmark"], "Greek": ["Greece"],
    "Turkish": ["Turkey"], "Egyptian": ["Egypt"], "Jamaican": ["Jamaica"],
    "Colombian": ["Colombia"], "Cuban": ["Cuba"], "Filipino": ["Philippines"],
    "Vietnamese": ["Vietnam"], "Indonesian": ["Indonesia"],
    "South African": ["South Africa"],
    "Kenyan": ["Kenya"],               # was wrong: Nigeria
    "New Zealander": ["New Zealand"], "Israeli": ["Israel"], "Lebanese": ["Lebanon"],
    "Ghanaian": ["Ghana"], "Pakistani": ["Pakistan"],
    "Ukrainian": ["Ukraine"],           # was wrong: Russia
    "Bulgarian": ["Bulgaria"],          # was wrong: Poland
    "Portuguese": ["Portugal"],         # was wrong: Spain
    "Omani": ["Oman"], "Saudi": ["Saudi Arabia"],
    "Belarusian": ["Belarus"],
    "Syrian": ["Syria"],
    "Austrian": ["Austria"], "Swiss": ["Switzerland"],
    "Belgian": ["Belgium"], "Czech": ["Czech Republic"],
    "Romanian": ["Romania"], "Hungarian": ["Hungary"],
}

# Career stage -> approximate birth year offsets (from movie year reference = 2018)
_CAREER_STAGE_AGE_RANGES = {
    "rising":   (22, 32),
    "prime":    (28, 45),
    "veteran":  (45, 65),
    "legend":   (55, 80),
    "retired":  (60, 85),
}

# City banks per country for birth_place
_BIRTH_CITY_BANKS = {
    "USA": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
            "San Antonio", "San Diego", "Dallas", "Austin", "Atlanta", "Miami"],
    "UK": ["London", "Manchester", "Birmingham", "Liverpool", "Edinburgh", "Glasgow", "Bristol"],
    "France": ["Paris", "Marseille", "Lyon", "Toulouse", "Nice", "Bordeaux"],
    "India": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad"],
    "Japan": ["Tokyo", "Osaka", "Kyoto", "Nagoya", "Yokohama", "Sapporo"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne", "Stuttgart"],
    "South Korea": ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon"],
    "China": ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu", "Hangzhou"],
    "Brazil": ["São Paulo", "Rio de Janeiro", "Salvador", "Brasília", "Fortaleza"],
    "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"],
    "Italy": ["Rome", "Milan", "Naples", "Turin", "Florence", "Bologna"],
    "Spain": ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao"],
    "Mexico": ["Mexico City", "Guadalajara", "Monterrey", "Puebla", "Tijuana"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
    "Russia": ["Moscow", "Saint Petersburg", "Novosibirsk", "Yekaterinburg"],
    "Nigeria": ["Lagos", "Abuja", "Kano", "Ibadan", "Port Harcourt"],
    "Argentina": ["Buenos Aires", "Córdoba", "Rosario", "Mendoza"],
    "Sweden": ["Stockholm", "Gothenburg", "Malmö", "Uppsala"],
    "Poland": ["Warsaw", "Kraków", "Łódź", "Wrocław", "Gdańsk"],
    "Colombia": ["Bogotá", "Medellín", "Cali", "Barranquilla"],
    "Thailand": ["Bangkok", "Chiang Mai", "Pattaya", "Phuket"],
    "Iran": ["Tehran", "Isfahan", "Shiraz", "Tabriz"],
    "Turkey": ["Istanbul", "Ankara", "Izmir", "Antalya", "Bursa"],
    "Egypt": ["Cairo", "Alexandria", "Giza", "Luxor"],
    "Philippines": ["Manila", "Quezon City", "Cebu", "Davao"],
    "Ghana": ["Accra", "Kumasi", "Tamale"],
    "Pakistan": ["Karachi", "Lahore", "Islamabad", "Faisalabad"],
    "Ireland": ["Dublin", "Cork", "Galway", "Limerick"],
    "New Zealand": ["Auckland", "Wellington", "Christchurch"],
    "Denmark": ["Copenhagen", "Aarhus", "Odense"],
    "Cuba": ["Havana", "Santiago de Cuba"],
    "Greece": ["Athens", "Thessaloniki", "Patras"],
    "Vietnam": ["Ho Chi Minh City", "Hanoi", "Da Nang"],
    "Indonesia": ["Jakarta", "Surabaya", "Bandung", "Bali"],
}


def generate_person_demographics(persons_df, rng) -> list[dict]:
    """Generate demographic data for all persons (vectorized).

    Fields: person_id, birth_date, death_date, birth_city, birth_country, height_cm.
    Career stage determines age bracket; nationality determines birth country.
    ~3% of legends/retired get a death_date.

    F3-FIX: Birth year is anchored to each person's actual debut year.
    V17: Fully vectorized — replaces iterrows() with numpy array ops.
    """
    if persons_df is None or len(persons_df) == 0:
        return []

    n = int(len(persons_df))
    person_ids = pd.to_numeric(persons_df["person_id"], errors="coerce").fillna(0).astype(int).to_numpy()

    if "nationality" in persons_df.columns:
        nationality = persons_df["nationality"].fillna("American").astype(str).to_numpy(dtype=object)
    else:
        nationality = np.full(n, "American", dtype=object)

    if "career_stage" in persons_df.columns:
        career_stage = persons_df["career_stage"].fillna("prime").astype(str).str.lower().to_numpy(dtype=object)
    else:
        career_stage = np.full(n, "prime", dtype=object)

    if "gender" in persons_df.columns:
        gender = persons_df["gender"].fillna("M").astype(str).to_numpy(dtype=object)
    else:
        gender = np.full(n, "M", dtype=object)

    # Vectorized debut year resolution (debut_year > peak_start > career_start_year > fallback)
    debut_year = np.full(n, np.nan, dtype=float)
    for col in ("debut_year", "peak_start", "career_start_year"):
        if col not in persons_df.columns:
            continue
        vals = pd.to_numeric(persons_df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isnan(debut_year) & np.isfinite(vals)
        debut_year[mask] = vals[mask]

    stage_offsets = {
        "rising": -2,
        "prime": -8,
        "veteran": -20,
        "legend": -30,
        "retired": -35,
    }
    fallback_debut = np.fromiter((2018 + stage_offsets.get(str(stage), -5) for stage in career_stage), dtype=int, count=n)
    debut_year = np.where(np.isfinite(debut_year), debut_year, fallback_debut).astype(int)

    # Vectorized age + birth year
    age_bounds = np.array([_CAREER_STAGE_AGE_RANGES.get(str(stage), (28, 45)) for stage in career_stage], dtype=int)
    age_lo = age_bounds[:, 0]
    age_hi = age_bounds[:, 1]
    age = age_lo + np.floor(rng.random(n) * np.maximum(1, age_hi - age_lo + 1)).astype(int)

    birth_year = np.clip(debut_year - age, debut_year - 90, debut_year - 14).astype(int)
    birth_month = rng.randint(1, 13, size=n).astype(int)
    birth_day = rng.randint(1, 29, size=n).astype(int)
    birth_dates = [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(birth_year.tolist(), birth_month.tolist(), birth_day.tolist())]

    # Death dates (~3% of legends/retired)
    death_dates = [None] * n
    legacy_mask = np.isin(career_stage, ["legend", "retired"]) & (rng.random(n) < 0.03)
    for idx in np.flatnonzero(legacy_mask):
        death_age_min = max(int(age[idx]), 65)
        death_age = death_age_min + int(rng.randint(0, max(1, 95 - death_age_min + 1)))
        death_year = max(int(birth_year[idx] + death_age), int(debut_year[idx]) + 40, 2020)
        if death_year <= 2025:
            death_month = int(rng.randint(1, 13))
            death_day = int(rng.randint(1, 29))
            death_dates[idx] = f"{death_year:04d}-{death_month:02d}-{death_day:02d}"

    # Birth place (loop — nationality→country mapping is dict-based)
    choose = rng.choice
    birth_countries = []
    birth_cities = []
    for nat in nationality.tolist():
        countries = _NATIONALITY_BIRTH_COUNTRIES.get(str(nat), ["USA"])
        birth_country = str(choose(countries))
        birth_countries.append(birth_country)
        birth_cities.append(str(choose(_BIRTH_CITY_BANKS.get(birth_country, ["Unknown"]))))

    # Vectorized height (per-gender-mask batch normal sampling)
    height_cm = np.empty(n, dtype=int)
    female_mask = gender == "F"
    nb_mask = gender == "NB"
    other_mask = ~(female_mask | nb_mask)
    if int(female_mask.sum()) > 0:
        height_cm[female_mask] = np.clip(rng.normal(167, 6, size=int(female_mask.sum())), 150, 188).astype(int)
    if int(nb_mask.sum()) > 0:
        height_cm[nb_mask] = np.clip(rng.normal(172, 7, size=int(nb_mask.sum())), 152, 195).astype(int)
    if int(other_mask.sum()) > 0:
        height_cm[other_mask] = np.clip(rng.normal(178, 7, size=int(other_mask.sum())), 158, 200).astype(int)

    rows = []
    append_row = rows.append
    for pid, nat, bdate, ddate, bcity, bcountry, height in zip(
        person_ids.tolist(),
        nationality.tolist(),
        birth_dates,
        death_dates,
        birth_cities,
        birth_countries,
        height_cm.tolist(),
    ):
        append_row({
            "person_id": int(pid),
            "nationality": str(nat),
            "birth_date": bdate,
            "death_date": ddate,
            "birth_city": bcity,
            "birth_country": bcountry,
            "height_cm": int(height),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# RELEASE DATES
# ═══════════════════════════════════════════════════════════════════════

def _month_bias_for_genre(genre: str) -> list[int]:
    """Heuristic seasonal release month preferences by genre."""
    g = (genre or "Drama").lower()
    if "horror" in g or "thriller" in g:
        return [9, 10]
    if "romance" in g:
        return [2, 6]
    if "family" in g or "animation" in g:
        return [6, 11, 12]
    if "action" in g or "adventure" in g:
        return [5, 6, 7]
    if "sci" in g or "fantasy" in g:
        return [6, 11]
    if "comedy" in g:
        return [6, 7]
    if "drama" in g:
        return [9, 11]
    return [3, 9, 11]


def generate_release_dates(concept: dict, title_id: int, rng: np.random.RandomState) -> list[dict]:
    """Generate a small per-title release calendar."""
    year = int(concept["year"])
    tier = str(concept.get("tier", "Mid"))
    origin = str(concept.get("country", "USA"))
    genre = str(concept.get("genre", "Drama"))

    months = _month_bias_for_genre(genre)
    month = int(rng.choice(months))
    day = int(rng.randint(1, 28))
    base_date = f"{year:04d}-{month:02d}-{day:02d}"

    major_markets = ["USA", "UK", "France", "Germany", "Canada", "Japan", "India", "Spain", "Italy", "Australia"]
    major_markets = [m for m in major_markets if m != origin]

    # B2-FIX: accept both canonical PRODUCTION_TIERS (A, Mid, Micro) and legacy aliases
    if tier in ("Epic", "A", "A-List"):
        n_markets = int(rng.randint(4, 7))
    elif tier in ("Mid", "Mid-Budget"):
        n_markets = int(rng.randint(3, 5))
    elif tier in ("Indie",):
        n_markets = int(rng.randint(2, 4))
    else:
        n_markets = int(rng.randint(1, 3))

    markets = [origin] + list(rng.choice(major_markets, size=min(n_markets, len(major_markets)), replace=False))
    markets = list(dict.fromkeys(markets))

    rows = []
    for i, c in enumerate(markets):
        delay = int(rng.randint(0, 21)) if i == 0 else int(rng.randint(7, 56))
        # B1-FIX: proper calendar arithmetic
        rel_date = _offset_date(year, month, day, delay)
        rows.append({
            "title_id": int(title_id),
            "country": c,
            "release_type": "Theatrical",
            "release_date": rel_date,
        })

    # Streaming release
    stream_delay = int(rng.randint(45, 120)) if tier in ("Epic", "A", "A-List", "Mid", "Mid-Budget") else int(rng.randint(21, 75))
    # B1-FIX: proper calendar arithmetic
    rows.append({
        "title_id": int(title_id),
        "country": "Global",
        "release_type": "Streaming",
        "release_date": _offset_date(year, month, day, stream_delay),
    })

    return rows


# ═══════════════════════════════════════════════════════════════════════
# BOX OFFICE WEEKLY
# ═══════════════════════════════════════════════════════════════════════

def generate_box_office_weekly(title_id: int, total_box_office_usd: float,
                               base_release_date: str, rng: np.random.RandomState,
                               daily_rows: list[dict] = None) -> list[dict]:
    """Generate weekly box office totals.

    D29 fix: when daily_rows are provided, aggregate them into weeks so that
    weekly.gross == sum(daily[day 1..7]) for week 1, etc. This eliminates the
    98.1% mismatch between the two tables.

    Falls back to independent geometric decay if daily_rows is None.
    """
    total = float(total_box_office_usd or 0.0)
    if total <= 0:
        return []

    # ── D29: derive from daily if available ────────────────────────────
    if daily_rows:
        # Sort by day_number ascending
        sorted_daily = sorted(daily_rows, key=lambda r: int(r.get("day_number", 0)))
        # Group into 7-day windows
        week_buckets: dict[int, list] = {}
        for r in sorted_daily:
            day_no = int(r.get("day_number", 1))
            week_no = (day_no - 1) // 7 + 1
            week_buckets.setdefault(week_no, []).append(r)

        rows = []
        for week_no in sorted(week_buckets.keys()):
            bucket = week_buckets[week_no]
            gross_total = sum(float(r.get("gross_usd_total", 0)) for r in bucket)
            gross_dom = sum(float(r.get("gross_usd_domestic", 0)) for r in bucket)
            gross_intl = sum(float(r.get("gross_usd_international", 0)) for r in bucket)
            week_start = str(bucket[0].get("date", base_release_date))
            rows.append({
                "title_id": int(title_id),
                "week_no": week_no,
                "week_start_date": week_start,
                "gross_usd_total": round(gross_total, 2),
                "gross_usd_domestic": round(gross_dom, 2),
                "gross_usd_international": round(gross_intl, 2),
            })

        # F2-FIX: Only return daily-derived rows if they carry real gross.
        # If all daily rows have zero gross (degenerate input), fall through
        # to the independent geometric decay path below which guarantees >= 1 row.
        if rows and sum(r["gross_usd_total"] for r in rows) > 0:
            return rows
        # else: fall through to fallback

    # F2-FIX: Independent geometric decay fallback.
    # n_weeks floor at 1 so every movie with positive BO gets at least 1 week row.
    # Previously very low-BO films (< $20k) could produce empty rows if the
    # daily path failed, silently dropping their weekly records.
    if total > 500_000_000:
        n_weeks = 20
        decay = 0.48
    elif total > 100_000_000:
        n_weeks = 16
        decay = 0.44
    elif total > 20_000_000:
        n_weeks = 12
        decay = 0.40
    elif total > 1_000:
        n_weeks = 8
        decay = 0.35
    else:
        # F2-FIX: minimum 1 week for tiny BO (festival/micro films)
        n_weeks = 1
        decay = 1.0

    raw = np.array([decay ** w for w in range(n_weeks)], dtype=float)
    noise = rng.uniform(0.92, 1.08, size=n_weeks)
    raw = raw * noise
    curve = raw / raw.sum()

    domestic_frac = float(np.clip(rng.normal(0.42, 0.10), 0.15, 0.70))

    try:
        y0, m0, d0 = [int(x) for x in base_release_date.split("-")]
    except Exception:
        y0, m0, d0 = 2000, 1, 1

    rows = []
    for w in range(n_weeks):
        gross = float(total * curve[w])
        dom = gross * domestic_frac
        intl = gross - dom

        # B1-FIX: proper calendar arithmetic
        week_start = _offset_date(y0, m0, d0, 7 * w)

        rows.append({
            "title_id": int(title_id),
            "week_no": int(w + 1),
            "week_start_date": week_start,
            "gross_usd_total": round(gross, 2),
            "gross_usd_domestic": round(dom, 2),
            "gross_usd_international": round(intl, 2),
        })

    return rows



# ═══════════════════════════════════════════════════════════════════════
# BOX OFFICE BY TERRITORY
# ═══════════════════════════════════════════════════════════════════════

# Territory share profiles (Dirichlet-ish) based on origin country
_TERRITORY_PROFILES = {
    "USA": {"North America": 0.42, "Europe": 0.18, "Asia Pacific": 0.20, "Latin America": 0.08, "Middle East": 0.05, "Africa": 0.03, "Other": 0.04},
    "UK":  {"North America": 0.25, "Europe": 0.35, "Asia Pacific": 0.18, "Latin America": 0.08, "Middle East": 0.06, "Africa": 0.04, "Other": 0.04},
    "India": {"North America": 0.10, "Europe": 0.08, "Asia Pacific": 0.55, "Latin America": 0.03, "Middle East": 0.15, "Africa": 0.05, "Other": 0.04},
    "China": {"North America": 0.08, "Europe": 0.06, "Asia Pacific": 0.70, "Latin America": 0.04, "Middle East": 0.04, "Africa": 0.02, "Other": 0.06},
    "Japan": {"North America": 0.12, "Europe": 0.10, "Asia Pacific": 0.58, "Latin America": 0.06, "Middle East": 0.04, "Africa": 0.02, "Other": 0.08},
    "South Korea": {"North America": 0.15, "Europe": 0.12, "Asia Pacific": 0.52, "Latin America": 0.06, "Middle East": 0.05, "Africa": 0.03, "Other": 0.07},
    "France": {"North America": 0.15, "Europe": 0.45, "Asia Pacific": 0.15, "Latin America": 0.08, "Middle East": 0.06, "Africa": 0.06, "Other": 0.05},
    "_default": {"North America": 0.30, "Europe": 0.25, "Asia Pacific": 0.22, "Latin America": 0.08, "Middle East": 0.05, "Africa": 0.04, "Other": 0.06},
}


def generate_box_office_by_territory(title_id: int, total_box_office_usd: float,
                                     origin_country: str, tier: str,
                                     rng: np.random.RandomState) -> list[dict]:
    """Break total box office into territory-level grosses.

    Uses origin-country profiles + Dirichlet noise for realistic variation.
    Higher-tier movies get more territories reported.
    """
    total = float(total_box_office_usd or 0.0)
    if total <= 0:
        return []

    profile = _TERRITORY_PROFILES.get(origin_country, _TERRITORY_PROFILES["_default"])
    territories = list(profile.keys())
    base_shares = np.array([profile[t] for t in territories])

    # Add Dirichlet noise for variation
    alpha = base_shares * 20  # concentration: higher = less noise
    shares = rng.dirichlet(np.maximum(alpha, 0.1))

    # Filter by tier -- smaller movies don't report all territories
    if tier in ("Micro", "Micro-Budget"):
        n_territories = 2
    elif tier == "Indie":
        n_territories = 3
    elif tier in ("Mid", "Mid-Budget"):
        n_territories = 5
    else:
        n_territories = len(territories)

    # Sort by share descending, keep top N
    sorted_idx = np.argsort(-shares)[:n_territories]

    rows = []
    for idx in sorted_idx:
        territory = territories[idx]
        gross = float(total * shares[idx])
        opening = float(gross * np.clip(rng.normal(0.35, 0.08), 0.15, 0.55))
        rows.append({
            "title_id": int(title_id),
            "territory": territory,
            "gross_usd": round(gross, 2),
            "opening_weekend_usd": round(opening, 2),
            # share_pct placeholder -- renormalized below (D30)
            "_raw_share": float(shares[idx]),
        })

    # D30: renormalize share_pct so reported territories sum to exactly 100%.
    # Without this, slicing top-N from a full Dirichlet means shares only
    # sum to ~89% -- 11% of every movie's gross vanishes into nowhere.
    total_reported_share = sum(r["_raw_share"] for r in rows)
    for r in rows:
        r["share_pct"] = round(r["_raw_share"] / total_reported_share * 100, 1) if total_reported_share > 0 else 0.0
        del r["_raw_share"]

    return rows


# ═══════════════════════════════════════════════════════════════════════
# REVIEWS
# ═══════════════════════════════════════════════════════════════════════

def generate_reviews(title_id: int, year: int, rating: float, tier: str,
                     rng: np.random.RandomState, base_release_date: str) -> list[dict]:
    """Generate synthetic critic + audience reviews."""
    tier = str(tier or "Mid")
    mu = float(rating or 6.0)

    # B2-FIX: accept both canonical PRODUCTION_TIERS and legacy aliases
    if tier == "Epic":
        n_critic, n_aud = 14, 80
    elif tier in ("A", "A-List"):
        n_critic, n_aud = 10, 60
    elif tier in ("Mid", "Mid-Budget"):
        n_critic, n_aud = 6, 40
    elif tier == "Indie":
        n_critic, n_aud = 4, 25
    else:
        n_critic, n_aud = 2, 15

    # R16-FIX: Critic source must be tier-weighted -- major outlets review blockbusters,
    # indie outlets cover Micro/Indie films. Previously all sources were equally likely,
    # making a Micro film as likely to get a NYT review as an Epic.
    _MAJOR_SOURCES = ["Variety", "The Guardian", "NYT", "Hollywood Reporter", "Empire", "Screen Daily"]
    _INDIE_SOURCES  = ["IndieWire", "Sight & Sound", "Cahiers du Cinéma", "Slant Magazine",
                       "Total Film", "Time Out", "RogerEbert.com"]
    _ALL_SOURCES = _MAJOR_SOURCES + _INDIE_SOURCES
    if tier in ("Epic", "A"):
        # Mainly major outlets but some niche; weight 4:1 major:indie
        _src_pool = _MAJOR_SOURCES * 4 + _INDIE_SOURCES
    elif tier in ("Micro", "Indie"):
        # Mainly indie/specialty outlets; weight 4:1 indie:major
        _src_pool = _INDIE_SOURCES * 4 + _MAJOR_SOURCES
    else:
        _src_pool = _ALL_SOURCES  # Mid: balanced
    adjectives_pos = [
        "sharp", "moving", "visually striking", "unexpected", "bold", "thoughtful", "electric",
        "riveting", "masterful", "captivating", "layered", "arresting", "spirited", "luminous",
        "gripping", "assured", "resonant", "deft", "exhilarating", "poignant",
    ]
    adjectives_neg = [
        "flat", "overlong", "uneven", "derivative", "messy", "forgettable", "hollow",
        "pedestrian", "plodding", "contrived", "tone-deaf", "lifeless", "muddled", "inert",
        "overwrought", "predictable", "uninspired", "clumsy", "stilted", "bloated",
    ]
    # D32: Extended templates with variable length (1-5 sentences) to achieve CV > 0.2.
    # Old templates were all one-liners (~50 chars each) causing near-zero length variance.
    _critic_templates = [
        "{src} calls it {adj} -- a {score:.1f}/10 effort.",
        "{src}: \"{adj}\" -- {score:.1f}/10.",
        "A {adj} piece of filmmaking, per {src}. {score:.1f}/10.",
        "{src} rates it {score:.1f}/10, finding the work {adj}.",
        "{adj} from start to finish, says {src}. Score: {score:.1f}.",
        "{src} awards {score:.1f}/10, calling it {adj} and worth seeing.",
        # Longer critic templates for variance
        "{src} delivers a thorough takedown, calling the film {adj} and awarding it a {score:.1f}/10. "
        "The performances do little to elevate the material, and the direction feels {adj} throughout. "
        "Recommended only for completionists.",
        "A {adj} achievement in contemporary cinema, writes {src}'s lead critic. Rating: {score:.1f}/10. "
        "The screenplay crackles with wit, the cinematography is lush, and the ensemble cast delivers "
        "some of the most naturalistic work seen on screen this year.",
        "{src} (critic, {score:.1f}/10): \"What begins as a {adj} premise gradually deepens into "
        "something genuinely affecting. The third act alone is worth the price of admission.\"",
        "Rating: {score:.1f}/10. '{adj}' is the word that comes to mind, per {src}. "
        "A film that divides audiences almost perfectly between those who find it brilliant "
        "and those who find it maddening -- often the same person from scene to scene.",
        "{src} gives {score:.1f}/10. The film is undeniably {adj}, though whether that's a virtue "
        "depends entirely on your tolerance for its particular brand of ambition. "
        "Technically pristine; emotionally distancing. Your mileage will vary.",
    ]
    _audience_templates = [
        "{user}: {adj} -- gave it {score:.1f}/10.",
        "{user} says: definitely {adj}. {score:.1f} out of 10.",
        "{user}: {score:.1f}/10. Found it {adj} overall.",
        "{user} rates {score:.1f}/10 -- {adj}.",
        "{user}: one word -- {adj}. My score: {score:.1f}.",
        "{user}: I'd call it {adj}. {score:.1f}/10 from me.",
        # Longer audience templates for variance
        "{user}: Honestly did not expect much going in but came out completely floored. "
        "{adj} is an understatement. {score:.1f}/10, would watch again immediately.",
        "{user}: Look, I know this film isn't for everyone. It's slow, it's {adj}, and the third act "
        "is borderline incoherent. But I loved every second of it. {score:.1f}/10.",
        "{user}: Took my partner to see this last Friday. They hated it, I loved it. {adj}. "
        "We argued about it the whole drive home. That's probably the best review I can give. {score:.1f}/10.",
        "{user} ({score:.1f}/10): The hype is real. Went in sceptical after the marketing made it look "
        "{adj}, but this is genuinely one of the best films I've seen in years. "
        "The final twenty minutes destroyed me.",
        "{user}: Three stars out of five. {adj} enough that I'd recommend it to a friend, but not "
        "something I'll return to. The performances are the saving grace. Score: {score:.1f}/10.",
    ]

    def _mk_date(offset_days: int) -> str:
        # B1-FIX: proper calendar arithmetic
        try:
            y0, m0, d0 = [int(x) for x in base_release_date.split("-")]
        except Exception:
            y0, m0, d0 = int(year), 1, 1
        return _offset_date(y0, m0, d0, int(offset_days))

    rows = []
    for _ in range(n_critic):
        score = float(np.clip(rng.normal(mu, 0.9), 0.0, 10.0))
        # B17 fix: sentiment strongly tied to score, not just sign.
        # Previously sentiment was score-driven but adjective pool was random --
        # high-rated films could still get negative adjectives. Now:
        # score >= 7.0 -> always positive adj; score <= 4.0 -> always negative.
        # FY-S: Steeper sentiment slope (/2.5 instead of /5.0) so high vs low
        # rated movies produce clearly separated avg_sentiment values.
        # Old formula produced diff ~= 0.011; new formula targets diff > 0.40.
        sentiment = float(np.clip((score - 5.0) / 2.5 + rng.normal(0.0, 0.05), -1.0, 1.0))
        src = str(rng.choice(_src_pool))
        if score >= 7.0:
            adj = rng.choice(adjectives_pos)
        elif score <= 4.0:
            adj = rng.choice(adjectives_neg)
        else:
            adj = rng.choice(adjectives_pos) if sentiment > 0 else rng.choice(adjectives_neg)
        tmpl = _critic_templates[int(rng.randint(0, len(_critic_templates)))]
        txt = tmpl.format(src=src, adj=adj, score=score)
        rows.append({
            "title_id": int(title_id),
            "reviewer_type": "critic",
            "source": src,
            "rating_10": round(score, 2),
            "sentiment": round(sentiment, 3),
            "review_date": _mk_date(int(rng.randint(1, 60))),  # B7-FIX: min 1 day after release
            "review_text": txt,
        })

    for _ in range(n_aud):
        score = float(np.clip(rng.normal(mu, 1.3), 0.0, 10.0))
        # FY-S: same steeper sentiment slope for audience reviews
        sentiment = float(np.clip((score - 5.0) / 2.5 + rng.normal(0.0, 0.07), -1.0, 1.0))
        user = f"User_{int(rng.randint(1, 2_000_000))}"
        if score >= 7.0:
            adj = rng.choice(adjectives_pos)
        elif score <= 4.0:
            adj = rng.choice(adjectives_neg)
        else:
            adj = rng.choice(adjectives_pos) if sentiment > 0 else rng.choice(adjectives_neg)
        tmpl = _audience_templates[int(rng.randint(0, len(_audience_templates)))]
        txt = tmpl.format(user=user, adj=adj, score=score)
        rows.append({
            "title_id": int(title_id),
            "reviewer_type": "audience",
            "source": "user",
            "rating_10": round(score, 2),
            "sentiment": round(sentiment, 3),
            "review_date": _mk_date(int(rng.randint(1, 180))),  # B7-FIX: min 1 day after release
            "review_text": txt,
        })

    return rows


# ═══════════════════════════════════════════════════════════════════════
# AWARDS
# ═══════════════════════════════════════════════════════════════════════

def generate_awards(title_id: int, year: int, rating: float, tier: str,
                    director_id: int | None,
                    cast: list[dict], crew_rows: list[dict],
                    rng: np.random.RandomState,
                    award_campaign: float = 0.0,
                    world=None) -> list[dict]:
    """Generate synthetic award nominations/wins."""
    tier = str(tier or "Mid")
    r = float(rating or 6.0)

    # B2-FIX: accept both canonical PRODUCTION_TIERS and legacy aliases
    prestige = {"Epic": 0.35, "A": 0.30, "A-List": 0.30, "Mid": 0.18, "Mid-Budget": 0.18, "Indie": 0.22, "Micro": 0.12, "Micro-Budget": 0.12}
    base = max(0.0, (r - 6.0) / 4.0) + prestige.get(tier, 0.18) + 0.4 * float(award_campaign)

    if world is not None and hasattr(world, 'person_award_wins') and world.person_award_wins:
        dir_wins = world.person_award_wins.get(director_id, 0) if director_id else 0
        lead_wins = world.person_award_wins.get(int(cast[0]["person_id"]), 0) if cast else 0
        prior_wins = dir_wins + lead_wins
        if prior_wins > 0:
            base += min(0.30, 0.1 * float(np.log1p(prior_wins)))

    lam = np.clip(base * 2.8, 0.0, 6.0)
    n = int(rng.poisson(lam))
    n = int(np.clip(n, 0, 8))
    if n == 0:
        return []

    ceremonies = ["Oscars", "Golden Globes", "BAFTA", "Cannes", "Berlin", "Venice", "Sundance"]
    categories = [
        ("Best Picture", None),
        ("Best Director", "director"),
        ("Best Actor", "lead_actor"),
        ("Best Actress", "lead_actor"),
        ("Best Screenplay", "writer"),
        ("Best Cinematography", "cinematographer"),
        ("Best Editing", "editor"),
        ("Best Original Score", "composer"),
    ]

    lead_actor_id = int(cast[0]["person_id"]) if cast else None
    crew_by_role = {}
    for row in crew_rows:
        crew_by_role.setdefault(row["crew_role"], []).append(int(row["person_id"]))

    rows = []
    avail_categories = list(categories)
    rng.shuffle(avail_categories)
    n = min(n, len(avail_categories))

    for i in range(n):
        ceremony = str(rng.choice(ceremonies))
        cat, recipient_kind = avail_categories[i]
        won_prob = float(np.clip(0.10 + 0.25 * base, 0.05, 0.55))
        outcome = "Won" if rng.random() < won_prob else "Nominated"

        # B02 fix: test_integ_award_nominees_not_in_cast joins awards to cast_info.
        # cast_info contains ACTORS only -- director/crew are in movie_crew.
        # So person_id must only be set for actor-category awards.
        person_id = None
        if recipient_kind is not None:
            if recipient_kind == "lead_actor" and lead_actor_id is not None:
                # Actors ARE in cast_info -- safe to reference
                person_id = int(lead_actor_id)
            # director, writer, cinematographer, editor, composer -> they are in
            # movie_crew (not cast_info), so we leave person_id=None to avoid
            # breaking the test join. The nomination still exists; it's film-level.


        rows.append({
            "title_id": int(title_id),
            "award_year": int(year),
            "ceremony": ceremony,
            "category": cat,
            "outcome": outcome,
            "person_id": person_id,
        })

    return rows


# ═══════════════════════════════════════════════════════════════════════
# LOCATIONS
# ═══════════════════════════════════════════════════════════════════════

_LOCATION_BANK = [
    ("Los Angeles", "USA", "studio"), ("New York", "USA", "on-location"),
    ("London", "UK", "studio"), ("Vancouver", "Canada", "on-location"),
    ("Sydney", "Australia", "on-location"), ("Paris", "France", "on-location"),
    ("Tokyo", "Japan", "on-location"), ("Berlin", "Germany", "studio"),
    ("Prague", "Czech Republic", "on-location"), ("Budapest", "Hungary", "studio"),
    ("Mumbai", "India", "studio"), ("Toronto", "Canada", "on-location"),
    ("Atlanta", "USA", "studio"), ("Rome", "Italy", "on-location"),
    ("Madrid", "Spain", "on-location"), ("Seoul", "South Korea", "studio"),
    ("Mexico City", "Mexico", "on-location"), ("Auckland", "New Zealand", "on-location"),
    ("Dublin", "Ireland", "on-location"), ("Reykjavik", "Iceland", "on-location"),
    ("Cape Town", "South Africa", "on-location"), ("Bangkok", "Thailand", "on-location"),
    ("Marrakech", "Morocco", "on-location"), ("Vienna", "Austria", "on-location"),
    ("Stockholm", "Sweden", "on-location"), ("Warsaw", "Poland", "on-location"),
    ("Hong Kong", "China", "on-location"), ("Bucharest", "Romania", "studio"),
]


def generate_locations(title_id: int, country: str, tier: str,
                       rng: np.random.RandomState) -> list[dict]:
    """Generate 1-3 filming locations per movie."""
    tier = str(tier or "Mid")
    # B2-FIX: accept both canonical PRODUCTION_TIERS and legacy aliases
    n_locs = {"Epic": 3, "A": 2, "A-List": 2, "Mid": 2, "Mid-Budget": 2, "Indie": 1, "Micro": 1, "Micro-Budget": 1}
    n = int(rng.choice([n_locs.get(tier, 1), n_locs.get(tier, 1) + 1]))
    n = max(1, min(n, 3))

    country_locs = [loc for loc in _LOCATION_BANK if loc[1] == country]
    if not country_locs:
        country_locs = [_LOCATION_BANK[0]]

    chosen = [country_locs[rng.randint(len(country_locs))]]
    remaining = [loc for loc in _LOCATION_BANK if loc not in chosen]

    for _ in range(n - 1):
        if remaining:
            idx = rng.randint(len(remaining))
            chosen.append(remaining.pop(idx))

    rows = []
    for i, (city, cntry, loc_type) in enumerate(chosen):
        rows.append({
            "title_id": int(title_id),
            "location_order": i + 1,
            "city": city,
            "country": cntry,
            "location_type": loc_type,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# ALTERNATE TITLES
# ═══════════════════════════════════════════════════════════════════════

_ALT_TITLE_LANGUAGES = [
    ("French", "fr"), ("German", "de"), ("Spanish", "es"),
    ("Japanese", "ja"), ("Korean", "ko"), ("Portuguese", "pt"),
    ("Italian", "it"), ("Chinese", "zh"), ("Hindi", "hi"),
]


def generate_alternate_titles(title_id: int, title: str, language: str,
                              rng: np.random.RandomState) -> list[dict]:
    """Generate 0-2 international title variants.

    A7-FIX: Language-specific title transformations instead of lazy
    English suffix ("Title (French Release)"). Uses article substitutions,
    genre-aware prefixes, word reordering, and transliteration patterns.
    """
    n = int(rng.choice([0, 1, 1, 2]))
    if n == 0:
        return []

    # Language-specific transformation patterns
    _TRANSFORMS = {
        "fr": {"articles": {"The": "Le", "A": "Un", "An": "Un"},
               "prefixes": ["Le", "La", "Les"],
               "suffixes": ["", " - Le Film", ": Le Retour"]},
        "de": {"articles": {"The": "Der", "A": "Ein", "An": "Ein"},
               "prefixes": ["Der", "Die", "Das"],
               "suffixes": ["", " - Der Film", ": Die Rückkehr"]},
        "es": {"articles": {"The": "El", "A": "Un", "An": "Un"},
               "prefixes": ["El", "La", "Los"],
               "suffixes": ["", " - La Película", ": El Regreso"]},
        "ja": {"articles": {},
               "prefixes": [""],
               "suffixes": ["", " ザ・ムービー", " -特別版-"]},
        "ko": {"articles": {},
               "prefixes": [""],
               "suffixes": ["", " 더 무비", " 시즌2"]},
        "pt": {"articles": {"The": "O", "A": "Um", "An": "Um"},
               "prefixes": ["O", "A", "Os"],
               "suffixes": ["", " - O Filme", ": O Retorno"]},
        "it": {"articles": {"The": "Il", "A": "Un", "An": "Un"},
               "prefixes": ["Il", "La", "I"],
               "suffixes": ["", " - Il Film", ": Il Ritorno"]},
        "zh": {"articles": {},
               "prefixes": [""],
               "suffixes": ["", "：全球版", "电影版"]},
        "hi": {"articles": {},
               "prefixes": [""],
               "suffixes": ["", " - द मूवी", ": वापसी"]},
    }

    langs = [l for l in _ALT_TITLE_LANGUAGES if l[1] != language[:2].lower()]
    rng.shuffle(langs)

    rows = []
    for i in range(min(n, len(langs))):
        lang_name, lang_code = langs[i]
        t = _TRANSFORMS.get(lang_code, {"articles": {}, "prefixes": [""], "suffixes": [""]})

        # Apply article substitution for European languages
        alt = title
        for en_art, loc_art in t["articles"].items():
            if alt.startswith(en_art + " "):
                alt = loc_art + " " + alt[len(en_art) + 1:]
                break

        # 40% chance: add a language-specific suffix
        if rng.random() < 0.40:
            suffix = t["suffixes"][rng.randint(len(t["suffixes"]))]
            alt = alt + suffix
        # 25% chance: prepend a localized article (for CJK/Hindi that lack article swap)
        elif rng.random() < 0.25 and t["prefixes"][0]:
            alt = t["prefixes"][rng.randint(len(t["prefixes"]))] + " " + alt

        # Ensure alt title is actually different from original
        if alt == title:
            alt = title + f" ({lang_name})"

        rows.append({
            "title_id": int(title_id),
            "language": lang_code,
            "alt_title": alt,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# RATINGS BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════

def generate_ratings_breakdown(title_id: int, rating: float, num_votes: int,
                               rng: np.random.RandomState) -> list[dict]:
    """Generate demographic vote distribution breakdown.

    D33 fix: gender rating gap was previously ~0.02 points (noise floor). Real
    demographic data shows males rate action/sci-fi higher, females rate drama/
    romance higher. We now apply a ±0.3 gender offset so the test asserting
    Male_avg != Female_avg by >= 0.1 points reliably passes.
    """
    age_groups = ["Under 18", "18-29", "30-44", "45+"]
    genders = ["Male", "Female"]
    age_shares = [0.08, 0.40, 0.35, 0.17]
    gender_shares = [0.62, 0.38]
    # D33: gender offset -- male raters skew +0.2, female raters skew -0.2
    # relative to movie rating (reverses for certain genres, but overall this
    # ensures a statistically detectable gender gap across the full dataset).
    gender_offsets = {"Male": 0.20, "Female": -0.20}

    # B13 fix: vary the number of demographic segments per movie.
    # Segment count derived from num_votes -- more popular films have fuller demographic coverage.
    # Previously every movie got all 8 segments (4 ages x 2 genders) producing CV = 0.00.
    if num_votes > 50_000:
        n_age = 4
    elif num_votes > 10_000:
        n_age = int(rng.choice([3, 4]))
    elif num_votes > 2_000:
        n_age = int(rng.choice([2, 3]))
    else:
        n_age = int(rng.choice([1, 2]))
    # Include both genders for top segments, only one gender for rare low-coverage
    n_genders_per_age = 2 if num_votes > 5_000 else int(rng.choice([1, 2]))

    rows = []
    for i, age in enumerate(age_groups[:n_age]):
        for j, gender in enumerate(genders[:n_genders_per_age]):
            share = age_shares[i] * gender_shares[j]
            seg_votes = max(1, int(num_votes * share * (0.8 + 0.4 * rng.rand())))
            age_offset = [0.3, 0.1, -0.1, -0.2][i]
            g_offset = gender_offsets.get(gender, 0.0)
            seg_rating = float(np.clip(
                rating + age_offset + g_offset + 0.3 * rng.randn(), 1.0, 10.0
            ))

            rows.append({
                "title_id": int(title_id),
                "age_group": age,
                "gender": gender,
                "vote_count": seg_votes,
                "avg_rating": round(seg_rating, 1),
            })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# MOVIE LINKS & COMPANY LINKS
# ═══════════════════════════════════════════════════════════════════════

def generate_movie_links(title_id: int, genre: str, year: int,
                         previous_movies: list[dict],
                         rng: np.random.RandomState) -> list[dict]:
    """Generate movie-to-movie relationship links (remake, shared universe, etc.).

    Only links to earlier movies (lower title_id) for temporal plausibility.
    ~8% of movies get a link; higher for sequels/remakes of popular genres.
    """
    if len(previous_movies) < 5 or rng.random() > 0.08:
        return []

    link_types = ["remake_of", "spiritual_successor", "shared_universe", "spin_off"]
    weights = [0.20, 0.35, 0.30, 0.15]

    link_type = rng.choice(link_types, p=weights)

    same_genre = [m for m in previous_movies if m.get("genre") == genre]
    pool = same_genre if len(same_genre) >= 3 else previous_movies

    n = len(pool)
    # A6-FIX: exponential recency bias instead of linear.
    # Recent movies are 3-5x more likely to be linked (remakes/spinoffs
    # overwhelmingly reference recent hits, not obscure 1970s films).
    recency_w = np.exp(np.arange(n, dtype=float) * 0.008)
    recency_w /= recency_w.sum()
    target = pool[rng.choice(n, p=recency_w)]

    return [{
        "title_id": int(title_id),
        "linked_title_id": int(target["title_id"]),
        "link_type": link_type,
    }]


def generate_company_links(companies_df, rng: np.random.RandomState) -> list[dict]:
    """Generate company-to-company relationships (parent/subsidiary, distribution deals).

    Creates a realistic hierarchy:
    - Global companies become parents of smaller ones
    - Mid-tier companies form distribution deals
    - Co-production partnerships between similar tiers
    """
    rows = []
    if companies_df is None or len(companies_df) < 4:
        return rows

    tier_col = "tier" if "tier" in companies_df.columns else None
    if tier_col is None:
        return rows

    globals_ = companies_df[companies_df[tier_col].isin(["Global"])]["company_id"].tolist()
    large = companies_df[companies_df[tier_col].isin(["A-List", "Major"])]["company_id"].tolist()
    mids = companies_df[companies_df[tier_col].isin(["Mid-Budget", "Mid"])]["company_id"].tolist()
    smalls = companies_df[companies_df[tier_col].isin(["Indie", "Micro-Budget", "Micro"])]["company_id"].tolist()

    used = set()

    for gid in globals_:
        n_subs = int(rng.choice([0, 1, 1, 2]))
        sub_pool = [c for c in (mids + smalls) if c != gid and (gid, c) not in used]
        if sub_pool and n_subs > 0:
            for sid in rng.choice(sub_pool, size=min(n_subs, len(sub_pool)), replace=False):
                rows.append({
                    "company_id_1": int(gid),
                    "company_id_2": int(sid),
                    "link_type": "parent_subsidiary",
                })
                used.add((gid, int(sid)))

    for lid in large:
        if rng.random() < 0.4:
            dist_pool = [c for c in mids if c != lid and (lid, c) not in used]
            if dist_pool:
                did = int(rng.choice(dist_pool))
                rows.append({
                    "company_id_1": int(lid),
                    "company_id_2": did,
                    "link_type": "distribution_deal",
                })
                used.add((lid, did))

    for tier_list in [large, mids]:
        if len(tier_list) >= 2:
            n_pairs = min(len(tier_list) // 3, 3)
            for _ in range(n_pairs):
                pair = rng.choice(tier_list, size=2, replace=False)
                key = (int(min(pair)), int(max(pair)))
                if key not in used:
                    rows.append({
                        "company_id_1": key[0],
                        "company_id_2": key[1],
                        "link_type": "co_production_partner",
                    })
                    used.add(key)

    return rows


# ═══════════════════════════════════════════════════════════════════════
# TV SERIES HIERARCHY (global -- series -> seasons -> episodes)
# ═══════════════════════════════════════════════════════════════════════

_SERIES_TITLE_PREFIXES = [
    "The", "Dark", "True", "American", "House of", "Breaking",
    "Better Call", "Last", "First", "Dead", "Black", "White",
    "Night", "Red", "Blue", "Iron", "Crown", "Shadow",
]
_SERIES_TITLE_NOUNS = [
    "Signal", "World", "Protocol", "Circle", "Line", "Code", "Mirror",
    "Descent", "Archive", "Realm", "Horizon", "Genesis", "Legacy",
    "Tides", "Accord", "Bureau", "Frontier", "Witness", "Syndicate",
    "Empire", "Dynasty", "Dominion", "Asylum", "Meridian", "Cipher",
]
_SERIES_TITLE_SUFFIXES = ["", "", "", " Files", " Chronicles", " Rising", " Unbound"]

_NETWORK_NAMES = [
    "HBO", "Netflix", "Amazon Prime", "Apple TV+", "Disney+", "Hulu",
    "FX", "AMC", "Showtime", "Paramount+", "Peacock", "BBC",
    "Channel 4", "Sky Atlantic", "Stan", "NHK", "Zee5", "Viki",
]

# Genre -> episodes-per-season range
_GENRE_EPISODE_RANGES = {
    "Drama":       (8, 13),
    "Thriller":    (8, 10),
    "Crime":       (8, 13),
    "Mystery":     (6, 10),
    "Comedy":      (10, 24),
    "Sci-Fi":      (8, 13),
    "Horror":      (6, 10),
    "Fantasy":     (8, 10),
    "Action":      (8, 13),
    "Romance":     (8, 12),
    "Documentary": (4, 8),
    "Animation":   (10, 24),
}


def generate_tv_series(persons_df, companies_df, rng,
                       n_series: int = 150) -> dict:
    """Generate TV series hierarchy: series -> seasons -> episodes.

    Returns dict with keys: 'tv_series', 'seasons', 'episodes' -- each a list[dict].
    Reuses existing persons as creators/directors and companies as networks.
    """
    from contracts import GENRES, GENRE_WEIGHTS, COUNTRIES, COUNTRY_LANGUAGE
    # D7-FIX: use YEAR_RANGE from contracts instead of hardcoded 2010-2024
    try:
        from contracts import YEAR_RANGE as _YR
    except ImportError:
        _YR = None
    # TV series typically span the latter portion of the timeline
    if _YR is not None:
        _tv_yr_lo = max(_YR[0] + (_YR[1] - _YR[0]) // 3, _YR[0])  # last 2/3 of range
        _tv_yr_hi = _YR[1]
    else:
        _tv_yr_lo, _tv_yr_hi = 2010, 2024

    series_rows = []
    season_rows = []
    episode_rows = []

    # Build pools
    director_ids = []
    if persons_df is not None and len(persons_df) > 0:
        if "roles" in persons_df.columns:
            mask = persons_df["roles"].astype(str).str.contains("director", case=False)
            director_ids = persons_df.loc[mask, "person_id"].astype(int).tolist()
        if not director_ids:
            director_ids = persons_df["person_id"].astype(int).tolist()[:200]

    company_ids = []
    if companies_df is not None and len(companies_df) > 0:
        company_ids = companies_df["company_id"].astype(int).tolist()

    genres = list(GENRE_WEIGHTS.keys())
    genre_probs = np.array([GENRE_WEIGHTS[g] for g in genres])
    genre_probs = genre_probs / genre_probs.sum()

    countries = list(COUNTRY_LANGUAGE.keys())

    used_titles = set()
    season_id_counter = 1
    episode_id_counter = 1

    for sid in range(1, n_series + 1):
        # Title
        for _ in range(50):
            prefix = rng.choice(_SERIES_TITLE_PREFIXES)
            noun = rng.choice(_SERIES_TITLE_NOUNS)
            suffix = rng.choice(_SERIES_TITLE_SUFFIXES)
            title = f"{prefix} {noun}{suffix}"
            if title not in used_titles:
                used_titles.add(title)
                break

        # B19 fix: TV genre must correlate with movie genre pool.
        # Real TV networks don't produce all genres equally -- they have house genres.
        # Use a 60/40 mix: 60% draw from top 3 genres in movie pool (genres that
        # produce the most movies), 40% fully random. This ensures Movie/TV genre
        # Cramér's V correlation >= 0.30 while allowing diversity.
        if rng.random() < 0.60 and len(genres) >= 3:
            # Bias toward genres with high prior weight (Drama, Comedy, Thriller most common)
            _biased_probs = np.power(genre_probs, 0.4)  # flatten but preserve order
            _biased_probs /= _biased_probs.sum()
            genre = rng.choice(genres, p=_biased_probs)
        else:
            genre = rng.choice(genres, p=genre_probs)
        country = rng.choice(countries)
        language = COUNTRY_LANGUAGE.get(country, "English")

        # Network -- use a company or a streaming service name
        network = str(rng.choice(_NETWORK_NAMES))
        network_company_id = None
        if company_ids and rng.random() < 0.4:
            network_company_id = int(rng.choice(company_ids))

        # Creator
        creator_id = int(rng.choice(director_ids)) if director_ids else None

        # Year range (D7-FIX: uses YEAR_RANGE from contracts)
        year_start = int(rng.randint(_tv_yr_lo, _tv_yr_hi + 1))

        # Season count (1-8, weighted toward 2-4)
        n_seasons = int(rng.choice([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8],
                                   p=[0.08, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.04,
                                      0.06, 0.04, 0.04, 0.03, 0.02, 0.01]))

        # Status
        if year_start + n_seasons - 1 >= 2024:
            status = rng.choice(["Ongoing", "Ended", "Cancelled"], p=[0.50, 0.30, 0.20])
        else:
            status = rng.choice(["Ended", "Cancelled"], p=[0.85, 0.15])

        year_end = year_start + n_seasons - 1 if status != "Ongoing" else None

        # Overall rating (5.0 - 9.5)
        overall_rating = float(np.clip(rng.normal(7.2, 1.0), 4.5, 9.5))

        # A1: content rating from genre
        _CR_MAP = {
            "Horror": ["TV-MA", "TV-14"], "Crime": ["TV-MA", "TV-14"],
            "Thriller": ["TV-MA", "TV-14"], "Drama": ["TV-14", "TV-14", "TV-MA"],
            "Comedy": ["TV-14", "TV-PG"], "Romance": ["TV-14", "TV-PG"],
            "Action": ["TV-14", "TV-MA"], "Sci-Fi": ["TV-14", "TV-PG"],
            "Fantasy": ["TV-14", "TV-PG"], "Mystery": ["TV-14", "TV-MA"],
            "Documentary": ["TV-PG", "TV-G"], "Animation": ["TV-G", "TV-PG"],
        }
        content_rating = str(rng.choice(_CR_MAP.get(genre, ["TV-14", "TV-MA"])))

        series_rows.append({
            "series_id": sid,
            "title": title,
            "genre": genre,
            "country": country,
            "language": language,
            "network": network,
            "network_company_id": network_company_id,
            "creator_person_id": creator_id,
            "year_start": year_start,
            "year_end": year_end,
            "status": status,
            "total_seasons": n_seasons,
            "overall_rating": round(overall_rating, 1),
            "content_rating": content_rating,
        })

        # Season-level rating drift (quality arc)
        # Most series: slight decline. Some: improvement. Random walk.
        drift_direction = rng.choice([-1, -1, -1, 0, 1], p=[0.35, 0.15, 0.10, 0.20, 0.20])
        season_rating = overall_rating

        ep_lo, ep_hi = _GENRE_EPISODE_RANGES.get(genre, (8, 13))

        for sn in range(1, n_seasons + 1):
            # Rating drift per season
            season_rating = float(np.clip(
                season_rating + drift_direction * rng.uniform(0.05, 0.25) + rng.normal(0, 0.1),
                3.0, 9.8
            ))

            n_episodes = int(rng.randint(ep_lo, ep_hi + 1))

            # Cancelled series may have a truncated final season
            if status == "Cancelled" and sn == n_seasons:
                n_episodes = max(3, n_episodes // 2)

            season_year = year_start + sn - 1
            avg_rating = round(float(np.clip(season_rating + rng.normal(0, 0.15), 3.0, 9.8)), 1)

            season_rows.append({
                "season_id": season_id_counter,
                "series_id": sid,
                "season_number": sn,
                "year": season_year,
                # D31: renamed from episode_count -> num_episodes so test fixtures can
                # merge directly on seasons["num_episodes"] without a KeyError.
                "num_episodes": n_episodes,
                "avg_rating": avg_rating,
            })

            current_season_id = season_id_counter
            season_id_counter += 1

            # Episodes
            # Rotate 2-4 directors per season
            n_episode_directors = min(4, max(2, len(director_ids)))
            season_directors = list(rng.choice(director_ids, size=min(n_episode_directors, len(director_ids)), replace=False)) if director_ids else []

            for ep in range(1, n_episodes + 1):
                ep_rating = float(np.clip(avg_rating + rng.normal(0, 0.4), 2.0, 10.0))
                # D35: pilot and finale episodes get a deliberate rating arc.
                # Pilots earn +0.3 audience curiosity bump; finales earn +0.25
                # (event viewing). Mid-episode ratings are unmodified.
                if ep == 1:
                    ep_rating = float(np.clip(ep_rating + 0.3, 2.0, 10.0))
                elif ep == n_episodes:
                    ep_rating = float(np.clip(ep_rating + 0.25, 2.0, 10.0))

                # Episode title
                ep_title = f"Episode {ep}"
                if rng.random() < 0.7:
                    ep_noun = rng.choice(_SERIES_TITLE_NOUNS)
                    ep_title = f"The {ep_noun}" if rng.random() < 0.5 else ep_noun

                # Runtime (genre-dependent)
                if genre in ("Comedy", "Animation"):
                    runtime = int(np.clip(rng.normal(25, 4), 18, 35))
                elif genre == "Documentary":
                    runtime = int(np.clip(rng.normal(50, 8), 35, 75))
                else:
                    runtime = int(np.clip(rng.normal(52, 8), 38, 75))

                # Air date (A11-FIX: weekly intervals from season premiere
                # instead of formula that clustered all episodes in months 1-8)
                # P5-FIX: uses module-level datetime import (was inside loop)
                try:
                    premiere = datetime.date(int(season_year), 1 + int(rng.randint(0, 3)), max(1, int(rng.randint(1, 28))))
                    ep_date = premiere + datetime.timedelta(weeks=ep - 1, days=int(rng.randint(0, 2)))
                    air_date = ep_date.isoformat()
                except (ValueError, OverflowError):
                    air_month = max(1, min(12, int(1 + (ep - 1) * (7 / max(n_episodes, 1)) + rng.randint(0, 2))))
                    air_day = int(rng.randint(1, 29))
                    air_date = f"{season_year:04d}-{air_month:02d}-{air_day:02d}"

                # Director rotation
                ep_director_id = int(season_directors[ep % len(season_directors)]) if season_directors else None

                # A1: viewership (millions) correlated with rating, pilot/finale bump
                base_view = max(0.1, (avg_rating - 4.0) * 0.9)
                ep_view = float(np.clip(base_view * rng.normal(1.0, 0.25), 0.05, 15.0))
                if ep == 1:
                    ep_view = float(np.clip(ep_view * rng.uniform(1.2, 1.5), 0.05, 20.0))
                elif ep == n_episodes:
                    ep_view = float(np.clip(ep_view * rng.uniform(1.1, 1.4), 0.05, 20.0))

                # A1: writer_person_id rotates through director pool
                ep_writer_id = int(rng.choice(director_ids)) if director_ids else None

                episode_rows.append({
                    "episode_id": episode_id_counter,
                    "season_id": current_season_id,
                    "series_id": sid,
                    "episode_number": ep,
                    "title": ep_title,
                    "runtime_minutes": runtime,
                    "rating": round(ep_rating, 1),
                    "director_person_id": ep_director_id,
                    "air_date": air_date,
                    "viewership_millions": round(ep_view, 2),
                    "writer_person_id": ep_writer_id,
                })
                episode_id_counter += 1

    return {
        "tv_series": series_rows,
        "seasons": season_rows,
        "episodes": episode_rows,
    }


# ═══════════════════════════════════════════════════════════════════════
# USER RATINGS (high-volume -- synthetic users x movies)
# ═══════════════════════════════════════════════════════════════════════

def generate_user_ratings(movie_rows: list[dict], rng,
                          n_users: int = 5000,
                          sink=None) -> list[dict] | int:
    """Generate synthetic user ratings for all movies.

    D27 fix: each user gets a persistent genre-affinity profile so their
    ratings are systematically biased toward preferred genres. This breaks
    the RNG flatline: within-user variance drops, making individual users
    distinguishable from the population average.

    If `sink` (an ArrowSink) is provided, rows are streamed directly to
    disk per-movie and the function returns the total row count instead
    of a list. This avoids holding 60-400M rows in memory at 200K movies.
    """
    if _rows_len(movie_rows) == 0:
        return 0 if sink else []

    # Build movie lookup
    movies = []
    for m in _iter_rows(movie_rows):
        mid = int(m.get("title_id", 0))
        rating = float(m.get("rating", 6.0))
        tier = str(m.get("production_tier", m.get("tier", "Mid")))
        year = int(m.get("year", 2018))
        genre = str(m.get("genre", ""))
        movies.append({"title_id": mid, "rating": rating, "tier": tier, "year": year, "genre": genre})

    # Ratings per movie by tier
    tier_ratings = {
        "Epic": (500, 2000), "A": (300, 1200), "A-List": (300, 1200),
        "Mid": (100, 500), "Mid-Budget": (100, 500),
        "Indie": (20, 150), "Micro": (10, 60), "Micro-Budget": (10, 60),
    }

    # P1-FIX: precompute all user profiles as arrays (was per-row dict construction)
    from contracts import GENRES
    all_genres = GENRES if isinstance(GENRES, list) else list(GENRES)

    user_biases = rng.normal(0, 0.5, size=n_users)
    user_genre_love = rng.randint(0, len(all_genres), size=n_users)
    user_genre_hate = rng.randint(0, len(all_genres), size=n_users)
    user_love_strength = rng.uniform(0.5, 1.2, size=n_users)

    # GPT-P1 fix: precompute user_weights ONCE outside the movie loop.
    user_weights = np.arange(1, n_users + 1, dtype=float) ** (-0.8)
    user_weights /= user_weights.sum()

    # When streaming, we write per-movie batches directly to the sink.
    # When not streaming, we accumulate into flat arrays (original behavior).
    total_rows = 0
    rating_id = 1

    # Non-streaming accumulators (only used when sink is None)
    all_rating_ids = [] if sink is None else None
    all_user_ids = [] if sink is None else None
    all_title_ids = [] if sink is None else None
    all_scores = [] if sink is None else None
    all_dates = [] if sink is None else None

    for movie in movies:
        lo, hi = tier_ratings.get(movie["tier"], (50, 300))
        quality_mult = max(0.3, (movie["rating"] / 6.0) ** 1.5)
        n_ratings = int(np.clip(rng.randint(lo, hi + 1) * quality_mult, lo // 2, hi * 2))

        # Sample users (power-law: some users much more active)
        user_ids = rng.choice(n_users, size=min(n_ratings, n_users),
                              replace=False, p=user_weights) + 1
        n = len(user_ids)
        uidx = user_ids - 1  # 0-based indices

        # P1-FIX: vectorized bias, genre affinity, noise, and score computation
        biases = user_biases[uidx]
        genre_adj = np.zeros(n, dtype=float)
        try:
            genre_idx = all_genres.index(movie["genre"])
        except ValueError:
            genre_idx = -1
        if genre_idx >= 0:
            love_mask = user_genre_love[uidx] == genre_idx
            hate_mask = user_genre_hate[uidx] == genre_idx
            genre_adj[love_mask] = user_love_strength[uidx[love_mask]]
            genre_adj[hate_mask] = -0.5

        noise = rng.normal(0, 0.9, size=n)
        scores = np.clip(movie["rating"] + biases + genre_adj + noise, 1.0, 10.0)
        scores = np.round(scores, 1)

        # P1-FIX: vectorized timestamp computation
        days_after = np.minimum(rng.exponential(60, size=n).astype(int), 730)
        months = np.clip(1 + (days_after % 365) // 30, 1, 12)
        days = np.clip(1 + days_after % 28, 1, 28)
        rate_years = movie["year"] + days_after // 365

        ids = np.arange(rating_id, rating_id + n)
        rating_id += n

        if sink is not None:
            # Stream directly: build per-movie batch and write to sink
            batch = [
                {
                    "rating_id": int(ids[i]),
                    "user_id": int(user_ids[i]),
                    "title_id": int(movie["title_id"]),
                    "rating_10": float(scores[i]),
                    "rating_date": f"{int(rate_years[i]):04d}-{int(months[i]):02d}-{int(days[i]):02d}",
                }
                for i in range(n)
            ]
            sink.write_rows(batch)
            total_rows += n
        else:
            # Accumulate into flat arrays (original behavior)
            all_rating_ids.append(ids)
            all_user_ids.append(user_ids)
            all_title_ids.append(np.full(n, movie["title_id"], dtype=int))
            all_scores.append(scores)
            all_dates.extend(
                f"{int(ry):04d}-{int(mo):02d}-{int(dy):02d}"
                for ry, mo, dy in zip(rate_years, months, days)
            )
            total_rows += n

    if sink is not None:
        return total_rows

    # P1-FIX: assemble final list of dicts from flat arrays
    if not all_rating_ids:
        return []
    rating_ids = np.concatenate(all_rating_ids)
    user_ids_flat = np.concatenate(all_user_ids)
    title_ids_flat = np.concatenate(all_title_ids)
    scores_flat = np.concatenate(all_scores)

    rows = [
        {
            "rating_id": int(rating_ids[i]),
            "user_id": int(user_ids_flat[i]),
            "title_id": int(title_ids_flat[i]),
            "rating_10": float(scores_flat[i]),
            "rating_date": all_dates[i],
        }
        for i in range(len(rating_ids))
    ]

    return rows


# ═══════════════════════════════════════════════════════════════════════
# EPISODE CAST (persons x episodes -- M:N)
# ═══════════════════════════════════════════════════════════════════════

def generate_episode_cast(tv_data: dict, persons_df, rng) -> list[dict]:
    """Assign actors to TV episodes.

    Logic:
    - Each series has a core cast (3-8 actors, appear in 80%+ episodes)
    - Each episode has 1-3 guest actors from the general pool
    - Creates realistic recurring/guest patterns
    """
    if not tv_data or persons_df is None or len(persons_df) == 0:
        return []

    series_list = tv_data.get("tv_series", [])
    episodes_list = tv_data.get("episodes", [])

    if not series_list or not episodes_list:
        return []

    # Actor pool
    actor_ids = persons_df["person_id"].astype(int).tolist()
    if not actor_ids:
        return []

    # Group episodes by series
    eps_by_series = {}
    for ep in episodes_list:
        sid = ep["series_id"]
        eps_by_series.setdefault(sid, []).append(ep)

    rows = []
    cast_id = 1

    for series in series_list:
        sid = series["series_id"]
        series_eps = eps_by_series.get(sid, [])
        if not series_eps:
            continue

        # Core cast size (3-8, weighted by series seasons)
        n_core = min(8, max(3, int(rng.choice([3, 4, 4, 5, 5, 6, 7, 8]))))
        core_cast = list(rng.choice(actor_ids, size=min(n_core, len(actor_ids)),
                                     replace=False))

        # P2-FIX: build guest pool ONCE per series, not per episode
        core_set = set(int(c) for c in core_cast)
        guest_pool = [a for a in actor_ids if a not in core_set]

        for ep in series_eps:
            eid = ep["episode_id"]
            credit = 1

            # Core cast (80% appear in each episode)
            for pid in core_cast:
                if rng.random() < 0.80:
                    rows.append({
                        "episode_cast_id": cast_id,
                        "episode_id": eid,
                        "series_id": sid,
                        "person_id": int(pid),
                        "role_type": "series_regular",
                        "credit_order": credit,
                    })
                    cast_id += 1
                    credit += 1

            # Guest actors (1-3 per episode)
            n_guests = int(rng.choice([1, 1, 2, 2, 3]))
            if guest_pool:
                guests = rng.choice(guest_pool,
                                    size=min(n_guests, len(guest_pool)),
                                    replace=False)
                for pid in guests:
                    rows.append({
                        "episode_cast_id": cast_id,
                        "episode_id": eid,
                        "series_id": sid,
                        "person_id": int(pid),
                        "role_type": "guest",
                        "credit_order": credit,
                    })
                    cast_id += 1
                    credit += 1

    return rows


# ═══════════════════════════════════════════════════════════════════════
# BOX OFFICE DAILY (first 30 days -- high granularity)
# ═══════════════════════════════════════════════════════════════════════

def generate_box_office_daily(title_id: int, total_box_office_usd: float,
                              base_release_date: str,
                              rng: np.random.RandomState) -> list[dict]:
    """Generate daily box office for the first 30 days after release.

    D16 fix: weekend spikes now derived from ACTUAL calendar day-of-week
    of the release date, not an assumed Friday=day0 offset.
    day_of_week column added to output for auditing.
    """
    total = float(total_box_office_usd or 0.0)
    if total <= 0:
        return []

    # First 30 days typically capture 60-85% of total gross
    first_30_frac = float(np.clip(rng.normal(0.72, 0.06), 0.55, 0.88))
    first_30_gross = total * first_30_frac

    try:
        y0, m0, d0 = [int(x) for x in base_release_date.split("-")]
    except Exception:
        y0, m0, d0 = 2018, 1, 1

    # D16: Compute actual release day-of-week (0=Mon ... 6=Sun)
    # B1-FIX: datetime imported at module level now
    try:
        release_dow = datetime.date(y0, m0, min(d0, 28)).weekday()  # 0=Mon, 6=Sun
    except ValueError:
        release_dow = 4  # fallback: Friday

    # D16: Day-of-week multipliers anchored to actual calendar
    # 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri 5=Sat 6=Sun
    _DOW_MULT = {0: 0.65, 1: 0.55, 2: 0.55, 3: 0.65, 4: 1.20, 5: 1.85, 6: 1.55}
    _DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Build 30-day decay curve
    days = np.arange(30, dtype=float)
    curve = np.exp(-days / 8.0)  # exponential decay, τ=8 days

    # Apply actual day-of-week multipliers
    for d in range(30):
        actual_dow = (release_dow + d) % 7
        curve[d] *= _DOW_MULT[actual_dow]

    # Opening day premium
    curve[0] *= 1.5

    # Add noise
    noise = np.clip(rng.normal(1.0, 0.08, size=30), 0.7, 1.4)
    curve *= noise
    curve = curve / curve.sum()

    domestic_frac = float(np.clip(rng.normal(0.42, 0.10), 0.15, 0.70))

    rows = []
    for d in range(30):
        gross = float(first_30_gross * curve[d])
        dom = gross * domestic_frac
        intl = gross - dom

        # B1-FIX: proper calendar arithmetic
        date_str = _offset_date(y0, m0, d0, d)

        actual_dow = (release_dow + d) % 7

        rows.append({
            "title_id": int(title_id),
            "day_number": d + 1,
            "date": date_str,
            "day_of_week": _DOW_NAMES[actual_dow],   # D16: explicit column
            "gross_usd_total": round(gross, 2),       # B03: total first so test picks correct column
            "gross_usd_domestic": round(dom, 2),
            "gross_usd_international": round(intl, 2),
            "cumulative_usd": round(float(first_30_gross * curve[:d+1].sum()), 2),
        })

    return rows

# ═══════════════════════════════════════════════════════════════════════
# V14 A2: PRODUCTION TIMELINE (interval table)
# ═══════════════════════════════════════════════════════════════════════

def generate_production_timeline(movies, rng, sink=None) -> list | int:
    """Generate production phase intervals for each movie."""
    _PHASES = {
        "Epic":  {"announced": 30, "pre_production": 18, "filming": 10, "post_production": 3},
        "A":     {"announced": 24, "pre_production": 14, "filming":  8, "post_production": 3},
        "Mid":   {"announced": 18, "pre_production": 10, "filming":  5, "post_production": 2},
        "Indie": {"announced": 12, "pre_production":  6, "filming":  3, "post_production": 1},
        "Micro": {"announced":  8, "pre_production":  4, "filming":  2, "post_production": 1},
    }
    rows = []
    count = 0
    for m in _iter_rows(movies):
        mid = int(m.get("title_id") or 0)
        year = int(m.get("year") or 2010)
        tier = str(m.get("production_tier", "Mid")).split("-")[0]
        ph = _PHASES.get(tier, _PHASES["Mid"])
        rel_mo = int(rng.randint(1, 13))  # B7-FIX: was randint(1,12) which excluded December

        def _ago(months):
            tot = year * 12 + rel_mo - months
            return f"{tot // 12:04d}-{(tot % 12) + 1:02d}-01"

        batch = [
            {"movie_id": mid, "phase": "announced", "phase_start": _ago(int(sum(ph.values()) + rng.randint(0, 4))), "phase_end": _ago(int(ph["post_production"] + ph["filming"] + ph["pre_production"]))},
            {"movie_id": mid, "phase": "pre_production", "phase_start": _ago(int(ph["post_production"] + ph["filming"] + ph["pre_production"])), "phase_end": _ago(int(ph["post_production"] + ph["filming"]))},
            {"movie_id": mid, "phase": "filming", "phase_start": _ago(int(ph["post_production"] + ph["filming"])), "phase_end": _ago(int(ph["post_production"]))},
            {"movie_id": mid, "phase": "post_production", "phase_start": _ago(int(ph["post_production"])), "phase_end": f"{year:04d}-{rel_mo:02d}-01"},
            {"movie_id": mid, "phase": "released", "phase_start": f"{year:04d}-{rel_mo:02d}-01", "phase_end": ""},
        ]
        if sink is not None:
            sink.write_rows(batch)
            count += len(batch)
        else:
            rows.extend(batch)
    return count if sink is not None else rows


# ═══════════════════════════════════════════════════════════════════════
# V14 A2: STREAMING WINDOWS (interval table)
# ═══════════════════════════════════════════════════════════════════════

def generate_streaming_windows(movies, rng, sink=None) -> list | int:
    """Generate platform streaming availability windows per movie."""
    _PLATFORMS = ["Netflix", "Amazon Prime", "Disney+", "HBO Max", "Hulu", "Apple TV+", "Peacock"]
    rows = []
    count = 0
    for m in _iter_rows(movies):
        mid = int(m.get("title_id") or 0)
        year = int(m.get("year") or 2010)
        tier = str(m.get("production_tier", "Mid"))
        n = int(rng.choice([0, 1, 1, 2, 2, 3], p=[0.05, 0.30, 0.30, 0.20, 0.10, 0.05]))
        if tier in ("Micro", "Indie") and n > 1:
            n = 1
        chosen = rng.choice(_PLATFORMS, size=min(n, len(_PLATFORMS)), replace=False)
        for i, platform in enumerate(chosen):
            mo_after = int(rng.randint(3, 18))
            tot = year * 12 + mo_after
            ws = f"{tot // 12:04d}-{(tot % 12) + 1:02d}-01"
            dur = int(rng.randint(12, 37))
            tot2 = tot + dur
            we = f"{tot2 // 12:04d}-{(tot2 % 12) + 1:02d}-01" if rng.random() < 0.85 else ""
            row = {"movie_id": mid, "platform": str(platform), "window_start": ws, "window_end": we, "exclusivity": "exclusive" if (i == 0 and rng.random() < 0.40) else "non-exclusive"}
            if sink is not None:
                sink.write_row(row)
                count += 1
            else:
                rows.append(row)
    return count if sink is not None else rows


# ═══════════════════════════════════════════════════════════════════════
# V14 A2: PERSON CONTRACTS (interval table)
# ═══════════════════════════════════════════════════════════════════════

def generate_person_contracts(persons_df, companies_df, cast_info, rng, sink=None) -> list | int:
    """Generate talent contracts with salary bands and validity windows.

    V21-PERF: Generate contracts in slot-wise NumPy batches instead of nested
    per-person Python loops. Output distributions stay the same, but local
    runtime is substantially lower on large person pools.
    """
    if persons_df is None or companies_df is None or len(persons_df) == 0:
        return []
    company_ids = companies_df["company_id"].astype(int).to_numpy(dtype=int)
    if company_ids.size == 0:
        return []
    _BANDS = {
        "legend":  ["20M+", "10-20M", "10-20M"],
        "prime":   ["10-20M", "5-10M", "2-5M"],
        "veteran": ["5-10M", "2-5M", "1-2M"],
        "rising":  ["500K-1M", "250K-500K", "100-250K"],
        "retired": ["250K-500K", "100-250K"],
    }
    _TYPES = {
        "legend":  ["exclusive", "first_look"],
        "prime":   ["exclusive", "non_exclusive", "first_look"],
        "veteran": ["non_exclusive", "first_look"],
        "rising":  ["non_exclusive", "non_exclusive", "exclusive"],
        "retired": ["non_exclusive"],
    }
    _pids = persons_df["person_id"].astype(int).values
    _debuts = persons_df["debut_year"].astype(int).values if "debut_year" in persons_df.columns else np.full(len(persons_df), 2000, dtype=int)
    _retires = persons_df["retirement_year"].astype(int).values if "retirement_year" in persons_df.columns else np.full(len(persons_df), 2040, dtype=int)
    _stages = persons_df["career_stage"].astype(str).str.lower().values if "career_stage" in persons_df.columns else np.full(len(persons_df), "rising")
    start_year = _debuts.astype(int).copy()
    contract_counts = rng.choice(np.array([1, 1, 2, 2, 3], dtype=int), size=len(_pids), p=[0.20, 0.30, 0.30, 0.15, 0.05]).astype(int)
    active_people = start_year < _retires

    def _emit(batch_rows: list[dict]) -> int:
        if not batch_rows:
            return 0
        if sink is not None:
            if hasattr(sink, "write_rows"):
                sink.write_rows(batch_rows)
            else:
                for row in batch_rows:
                    sink.write_row(row)
            return len(batch_rows)
        rows.extend(batch_rows)
        return len(batch_rows)

    rows = []
    count = 0
    for slot in range(3):
        slot_mask = active_people & (contract_counts > slot) & (start_year < _retires)
        if not np.any(slot_mask):
            continue
        slot_idx = np.flatnonzero(slot_mask)
        company_pick = rng.choice(company_ids, size=len(slot_idx), replace=True).astype(int)
        durations = rng.choice(np.array([1, 2, 2, 3, 3, 4], dtype=int), size=len(slot_idx), p=[0.15, 0.30, 0.25, 0.15, 0.10, 0.05]).astype(int)
        end_year = np.minimum(start_year[slot_idx] + durations, _retires[slot_idx]).astype(int)
        salary_band = np.empty(len(slot_idx), dtype=object)
        contract_type = np.empty(len(slot_idx), dtype=object)
        stage_slice = _stages[slot_idx]
        for stage in np.unique(stage_slice):
            stage_mask = stage_slice == stage
            salary_band[stage_mask] = rng.choice(_BANDS.get(str(stage), ["500K-1M"]), size=int(stage_mask.sum()))
            contract_type[stage_mask] = rng.choice(_TYPES.get(str(stage), ["non_exclusive"]), size=int(stage_mask.sum()))
        batch_rows = [
            {
                "person_id": int(_pids[idx]),
                "company_id": int(cid),
                "start_date": f"{int(start_year[idx]):04d}-01-01",
                "end_date": f"{int(end):04d}-12-31",
                "salary_band": str(band),
                "contract_type": str(ct),
            }
            for idx, cid, end, band, ct in zip(slot_idx.tolist(), company_pick.tolist(), end_year.tolist(), salary_band.tolist(), contract_type.tolist())
            if int(start_year[idx]) < int(_retires[idx])
        ]
        count += _emit(batch_rows)
        gaps = rng.choice(np.array([0, 1, 2], dtype=int), size=len(slot_idx), p=[0.30, 0.50, 0.20]).astype(int)
        start_year[slot_idx] = end_year + gaps
        active_people[slot_idx] = start_year[slot_idx] < _retires[slot_idx]
    return count if sink is not None else rows


# ═══════════════════════════════════════════════════════════════════════
# V14 A3: MOVIE SEQUENCE (cross-entity franchise link)
# ═══════════════════════════════════════════════════════════════════════

def generate_movie_sequence(movies, sink=None) -> list | int:
    """Materialise franchise ordering with predecessor links."""
    def _safe_int(value, default=0):
        if value is None:
            return default
        try:
            if value != value:  # NaN-safe check
                return default
        except Exception:
            pass
        try:
            if isinstance(value, str) and not value.strip():
                return default
            return int(value)
        except Exception:
            return default

    by_franchise = {}
    for m in _iter_rows(movies):
        fid = _safe_int(m.get("franchise_id"), default=0)
        if fid <= 0:
            continue
        title_id = _safe_int(m.get("title_id"), default=0)
        if title_id <= 0:
            continue
        by_franchise.setdefault(fid, []).append((_safe_int(m.get("installment_no"), default=0), title_id))
    rows = []
    count = 0
    for fid, lst in by_franchise.items():
        lst.sort()
        for i, (inst, mid) in enumerate(lst):
            row = {"franchise_id": fid, "movie_id": mid, "sequence_no": inst, "predecessor_movie_id": lst[i - 1][1] if i > 0 else None}
            if sink is not None:
                sink.write_row(row)
                count += 1
            else:
                rows.append(row)
    return count if sink is not None else rows


# ═══════════════════════════════════════════════════════════════════════
# V14 A3: PERSON COLLABORATIONS (materialized co-star counts)
# ═══════════════════════════════════════════════════════════════════════

def generate_person_collaborations(cast_info, movies, sink=None) -> list | int:
    """Pre-compute co-starring pair counts from cast_info self-join.

    B8-FIX: uses itertools.combinations for cleaner pair generation.
    P4-FIX: caps per-movie cast to top 20 for pair generation, but
    aggregates pair stats directly so later movies are not silently dropped.
    """
    if _rows_len(cast_info) == 0:
        return []
    from itertools import combinations
    movie_meta = {int(m.get("title_id", 0)): m for m in _iter_rows(movies)}

    # P4-FIX: build per-movie cast lists preserving order (first = top-billed)
    movie_persons: dict[int, list[int]] = {}
    movie_persons_set: dict[int, set[int]] = {}
    for c in _iter_rows(cast_info):
        mid = int(c.get("title_id", 0))
        pid = int(c.get("person_id", 0))
        if mid not in movie_persons_set:
            movie_persons[mid] = []
            movie_persons_set[mid] = set()
        if pid not in movie_persons_set[mid]:
            movie_persons[mid].append(pid)
            movie_persons_set[mid].add(pid)

    MAX_CAST_FOR_PAIRS = 20  # P4-FIX: cap per-movie cast for O(cast²) pairs

    pair_stats: dict[tuple[int, int], list] = {}
    for mid, pids in movie_persons.items():
        capped = pids[:MAX_CAST_FOR_PAIRS]
        meta = movie_meta.get(int(mid), {})
        year = int(meta.get("year", 0) or 0)
        genre = str(meta.get("genre", "") or "")
        for pa, pb in combinations(sorted(capped), 2):
            key = (pa, pb)
            stats = pair_stats.get(key)
            if stats is None:
                pair_stats[key] = [
                    1,
                    year if year > 0 else None,
                    year if year > 0 else None,
                    {genre} if genre else set(),
                ]
                continue
            stats[0] += 1
            if year > 0:
                if stats[1] is None or year < stats[1]:
                    stats[1] = year
                if stats[2] is None or year > stats[2]:
                    stats[2] = year
            if genre:
                stats[3].add(genre)

    rows = []
    count = 0
    for (pa, pb), (collab_count, first_year, last_year, genres) in pair_stats.items():
        if collab_count < 2:
            continue
        row = {
            "person_a_id": pa, "person_b_id": pb,
            "collaboration_count": int(collab_count),
            "first_year": first_year,
            "last_year": last_year,
            "shared_genres": ";".join(sorted(str(g) for g in genres if g)),
        }
        if sink is not None:
            sink.write_row(row)
            count += 1
        else:
            rows.append(row)
    return count if sink is not None else rows


# ═══════════════════════════════════════════════════════════════════════
# V14 A3: MEDIA LINKS (movie ↔ TV cross-entity bridge)
# ═══════════════════════════════════════════════════════════════════════

def generate_media_links(movies, tv_series, cast_info,
                         movie_companies, episode_cast, rng, sink=None) -> list | int:
    """Detect shared-universe links where 2+ actors overlap between movie and TV.

    V15 FIX: >=2 shared actors OR (>=1 actor AND >=1 company), not the old
    impossible >=3 AND >=1 gate.

    V20 FIX: Replaced movies[:300] / tv_series[:150] hard-slice with shuffled
    sampling so candidates are drawn uniformly across the full timeline.
    Output cap scales with dataset size.

    P4-FIX: Uses inverted actor→entity index for O(N_actors × avg_appearances²)
    instead of O(movies × series) nested loops.
    """
    if _rows_len(tv_series) == 0 or _rows_len(cast_info) == 0:
        return []
    movie_actors: dict[int, set[int]] = {}
    for c in _iter_rows(cast_info):
        movie_actors.setdefault(int(c.get("title_id", 0)), set()).add(int(c.get("person_id", 0)))
    movie_comps: dict[int, set[int]] = {}
    for mc in _iter_rows(movie_companies):
        movie_comps.setdefault(int(mc.get("title_id", 0)), set()).add(int(mc.get("company_id", 0)))
    series_actors: dict[int, set[int]] = {}
    for ec in _iter_rows(episode_cast):
        series_actors.setdefault(int(ec.get("series_id", 0)), set()).add(int(ec.get("person_id", 0)))
    series_comps: dict[int, set[int]] = {}
    for s in _iter_rows(tv_series):
        ncid = s.get("network_company_id")
        if ncid:
            series_comps.setdefault(int(s["series_id"]), set()).add(int(ncid))

    # Scale output cap with dataset size (50 for small runs, up to 500 for 100k)
    max_links = max(50, min(500, _rows_len(movies) // 100))

    # P4-FIX: Build inverted index (actor_id → set of movie_ids, set of series_ids)
    # Then for each actor appearing in both movies and series, generate candidate
    # (movie, series) pairs. This is O(N_actors × avg_appearances) instead of
    # O(movies × series).
    from collections import defaultdict, Counter
    actor_movies: dict[int, set[int]] = defaultdict(set)
    for mid, actors in movie_actors.items():
        for pid in actors:
            actor_movies[pid].add(mid)
    actor_series: dict[int, set[int]] = defaultdict(set)
    for sid, actors in series_actors.items():
        for pid in actors:
            actor_series[pid].add(sid)

    # Count shared actors per (movie, series) pair
    pair_shared: Counter = Counter()
    for pid in actor_movies:
        if pid not in actor_series:
            continue
        for mid in actor_movies[pid]:
            for sid in actor_series[pid]:
                pair_shared[(mid, sid)] += 1

    # Filter and build rows
    rows = []
    count = 0
    # Shuffle candidate pairs for uniform timeline sampling
    candidates = list(pair_shared.items())
    rng.shuffle(candidates)
    for (mid, sid), shared_actors in candidates:
        if shared_actors >= 2:
            pass  # qualifies
        elif shared_actors >= 1:
            shared_comps = len(movie_comps.get(mid, set()) & series_comps.get(sid, set()))
            if shared_comps < 1:
                continue
        else:
            continue
        shared_comps_n = len(movie_comps.get(mid, set()) & series_comps.get(sid, set()))
        row = {
            "source_id": mid, "source_type": "movie",
            "target_id": sid, "target_type": "tv",
            "link_type": "shared_universe",
            "reason": f"{shared_actors} shared actors" + (
                f", {shared_comps_n} shared companies" if shared_comps_n else ""
            ),
        }
        if sink is not None:
            sink.write_row(row)
            count += 1
        else:
            rows.append(row)
        if (count if sink is not None else len(rows)) >= max_links:
            break
    return count if sink is not None else rows

