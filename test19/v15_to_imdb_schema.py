"""Convert synthetic movie data into JOB/IMDB-compatible CSVs (full 21-table core)."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from imdb_job_contract import JOB_CORE_TABLES, JOB_TABLE_COLUMNS


def _read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    if required:
        raise FileNotFoundError(f"Required source file missing: {path}")
    return pd.DataFrame()


def _safe_int(v, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _maybe_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return int(v)
    except Exception:
        return None


def _country_code(country: str) -> str:
    mapping = {
        "USA": "us", "United States": "us", "UK": "gb", "United Kingdom": "gb", "India": "in",
        "Japan": "jp", "South Korea": "kr", "Korea": "kr", "China": "cn", "France": "fr",
        "Germany": "de", "Italy": "it", "Spain": "es", "Canada": "ca", "Australia": "au",
        "Brazil": "br", "Mexico": "mx", "Russia": "ru", "Sweden": "se", "Denmark": "dk",
        "Norway": "no", "Finland": "fi", "Netherlands": "nl", "Belgium": "be", "Switzerland": "ch",
        "Austria": "at", "Portugal": "pt", "Czech Republic": "cz", "Hungary": "hu", "Romania": "ro",
        "Ukraine": "ua", "Poland": "pl", "Turkey": "tr", "Greece": "gr", "Argentina": "ar",
        "Colombia": "co", "Chile": "cl", "Peru": "pe", "Egypt": "eg", "South Africa": "za",
        "Nigeria": "ng", "Kenya": "ke", "Morocco": "ma", "Thailand": "th", "Indonesia": "id",
        "Philippines": "ph", "Vietnam": "vn", "Malaysia": "my", "Singapore": "sg", "Iran": "ir",
        "Israel": "il", "Saudi Arabia": "sa", "UAE": "ae", "Pakistan": "pk", "Bangladesh": "bd",
        "New Zealand": "nz", "Ireland": "ie", "Hong Kong": "hk", "Taiwan": "tw",
    }
    return mapping.get(str(country).strip(), "us")


def _role_id_from_gender(gender: str) -> int:
    g = str(gender).strip().lower()
    if g in {"f", "female", "woman"}:
        return 2
    return 1


def _crew_role_to_role_id(crew_role: str) -> int:
    """Map pipeline crew_role to real IMDB role_type.id."""
    mapping = {
        "producer": 3,
        "writer": 4,
        "cinematographer": 5,
        "composer": 6,
        "costume_designer": 7,
        # 8 = director (handled separately)
        "editor": 9,
        # 10 = guest
        # Everything else maps to miscellaneous (11)
    }
    return mapping.get(str(crew_role).strip().lower(), 11)


def _link_type_id(link_type: str) -> int:
    lt = str(link_type).strip().lower()
    mapping = {
        "follows": 1,
        "prequel": 1,
        "followed by": 2,
        "sequel": 2,
        "remake": 3,
        "remake of": 3,
        "remade as": 4,
        "references": 5,
        "referenced in": 6,
        "spoofs": 7,
        "spoofed in": 8,
        "features": 9,
        "featured in": 10,
        "spin-off from": 11,
        "spin_off_from": 11,
        "spin-off": 12,
        "spin_off": 12,
        "version of": 13,
        "similar": 14,
        "similar to": 14,
        "edited from": 15,
        "edited into": 16,
        "alternate language version of": 17,
    }
    return mapping.get(lt, 18)


def _company_type_id(role: str) -> int:
    """Map to real IMDB company_type: 1=distributors, 2=production, 3=sfx, 4=misc."""
    r = str(role).strip().lower()
    if "distrib" in r:
        return 1
    if "effect" in r or "vfx" in r:
        return 3
    if "production" in r or "producer" in r:
        return 2
    return 4


def _write_table(df: pd.DataFrame, table: str, out_dir: Path):
    cols = JOB_TABLE_COLUMNS[table]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.to_csv(out_dir / f"{table}.csv", index=False)
    print(f"  OK  {table:<16} {len(df):>10,} rows")


def _validate_core_headers(out_dir: Path):
    missing = []
    wrong = []
    for table, cols in JOB_TABLE_COLUMNS.items():
        p = out_dir / f"{table}.csv"
        if not p.exists():
            missing.append(table)
            continue
        got = list(pd.read_csv(p, nrows=0).columns)
        if got != cols:
            wrong.append((table, cols, got))
    if missing or wrong:
        lines = []
        if missing:
            lines.append("Missing JOB tables: " + ", ".join(sorted(missing)))
        for table, exp, got in wrong:
            lines.append(f"Header mismatch for {table}: expected {exp}, got {got}")
        raise RuntimeError("\n".join(lines))


def _make_static_tables() -> Dict[str, pd.DataFrame]:
    # All IDs match real IMDB exactly
    kind_type = pd.DataFrame([
        {"id": 1, "kind": "movie"},
        {"id": 2, "kind": "tv series"},
        {"id": 3, "kind": "tv movie"},
        {"id": 4, "kind": "video movie"},
        {"id": 5, "kind": "tv mini series"},
        {"id": 6, "kind": "video game"},
        {"id": 7, "kind": "episode"},
    ])
    role_type = pd.DataFrame([
        {"id": 1, "role": "actor"},
        {"id": 2, "role": "actress"},
        {"id": 3, "role": "producer"},
        {"id": 4, "role": "writer"},
        {"id": 5, "role": "cinematographer"},
        {"id": 6, "role": "composer"},
        {"id": 7, "role": "costume designer"},
        {"id": 8, "role": "director"},
        {"id": 9, "role": "editor"},
        {"id": 10, "role": "guest"},
        {"id": 11, "role": "miscellaneous"},
    ])
    company_type = pd.DataFrame([
        {"id": 1, "kind": "distributors"},
        {"id": 2, "kind": "production companies"},
        {"id": 3, "kind": "special effects companies"},
        {"id": 4, "kind": "miscellaneous companies"},
    ])
    link_type = pd.DataFrame([
        {"id": 1, "link": "follows"},
        {"id": 2, "link": "followed by"},
        {"id": 3, "link": "remake of"},
        {"id": 4, "link": "remade as"},
        {"id": 5, "link": "references"},
        {"id": 6, "link": "referenced in"},
        {"id": 7, "link": "spoofs"},
        {"id": 8, "link": "spoofed in"},
        {"id": 9, "link": "features"},
        {"id": 10, "link": "featured in"},
        {"id": 11, "link": "spin-off from"},
        {"id": 12, "link": "spin-off"},
        {"id": 13, "link": "version of"},
        {"id": 14, "link": "similar to"},
        {"id": 15, "link": "edited from"},
        {"id": 16, "link": "edited into"},
        {"id": 17, "link": "alternate language version of"},
        {"id": 18, "link": "unknown link"},
    ])
    info_type = pd.DataFrame([
        {"id": 1, "info": "runtimes"},
        {"id": 3, "info": "genres"},
        {"id": 4, "info": "languages"},
        {"id": 7, "info": "color info"},
        {"id": 8, "info": "countries"},
        {"id": 16, "info": "certificates"},
        {"id": 21, "info": "birth date"},
        {"id": 22, "info": "birth notes"},
        {"id": 26, "info": "death date"},
        {"id": 34, "info": "height"},
        {"id": 99, "info": "taglines"},
        {"id": 100, "info": "plot"},
        {"id": 101, "info": "votes"},
        {"id": 102, "info": "rating"},
        {"id": 103, "info": "budget"},
        {"id": 104, "info": "box office"},
        {"id": 105, "info": "production tier"},
        {"id": 500, "info": "nationality"},
        {"id": 501, "info": "biography"},
        {"id": 502, "info": "gender"},
        {"id": 503, "info": "career stage"},
    ])
    comp_cast_type = pd.DataFrame([
        {"id": 1, "kind": "cast"},
        {"id": 2, "kind": "crew"},
        {"id": 3, "kind": "complete"},
        {"id": 4, "kind": "complete+verified"},
    ])
    return {
        "kind_type": kind_type,
        "role_type": role_type,
        "company_type": company_type,
        "link_type": link_type,
        "info_type": info_type,
        "comp_cast_type": comp_cast_type,
    }


def _generate_aka_name(name_df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    rows: List[Dict] = []
    next_id = 1
    for r in name_df.itertuples(index=False):
        pid = int(getattr(r, "id"))
        full = str(getattr(r, "name", "")).strip()
        if not full:
            continue
        if rng.rand() > 0.12:
            continue
        parts = [p for p in full.replace("-", " ").split() if p]
        if len(parts) == 1:
            aliases = [f"{parts[0]} {parts[0]}"]
        else:
            first, last = parts[0], parts[-1]
            aliases = [
                f"{first[0]}. {last}",
                f"{last}, {first}",
                f"{first} {last[0]}",
            ]
        rng.shuffle(aliases)
        take = 1 if rng.rand() < 0.72 else 2
        seen = set()
        for alias in aliases[:take]:
            alias = alias.strip()
            if not alias or alias == full or alias in seen:
                continue
            seen.add(alias)
            rows.append({
                "id": next_id,
                "person_id": pid,
                "name": alias,
                "imdb_index": None,
                "name_pcode_cf": None,
                "name_pcode_nf": None,
                "surname_pcode": None,
                "md5sum": hashlib.md5(alias.encode("utf-8", errors="ignore")).hexdigest(),
            })
            next_id += 1
    return pd.DataFrame(rows)


def _build_complete_cast(movie_df: pd.DataFrame, cast_df: pd.DataFrame, crew_df: pd.DataFrame, directors_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(777)
    cast_counts = cast_df.groupby("title_id").size().to_dict() if not cast_df.empty else {}
    crew_counts = {}
    if not crew_df.empty and "title_id" in crew_df.columns:
        crew_counts = crew_df.groupby("title_id").size().to_dict()
    dir_counts = directors_df.groupby("title_id").size().to_dict() if not directors_df.empty else {}

    tier_bonus = {
        "Epic": 0.55,
        "A": 0.40,
        "Mid": 0.25,
        "Indie": 0.15,
        "Micro": 0.05,
    }

    rows: List[Dict] = []
    rid = 1
    for m in movie_df.itertuples(index=False):
        mid = int(getattr(m, "title_id"))
        tier = str(getattr(m, "production_tier", "Mid"))
        c_n = int(cast_counts.get(mid, 0))
        crew_n = int(crew_counts.get(mid, 0) + dir_counts.get(mid, 0))

        score = np.log1p(c_n) * 0.25 + np.log1p(crew_n) * 0.18 + tier_bonus.get(tier, 0.2)
        verified_p = float(np.clip(0.12 + 0.22 * score, 0.08, 0.90))

        # Subject 1: cast
        status_id = 4 if rng.rand() < verified_p else 3
        rows.append({"id": rid, "movie_id": mid, "subject_id": 1, "status_id": status_id})
        rid += 1

        # Subject 2: crew (probabilistic-rich; include often but not always)
        if crew_n > 0 or rng.rand() < 0.42:
            crew_verified_p = float(np.clip(verified_p - 0.08 + (0.03 if crew_n > 4 else 0.0), 0.05, 0.85))
            status_id = 4 if rng.rand() < crew_verified_p else 3
            rows.append({"id": rid, "movie_id": mid, "subject_id": 2, "status_id": status_id})
            rid += 1

    return pd.DataFrame(rows)


def convert(base_dir: Path, out_dir: Path, strict_job: bool, include_extras: bool):
    ent = base_dir / "entities"

    movie = _read_csv(base_dir / "movie.csv", required=True)
    cast = _read_csv(base_dir / "cast_info.csv", required=True)
    movie_companies = _read_csv(base_dir / "movie_companies.csv", required=True)
    movie_keyword = _read_csv(base_dir / "movie_keyword.csv", required=True)
    movie_links = _read_csv(base_dir / "movie_links.csv")
    aka_titles = _read_csv(base_dir / "alternate_titles.csv")
    movie_crew = _read_csv(base_dir / "movie_crew.csv")
    movie_directors = _read_csv(base_dir / "movie_directors.csv")
    person_demo = _read_csv(base_dir / "person_demographics.csv")
    tv_series = _read_csv(base_dir / "tv_series.csv")
    episodes = _read_csv(base_dir / "episodes.csv")

    persons = _read_csv(base_dir / "persons_enriched.csv")
    if persons.empty:
        persons = _read_csv(ent / "person.csv", required=True)

    companies = _read_csv(base_dir / "companies_enriched.csv")
    if companies.empty:
        companies = _read_csv(ent / "company.csv", required=True)

    keywords = _read_csv(ent / "keyword.csv", required=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_csv in out_dir.glob("*.csv"):
        try:
            old_csv.unlink()
        except PermissionError:
            print(f"  WARN locked CSV not removed: {old_csv.name} (continuing)")
    # Standardize key IDs.
    if "person_id" not in persons.columns:
        persons["person_id"] = np.arange(1, len(persons) + 1)
    if "company_id" not in companies.columns:
        companies["company_id"] = np.arange(1, len(companies) + 1)

    static = _make_static_tables()

    # title (movies)
    title_movies = pd.DataFrame({
        "id": movie["title_id"].astype(int),
        "title": movie["title"].astype(str),
        "imdb_index": None,
        "kind_id": 1,
        "production_year": pd.to_numeric(movie.get("year"), errors="coerce").astype("Int64"),
        "imdb_id": movie["title_id"].astype(int),
        "phonetic_code": None,
        "episode_of_id": None,
        "season_nr": None,
        "episode_nr": None,
        "series_years": None,
        "md5sum": None,
    })

    # title (TV series + episodes)
    title_parts = [title_movies]
    max_title_id = int(title_movies["id"].max()) if len(title_movies) > 0 else 0

    # Map series_id -> title.id so episodes can reference via episode_of_id
    series_id_to_title_id = {}
    if not tv_series.empty and "series_id" in tv_series.columns:
        series_title_ids = np.arange(max_title_id + 1, max_title_id + 1 + len(tv_series))
        for sid, tid in zip(tv_series["series_id"].astype(int), series_title_ids):
            series_id_to_title_id[sid] = int(tid)
        year_start = pd.to_numeric(tv_series.get("year_start"), errors="coerce").astype("Int64")
        year_end = pd.to_numeric(tv_series.get("year_end"), errors="coerce").astype("Int64")
        series_years_str = (year_start.astype(object).astype(str) + "-" + year_end.astype(object).fillna("").astype(str)).where(year_start.notna(), None)
        title_tv = pd.DataFrame({
            "id": series_title_ids,
            "title": tv_series["title"].astype(str),
            "imdb_index": None,
            "kind_id": 2,  # tv series
            "production_year": year_start,
            "imdb_id": series_title_ids,
            "phonetic_code": None,
            "episode_of_id": None,
            "season_nr": None,
            "episode_nr": None,
            "series_years": series_years_str,
            "md5sum": None,
        })
        title_parts.append(title_tv)
        max_title_id = int(series_title_ids.max())
        print(f"  +   TV series in title:     {len(title_tv):,} rows (kind_id=2)")

    if not episodes.empty and "episode_id" in episodes.columns:
        ep_title_ids = np.arange(max_title_id + 1, max_title_id + 1 + len(episodes))
        ep_series_ids = pd.to_numeric(episodes.get("series_id"), errors="coerce").fillna(0).astype(int)
        title_episodes = pd.DataFrame({
            "id": ep_title_ids,
            "title": episodes["title"].astype(str) if "title" in episodes.columns else "Episode",
            "imdb_index": None,
            "kind_id": 7,  # episode (real IMDB)
            "production_year": pd.to_numeric(episodes.get("air_date", "").astype(str).str[:4], errors="coerce").astype("Int64") if "air_date" in episodes.columns else None,
            "imdb_id": ep_title_ids,
            "phonetic_code": None,
            "episode_of_id": ep_series_ids.map(series_id_to_title_id).astype("Int64"),
            "season_nr": pd.to_numeric(episodes.get("episode_number", None), errors="coerce").astype("Int64") if "episode_number" in episodes.columns else None,
            "episode_nr": pd.to_numeric(episodes.get("episode_number", None), errors="coerce").astype("Int64") if "episode_number" in episodes.columns else None,
            "series_years": None,
            "md5sum": None,
        })
        # Fix season_nr: derive from seasons table if available
        seasons = _read_csv(base_dir / "seasons.csv")
        if not seasons.empty and "season_id" in seasons.columns and "season_id" in episodes.columns:
            season_map = dict(zip(
                pd.to_numeric(seasons["season_id"], errors="coerce").fillna(0).astype(int),
                pd.to_numeric(seasons["season_number"], errors="coerce").fillna(1).astype(int),
            ))
            ep_season_ids = pd.to_numeric(episodes["season_id"], errors="coerce").fillna(0).astype(int)
            title_episodes["season_nr"] = ep_season_ids.map(season_map).astype("Int64")

        title_parts.append(title_episodes)
        max_title_id = int(ep_title_ids.max())
        print(f"  +   episodes in title:      {len(title_episodes):,} rows (kind_id=7)")

    title = pd.concat(title_parts, ignore_index=True)

    # name
    name = pd.DataFrame({
        "id": persons["person_id"].astype(int),
        "name": persons["name"].astype(str),
        "imdb_index": None,
        "imdb_id": persons["person_id"].astype(int),
        "gender": (persons["gender"].astype(str) if "gender" in persons.columns else pd.Series(["U"] * len(persons))),
        "name_pcode_cf": None,
        "name_pcode_nf": None,
        "surname_pcode": None,
        "md5sum": None,
    })

    # char_name and cast_info
    cast_char = cast.get("character_name", pd.Series(["Unknown"] * len(cast))).fillna("Unknown").astype(str).str.strip()
    cast_char = cast_char.replace("", "Unknown")
    char_unique = pd.Series(sorted(set(cast_char.tolist())))
    char_name = pd.DataFrame({
        "id": np.arange(1, len(char_unique) + 1),
        "name": char_unique,
        "imdb_index": None,
        "imdb_id": np.arange(1, len(char_unique) + 1),
        "name_pcode_nf": None,
        "surname_pcode": None,
        "md5sum": None,
    })
    char_map = dict(zip(char_name["name"], char_name["id"]))
    person_gender = dict(zip(name["id"], name["gender"]))

    cast_work = cast.copy()
    if "billing_order" not in cast_work.columns:
        cast_work["billing_order"] = cast_work.groupby("title_id").cumcount() + 1
    if "character_description" not in cast_work.columns:
        cast_work["character_description"] = cast_work.get("archetype", "")
    cast_work["_char"] = cast_char

    cast_info_actors = pd.DataFrame({
        "id": np.arange(1, len(cast_work) + 1),
        "person_id": pd.to_numeric(cast_work["person_id"], errors="coerce").fillna(0).astype(int),
        "movie_id": pd.to_numeric(cast_work["title_id"], errors="coerce").fillna(0).astype(int),
        "person_role_id": cast_work["_char"].map(char_map).astype("Int64"),
        "note": cast_work["character_description"].astype(str),
        "nr_order": pd.to_numeric(cast_work["billing_order"], errors="coerce").fillna(0).astype(int),
        "role_id": cast_work["person_id"].map(lambda pid: _role_id_from_gender(person_gender.get(_safe_int(pid), "U"))).astype(int),
    })

    # --- Append directors to cast_info with role_id=8 (real IMDB) ---
    cast_parts = [cast_info_actors]
    next_cast_id = len(cast_info_actors) + 1

    if not movie_directors.empty and "director_id" in movie_directors.columns:
        dir_rows = pd.DataFrame({
            "id": np.arange(next_cast_id, next_cast_id + len(movie_directors)),
            "person_id": pd.to_numeric(movie_directors["director_id"], errors="coerce").fillna(0).astype(int),
            "movie_id": pd.to_numeric(movie_directors["title_id"], errors="coerce").fillna(0).astype(int),
            "person_role_id": None,
            "note": None,
            "nr_order": np.arange(1, len(movie_directors) + 1),
            "role_id": 8,  # director in real IMDB
        })
        cast_parts.append(dir_rows)
        next_cast_id += len(movie_directors)
        print(f"  +   directors in cast_info: {len(dir_rows):,} rows (role_id=8)")

    # --- Append crew to cast_info with role_ids 4-11 ---
    if not movie_crew.empty and "person_id" in movie_crew.columns:
        crew_rows = pd.DataFrame({
            "id": np.arange(next_cast_id, next_cast_id + len(movie_crew)),
            "person_id": pd.to_numeric(movie_crew["person_id"], errors="coerce").fillna(0).astype(int),
            "movie_id": pd.to_numeric(movie_crew["title_id"], errors="coerce").fillna(0).astype(int),
            "person_role_id": None,
            "note": movie_crew["crew_role"].astype(str) if "crew_role" in movie_crew.columns else None,
            "nr_order": pd.to_numeric(movie_crew.get("credit_order", 0), errors="coerce").fillna(0).astype(int),
            "role_id": movie_crew["crew_role"].map(_crew_role_to_role_id).astype(int) if "crew_role" in movie_crew.columns else 11,
        })
        cast_parts.append(crew_rows)
        next_cast_id += len(movie_crew)
        print(f"  +   crew in cast_info:      {len(crew_rows):,} rows (role_ids 4-11)")

    cast_info = pd.concat(cast_parts, ignore_index=True)
    cast_info["id"] = np.arange(1, len(cast_info) + 1)  # re-number sequentially

    # company_name
    company_name = pd.DataFrame({
        "id": companies["company_id"].astype(int),
        "name": companies["name"].astype(str),
        "country_code": ((companies["country"] if "country" in companies.columns else pd.Series(["USA"] * len(companies))).map(_country_code)),
        "imdb_id": companies["company_id"].astype(int),
        "name_pcode_nf": None,
        "name_pcode_sf": None,
        "md5sum": None,
    })

    # movie_companies
    mc = movie_companies.copy()
    movie_companies_out = pd.DataFrame({
        "id": np.arange(1, len(mc) + 1),
        "movie_id": pd.to_numeric(mc["title_id"], errors="coerce").fillna(0).astype(int),
        "company_id": pd.to_numeric(mc["company_id"], errors="coerce").fillna(0).astype(int),
        "company_type_id": ((mc["role"] if "role" in mc.columns else pd.Series(["production"] * len(mc))).map(_company_type_id).astype(int)),
        "note": ((mc["role"] if "role" in mc.columns else pd.Series([""] * len(mc))).astype(str)),
    })

    # keyword, movie_keyword
    keyword = pd.DataFrame({
        "id": keywords["keyword_id"].astype(int),
        "keyword": keywords["keyword"].astype(str),
        "phonetic_code": None,
    })
    mk = movie_keyword.copy()
    movie_keyword_out = pd.DataFrame({
        "id": np.arange(1, len(mk) + 1),
        "movie_id": pd.to_numeric(mk["title_id"], errors="coerce").fillna(0).astype(int),
        "keyword_id": pd.to_numeric(mk["keyword_id"], errors="coerce").fillna(0).astype(int),
    })

    # movie_link
    ml = movie_links.copy()
    if ml.empty:
        movie_link = pd.DataFrame(columns=JOB_TABLE_COLUMNS["movie_link"])
    else:
        movie_link = pd.DataFrame({
            "id": np.arange(1, len(ml) + 1),
            "movie_id": pd.to_numeric(ml["title_id"], errors="coerce").fillna(0).astype(int),
            "linked_movie_id": pd.to_numeric(ml["linked_title_id"], errors="coerce").fillna(0).astype(int),
            "link_type_id": ml.get("link_type", "unknown").map(_link_type_id).astype(int),
        })

    # aka_title
    at = aka_titles.copy()
    if at.empty:
        aka_title = pd.DataFrame(columns=JOB_TABLE_COLUMNS["aka_title"])
    else:
        aka_title = pd.DataFrame({
            "id": np.arange(1, len(at) + 1),
            "movie_id": pd.to_numeric(at["title_id"], errors="coerce").fillna(0).astype(int),
            "title": ((at["alt_title"] if "alt_title" in at.columns else pd.Series([""] * len(at))).astype(str)),
            "imdb_index": None,
            "kind_id": 1,
            "production_year": None,
            "phonetic_code": None,
            "episode_of_id": None,
            "season_nr": None,
            "episode_nr": None,
            "note": ((at["language"] if "language" in at.columns else pd.Series([""] * len(at))).astype(str)),
            "md5sum": None,
        })

    # aka_name
    rng = np.random.RandomState(42)
    aka_name = _generate_aka_name(name, rng)

    # movie_info
    mi_rows: List[Dict] = []
    for r in movie.itertuples(index=False):
        mid = int(getattr(r, "title_id"))

        def add(info_type_id: int, value):
            if value is None:
                return
            sval = str(value).strip()
            if not sval or sval.lower() == "nan":
                return
            mi_rows.append({"movie_id": mid, "info_type_id": info_type_id, "info": sval, "note": None})

        add(3, getattr(r, "genre", None))
        add(8, getattr(r, "country", None))
        add(4, getattr(r, "language", None))
        add(16, getattr(r, "certification", None))
        add(7, getattr(r, "color_format", None))
        runtime = _maybe_int(getattr(r, "runtime_minutes", None))
        if runtime is not None and runtime > 0:
            add(1, f"{runtime} min")
        add(99, getattr(r, "tagline", None))
        add(100, getattr(r, "plot_summary", None))
        budget = _maybe_int(getattr(r, "budget_usd", None))
        gross = _maybe_int(getattr(r, "box_office_usd", None))
        if budget is not None:
            add(103, f"${budget:,}")
        if gross is not None:
            add(104, f"${gross:,}")
        add(105, getattr(r, "production_tier", None))

    movie_info = pd.DataFrame(mi_rows)
    if movie_info.empty:
        movie_info = pd.DataFrame(columns=JOB_TABLE_COLUMNS["movie_info"])
    else:
        movie_info.insert(0, "id", np.arange(1, len(movie_info) + 1))

    # movie_info_idx
    mix_rows: List[Dict] = []
    for r in movie.itertuples(index=False):
        mid = int(getattr(r, "title_id"))
        rating = getattr(r, "rating", None)
        votes = getattr(r, "num_votes", None)
        if rating is not None and not pd.isna(rating):
            mix_rows.append({"movie_id": mid, "info_type_id": 102, "info": f"{float(rating):.1f}", "note": None})
        if votes is not None and not pd.isna(votes):
            mix_rows.append({"movie_id": mid, "info_type_id": 101, "info": f"{int(votes)}", "note": None})
    movie_info_idx = pd.DataFrame(mix_rows)
    if movie_info_idx.empty:
        movie_info_idx = pd.DataFrame(columns=JOB_TABLE_COLUMNS["movie_info_idx"])
    else:
        movie_info_idx.insert(0, "id", np.arange(1, len(movie_info_idx) + 1))

    # person_info
    pi_rows: List[Dict] = []
    demo_by_person = person_demo.set_index("person_id") if not person_demo.empty and "person_id" in person_demo.columns else None

    for r in persons.itertuples(index=False):
        pid = int(getattr(r, "person_id"))

        def padd(info_type_id: int, value):
            if value is None:
                return
            sval = str(value).strip()
            if not sval or sval.lower() == "nan":
                return
            pi_rows.append({"person_id": pid, "info_type_id": info_type_id, "info": sval, "note": None})

        padd(500, getattr(r, "nationality", None))
        padd(501, getattr(r, "bio", None))
        padd(502, getattr(r, "gender", None))
        padd(503, getattr(r, "career_stage", None))

        if demo_by_person is not None and pid in demo_by_person.index:
            drow = demo_by_person.loc[pid]
            if isinstance(drow, pd.DataFrame):
                drow = drow.iloc[0]
            padd(21, drow.get("birth_date"))
            birth_note = ", ".join([str(drow.get("birth_city", "")).strip(), str(drow.get("birth_country", "")).strip()]).strip(", ").strip()
            padd(22, birth_note)
            padd(26, drow.get("death_date"))
            h = drow.get("height_cm")
            if h is not None and not pd.isna(h):
                padd(34, f"{float(h):.1f} cm")

    person_info = pd.DataFrame(pi_rows)
    if person_info.empty:
        person_info = pd.DataFrame(columns=JOB_TABLE_COLUMNS["person_info"])
    else:
        person_info.insert(0, "id", np.arange(1, len(person_info) + 1))

    # complete_cast
    complete_cast = _build_complete_cast(movie, cast, movie_crew, movie_directors)

    # write core tables
    core_map = {
        "title": title,
        "name": name,
        "cast_info": cast_info,
        "char_name": char_name,
        "company_name": company_name,
        "movie_companies": movie_companies_out,
        "movie_keyword": movie_keyword_out,
        "keyword": keyword,
        "movie_link": movie_link,
        "aka_title": aka_title,
        "aka_name": aka_name,
        "movie_info": movie_info,
        "movie_info_idx": movie_info_idx,
        "person_info": person_info,
        "complete_cast": complete_cast,
        **static,
    }

    for table in JOB_CORE_TABLES:
        _write_table(core_map.get(table, pd.DataFrame()), table, out_dir)

    if strict_job:
        _validate_core_headers(out_dir)
        print("  JOB strict schema validation: OK")

    if include_extras:
        extras = [
            "movie.csv", "cast_info.csv", "movie_companies.csv", "movie_keyword.csv", "movie_links.csv",
            "movie_crew.csv", "movie_directors.csv", "release_dates.csv", "ratings_breakdown.csv",
            "reviews.csv", "awards.csv", "locations.csv", "world_events.csv", "production_timeline.csv",
            "streaming_windows.csv", "person_contracts.csv", "movie_sequence.csv", "person_collaborations.csv",
            "tv_series.csv", "seasons.csv", "episodes.csv", "episode_cast.csv", "box_office_daily.csv",
            "box_office_weekly.csv", "box_office_by_territory.csv", "person_demographics.csv",
            "persons_enriched.csv", "companies_enriched.csv",
        ]
        core_csv_names = {f"{t}.csv" for t in JOB_CORE_TABLES}
        for fname in extras:
            src = base_dir / fname
            if src.exists():
                df = pd.read_csv(src, low_memory=False)
                dst_name = fname if fname not in core_csv_names else f"extra_{fname}"
                df.to_csv(out_dir / dst_name, index=False)
                print(f"  OK  extra:{dst_name:<24} {len(df):>10,} rows")
        edge_src = base_dir / "graph" / "edge_graph.csv"
        if edge_src.exists():
            edf = pd.read_csv(edge_src, low_memory=False)
            edf.to_csv(out_dir / "extra_edges.csv", index=False)
            print(f"  OK  extra:extra_edges.csv{'':<11} {len(edf):>10,} rows")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to strict JOB/IMDB core schema")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent),
                        help="Source dataset directory (contains movie.csv, entities/, graph/)")
    parser.add_argument("--out-dir", default=str((Path(__file__).resolve().parent / "imdb_schema").resolve()),
                        help="Output directory for converted CSV tables")
    parser.add_argument("--strict-job", action=argparse.BooleanOptionalAction, default=True,
                        help="Validate and fail if any of the 21 JOB tables/headers are missing or mismatched")
    parser.add_argument("--include-extras", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy non-JOB research tables into output directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"Source: {base_dir}")
    print(f"Output: {out_dir}")
    convert(base_dir=base_dir, out_dir=out_dir, strict_job=bool(args.strict_job), include_extras=bool(args.include_extras))
    print("Done.")


if __name__ == "__main__":
    main()



