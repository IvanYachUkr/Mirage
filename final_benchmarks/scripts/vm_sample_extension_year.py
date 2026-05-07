"""Read-only sample/audit for a Mirage Step100 extension year checkpoint."""

from __future__ import annotations

import json
import statistics
import csv
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.ipc as ipc


BASE = Path("/home/vanya/DATA_SYS_LAB/test42")
SHARDS = BASE / "_step100_resume" / "shards"
YEAR = 2051


def read_arrow(path: Path):
    try:
        return ipc.open_file(str(path)).read_all()
    except Exception:
        return ipc.open_stream(str(path)).read_all()


def table_rows(name: str, year: int = YEAR) -> list[dict]:
    path = SHARDS / name / f"year={year}.arrow"
    if not path.exists():
        return []
    return read_arrow(path).to_pylist()


def table_obj(name: str, year: int = YEAR):
    path = SHARDS / name / f"year={year}.arrow"
    if not path.exists():
        return None
    return read_arrow(path)


def compact(row: dict, keys: list[str]) -> dict:
    return {k: row.get(k) for k in keys if k in row}


def main() -> None:
    tables = {
        name: table_obj(name)
        for name in [
            "movie",
            "cast_info",
            "movie_companies",
            "movie_keyword",
            "movie_crew",
            "release_dates",
            "box_office_daily",
        ]
    }
    movies = tables["movie"].to_pylist() if tables["movie"] is not None else []
    cast = tables["cast_info"].to_pylist() if tables["cast_info"] is not None else []
    companies = tables["movie_companies"].to_pylist() if tables["movie_companies"] is not None else []
    keywords = tables["movie_keyword"].to_pylist() if tables["movie_keyword"] is not None else []
    crew = tables["movie_crew"].to_pylist() if tables["movie_crew"] is not None else []
    release_dates = tables["release_dates"].to_pylist() if tables["release_dates"] is not None else []
    box_daily = tables["box_office_daily"].to_pylist() if tables["box_office_daily"] is not None else []

    person_names: dict[int, str] = {}
    person_path = BASE / "_step100_resume" / "checkpoint" / "latest" / "persons.arrow"
    if person_path.exists():
        for row in read_arrow(person_path).select(["person_id", "name"]).to_pylist():
            if row.get("person_id") is not None:
                person_names[int(row["person_id"])] = str(row.get("name") or "")

    company_names: dict[int, str] = {}
    company_path = BASE / "_step100_resume" / "checkpoint" / "latest" / "companies.arrow"
    if company_path.exists():
        company_cols = read_arrow(company_path)
        company_id_col = "company_id" if "company_id" in company_cols.column_names else "id"
        company_name_col = "name" if "name" in company_cols.column_names else "company_name"
        for row in company_cols.select([company_id_col, company_name_col]).to_pylist():
            if row.get(company_id_col) is not None:
                company_names[int(row[company_id_col])] = str(row.get(company_name_col) or "")

    keyword_names: dict[int, str] = {}
    keyword_path = BASE / "entities" / "keyword.csv"
    if keyword_path.exists():
        with keyword_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                kid = row.get("keyword_id") or row.get("id")
                name = row.get("keyword") or row.get("name")
                if kid:
                    keyword_names[int(kid)] = str(name or "")

    columns = {name: table.column_names if table is not None else [] for name, table in tables.items()}

    movie_id_candidates = ["movie_id", "id", "title_id", "movie"]

    def infer_key(row: dict, candidates: list[str] = movie_id_candidates) -> int | None:
        for key in candidates:
            value = row.get(key)
            if value is not None:
                return value
        return None

    movie_ids = {infer_key(r) for r in movies}
    movie_ids.discard(None)

    def movie_key(row: dict) -> int | None:
        return infer_key(row)

    cast_by_movie = Counter(movie_key(r) for r in cast)
    company_by_movie = Counter(movie_key(r) for r in companies)
    keyword_by_movie = Counter(movie_key(r) for r in keywords)
    crew_by_movie = Counter(movie_key(r) for r in crew)
    daily_by_movie = Counter(movie_key(r) for r in box_daily)

    titles = [str(r.get("title", "")).strip() for r in movies]
    years = Counter(r.get("production_year") or r.get("year") for r in movies)
    kinds = Counter(r.get("kind_id") or r.get("kind") for r in movies)

    old_person_refs = 0
    new_person_refs = 0
    person_ids = []
    for row in cast + crew:
        pid = row.get("person_id")
        if isinstance(pid, int):
            person_ids.append(pid)
            if pid <= 240000:
                old_person_refs += 1
            else:
                new_person_refs += 1

    old_company_refs = 0
    new_company_refs = 0
    company_ids = []
    for row in companies:
        cid = row.get("company_id")
        if isinstance(cid, int):
            company_ids.append(cid)
            if cid <= 9000:
                old_company_refs += 1
            else:
                new_company_refs += 1

    bad = {
        "movies_with_empty_title": sum(1 for title in titles if not title),
        "movie_rows_wrong_year": sum(
            1 for row in movies if (row.get("production_year") or row.get("year")) != YEAR
        ),
        "movies_without_cast": sum(1 for mid in movie_ids if cast_by_movie[mid] == 0),
        "movies_without_company": sum(1 for mid in movie_ids if company_by_movie[mid] == 0),
        "movies_without_keyword": sum(1 for mid in movie_ids if keyword_by_movie[mid] == 0),
        "movies_without_crew": sum(1 for mid in movie_ids if crew_by_movie[mid] == 0),
        "movies_without_daily_box_office": sum(1 for mid in movie_ids if daily_by_movie[mid] == 0),
    }

    sample_movies = movies[:5]
    sample_ids = [movie_key(row) for row in sample_movies]

    samples = []
    for movie in sample_movies:
        mid = movie_key(movie)
        samples.append(
            {
                "movie": compact(
                    movie,
                    [
                        "id",
                        "movie_id",
                        "title",
                        "production_year",
                        "year",
                        "kind_id",
                        "genre",
                        "language",
                        "country",
                    ],
                ),
                "cast": [
                    {
                        **compact(r, ["person_id", "character_name", "billing_order", "archetype"]),
                        "person_name": person_names.get(r.get("person_id"), ""),
                    }
                    for r in cast
                    if movie_key(r) == mid
                ][:6],
                "companies": [
                    {
                        **compact(r, ["company_id", "role"]),
                        "company_name": company_names.get(r.get("company_id"), ""),
                    }
                    for r in companies
                    if movie_key(r) == mid
                ][:5],
                "keywords": [
                    {
                        **compact(r, ["keyword_id"]),
                        "keyword": keyword_names.get(r.get("keyword_id"), ""),
                    }
                    for r in keywords
                    if movie_key(r) == mid
                ][:8],
                "crew": [
                    {
                        **compact(r, ["person_id", "crew_role", "department"]),
                        "person_name": person_names.get(r.get("person_id"), ""),
                    }
                    for r in crew
                    if movie_key(r) == mid
                ][:6],
                "release_dates": [compact(r, ["country_code", "release_date", "note"]) for r in release_dates if movie_key(r) == mid][:5],
            }
        )

    summary = {
        "year": YEAR,
        "table_rows": {
            "movie": len(movies),
            "cast_info": len(cast),
            "movie_companies": len(companies),
            "movie_keyword": len(keywords),
            "movie_crew": len(crew),
            "release_dates": len(release_dates),
            "box_office_daily": len(box_daily),
        },
        "columns": columns,
        "movie_id_count": len(movie_ids),
        "movie_years": dict(years),
        "movie_kinds": dict(kinds),
        "per_movie_medians": {
            "cast": statistics.median(cast_by_movie[mid] for mid in movie_ids) if movie_ids else None,
            "companies": statistics.median(company_by_movie[mid] for mid in movie_ids) if movie_ids else None,
            "keywords": statistics.median(keyword_by_movie[mid] for mid in movie_ids) if movie_ids else None,
            "crew": statistics.median(crew_by_movie[mid] for mid in movie_ids) if movie_ids else None,
        },
        "old_new_refs": {
            "old_person_refs": old_person_refs,
            "new_person_refs": new_person_refs,
            "old_company_refs": old_company_refs,
            "new_company_refs": new_company_refs,
            "min_person_id": min(person_ids) if person_ids else None,
            "max_person_id": max(person_ids) if person_ids else None,
            "min_company_id": min(company_ids) if company_ids else None,
            "max_company_id": max(company_ids) if company_ids else None,
        },
        "basic_failures": bad,
        "samples": samples,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
