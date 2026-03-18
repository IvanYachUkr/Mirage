"""
Backfill missing keywords into movie_keyword.csv
=================================================
Post-hoc assigns the 29 keywords that had NaN pop_weight during step 7
to appropriate movies based on genre matching. Reversible via backup file.

This script:
1. Backs up movie_keyword.csv to movie_keyword.csv.pre_backfill
2. Identifies which keywords are under-represented
3. Assigns them to genre-matching movies using the same pop_weight distribution
   that pick_keywords would have used
4. Writes a manifest of all changes for reversibility

Usage:
    python backfill_missing_keywords.py
    python backfill_missing_keywords.py --revert  # undo all changes
"""
import os
import sys
import shutil
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# The 29 keyword IDs that had NaN pop_weight during the step 7 run
BACKFILL_KEYWORD_IDS = list(range(3603, 3632))  # 3603 through 3631


def _expected_assignment_count(pop_weight: float, total_movies: int,
                                avg_kw_per_movie: float, total_kw_weight: float) -> int:
    """Estimate how many movies a keyword SHOULD appear in based on its pop_weight.

    pick_keywords samples ~3-5 keywords per movie from the full pool.
    A keyword's expected frequency is proportional to its share of total weight.
    """
    weight_share = pop_weight / total_kw_weight
    expected = weight_share * total_movies * avg_kw_per_movie
    return max(1, int(round(expected)))


def main():
    parser = argparse.ArgumentParser(description="Backfill missing keywords")
    parser.add_argument("--revert", action="store_true",
                        help="Revert to pre-backfill state")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing")
    args = parser.parse_args()

    mk_path = BASE_DIR / "movie_keyword.csv"
    backup_path = BASE_DIR / "movie_keyword.csv.pre_backfill"
    manifest_path = BASE_DIR / "backfill_manifest.json"

    # ─── REVERT MODE ──────────────────────────────────────────────────
    if args.revert:
        if not backup_path.exists():
            print("ERROR: No backup file found. Nothing to revert.")
            sys.exit(1)
        shutil.copy2(backup_path, mk_path)
        print(f"Reverted movie_keyword.csv from backup ({backup_path})")
        return

    # ─── LOAD DATA ────────────────────────────────────────────────────
    mk = pd.read_csv(mk_path)
    kw = pd.read_csv(BASE_DIR / "entities" / "keyword.csv")
    movies = pd.read_csv(BASE_DIR / "movie.csv", usecols=["title_id", "genre"])

    print(f"Loaded {len(mk)} movie_keyword rows, {len(movies)} movies, {len(kw)} keywords")

    # ─── IDENTIFY UNDER-REPRESENTED KEYWORDS ──────────────────────────
    total_kw_weight = kw["pop_weight"].sum()
    avg_kw_per_movie = mk.groupby("title_id").size().mean()
    total_movies = movies["title_id"].nunique()

    # Current assignment counts for backfill keywords
    backfill_kw = kw[kw["keyword_id"].isin(BACKFILL_KEYWORD_IDS)].copy()
    current_counts = mk[mk["keyword_id"].isin(BACKFILL_KEYWORD_IDS)].groupby("keyword_id").size()

    assignments_needed = []
    for _, row in backfill_kw.iterrows():
        kid = row["keyword_id"]
        expected = _expected_assignment_count(
            row["pop_weight"], total_movies, avg_kw_per_movie, total_kw_weight
        )
        current = current_counts.get(kid, 0)
        deficit = max(0, expected - current)
        if deficit > 0:
            assignments_needed.append({
                "keyword_id": kid,
                "keyword": row["keyword"],
                "topic_genre": row["topic_genre"],
                "pop_weight": row["pop_weight"],
                "expected": expected,
                "current": current,
                "deficit": deficit,
            })

    if not assignments_needed:
        print("All backfill keywords are already at expected levels. Nothing to do.")
        return

    print(f"\n{len(assignments_needed)} keywords need backfill:")
    for a in sorted(assignments_needed, key=lambda x: -x["deficit"]):
        print(f"  {a['keyword']:30s} {a['topic_genre']:12s} "
              f"expected={a['expected']:4d}  current={a['current']:3d}  "
              f"deficit={a['deficit']:4d}")

    # ─── GENRE MAPPING ────────────────────────────────────────────────
    # Parse multi-genre strings (e.g., "Action, Comedy") into sets
    movie_genres = {}
    for _, row in movies.iterrows():
        tid = row["title_id"]
        genre_str = str(row["genre"]) if pd.notna(row["genre"]) else ""
        genres = {g.strip() for g in genre_str.replace("/", ",").split(",") if g.strip()}
        movie_genres[tid] = genres

    # Build genre -> [title_ids] index
    genre_to_movies = {}
    for tid, genres in movie_genres.items():
        for g in genres:
            genre_to_movies.setdefault(g, []).append(tid)

    # Movies already assigned to each keyword (to avoid duplicates)
    existing_assignments = set()
    for _, row in mk.iterrows():
        existing_assignments.add((row["title_id"], row["keyword_id"]))

    # ─── ASSIGN KEYWORDS ─────────────────────────────────────────────
    rng = np.random.RandomState(42)  # reproducible
    new_rows = []

    # Keyword-to-genre affinity rules for better matching
    # Some keywords match multiple genres
    KEYWORD_GENRE_AFFINITY = {
        "based-on-comic": ["Superhero", "Action", "Adventure", "Sci-Fi"],
        "based-on-novel": ["Drama", "Romance", "Thriller", "Mystery", "Horror"],
        "blood": ["Horror", "Action", "Thriller", "Crime"],
        "character-name-in-title": ["Biography", "Drama", "Comedy", "Action"],
        "claw": ["Horror", "Action", "Superhero", "Sci-Fi"],
        "computer-animated-movie": ["Animation", "Family", "Comedy", "Adventure"],
        "computer-animation": ["Animation", "Family", "Comedy", "Adventure"],
        "death": ["Drama", "Horror", "Thriller", "Crime", "War", "Action"],
        "dignity": ["Drama", "Biography", "History"],
        "female-nudity": ["Drama", "Thriller", "Horror", "Romance"],
        "fight": ["Action", "Martial Arts", "Superhero", "Crime", "Thriller"],
        "hero": ["Action", "Adventure", "Superhero", "Fantasy", "Sci-Fi"],
        "hospital": ["Drama", "Horror", "Thriller", "Comedy"],
        "laser": ["Sci-Fi", "Action", "Fantasy", "Superhero"],
        "loner": ["Drama", "Western", "Thriller", "Crime"],
        "magnet": ["Sci-Fi", "Action"],
        "marvel-cinematic-universe": ["Superhero", "Action", "Adventure", "Sci-Fi"],
        "marvel-comics": ["Superhero", "Action", "Adventure"],
        "murder": ["Crime", "Thriller", "Mystery", "Horror", "Drama"],
        "murder-in-title": ["Crime", "Mystery", "Thriller", "Horror"],
        "nerd": ["Comedy", "Sci-Fi", "Drama"],
        "revenge": ["Action", "Thriller", "Crime", "Drama", "Western"],
        "second-part": ["Action", "Adventure", "Sci-Fi", "Fantasy", "Thriller"],
        "sequel": ["Action", "Horror", "Comedy", "Adventure", "Sci-Fi"],
        "superhero": ["Superhero", "Action", "Adventure", "Sci-Fi"],
        "tv-special": ["Comedy", "Drama", "Reality-TV", "Documentary"],
        "violence": ["Action", "Crime", "Thriller", "Horror", "War"],
        "web": ["Thriller", "Sci-Fi", "Crime", "Horror"],
        "10,000-mile-club": ["Comedy", "Adventure", "Drama"],
    }

    for assignment in assignments_needed:
        kid = assignment["keyword_id"]
        kw_name = assignment["keyword"]
        deficit = assignment["deficit"]

        # Get candidate movies from matching genres
        matching_genres = KEYWORD_GENRE_AFFINITY.get(kw_name, [assignment["topic_genre"]])
        candidate_tids = set()
        for g in matching_genres:
            candidate_tids.update(genre_to_movies.get(g, []))

        # Remove movies that already have this keyword
        candidate_tids -= {tid for (tid, kid2) in existing_assignments if kid2 == kid}

        if not candidate_tids:
            print(f"  WARNING: No candidate movies for '{kw_name}'")
            continue

        candidate_list = sorted(candidate_tids)
        n_assign = min(deficit, len(candidate_list))

        # Sample without replacement, weighted slightly toward movies with
        # fewer keywords (to balance keyword distribution)
        movie_kw_counts = mk.groupby("title_id").size().reindex(candidate_list, fill_value=0)
        # Inverse weight: movies with fewer keywords are more likely to get new ones
        weights = 1.0 / (movie_kw_counts.values + 1.0)
        weights = weights / weights.sum()

        chosen = rng.choice(candidate_list, size=n_assign, replace=False, p=weights)

        for tid in chosen:
            new_rows.append({"title_id": tid, "keyword_id": kid})
            existing_assignments.add((tid, kid))

        print(f"  Assigned '{kw_name}' to {n_assign} movies "
              f"(from {len(candidate_list)} candidates)")

    print(f"\nTotal new assignments: {len(new_rows)}")

    if args.dry_run:
        print("DRY RUN — no files written.")
        return

    # ─── BACKUP & WRITE ──────────────────────────────────────────────
    shutil.copy2(mk_path, backup_path)
    print(f"Backed up to {backup_path}")

    # Append new rows
    new_df = pd.DataFrame(new_rows)
    mk_updated = pd.concat([mk, new_df], ignore_index=True)
    mk_updated.to_csv(mk_path, index=False)
    print(f"Written {len(mk_updated)} rows to {mk_path} (+{len(new_rows)} new)")

    # Save manifest for auditability
    manifest = {
        "backfill_date": pd.Timestamp.now().isoformat(),
        "total_new_assignments": len(new_rows),
        "keywords_backfilled": [a["keyword"] for a in assignments_needed],
        "new_rows": [{"title_id": int(r["title_id"]), "keyword_id": int(r["keyword_id"])} for r in new_rows],
        "backup_file": str(backup_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest to {manifest_path}")
    print(f"\nTo revert: python backfill_missing_keywords.py --revert")


if __name__ == "__main__":
    main()
