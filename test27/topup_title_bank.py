"""Deterministically top up title_bank.csv to a target count without any API calls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from contracts import DECADE_WEIGHTS, GENRES, generate_compositional_title


def _year_sample(rng: np.random.RandomState) -> int:
    decades = sorted(DECADE_WEIGHTS.keys())
    probs = np.array([DECADE_WEIGHTS[d] for d in decades], dtype=float)
    probs = probs / probs.sum()
    decade = int(rng.choice(decades, p=probs))
    return decade + int(rng.randint(0, 10))


def _planned_years(
    existing_years: pd.Series,
    target_count: int,
    start_year: int,
    end_year: int,
    rng: np.random.RandomState,
) -> list[int]:
    if end_year < start_year:
        raise ValueError(f"end_year {end_year} must be >= start_year {start_year}")

    years = list(range(int(start_year), int(end_year) + 1))
    desired = {year: target_count // len(years) for year in years}
    for year in years[: target_count % len(years)]:
        desired[year] += 1

    existing_counts = (
        pd.to_numeric(existing_years, errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .to_dict()
    )
    planned: list[int] = []
    for year in years:
        deficit = max(0, int(desired[year]) - int(existing_counts.get(year, 0)))
        planned.extend([year] * deficit)
    rng.shuffle(planned)
    return planned


def _tagline(genre: str, rng: np.random.RandomState) -> str:
    pool = {
        "Action": ["No rules. No limits.", "When chaos rises, heroes answer."],
        "Drama": ["Every choice leaves a scar.", "Some truths refuse to stay buried."],
        "Comedy": ["Nothing goes to plan.", "Bad ideas, perfect timing."],
        "Sci-Fi": ["The future has a price.", "Beyond the edge of known."],
        "Thriller": ["Trust fractures first.", "The clock is already running."],
        "Horror": ["It was never empty.", "Fear always finds a door."],
        "Fantasy": ["Legends wake in silence.", "Magic remembers everything."],
    }
    opts = pool.get(genre, ["Everything changes tonight.", "No one gets out unchanged."])
    return opts[int(rng.randint(0, len(opts)))]


def topup(
    base_dir: Path,
    target_count: int,
    seed: int,
    start_year: int | None = None,
    end_year: int | None = None,
):
    import os
    os.makedirs(base_dir / "entities", exist_ok=True)
    tb_path = base_dir / "entities" / "title_bank.csv"

    if tb_path.exists():
        df = pd.read_csv(tb_path, low_memory=False)
        cur = len(df)
        used = set(df["title"].astype(str).tolist()) if "title" in df.columns else set()
    else:
        df = pd.DataFrame(columns=["title", "tagline", "genre_hint", "year", "award_contender"])
        cur = 0
        used = set()

    if cur >= target_count and start_year is None and end_year is None:
        print(f"title_bank already has {cur} rows (target {target_count})")
        return

    rng = np.random.RandomState(seed)
    rows = []
    genre_probs = np.ones(len(GENRES), dtype=float) / max(1, len(GENRES))
    if (start_year is None) != (end_year is None):
        raise ValueError("start_year and end_year must be provided together")
    if start_year is not None and end_year is not None:
        planned_years = _planned_years(
            df["year"] if "year" in df.columns else pd.Series(dtype=float),
            int(target_count),
            int(start_year),
            int(end_year),
            rng,
        )
        needed = len(planned_years)
    else:
        needed = max(0, target_count - cur)
        planned_years = []

    if needed <= 0:
        print(f"title_bank already satisfies target/distribution ({cur} rows)")
        return

    for idx in range(needed):
        title = generate_compositional_title(rng, used)
        used.add(title)
        genre = str(rng.choice(GENRES, p=genre_probs))
        year = planned_years[idx] if planned_years else _year_sample(rng)
        rows.append({
            "title": title,
            "tagline": _tagline(genre, rng),
            "genre_hint": genre,
            "year": year,
            "award_contender": bool(rng.rand() < 0.08),
        })

    add_df = pd.DataFrame(rows)

    # Preserve any extra columns from existing bank.
    for col in df.columns:
        if col not in add_df.columns:
            add_df[col] = None
    for col in add_df.columns:
        if col not in df.columns:
            df[col] = None

    out = pd.concat([df[df.columns], add_df[df.columns]], ignore_index=True)
    out.to_csv(tb_path, index=False)
    print(f"Added {needed} titles to {tb_path}")
    print(f"New size: {len(out)}")


def main():
    parser = argparse.ArgumentParser(description="Deterministic title bank top-up")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--target-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()

    topup(
        Path(args.base_dir).resolve(),
        int(args.target_count),
        int(args.seed),
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
