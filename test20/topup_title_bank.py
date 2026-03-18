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


def topup(base_dir: Path, target_count: int, seed: int):
    tb_path = base_dir / "entities" / "title_bank.csv"
    if not tb_path.exists():
        raise FileNotFoundError(f"Missing title bank: {tb_path}")

    df = pd.read_csv(tb_path, low_memory=False)
    cur = len(df)
    if cur >= target_count:
        print(f"title_bank already has {cur} rows (target {target_count})")
        return

    rng = np.random.RandomState(seed)
    used = set(df["title"].astype(str).tolist()) if "title" in df.columns else set()

    rows = []
    needed = target_count - cur
    genre_probs = np.ones(len(GENRES), dtype=float) / max(1, len(GENRES))

    for _ in range(needed):
        title = generate_compositional_title(rng, used)
        used.add(title)
        genre = str(rng.choice(GENRES, p=genre_probs))
        rows.append({
            "title": title,
            "tagline": _tagline(genre, rng),
            "genre_hint": genre,
            "year": _year_sample(rng),
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
    args = parser.parse_args()

    topup(Path(args.base_dir).resolve(), int(args.target_count), int(args.seed))


if __name__ == "__main__":
    main()
