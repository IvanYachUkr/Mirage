"""
V19 Pipeline -- assembly.py
===========================
Movie component selection: director, cast, companies, crew, title, keywords.

This rewrite keeps the public API stable while cleaning up three chronic problems
from the previous version:

1. Too much orchestration leaked into local selection code.
2. Several scoring passes mixed vectorised blocks with thousands of tiny Python
   loops, so runtime scaled badly as the world grew.
3. Some temporal / event-system signals existed in world state but were not
   consistently consumed here, especially country overrides and temporal edge
   validity.

The module still exports the same core entry points expected by generate_movies:
    - sample_movie_concept
    - pick_director
    - pick_co_director
    - pick_companies
    - pick_cast
    - pick_title
    - pick_keywords
    - pick_crew
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from contracts import (
    ARCHETYPES,
    CAST_SIZE_RANGES,
    COUNTRIES,
    COUNTRY_LANGUAGE,
    COUNTRY_WEIGHTS,
    CREW_DEPARTMENTS,
    DECADE_WEIGHTS,
    FRANCHISE_CONFIG,
    GENRES,
    GENRE_WEIGHTS,
    PRODUCTION_TIERS,
    TIER_WEIGHTS,
    YEAR_RANGE,
    generate_compositional_title,
)
from financials import edge_is_active
from utils import (
    TIER_TO_LATENT_IDX,
    TONE_STYLE_HINTS,
    canonical_company_genre_vector,
    normalize_weights,
    project_genres_to_company_basis,
)
from world_state import WorldState, get_person_latent, latent_similarity_batch


# ---------------------------------------------------------------------------
# Geography / scale helpers
# ---------------------------------------------------------------------------

_NATIONALITY_TO_COUNTRY: dict[str, str] = {
    "American": "USA", "British": "UK", "French": "France", "German": "Germany",
    "Indian": "India", "Japanese": "Japan", "Chinese": "China", "Korean": "South Korea",
    "Italian": "Italy", "Spanish": "Spain", "Brazilian": "Brazil",
    "Mexican": "Mexico", "Australian": "Australia", "Canadian": "Canada",
    "Russian": "Russia", "Nigerian": "Nigeria", "Swedish": "Sweden",
    "Danish": "Denmark", "Norwegian": "Norway", "Polish": "Poland",
    "Turkish": "Turkey", "Iranian": "Iran", "Argentine": "Argentina",
    "Colombian": "Colombia", "Egyptian": "Egypt", "South African": "South Africa",
    "Thai": "Thailand", "Indonesian": "Indonesia", "Filipino": "Philippines",
    "Pakistani": "Pakistan", "Bangladeshi": "Bangladesh", "Greek": "Greece",
    "Dutch": "Netherlands", "Belgian": "Belgium", "Swiss": "Switzerland",
    "Austrian": "Austria", "Portuguese": "Portugal", "Czech": "Czech Republic",
    "Romanian": "Romania", "Hungarian": "Hungary", "Ukrainian": "Ukraine",
}

_GEO_BOOST_BY_TIER: dict[str, float] = {
    "Epic": 1.5,
    "A": 2.0,
    "Mid": 3.0,
    "Indie": 4.5,
    "Micro": 5.0,
}

_DYNAMIC_CAST_BASE: dict[str, tuple[int, int]] = {
    "Epic": (16, 44),
    "A": (9, 24),
    "Mid": (4, 12),
    "Indie": (2, 7),
    "Micro": (1, 4),
}

_BLOCKBUSTER_GENRES = {
    "Action", "Sci-Fi", "Fantasy", "Superhero", "Adventure", "War",
}

_MAJOR_HUBS = {
    "USA", "UK", "China", "India", "Japan", "France", "Germany",
    "South Korea", "Australia", "Canada", "Brazil", "Italy", "Spain",
}

_COUNTRY_TO_MARKET = {
    "USA": "North America",
    "Canada": "North America",
    "UK": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Sweden": "Europe",
    "India": "Asia",
    "Japan": "Asia",
    "South Korea": "Asia",
    "China": "Asia",
    "Australia": "Oceania",
    "Nigeria": "Africa",
    "Brazil": "South America",
    "Mexico": "Latin America",
    "Argentina": "Latin America",
}

_GENRE_TONE = {
    "Action": "intense",
    "Drama": "emotional",
    "Comedy": "light",
    "Sci-Fi": "cerebral",
    "Horror": "dark",
    "Romance": "warm",
    "Thriller": "suspenseful",
    "Fantasy": "epic",
    "Animation": "whimsical",
    "Documentary": "observational",
    "Crime": "gritty",
    "Mystery": "atmospheric",
    "War": "somber",
}

_GENRE_TIER_DIST = {
    "Action":      np.array([0.15, 0.30, 0.35, 0.15, 0.05]),
    "Sci-Fi":      np.array([0.12, 0.25, 0.35, 0.20, 0.08]),
    "Fantasy":     np.array([0.15, 0.30, 0.30, 0.18, 0.07]),
    "Animation":   np.array([0.10, 0.25, 0.40, 0.20, 0.05]),
    "Drama":       np.array([0.03, 0.12, 0.40, 0.35, 0.10]),
    "Comedy":      np.array([0.02, 0.10, 0.45, 0.35, 0.08]),
    "Thriller":    np.array([0.05, 0.15, 0.40, 0.30, 0.10]),
    "Horror":      np.array([0.02, 0.05, 0.25, 0.45, 0.23]),
    "Romance":     np.array([0.02, 0.08, 0.35, 0.40, 0.15]),
    "Documentary": np.array([0.00, 0.02, 0.15, 0.45, 0.38]),
    "Crime":       np.array([0.03, 0.12, 0.40, 0.35, 0.10]),
    "Mystery":     np.array([0.02, 0.10, 0.38, 0.38, 0.12]),
    "War":         np.array([0.06, 0.18, 0.34, 0.28, 0.14]),
}


# ---------------------------------------------------------------------------
# Character / tagline banks
# ---------------------------------------------------------------------------

_GENRE_CHAR_NAMES = {
    "action": {
        "M": ["Jake Reaper", "Sgt. Stone", "Rex Viper", "Duke Slater", "Marcus Blaze",
              "Hawk Jensen", "Brick Malone", "Logan Cruz", "Axel Storm", "Kane Bishop"],
        "F": ["Maya Cruz", "Elena Black", "Tara Fox", "Nina Cortez", "Sierra Voss",
              "Jordan Steele", "Athena Sharp", "Raven Cole", "Zara Knight", "Jade Fury"],
    },
    "comedy": {
        "M": ["Buddy Feldman", "Phil Bumble", "Benny Chuckles", "Gus Wobble", "Norm Dinkle",
              "Larry Fink", "Doug Peppers", "Ted Crumble", "Milo Pratt", "Chip Wadsworth"],
        "F": ["Liz Trotter", "Margot Fizz", "Diane Pratt", "Sally Sparks", "Patty Loop",
              "Dottie Banks", "Brenda Pluck", "Wendy Quirk", "Greta Bloom", "Faye Nibbles"],
    },
    "drama": {
        "M": ["Thomas Mercer", "William Hale", "James Whitfield", "Michael Carey", "Arthur Webb",
              "Daniel Cross", "Edward Blake", "Henry Thorne", "Robert Ashworth", "Samuel Voss"],
        "F": ["Claire Ashton", "Rebecca Forsythe", "Eleanor Voss", "Isabelle Dunn", "Catherine Pierce",
              "Margaret Hayes", "Vivian Cross", "Helen Marsh", "Grace Whitmore", "Lillian Ford"],
    },
    "sci-fi": {
        "M": ["Commander Kael", "Orion Voss", "Axel Quantum", "Professor Marsh", "Tau Epsilon",
              "Dr. Renn Solaris", "Major Atlas", "Cipher-9", "Zane Helix", "Capt. Holt"],
        "F": ["Dr. Nova Rix", "Zara-7", "Captain Sera Blaine", "Lyra Xenon", "Juno Six",
              "Aria Nebula", "Lt. Kira Voss", "Mira Starfall", "Echo Prime", "Dr. Elise Kepler"],
    },
    "horror": {
        "M": ["The Hollow Man", "Pastor Vex", "Dr. Graves", "The Watcher", "Father Morrow",
              "The Reaper", "Brother Silent", "Mr. Wilt", "The Surgeon", "Deacon Ash"],
        "F": ["Sister Agnes", "Darla Crane", "Mira Blackwood", "Evelyn Shade", "Ruby Thorn",
              "Moira Glass", "The Bride", "Nurse Hallow", "Lily Grave", "Constance Veil"],
    },
    "romance": {
        "M": ["Julien Marchand", "Daniel Hart", "Oliver Reed", "Marcus Cavanaugh", "Leo Ashford",
              "Sebastian Cole", "Ethan Sinclair", "Alexander Frost", "Gabriel Montague", "Theo Fairchild"],
        "F": ["Sophia Belmont", "Lily Fairweather", "Camille Laurent", "Rose Delacroix", "Natalie Summers",
              "Isabelle Chase", "Vivienne Hart", "Clara Beaumont", "Juliana Voss", "Eloise Wren"],
    },
    "thriller": {
        "M": ["Victor Kane", "Dr. Orin West", "Agent Cole Bishop", "Mason Hargrave", "Pierce Ashton",
              "Det. Jack Mercer", "Marcus Holt", "Raymond Cross", "Felix Strand", "Callum Grey"],
        "F": ["Det. Sarah Voss", "Nadia Cipher", "Claire Devlin", "Alina Markova", "Vera Cross",
              "Agent Lena Park", "Dr. Maren Hale", "Irina Kozlov", "Diana Thorn", "Cassandra Wren"],
    },
    "animation": {
        "N": ["Sparky", "Captain Fluffbeard", "Zippy the Fox", "Old Sage Oak", "Ember",
              "Pip Starlight", "Shadow Paw", "Snickers", "Blaze", "Whiskers",
              "Luna Moonwhisker", "Princess Coral", "Queen Blossom", "Petal", "Twinkle",
              "Prince Bramble", "Sir Hoot", "Nutmeg", "Frost", "Sunbeam"],
    },
    "fantasy": {
        "M": ["Lord Aldric", "Theron the Brave", "Grimjaw", "Prince Kael", "Dregan Ironhelm",
              "Sir Gideon", "Warden Kross", "Orin Darkfire", "Bael Stonehelm", "Ragnar Wolfsbane"],
        "F": ["Elara Windborne", "Lady Ashworth", "Sorceress Ilya", "Mira Frostweaver", "Celeste Moonshadow",
              "Queen Seraphina", "Priestess Yara", "Isolde Brightveil", "Runa Starwoven", "Sylve Thornheart"],
    },
    "documentary": {
        "N": ["Narrator", "Subject A", "Expert Witness", "The Director", "Interview Subject",
              "Commentator", "Field Reporter", "Historian", "Survivor", "Scientist",
              "Analyst", "Activist", "Witness", "Researcher", "Advocate"],
    },
}

_GENRE_KEY_MAP = {
    "science fiction": "sci-fi",
    "scifi": "sci-fi",
    "sci fi": "sci-fi",
    "animated": "animation",
    "adventure": "action",
    "war": "action",
    "mystery": "thriller",
    "crime": "thriller",
    "noir": "thriller",
    "musical": "comedy",
    "family": "animation",
    "historical": "drama",
    "period": "drama",
    "western": "action",
    "superhero": "action",
}

_GENRE_TO_ARCHETYPES = {
    "action": ["Lead Hero", "Lead Villain", "Sidekick", "Henchman", "Supporting"],
    "drama": ["Lead Hero", "Mentor", "Supporting", "Love Interest", "Authority Figure"],
    "comedy": ["Comic Relief", "Sidekick", "Love Interest", "Supporting", "Lead Hero"],
    "horror": ["Victim", "Lead Villain", "Lead Hero", "Mysterious Stranger", "Supporting"],
    "sci-fi": ["Lead Hero", "Lead Villain", "Sidekick", "Authority Figure", "Supporting"],
    "fantasy": ["Lead Hero", "Lead Villain", "Mentor", "Mysterious Stranger", "Sidekick"],
    "thriller": ["Lead Hero", "Lead Villain", "Mysterious Stranger", "Authority Figure", "Supporting"],
    "romance": ["Love Interest", "Lead Hero", "Sidekick", "Mentor", "Supporting"],
    "animation": ["Lead Hero", "Sidekick", "Lead Villain", "Comic Relief", "Supporting"],
    "documentary": ["Authority Figure", "Supporting", "Mentor", "Lead Hero"],
    "crime": ["Lead Hero", "Lead Villain", "Henchman", "Authority Figure", "Supporting"],
    "mystery": ["Lead Hero", "Lead Villain", "Mysterious Stranger", "Victim", "Supporting"],
    "war": ["Lead Hero", "Lead Villain", "Henchman", "Mentor", "Supporting"],
}

_TAGLINE_TEMPLATES = {
    "Action": [
        "No rules. No limits. No mercy.",
        "The only way out is through.",
        "When the dust settles, only one will stand.",
        "Payback has a new name.",
        "This time, it's personal.",
        "Some fights can't be won. This one must be.",
        "Heroes aren't born. They're forged.",
        "The clock is ticking.",
    ],
    "Comedy": [
        "Expect the unexpected. Then laugh.",
        "Life's a mess. Might as well enjoy it.",
        "Rules were made to be broken... hilariously.",
        "Some mistakes are worth repeating.",
        "You can't make this stuff up. Actually, we did.",
        "Normal is overrated.",
        "The worst plan ever... might just work.",
        "Chaos has never been this much fun.",
    ],
    "Drama": [
        "Every family has its secrets.",
        "The truth will set you free. Eventually.",
        "Some wounds never heal.",
        "A story that needs to be told.",
        "Behind every silence lies a story.",
        "The hardest battles are fought within.",
        "What we leave behind defines who we are.",
        "Sometimes the only way forward is to look back.",
    ],
    "Sci-Fi": [
        "The future is closer than you think.",
        "Humanity's greatest discovery... is its greatest threat.",
        "Beyond the stars. Beyond reason.",
        "Evolution doesn't ask permission.",
        "The universe doesn't care about your plans.",
        "First contact. Last chance.",
        "In space, the rules are different.",
        "What if everything you knew was designed?",
    ],
    "Horror": [
        "Don't look back.",
        "Some doors should stay closed.",
        "The darkness is listening.",
        "Fear is just the beginning.",
        "They're already inside.",
        "You can't escape what's in your head.",
        "Pray it doesn't find you.",
        "The dead don't rest here.",
    ],
    "Romance": [
        "Love finds a way. It always does.",
        "Two hearts. One chance.",
        "The greatest risk is not falling at all.",
        "Some stories are written in the stars.",
        "Love doesn't follow the rules.",
        "Where there's love, there's a way.",
        "Falling was the easy part.",
        "The heart wants what it wants.",
    ],
    "Thriller": [
        "Trust no one.",
        "The truth is the most dangerous weapon.",
        "Everyone has a breaking point.",
        "Nothing is what it seems.",
        "The game was rigged from the start.",
        "How far would you go to survive?",
        "Secrets have consequences.",
        "The closer you look, the less you see.",
    ],
    "Fantasy": [
        "Legends are born in darkness.",
        "A world beyond imagination.",
        "The prophecy was just the beginning.",
        "Magic always comes with a price.",
        "One realm. One chance. One destiny.",
        "The old magic is awakening.",
        "Kingdoms will fall. Heroes will rise.",
        "Beyond the veil lies another world.",
    ],
    "Animation": [
        "Adventure is just around the corner.",
        "Big dreams come in small packages.",
        "A journey beyond imagination.",
        "Some heroes are unexpected.",
        "Believe in the impossible.",
        "The adventure of a lifetime.",
    ],
    "Documentary": [
        "The story they didn't want told.",
        "Truth is stranger than fiction.",
        "See the world as it really is.",
        "A story that changes everything.",
        "What you don't know can change you.",
        "The untold story. Until now.",
    ],
    "Crime": [
        "In this city, everyone has a price.",
        "Justice has a dark side.",
        "The line between law and chaos.",
        "Every crime tells a story.",
        "Honor among thieves is a myth.",
        "The streets remember everything.",
    ],
    "Mystery": [
        "The answer is hiding in plain sight.",
        "Every clue leads deeper.",
        "Some puzzles are better left unsolved.",
        "The truth is buried. Start digging.",
        "What happened that night?",
        "Not everything can be explained.",
    ],
}

_CREW_DEPT = {
    "writer": "Production",
    "director_of_photography": "Camera",
    "cinematographer": "Camera",
    "editor": "Post-Production",
    "composer": "Sound",
    "sound_designer": "Sound",
    "costume_designer": "Art",
    "production_designer": "Art",
    "visual_effects_supervisor": "VFX",
    "vfx_supervisor": "VFX",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        val = float(default)
    else:
        try:
            val = float(raw)
        except Exception:
            val = float(default)
    if lo is not None:
        val = max(float(lo), val)
    if hi is not None:
        val = min(float(hi), val)
    return float(val)


def _workspace_priors(world: WorldState):
    return getattr(getattr(getattr(world, "workspace", None), "config", None), "priors", None)


def _prior_float(
    world: WorldState,
    name: str,
    default: float,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    priors = _workspace_priors(world)
    try:
        value = float(getattr(priors, name, default)) if priors is not None else float(default)
    except Exception:
        value = float(default)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return float(value)


def _safe01(value, default: float = 0.5) -> float:
    try:
        val = float(value)
    except Exception:
        val = float(default)
    if val != val:
        val = float(default)
    return float(max(0.0, min(1.0, val)))


def _normalise_dict_weights(weights: dict[str, float], floor: float = 1e-6) -> dict[str, float]:
    if not weights:
        return {}
    arr = {k: max(float(floor), float(v)) for k, v in weights.items()}
    s = sum(arr.values())
    if s <= 0:
        u = 1.0 / max(1, len(arr))
        return {k: u for k in arr}
    return {k: v / s for k, v in arr.items()}


def _cosine_sim(a, b) -> float:
    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_np) * np.linalg.norm(b_np))
    if denom < 1e-10:
        return 0.0
    return float(max(0.0, min(1.0, float(np.dot(a_np, b_np)) / denom)))


def _concept_csv_target(concept: dict) -> list[float]:
    genre = concept.get("genre", "Drama")
    tier = concept.get("tier", "Mid")
    genre_map = {
        "Action":      [0.3, 0.8, 0.4, 0.9, 0.3, 0.8, 0.7, 0.8],
        "Drama":       [0.6, 0.2, 0.7, 0.2, 0.5, 0.3, 0.3, 0.2],
        "Comedy":      [0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4],
        "Sci-Fi":      [0.4, 0.7, 0.3, 0.6, 0.4, 0.6, 0.7, 0.7],
        "Horror":      [0.3, 0.3, 0.4, 0.7, 0.3, 0.8, 0.4, 0.3],
        "Romance":     [0.7, 0.2, 0.6, 0.2, 0.8, 0.3, 0.3, 0.2],
        "Thriller":    [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.4, 0.4],
        "Fantasy":     [0.5, 0.8, 0.2, 0.5, 0.4, 0.5, 0.8, 0.8],
        "Animation":   [0.6, 0.6, 0.2, 0.5, 0.5, 0.4, 0.7, 0.6],
        "Documentary": [0.4, 0.1, 0.9, 0.1, 0.3, 0.6, 0.1, 0.1],
        "Crime":       [0.4, 0.4, 0.6, 0.5, 0.6, 0.5, 0.4, 0.4],
        "Mystery":     [0.5, 0.3, 0.5, 0.3, 0.6, 0.7, 0.3, 0.3],
        "War":         [0.3, 0.7, 0.5, 0.8, 0.3, 0.7, 0.6, 0.7],
    }
    tier_shift = {"Epic": 0.15, "A": 0.08, "Mid": 0.0, "Indie": -0.10, "Micro": -0.15}
    base = list(genre_map.get(genre, [0.5] * 8))
    shift = tier_shift.get(tier, 0.0)
    for dim in (1, 6, 7):
        base[dim] = max(-1.0, min(1.0, base[dim] + shift))
    return base


def _concept_latent_targets(concept: dict) -> tuple[float, float, float]:
    genre = str(concept.get("genre", "Drama"))
    tier = str(concept.get("tier", "Mid"))
    risk_target = {
        "Action": 0.68, "Thriller": 0.62, "Crime": 0.57, "Sci-Fi": 0.63,
        "Fantasy": 0.58, "Horror": 0.74, "Mystery": 0.53, "Comedy": 0.47,
        "Animation": 0.46, "Drama": 0.36, "Romance": 0.32, "Documentary": 0.24,
        "War": 0.59,
    }.get(genre, 0.50)
    ambition_target = {"Epic": 0.72, "A": 0.64, "Mid": 0.55, "Indie": 0.70, "Micro": 0.66}.get(tier, 0.56)
    prestige_target = {"Epic": 0.78, "A": 0.68, "Mid": 0.57, "Indie": 0.52, "Micro": 0.42}.get(tier, 0.56)
    return float(risk_target), float(ambition_target), float(prestige_target)


def _shortlist_budget(world: WorldState, kind: str, default: int) -> int:
    priors = _workspace_priors(world)
    base = int(default)
    if priors is not None:
        try:
            base = max(8, int(getattr(priors, "shortlist_size", default)))
        except Exception:
            base = int(default)
    if kind == "director":
        return max(12, min(64, max(12, base // 2)))
    if kind == "company":
        return max(10, min(40, max(10, base // 3)))
    if kind == "crew":
        return max(12, min(48, max(12, base // 2)))
    return max(8, int(base))


def _shortlist_indices(weights: np.ndarray, shortlist_size: int, rng, exploration_share: float = 0.25) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    valid = np.flatnonzero(w > 0)
    if valid.size <= shortlist_size:
        return valid
    shortlist_size = max(1, int(shortlist_size))
    exploration_share = float(np.clip(exploration_share, 0.0, 0.60))
    anchor_size = min(valid.size, max(1, int(round(shortlist_size * (1.0 - exploration_share)))))
    valid_weights = w[valid]
    anchor_local = np.argpartition(valid_weights, -anchor_size)[-anchor_size:]
    anchor = valid[anchor_local]
    anchor = anchor[np.argsort(w[anchor])[::-1]]
    explore_size = shortlist_size - anchor.size
    if explore_size <= 0:
        return anchor[:shortlist_size]
    remaining = np.setdiff1d(valid, anchor, assume_unique=False)
    if remaining.size == 0:
        return anchor[:shortlist_size]
    band_size = min(remaining.size, max(explore_size * 4, shortlist_size))
    band_local = np.argpartition(w[remaining], -band_size)[-band_size:]
    band = remaining[band_local]
    band_weights = normalize_weights(w[band])
    explore = rng.choice(band, size=min(explore_size, band.size), replace=False, p=band_weights)
    merged = np.unique(np.concatenate([anchor, explore]))
    merged = merged[np.argsort(w[merged])[::-1]]
    return merged[:shortlist_size]


def _active_year_subset(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if "debut_year" in df.columns:
        debut_ok = df["debut_year"].fillna(1900).astype(int) <= int(year)
    else:
        debut_ok = pd.Series(True, index=df.index)
    if "retirement_year" in df.columns:
        retire_ok = df["retirement_year"].fillna(2100).astype(float) >= float(year)
    else:
        retire_ok = pd.Series(True, index=df.index)
    out = df[debut_ok & retire_ok]
    return out if len(out) > 0 else df


def _edge_payload_weight(entry) -> float:
    if isinstance(entry, dict):
        return float(entry.get("weight", 0.0) or 0.0)
    try:
        return float(entry)
    except Exception:
        return 0.0


def _ensure_company_lookup_cache(world: WorldState) -> None:
    if getattr(world, "_company_by_tier_genre", None) is not None:
        return
    world._company_by_tier_genre = {}
    if world.companies is None or len(world.companies) == 0:
        return
    has_tier = "tier" in world.companies.columns
    has_genre = "specialty_genres" in world.companies.columns
    cids = world.companies["company_id"].astype(int).values
    tiers = world.companies["tier"].astype(str).values if has_tier else np.full(len(cids), "")
    genres = world.companies["specialty_genres"].fillna("").astype(str).values if has_genre else np.full(len(cids), "")
    for cid, tier, gspec in zip(cids, tiers, genres):
        pieces = [g.strip().lower() for g in str(gspec).replace(",", ";").split(";") if g.strip()]
        world._company_by_tier_genre.setdefault((str(tier), ""), set()).add(int(cid))
        for g in pieces:
            world._company_by_tier_genre.setdefault((str(tier), g), set()).add(int(cid))
            world._company_by_tier_genre.setdefault(("", g), set()).add(int(cid))


def _ensure_actor_workload_counter(world: WorldState) -> None:
    if getattr(world, "_yearly_workload", None) is not None:
        return
    world._yearly_workload = Counter()
    for pid, years in getattr(world, "person_recent", {}).items():
        for y in years:
            world._yearly_workload[(int(pid), int(y))] += 1


def _bounded_positions(weights: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(weights) == 0:
        return np.array([], dtype=np.int32)
    if len(weights) <= limit:
        return np.arange(len(weights), dtype=np.int32)
    top_count = max(1, min(limit, int(round(limit * 0.70))))
    if top_count >= len(weights):
        top_idx = np.arange(len(weights), dtype=np.int32)
    else:
        top_idx = np.argpartition(weights, -top_count)[-top_count:].astype(np.int32, copy=False)
    order = np.argsort(weights[top_idx])[::-1]
    top_idx = top_idx[order].astype(np.int32, copy=False)
    extra = max(0, limit - len(top_idx))
    if extra <= 0:
        return top_idx[:limit].astype(np.int32, copy=False)
    mask = np.ones(len(weights), dtype=bool)
    mask[top_idx] = False
    remaining = np.flatnonzero(mask)
    if remaining.size == 0:
        return top_idx[:limit].astype(np.int32, copy=False)
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    explore = rng.choice(remaining, size=min(extra, remaining.size), replace=False)
    return np.concatenate([top_idx, np.asarray(explore, dtype=np.int32)])[:limit].astype(np.int32, copy=False)


def _recent_window_count(world: WorldState, pid: int, year: int) -> float:
    _ensure_actor_workload_counter(world)
    workload = getattr(world, "_yearly_workload", None) or {}
    total = 0
    for offset in (-2, -1, 0, 1, 2):
        total += int(workload.get((int(pid), int(year + offset)), 0))
    return float(total)


def _bounded_neighbor_rows(
    world: WorldState,
    edge_type: str,
    pid: int,
    year: int,
    *,
    limit: int,
    seed: int,
) -> list[tuple[int, float, int, int]]:
    if limit <= 0:
        return []
    graph = getattr(world, "graph", None)
    if graph is not None and hasattr(graph, "sample_bounded_neighbors"):
        try:
            rows = graph.sample_bounded_neighbors(edge_type, int(pid), year, limit=int(limit), seed=int(seed))
            if rows:
                return rows
        except Exception:
            pass

    fallback = world._friend_adj_all.get(int(pid), []) if edge_type == "friendship" else world._rival_adj_all.get(int(pid), [])
    if len(fallback) <= limit:
        return list(fallback)
    weights = np.asarray([float(row[1]) for row in fallback], dtype=np.float32)
    chosen = _bounded_positions(weights, int(limit), int(seed))
    return [fallback[int(idx)] for idx in chosen]


@dataclass(slots=True)
class CastYearCache:
    year: int
    candidates: pd.DataFrame
    pids: np.ndarray
    pop_weight: np.ndarray
    peak_start: np.ndarray
    peak_end: np.ndarray
    stage_vals: np.ndarray
    career_stage_mult: np.ndarray
    yearly_max: np.ndarray
    actor_genders: np.ndarray | None
    actor_nationalities: np.ndarray | None
    nat_country: np.ndarray | None
    ga_lower: np.ndarray | None
    st_lower: np.ndarray | None
    market_fit_lower: np.ndarray | None
    actor_tag_sets: list[set[str]] | None
    actor_tag_bitmasks: np.ndarray | None
    li_arr: np.ndarray | None
    valid_li: np.ndarray | None
    cand_agencies: np.ndarray
    cand_communities: np.ndarray
    budget_pref: np.ndarray | None


@dataclass(slots=True)
class CrewYearPool:
    year: int
    role: str
    person_ids: np.ndarray
    weights: np.ndarray
    genre_affinity_lower: np.ndarray | None
    pid_to_local: dict[int, int]
    genre_match_cache: dict[str, np.ndarray] = field(default_factory=dict)
    genre_weight_cache: dict[str, np.ndarray] = field(default_factory=dict)
    genre_band_cache: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class SelectionYearState:
    year: int
    actor_cache: CastYearCache
    pid_to_local: dict[int, int]
    film_count: np.ndarray
    recent_window: np.ndarray
    yearly_workload: np.ndarray
    unused_flags: np.ndarray
    award_recent: np.ndarray
    actor_views: dict[tuple[Any, ...], "ActorStaticBlock"] = field(default_factory=dict)
    actor_pc_affinity_cache: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    director_pc_affinity_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def get_actor_view(self, concept_key: tuple[Any, ...]) -> "ActorStaticBlock" | None:
        return self.actor_views.get(tuple(concept_key))

    def record_cast_selection(self, actor_ids: Iterable[int], year: int) -> None:
        for pid in actor_ids:
            local_idx = self.pid_to_local.get(int(pid))
            if local_idx is None:
                continue
            self.film_count[local_idx] += 1.0
            self.yearly_workload[local_idx] += 1.0
            self.recent_window[local_idx] += 1.0
            self.unused_flags[local_idx] = False


@dataclass(slots=True)
class CastEnsembleState:
    cast_ids: list[int] = field(default_factory=list)
    cast_set: set[int] = field(default_factory=set)
    agencies: set[int] = field(default_factory=set)
    communities: set[int] = field(default_factory=set)
    genders: set[str] = field(default_factory=set)
    nationalities: set[str] = field(default_factory=set)
    tag_bitmasks: list[np.uint64] = field(default_factory=list)
    latent_anchor_ids: list[int] = field(default_factory=list)
    friend_frontier: dict[int, float] = field(default_factory=dict)
    rival_penalty: dict[int, float] = field(default_factory=dict)
    friend_frontier_vec: np.ndarray | None = None
    rival_penalty_vec: np.ndarray | None = None
    blocked_local_mask: np.ndarray | None = None
    frontier_local_idx: set[int] = field(default_factory=set)
    rival_local_idx: set[int] = field(default_factory=set)
    forbidden: set[int] = field(default_factory=set)

    def add_actor(self, world: WorldState, static: "ActorStaticBlock", local_idx: int, year: int) -> None:
        pid = int(static.pids[local_idx])
        if pid in self.cast_set:
            return
        self.cast_ids.append(pid)
        self.cast_set.add(pid)
        agency = int(static.cand_agencies[local_idx]) if len(static.cand_agencies) > local_idx else -1
        if agency >= 0:
            self.agencies.add(agency)
        community = int(static.cand_communities[local_idx]) if len(static.cand_communities) > local_idx else -1
        if community >= 0:
            self.communities.add(community)
        if static.actor_genders is not None:
            self.genders.add(str(static.actor_genders[local_idx]))
        if static.actor_nationalities is not None:
            self.nationalities.add(str(static.actor_nationalities[local_idx]))
        if static.actor_tag_bitmasks is not None:
            self.tag_bitmasks.append(np.uint64(static.actor_tag_bitmasks[local_idx]))
        if static.li_arr is not None:
            li = int(static.li_arr[local_idx])
            if li >= 0:
                self.latent_anchor_ids.append(li)
        if self.blocked_local_mask is not None and 0 <= local_idx < len(self.blocked_local_mask):
            self.blocked_local_mask[local_idx] = True

        friend_limit = max(24, min(96, max(24, len(static.pids) // 192)))
        rival_limit = max(16, min(72, max(16, len(static.pids) // 256)))
        base_seed = (
            int(getattr(world, "seed", 0)) * 1_315_423_911
            + int(year) * 2_654_435_761
            + int(pid) * 97_531
            + len(self.cast_ids) * 17
        ) & 0xFFFFFFFF
        friend_iter = _bounded_neighbor_rows(world, "friendship", pid, year, limit=friend_limit, seed=base_seed)
        rival_iter = _bounded_neighbor_rows(world, "rivalry", pid, year, limit=rival_limit, seed=base_seed ^ 0x9E3779B9)
        for nbr, weight, _vf, _vt in friend_iter:
            nbr_idx = static.pid_to_idx.get(int(nbr))
            if nbr_idx is None or int(nbr) in self.cast_set:
                continue
            boost = float(1.0 + 4.0 * max(0.15, float(weight)))
            self.friend_frontier[int(nbr_idx)] = max(
                float(self.friend_frontier.get(int(nbr_idx), 0.0)),
                boost,
            )
            if self.friend_frontier_vec is not None:
                self.friend_frontier_vec[int(nbr_idx)] = max(
                    float(self.friend_frontier_vec[int(nbr_idx)]),
                    boost,
                )
            self.frontier_local_idx.add(int(nbr_idx))
        for nbr, weight, _vf, _vt in rival_iter:
            nbr_idx = static.pid_to_idx.get(int(nbr))
            if nbr_idx is None or int(nbr) in self.cast_set:
                continue
            penalty = max(0.0, 1.0 - max(0.6, float(weight)))
            current = float(self.rival_penalty.get(int(nbr_idx), 1.0))
            self.rival_penalty[int(nbr_idx)] = min(current, penalty)
            if self.rival_penalty_vec is not None:
                self.rival_penalty_vec[int(nbr_idx)] = min(
                    float(self.rival_penalty_vec[int(nbr_idx)]),
                    penalty,
                )
            if self.blocked_local_mask is not None:
                self.blocked_local_mask[int(nbr_idx)] = True
            self.rival_local_idx.add(int(nbr_idx))
            self.forbidden.add(int(nbr))


def _build_year_mask(df: pd.DataFrame, year: int) -> np.ndarray:
    debut_arr = pd.to_numeric(df.get("debut_year", 1900), errors="coerce").fillna(1900).astype(int).to_numpy()
    retire_arr = pd.to_numeric(df.get("retirement_year", 2100), errors="coerce").fillna(2100).astype(float).to_numpy()
    mask = (debut_arr <= year) & (retire_arr >= float(year))
    if mask.sum() < 10:
        mask = np.ones(len(df), dtype=bool)
    return mask


def _get_cast_year_cache(world: WorldState, year: int) -> CastYearCache:
    cache_map = getattr(world, "_cast_year_cache", None)
    if cache_map is None:
        cache_map = {}
        world._cast_year_cache = cache_map
    cached = cache_map.get(int(year))
    if cached is not None:
        return cached

    if year in getattr(world, "_year_cache", {}):
        mask = world._year_cache[year]
    else:
        mask = _build_year_mask(world.actors, year)
        world._year_cache[year] = mask

    candidates = world.actors.loc[mask].reset_index(drop=True)
    pids = candidates["person_id"].astype(int).to_numpy()
    if "peak_start" in candidates.columns:
        peak_start = pd.to_numeric(candidates["peak_start"], errors="coerce").fillna(1900).astype(int).to_numpy()
    else:
        peak_start = np.full(len(candidates), 1900, dtype=int)
    if "peak_end" in candidates.columns:
        peak_end = pd.to_numeric(candidates["peak_end"], errors="coerce").fillna(1901).astype(int).to_numpy()
    else:
        peak_end = np.full(len(candidates), 1901, dtype=int)
    stage_vals = candidates["career_stage"].fillna("prime").astype(str).str.lower().to_numpy() if "career_stage" in candidates.columns else np.full(len(candidates), "prime")
    stage_map = {
        "legend": _env_float("V16_LEGEND_MULT", 15.0, 4.0, 30.0),
        "prime": _env_float("V16_PRIME_MULT", 5.0, 1.0, 12.0),
        "veteran": _env_float("V16_VETERAN_MULT", 2.5, 0.5, 10.0),
        "rising": _env_float("V16_RISING_MULT", 1.0, 0.2, 4.0),
        "retired": _env_float("V16_RETIRED_MULT", 0.08, 0.01, 1.5),
    }
    career_stage_mult = np.array([stage_map.get(str(s), 1.0) for s in stage_vals], dtype=float)
    yearly_max = candidates["yearly_max"].to_numpy(dtype=float) if "yearly_max" in candidates.columns else np.full(len(candidates), 5.0, dtype=float)
    yearly_max = np.where(np.isnan(yearly_max) | (yearly_max <= 0), 5.0, yearly_max)
    actor_genders = candidates["gender"].fillna("unknown").astype(str).str.lower().to_numpy() if "gender" in candidates.columns else None
    actor_nationalities = candidates["nationality"].fillna("unknown").astype(str).str.lower().to_numpy() if "nationality" in candidates.columns else None
    nat_country = None
    if "nationality" in candidates.columns:
        nat_country = np.array([_NATIONALITY_TO_COUNTRY.get(v, "") for v in candidates["nationality"].fillna("").astype(str).to_numpy()], dtype=object)
    if "_ga_lower" in candidates.columns:
        ga_lower = candidates["_ga_lower"].fillna("").astype(str).to_numpy()
    elif "genre_affinity" in candidates.columns:
        ga_lower = candidates["genre_affinity"].fillna("").astype(str).str.lower().to_numpy()
    else:
        ga_lower = None
    if "_st_lower" in candidates.columns:
        st_lower = candidates["_st_lower"].fillna("").astype(str).to_numpy()
    elif "style_tags" in candidates.columns:
        st_lower = candidates["style_tags"].fillna("").astype(str).str.lower().to_numpy()
    else:
        st_lower = None
    market_fit_lower = candidates["market_fit"].fillna("").astype(str).str.lower().to_numpy() if "market_fit" in candidates.columns else None
    actor_tag_sets = [set(t.strip().lower() for t in raw.replace(",", ";").split(";") if t.strip()) for raw in st_lower] if st_lower is not None else None
    tag_bit_map = _ensure_tag_bit_mapping(world)
    actor_tag_bitmasks = _build_tag_bitmasks(actor_tag_sets, tag_bit_map)
    latent_idx = getattr(world, "_latent_pid_to_idx", None)
    li_arr = np.array([latent_idx.get(int(pid), -1) for pid in pids], dtype=int) if latent_idx is not None else None
    valid_li = li_arr >= 0 if li_arr is not None else None
    cand_agencies = np.array([world.person_agency.get(int(pid), -1) for pid in pids], dtype=np.int32)
    communities = getattr(world, "communities", None) or {}
    cand_communities = np.array([communities.get(int(pid), -1) for pid in pids], dtype=np.int32)

    budget_pref = None
    if li_arr is not None and getattr(world, "_latent_bbp_normed", None) is not None:
        safe = np.clip(li_arr, 0, len(world._latent_bbp_normed) - 1)
        budget_pref = world._latent_bbp_normed[safe].copy()
        budget_pref[~valid_li] = 0.0

    cached = CastYearCache(
        year=int(year),
        candidates=candidates,
        pids=pids,
        pop_weight=candidates["pop_weight"].astype(float).to_numpy().copy(),
        peak_start=peak_start,
        peak_end=peak_end,
        stage_vals=stage_vals,
        career_stage_mult=career_stage_mult,
        yearly_max=yearly_max,
        actor_genders=actor_genders,
        actor_nationalities=actor_nationalities,
        nat_country=nat_country,
        ga_lower=ga_lower,
        st_lower=st_lower,
        market_fit_lower=market_fit_lower,
        actor_tag_sets=actor_tag_sets,
        actor_tag_bitmasks=actor_tag_bitmasks,
        li_arr=li_arr,
        valid_li=valid_li,
        cand_agencies=cand_agencies,
        cand_communities=cand_communities,
        budget_pref=budget_pref,
    )
    cache_map[int(year)] = cached
    return cached


def _award_recent_pid_set(world: WorldState, year: int) -> set[int]:
    award_recent_pids: set[int] = set()
    if getattr(world, "person_award_wins", None):
        for raw_pid, award_info in world.person_award_wins.items():
            pid = int(raw_pid)
            if isinstance(award_info, dict):
                if year - int(award_info.get("year", 0) or 0) <= 3:
                    award_recent_pids.add(pid)
            elif isinstance(award_info, int) and award_info > 0:
                award_recent_pids.add(pid)
    return award_recent_pids


def _get_selection_year_state(world: WorldState, year: int) -> SelectionYearState:
    cache_map = getattr(world, "_selection_year_state_cache", None)
    if cache_map is None:
        cache_map = {}
        world._selection_year_state_cache = cache_map
    cached = cache_map.get(int(year))
    if cached is not None:
        return cached

    actor_cache = _get_cast_year_cache(world, year)
    _ensure_actor_workload_counter(world)
    pids = actor_cache.pids
    recent_window = np.fromiter(
        (_recent_window_count(world, int(pid), int(year)) for pid in pids),
        dtype=float,
        count=len(pids),
    )
    cached = SelectionYearState(
        year=int(year),
        actor_cache=actor_cache,
        pid_to_local={int(pid): idx for idx, pid in enumerate(pids)},
        film_count=np.fromiter((world.person_film_count.get(int(pid), 0) for pid in pids), dtype=float, count=len(pids)),
        recent_window=recent_window,
        yearly_workload=np.fromiter((world._yearly_workload.get((int(pid), int(year)), 0) for pid in pids), dtype=float, count=len(pids)),
        unused_flags=(recent_window <= 0).astype(bool),
        award_recent=(
            np.isin(pids, np.array(sorted(_award_recent_pid_set(world, year)), dtype=int))
            if len(pids) else np.zeros(0, dtype=bool)
        ),
    )
    cache_map[int(year)] = cached
    return cached


def _get_crew_year_pool(world: WorldState, role: str, year: int) -> CrewYearPool | None:
    cache_map = getattr(world, "_crew_year_pool_cache", None)
    if cache_map is None:
        cache_map = {}
        world._crew_year_pool_cache = cache_map
    cache_key = (int(year), str(role))
    cached = cache_map.get(cache_key)
    if cached is not None:
        return cached

    pool = getattr(world, "crew_pools", {}).get(role) if hasattr(world, "crew_pools") else None
    if pool is None or len(pool) == 0:
        fallback_attr = CREW_DEPARTMENTS.get(role, {}).get("pool_fallback", "persons")
        pool = getattr(world, fallback_attr, None)
        if pool is None or len(pool) == 0:
            pool = world.actors
    if pool is None or len(pool) == 0:
        return None

    mask = _build_year_mask(pool, year)
    active = pool.loc[mask].reset_index(drop=True)
    if len(active) == 0:
        return None
    weights = active.get("pop_weight", pd.Series(np.ones(len(active), dtype=float))).astype(float).to_numpy()
    genre_affinity_lower = active["genre_affinity"].fillna("").astype(str).str.lower().to_numpy() if "genre_affinity" in active.columns else None
    cached = CrewYearPool(
        year=int(year),
        role=str(role),
        person_ids=active["person_id"].astype(int).to_numpy(),
        weights=weights,
        genre_affinity_lower=genre_affinity_lower,
        pid_to_local={int(pid): idx for idx, pid in enumerate(active["person_id"].astype(int).to_numpy())},
    )
    cache_map[cache_key] = cached
    return cached


def _crew_genre_match_mask(pool: CrewYearPool, genre: str) -> np.ndarray:
    genre_key = str(genre or "").lower().strip()
    if not genre_key or pool.genre_affinity_lower is None:
        return np.zeros(len(pool.person_ids), dtype=bool)
    cached = pool.genre_match_cache.get(genre_key)
    if cached is not None:
        return cached
    mask = np.char.find(pool.genre_affinity_lower.astype(str), genre_key) >= 0
    pool.genre_match_cache[genre_key] = mask
    return mask


def _crew_genre_weights(pool: CrewYearPool, genre: str) -> np.ndarray:
    genre_key = str(genre or "").lower().strip()
    if not genre_key:
        return pool.weights
    cached = pool.genre_weight_cache.get(genre_key)
    if cached is not None:
        return cached
    weights = pool.weights.copy()
    if pool.genre_affinity_lower is not None:
        weights *= (1.0 + 0.60 * _crew_genre_match_mask(pool, genre_key).astype(float))
    pool.genre_weight_cache[genre_key] = weights
    return weights


def _crew_candidate_band(world: WorldState, pool: CrewYearPool, genre: str, target_n: int) -> np.ndarray:
    genre_key = str(genre or "").lower().strip()
    cached = pool.genre_band_cache.get(genre_key)
    if cached is not None:
        return cached
    weights = _crew_genre_weights(pool, genre_key)
    shortlist = _shortlist_budget(world, "crew", max(32, int(target_n) * 8))
    band_size = min(len(pool.person_ids), max(shortlist * 6, 96))
    band = _shortlist_indices(
        weights,
        band_size,
        world.rng,
        exploration_share=_prior_float(world, "crew_exploration_share", 0.35, lo=0.0, hi=0.60),
    )
    if band.size == 0:
        band = np.flatnonzero(weights > 0)
    pool.genre_band_cache[genre_key] = band.astype(np.int32, copy=False)
    return pool.genre_band_cache[genre_key]


def _person_company_multiplier(world: WorldState, pids: np.ndarray, year: int, tier: str, genre: str) -> np.ndarray:
    # Cache P-C scoring once per actor year universe x (tier, genre).
    _ensure_company_lookup_cache(world)
    suitable = (
        world._company_by_tier_genre.get((tier, genre.lower()), set())
        | world._company_by_tier_genre.get(("", genre.lower()), set())
        | world._company_by_tier_genre.get((tier, ""), set())
    )
    if not suitable:
        return np.ones(len(pids), dtype=float)
    selection = _get_selection_year_state(world, year)
    cache_key = (str(tier), str(genre).lower())
    cached = selection.actor_pc_affinity_cache.get(cache_key)
    if cached is None or len(cached) != len(selection.actor_cache.pids):
        cached = world.compute_pc_affinity_batch(selection.actor_cache.pids, suitable)
        selection.actor_pc_affinity_cache[cache_key] = cached
    if np.array_equal(pids, selection.actor_cache.pids):
        return cached
    take = np.array([selection.pid_to_local.get(int(pid), -1) for pid in pids], dtype=int)
    out = np.ones(len(pids), dtype=float)
    valid = take >= 0
    out[valid] = cached[take[valid]]
    return out


def _director_company_multiplier(world: WorldState, dir_pids: np.ndarray, year: int, tier: str, genre: str) -> np.ndarray:
    _ensure_company_lookup_cache(world)
    suitable = (
        world._company_by_tier_genre.get((tier, genre.lower()), set())
        | world._company_by_tier_genre.get(("", genre.lower()), set())
        | world._company_by_tier_genre.get((tier, ""), set())
    )
    if not suitable:
        return np.ones(len(dir_pids), dtype=float)
    selection = _get_selection_year_state(world, year)
    cache_key = (str(tier), str(genre).lower())
    cached = selection.director_pc_affinity_cache.get(cache_key)
    if cached is None or len(cached[0]) != len(dir_pids) or not np.array_equal(cached[0], dir_pids):
        cached = (dir_pids.copy(), world.compute_pc_affinity_batch(dir_pids, suitable))
        selection.director_pc_affinity_cache[cache_key] = cached
    return cached[1]


def _director_edge_arrays(world: WorldState, director_id: int, pids: np.ndarray, year: int) -> tuple[np.ndarray, np.ndarray]:
    cache_map = getattr(world, "_director_year_edge_cache", None)
    if cache_map is None:
        cache_map = {}
        world._director_year_edge_cache = cache_map
    cache_key = (int(year), int(director_id))
    cached = cache_map.get(cache_key)
    if cached is None:
        pref_map: dict[int, float] = {}
        avoid_set: set[int] = set()
        graph = getattr(world, "graph", None)
        pref_edges = graph.get_director_prefs(int(director_id), year) if graph is not None else []
        for aid, weight, _valid_from, _valid_to in pref_edges:
            pref_map[int(aid)] = float(weight)
        avoid_edges = graph.get_director_avoids(int(director_id), year) if graph is not None else []
        for aid, _valid_from, _valid_to in avoid_edges:
            avoid_set.add(int(aid))
        cached = (pref_map, avoid_set)
        cache_map[cache_key] = cached
    else:
        pref_map, avoid_set = cached

    pref = np.ones(len(pids), dtype=float)
    if pref_map:
        arr = np.array([pref_map.get(int(pid), -1.0) for pid in pids], dtype=float)
        mask = arr >= 0.0
        arr = np.clip(arr, 0.0, 1.0)
        pref[mask] = 1.0 + 9.0 * arr[mask]

    avoid_mask = np.isin(pids, np.array(list(avoid_set), dtype=int)) if avoid_set else np.zeros(len(pids), dtype=bool)
    return pref, avoid_mask


@dataclass(slots=True)
class ActorStaticBlock:
    key: tuple[Any, ...]
    candidate_idx: np.ndarray
    pids: np.ndarray
    immutable_scores: np.ndarray
    base_focus_idx: np.ndarray
    yearly_max: np.ndarray
    stage_vals: np.ndarray
    cooldown_floor: np.ndarray
    cooldown_decay: np.ndarray
    li_arr: np.ndarray | None
    actor_genders: np.ndarray | None
    actor_nationalities: np.ndarray | None
    actor_tag_sets: list[set[str]] | None
    actor_tag_bitmasks: np.ndarray | None  # uint64 bitmask per candidate
    pid_to_idx: dict[int, int]
    top_star_mask: np.ndarray
    cand_agencies: np.ndarray
    cand_communities: np.ndarray
    _sim_threshold: float


ActorConceptView = ActorStaticBlock


# ---------------------------------------------------------------------------
# Tag bitmask helpers (vectorized Jaccard over uint64 arrays)
# ---------------------------------------------------------------------------

def _popcount_vec(arr: np.ndarray) -> np.ndarray:
    """Vectorized popcount for a uint64 numpy array (bit-parallel)."""
    x = arr.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def _ensure_tag_bit_mapping(world: WorldState) -> dict[str, int]:
    """Build a tag → bit-position mapping (cached on world, O(N) once)."""
    if getattr(world, "_tag_bit_map", None) is not None:
        return world._tag_bit_map
    all_tags: set[str] = set()
    for df in (world.actors, world.persons):
        if df is not None and "style_tags" in df.columns:
            for raw in df["style_tags"].fillna("").astype(str).values:
                for t in raw.replace(",", ";").split(";"):
                    t = t.strip().lower()
                    if t:
                        all_tags.add(t)
            break
    sorted_tags = sorted(all_tags)[:64]  # uint64 supports up to 64 tags
    world._tag_bit_map = {tag: i for i, tag in enumerate(sorted_tags)}
    return world._tag_bit_map


def _build_tag_bitmasks(tag_sets: list[set[str]] | None, bit_map: dict[str, int]) -> np.ndarray | None:
    """Convert a list of tag sets into a uint64 bitmask array."""
    if tag_sets is None:
        return None
    n = len(tag_sets)
    bitmasks = np.zeros(n, dtype=np.uint64)
    for i, tags in enumerate(tag_sets):
        mask = np.uint64(0)
        for t in tags:
            bit = bit_map.get(t)
            if bit is not None:
                mask |= np.uint64(1) << np.uint64(bit)
        bitmasks[i] = mask
    return bitmasks


def _build_actor_static_block(world: WorldState, concept: dict) -> ActorStaticBlock:
    year = int(concept["year"])
    genre = str(concept["genre"])
    tier = str(concept["tier"])
    tone = str(concept.get("tone", "neutral"))
    movie_country = str(concept.get("country", ""))
    franchise = concept.get("franchise")
    selection = _get_selection_year_state(world, year)
    cache = selection.actor_cache
    concept_key = (
        int(year),
        genre.lower(),
        str(tier),
        movie_country,
        tone.lower(),
        bool(franchise and franchise.get("movies_generated", 0) > 0),
    )
    cached = selection.get_actor_view(concept_key)
    if cached is not None:
        return cached

    cast_size = _sample_cast_size(world, concept)
    candidate_idx = np.arange(len(cache.pids), dtype=np.int32)
    min_pool = max(cast_size * 4, 50)
    genre_lower = genre.lower()

    if candidate_idx.size > min_pool and cache.ga_lower is not None:
        genre_mask = np.char.find(cache.ga_lower.astype(str), genre_lower) >= 0
        filtered_idx = candidate_idx[genre_mask[candidate_idx]]
        if filtered_idx.size >= min_pool:
            candidate_idx = filtered_idx

    if candidate_idx.size > min_pool and cache.budget_pref is not None:
        tier_idx = TIER_TO_LATENT_IDX.get(tier, 2)
        tier_ok = cache.budget_pref[candidate_idx, tier_idx] >= 0.05
        filtered_idx = candidate_idx[tier_ok]
        if filtered_idx.size >= min_pool:
            candidate_idx = filtered_idx

    pids = cache.pids[candidate_idx]
    n = len(candidate_idx)
    base_scores = cache.pop_weight[candidate_idx].copy()

    peak_s = cache.peak_start[candidate_idx]
    peak_e = cache.peak_end[candidate_idx]
    valid = peak_e >= peak_s
    in_peak = valid & (peak_s <= year) & (year <= peak_e)
    career_mult = np.where(in_peak, 3.0, 1.0)

    stage_vals = cache.stage_vals[candidate_idx]
    career_stage_mult = cache.career_stage_mult[candidate_idx]

    geo_boost = _GEO_BOOST_BY_TIER.get(tier, 2.0)
    if movie_country and cache.nat_country is not None:
        nat_country = cache.nat_country[candidate_idx]
        nationality_mult = np.where(nat_country == movie_country, geo_boost, 1.0)
        actor_nats = cache.actor_nationalities[candidate_idx] if cache.actor_nationalities is not None else None
    else:
        nationality_mult = np.ones(n, dtype=float)
        actor_nats = None

    if cache.ga_lower is not None:
        genre_match = np.char.find(cache.ga_lower[candidate_idx].astype(str), genre_lower) >= 0
    else:
        genre_match = np.zeros(n, dtype=bool)
    genre_mult = np.where(genre_match, 5.0, 1.0)

    hints = TONE_STYLE_HINTS.get(tone, [tone])
    if hints:
        if cache.st_lower is not None:
            st_values = cache.st_lower[candidate_idx].astype(str)
            tagged = np.array([f";{raw.replace(',', ';')};" for raw in st_values], dtype=object)
            style_match = np.zeros(n, dtype=bool)
            for hint in hints:
                if hint:
                    style_match |= np.char.find(tagged.astype(str), f";{hint};") >= 0
            style_boost = _prior_float(world, "cast_style_multiplier", 2.0, lo=1.0, hi=4.0)
            style_mult = np.where(style_match, style_boost, 1.0)
            actor_tag_sets = [cache.actor_tag_sets[int(i)] for i in candidate_idx] if cache.actor_tag_sets is not None else None
        else:
            style_mult = np.ones(n, dtype=float)
            actor_tag_sets = None
    else:
        style_mult = np.ones(n, dtype=float)
        actor_tag_sets = [cache.actor_tag_sets[int(i)] for i in candidate_idx] if cache.actor_tag_sets is not None else None

    avoid_genre_mult = np.ones(n, dtype=float)
    sparse_avoid = getattr(world, "_latent_avoid_genres", None)
    if sparse_avoid:
        for i, pid in enumerate(pids):
            if genre in sparse_avoid.get(int(pid), set()):
                avoid_genre_mult[i] = 0.15

    market_mult = np.ones(n, dtype=float)
    if cache.market_fit_lower is not None and movie_country:
        target_market = _COUNTRY_TO_MARKET.get(movie_country, "")
        if target_market:
            mf = cache.market_fit_lower[candidate_idx].astype(str)
            market_mult = np.where(
                (np.char.find(mf, "global") >= 0) | (np.char.find(mf, target_market.lower()) >= 0),
                2.0,
                1.0,
            )

    collab_mult = np.ones(n, dtype=float)
    latent_collab = getattr(world, "_latent_collab", None)
    li_arr = cache.li_arr[candidate_idx] if cache.li_arr is not None else None
    valid_li = cache.valid_li[candidate_idx] if cache.valid_li is not None else None

    if li_arr is not None and latent_collab is not None and valid_li.any():
        safe_li = np.clip(li_arr, 0, len(latent_collab) - 1)
        styles = latent_collab[safe_li]
        cs = cast_size
        style_to_mult = {
            "solo": 0.6 if cs >= 6 else (1.5 if cs <= 3 else 1.0),
            "ensemble": 2.0 if cs >= 6 else (0.8 if cs <= 2 else 1.2),
            "chameleon": 1.0,
            "mentorship": 1.3 if cs >= 4 else 1.0,
        }
        for style_name, mult in style_to_mult.items():
            mask = valid_li & (styles == style_name)
            collab_mult[mask] = mult

    tier_ctro = {"Epic": 2.5, "A": 1.8, "Mid": 1.0, "Indie": 0.3, "Micro": 0.1}.get(tier, 1.0)
    controversy_mult = np.ones(n, dtype=float)
    if li_arr is not None and getattr(world, "_latent_controversy", None) is not None and tier_ctro > 0:
        safe = np.clip(li_arr, 0, len(world._latent_controversy) - 1)
        vals = np.where(valid_li, world._latent_controversy[safe], 0.15)
        controversy_mult = np.maximum(0.2, 1.0 - tier_ctro * vals * 0.4)

    tier_vol = {"Epic": 1.5, "A": 1.0, "Mid": 0.5, "Indie": 0.0, "Micro": 0.0}.get(tier, 0.5)
    volatility_mult = np.ones(n, dtype=float)
    if li_arr is not None and getattr(world, "_latent_volatility", None) is not None and tier_vol > 0:
        safe = np.clip(li_arr, 0, len(world._latent_volatility) - 1)
        vals = np.where(valid_li, world._latent_volatility[safe], 0.4)
        volatility_mult = np.maximum(0.4, 1.0 - tier_vol * vals * 0.4)

    csv_mult = np.ones(n, dtype=float)
    csv_target = np.asarray(_concept_csv_target(concept), dtype=np.float32)
    norm = float(np.linalg.norm(csv_target))
    csv_target = csv_target / norm if norm > 1e-10 else csv_target
    if li_arr is not None and getattr(world, "_latent_csv_normed", None) is not None:
        safe = np.clip(li_arr, 0, len(world._latent_csv_normed) - 1)
        sims = np.clip(world._latent_csv_normed[safe] @ csv_target, 0.0, 1.0)
        sims[~valid_li] = 0.0
        csv_mult = 0.7 + 0.8 * sims

    company_aff_mult = _person_company_multiplier(world, pids, year, tier, genre)

    actor_genders = cache.actor_genders[candidate_idx] if cache.actor_genders is not None else None
    cand_agencies = cache.cand_agencies[candidate_idx]
    cand_communities = cache.cand_communities[candidate_idx]
    immutable_scores = base_scores.copy()
    immutable_scores *= career_mult
    immutable_scores *= career_stage_mult
    immutable_scores *= nationality_mult
    immutable_scores *= genre_mult
    immutable_scores *= style_mult
    immutable_scores *= avoid_genre_mult
    immutable_scores *= market_mult
    immutable_scores *= collab_mult
    immutable_scores *= controversy_mult
    immutable_scores *= volatility_mult
    immutable_scores *= csv_mult
    immutable_scores *= company_aff_mult
    top_cut = float(np.quantile(immutable_scores, 0.95)) if len(immutable_scores) > 30 else float(np.max(immutable_scores) if len(immutable_scores) else 0.0)
    top_star_mask = immutable_scores >= top_cut
    sorted_base = np.sort(immutable_scores)[::-1]
    sim_threshold = float(sorted_base[min(199, len(sorted_base) - 1)]) * 0.5 if len(sorted_base) else 0.0
    base_focus_size = min(
        len(candidate_idx),
        max(_shortlist_budget(world, "cast", 120) * 5, 320),
    )
    base_focus_idx = _shortlist_indices(
        immutable_scores,
        base_focus_size,
        world.rng,
        exploration_share=_prior_float(world, "cast_base_focus_exploration", 0.45, lo=0.0, hi=0.60),
    )

    cooldown_floor = np.ones(n, dtype=np.float32)
    cooldown_decay = np.full(n, 0.15, dtype=np.float32)
    if len(stage_vals):
        stage_vals_arr = np.asarray(stage_vals, dtype=object)
        for stage_name, floor in {
            "legend": 0.70,
            "prime": 0.45,
            "veteran": 0.50,
            "rising": 0.25,
            "retired": 0.05,
        }.items():
            stage_mask = stage_vals_arr == stage_name
            if not np.any(stage_mask):
                continue
            cooldown_floor[stage_mask] = np.float32(floor)
            cooldown_decay[stage_mask] = np.float32({
                "legend": 0.07,
                "prime": 0.13,
                "veteran": 0.10,
                "rising": 0.18,
                "retired": 0.30,
            }.get(stage_name, 0.15))

    actor_tag_bitmasks = cache.actor_tag_bitmasks[candidate_idx].copy() if cache.actor_tag_bitmasks is not None else None

    block = ActorStaticBlock(
        key=concept_key,
        candidate_idx=candidate_idx.copy(),
        pids=pids,
        immutable_scores=immutable_scores,
        base_focus_idx=base_focus_idx.astype(np.int32, copy=False),
        yearly_max=cache.yearly_max[candidate_idx].copy(),
        stage_vals=stage_vals,
        cooldown_floor=cooldown_floor,
        cooldown_decay=cooldown_decay,
        li_arr=li_arr,
        actor_genders=actor_genders,
        actor_nationalities=actor_nats,
        actor_tag_sets=actor_tag_sets,
        actor_tag_bitmasks=actor_tag_bitmasks,
        pid_to_idx={int(pid): i for i, pid in enumerate(pids)},
        top_star_mask=top_star_mask,
        cand_agencies=cand_agencies,
        cand_communities=cand_communities,
        _sim_threshold=sim_threshold,
    )
    selection.actor_views[concept_key] = block
    return block


def _expand_cast_focus_indices(
    world: WorldState,
    static: ActorStaticBlock,
    scores: np.ndarray,
    cast_id_list: list[int],
    year: int,
    shortlist_size: int,
) -> np.ndarray:
    focus = _shortlist_indices(
        scores,
        max(shortlist_size * 2, 96),
        world.rng,
        exploration_share=_prior_float(world, "cast_focus_exploration", 0.40, lo=0.0, hi=0.60),
    )
    if focus.size == 0 or not cast_id_list:
        return focus
    graph = getattr(world, "graph", None)
    frontier: list[int] = []
    for cid in cast_id_list:
        friend_iter = graph.iter_friend_neighbors(int(cid), year) if graph is not None else world._friend_adj_all.get(int(cid), [])
        rival_iter = graph.iter_rival_neighbors(int(cid), year) if graph is not None else world._rival_adj_all.get(int(cid), [])
        for nbr, _w, _vf, _vt in friend_iter:
            idx = static.pid_to_idx.get(int(nbr))
            if idx is not None and scores[idx] > 0:
                frontier.append(int(idx))
        for nbr, _w, _vf, _vt in rival_iter:
            idx = static.pid_to_idx.get(int(nbr))
            if idx is not None and scores[idx] > 0:
                frontier.append(int(idx))
    if not frontier:
        return focus
    frontier_arr = np.unique(np.asarray(frontier, dtype=np.int32))
    merged = np.unique(np.concatenate([focus.astype(np.int32, copy=False), frontier_arr]))
    cap = max(shortlist_size * 3, 160)
    if merged.size > cap:
        merged = merged[np.argsort(scores[merged])[::-1][:cap]]
    return merged


def _expand_cast_focus_indices_cached(
    world: WorldState,
    scores: np.ndarray,
    shortlist_size: int,
    frontier_local_indices: Iterable[int] | None = None,
) -> np.ndarray:
    focus = _shortlist_indices(
        scores,
        max(shortlist_size * 2, 96),
        world.rng,
        exploration_share=_prior_float(world, "cast_focus_exploration", 0.40, lo=0.0, hi=0.60),
    )
    extras: list[int] = []
    if frontier_local_indices is not None:
        for idx in frontier_local_indices:
            local_idx = int(idx)
            if 0 <= local_idx < len(scores) and scores[local_idx] > 0:
                extras.append(local_idx)
    if not extras:
        return focus
    extra_arr = np.unique(np.asarray(extras, dtype=np.int32))
    if focus.size == 0:
        return extra_arr
    merged = np.unique(np.concatenate([focus.astype(np.int32, copy=False), extra_arr]))
    cap = max(shortlist_size * 3, 160)
    if merged.size > cap:
        merged = merged[np.argsort(scores[merged])[::-1][:cap]]
    return merged


def _cap_focus_indices(scores: np.ndarray, indices: np.ndarray, cap: int) -> np.ndarray:
    if indices.size <= cap:
        return indices
    return indices[np.argsort(scores[indices])[::-1][:cap]]


def _priority_cast_indices(
    static: ActorStaticBlock,
    director_pref_boost: np.ndarray,
    award_mask: np.ndarray,
    franchise_pool_mask: np.ndarray,
    limit: int,
) -> np.ndarray:
    extras: list[np.ndarray] = []
    if np.any(director_pref_boost > 1.0):
        pref = np.flatnonzero(director_pref_boost > 1.0)
        if pref.size > limit:
            pref = pref[np.argsort(static.immutable_scores[pref] * director_pref_boost[pref])[::-1][:limit]]
        extras.append(pref.astype(np.int32, copy=False))
    if np.any(award_mask):
        award_idx = np.flatnonzero(award_mask)
        if award_idx.size > limit:
            award_idx = award_idx[np.argsort(static.immutable_scores[award_idx])[::-1][:limit]]
        extras.append(award_idx.astype(np.int32, copy=False))
    if np.any(franchise_pool_mask):
        pool_idx = np.flatnonzero(franchise_pool_mask)
        if pool_idx.size > limit:
            pool_idx = pool_idx[np.argsort(static.immutable_scores[pool_idx])[::-1][:limit]]
        extras.append(pool_idx.astype(np.int32, copy=False))
    if not extras:
        return np.zeros(0, dtype=np.int32)
    merged = np.unique(np.concatenate(extras))
    if merged.size > limit * 2:
        merged = _cap_focus_indices(static.immutable_scores, merged, limit * 2)
    return merged.astype(np.int32, copy=False)


def _sample_cast_size(world: WorldState, concept: dict) -> int:
    tier = str(concept.get("tier", "Mid"))
    genre = str(concept.get("genre", "Drama"))
    franchise = concept.get("franchise")
    lo, hi = _DYNAMIC_CAST_BASE.get(tier, CAST_SIZE_RANGES.get(tier, (3, 8)))
    if genre in _BLOCKBUSTER_GENRES:
        hi += 8 if tier in ("Epic", "A") else 2
    if franchise is not None:
        hi += 10 if tier in ("Epic", "A") else 2
    cast_max_cap = int(_env_float("V16_CAST_MAX", 120.0, 40.0, 240.0))
    hi = max(lo, min(hi, cast_max_cap))
    if tier == "Epic":
        tail_p = 0.10 + (0.08 if franchise is not None else 0.0) + (0.05 if genre in _BLOCKBUSTER_GENRES else 0.0)
        tail_p *= _env_float("V16_EPIC_TAIL_SCALE", 1.0, 0.4, 3.0)
        if world.rng.random() < min(0.55, tail_p):
            return int(np.clip(world.rng.lognormal(mean=3.75, sigma=0.42), 48, cast_max_cap))
    if tier == "A" and (franchise is not None or genre in _BLOCKBUSTER_GENRES):
        a_tail_p = _env_float("V16_A_TAIL_PROB", 0.06, 0.01, 0.25)
        if world.rng.random() < a_tail_p:
            hi_a = int(min(cast_max_cap, 86))
            return int(world.rng.randint(30, max(31, hi_a)))
    return int(world.rng.randint(lo, hi + 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_movie_concept(world: WorldState, movie_id: int, forced_year: int = None) -> dict:
    rng = world.rng
    franchise = world.movie_franchise_map.get(movie_id)
    if franchise:
        genre = franchise["genre"]
        tier = franchise["tier"]
        installment = franchise["movies_generated"] + 1
    else:
        genre_weights = dict(GENRE_WEIGHTS)
        for g, delta in getattr(world, "genre_weight_overrides", {}).items():
            if g in genre_weights:
                genre_weights[g] = max(0.005, float(genre_weights[g]) + float(delta))
        genre_weights = _normalise_dict_weights(genre_weights)
        genre = rng.choice(list(genre_weights.keys()), p=list(genre_weights.values()))
        tier_dist = _GENRE_TIER_DIST.get(genre)
        if tier_dist is not None:
            tier = rng.choice(PRODUCTION_TIERS, p=tier_dist / tier_dist.sum())
        else:
            base_tier = _normalise_dict_weights({k: float(v) for k, v in TIER_WEIGHTS.items()})
            tier = rng.choice(list(base_tier.keys()), p=list(base_tier.values()))
        installment = None

    if forced_year is not None:
        year = int(forced_year)
    elif YEAR_RANGE:
        year = int(rng.randint(YEAR_RANGE[0], YEAR_RANGE[1] + 1))
    else:
        decades = list(DECADE_WEIGHTS.keys())
        dweights = _normalise_dict_weights({str(k): float(v) for k, v in DECADE_WEIGHTS.items()})
        decade = int(rng.choice(decades, p=[dweights[str(k)] for k in decades]))
        year = int(decade + rng.randint(0, 10))

    country_weights = {k: float(v) for k, v in COUNTRY_WEIGHTS.items()}
    for c, multiplier in getattr(world, "country_weight_overrides", {}).items():
        if c in country_weights:
            country_weights[c] = max(1e-6, country_weights[c] * float(multiplier))
    country_weights = _normalise_dict_weights(country_weights)
    country = rng.choice(list(country_weights.keys()), p=list(country_weights.values()))
    language = COUNTRY_LANGUAGE.get(country, "English")

    if franchise is None and country not in _MAJOR_HUBS and tier in ("Epic", "A"):
        tier = "Mid"

    month_weights = np.array([0.06, 0.06, 0.07, 0.07, 0.10, 0.11, 0.11, 0.10, 0.07, 0.07, 0.09, 0.09], dtype=float)
    if genre in ("Horror", "Thriller") and rng.random() < 0.4:
        month_weights[8] += 0.06
        month_weights[9] += 0.04
    elif genre == "Romance" and rng.random() < 0.3:
        month_weights[1] += 0.06
        month_weights[5] += 0.04
    elif genre in ("Action", "Sci-Fi", "Fantasy") and rng.random() < 0.35:
        month_weights[4] += 0.05
        month_weights[5] += 0.04
    elif genre == "Drama" and rng.random() < 0.3:
        month_weights[8] += 0.04
        month_weights[9] += 0.06
    month = int(rng.choice(range(1, 13), p=month_weights / month_weights.sum()))

    writer_director_prob = {"Indie": 0.20, "Micro": 0.25, "Mid": 0.12, "A": 0.08, "Epic": 0.06}
    return {
        "movie_id": int(movie_id),
        "genre": genre,
        "tier": tier,
        "year": int(year),
        "country": country,
        "language": language,
        "tone": _GENRE_TONE.get(genre, "neutral"),
        "month": month,
        "franchise": franchise,
        "installment": installment,
        "is_writer_director": bool(rng.random() < writer_director_prob.get(tier, 0.10)),
    }


def pick_director(world: WorldState, concept: dict) -> int | None:
    if len(world.directors) == 0:
        return None
    genre = str(concept["genre"])
    tier = str(concept.get("tier", "Mid"))
    franchise = concept.get("franchise")
    year = int(concept.get("year", 2000))

    if franchise and franchise.get("director_id") is not None and world.rng.random() < 0.80:
        return int(franchise["director_id"])

    dirs = _active_year_subset(world.directors, year)
    if "debut_year" in dirs.columns:
        span = year - dirs["debut_year"].fillna(1970).astype(int)
        soft = dirs[(span >= 0) & (span <= 60)]
        if len(soft) >= 3:
            dirs = soft

    weights = dirs["pop_weight"].astype(float).values.copy()
    if "genre_affinity" in dirs.columns:
        ga = dirs["genre_affinity"].fillna("").astype(str).str.lower()
        weights *= np.where(ga.str.contains(genre.lower(), regex=False).values, 3.0, 1.0)

    movie_country = str(concept.get("country", ""))
    if movie_country and "nationality" in dirs.columns:
        nat = dirs["nationality"].fillna("").astype(str).values
        dir_country = np.array([_NATIONALITY_TO_COUNTRY.get(n, "") for n in nat])
        geo_boost = _GEO_BOOST_BY_TIER.get(tier, 2.0) * 0.6
        weights *= np.where(dir_country == movie_country, max(1.5, geo_boost), 1.0)

    dir_pids = dirs["person_id"].astype(int).values
    film_counts = np.array([world.director_film_count.get(int(pid), 0) for pid in dir_pids], dtype=float)
    weights *= np.where(film_counts > 80, 0.3, np.where(film_counts > 50, 0.5, np.where(film_counts > 30, 0.7, 1.0)))

    shortlist = _shortlist_indices(
        weights,
        _shortlist_budget(world, "director", 24),
        world.rng,
        exploration_share=_prior_float(world, "director_exploration_share", 0.30, lo=0.0, hi=0.60),
    )
    if shortlist.size > 0 and shortlist.size < len(dirs):
        dirs = dirs.iloc[shortlist]
        weights = weights[shortlist]
        dir_pids = dirs["person_id"].astype(int).values

    risk_target, ambition_target, prestige_target = _concept_latent_targets(concept)
    alignment = np.ones(len(dirs), dtype=float)
    latent_lookup = getattr(world, "_latent_pid_to_idx", {}) or {}
    latent_indices = np.array([latent_lookup.get(int(pid), -1) for pid in dir_pids], dtype=int)
    valid_latent = latent_indices >= 0
    if valid_latent.any():
        valid_idx = latent_indices[valid_latent]
        risk_align = 1.0 - np.abs(world._latent_risk[valid_idx] - risk_target)
        ambition_align = 1.0 - np.abs(world._latent_ambition[valid_idx] - ambition_target)
        prestige_source = getattr(world, "_latent_public_reputation", None)
        if prestige_source is not None:
            prestige_align = 1.0 - np.abs(prestige_source[valid_idx] - prestige_target)
        else:
            prestige_align = np.array([
                1.0 - abs(_safe01(get_person_latent(world, int(pid)).get("public_reputation"), 0.5) - prestige_target)
                for pid in dir_pids[valid_latent]
            ], dtype=float)
        alignment[valid_latent] = 0.40 * risk_align + 0.35 * ambition_align + 0.25 * prestige_align
    if (~valid_latent).any():
        for arr_idx, pid in zip(np.flatnonzero(~valid_latent), dir_pids[~valid_latent]):
            lv = get_person_latent(world, int(pid))
            risk_align = 1.0 - abs(_safe01(lv.get("risk_tolerance"), 0.5) - risk_target)
            ambition_align = 1.0 - abs(_safe01(lv.get("artistic_ambition"), 0.5) - ambition_target)
            prestige_align = 1.0 - abs(_safe01(lv.get("public_reputation"), 0.5) - prestige_target)
            alignment[arr_idx] = 0.40 * risk_align + 0.35 * ambition_align + 0.25 * prestige_align
    weights *= np.clip(0.70 + 0.80 * alignment, 0.35, 1.75)

    csv_target = np.asarray(_concept_csv_target(concept), dtype=np.float32)
    csv_target_norm = float(np.linalg.norm(csv_target))
    if csv_target_norm > 1e-10:
        csv_target = csv_target / csv_target_norm
    csv_alignment = np.ones(len(dirs), dtype=float)
    if valid_latent.any() and getattr(world, "_latent_csv_normed", None) is not None:
        csv_alignment[valid_latent] = np.clip(
            world._latent_csv_normed[latent_indices[valid_latent]] @ csv_target,
            0.0,
            1.0,
        )
    if (~valid_latent).any():
        for arr_idx, pid in zip(np.flatnonzero(~valid_latent), dir_pids[~valid_latent]):
            csv_alignment[arr_idx] = _cosine_sim(
                get_person_latent(world, int(pid)).get("creative_style_vector", [0.0] * 8),
                csv_target,
            )
    weights *= np.clip(0.80 + 0.50 * csv_alignment, 0.60, 1.40)

    pc_mult = _director_company_multiplier(world, dir_pids, year, tier, genre)
    if len(pc_mult):
        # Scale from 1.8 (brand-fit match) to 2.0 for directors
        weights *= np.where(pc_mult > 1.0, pc_mult * (2.0 / 1.8), 1.0)

    probs = normalize_weights(weights)
    if float(np.sum(probs)) <= 0:
        return None
    chosen = int(world.rng.choice(len(dirs), p=probs))
    did = int(dirs.iloc[chosen]["person_id"])
    world.director_film_count[did] += 1
    if hasattr(world, "director_recent"):
        world.director_recent[did].append(int(year))
    return did


def pick_co_director(world: WorldState, concept: dict, primary_dir_id: int) -> int | None:
    tier = str(concept.get("tier", "Mid"))
    genre = str(concept.get("genre", "Drama"))
    prob = {"Epic": 0.10, "A": 0.07, "Mid": 0.04, "Indie": 0.01, "Micro": 0.0}.get(tier, 0.02)
    if world.rng.random() >= prob or len(world.directors) < 2:
        return None
    dirs = _active_year_subset(world.directors, int(concept.get("year", 2000)))
    weights = dirs["pop_weight"].astype(float).values.copy()
    if "genre_affinity" in dirs.columns:
        ga = dirs["genre_affinity"].fillna("").astype(str).str.lower()
        weights *= np.where(ga.str.contains(genre.lower(), regex=False).values, 3.0, 1.0)
    weights[dirs["person_id"].astype(int).values == int(primary_dir_id)] = 0.0
    probs = normalize_weights(weights)
    if float(np.sum(probs)) <= 0:
        return None
    chosen = int(world.rng.choice(len(dirs), p=probs))
    did = int(dirs.iloc[chosen]["person_id"])
    world.director_film_count[did] += 1
    return did


def pick_companies(world: WorldState, concept: dict, director_id: int) -> list[dict]:
    comps = world.companies
    tier = str(concept["tier"])
    genre = str(concept["genre"])
    year = int(concept["year"])
    franchise = concept.get("franchise")

    if franchise and franchise.get("company_ids"):
        rows = []
        for cid in franchise["company_ids"]:
            if int(cid) in set(comps["company_id"].astype(int).tolist()):
                rows.append({"company_id": int(cid), "role": "Production" if not rows else "Co-Production"})
        if rows:
            return rows

    active = comps
    if "founded_year" in active.columns:
        active = active[active["founded_year"].fillna(0).astype(int) <= year]
    if "defunct_year" in active.columns:
        active = active[active["defunct_year"].isna() | (active["defunct_year"].fillna(2100).astype(float) >= float(year))]
    if len(active) < 3:
        active = comps

    n_companies = int(world.rng.choice([1, 1, 2, 2, 3]))
    n_companies = min(n_companies, len(active))
    weights = active["pop_weight"].astype(float).values.copy()

    if "tier" in active.columns:
        tier_match = active["tier"].astype(str).values == tier
        weights *= np.where(tier_match, 8.0, 0.2)
    if "specialty_genres" in active.columns:
        sg = active["specialty_genres"].fillna("").astype(str).str.lower()
        match = sg.str.contains(genre.lower(), regex=False).values
        weights *= np.where(match, 15.0, 0.25)

    shortlist = _shortlist_indices(
        weights,
        _shortlist_budget(world, "company", 18),
        world.rng,
        exploration_share=_prior_float(world, "company_primary_exploration_share", 0.35, lo=0.0, hi=0.60),
    )
    if shortlist.size > 0 and shortlist.size < len(active):
        active = active.iloc[shortlist]
        weights = weights[shortlist]

    cids = active["company_id"].astype(int).values
    tier_idx = TIER_TO_LATENT_IDX.get(tier, 2)
    risk_target, _, prestige_target = _concept_latent_targets(concept)
    align = np.ones(len(active), dtype=float)
    concept_genre_basis = project_genres_to_company_basis([genre])
    company_latent_lookup = getattr(world, "_pc_c_idx", {}) or {}
    company_latent_indices = np.array([company_latent_lookup.get(int(cid), -1) for cid in cids], dtype=int)
    valid_company_latent = company_latent_indices >= 0
    if valid_company_latent.any() and getattr(world, "_pc_ready", False):
        valid_idx = company_latent_indices[valid_company_latent]
        risk_align = 1.0 - np.abs(world._pc_c_risk[valid_idx] - risk_target)
        prestige_align = 1.0 - np.abs(world._pc_c_prestige[valid_idx] - prestige_target)
        focus = world._pc_c_budget[valid_idx, tier_idx]
        genre_fit = np.clip(world._pc_c_genre[valid_idx] @ concept_genre_basis, 0.0, 1.0)
        align[valid_company_latent] = 0.28 * risk_align + 0.22 * prestige_align + 0.30 * focus + 0.20 * genre_fit
    if (~valid_company_latent).any():
        for arr_idx, cid in zip(np.flatnonzero(~valid_company_latent), cids[~valid_company_latent]):
            lv = world.company_latent.get(int(cid), {}) if hasattr(world, "company_latent") else {}
            risk_align = 1.0 - abs(_safe01(lv.get("risk_appetite"), 0.5) - risk_target)
            prestige_align = 1.0 - abs(_safe01(lv.get("prestige_score"), 0.5) - prestige_target)
            bbp = lv.get("budget_tier_focus") if isinstance(lv, dict) else None
            focus = _safe01(bbp[tier_idx], 0.5) if isinstance(bbp, list) and len(bbp) > tier_idx else 0.5
            gp = canonical_company_genre_vector(lv.get("genre_portfolio") if isinstance(lv, dict) else None)
            genre_fit = float(np.clip(gp @ concept_genre_basis, 0.0, 1.0))
            align[arr_idx] = 0.28 * risk_align + 0.22 * prestige_align + 0.30 * focus + 0.20 * genre_fit
    weights *= np.clip(0.65 + 0.90 * align, 0.25, 1.80)
    weights = normalize_weights(weights)
    if len(active) == 0 or float(np.sum(weights)) <= 0:
        return []

    primary_shortlist = _shortlist_indices(
        weights,
        min(len(active), _shortlist_budget(world, "company", 18)),
        world.rng,
        exploration_share=_prior_float(world, "company_primary_exploration_share", 0.35, lo=0.0, hi=0.60),
    )
    if primary_shortlist.size == 0:
        return []
    primary_probs = normalize_weights(weights[primary_shortlist])
    chosen_idx = int(world.rng.choice(primary_shortlist, p=primary_probs))
    chosen_indices = [chosen_idx]
    first_cid = int(active.iloc[chosen_idx]["company_id"])

    if n_companies > 1:
        family = getattr(world, "_merge_families", {}) or getattr(world, "company_family", {})
        first_family = family.get(first_cid, set())
        partner_weights = weights.copy()
        for i, cid in enumerate(cids):
            # On-demand C-C scoring: replaces precomputed company_affinity/rivalry index
            cc_aff = world.compute_cc_affinity(first_cid, int(cid))
            if cc_aff > 0:
                partner_weights[i] *= (1.0 + 3.0 * cc_aff)
            cc_riv = world.compute_cc_rivalry(first_cid, int(cid))
            if cc_riv > 0:
                partner_weights[i] *= max(0.05, 1.0 - cc_riv)
            if int(cid) in first_family:
                partner_weights[i] *= 6.0
        for _ in range(n_companies - 1):
            pw = partner_weights.copy()
            for ci in chosen_indices:
                pw[ci] = 0.0
            if np.count_nonzero(pw > 0) == 0:
                break
            sl = _shortlist_indices(
                pw,
                min(len(active), _shortlist_budget(world, "company", 18)),
                world.rng,
                exploration_share=_prior_float(world, "company_secondary_exploration_share", 0.40, lo=0.0, hi=0.60),
            )
            if sl.size == 0:
                break
            probs = normalize_weights(pw[sl])
            idx = int(world.rng.choice(sl, p=probs))
            if idx not in chosen_indices:
                chosen_indices.append(idx)

    rows = []
    for pos, idx in enumerate(chosen_indices):
        cid = int(active.iloc[idx]["company_id"])
        world.company_film_count[cid] += 1
        if hasattr(world, "company_recent"):
            world.company_recent[cid].append(int(year))
        rows.append({"company_id": cid, "role": "Production" if pos == 0 else "Co-Production"})
    return rows


def _pick_character_name(genre: str, archetype: str, slot: int, world: WorldState, gender: str = "M", used_names: set | None = None) -> str:
    if used_names is None:
        used_names = set()
    if not hasattr(world, "_used_char_names_global"):
        world._used_char_names_global = set()
    global_used = world._used_char_names_global
    if hasattr(world, "character_bank") and world.character_bank is not None and len(world.character_bank) > 0:
        bank = world.character_bank
        key = genre.lower().strip()
        hints = _GENRE_TO_ARCHETYPES.get(key, [])
        if hints and "archetype" in bank.columns:
            pool = bank[bank["archetype"].isin(hints + [archetype])]
            if len(pool) < 20:
                pool = bank
        elif "archetype" in bank.columns:
            pool = bank[bank["archetype"] == archetype]
            if len(pool) == 0:
                pool = bank
        else:
            pool = bank
        if "gender" in pool.columns and gender in ("M", "F"):
            gendered = pool[pool["gender"] == gender]
            if len(gendered) >= 10:
                pool = gendered
        candidates = pool["character_name"].astype(str).values
        order = list(range(len(candidates)))
        world.rng.shuffle(order)
        for idx in order:
            name = str(candidates[idx])
            if name not in global_used and name not in used_names:
                used_names.add(name)
                global_used.add(name)
                return name
        for idx in order:
            name = str(candidates[idx])
            if name not in used_names:
                used_names.add(name)
                global_used.add(name)
                return name
    genre_key = genre.lower().strip()
    genre_key = genre_key if genre_key in _GENRE_CHAR_NAMES else _GENRE_KEY_MAP.get(genre_key)
    if genre_key and genre_key in _GENRE_CHAR_NAMES:
        bank = _GENRE_CHAR_NAMES[genre_key]
        names = bank.get(gender, bank.get("N", [])) or next(iter(bank.values()))
        order = list(range(len(names)))
        world.rng.shuffle(order)
        for idx in order:
            if names[idx] not in used_names:
                used_names.add(names[idx])
                return names[idx]
    fallback = f"Character_{slot}_{world.rng.randint(1000, 9999)}"
    used_names.add(fallback)
    return fallback


def _pick_cast_fast(world: WorldState, concept: dict, director_id: int, max_retries: int = 3) -> tuple[list[dict], list[tuple[int, int]]]:
    year = int(concept["year"])
    cast_size = _sample_cast_size(world, concept)
    static = _build_actor_static_block(world, concept)
    selection = _get_selection_year_state(world, year)
    pids = static.pids
    n_cand = len(pids)
    director_pref_boost, avoid_mask = _director_edge_arrays(world, int(director_id) if director_id is not None else -1, pids, year)
    cast_shortlist_size = _shortlist_budget(world, "cast", 120)
    candidate_local_idx = static.candidate_idx

    if not hasattr(world, "_pid_to_gender"):
        world._pid_to_gender = {}
        if world.persons is not None and "gender" in world.persons.columns:
            _g_pids = world.persons["person_id"].astype(int).values
            _g_vals = world.persons["gender"].fillna("M").astype(str).values
            for _gi in range(len(_g_pids)):
                world._pid_to_gender[int(_g_pids[_gi])] = _g_vals[_gi]

    franchise = concept.get("franchise")
    franchise_pool = set(int(p) for p in franchise.get("cast_pool", [])) if franchise and franchise.get("movies_generated", 0) > 0 else set()
    franchise_pool_mask = np.isin(pids, np.array(sorted(franchise_pool), dtype=int)) if franchise_pool else np.zeros(n_cand, dtype=bool)
    award_mask = selection.award_recent[candidate_local_idx] if len(candidate_local_idx) else np.zeros(0, dtype=bool)
    priority_focus = _priority_cast_indices(
        static,
        director_pref_boost,
        award_mask,
        franchise_pool_mask,
        max(cast_shortlist_size, 64),
    )
    base_focus = static.base_focus_idx
    base_focus_cap = max(cast_shortlist_size * 4, 224)

    cast: list[dict] = []
    competition_pairs: list[tuple[int, int]] = []
    for _attempt in range(max_retries):
        cast.clear()
        competition_pairs.clear()
        attempt_film_count = selection.film_count.copy()
        attempt_recent_window = selection.recent_window.copy()
        attempt_yearly_workload = selection.yearly_workload.copy()
        attempt_unused_flags = selection.unused_flags.copy()
        ensemble = CastEnsembleState(
            friend_frontier_vec=np.zeros(n_cand, dtype=np.float32),
            rival_penalty_vec=np.ones(n_cand, dtype=np.float32),
            blocked_local_mask=np.zeros(n_cand, dtype=bool),
        )
        slot_exploration_empty = _prior_float(world, "cast_slot_exploration_empty", 0.35, lo=0.0, hi=0.60)
        slot_exploration_filled = _prior_float(world, "cast_slot_exploration_filled", 0.30, lo=0.0, hi=0.60)
        community_match_mult = _prior_float(world, "cast_community_match_multiplier", 1.5, lo=1.0, hi=3.0)

        for slot in range(cast_size):
            frontier_candidates = ensemble.frontier_local_idx | ensemble.rival_local_idx
            if frontier_candidates:
                frontier_arr = np.fromiter(frontier_candidates, dtype=np.int32)
                candidate_focus = np.unique(np.concatenate([base_focus, priority_focus, frontier_arr]))
            elif priority_focus.size:
                candidate_focus = np.unique(np.concatenate([base_focus, priority_focus]))
            else:
                candidate_focus = base_focus
            candidate_focus = _cap_focus_indices(static.immutable_scores, candidate_focus.astype(np.int32, copy=False), base_focus_cap)
            if candidate_focus.size == 0:
                break

            focus_local = candidate_local_idx[candidate_focus]
            film_count = attempt_film_count[focus_local]
            recent_window = attempt_recent_window[focus_local]
            yearly_workload = attempt_yearly_workload[focus_local]
            unused_flags = attempt_unused_flags[focus_local]

            focus_scores = static.immutable_scores[candidate_focus].copy()
            focus_scores *= np.clip(1.0 + 0.22 * np.log1p(film_count), 1.0, 3.4)
            cooldown_mult = np.where(
                recent_window > 0,
                np.maximum(static.cooldown_floor[candidate_focus], 1.0 - static.cooldown_decay[candidate_focus] * recent_window),
                1.0,
            )
            focus_scores *= cooldown_mult
            focus_scores *= np.where(yearly_workload >= static.yearly_max[candidate_focus], 0.1, 1.0)
            focus_scores *= director_pref_boost[candidate_focus]
            focus_scores[avoid_mask[candidate_focus]] = 0.0

            if ensemble.blocked_local_mask is not None:
                focus_scores[ensemble.blocked_local_mask[candidate_focus]] = 0.0

            if slot >= 2:
                focus_scores[unused_flags] *= 1.5
            if slot >= 4:
                focus_scores[static.top_star_mask[candidate_focus]] *= _env_float("V16_TOP_STAR_SLOT_PENALTY", 0.82, 0.5, 1.0)
            if award_mask.any():
                focus_scores[award_mask[candidate_focus] & (focus_scores > 0)] *= 1.5
            if franchise_pool:
                focus_scores[franchise_pool_mask[candidate_focus] & (focus_scores > 0)] *= 2.0

            if ensemble.cast_ids:
                focus_pids = pids[candidate_focus]
                if ensemble.agencies:
                    agency_match = np.isin(static.cand_agencies[candidate_focus], list(ensemble.agencies))
                    focus_scores[(focus_scores > 0) & agency_match] *= 2.0
                if ensemble.communities:
                    community_match = np.isin(static.cand_communities[candidate_focus], list(ensemble.communities))
                    focus_scores[(focus_scores > 0) & community_match] *= community_match_mult
                if ensemble.genders and static.actor_genders is not None:
                    gender_novel = np.array([g not in ensemble.genders for g in static.actor_genders[candidate_focus]], dtype=bool)
                    focus_scores[(focus_scores > 0) & gender_novel] *= 1.5
                if len(ensemble.nationalities) >= 2 and static.actor_nationalities is not None:
                    nat_novel = np.array([n not in ensemble.nationalities for n in static.actor_nationalities[candidate_focus]], dtype=bool)
                    focus_scores[(focus_scores > 0) & nat_novel] *= 1.3
                if ensemble.tag_bitmasks and static.actor_tag_bitmasks is not None:
                    cand_bm = static.actor_tag_bitmasks[candidate_focus]
                    forbidden_mask = np.array([int(pid) in ensemble.forbidden for pid in focus_pids], dtype=bool)
                    has_tags = cand_bm > np.uint64(0)
                    eligible = (focus_scores > 0) & has_tags & ~forbidden_mask
                    if eligible.any():
                        max_jaccard = np.zeros(len(candidate_focus), dtype=np.float32)
                        for cbm in ensemble.tag_bitmasks:
                            inter = _popcount_vec(cand_bm & cbm)
                            union = _popcount_vec(cand_bm | cbm)
                            jaccard = np.where(union > 0, inter.astype(np.float32) / union.astype(np.float32), 0.0)
                            np.maximum(max_jaccard, jaccard, out=max_jaccard)
                        focus_scores[eligible & (max_jaccard > 0.5)] *= 0.5
                if ensemble.friend_frontier_vec is not None:
                    friend_boost = ensemble.friend_frontier_vec[candidate_focus]
                    fb_mask = friend_boost > 0
                    focus_scores[fb_mask] *= friend_boost[fb_mask]
                else:
                    fb_mask = np.zeros(len(candidate_focus), dtype=bool)
                if ensemble.rival_penalty_vec is not None:
                    focus_scores *= ensemble.rival_penalty_vec[candidate_focus]
                need_similarity = np.flatnonzero((focus_scores >= static._sim_threshold) & ~fb_mask)
                if need_similarity.size > 0 and static.li_arr is not None and ensemble.latent_anchor_ids:
                    ns_li = static.li_arr[candidate_focus[need_similarity]]
                    valid_li = ns_li >= 0
                    if valid_li.any():
                        local_targets = need_similarity[valid_li]
                        max_sim = latent_similarity_batch(world, ns_li[valid_li], np.asarray(ensemble.latent_anchor_ids[:3], dtype=int))
                        boost = max_sim > 0.1
                        if boost.any():
                            focus_scores[local_targets[boost]] *= (1.0 + max_sim[boost])

            valid_local = _shortlist_indices(
                focus_scores,
                cast_shortlist_size,
                world.rng,
                exploration_share=slot_exploration_filled if ensemble.cast_ids else slot_exploration_empty,
            )
            if valid_local.size == 0:
                break
            valid = candidate_focus[valid_local]
            exp = 2.5 if slot == 0 else (1.3 if slot == 1 else 0.7)
            probs = normalize_weights(focus_scores[valid_local] ** exp)
            chosen_local = int(world.rng.choice(valid_local, p=probs))
            chosen_idx = int(candidate_focus[chosen_local])
            pid = int(pids[chosen_idx])

            if slot <= 1 and valid.size >= 2 and len(competition_pairs) < 4:
                order = valid[np.argsort(focus_scores[valid_local])[::-1]]
                for alt_idx in order:
                    if int(alt_idx) != chosen_idx:
                        competition_pairs.append((min(pid, int(pids[alt_idx])), max(pid, int(pids[alt_idx]))))
                        break

            cast.append({"person_id": pid, "billing_order": slot + 1})
            chosen_focus_local = int(candidate_local_idx[chosen_idx])
            attempt_film_count[chosen_focus_local] += 1.0
            attempt_recent_window[chosen_focus_local] += 1.0
            attempt_yearly_workload[chosen_focus_local] += 1.0
            attempt_unused_flags[chosen_focus_local] = False
            ensemble.add_actor(world, static, chosen_idx, year)

        if len(cast) >= max(1, cast_size - 1):
            break

    committed_ids = [int(row["person_id"]) for row in cast]
    if committed_ids:
        if not hasattr(world, "person_film_count") or world.person_film_count is None:
            world.person_film_count = Counter()
        if not hasattr(world, "person_recent") or world.person_recent is None:
            world.person_recent = {}
        for pid in committed_ids:
            world.person_film_count[pid] = int(world.person_film_count.get(pid, 0)) + 1
            world.person_recent.setdefault(pid, []).append(year)
            if getattr(world, "_yearly_workload", None) is not None:
                world._yearly_workload[(pid, year)] += 1
        selection.record_cast_selection(committed_ids, year)

    _assign_archetypes(cast, concept, world)
    return cast, competition_pairs


def pick_cast(world: WorldState, concept: dict, director_id: int, max_retries: int = 3) -> tuple[list[dict], list[tuple[int, int]]]:
    return _pick_cast_fast(world, concept, director_id, max_retries=max_retries)

    tier = str(concept["tier"])
    year = int(concept["year"])
    cast_size = _sample_cast_size(world, concept)
    static = _build_actor_static_block(world, concept)
    pids = static.pids
    n_cand = len(pids)
    graph = getattr(world, "graph", None)
    director_pref_boost, avoid_mask = _director_edge_arrays(world, int(director_id) if director_id is not None else -1, pids, year)
    cast_shortlist_size = _shortlist_budget(world, "cast", 120)

    cast: list[dict] = []
    competition_pairs: list[tuple[int, int]] = []
    cast_ids: set[int] = set()
    cast_id_list: list[int] = []
    forbidden: set[int] = set()

    # Pre-build pid→gender dict (cached on world, O(N_persons) once)
    if not hasattr(world, '_pid_to_gender'):
        world._pid_to_gender = {}
        if world.persons is not None and "gender" in world.persons.columns:
            _g_pids = world.persons["person_id"].astype(int).values
            _g_vals = world.persons["gender"].fillna("M").astype(str).values
            for _gi in range(len(_g_pids)):
                world._pid_to_gender[int(_g_pids[_gi])] = _g_vals[_gi]

    # Pre-build award-recent set for O(1) lookup per candidate
    award_recent_pids = set()
    if getattr(world, "person_award_wins", None):
        for _aw_pid, _aw_info in world.person_award_wins.items():
            if isinstance(_aw_info, dict):
                if year - int(_aw_info.get("year", 0) or 0) <= 3:
                    award_recent_pids.add(int(_aw_pid))
            elif isinstance(_aw_info, int) and _aw_info > 0:
                award_recent_pids.add(int(_aw_pid))

    # Pre-build franchise cast pool set
    franchise = concept.get("franchise")
    franchise_pool = set()
    if franchise and franchise.get("movies_generated", 0) > 0:
        franchise_pool = set(int(p) for p in franchise.get("cast_pool", []))
    award_mask = np.isin(pids, np.array(sorted(award_recent_pids), dtype=int)) if award_recent_pids else np.zeros(n_cand, dtype=bool)
    franchise_pool_mask = np.isin(pids, np.array(sorted(franchise_pool), dtype=int)) if franchise_pool else np.zeros(n_cand, dtype=bool)
    for _attempt in range(max_retries):
        cast.clear()
        competition_pairs.clear()
        cast_ids.clear()
        cast_id_list.clear()
        forbidden.clear()
        unused_flags = static.unused_flags.copy()

        # Pre-compute the static portion of scores ONCE.  All 15 multipliers
        # below are identical every slot — only per-slot adjustments differ.
        static_product = static.base_scores.copy()
        static_product *= static.career_mult
        static_product *= static.career_stage_mult
        static_product *= static.genre_mult
        static_product *= static.nationality_mult
        static_product *= static.style_mult
        static_product *= static.avoid_genre_mult
        static_product *= static.market_mult
        static_product *= static.collab_mult
        static_product *= static.controversy_mult
        static_product *= static.volatility_mult
        static_product *= static.csv_mult
        static_product *= static.company_aff_mult
        static_product *= static.cooldown_mult
        static_product *= static.workload_mult
        static_product *= director_pref_boost
        static_product[avoid_mask] = 0.0

        for slot in range(cast_size):
            scores = static_product.copy()

            if cast_ids or forbidden:
                for cid in cast_ids | forbidden:
                    idx = static.pid_to_idx.get(int(cid))
                    if idx is not None:
                        scores[idx] = 0.0

            if slot >= 2:
                scores[unused_flags] *= 1.5
            if slot >= 4:
                scores[static.top_star_mask] *= _env_float("V16_TOP_STAR_SLOT_PENALTY", 0.82, 0.5, 1.0)

            # Awards bump — vectorized via pre-built set
            if award_recent_pids:
                scores[award_mask & (scores > 0)] *= 1.5

            # Franchise cast pool boost — vectorized via pre-built set
            if franchise_pool:
                scores[franchise_pool_mask & (scores > 0)] *= 2.0

            candidate_focus = _expand_cast_focus_indices(
                world,
                static,
                scores,
                cast_id_list,
                year,
                cast_shortlist_size,
            )
            if candidate_focus.size == 0:
                break

            focus_scores = scores[candidate_focus].copy()

            if cast_id_list:
                focus_pids = pids[candidate_focus]

                cast_agency_set = {world.person_agency.get(cid) for cid in cast_id_list}
                cast_agency_set.discard(None)
                if cast_agency_set:
                    agency_match = np.isin(static.cand_agencies[candidate_focus], list(cast_agency_set))
                    focus_scores[(focus_scores > 0) & agency_match] *= 2.0

                cast_com_set = set()
                for cid in cast_id_list:
                    c = static.cand_communities[static.pid_to_idx[cid]] if cid in static.pid_to_idx else -1
                    if c >= 0:
                        cast_com_set.add(c)
                if cast_com_set:
                    com_match = np.isin(static.cand_communities[candidate_focus], list(cast_com_set))
                    focus_scores[(focus_scores > 0) & com_match] *= 1.5

                cast_friend_pids = set()
                for cid in cast_id_list:
                    if graph is not None:
                        for nbr, _w, _vf, _vt in graph.iter_friend_neighbors(int(cid), year):
                            cast_friend_pids.add(int(nbr))
                    else:
                        for nbr, _w, _vf, _vt in world._friend_adj_all.get(int(cid), []):
                            cast_friend_pids.add(int(nbr))

                cast_genders = Counter(
                    static.actor_genders[static.pid_to_idx[cid]]
                    for cid in cast_id_list
                    if static.actor_genders is not None and cid in static.pid_to_idx
                )
                cast_nats = Counter(
                    static.actor_nationalities[static.pid_to_idx[cid]]
                    for cid in cast_id_list
                    if static.actor_nationalities is not None and cid in static.pid_to_idx
                )
                if cast_genders and static.actor_genders is not None:
                    gender_present = set(cast_genders.keys())
                    gender_novel = np.array(
                        [g not in gender_present for g in static.actor_genders[candidate_focus]],
                        dtype=bool,
                    )
                    focus_scores[(focus_scores > 0) & gender_novel] *= 1.5
                if len(cast_nats) >= 2 and static.actor_nationalities is not None:
                    nat_present = set(cast_nats.keys())
                    nat_novel = np.array(
                        [n not in nat_present for n in static.actor_nationalities[candidate_focus]],
                        dtype=bool,
                    )
                    focus_scores[(focus_scores > 0) & nat_novel] *= 1.3

                if static.actor_tag_bitmasks is not None:
                    cast_bitmasks = np.array(
                        [static.actor_tag_bitmasks[static.pid_to_idx[cid]] for cid in cast_id_list if cid in static.pid_to_idx],
                        dtype=np.uint64,
                    )
                    if cast_bitmasks.size > 0:
                        cand_bm = static.actor_tag_bitmasks[candidate_focus]
                        friend_mask = (
                            np.isin(focus_pids, np.array(list(cast_friend_pids), dtype=int))
                            if cast_friend_pids else np.zeros(len(candidate_focus), dtype=bool)
                        )
                        has_tags = cand_bm > np.uint64(0)
                        eligible = (focus_scores > 0) & has_tags & ~friend_mask
                        if eligible.any():
                            max_jaccard = np.zeros(len(candidate_focus), dtype=np.float32)
                            for cbm in cast_bitmasks:
                                inter = _popcount_vec(cand_bm & cbm)
                                union = _popcount_vec(cand_bm | cbm)
                                jaccard = np.where(union > 0, inter.astype(np.float32) / union.astype(np.float32), 0.0)
                                np.maximum(max_jaccard, jaccard, out=max_jaccard)
                            focus_scores[eligible & (max_jaccard > 0.5)] *= 0.5

                friend_boost = np.zeros(len(candidate_focus), dtype=np.float32)
                rival_penalty = np.ones(len(candidate_focus), dtype=np.float32)
                focus_pos = {int(global_idx): pos for pos, global_idx in enumerate(candidate_focus)}
                for cid in cast_id_list:
                    friend_iter = graph.iter_friend_neighbors(int(cid), year) if graph is not None else world._friend_adj_all.get(int(cid), [])
                    rival_iter = graph.iter_rival_neighbors(int(cid), year) if graph is not None else world._rival_adj_all.get(int(cid), [])
                    for nbr, w, _vf, _vt in friend_iter:
                        idx = static.pid_to_idx.get(int(nbr))
                        local_idx = focus_pos.get(int(idx)) if idx is not None else None
                        if local_idx is not None and focus_scores[local_idx] > 0 and friend_boost[local_idx] == 0:
                            friend_boost[local_idx] = 1.0 + 4.0 * max(0.15, w)
                    for nbr, w, _vf, _vt in rival_iter:
                        idx = static.pid_to_idx.get(int(nbr))
                        local_idx = focus_pos.get(int(idx)) if idx is not None else None
                        if local_idx is not None and focus_scores[local_idx] > 0:
                            rival_penalty[local_idx] = min(
                                rival_penalty[local_idx],
                                max(0.0, 1.0 - max(0.6, w)),
                            )
                fb_mask = friend_boost > 0
                focus_scores[fb_mask] *= friend_boost[fb_mask]
                focus_scores *= rival_penalty

                need_similarity = np.flatnonzero((focus_scores >= static._sim_threshold) & ~fb_mask)
                if need_similarity.size > 0 and static.li_arr is not None:
                    latent_idx = getattr(world, "_latent_pid_to_idx", {}) or {}
                    cast_li = np.array([latent_idx.get(cid, -1) for cid in cast_id_list[:3]], dtype=int)
                    cast_li = cast_li[cast_li >= 0]
                    if cast_li.size > 0:
                        ns_li = static.li_arr[candidate_focus[need_similarity]]
                        valid_li = ns_li >= 0
                        if valid_li.any():
                            local_targets = need_similarity[valid_li]
                            max_sim = latent_similarity_batch(world, ns_li[valid_li], cast_li)
                            boost = max_sim > 0.1
                            if boost.any():
                                focus_scores[local_targets[boost]] *= (1.0 + max_sim[boost])

            valid_local = _shortlist_indices(
                focus_scores,
                cast_shortlist_size,
                world.rng,
                exploration_share=0.30 if cast_id_list else 0.35,
            )
            if valid_local.size == 0:
                break
            valid = candidate_focus[valid_local]
            exp = 2.5 if slot == 0 else (1.3 if slot == 1 else 0.7)
            probs = normalize_weights(focus_scores[valid_local] ** exp)
            chosen_local = int(world.rng.choice(valid_local, p=probs))
            chosen_idx = int(candidate_focus[chosen_local])
            pid = int(pids[chosen_idx])

            if slot <= 1 and valid.size >= 2 and len(competition_pairs) < 4:
                order = valid[np.argsort(focus_scores[valid_local])[::-1]]
                for alt_idx in order:
                    if int(alt_idx) != chosen_idx:
                        competition_pairs.append((min(pid, int(pids[alt_idx])), max(pid, int(pids[alt_idx]))))
                        break

            cast.append({"person_id": pid, "billing_order": slot + 1})
            cast_ids.add(pid)
            cast_id_list.append(pid)
            world.person_film_count[pid] += 1
            world.person_recent[pid].append(year)
            if getattr(world, "_yearly_workload", None) is not None:
                world._yearly_workload[(pid, year)] += 1
            idx = static.pid_to_idx.get(pid)
            if idx is not None:
                unused_flags[idx] = False

            # Keep direct rivals from being added later in the same cast
            rival_source = graph.iter_rival_neighbors(int(pid), year) if graph is not None else world._rival_adj_all.get(int(pid), [])
            rival_neighbors = set(nbr for (nbr, _w, _vf, _vt) in rival_source)
            forbidden.update(int(nbr) for nbr in rival_neighbors if int(nbr) not in cast_ids)

        if len(cast) >= max(1, cast_size - 1):
            break

    _assign_archetypes(cast, concept, world)

    return cast, competition_pairs


# ---------------------------------------------------------------------------
# Latent-profile archetype assignment
# ---------------------------------------------------------------------------

# Target latent profiles: [reputation, risk_tolerance, ambition, controversy]
# Each archetype has an "ideal" actor profile — the scoring function measures
# how close a real actor is to that ideal.
_ARCHETYPE_TARGET = {
    "Lead Hero":      np.array([0.80, 0.55, 0.70, 0.12], dtype=np.float32),
    "Lead Villain":   np.array([0.60, 0.75, 0.65, 0.45], dtype=np.float32),
    "Love Interest":  np.array([0.60, 0.30, 0.50, 0.08], dtype=np.float32),
    "Mentor":         np.array([0.75, 0.30, 0.65, 0.08], dtype=np.float32),
    "Sidekick":       np.array([0.40, 0.45, 0.40, 0.15], dtype=np.float32),
    "Comic Relief":   np.array([0.35, 0.50, 0.35, 0.22], dtype=np.float32),
    "Supporting":     np.array([0.40, 0.45, 0.45, 0.15], dtype=np.float32),
    "Henchman":       np.array([0.25, 0.60, 0.30, 0.30], dtype=np.float32),
    "Extra":          np.array([0.20, 0.40, 0.25, 0.10], dtype=np.float32),
}

# Billing tier → eligible archetypes.  Preserves the slot → importance hierarchy.
_ARCHETYPE_SLOT_TIERS: dict[int, list[str]] = {
    0: ["Lead Hero", "Lead Villain"],
    1: ["Love Interest", "Sidekick", "Mentor", "Lead Villain"],
    2: ["Mentor", "Comic Relief", "Sidekick", "Lead Villain"],
}
_ARCHETYPE_GENERAL = ["Supporting", "Extra", "Henchman"]

# Archetypes that can appear at most ONCE per movie.
_ARCHETYPE_UNIQUE = {"Lead Hero", "Lead Villain", "Mentor"}

# Career stage bonuses: legends gravitate to Mentor, prime actors to leads, etc.
_CAREER_ARCHETYPE_BONUS: dict[str, dict[str, float]] = {
    "legend":  {"Mentor": 0.30, "Lead Hero": 0.12, "Lead Villain": 0.08},
    "prime":   {"Lead Hero": 0.20, "Lead Villain": 0.18, "Love Interest": 0.10},
    "veteran": {"Mentor": 0.22, "Supporting": 0.08, "Lead Villain": 0.10},
    "rising":  {"Sidekick": 0.20, "Supporting": 0.12, "Love Interest": 0.10},
    "retired": {"Mentor": 0.18, "Extra": 0.08, "Supporting": 0.05},
}

# Genre-specific archetype bonuses.
_GENRE_ARCHETYPE_BONUS: dict[str, dict[str, float]] = {
    "Horror":       {"Lead Villain": 0.25, "Henchman": 0.08},
    "Romance":      {"Love Interest": 0.28, "Supporting": 0.05},
    "Comedy":       {"Comic Relief": 0.25, "Sidekick": 0.10},
    "Action":       {"Lead Hero": 0.18, "Henchman": 0.10, "Sidekick": 0.08},
    "Thriller":     {"Lead Villain": 0.18, "Lead Hero": 0.10},
    "Crime":        {"Lead Villain": 0.15, "Henchman": 0.10},
    "Sci-Fi":       {"Lead Hero": 0.12, "Mentor": 0.10},
    "Fantasy":      {"Lead Hero": 0.12, "Mentor": 0.12, "Lead Villain": 0.10},
    "Drama":        {"Lead Hero": 0.10, "Mentor": 0.10, "Supporting": 0.08},
    "War":          {"Lead Hero": 0.15, "Mentor": 0.12, "Henchman": 0.08},
    "Mystery":      {"Lead Hero": 0.10, "Lead Villain": 0.15, "Sidekick": 0.10},
    "Documentary":  {"Supporting": 0.10},
    "Animation":    {"Sidekick": 0.12, "Comic Relief": 0.15, "Lead Villain": 0.10},
}


def _archetype_score(
    lv: dict, archetype: str, career_stage: str, genre: str, jitter: float,
) -> float:
    """Score how well a person's latent profile matches an archetype.

    Returns a value roughly in [0, 1.5] where higher = better match.
    The *jitter* parameter (drawn from rng) adds controlled randomness so
    that the same actor profile doesn't deterministically get the same role.
    """
    target = _ARCHETYPE_TARGET[archetype]
    person = np.array([
        _safe01(lv.get("public_reputation"), 0.5),
        _safe01(lv.get("risk_tolerance"), 0.5),
        _safe01(lv.get("artistic_ambition"), 0.5),
        _safe01(lv.get("controversy_score"), 0.15),
    ], dtype=np.float32)

    # Mean absolute error → similarity (1.0 = perfect match)
    similarity = float(1.0 - np.mean(np.abs(person - target)))

    # Career stage bonus
    similarity += _CAREER_ARCHETYPE_BONUS.get(career_stage, {}).get(archetype, 0.0)

    # Genre bonus
    similarity += _GENRE_ARCHETYPE_BONUS.get(genre, {}).get(archetype, 0.0)

    # Collaboration style alignment
    collab = str(lv.get("collaboration_style", "ensemble")).lower()
    if archetype in ("Lead Hero", "Lead Villain") and collab == "solo":
        similarity += 0.08
    elif archetype in ("Sidekick", "Supporting") and collab == "ensemble":
        similarity += 0.06
    elif archetype == "Mentor" and collab == "mentorship":
        similarity += 0.12

    # Controlled jitter: enough to shuffle close scores, not enough to
    # override strong matches.  Scaled to ~10% of typical score range.
    similarity += jitter * 0.12

    return float(similarity)


def _ensure_career_stage_cache(world: WorldState) -> None:
    """Build pid → career_stage cache (O(N) once, O(1) thereafter)."""
    if getattr(world, "_pid_to_career_stage", None) is not None:
        return
    world._pid_to_career_stage = {}
    for df in (world.actors, world.persons):
        if df is not None and "career_stage" in df.columns and "person_id" in df.columns:
            pids = df["person_id"].astype(int).values
            stages = df["career_stage"].fillna("prime").astype(str).str.lower().values
            for pid, stage in zip(pids, stages):
                world._pid_to_career_stage[int(pid)] = stage
            break


def _assign_archetypes(cast: list[dict], concept: dict, world: WorldState) -> None:
    """Assign archetypes by matching each actor's latent profile to role ideals.

    Billing order is preserved (slot 0 = highest billed). For each slot the
    best-matching *eligible* archetype is chosen, respecting tier constraints
    and uniqueness rules. A small random jitter prevents deterministic mapping.
    """
    genre = str(concept.get("genre", "Drama"))
    n = len(cast)
    if n == 0:
        return

    _ensure_career_stage_cache(world)
    used_unique: set[str] = set()
    used_names: set[str] = set()

    # Pre-draw jitter values for each slot (one rng call, reproducible)
    jitters = world.rng.uniform(-1.0, 1.0, size=n)

    for i, row in enumerate(cast):
        pid = int(row["person_id"])
        lv = get_person_latent(world, pid)
        career_stage = world._pid_to_career_stage.get(pid, "prime")

        # Determine eligible archetypes for this billing position
        if i in _ARCHETYPE_SLOT_TIERS:
            eligible = list(_ARCHETYPE_SLOT_TIERS[i])
        else:
            eligible = list(_ARCHETYPE_GENERAL)

        # Remove already-used unique archetypes
        eligible = [a for a in eligible if a not in used_unique]

        # Fallback: if all eligible were taken, open up general pool
        if not eligible:
            eligible = [a for a in _ARCHETYPE_GENERAL if a not in used_unique]
        if not eligible:
            eligible = list(_ARCHETYPE_GENERAL)

        # Score each eligible archetype
        best_arch = eligible[0]
        best_score = -999.0
        for arch in eligible:
            score = _archetype_score(lv, arch, career_stage, genre, float(jitters[i]))
            if score > best_score:
                best_score = score
                best_arch = arch

        # Mark unique archetypes as used
        if best_arch in _ARCHETYPE_UNIQUE:
            used_unique.add(best_arch)

        # Assign archetype and character name
        gender = world._pid_to_gender.get(pid, "M")
        row["archetype"] = best_arch
        row["character_name"] = _pick_character_name(
            str(concept["genre"]), best_arch, i, world,
            gender=gender, used_names=used_names,
        )


def _generate_tagline(genre: str, world) -> str:
    templates = _TAGLINE_TEMPLATES.get(genre, _TAGLINE_TEMPLATES.get("Drama", []))
    return world.py_rng.choice(templates) if templates else ""


def pick_title(world: WorldState, concept: dict) -> tuple[str, str, bool]:
    genre = str(concept["genre"])
    franchise = concept.get("franchise")
    if franchise and franchise["movies_generated"] > 0:
        base = franchise["name"]
        inst = concept["installment"]
        title = f"{base} {inst}" if inst <= 3 else f"{base}: Part {inst}"
        if title not in world.used_titles:
            world.used_titles.add(title)
            return title, _generate_tagline(genre, world), False

    if world.title_bank is not None and len(world.title_bank) > 0:
        if "genre_hint" in world.title_bank.columns:
            pool = world.title_bank[(world.title_bank["genre_hint"] == genre) & (~world.title_bank["title"].isin(world.used_titles))]
        else:
            pool = world.title_bank[~world.title_bank["title"].isin(world.used_titles)]
        if len(pool) == 0:
            pool = world.title_bank[~world.title_bank["title"].isin(world.used_titles)]
        if len(pool) > 0:
            row = pool.sample(1, random_state=world.rng.randint(0, 2**31)).iloc[0]
            title = str(row["title"])
            tagline = str(row.get("tagline", ""))
            if not tagline or tagline == "nan":
                tagline = _generate_tagline(genre, world)
            ac_raw = row.get("award_contender", False)
            award_contender = bool(ac_raw) if not (isinstance(ac_raw, float) and ac_raw != ac_raw) else False
            world.used_titles.add(title)
            if franchise and franchise["movies_generated"] == 0:
                franchise["name"] = title
            return title, tagline, award_contender

    title = generate_compositional_title(world.py_rng, world.used_titles)
    world.used_titles.add(title)
    if franchise and franchise["movies_generated"] == 0:
        franchise["name"] = title
    return title, _generate_tagline(genre, world), False


def _ensure_company_genre_cache(world: WorldState) -> None:
    """Build a company_id→set[str] cache of each company's specialty genres."""
    if getattr(world, "_company_genre_cache", None) is not None:
        return
    world._company_genre_cache = {}
    if world.companies is None or len(world.companies) == 0:
        return
    if "specialty_genres" not in world.companies.columns:
        return
    cids = world.companies["company_id"].astype(int).values
    genres = world.companies["specialty_genres"].fillna("").astype(str).values
    for cid, gspec in zip(cids, genres):
        pieces = {g.strip() for g in str(gspec).replace(",", ";").split(";") if g.strip()}
        if pieces:
            world._company_genre_cache[int(cid)] = pieces


# Genre-family clusters: a keyword tagged "Action" should also get a small
# boost from a Thriller/Crime-focused company, since those genres share
# thematic affinity.
_GENRE_CLUSTERS = [
    {"Action", "Thriller", "Crime"},
    {"Drama", "Romance", "War"},
    {"Sci-Fi", "Fantasy", "Animation"},
    {"Horror", "Mystery"},
    {"Comedy"},
    {"Documentary"},
]

# Pre-build a genre→cluster mapping for O(1) lookup.
_GENRE_TO_CLUSTER_IDX: dict[str, int] = {}
for _ci, _cluster in enumerate(_GENRE_CLUSTERS):
    for _g in _cluster:
        _GENRE_TO_CLUSTER_IDX[_g] = _ci


def pick_keywords(world: WorldState, concept: dict, n: int = None, company_ids: list | None = None) -> list[int]:
    if n is None:
        n = int(world.rng.randint(2, 6))
    kw = world.keywords
    if kw is None or len(kw) == 0:
        return []
    weights = kw["pop_weight"].astype(float).values.copy() if "pop_weight" in kw.columns else np.ones(len(kw), dtype=float)
    if "topic_genre" in kw.columns:
        weights *= np.where((kw["topic_genre"] == concept["genre"]).values, 3.0, 1.0)

    if company_ids and "topic_genre" in kw.columns:
        _ensure_company_genre_cache(world)

        # Collect the actual specialty genres of the producing companies.
        # Fall back to the movie's own genre if the company has none.
        company_genres: set[str] = set()
        for cid in company_ids[:3]:
            cg = world._company_genre_cache.get(int(cid))
            if cg:
                company_genres.update(cg)
        if not company_genres:
            company_genres.add(str(concept["genre"]))

        # Expand each company genre to its family cluster, so a Thriller
        # company also boosts Action/Crime keywords.
        boost_genres: set[str] = set(company_genres)
        seen_clusters: set[int] = set()
        for g in company_genres:
            cidx = _GENRE_TO_CLUSTER_IDX.get(g)
            if cidx is not None and cidx not in seen_clusters:
                seen_clusters.add(cidx)
                boost_genres.update(_GENRE_CLUSTERS[cidx])

        # Build keyword-level boost vector from the expanded genre set.
        kw_topic = kw["topic_genre"].values
        cluster_weights = np.zeros(len(kw), dtype=float)
        for bg in boost_genres:
            # Direct match with company specialty → strong boost
            direct = bg in company_genres
            match = (kw_topic == bg).astype(float)
            cluster_weights += match * (1.0 if direct else 0.5)

        if float(cluster_weights.sum()) > 0:
            total = float(weights.sum())
            cluster_weights = cluster_weights / cluster_weights.sum() * total * 0.67
            weights = weights * 0.60 + cluster_weights

    probs = normalize_weights(weights)
    n = min(int(n), len(kw))
    chosen = world.rng.choice(len(kw), size=n, replace=False, p=probs)
    return [int(kw.iloc[int(i)]["keyword_id"]) for i in chosen]


def pick_crew(world: WorldState, concept: dict, director_id: int, cast: list[dict]) -> list[dict]:
    year = int(concept["year"])
    genre = str(concept.get("genre", "Drama"))
    tier = str(concept.get("tier", "Mid-Budget"))
    used = {int(director_id)} if director_id is not None else set()
    for c in cast:
        try:
            used.add(int(c.get("person_id")))
        except Exception:
            pass

    def sample_ids(pool: CrewYearPool | None, n: int, genre_boost_genres: Iterable[str] | None = None, preferred_ids: set[int] | None = None) -> list[int]:
        if pool is None or n <= 0 or len(pool.person_ids) == 0:
            return []
        band_idx = _crew_candidate_band(world, pool, genre, n)
        if band_idx.size == 0:
            return []
        if preferred_ids:
            pref_idx = [pool.pid_to_local.get(int(pid)) for pid in preferred_ids]
            pref_idx = [int(idx) for idx in pref_idx if idx is not None]
            if pref_idx:
                band_idx = np.unique(np.concatenate([band_idx, np.asarray(pref_idx, dtype=np.int32)]))
        pool_ids = pool.person_ids[band_idx]
        weights = _crew_genre_weights(pool, genre)[band_idx].copy()
        if len(weights) == 0:
            return []
        band_local = {int(pid): idx for idx, pid in enumerate(pool_ids)}
        if genre_boost_genres and str(genre).lower() in {str(g).lower() for g in genre_boost_genres}:
            boosted_mask = _crew_genre_match_mask(pool, genre)[band_idx]
            if boosted_mask.any():
                weights[boosted_mask] *= 1.5
        if preferred_ids:
            if len(preferred_ids) <= 24:
                for pid in preferred_ids:
                    local_idx = band_local.get(int(pid))
                    if local_idx is not None:
                        weights[int(local_idx)] *= 8.0
            else:
                pref_mask = np.isin(pool_ids, np.array(sorted(preferred_ids), dtype=int))
                weights[pref_mask] *= 8.0

        if used:
            if len(used) <= 32:
                for pid in used:
                    local_idx = band_local.get(int(pid))
                    if local_idx is not None:
                        weights[int(local_idx)] = 0.0
            else:
                weights[np.isin(pool_ids, np.array(sorted(used), dtype=int))] = 0.0
        eligible = _shortlist_indices(
            weights,
            min(len(pool_ids), _shortlist_budget(world, "crew", max(32, n * 8))),
            world.rng,
            exploration_share=_prior_float(world, "crew_exploration_share", 0.30, lo=0.0, hi=0.60),
        )
        if eligible.size == 0:
            eligible = np.flatnonzero(weights > 0)
        if eligible.size == 0:
            return []

        chosen_count = min(int(n), int(eligible.size))
        probs = normalize_weights(weights[eligible])
        chosen = world.rng.choice(eligible, size=chosen_count, replace=False, p=probs)
        ids: list[int] = []
        for idx in chosen:
            pid = int(pool_ids[int(idx)])
            if pid not in used:
                ids.append(pid)
                used.add(pid)
        return ids

    crew_rows = []
    credit = 1
    for role, cfg in CREW_DEPARTMENTS.items():
        count = int(cfg["count"].get(tier, 0))
        if count <= 0:
            continue
        pool = _get_crew_year_pool(world, role, year)
        if pool is None:
            if not hasattr(world, "_crew_fallback_warned"):
                world._crew_fallback_warned = set()
            if role not in world._crew_fallback_warned:
                print(f"  [WARN] H1: crew pool empty for role='{role}' -- falling back to actor pool")
                world._crew_fallback_warned.add(role)
            pool = _get_crew_year_pool(world, "actor_fallback", year)
        if role == "writer" and director_id is not None:
            if not hasattr(world, "director_writer_history"):
                world.director_writer_history = {}
            prev_writers = world.director_writer_history.get(int(director_id), set())
        else:
            prev_writers = None
        for pid in sample_ids(pool, count, genre_boost_genres=cfg.get("genre_boost"), preferred_ids=prev_writers):
            crew_rows.append({
                "person_id": int(pid),
                "crew_role": role,
                "credit_order": credit,
                "department": _CREW_DEPT.get(role, "Production"),
            })
            if role == "writer" and director_id is not None:
                if not hasattr(world, "director_writer_history"):
                    world.director_writer_history = {}
                world.director_writer_history.setdefault(int(director_id), set()).add(int(pid))
            credit += 1
    return crew_rows
