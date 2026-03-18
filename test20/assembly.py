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
from dataclasses import dataclass
from typing import Iterable

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
from utils import TIER_TO_LATENT_IDX, TONE_STYLE_HINTS, normalize_weights
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
    priors = getattr(getattr(getattr(world, "workspace", None), "config", None), "priors", None)
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


def _person_company_multiplier(world: WorldState, pids: np.ndarray, tier: str, genre: str) -> np.ndarray:
    # On-demand P-C scoring: replaces precomputed person_company_affinity index
    _ensure_company_lookup_cache(world)
    suitable = (
        world._company_by_tier_genre.get((tier, genre.lower()), set())
        | world._company_by_tier_genre.get(("", genre.lower()), set())
        | world._company_by_tier_genre.get((tier, ""), set())
    )
    if not suitable:
        return np.ones(len(pids), dtype=float)
    return world.compute_pc_affinity_batch(pids, suitable)


def _director_edge_arrays(world: WorldState, director_id: int, pids: np.ndarray, year: int) -> tuple[np.ndarray, np.ndarray]:
    aff = world.affinity_index or {}
    pref_map = {}
    for edge in aff.get("director_prefs", {}).get(int(director_id), []):
        if isinstance(edge, dict):
            aid = int(edge.get("actor_id", -1))
            if edge_is_active(world, "mentorship", int(director_id), aid, 0, year, float(edge.get("weight", 0.5) or 0.5), edge.get("valid_from"), edge.get("valid_to"))[0]:
                pref_map[aid] = float(edge.get("weight", 0.5) or 0.5)
        else:
            try:
                aid = int(edge[0])
                pref_map[aid] = float(edge[1]) if len(edge) > 1 else 0.5
            except Exception:
                continue
    pref = np.ones(len(pids), dtype=float)
    if pref_map:
        arr = np.array([pref_map.get(int(pid), -1.0) for pid in pids], dtype=float)
        mask = arr >= 0.0
        arr = np.clip(arr, 0.0, 1.0)
        pref[mask] = 1.0 + 9.0 * arr[mask]

    avoid_set: set[int] = set()
    for edge in aff.get("director_avoids", {}).get(int(director_id), []):
        if isinstance(edge, dict):
            aid = int(edge.get("actor_id", -1))
            if edge_is_active(world, "avoid", int(director_id), aid, 0, year, 1.0, edge.get("valid_from"), edge.get("valid_to"))[0]:
                avoid_set.add(aid)
        else:
            try:
                avoid_set.add(int(edge))
            except Exception:
                pass
    avoid_mask = np.isin(pids, np.array(list(avoid_set), dtype=int)) if avoid_set else np.zeros(len(pids), dtype=bool)
    return pref, avoid_mask


@dataclass(slots=True)
class ActorStaticBlock:
    candidates: pd.DataFrame
    pids: np.ndarray
    base_scores: np.ndarray
    career_mult: np.ndarray
    career_stage_mult: np.ndarray
    nationality_mult: np.ndarray
    genre_mult: np.ndarray
    style_mult: np.ndarray
    avoid_genre_mult: np.ndarray
    market_mult: np.ndarray
    collab_mult: np.ndarray
    controversy_mult: np.ndarray
    volatility_mult: np.ndarray
    csv_mult: np.ndarray
    company_aff_mult: np.ndarray
    cooldown_mult: np.ndarray
    workload_mult: np.ndarray
    unused_flags: np.ndarray
    actor_genders: np.ndarray | None
    actor_nationalities: np.ndarray | None
    actor_tag_sets: list[set[str]] | None
    actor_tag_bitmasks: np.ndarray | None  # uint64 bitmask per candidate
    pid_to_idx: dict[int, int]
    top_star_mask: np.ndarray
    cand_agencies: np.ndarray
    cand_communities: np.ndarray
    _sim_threshold: float


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
    cast_size = _sample_cast_size(world, concept)

    if year in getattr(world, "_year_cache", {}):
        # P2-FIX: cache stores boolean masks, not DataFrame copies
        mask = world._year_cache[year]
        candidates = world.actors[mask].copy()
    else:
        base = _active_year_subset(world.actors, year).copy()
        if "genre_affinity" in base.columns:
            base["_ga_lower"] = base["genre_affinity"].fillna("").astype(str).str.lower()
        if "style_tags" in base.columns:
            base["_st_lower"] = base["style_tags"].fillna("").astype(str).str.lower()
        # P2-FIX: store mask in cache for lazily-computed years
        world._year_cache[year] = np.ones(len(world.actors), dtype=bool)
        candidates = base.copy()

    min_pool = max(cast_size * 4, 50)

    if len(candidates) > min_pool and "_ga_lower" in candidates.columns:
        filtered = candidates[candidates["_ga_lower"].str.contains(genre.lower(), regex=False)]
        if len(filtered) >= min_pool:
            candidates = filtered

    if len(candidates) > min_pool and world.person_latent:
        tier_idx = TIER_TO_LATENT_IDX.get(tier, 2)
        pids = candidates["person_id"].astype(int).values
        tier_ok = np.ones(len(candidates), dtype=bool)
        for i, pid in enumerate(pids):
            lv = world.person_latent.get(int(pid))
            if lv is not None:
                bbp = lv.get("budget_band_pref")
                if isinstance(bbp, list) and len(bbp) > tier_idx:
                    tier_ok[i] = float(bbp[tier_idx]) >= 0.05
        filtered = candidates[tier_ok]
        if len(filtered) >= min_pool:
            candidates = filtered

    _ensure_actor_workload_counter(world)
    if len(candidates) > min_pool:
        pids = candidates["person_id"].astype(int).values
        ym = candidates["yearly_max"].values.astype(float) if "yearly_max" in candidates.columns else np.full(len(candidates), 5.0)
        ym = np.where(np.isnan(ym) | (ym <= 0), 5.0, ym)
        wl = world._yearly_workload
        film_counts = np.array([wl.get((int(pid), year), 0) for pid in pids], dtype=float)
        keep = film_counts < ym
        filtered = candidates[keep]
        if len(filtered) >= min_pool:
            candidates = filtered

    pids = candidates["person_id"].astype(int).values
    n = len(candidates)
    base_scores = candidates["pop_weight"].astype(float).values.copy()
    hist = np.array([world.person_film_count.get(int(pid), 0) for pid in pids], dtype=float)
    star_momentum = np.clip(1.0 + 0.22 * np.log1p(hist), 1.0, 3.4)
    base_scores *= star_momentum

    if "peak_start" in candidates.columns and "peak_end" in candidates.columns:
        peak_s = candidates["peak_start"].fillna(1900).astype(int).values
        peak_e = candidates["peak_end"].fillna(1901).astype(int).values
        valid = peak_e >= peak_s
        in_peak = valid & (peak_s <= year) & (year <= peak_e)
        career_mult = np.where(in_peak, 3.0, 1.0)
    else:
        career_mult = np.ones(n, dtype=float)

    stage_map = {
        "legend": _env_float("V16_LEGEND_MULT", 15.0, 4.0, 30.0),
        "prime": _env_float("V16_PRIME_MULT", 5.0, 1.0, 12.0),
        "veteran": _env_float("V16_VETERAN_MULT", 2.5, 0.5, 10.0),
        "rising": _env_float("V16_RISING_MULT", 1.0, 0.2, 4.0),
        "retired": _env_float("V16_RETIRED_MULT", 0.08, 0.01, 1.5),
    }
    stage_vals = candidates["career_stage"].fillna("prime").astype(str).str.lower().values if "career_stage" in candidates.columns else np.full(n, "prime")
    career_stage_mult = np.array([stage_map.get(str(s), 1.0) for s in stage_vals], dtype=float)

    movie_country = str(concept.get("country", ""))
    geo_boost = _GEO_BOOST_BY_TIER.get(tier, 2.0)
    if movie_country and "nationality" in candidates.columns:
        nat = candidates["nationality"].fillna("").astype(str).values
        nat_country = np.array([_NATIONALITY_TO_COUNTRY.get(v, "") for v in nat])
        nationality_mult = np.where(nat_country == movie_country, geo_boost, 1.0)
        actor_nats = candidates["nationality"].fillna("unknown").astype(str).str.lower().values
    else:
        nationality_mult = np.ones(n, dtype=float)
        actor_nats = None

    if "_ga_lower" in candidates.columns:
        genre_match = candidates["_ga_lower"].str.contains(genre.lower(), regex=False).values
    elif "genre_affinity" in candidates.columns:
        ga = candidates["genre_affinity"].fillna("").astype(str).str.lower()
        genre_match = ga.str.contains(genre.lower(), regex=False).values
    else:
        genre_match = np.zeros(n, dtype=bool)
    genre_mult = np.where(genre_match, 5.0, 1.0)

    hints = TONE_STYLE_HINTS.get(tone, [tone])
    if hints:
        if "_st_lower" in candidates.columns:
            st_col = candidates["_st_lower"]
        elif "style_tags" in candidates.columns:
            st_col = candidates["style_tags"].fillna("").astype(str).str.lower()
        else:
            st_col = None
        if st_col is not None:
            tagged = ";" + st_col.str.replace(",", ";", regex=False) + ";"
            style_match = np.zeros(n, dtype=bool)
            for hint in hints:
                if hint:
                    style_match |= tagged.str.contains(f";{hint};", regex=False).values
            style_mult = np.where(style_match, 2.0, 1.0)
            actor_tag_sets = [set(t.strip().lower() for t in raw.replace(",", ";").split(";") if t.strip()) for raw in st_col.values]
        else:
            style_mult = np.ones(n, dtype=float)
            actor_tag_sets = None
    else:
        style_mult = np.ones(n, dtype=float)
        actor_tag_sets = None

    avoid_genre_mult = np.ones(n, dtype=float)
    sparse_avoid = getattr(world, "_latent_avoid_genres", None)
    if sparse_avoid:
        for i, pid in enumerate(pids):
            if genre in sparse_avoid.get(int(pid), set()):
                avoid_genre_mult[i] = 0.15

    market_mult = np.ones(n, dtype=float)
    if "market_fit" in candidates.columns and movie_country:
        target_market = _COUNTRY_TO_MARKET.get(movie_country, "")
        if target_market:
            mf = candidates["market_fit"].fillna("").astype(str).str.lower()
            market_mult = np.where(
                mf.str.contains("global", regex=False).values | mf.str.contains(target_market.lower(), regex=False).values,
                2.0,
                1.0,
            )

    collab_mult = np.ones(n, dtype=float)
    latent_idx = getattr(world, "_latent_pid_to_idx", None)
    latent_collab = getattr(world, "_latent_collab", None)

    # Build latent index array ONCE — reused for collab, controversy, volatility, CSV
    li_arr = None
    valid_li = None
    if latent_idx is not None:
        li_arr = np.array([latent_idx.get(int(pid), -1) for pid in pids], dtype=int)
        valid_li = li_arr >= 0

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

    company_aff_mult = _person_company_multiplier(world, pids, tier, genre)

    cooldown_floor = {"legend": 0.70, "prime": 0.45, "veteran": 0.50, "rising": 0.25, "retired": 0.05}
    cooldown_decay = {"legend": 0.07, "prime": 0.13, "veteran": 0.10, "rising": 0.18, "retired": 0.30}
    cooldown_mult = np.ones(n, dtype=float)
    workload_mult = np.ones(n, dtype=float)
    unused_flags = np.zeros(n, dtype=bool)
    ym = candidates["yearly_max"].values.astype(float) if "yearly_max" in candidates.columns else np.full(n, 5.0)
    ym = np.where(np.isnan(ym) | (ym <= 0), 5.0, ym)
    for i, pid in enumerate(pids):
        stage = str(stage_vals[i])
        recent = world.person_recent.get(int(pid))
        if not recent:
            unused_flags[i] = True
        else:
            recent_window = sum(1 for y in recent if abs(y - year) <= 2)
            if recent_window > 0:
                cooldown_mult[i] = max(cooldown_floor.get(stage, 0.3), 1.0 - cooldown_decay.get(stage, 0.15) * recent_window)
            if world._yearly_workload.get((int(pid), year), 0) >= int(ym[i]):
                workload_mult[i] = 0.1

    actor_genders = candidates["gender"].fillna("unknown").astype(str).str.lower().values if "gender" in candidates.columns else None
    top_cut = float(np.quantile(base_scores, 0.95)) if len(base_scores) > 30 else float(np.max(base_scores) if len(base_scores) else 0.0)
    top_star_mask = base_scores >= top_cut

    # Pre-build agency / community arrays (avoids rebuilding per slot)
    cand_agencies = np.array([world.person_agency.get(int(pid), -1) for pid in pids], dtype=np.int32)
    communities = getattr(world, "communities", None) or {}
    cand_communities = np.array([communities.get(int(pid), -1) for pid in pids], dtype=np.int32)

    sorted_base = np.sort(base_scores * career_mult * career_stage_mult)[::-1]
    sim_threshold = float(sorted_base[min(199, len(sorted_base) - 1)]) * 0.5 if len(sorted_base) else 0.0

    # Build tag bitmasks for vectorized Jaccard
    tag_bit_map = _ensure_tag_bit_mapping(world)
    actor_tag_bitmasks = _build_tag_bitmasks(actor_tag_sets, tag_bit_map)

    return ActorStaticBlock(
        candidates=candidates,
        pids=pids,
        base_scores=base_scores,
        career_mult=career_mult,
        career_stage_mult=career_stage_mult,
        nationality_mult=nationality_mult,
        genre_mult=genre_mult,
        style_mult=style_mult,
        avoid_genre_mult=avoid_genre_mult,
        market_mult=market_mult,
        collab_mult=collab_mult,
        controversy_mult=controversy_mult,
        volatility_mult=volatility_mult,
        csv_mult=csv_mult,
        company_aff_mult=company_aff_mult,
        cooldown_mult=cooldown_mult,
        workload_mult=workload_mult,
        unused_flags=unused_flags,
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

    shortlist = _shortlist_indices(weights, _shortlist_budget(world, "director", 24), world.rng, exploration_share=0.30)
    if shortlist.size > 0 and shortlist.size < len(dirs):
        dirs = dirs.iloc[shortlist].copy()
        weights = weights[shortlist]
        dir_pids = dirs["person_id"].astype(int).values

    risk_target, ambition_target, prestige_target = _concept_latent_targets(concept)
    alignment = np.ones(len(dirs), dtype=float)
    for i, pid in enumerate(dir_pids):
        lv = get_person_latent(world, int(pid))
        risk_align = 1.0 - abs(_safe01(lv.get("risk_tolerance"), 0.5) - risk_target)
        ambition_align = 1.0 - abs(_safe01(lv.get("artistic_ambition"), 0.5) - ambition_target)
        prestige_align = 1.0 - abs(_safe01(lv.get("public_reputation"), 0.5) - prestige_target)
        alignment[i] = 0.40 * risk_align + 0.35 * ambition_align + 0.25 * prestige_align
    weights *= np.clip(0.70 + 0.80 * alignment, 0.35, 1.75)

    csv_target = _concept_csv_target(concept)
    csv_alignment = np.ones(len(dirs), dtype=float)
    for i, pid in enumerate(dir_pids):
        csv_alignment[i] = _cosine_sim(get_person_latent(world, int(pid)).get("creative_style_vector", [0.0] * 8), csv_target)
    weights *= np.clip(0.80 + 0.50 * csv_alignment, 0.60, 1.40)

    _ensure_company_lookup_cache(world)
    suitable = (
        world._company_by_tier_genre.get((tier, genre.lower()), set())
        | world._company_by_tier_genre.get(("", genre.lower()), set())
        | world._company_by_tier_genre.get((tier, ""), set())
    )
    # On-demand P-C scoring for director-company affinity
    if suitable:
        pc_mult = world.compute_pc_affinity_batch(dir_pids, suitable)
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

    active = comps.copy()
    if "founded_year" in active.columns:
        active = active[active["founded_year"].fillna(0).astype(int) <= year]
    if "defunct_year" in active.columns:
        active = active[active["defunct_year"].isna() | (active["defunct_year"].fillna(2100).astype(float) >= float(year))]
    if len(active) < 3:
        active = comps.copy()

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

    shortlist = _shortlist_indices(weights, _shortlist_budget(world, "company", 18), world.rng, exploration_share=0.35)
    if shortlist.size > 0 and shortlist.size < len(active):
        active = active.iloc[shortlist].copy()
        weights = weights[shortlist]

    cids = active["company_id"].astype(int).values
    tier_idx = TIER_TO_LATENT_IDX.get(tier, 2)
    risk_target, _, prestige_target = _concept_latent_targets(concept)
    align = np.ones(len(active), dtype=float)
    genre_axis = ["Action", "Drama", "Comedy", "Sci-Fi", "Horror", "Romance", "Thriller", "Fantasy", "Mystery", "Documentary", "Crime", "Animation"]
    gidx = genre_axis.index(genre) if genre in genre_axis else -1
    for i, cid in enumerate(cids):
        lv = world.company_latent.get(int(cid), {}) if hasattr(world, "company_latent") else {}
        risk_align = 1.0 - abs(_safe01(lv.get("risk_appetite"), 0.5) - risk_target)
        prestige_align = 1.0 - abs(_safe01(lv.get("prestige_score"), 0.5) - prestige_target)
        bbp = lv.get("budget_tier_focus") if isinstance(lv, dict) else None
        focus = _safe01(bbp[tier_idx], 0.5) if isinstance(bbp, list) and len(bbp) > tier_idx else 0.5
        gp = lv.get("genre_portfolio") if isinstance(lv, dict) else None
        genre_fit = _safe01(gp[gidx], 0.5) if isinstance(gp, list) and gidx >= 0 and len(gp) > gidx else 0.5
        align[i] = 0.28 * risk_align + 0.22 * prestige_align + 0.30 * focus + 0.20 * genre_fit
    weights *= np.clip(0.65 + 0.90 * align, 0.25, 1.80)
    weights = normalize_weights(weights)
    if len(active) == 0 or float(np.sum(weights)) <= 0:
        return []

    primary_shortlist = _shortlist_indices(weights, min(len(active), _shortlist_budget(world, "company", 18)), world.rng, exploration_share=0.35)
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
            sl = _shortlist_indices(pw, min(len(active), _shortlist_budget(world, "company", 18)), world.rng, exploration_share=0.40)
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


def pick_cast(world: WorldState, concept: dict, director_id: int, max_retries: int = 3) -> tuple[list[dict], list[tuple[int, int]]]:
    tier = str(concept["tier"])
    year = int(concept["year"])
    cast_size = _sample_cast_size(world, concept)
    static = _build_actor_static_block(world, concept)
    pids = static.pids
    n_cand = len(pids)
    director_pref_boost, avoid_mask = _director_edge_arrays(world, int(director_id) if director_id is not None else -1, pids, year)

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

    for _attempt in range(max_retries):
        cast.clear()
        competition_pairs.clear()
        cast_ids.clear()
        cast_id_list.clear()
        forbidden.clear()
        static.unused_flags[:] = static.unused_flags

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
                scores[static.unused_flags] *= 1.5
            if slot >= 4:
                scores[static.top_star_mask] *= _env_float("V16_TOP_STAR_SLOT_PENALTY", 0.82, 0.5, 1.0)

            if cast_id_list:
                # ---- Agency boost (pre-built array, np.isin) ----
                cast_agency_set = {world.person_agency.get(cid) for cid in cast_id_list}
                cast_agency_set.discard(None)
                if cast_agency_set:
                    agency_match = np.isin(static.cand_agencies, list(cast_agency_set))
                    scores[agency_match & (scores > 0)] *= 2.0

                # ---- Community boost (pre-built array, np.isin) ----
                cast_com_set = set()
                for cid in cast_id_list:
                    c = static.cand_communities[static.pid_to_idx[cid]] if cid in static.pid_to_idx else -1
                    if c >= 0:
                        cast_com_set.add(c)
                if cast_com_set:
                    com_match = np.isin(static.cand_communities, list(cast_com_set))
                    scores[com_match & (scores > 0)] *= 1.5

                # ---- Ensemble diversity (gender/nationality/tag overlap) ----
                if static.actor_tag_sets is not None:
                    # Build friend set from adjacency for diversity check
                    cast_friend_pids = set()
                    for cid in cast_id_list:
                        for (nbr, _w, _vf, _vt) in world._friend_adj_all.get(int(cid), []):
                            cast_friend_pids.add(nbr)
                    cast_tag_sets = [static.actor_tag_sets[static.pid_to_idx[cid]] for cid in cast_id_list if cid in static.pid_to_idx]
                    cast_genders = Counter(static.actor_genders[static.pid_to_idx[cid]] for cid in cast_id_list if static.actor_genders is not None and cid in static.pid_to_idx)
                    cast_nats = Counter(static.actor_nationalities[static.pid_to_idx[cid]] for cid in cast_id_list if static.actor_nationalities is not None and cid in static.pid_to_idx)
                    # Vectorized gender/nationality diversity
                    active_mask = scores > 0
                    if cast_genders and static.actor_genders is not None:
                        gender_present = set(cast_genders.keys())
                        gender_novel = np.array([g not in gender_present for g in static.actor_genders], dtype=bool)
                        scores[active_mask & gender_novel] *= 1.5
                    if len(cast_nats) >= 2 and static.actor_nationalities is not None:
                        nat_present = set(cast_nats.keys())
                        nat_novel = np.array([n not in nat_present for n in static.actor_nationalities], dtype=bool)
                        scores[active_mask & nat_novel] *= 1.3
                    # Tag overlap penalty — vectorized via uint64 bitmask Jaccard.
                    # Same semantics as before: penalize candidates with >50%
                    # Jaccard overlap with any single cast member, exempt friends.
                    if cast_tag_sets and static.actor_tag_bitmasks is not None:
                        cast_bitmasks = np.array(
                            [static.actor_tag_bitmasks[static.pid_to_idx[cid]]
                             for cid in cast_id_list if cid in static.pid_to_idx],
                            dtype=np.uint64,
                        )
                        # Build friend exclusion mask (friends skip penalty)
                        friend_mask = np.isin(pids, np.array(list(cast_friend_pids), dtype=int)) if cast_friend_pids else np.zeros(n_cand, dtype=bool)
                        # Has-tags mask: candidates with at least one tag bit set
                        has_tags = static.actor_tag_bitmasks > np.uint64(0)
                        # Eligible for penalty: active, has tags, not a friend
                        eligible = active_mask & has_tags & ~friend_mask

                        if eligible.any() and len(cast_bitmasks) > 0:
                            max_jaccard = np.zeros(n_cand, dtype=np.float32)
                            cand_bm = static.actor_tag_bitmasks  # full array
                            for cbm in cast_bitmasks:
                                inter = _popcount_vec(cand_bm & cbm)
                                union = _popcount_vec(cand_bm | cbm)
                                jaccard = np.where(union > 0, inter.astype(np.float32) / union.astype(np.float32), 0.0)
                                np.maximum(max_jaccard, jaccard, out=max_jaccard)
                            penalty_mask = eligible & (max_jaccard > 0.5)
                            scores[penalty_mask] *= 0.5

                # ---- Friendship / rivalry via adjacency traversal ----
                # O(cast_size × avg_degree) instead of O(candidates × cast_size)
                friend_boost = np.zeros(n_cand, dtype=np.float32)
                rival_penalty = np.ones(n_cand, dtype=np.float32)
                for cid in cast_id_list:
                    # Traverse this cast member's friends
                    for (nbr, w, vf, vt) in world._friend_adj_all.get(int(cid), []):
                        idx = static.pid_to_idx.get(nbr)
                        if idx is not None and scores[idx] > 0 and friend_boost[idx] == 0:
                            if edge_is_active(world, "friendship", min(cid, nbr), max(cid, nbr), 0, year, w, vf, vt)[0]:
                                friend_boost[idx] = 1.0 + 4.0 * max(0.15, w)
                    # Traverse this cast member's rivals
                    for (nbr, w, vf, vt) in world._rival_adj_all.get(int(cid), []):
                        idx = static.pid_to_idx.get(nbr)
                        if idx is not None and scores[idx] > 0:
                            if edge_is_active(world, "rivalry", min(cid, nbr), max(cid, nbr), 0, year, w, vf, vt)[0]:
                                rival_penalty[idx] = min(rival_penalty[idx], max(0.0, 1.0 - max(0.6, w)))
                # Apply friendship boost and rivalry penalty
                fb_mask = friend_boost > 0
                scores[fb_mask] *= friend_boost[fb_mask]
                scores *= rival_penalty

                # ---- Latent similarity (full 6-component, batched) ----
                need_similarity = np.flatnonzero((scores >= static._sim_threshold) & ~fb_mask)
                if need_similarity.size > 0 and cast_id_list:
                    cast_li = np.array([world._latent_pid_to_idx.get(cid, -1) for cid in cast_id_list[:3]], dtype=int)
                    cast_li = cast_li[cast_li >= 0]
                    if cast_li.size > 0:
                        ns_li = np.array([world._latent_pid_to_idx.get(int(pids[i]), -1) for i in need_similarity], dtype=int)
                        valid = ns_li >= 0
                        if valid.any():
                            valid_idx = need_similarity[valid]
                            max_sim = latent_similarity_batch(world, ns_li[valid], cast_li)
                            boost = max_sim > 0.1
                            if boost.any():
                                scores[valid_idx[boost]] *= (1.0 + max_sim[boost])

            # Awards bump — vectorized via pre-built set
            if award_recent_pids:
                award_mask = np.isin(pids, np.array(list(award_recent_pids), dtype=int))
                scores[award_mask & (scores > 0)] *= 1.5

            # Franchise cast pool boost — vectorized via pre-built set
            if franchise_pool:
                fpool_mask = np.isin(pids, np.array(list(franchise_pool), dtype=int))
                scores[fpool_mask & (scores > 0)] *= 2.0

            valid = np.flatnonzero(scores > 0)
            if valid.size == 0:
                break
            exp = 2.5 if slot == 0 else (1.3 if slot == 1 else 0.7)
            probs = normalize_weights(scores[valid] ** exp)
            chosen_idx = int(world.rng.choice(valid, p=probs))
            pid = int(pids[chosen_idx])

            if slot <= 1 and valid.size >= 2 and len(competition_pairs) < 4:
                order = valid[np.argsort(scores[valid])[::-1]]
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
                static.unused_flags[idx] = False

            # Keep direct rivals from being added later in the same cast
            rival_neighbors = set(nbr for (nbr, _w, _vf, _vt) in world._rival_adj_all.get(int(pid), []))
            for cid in cast_id_list:
                if cid in rival_neighbors:
                    forbidden.add(cid)

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

    def sample_ids(df: pd.DataFrame, n: int, genre_boost_genres: Iterable[str] | None = None) -> list[int]:
        if df is None or len(df) == 0 or n <= 0:
            return []
        pool = _active_year_subset(df, year)
        if len(pool) == 0:
            return []
        w = pool.get("pop_weight", pd.Series(np.ones(len(pool), dtype=float))).astype(float).values
        if "genre_affinity" in pool.columns:
            ga = pool["genre_affinity"].fillna("").astype(str).str.lower()
            w *= (1.0 + 0.60 * ga.str.contains(genre.lower(), regex=False).astype(float).values)
        if genre_boost_genres and genre in set(genre_boost_genres):
            w *= 1.5
        probs = normalize_weights(w)
        n = min(int(n), len(pool))
        chosen = world.rng.choice(len(pool), size=n, replace=False, p=probs)
        ids = []
        for idx in chosen:
            pid = int(pool.iloc[int(idx)]["person_id"])
            if pid not in used:
                ids.append(pid)
                used.add(pid)
        if len(ids) < n:
            remaining = pool[~pool["person_id"].astype(int).isin(used)]
            if len(remaining) > 0:
                probs2 = normalize_weights(remaining.get("pop_weight", pd.Series(np.ones(len(remaining), dtype=float))).astype(float).values)
                extra = world.rng.choice(len(remaining), size=min(n - len(ids), len(remaining)), replace=False, p=probs2)
                for idx in extra:
                    pid = int(remaining.iloc[int(idx)]["person_id"])
                    ids.append(pid)
                    used.add(pid)
        return ids

    crew_rows = []
    credit = 1
    for role, cfg in CREW_DEPARTMENTS.items():
        count = int(cfg["count"].get(tier, 0))
        if count <= 0:
            continue
        pool = getattr(world, "crew_pools", {}).get(role) if hasattr(world, "crew_pools") else None
        if pool is None or len(pool) == 0:
            fallback_attr = cfg.get("pool_fallback", "persons")
            pool = getattr(world, fallback_attr, None)
            if pool is None or len(pool) == 0:
                if not hasattr(world, "_crew_fallback_warned"):
                    world._crew_fallback_warned = set()
                if role not in world._crew_fallback_warned:
                    print(f"  [WARN] H1: crew pool empty for role='{role}' -- falling back to actor pool")
                    world._crew_fallback_warned.add(role)
                pool = world.actors
        if role == "writer" and director_id is not None:
            if not hasattr(world, "director_writer_history"):
                world.director_writer_history = {}
            prev_writers = world.director_writer_history.get(int(director_id), set())
            if prev_writers and len(pool) > 0 and "person_id" in pool.columns:
                pool = pool.copy()
                if "pop_weight" not in pool.columns:
                    pool["pop_weight"] = 1.0
                mask = pool["person_id"].astype(int).isin(prev_writers)
                pool.loc[mask, "pop_weight"] = pool.loc[mask, "pop_weight"] * 8.0
        for pid in sample_ids(pool, count, genre_boost_genres=cfg.get("genre_boost")):
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
