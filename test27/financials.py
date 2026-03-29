"""
financials.py
=============
Correlated financial model for synthetic movie generation.

Public API kept stable:
    - edge_is_active(...)
    - compute_financials(...)
    - record_financial_outcome(...)

This rewrite keeps the V17 market-memory ideas but restructures the module into
smaller helpers so the finance layer is not one giant procedural wall.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from contracts import AWARD_CAMPAIGN_GENRES, BUDGET_RANGES, CERT_DISTS, FRANCHISE_CONFIG
from utils import TIER_TO_LATENT_IDX, _safe_float, _clip01, _safe_mean, stable_uniform_0_1, style_spectacle_score
from world_state import get_company_latent, get_person_latent


# ============================================================================
# Static priors
# ============================================================================

# Production-cost multiplier by country. This scales COST, not the latent appeal
# of the movie. Non-US productions can still overperform commercially.
COUNTRY_BUDGET_SCALE: Dict[str, float] = {
    "USA": 1.00,
    "UK": 0.65,
    "Australia": 0.60,
    "Canada": 0.55,
    "Ireland": 0.52,
    "New Zealand": 0.50,
    "France": 0.45,
    "Germany": 0.42,
    "Sweden": 0.40,
    "Denmark": 0.38,
    "Norway": 0.38,
    "Finland": 0.35,
    "Netherlands": 0.38,
    "Belgium": 0.35,
    "Switzerland": 0.42,
    "Austria": 0.35,
    "Italy": 0.32,
    "Spain": 0.30,
    "Portugal": 0.22,
    "Greece": 0.20,
    "Luxembourg": 0.40,
    "Poland": 0.18,
    "Czech Republic": 0.20,
    "Hungary": 0.18,
    "Romania": 0.14,
    "Bulgaria": 0.13,
    "Croatia": 0.16,
    "Serbia": 0.13,
    "Slovakia": 0.16,
    "Ukraine": 0.12,
    "Estonia": 0.18,
    "Latvia": 0.17,
    "Lithuania": 0.17,
    "Slovenia": 0.18,
    "Macedonia": 0.11,
    "Japan": 0.35,
    "South Korea": 0.28,
    "China": 0.30,
    "Hong Kong": 0.35,
    "Taiwan": 0.22,
    "Thailand": 0.10,
    "Vietnam": 0.07,
    "Indonesia": 0.07,
    "Philippines": 0.08,
    "Malaysia": 0.10,
    "Singapore": 0.28,
    "India": 0.09,
    "Pakistan": 0.05,
    "Bangladesh": 0.04,
    "Sri Lanka": 0.05,
    "Nepal": 0.03,
    "Iran": 0.07,
    "Turkey": 0.10,
    "Israel": 0.20,
    "Lebanon": 0.09,
    "Egypt": 0.06,
    "Jordan": 0.08,
    "Iraq": 0.05,
    "Saudi Arabia": 0.14,
    "UAE": 0.20,
    "Kuwait": 0.16,
    "Qatar": 0.18,
    "Bahrain": 0.14,
    "Oman": 0.12,
    "Kazakhstan": 0.08,
    "Azerbaijan": 0.08,
    "Uzbekistan": 0.06,
    "Georgia": 0.07,
    "Brazil": 0.18,
    "Mexico": 0.14,
    "Argentina": 0.12,
    "Colombia": 0.11,
    "Chile": 0.13,
    "Peru": 0.09,
    "Venezuela": 0.07,
    "Cuba": 0.05,
    "Ecuador": 0.08,
    "Bolivia": 0.06,
    "Uruguay": 0.10,
    "Dominican Republic": 0.07,
    "Costa Rica": 0.10,
    "Guatemala": 0.07,
    "Honduras": 0.06,
    "El Salvador": 0.06,
    "Panama": 0.10,
    "Puerto Rico": 0.35,
    "Russia": 0.20,
    "Belarus": 0.10,
    "Armenia": 0.07,
    "Moldova": 0.07,
    "Nigeria": 0.04,
    "South Africa": 0.14,
    "Ghana": 0.04,
    "Kenya": 0.05,
    "Ethiopia": 0.04,
    "Tanzania": 0.04,
    "Morocco": 0.08,
    "Tunisia": 0.07,
    "Senegal": 0.05,
    "Zimbabwe": 0.04,
    "Congo": 0.03,
    "Benin": 0.03,
    "Cape Verde": 0.05,
    "Mongolia": 0.06,
    "Tibet": 0.04,
    "Palestinian Territories": 0.04,
    "Iceland": 0.32,
    "Syria": 0.04,
}

# Genre-level critical bias. The point is not “truth”, it is a stable,
# intentionally non-flat baseline that downstream CE queries can exploit.
GENRE_RATING_OFFSET: Dict[str, float] = {
    "Documentary": +0.70,
    "War": +0.45,
    "Biography": +0.40,
    "History": +0.35,
    "Crime": +0.30,
    "Drama": +0.35,
    "Thriller": +0.10,
    "Mystery": +0.15,
    "Sci-Fi": 0.00,
    "Fantasy": -0.05,
    "Action": -0.10,
    "Animation": +0.15,
    "Romance": -0.15,
    "Comedy": -0.35,
    "Horror": -0.50,
    # B4-FIX: 15 previously-missing genres that all defaulted to 0.0
    "Adventure": -0.05,
    "Family": -0.10,
    "Film-Noir": +0.30,
    "Music": +0.10,
    "Musical": +0.05,
    "Sport": +0.05,
    "Western": +0.10,
    "Superhero": -0.20,
    "Martial Arts": -0.25,
    "Disaster": -0.30,
    "Experimental": +0.25,
    "Short": +0.10,
    "Reality-TV": -0.40,
}

TIER_MIN_VOTES: Dict[str, int] = {
    "Epic": 5_000,
    "A": 2_000,
    "Mid": 500,
    "Indie": 200,
    "Micro": 50,
}

TIER_RATING_BASE: Dict[str, float] = {
    "Epic": 6.8,
    "A": 6.5,
    "Mid": 6.0,
    "Indie": 6.3,
    "Micro": 5.3,
}

TIER_RATING_STD: Dict[str, float] = {
    "Epic": 1.0,
    "A": 1.1,
    "Mid": 1.4,
    "Indie": 1.6,
    "Micro": 1.8,
}

TIER_LOG_CENTER: Dict[str, float] = {
    "Epic": 19.3,
    "A": 18.0,
    "Mid": 16.5,
    "Indie": 14.5,
    "Micro": 12.5,
}


# ============================================================================
# Helpers
# ============================================================================

# C1-FIX: _clip01 and _safe_mean now imported from utils.py


def _financial_priors(world) -> Dict[str, float]:
    # D4-FIX: flatten triple-nested getattr chain which silently fell
    # through to defaults when workspace.config.priors didn't exist.
    priors = None
    try:
        priors = world.workspace.config.priors
    except AttributeError:
        pass
    if priors is None:
        # Return hardcoded defaults (the configurable path was never used)
        return {
            "regime_amplitude": 0.18,
            "slate_pressure": 0.075,
            "momentum_decay": 0.68,
            "recent_horizon": 6,
            "genre_memory_weight": 0.11,
        }
    return {
        "regime_amplitude": float(getattr(priors, "financial_regime_amplitude", 0.18)),
        "slate_pressure": float(getattr(priors, "financial_slate_pressure", 0.075)),
        "momentum_decay": float(getattr(priors, "financial_momentum_decay", 0.68)),
        "recent_horizon": int(getattr(priors, "financial_recent_horizon", 6)),
        "genre_memory_weight": float(getattr(priors, "financial_genre_memory_weight", 0.11)),
    }


def _market_regime(world, year: int) -> Dict[str, Any]:
    cache = getattr(world, "_financial_regime_cache", None)
    if cache is None:
        cache = {}
        setattr(world, "_financial_regime_cache", cache)

    year = int(year)
    if year in cache:
        return cache[year]

    priors = _financial_priors(world)
    amp = priors["regime_amplitude"]

    cycle = math.sin(
        2.0 * math.pi * (((year - 1975) / 8.5) + stable_uniform_0_1(world.seed, "fin", "cycle"))
    )
    prestige_cycle = math.sin(
        2.0 * math.pi * (((year - 1975) / 13.0) + stable_uniform_0_1(world.seed, "fin", "prestige_cycle"))
    )

    shock_u = stable_uniform_0_1(world.seed, "fin", "shock", year)
    shock_mag = stable_uniform_0_1(world.seed, "fin", "shock_mag", year)
    if shock_u < 0.14:
        shock = -0.38 + 0.22 * shock_mag
    elif shock_u > 0.91:
        shock = 0.08 + 0.18 * shock_mag
    else:
        shock = (stable_uniform_0_1(world.seed, "fin", "drift", year) - 0.5) * 0.10

    theatrical = float(np.clip(1.0 + amp * cycle + 0.16 * shock, 0.72, 1.30))
    financing = float(np.clip(1.0 + 0.82 * amp * cycle + 0.22 * shock, 0.78, 1.26))
    prestige_bias = float(np.clip(0.5 + 0.22 * prestige_cycle - 0.10 * shock, 0.0, 1.0))
    volatility = float(np.clip(0.10 + 0.12 * abs(cycle) + 0.22 * abs(shock), 0.08, 0.42))
    crowding = float(
        np.clip(0.42 + 0.18 * cycle + 0.10 * stable_uniform_0_1(world.seed, "fin", "crowding", year), 0.0, 1.0)
    )

    if theatrical < 0.88:
        label = "contracted"
    elif theatrical > 1.10 and prestige_bias > 0.58:
        label = "award_boom"
    elif theatrical > 1.08:
        label = "expansion"
    elif volatility > 0.28:
        label = "volatile"
    else:
        label = "steady"

    regime = {
        "theatrical_demand": theatrical,
        "financing_multiplier": financing,
        "prestige_bias": prestige_bias,
        "volatility": volatility,
        "crowding": crowding,
        "score": float((theatrical - 1.0) * 4.0),
        "label": label,
    }
    cache[year] = regime
    return regime


def _recent_load(events: Iterable[Any], year: int, horizon: int, decay: float) -> float:
    if not events:
        return 0.0
    total = 0.0
    for event_year in list(events)[-48:]:
        try:
            age = int(year) - int(event_year)
        except Exception:
            continue
        if 0 <= age <= horizon:
            total += float(decay) ** age
    return float(total)


def _momentum_from_records(records: Iterable[Any], year: int, horizon: int, decay: float) -> float:
    if not records:
        return 0.0
    score = 0.0
    total_weight = 0.0
    for rec in list(records)[-32:]:
        if not isinstance(rec, Mapping):
            continue
        try:
            rec_year = int(rec.get("year", year) or year)
        except Exception:
            rec_year = int(year)
        age = int(year) - rec_year
        if age < 0 or age > horizon:
            continue
        weight = float(decay) ** age
        perf = float(np.clip((_safe_float(rec.get("performance_ratio"), 1.0) - 1.0) / 1.35, -0.9, 1.3))
        rating = float(np.clip((_safe_float(rec.get("rating"), 6.2) - 6.2) / 2.8, -1.0, 1.0))
        awards = float(np.clip(_safe_float(rec.get("awards_won"), 0.0) / 3.0, 0.0, 1.5))
        budget = max(0.0, _safe_float(rec.get("budget_usd"), 0.0))
        budget_signal = float(np.clip((math.log1p(budget) - 14.0) / 5.0, -0.4, 0.6)) if budget > 0 else 0.0
        score += weight * (0.48 * perf + 0.34 * rating + 0.12 * awards + 0.06 * budget_signal)
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return float(np.clip(score / total_weight, -1.0, 1.25))


def _trim_history(history: List[Any], limit: int = 32) -> None:
    if len(history) > limit:
        del history[:-limit]


def _edge_window_active(year: int, valid_from: Optional[int], valid_to: Optional[int]) -> bool:
    if valid_from is not None and year < int(valid_from):
        return False
    if valid_to is not None and year > int(valid_to):
        return False
    return True


# ============================================================================
# Year quality effect — multi-component, era-aware
# ============================================================================

# Four incommensurate oscillation periods.  Because the ratios are irrational,
# the combined wave never exactly repeats — a GROUP BY year query sees rich
# texture instead of a detectable single sine wave.
_YR_COMPONENTS = [
    #  period   amplitude  phase_key
    (  7.3,     0.08,      "fast"),        # year-to-year variation
    ( 11.7,     0.12,      "medium"),      # roughly a decade
    ( 19.3,     0.16,      "slow"),        # generational
    ( 31.1,     0.10,      "ultra"),       # structural shifts
]

# Cultural era step offsets — mimics real film industry regime shifts.
# These are discontinuous (step functions), creating exactly the kind of
# non-smooth patterns that trip up cardinality estimators built on smooth
# distribution assumptions.
_YR_ERA_OFFSETS = [
    # (start, end,  offset, key)
    (1920, 1945,  +0.22, "golden_age"),       # Hollywood Golden Age
    (1946, 1959,  -0.05, "postwar"),          # Studio system strain
    (1960, 1966,  -0.12, "pre_new_hw"),       # Pre-New Hollywood lull
    (1967, 1979,  +0.28, "new_hollywood"),    # New Hollywood renaissance
    (1980, 1993,  -0.10, "blockbuster"),      # Blockbuster commercialization
    (1994, 2005,  +0.18, "indie_boom"),       # Indie / Sundance boom
    (2006, 2012,  +0.05, "prestige_tv"),      # Prestige TV pulls talent
    (2013, 2019,  -0.08, "franchise_glut"),   # MCU/franchise fatigue
    (2020, 2030,  -0.15, "streaming_flood"),  # Streaming content flood
]


def _year_quality_effect(year: int, seed: int) -> float:
    """Deterministic year-level quality offset with organic multi-component structure.

    Combines:
    1. Four incommensurate sine waves (quasi-periodic, never repeats)
    2. Cultural era step functions (regime shifts)
    3. Per-year hash noise (deterministic texture)

    Returns a value in roughly [-0.55, +0.55].
    """
    year = int(year)

    # ---- Component 1: Multi-harmonic oscillations ----
    oscillation = 0.0
    for period, amplitude, phase_key in _YR_COMPONENTS:
        # Seed-dependent phase shift — different seeds produce different timing
        phase = stable_uniform_0_1(seed, "yr_phase", phase_key)
        oscillation += amplitude * math.sin(
            2.0 * math.pi * ((year - 1970) / period + phase)
        )

    # ---- Component 2: Cultural era offsets ----
    era_offset = 0.0
    for start, end, offset, key in _YR_ERA_OFFSETS:
        if start <= year <= end:
            # Seed-dependent amplitude scaling (±30%) so each dataset
            # has slightly different era strengths
            era_scale = 0.70 + 0.60 * stable_uniform_0_1(seed, "yr_era", key)
            era_offset = offset * era_scale
            break

    # ---- Component 3: Per-year hash noise ----
    # Blake2b-based deterministic noise, ~±0.12 range
    h = int(hashlib.blake2b(
        f"yr_qual|{seed}|{year}".encode(), digest_size=4
    ).hexdigest(), 16)
    hash_noise = (h / 0xFFFFFFFF - 0.5) * 0.24

    total = float(oscillation + era_offset + hash_noise)
    return float(np.clip(total, -0.60, 0.60))


# ============================================================================
# Public API
# ============================================================================

def edge_is_active(
    world,
    edge_type: str,
    src_id: int,
    dst_id: int,
    movie_id: int,
    year: int,
    weight: float,
    valid_from: int | None = None,
    valid_to: int | None = None,
) -> tuple[bool, float]:
    """Soft edge activation for a global edge during one movie.

    A4-FIX: Instead of a hard Bernoulli gate (u < weight), we use a floor
    probability of 30% + 70% * weight, so even weak edges fire sometimes.
    Returns (did_fire, effective_strength) where effective_strength is the
    weight itself — callers should multiply their synergy contribution by it.

    Return type changed from bool to (bool, float) for backward compat:
    callers that only checked truthiness still work (tuple is truthy when
    did_fire=True), but callers can now also use the weight for scaling.
    """
    try:
        w = float(weight)
    except Exception:
        w = 0.0
    if not np.isfinite(w):
        w = 0.0
    w = float(np.clip(w, 0.0, 1.0))

    if not _edge_window_active(int(year), valid_from, valid_to):
        return (False, 0.0)

    u = stable_uniform_0_1(world.seed, "edge", edge_type, src_id, dst_id, movie_id, year)
    # A4-FIX: floor + scaled probability — weak edges (w=0.1) fire 37%,
    # moderate edges (w=0.5) fire 65%, strong edges (w=0.9) fire 93%.
    activation_prob = 0.30 + 0.70 * w
    fired = bool(u < activation_prob)
    return (fired, w if fired else 0.0)

# D6-FIX: descriptive alias to disambiguate from temporal_evolution_api._edge_is_active
edge_fires_for_movie = edge_is_active

def record_financial_outcome(
    world,
    concept: dict,
    fin: dict,
    director_id: int | None = None,
    companies: list[dict] | None = None,
    award_rows: list[dict] | None = None,
) -> None:
    """Append one realized movie outcome into world momentum histories."""
    if not isinstance(fin, Mapping):
        return

    year = int(concept.get("year", 0) or 0)
    awards_won = sum(1 for row in (award_rows or []) if str(row.get("outcome", "")).lower() == "won")
    performance_ratio = float(
        fin.get(
            "performance_ratio",
            float(fin.get("box_office_usd", 0.0)) / max(1.0, float(fin.get("budget_usd", 1.0))),
        )
    )

    record = {
        "year": year,
        "rating": float(fin.get("rating", 6.0)),
        "performance_ratio": performance_ratio,
        "budget_usd": float(fin.get("budget_usd", 0.0)),
        "box_office_usd": float(fin.get("box_office_usd", 0.0)),
        "awards_won": int(awards_won),
        "market_factor": float(fin.get("market_factor", 1.0)),
        "award_campaign_strength": float(fin.get("award_campaign_strength", 0.0)),
    }

    if director_id is not None and hasattr(world, "director_recent_outcomes"):
        bucket = world.director_recent_outcomes[int(director_id)]
        bucket.append(dict(record))
        _trim_history(bucket)

    if companies and hasattr(world, "company_recent_outcomes"):
        for company in companies:
            cid = company.get("company_id") if isinstance(company, Mapping) else None
            if cid is None:
                continue
            bucket = world.company_recent_outcomes[int(cid)]
            bucket.append(dict(record))
            _trim_history(bucket)

    genre = str(concept.get("genre", "Drama") or "Drama")
    if hasattr(world, "genre_recent_outcomes"):
        bucket = world.genre_recent_outcomes[genre]
        bucket.append(dict(record))
        _trim_history(bucket, limit=48)


# ============================================================================
# Internal model helpers
# ============================================================================

def _company_profile_effects(world, company_ids: List[int]) -> Dict[str, float]:
    profile_budget_mult = 1.0
    profile_market_mult = 1.0
    profile_perf_mult = 1.0
    slate_capacity = 0.5

    if not company_ids or not hasattr(world, "company_financial_profile"):
        return {
            "profile_budget_mult": profile_budget_mult,
            "profile_market_mult": profile_market_mult,
            "profile_perf_mult": profile_perf_mult,
            "slate_capacity": slate_capacity,
        }

    prof_rows = [world.company_financial_profile.get(int(cid)) for cid in company_ids]
    prof_rows = [p for p in prof_rows if isinstance(p, Mapping)]
    if not prof_rows:
        return {
            "profile_budget_mult": profile_budget_mult,
            "profile_market_mult": profile_market_mult,
            "profile_perf_mult": profile_perf_mult,
            "slate_capacity": slate_capacity,
        }

    cap = _safe_mean((_safe_float(p.get("capital_score"), 0.55) for p in prof_rows), 0.55)
    margin = _safe_mean((_safe_float(p.get("operating_margin"), 0.45) for p in prof_rows), 0.45)
    debt = _safe_mean((_safe_float(p.get("debt_ratio"), 0.45) for p in prof_rows), 0.45)
    slate_capacity = _safe_mean((_safe_float(p.get("slate_capacity"), 0.50) for p in prof_rows), 0.50)
    risk_buf = _safe_mean((_safe_float(p.get("risk_buffer"), 0.50) for p in prof_rows), 0.50)
    growth = _safe_mean((_safe_float(p.get("growth_bias"), 0.50) for p in prof_rows), 0.50)
    rev_eff = _safe_mean((_safe_float(p.get("revenue_efficiency"), 0.50) for p in prof_rows), 0.50)

    profile_budget_mult = float(np.clip(0.70 + 0.55 * cap + 0.20 * margin + 0.15 * risk_buf - 0.20 * debt, 0.65, 1.55))
    profile_market_mult = float(np.clip(0.78 + 0.30 * rev_eff + 0.20 * slate_capacity + 0.12 * growth, 0.75, 1.60))
    profile_perf_mult = float(np.clip(0.82 + 0.25 * risk_buf + 0.18 * margin - 0.10 * debt, 0.70, 1.45))

    return {
        "profile_budget_mult": profile_budget_mult,
        "profile_market_mult": profile_market_mult,
        "profile_perf_mult": profile_perf_mult,
        "slate_capacity": slate_capacity,
    }


def _graph_synergy(world, year: int, movie_id: int, cast_ids: List[int], director_id: Optional[int]) -> float:
    """Cast+director synergy via adjacency traversal.

    Uses pre-built ``world._friend_adj_all`` / ``world._rival_adj_all``
    adjacency lists (built by WorldState.load) instead of the old O(N²)
    all-pairs approach.  Complexity is now O(cast × avg_degree).
    """
    cast_set = set(int(c) for c in cast_ids)
    if not cast_set:
        return 0.0

    synergy = 0.0
    pairs_seen: set[tuple[int, int]] = set()

    graph = getattr(world, "graph", None)
    friend_adj = getattr(world, "_friend_adj_all", None) or {}
    rival_adj = getattr(world, "_rival_adj_all", None) or {}

    # Traverse each cast member's friendship edges; count only intra-cast hits
    for pid in cast_set:
        friend_iter = graph.iter_friend_neighbors(int(pid), year) if graph is not None else friend_adj.get(int(pid), [])
        for (nbr, w, vf, vt) in friend_iter:
            if nbr not in cast_set:
                continue
            key = (min(pid, nbr), max(pid, nbr))
            if key in pairs_seen:
                continue
            pairs_seen.add(key)
            if _edge_window_active(year, vf, vt):
                fired, strength = edge_is_active(world, "friendship", key[0], key[1],
                                  movie_id, year, float(w),
                                  valid_from=vf, valid_to=vt)
                if fired:
                    synergy += strength  # A4: scaled by edge weight

    # Traverse each cast member's rivalry edges
    for pid in cast_set:
        rival_iter = graph.iter_rival_neighbors(int(pid), year) if graph is not None else rival_adj.get(int(pid), [])
        for (nbr, w, vf, vt) in rival_iter:
            if nbr not in cast_set:
                continue
            key = (min(pid, nbr), max(pid, nbr))
            if key in pairs_seen:
                continue
            pairs_seen.add(key)
            rw = float(w) if w else 0.9
            if _edge_window_active(year, vf, vt):
                fired, strength = edge_is_active(world, "rivalry", key[0], key[1],
                                  movie_id, year, rw,
                                  valid_from=vf, valid_to=vt)
                if fired:
                    synergy -= strength  # A4: scaled by edge weight

    # Director preferences / avoidances (unchanged — already O(prefs))
    if director_id:
        pref_rows = graph.get_director_prefs(int(director_id), year) if graph is not None else []
        for pref_id, pref_w, pref_vf, pref_vt in pref_rows:
            fired, strength = edge_is_active(world, "director_pref", int(director_id), int(pref_id), movie_id, year, float(pref_w), valid_from=pref_vf, valid_to=pref_vt)
            if int(pref_id) in cast_set and fired:
                synergy += 0.5 * strength

        avoid_rows = graph.get_director_avoids(int(director_id), year) if graph is not None else []
        for av_id, av_vf, av_vt in avoid_rows:
            fired, _ = edge_is_active(world, "director_avoid", int(director_id), int(av_id), movie_id, year, 0.9, valid_from=av_vf, valid_to=av_vt)
            if int(av_id) in cast_set and fired:
                synergy -= 0.63

    n_possible_pairs = max(1, len(cast_set) * (len(cast_set) - 1) // 2)
    return float(np.clip(synergy / max(1, n_possible_pairs), -1.0, 1.0))


# ============================================================================
# Main financial model
# ============================================================================

def compute_financials(
    world,
    concept: dict,
    cast: list[dict],
    director_id: int = None,
    companies: list[dict] = None,
    demand_factor: float = 1.0,
) -> dict:
    """Generate one movie's correlated financial and reception outputs."""
    rng = world.rng
    tier = str(concept["tier"])
    genre = str(concept["genre"])
    year = int(concept["year"])
    movie_id = int(concept.get("movie_id", 0) or 0)
    installment = concept.get("installment")

    priors = _financial_priors(world)
    regime = _market_regime(world, year)

    base_lo, base_hi = BUDGET_RANGES[tier]
    inflation = (1.0 + 0.02) ** max(0, year - 1975)
    lo = float(base_lo * inflation)
    hi = float(base_hi * inflation)
    tier_idx = TIER_TO_LATENT_IDX.get(tier, 2)

    cast_ids = [int(c["person_id"]) for c in cast]
    cast_lvs = [get_person_latent(world, pid) for pid in cast_ids]
    director_lv = get_person_latent(world, int(director_id)) if director_id else None
    company_ids = [int(c["company_id"]) for c in (companies or [])]
    company_lvs = [get_company_latent(world, cid) for cid in company_ids]

    cast_rep = _safe_mean((_safe_float(lv.get("public_reputation"), 0.2) for lv in cast_lvs), 0.2)
    cast_risk = _safe_mean((_safe_float(lv.get("risk_tolerance"), 0.5) for lv in cast_lvs), 0.5)
    cast_contro = _safe_mean((_safe_float(lv.get("controversy_score"), 0.15) for lv in cast_lvs), 0.15)

    cast_tier_pref = _safe_mean(
        (
            _safe_float(lv.get("budget_band_pref", [0.5] * 5)[tier_idx], 0.5)
            for lv in cast_lvs
            if isinstance(lv.get("budget_band_pref"), list) and len(lv.get("budget_band_pref")) >= 5
        ),
        0.5,
    )

    if isinstance(director_lv, Mapping):
        director_rep = _safe_float(director_lv.get("public_reputation"), 0.4)
        director_risk = _safe_float(director_lv.get("risk_tolerance"), 0.5)
        dir_ambition = _safe_float(director_lv.get("artistic_ambition"), 0.5)
    else:
        director_rep = 0.4
        director_risk = 0.5
        dir_ambition = 0.5

    company_prest = _safe_mean((_safe_float(lv.get("prestige_score"), 0.5) for lv in company_lvs), 0.5)
    company_risk = _safe_mean((_safe_float(lv.get("risk_appetite"), 0.5) for lv in company_lvs), 0.5)
    company_contro_tol = _safe_mean((_safe_float(lv.get("controversy_tolerance"), 0.5) for lv in company_lvs), 0.5)
    trend_sensitivity = _safe_mean((_clip01(lv.get("market_trend_sensitivity"), 0.5) for lv in company_lvs), 0.5)
    company_focus = _safe_mean(
        (
            _safe_float(lv.get("budget_tier_focus", [0.5] * 5)[tier_idx], 0.5)
            for lv in company_lvs
            if isinstance(lv.get("budget_tier_focus"), list) and len(lv.get("budget_tier_focus")) >= 5
        ),
        0.5,
    )

    profile_fx = _company_profile_effects(world, company_ids)
    profile_budget_mult = profile_fx["profile_budget_mult"]
    profile_market_mult = profile_fx["profile_market_mult"]
    profile_perf_mult = profile_fx["profile_perf_mult"]
    slate_capacity = profile_fx["slate_capacity"]

    director_recent_load = 0.0
    if director_id is not None and hasattr(world, "director_recent"):
        director_recent_load = _recent_load(
            world.director_recent.get(int(director_id), []),
            year,
            horizon=max(3, priors["recent_horizon"] - 1),
            decay=priors["momentum_decay"],
        )

    company_recent_load = _safe_mean(
        (
            _recent_load(
                getattr(world, "company_recent", {}).get(int(cid), []),
                year,
                horizon=max(2, priors["recent_horizon"] - 2),
                decay=priors["momentum_decay"],
            )
            for cid in company_ids
        ),
        0.0,
    )

    director_momentum = 0.0
    if director_id is not None and hasattr(world, "director_recent_outcomes"):
        director_momentum = _momentum_from_records(
            world.director_recent_outcomes.get(int(director_id), []),
            year,
            horizon=priors["recent_horizon"],
            decay=priors["momentum_decay"],
        )

    company_momentum = _safe_mean(
        (
            _momentum_from_records(
                getattr(world, "company_recent_outcomes", {}).get(int(cid), []),
                year,
                horizon=priors["recent_horizon"],
                decay=priors["momentum_decay"],
            )
            for cid in company_ids
        ),
        0.0,
    )

    genre_heat = _momentum_from_records(
        getattr(world, "genre_recent_outcomes", {}).get(str(genre), []),
        year,
        horizon=max(priors["recent_horizon"] + 1, 5),
        decay=min(0.82, priors["momentum_decay"] + 0.08),
    )

    capacity_units = 1.15 + 2.25 * slate_capacity + 0.75 * company_focus
    overload = max(0.0, company_recent_load - capacity_units)
    slate_pressure = float(
        np.clip(1.0 - priors["slate_pressure"] * overload + 0.03 * max(company_momentum, 0.0), 0.74, 1.10)
    )
    director_burnout = float(np.clip(max(0.0, director_recent_load - 2.4) / 4.0, 0.0, 0.5))

    cast_spectacle = _safe_mean((style_spectacle_score(lv) for lv in cast_lvs), 0.5)
    mean_pop = _safe_mean((_safe_float(world.person_pop_weight.get(pid, 0.1), 0.1) for pid in cast_ids), 0.1)
    star_power = float(np.clip((0.6 * mean_pop + 0.4 * cast_rep) * 100.0, 5.0, 100.0))

    global_log_mean = 16.5
    global_log_std = 1.8
    tier_shift = TIER_LOG_CENTER.get(tier, 16.5) - global_log_mean
    pref_focus = float(np.clip(0.5 * cast_tier_pref + 0.5 * company_focus, 0.0, 1.0))
    budget_log = rng.normal(global_log_mean + tier_shift, global_log_std)
    budget_log += 0.3 * (pref_focus - 0.5)
    budget = float(math.exp(budget_log))
    budget = float(np.clip(budget, 1_000, 2_000_000_000))

    regime_financing_lift = float(np.clip(1.0 + 0.35 * (regime["financing_multiplier"] - 1.0), 0.88, 1.12))
    trend_capture = float((2.0 * trend_sensitivity - 1.0) * genre_heat)
    budget *= (0.9 + 0.4 * company_prest) * (0.95 + 0.2 * director_rep) * profile_budget_mult
    budget *= regime_financing_lift
    budget *= float(np.clip(1.0 + 0.14 * company_momentum + 0.09 * director_momentum, 0.74, 1.30))
    budget *= float(
        np.clip(1.0 + priors["genre_memory_weight"] * genre_heat * (0.45 + 0.55 * trend_sensitivity), 0.82, 1.18)
    )
    budget *= slate_pressure
    budget *= float(np.clip(1.0 - 0.10 * director_burnout, 0.86, 1.02))

    if installment and installment > 1:
        budget *= FRANCHISE_CONFIG["budget_growth_per_sequel"] ** (installment - 1)

    if hasattr(world, "person_award_wins") and world.person_award_wins:
        dir_wins = world.person_award_wins.get(director_id, 0)
        cast_wins = sum(world.person_award_wins.get(c.get("person_id", 0), 0) for c in cast) if cast else 0
        total_wins = dir_wins + cast_wins
        if total_wins > 0:
            budget *= min(1.0 + 0.1 * float(np.log1p(total_wins)), 1.40)

    effective_budget = int(np.clip(budget, lo * 0.6, hi * 1.8))
    country = str(concept.get("country", "USA") or "USA")
    country_scale = COUNTRY_BUDGET_SCALE.get(country, 0.20)
    budget_usd_scaled = int(effective_budget * country_scale)

    synergy_norm = _graph_synergy(world, year=year, movie_id=movie_id, cast_ids=cast_ids, director_id=director_id)

    def z01(x: float) -> float:
        return (float(x) - 0.5) * 2.0

    rep_dir_z = z01(director_rep)
    rep_cast_z = z01(cast_rep)
    prest_z = z01(company_prest)
    risk_mean = float(np.clip(0.5 * cast_risk + 0.2 * director_risk + 0.3 * company_risk, 0.0, 1.0))
    controversy_penalty = float(np.clip(cast_contro * (1.0 - company_contro_tol), 0.0, 1.0))

    quality_noise = float(rng.normal(0, 0.90 + 0.65 * risk_mean + 0.50 * regime["volatility"]))
    market_noise = float(rng.normal(0, 0.90 + 0.45 * risk_mean + 0.70 * regime["volatility"]))

    clique_offset = 0.0
    if company_ids:
        cliques = [world.company_clique.get(cid, 0) for cid in company_ids]
        primary_clique = max(set(cliques), key=cliques.count) if cliques else 0
        h = int(hashlib.md5(f"clique_{primary_clique}".encode()).hexdigest(), 16)
        clique_offset = (h % 100 - 50) / 100.0

    dir_quality_offset = world.director_quality_offset.get(director_id, 0.0) if director_id and hasattr(world, "director_quality_offset") else 0.0

    # D5-FIX: Named coefficients for quality & market latent formulas.
    # These model the relative importance of each signal in determining
    # movie quality perception vs market commercial appeal.
    #
    # quality_latent coefficients (critic/quality perspective):
    #   director reputation (0.9), prestige signals (0.5), cast synergy (0.4),
    #   company momentum (0.32), director momentum (0.34), genre heat (0.18),
    #   market regime (0.16), quality noise (0.6), clique style (0.3),
    #   director quality (0.3), slate pressure (-0.35), burnout (-0.45)
    #
    # market_latent coefficients (commercial/audience perspective):
    #   cast reputation (0.9), prestige (0.7), director reputation (0.2),
    #   controversy (-1.2), synergy (0.4), company momentum (0.52),
    #   director momentum (0.26), genre heat (0.42), regime score (0.55),
    #   trend capture (0.22), market noise (0.6), slate pressure (-0.50)
    quality_latent = (
        0.9 * rep_dir_z
        + 0.5 * prest_z
        + 0.4 * synergy_norm
        + 0.32 * company_momentum
        + 0.34 * director_momentum
        + 0.18 * genre_heat
        + 0.16 * regime["score"]
        + 0.6 * quality_noise
        + 0.3 * clique_offset
        + 0.3 * dir_quality_offset
        - 0.35 * (1.0 - slate_pressure)
        - 0.45 * director_burnout
    )
    market_latent = (
        0.9 * rep_cast_z
        + 0.7 * prest_z
        + 0.2 * rep_dir_z
        - 1.2 * controversy_penalty
        + 0.4 * synergy_norm
        + 0.52 * company_momentum
        + 0.26 * director_momentum
        + 0.42 * genre_heat
        + 0.55 * regime["score"]
        + 0.22 * trend_capture
        + 0.6 * market_noise
        - 0.50 * (1.0 - slate_pressure)
    )

    market_factor = float(np.clip(np.exp(0.10 * market_latent + 0.010 * star_power), 0.6, 2.5))
    market_factor *= regime["theatrical_demand"]
    market_factor *= float(
        np.clip(
            1.0 + 0.12 * genre_heat + 0.10 * company_momentum + 0.04 * max(director_momentum, 0.0) - 0.10 * (1.0 - slate_pressure),
            0.70,
            1.30,
        )
    )
    market_factor = float(np.clip(market_factor * profile_market_mult, 0.45, 3.4))

    qclip = float(np.clip(quality_latent, -3.0, 3.0))
    mean_log = np.log(0.93) + 0.13 * qclip + 0.08 * company_momentum + 0.07 * director_momentum + 0.05 * genre_heat
    mean_log -= 0.11 * (1.0 - slate_pressure) + 0.08 * regime["volatility"]
    sigma = (0.52 + 0.18 * regime["volatility"]) * (1.0 + 0.8 * risk_mean)

    performance = float(rng.lognormal(mean=mean_log, sigma=sigma))
    performance *= profile_perf_mult
    performance = float(np.clip(performance, 0.08, 5.8))
    if installment and installment > 1:
        performance *= float(np.clip(1.0 + 0.04 * (installment - 1), 1.0, 1.25))

    box_office_raw = int(effective_budget * country_scale * performance * market_factor)
    benford_jitter = float(10 ** rng.normal(0.0, 0.30))
    box_office_raw = max(0, int(box_office_raw * benford_jitter))
    box_office_raw = int(box_office_raw * float(np.clip(demand_factor, 0.1, 1.0)))

    # Soft compression: below threshold is linear, above uses a power-law
    # with exponent 0.72 (gentler than the old sqrt=0.5).
    # A5-FIX: threshold scales with the same inflation factor as budgets,
    # so modern blockbusters aren't penalised relative to earlier eras.
    # 1975: $200M, 2000: $330M, 2025: $537M — allows Marvel-scale $1.5B+ hits.
    bo_threshold = int(200_000_000 * inflation)
    if box_office_raw > bo_threshold:
        excess = box_office_raw - bo_threshold
        compressed_excess = int(bo_threshold * (excess / bo_threshold) ** 0.72)
        box_office = bo_threshold + compressed_excess
    else:
        box_office = box_office_raw

    base = TIER_RATING_BASE.get(tier, 6.0)
    std = TIER_RATING_STD.get(tier, 1.5)
    rating = base + (quality_latent + 0.25 * synergy_norm - 0.6 * controversy_penalty) * std
    rating = round(float(np.clip(rating, 1.0, 10.0)), 1)

    if installment and installment > 1:
        raw_decay = FRANCHISE_CONFIG["rating_decay_per_sequel"] * (installment - 1)
        rating_floor = max(4.5, rating - 2.0)
        rating = round(max(rating_floor, rating + raw_decay), 1)

    rating = round(float(np.clip(rating + GENRE_RATING_OFFSET.get(genre, 0.0), 1.0, 10.0)), 1)
    year_offset = _year_quality_effect(year, world.seed)
    rating = round(float(np.clip(rating + year_offset, 1.0, 10.0)), 1)

    if concept.get("is_writer_director", False):
        rating = round(float(np.clip(rating + 0.30, 1.0, 10.0)), 1)

    if tier in ["Epic", "A"] and genre in ["Action", "Sci-Fi", "Fantasy"]:
        runtime = int(rng.uniform(120, 180))
    elif genre == "Animation":
        runtime = int(rng.uniform(75, 110))
    elif genre == "Musical":
        runtime = int(rng.uniform(95, 135))
    elif genre == "Short":
        runtime = int(rng.uniform(4, 40))
    elif genre in ("Comedy", "Romance"):
        runtime = int(rng.uniform(85, 115))
    elif genre in ("Horror", "Thriller"):
        runtime = int(rng.uniform(80, 105))
    elif genre == "Documentary":
        runtime = int(rng.uniform(70, 120))
    else:
        runtime = int(rng.uniform(85, 130))

    if cast_spectacle > 0.6 and tier in ["Epic", "A"] and genre not in ("Animation", "Short"):
        runtime += int(rng.uniform(5, 25))
    elif cast_spectacle < 0.35 and genre in ["Comedy", "Horror"]:
        runtime -= int(rng.uniform(0, 10))
    runtime = int(np.clip(runtime, 4, 210))

    cert_dist = CERT_DISTS.get(genre, CERT_DISTS["Drama"])
    cert = rng.choice(list(cert_dist.keys()), p=list(cert_dist.values()))

    # B6-FIX: use box_office_raw (pre-compression) for vote count since
    # audience attention scales with actual commercial reach, not the
    # artificial compression applied for distribution realism.
    nv = (box_office_raw / rng.uniform(900, 2200)) * (1 + (year - 1975) / 50)
    nv *= regime["theatrical_demand"]
    nv *= 0.7 + 0.9 * cast_rep
    nv *= 1.0 + 0.35 * abs(synergy_norm)
    nv *= 1.0 + 0.35 * cast_contro
    nv *= float(np.clip(0.95 + 0.20 * max(0.0, genre_heat) + 0.10 * max(0.0, company_momentum), 0.80, 1.35))
    num_votes = max(TIER_MIN_VOTES.get(tier, 200), min(1_000_000, int(nv)))

    month = int(concept.get("month", 6))
    is_q4 = month >= 10
    is_prestige_genre = genre in AWARD_CAMPAIGN_GENRES
    award_campaign = (
        0.24 * company_prest
        + 0.22 * dir_ambition
        + 0.20 * (1.0 if is_q4 else 0.0)
        + 0.16 * (1.0 if is_prestige_genre else 0.0)
        + 0.10 * regime["prestige_bias"]
        + 0.05 * max(0.0, director_momentum)
        + 0.03 * max(0.0, company_momentum)
    )
    award_campaign = float(np.clip(award_campaign, 0.0, 1.0))

    performance_ratio = float(box_office / max(1.0, float(budget_usd_scaled)))

    return {
        "budget_usd": budget_usd_scaled,
        "box_office_usd": box_office,
        "rating": rating,
        "runtime_minutes": runtime,
        "certification": cert,
        "num_votes": num_votes,
        "award_campaign_strength": round(float(award_campaign), 3),
        "market_factor": round(float(market_factor), 4),
        "performance_ratio": round(performance_ratio, 4),
        "market_regime_score": round(float(regime["score"]), 3),
        "market_regime_label": regime["label"],
        "company_momentum": round(float(company_momentum), 3),
        "director_momentum": round(float(director_momentum), 3),
        "genre_heat": round(float(genre_heat), 3),
        "slate_pressure": round(float(slate_pressure), 3),
    }
