"""
V13 Pipeline -- utils.py
========================
Shared utility functions and constants for movie assembly.
Extracted from generate_movies.py for modular architecture.
"""
import numpy as np
import hashlib


# ═══════════════════════════════════════════════════════════════════════
# WEIGHT HELPERS
# ═══════════════════════════════════════════════════════════════════════

def normalize_weights(w):
    """Normalize weights to sum to 1, handling zeros and NaN."""
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)  # replace NaN/Inf with 0
    w = np.maximum(w, 0.0)  # clamp negatives
    s = w.sum()
    if s <= 0:
        return np.ones(len(w)) / len(w)
    return w / s


def _safe_float(x, default=0.0):
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v != v:  # NaN
        return float(default)
    return v


# C1-FIX: consolidated from duplicate definitions in financials.py and world_state.py
def _clip01(value, default=0.5):
    """Clip value to [0, 1] with NaN/error safety."""
    try:
        x = float(value)
    except Exception:
        x = float(default)
    if x != x:  # NaN
        x = float(default)
    return max(0.0, min(1.0, x))


def _safe_mean(values, default=0.0):
    """NaN-safe mean over an iterable of numeric values."""
    clean = []
    for value in values:
        try:
            v = float(value)
        except Exception:
            continue
        if v == v:  # not NaN
            clean.append(v)
    if not clean:
        return float(default)
    return float(sum(clean) / len(clean))


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISTIC HASHING
# ═══════════════════════════════════════════════════════════════════════

def stable_uniform_0_1(*parts) -> float:
    """Deterministic uniform[0,1) derived from stringified parts.

    Uses BLAKE2b (not Python hash, which is salted per-process).
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode('utf-8'))
        h.update(b'|')
    return int.from_bytes(h.digest(), 'big') / 2**64


# ═══════════════════════════════════════════════════════════════════════
# LATENT VARIABLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Latent tier order matches generate_latent_vars_api.py prompts:
# [Micro, Indie, Mid, A, Epic]
LATENT_TIER_ORDER = ["Micro", "Indie", "Mid", "A", "Epic"]
TIER_TO_LATENT_IDX = {t: i for i, t in enumerate(LATENT_TIER_ORDER)}

# D2-FIX: extend with common aliases that appear across modules
TIER_TO_LATENT_IDX.update({
    "A-List": 3, "Mid-Budget": 2, "Micro-Budget": 0,
    "Major": 3, "Global": 4,
})

# D2-FIX: Canonical movie-tier names. Maps any alias → canonical name.
_TIER_ALIASES = {
    "Epic": "Epic", "A": "A", "Mid": "Mid", "Indie": "Indie", "Micro": "Micro",
    "A-List": "A", "Mid-Budget": "Mid", "Micro-Budget": "Micro",
}


def normalize_movie_tier(tier: str) -> str:
    """Map any tier name variant to canonical movie production tier.

    D2-FIX: resolves the inconsistency where ``secondary_tables.py`` used
    ``A-List`` / ``Mid-Budget`` / ``Micro-Budget`` (company-tier names)
    for movie-tier lookups, causing silent fallthrough to defaults.
    """
    return _TIER_ALIASES.get(str(tier).strip(), "Mid")


def style_spectacle_score(lv: dict) -> float:
    """Map creative_style_vector[7] (intimate..spectacle) to [0,1]."""
    vec = lv.get("creative_style_vector")
    if isinstance(vec, list) and len(vec) >= 8:
        try:
            x = float(vec[7])
        except Exception:
            return 0.5
        if x != x:
            return 0.5
        return max(0.0, min(1.0, (x + 1.0) / 2.0))
    return 0.5


# ═══════════════════════════════════════════════════════════════════════
# TONE -> STYLE TAG MAPPING
# ═══════════════════════════════════════════════════════════════════════

# Heuristic mapping from sampled movie "tone" -> style tag hints.
# Used in casting to boost actors whose style tags fit the movie tone.
TONE_STYLE_HINTS = {
    "intense": ["intense", "explosive", "physical", "menacing"],
    "emotional": ["vulnerable", "understated", "theatrical", "lyrical"],
    "light": ["comedic", "magnetic", "improvisational"],
    "cerebral": ["cerebral", "minimalist"],
    "dark": ["menacing", "neo-noir", "slow-burn", "atmospheric"],
    "warm": ["vulnerable", "magnetic", "naturalistic", "lyrical"],
    "suspenseful": ["menacing", "neo-noir", "slow-burn", "atmospheric"],
    "epic": ["epic-scale", "visual-spectacle", "explosive", "physical"],
    "atmospheric": ["atmospheric", "neo-noir", "slow-burn"],
    "observational": ["naturalistic", "documentary-style", "handheld"],
    "gritty": ["intense", "neo-noir", "handheld", "naturalistic"],
    "whimsical": ["comedic", "theatrical", "lyrical", "surrealist"],
    "neutral": [],
}
