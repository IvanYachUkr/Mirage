"""
V13 Pipeline -- generate_edges_hybrid.py
=========================================
Step 2 of the Hybrid Graph Architecture:
Consumes latent variables (from generate_latent_vars_api.py) and
builds edges PROCEDURALLY using cosine similarity, genre overlap,
and feature-based rules.

ALL edges are:
- Symmetric by construction (no asymmetric friendships)
- Deterministic (seeded RNG)
- Deduplicated (impossible to have duplicates)
- Typed across 3 categories: person<->person, person<->company, company<->company

Also runs Louvain community detection on the friendship subgraph.

Usage:
    python test9/generate_edges_hybrid.py
"""
import json
import math
import os
import sys
import csv
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from contracts import (
    GENRES, CAREER_STAGES, STYLE_TAGS, EDGE_TYPES,
    RELATIONSHIP_TARGETS, load_json_batch,
)

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"
GRAPH_DIR = BASE_DIR / "graph"

SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# MATH UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def dot_overlap(a, b):
    """Weighted dot-product overlap (elements in [0,1])."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.sum(a * b))


def sigmoid(x, scale=1.0, bias=0.0):
    """Sigmoid function."""
    return 1.0 / (1.0 + math.exp(-scale * (x - bias)))


def genre_str_to_vector(genre_str, genre_list=GENRES):
    """Convert genre string(s) to a binary vector."""
    vec = [0.0] * len(genre_list)
    if isinstance(genre_str, list):
        genres = genre_str
    elif isinstance(genre_str, str):
        genres = [g.strip() for g in genre_str.replace("|", ";").split(";")]
    else:
        genres = []
    for g in genres:
        for i, canonical in enumerate(genre_list):
            if g.lower() == canonical.lower():
                vec[i] = 1.0
                break
    return vec


# ═══════════════════════════════════════════════════════════════════════
# EDGE GENERATION
# ═══════════════════════════════════════════════════════════════════════

# Stochastic per-person degree caps (seeded by person_id for reproducibility).
# Two-mechanism design:
#   1. Stochastic cap -> within-stage variance (not all legends at exactly 80)
#   2. Adaptive threshold -> cluster structure (similar people connect first)
#
# Rising N(2, 3) [0, 8]:  ~30% get cap=0 (true unknowns, never connect pre-assembly)
# Prime  N(10, 6) [1, 30]: some prime actors quite peripheral
# Veteran N(22,10) [3, 60]: established but varied
# Legend N(40,18) [5,100]: reclusive niche legends at 15, generalist stars at 80
# Retired N(14, 8) [1, 35]:
_CAP_PARAMS = {
    "rising":  (2,  3,  0,   8),
    "prime":   (10, 6,  1,  30),
    "veteran": (22, 10, 3,  60),
    "legend":  (40, 18, 5, 100),
    "retired": (14, 8,  1,  35),
}
_CAP_SEED_OFFSET = 9173  # offset to avoid colliding with main RNG seeds

# V15 scaling parameters for the two-phase P-P edge generator.
# CROSS_GENRE_K:    how many cross-genre candidates each person samples in Phase 2.
#                   Higher -> more inter-community bridges but slower.
# CROSS_GENRE_BUMP: extra threshold added on top of the adaptive base for cross-genre
#                   pairs. 0.10 means only very similar cross-genre pairs connect,
#                   preserving dense within-community / sparse between-community structure.
_CROSS_GENRE_K    = 150
_CROSS_GENRE_BUMP = 0.10

# P-C scaling: random supplement companies evaluated per person beyond genre match.
_PC_GENRE_SUPPLEMENT = 25

_UNDIRECTED_EDGE_TYPES = {
    "friendship",
    "rivalry",
    "co_production",
    "market_rival",
    "collaboration",
    "chemistry",
}


def _primary_genre(p) -> str:
    """Extract primary (first) genre from a person's genre_affinity field."""
    ga = p.get("genre_affinity", [])
    if isinstance(ga, str):
        parts = [g.strip() for g in ga.replace(";", ",").split(",") if g.strip()]
        return parts[0].lower() if parts else "__other__"
    if isinstance(ga, (list, tuple)) and ga:
        return str(ga[0]).strip().lower()
    return "__other__"


def _person_stochastic_cap(pid: int, stage: str) -> int:
    """Reproducible per-person degree cap sampled from stage distribution."""
    mu, sigma, lo, hi = _CAP_PARAMS.get(stage, (10, 5, 1, 30))
    rng_local = np.random.RandomState((int(pid) + _CAP_SEED_OFFSET) % (2**31))
    cap = int(round(rng_local.normal(mu, sigma)))
    return max(lo, min(hi, cap))


def generate_person_person_edges(persons, latent_map, rng, max_edges_per_person=None):
    """Generate person<->person edges from latent variables.

    Edge types: friendship, rivalry, mentorship, avoid.

    V15 SCALING FIX -- Two-phase approach replaces O(n²) loop.
    ─────────────────────────────────────────────────────────────
    Phase 1  (Within-genre, dense communities):
      For each primary-genre bucket compare ALL pairs within the bucket.
      Standard adaptive threshold (0.58 base). -> Tight intra-genre clusters.

    Phase 2  (Cross-genre, sparse bridges):
      Each person i samples _CROSS_GENRE_K=150 candidates j>i from OTHER
      genre buckets.  Threshold raised by _CROSS_GENRE_BUMP=0.10.
      -> Only very similar cross-genre people connect (legends who transcend
        genre) -> sparse inter-community edges, preserved Louvain communities.

    Complexity: O(Sigma Bk² + n.K) ~= 15-20M pairs vs 286M for plain O(n²).

    Pair guarantee: every (i,j) is evaluated AT MOST ONCE.
      Phase 1 covers all intra-genre pairs with i<j (within bucket).
      Phase 2 enforces j>i on cross-genre candidates -> zero overlap.

    All original mechanisms preserved unchanged:
      - _person_stochastic_cap degree caps
      - Adaptive threshold:  0.58 / (1 + 0.12 * deg^0.65)
      - Edge type classification (friendship/rivalry/mentorship/avoid)
      - Weight jitter formula
    """
    edges = []
    n = len(persons)
    edge_count = Counter()
    stage_ord = {"rising": 1, "prime": 2, "veteran": 3, "legend": 4, "retired": 5}

    person_by_id = {p["person_id"]: p for p in persons}
    person_cap   = {
        p["person_id"]: _person_stochastic_cap(
            p["person_id"], p.get("career_stage", "prime")
        )
        for p in persons
    }

    # Pre-compute per-person feature arrays (avoids repeated dict lookups)
    pid_list   = [p["person_id"] for p in persons]
    lv_list    = [latent_map.get(p["person_id"]) for p in persons]
    has_lv     = [lv is not None for lv in lv_list]
    csv_list   = [lv.get("creative_style_vector", [0]*8) if lv else [0]*8 for lv in lv_list]
    risk_list  = [lv.get("risk_tolerance",  0.5)         if lv else 0.5  for lv in lv_list]
    stage_list = [stage_ord.get(p.get("career_stage", "prime"), 2)        for p in persons]
    ctrv_list  = [lv.get("controversy_score", 0.0)       if lv else 0.0  for lv in lv_list]
    gnr_list   = [genre_str_to_vector(p.get("genre_affinity", []))        for p in persons]

    # Genre bucket construction (primary genre only -> no duplicate intra comparisons)
    pg_list = [_primary_genre(persons[i]) for i in range(n)]
    genre_buckets: dict = defaultdict(list)
    for idx, pg in enumerate(pg_list):
        genre_buckets[pg].append(idx)

    # ── Scoring closure ───────────────────────────────────────────────────────
    def _try_pair(i: int, j: int, threshold_extra: float = 0.0):
        pid_a, pid_b = pid_list[i], pid_list[j]
        if not has_lv[i] or not has_lv[j]:
            return
        if (edge_count[pid_a] >= person_cap[pid_a] or
                edge_count[pid_b] >= person_cap[pid_b]):
            return

        style_sim_v   = cosine_similarity(csv_list[i], csv_list[j])
        ovlp          = dot_overlap(gnr_list[i], gnr_list[j])
        union         = max(sum(gnr_list[i]) + sum(gnr_list[j]) - ovlp, 1.0)
        genre_jaccard = ovlp / union
        risk_match    = 1.0 - abs(risk_list[i] - risk_list[j])
        stage_diff    = abs(stage_list[i] - stage_list[j])
        controversy_a = ctrv_list[i]
        controversy_b = ctrv_list[j]

        raw_score = (
            0.40 * style_sim_v +
            0.25 * genre_jaccard +
            0.15 * risk_match +
            0.10 * (1.0 - stage_diff / 4.0) +
            0.10 * rng.rand()
        )
        p_edge = sigmoid(raw_score, scale=8.0, bias=0.55)

        # Adaptive threshold -- same formula as before + optional cross-genre bump
        thresh = max(
            0.58 / (1.0 + 0.12 * (edge_count[pid_a] ** 0.65)),
            0.58 / (1.0 + 0.12 * (edge_count[pid_b] ** 0.65)),
        ) + threshold_extra

        if p_edge < thresh:
            return

        src_id, dst_id = pid_a, pid_b
        if controversy_a > 0.7 or controversy_b > 0.7:
            if abs(controversy_a - controversy_b) > 0.4:
                etype, sign = "avoid", "-"
                weight = round(min(controversy_a, controversy_b) + 0.1, 2)
            else:
                etype, sign = "friendship", "+"
                base   = 0.25 + 0.55 * (p_edge + style_sim_v) / 2.0
                weight = round(base + rng.uniform(-0.08, 0.08), 2)
        elif style_sim_v > 0.65 and stage_diff >= 2:
            etype, sign = "mentorship", "+"
            base   = 0.28 + 0.54 * style_sim_v
            weight = round(base + rng.uniform(-0.06, 0.06), 2)
            if stage_list[i] < stage_list[j]:
                src_id, dst_id = pid_b, pid_a
        elif style_sim_v < 0.30 and genre_jaccard > 0.35 and stage_diff <= 1:
            etype, sign = "rivalry", "-"
            base   = 0.25 + 0.50 * genre_jaccard + 0.15 * (0.30 - style_sim_v)
            weight = round(base + rng.uniform(-0.06, 0.06), 2)
        elif style_sim_v > 0.70 or (p_edge > 0.35 and style_sim_v > 0.55):
            etype, sign = "friendship", "+"
            base   = 0.28 + 0.55 * (p_edge + style_sim_v) / 2.0
            weight = round(base + rng.uniform(-0.08, 0.08), 2)
        else:
            return

        weight = max(0.10, min(0.88, weight))
        xg = " [cross-genre]" if threshold_extra > 0 else ""
        edges.append({
            "src_id":      src_id,
            "dst_id":      dst_id,
            "src_name":    person_by_id.get(src_id, {}).get("name", "?"),
            "dst_name":    person_by_id.get(dst_id, {}).get("name", "?"),
            "edge_type":   etype,
            "sign":        sign,
            "weight":      weight,
            "source_kind": "latent_hybrid",
            "reason":      f"style={style_sim_v:.2f} jaccard={genre_jaccard:.2f} "
                           f"stage_gap={stage_diff}{xg}",
        })
        edge_count[pid_a] += 1
        edge_count[pid_b] += 1

    # ── Phase 1: Within-genre (dense community formation) ────────────────────
    # persons[] is pre-sorted legends->rising, so lower idx = higher career stage.
    # Iterating bucket indices ascending means legends fill caps first -> hubs form.
    phase1_pairs = 0
    for pg, idxs in genre_buckets.items():
        for a in range(len(idxs)):
            i = idxs[a]
            if not has_lv[i]:
                continue
            if edge_count[pid_list[i]] >= person_cap[pid_list[i]]:
                continue
            for b in range(a + 1, len(idxs)):
                j = idxs[b]
                _try_pair(i, j, 0.0)
                phase1_pairs += 1
    print(f"  Phase 1 (within-genre, {len(genre_buckets)} buckets): "
          f"{phase1_pairs:,} pairs -> {len(edges)} edges")

    # ── Phase 2: Cross-genre bridges (sparse inter-community connections) ─────
    # j > i ensures each cross-genre pair is evaluated exactly once.
    # Raised threshold keeps bridges rare -- only truly similar cross-genre
    # people (generalist legends) will connect here.
    p2_before = len(edges)
    for i in range(n):
        if not has_lv[i]:
            continue
        if edge_count[pid_list[i]] >= person_cap[pid_list[i]]:
            continue
        remaining = n - i - 1
        if remaining <= 0:
            continue
        n_sample = min(_CROSS_GENRE_K * 3, remaining)
        # rng.choice(remaining) gives offsets in [0, remaining-1]; shift to [i+1, n-1]
        raw = rng.choice(remaining, size=n_sample, replace=False) + (i + 1)
        cross_js = [j for j in raw if pg_list[j] != pg_list[i]][:_CROSS_GENRE_K]
        for j in cross_js:
            _try_pair(i, j, _CROSS_GENRE_BUMP)

    print(f"  Phase 2 (cross-genre, K={_CROSS_GENRE_K}): "
          f"{len(edges) - p2_before} additional edges")
    return edges





def generate_person_company_edges(persons, companies, person_latent, company_latent, rng):
    """Generate person<->company edges from latent variables.

    Edge types: employment, blacklist, brand_fit

    V15 SCALING FIX -- Genre pre-filtering:
      Pre-group companies by specialty_genres. Each person evaluates only
      genre-overlapping companies + _PC_GENRE_SUPPLEMENT random ones.
      Old: O(n x m) = 23,914 x 1,024 = 24.5M pairs
      New: O(n x avg_match) ~= 23,914 x ~120 = ~2.9M pairs (9x speedup)
    """
    edges = []

    p_latent_map = {lv["person_id"]: lv for lv in person_latent}
    c_latent_map = {lv["company_id"]: lv for lv in company_latent}
    person_by_id = {p["person_id"]: p for p in persons}
    company_by_id = {c["company_id"]: c for c in companies}

    # Pre-index companies by specialty_genres (lowercase)
    companies_by_genre: dict = defaultdict(list)
    for c in companies:
        for g in c.get("specialty_genres", []):
            companies_by_genre[g.lower()].append(c)

    n_companies = len(companies)

    for p in persons:
        pid = p["person_id"]
        plv = p_latent_map.get(pid)
        if plv is None:
            continue

        p_risk        = plv.get("risk_tolerance", 0.5)
        p_controversy = plv.get("controversy_score", 0.0)
        p_budget_pref = plv.get("budget_band_pref", [0.5]*5)
        p_genre       = genre_str_to_vector(p.get("genre_affinity", []))

        # Candidate companies: genre-overlapping + random supplement
        candidate_cids: set = set()
        person_genres = p.get("genre_affinity", [])
        if isinstance(person_genres, str):
            person_genres = [g.strip() for g in person_genres.split(",") if g.strip()]
        for g in person_genres:
            for c in companies_by_genre.get(g.lower(), []):
                candidate_cids.add(c["company_id"])
        # Random supplement -- ensures cross-genre brand_fit edges can exist
        supp_n = min(_PC_GENRE_SUPPLEMENT, n_companies)
        for idx in rng.choice(n_companies, size=supp_n, replace=False):
            candidate_cids.add(companies[idx]["company_id"])

        for c in companies:
            cid = c["company_id"]
            if cid == pid:              # self-loop guard
                continue
            if cid not in candidate_cids:
                continue
            clv = c_latent_map.get(cid)
            if clv is None:
                continue

            c_risk            = clv.get("risk_appetite", 0.5)
            c_controversy_tol = clv.get("controversy_tolerance", 0.5)
            c_budget_focus    = clv.get("budget_tier_focus", [0.2]*5)
            c_genre           = clv.get("genre_portfolio", [0.083]*12)

            risk_match     = 1.0 - abs(p_risk - c_risk)
            budget_overlap = dot_overlap(p_budget_pref, c_budget_focus)
            # Truncate to shorter dimension (person=28-dim, company=12-dim)
            ndim         = min(len(p_genre), len(c_genre))
            genre_overlap  = dot_overlap(p_genre[:ndim], c_genre[:ndim])

            raw_score = (
                0.30 * risk_match +
                0.30 * budget_overlap +
                0.30 * genre_overlap +
                0.10 * rng.rand()
            )

            # Blacklist: controversial person + intolerant company
            if p_controversy > 0.6 and c_controversy_tol < 0.3:
                if raw_score > 0.3:
                    edges.append({
                        "src_id":      cid,
                        "dst_id":      pid,
                        "src_name":    c.get("name", "?"),
                        "dst_name":    p.get("name", "?"),
                        "edge_type":   "blacklist",
                        "sign":        "-",
                        "weight":      round(0.5 + 0.3 * p_controversy, 2),
                        "source_kind": "latent_hybrid",
                        "reason":      f"controversy={p_controversy:.2f} vs tolerance={c_controversy_tol:.2f}",
                    })
                continue

            # Brand fit: high affinity (raised threshold: 0.55 -> 0.60 to avoid edge explosion)
            if raw_score > 0.60:
                edges.append({
                    "src_id":      cid,
                    "dst_id":      pid,
                    "src_name":    c.get("name", "?"),
                    "dst_name":    p.get("name", "?"),
                    "edge_type":   "brand_fit",
                    "sign":        "+",
                    "weight":      round(raw_score, 2),
                    "source_kind": "latent_hybrid",
                    "reason":      f"risk={risk_match:.2f} budget={budget_overlap:.2f} genre={genre_overlap:.2f}",
                })

    return edges



def generate_company_company_edges(companies, company_latent, rng):
    """Generate company<->company edges from latent variables.
    
    Edge types: co_production, market_rival
    """
    edges = []
    c_latent_map = {lv["company_id"]: lv for lv in company_latent}
    company_by_id = {c["company_id"]: c for c in companies}
    
    n = len(companies)
    for i in range(n):
        c_a = companies[i]
        cid_a = c_a["company_id"]
        clv_a = c_latent_map.get(cid_a)
        if clv_a is None:
            continue
        
        genre_a = clv_a.get("genre_portfolio", [0.083]*12)
        tier_a = clv_a.get("budget_tier_focus", [0.2]*5)
        
        for j in range(i + 1, n):
            c_b = companies[j]
            cid_b = c_b["company_id"]
            clv_b = c_latent_map.get(cid_b)
            if clv_b is None:
                continue
            
            genre_b = clv_b.get("genre_portfolio", [0.083]*12)
            tier_b = clv_b.get("budget_tier_focus", [0.2]*5)
            
            genre_overlap = dot_overlap(genre_a, genre_b)
            tier_overlap = dot_overlap(tier_a, tier_b)
            
            # Same genre + same tier = market rivals
            if genre_overlap > 0.5 and tier_overlap > 0.5:
                edges.append({
                    "src_id": cid_a,
                    "dst_id": cid_b,
                    "src_name": c_a.get("name", "?"),
                    "dst_name": c_b.get("name", "?"),
                    "edge_type": "market_rival",
                    "sign": "-",
                    "weight": round(genre_overlap * tier_overlap, 2),
                    "source_kind": "latent_hybrid",
                    "reason": f"genre_overlap={genre_overlap:.2f} tier_overlap={tier_overlap:.2f}",
                })
            # Complementary tiers + overlapping genres = co-production candidates
            elif genre_overlap > 0.3 and tier_overlap < 0.3:
                edges.append({
                    "src_id": cid_a,
                    "dst_id": cid_b,
                    "src_name": c_a.get("name", "?"),
                    "dst_name": c_b.get("name", "?"),
                    "edge_type": "co_production",
                    "sign": "+",
                    "weight": round(genre_overlap * 0.7, 2),
                    "source_kind": "latent_hybrid",
                    "reason": f"genre_shared={genre_overlap:.2f} complementary_tiers",
                })
    
    return edges


# ═══════════════════════════════════════════════════════════════════════
# SOCIAL GRAPH ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════

def _add_serendipitous_edges(pp_edges: list, persons: list, latent_map: dict, rng) -> list:
    """Phase 3: Random cross-community 'serendipitous' friendship edges.

    Real social networks have unexpected connections that procedural similarity
    scoring can never produce (same film school, shared agent, random festival
    encounter). These edges punch wormholes through community walls at random
    spots, making the graph topology genuinely unpredictable.

    Stage-gated injection probability:
      legend:  65% chance of 1-3 serendipitous edges
      veteran: 35% chance of 1-2
      prime:   15% chance of 1
      rising:   5% chance of 1

    Tagged with source_kind='serendipitous' so queries can distinguish them.
    Weight range 0.30-0.55 (moderate -- strong enough to matter, weak enough
    to not dominate over within-community ties).
    """
    _SEREN_PROB   = {"legend": 0.65, "veteran": 0.35, "prime": 0.15, "rising": 0.05, "retired": 0.05}
    _SEREN_N      = {"legend": 3,    "veteran": 2,    "prime": 1,    "rising": 1,    "retired": 1}
    _CROSS_GENRE_MIN_DIST = 1  # must be different primary genre
    pg_list = [_primary_genre(p) for p in persons]

    # Existing pair set for dedup
    existing = {(min(e["src_id"], e["dst_id"]), max(e["src_id"], e["dst_id"]))
                for e in pp_edges}

    name_map = {p["person_id"]: p.get("name", "?") for p in persons}
    n = len(persons)
    added = 0

    for i, p in enumerate(persons):
        pid   = p["person_id"]
        stage = p.get("career_stage", "prime")
        prob  = _SEREN_PROB.get(stage, 0.05)
        if rng.rand() >= prob:
            continue
        n_edges = rng.randint(1, _SEREN_N.get(stage, 1) + 1)

        my_genre = pg_list[i]
        # Sample candidates from different genre + randomly scattered nationality
        n_sample = min(n_edges * 20, n - 1)
        raw = rng.choice(n, size=n_sample, replace=False)
        cross = [j for j in raw if j != i and pg_list[j] != my_genre]

        for j in cross[:n_edges]:
            pid_b = persons[j]["person_id"]
            pair  = (min(pid, pid_b), max(pid, pid_b))
            if pair in existing:
                continue
            existing.add(pair)
            w = round(0.30 + 0.25 * rng.rand(), 2)
            pp_edges.append({
                "src_id":      min(pid, pid_b),
                "dst_id":      max(pid, pid_b),
                "src_name":    name_map.get(min(pid, pid_b), "?"),
                "dst_name":    name_map.get(max(pid, pid_b), "?"),
                "edge_type":   "friendship",
                "sign":        "+",
                "weight":      w,
                "source_kind": "serendipitous",
                "reason":      f"serendipitous cross-genre bridge",
            })
            added += 1

    print(f"  Serendipitous edges added: {added}")
    return pp_edges


def _add_triadic_closure(pp_edges: list, persons: list, rng) -> list:
    """Triadic closure: friend-of-friend becomes friend.

    If A--B and B--C are friendship edges, close the open triad A--C with a
    career-stage-gated probability. Creates tight friend GROUPS (high clustering
    coefficient) rather than just pairwise links.

    Stage-gated closure probability per open triad:
      legend:  25%  -- very tight A-list circles (realistic)
      veteran: 12%  -- established but more open
      prime:    5%  -- still building network
      rising:   0%  -- open network, not yet embedded

    Only applied to actors/directors (not all persons) to keep it focused.
    Respects existing per-person degree caps to avoid runaway degree growth.
    """
    _CLOSURE_PROB = {"legend": 0.25, "veteran": 0.12, "prime": 0.05, "rising": 0.0, "retired": 0.03}
    _CAP_EXTRA     = 8   # triadic closure can add up to 8 extra edges beyond cap

    stage_map = {p["person_id"]: p.get("career_stage", "prime") for p in persons}
    name_map  = {p["person_id"]: p.get("name", "?")             for p in persons}

    # Build friendship adjacency (only friendship sign=+)
    friends: dict = defaultdict(set)
    for e in pp_edges:
        if e.get("edge_type") == "friendship" and e.get("sign") == "+":
            a, b = e["src_id"], e["dst_id"]
            friends[a].add(b)
            friends[b].add(a)

    existing = {(min(e["src_id"], e["dst_id"]), max(e["src_id"], e["dst_id"]))
                for e in pp_edges}
    closure_count: dict = defaultdict(int)

    new_edges = []
    # Use sorted node list for determinism
    for pid_b in sorted(friends.keys()):
        flist = list(friends[pid_b])
        for ai in range(len(flist)):
            pid_a = flist[ai]
            for pid_c in flist[ai + 1:]:
                pair = (min(pid_a, pid_c), max(pid_a, pid_c))
                if pair in existing:
                    continue
                # Apply closure prob of the HIGHER-status node (the hub)
                highest_stage = max(
                    _CLOSURE_PROB.get(stage_map.get(pid_a, "prime"), 0.05),
                    _CLOSURE_PROB.get(stage_map.get(pid_b, "prime"), 0.05),
                    _CLOSURE_PROB.get(stage_map.get(pid_c, "prime"), 0.05),
                )
                if rng.rand() >= highest_stage:
                    continue
                if closure_count[pid_a] >= _CAP_EXTRA or closure_count[pid_c] >= _CAP_EXTRA:
                    continue
                existing.add(pair)
                closure_count[pid_a] += 1
                closure_count[pid_c] += 1
                w = round(0.38 + 0.32 * rng.rand(), 2)
                new_edges.append({
                    "src_id":      min(pid_a, pid_c),
                    "dst_id":      max(pid_a, pid_c),
                    "src_name":    name_map.get(min(pid_a, pid_c), "?"),
                    "dst_name":    name_map.get(max(pid_a, pid_c), "?"),
                    "edge_type":   "friendship",
                    "sign":        "+",
                    "weight":      w,
                    "source_kind": "triadic_closure",
                    "reason":      f"mutual friend {name_map.get(pid_b, '?')}",
                })

    print(f"  Triadic closure edges added: {len(new_edges)}")
    return pp_edges + new_edges


# ═══════════════════════════════════════════════════════════════════════
# COMMUNITY DETECTION (Louvain-inspired label propagation)
# ═══════════════════════════════════════════════════════════════════════

def detect_communities_louvain(edges, min_community_size=3, max_iterations=100, seed=42,
                               max_community_fraction=0.25):
    """Weighted label propagation on friendship subgraph.

    D10 fix: max_community_fraction=0.25 (default) -- any community that
    exceeds 25% of all nodes is forcibly split. This prevents the 80%%
    mega-community that appeared in V12 where label propagation converged
    to a single dominant community. No manual resolution parameter needed.

    Returns: {node_id: community_id}
    """
    rng = np.random.RandomState(seed)

    # Build adjacency from friendship edges only
    adj = defaultdict(list)  # node_id -> [(neighbor_id, weight)]
    nodes = set()

    for e in edges:
        if e.get("edge_type") == "friendship" and e.get("sign") == "+":
            src, dst = e["src_id"], e["dst_id"]
            w = e.get("weight", 0.5)
            adj[src].append((dst, w))
            adj[dst].append((src, w))
            nodes.add(src)
            nodes.add(dst)

    if not nodes:
        return {}

    # Initialize: each node is its own community
    labels = {n: n for n in nodes}
    node_list = list(nodes)

    for iteration in range(max_iterations):
        rng.shuffle(node_list)
        changed = 0

        for node in node_list:
            if not adj[node]:
                continue

            # Weighted vote from neighbors
            votes = defaultdict(float)
            for neighbor, weight in adj[node]:
                votes[labels[neighbor]] += weight

            if votes:
                best_label = max(votes, key=votes.get)
                if labels[node] != best_label:
                    labels[node] = best_label
                    changed += 1

        if changed == 0:
            break

    # D10: Forcibly split communities that exceed max_community_fraction
    n_nodes = len(nodes)
    max_size = max(int(max_community_fraction * n_nodes), min_community_size + 1)
    next_label = max(labels.values()) + 1

    for _split_pass in range(5):  # up to 5 rounds of splitting
        community_members = defaultdict(list)
        for node, label in labels.items():
            community_members[label].append(node)

        oversized = {k: v for k, v in community_members.items() if len(v) > max_size}
        if not oversized:
            break

        for label, members in oversized.items():
            # Sort by internal connectivity (least-connected first -> new community seed)
            internal_w = {}
            for node in members:
                w_sum = sum(w for nb, w in adj[node] if labels[nb] == label)
                internal_w[node] = w_sum
            sorted_members = sorted(members, key=lambda n: internal_w[n])

            # Move the bottom half to a new label
            split_size = len(members) // 2
            for node in sorted_members[:split_size]:
                labels[node] = next_label
            next_label += 1

    # Renumber communities and merge tiny ones
    community_members = defaultdict(list)
    for node, label in labels.items():
        community_members[label].append(node)

    # Find tiny communities and merge into nearest large one
    large_comms = {k: v for k, v in community_members.items() if len(v) >= min_community_size}
    tiny_comms = {k: v for k, v in community_members.items() if len(v) < min_community_size}

    for tiny_label, tiny_nodes in tiny_comms.items():
        # Find nearest large community by total edge weight
        best_large = None
        best_weight = -1
        for node in tiny_nodes:
            for neighbor, w in adj[node]:
                n_label = labels[neighbor]
                if n_label in large_comms and w > best_weight:
                    best_large = n_label
                    best_weight = w

        if best_large is not None:
            for node in tiny_nodes:
                labels[node] = best_large
        # else: keep as orphan community

    # Final renumbering
    unique_labels = sorted(set(labels.values()))
    label_map = {old: new + 1 for new, old in enumerate(unique_labels)}

    return {node: label_map[label] for node, label in labels.items()}



# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL DIMENSION
# ═══════════════════════════════════════════════════════════════════════

def add_temporal_edges(edges, persons):
    """Add valid_from/valid_to to edges based on career timelines.

    v10: Uses actual debut_year/retirement_year when available (set by
    _assign_career_timelines which is YEAR_RANGE-aware). Falls back to
    stage-based estimates for persons missing timeline data.
    """
    person_map = {}
    for p in persons:
        pid = p.get("person_id")
        if pid is None:
            continue
        # Prefer actual timeline data from person record
        debut = p.get("debut_year")
        retire = p.get("retirement_year")
        if debut is not None and retire is not None:
            person_map[pid] = (int(debut), int(retire))
        else:
            # Fallback: approximate from career stage
            stage = p.get("career_stage", "prime")
            if stage == "legend":
                person_map[pid] = (1975, 2035)
            elif stage == "veteran":
                person_map[pid] = (1985, 2030)
            elif stage == "prime":
                person_map[pid] = (1995, 2035)
            elif stage == "rising":
                # R10-FIX: valid_to=2100 gave 90-year careers. 2060 = 50-year estimate.
                person_map[pid] = (2010, 2060)

            else:  # retired
                person_map[pid] = (1975, 2015)

    for e in edges:
        src_range = person_map.get(e["src_id"], (1975, 2100))
        dst_range = person_map.get(e["dst_id"], (1975, 2100))
        # Overlap: relationship valid when both are active
        valid_from = max(src_range[0], dst_range[0])
        valid_to = min(src_range[1], dst_range[1])
        if valid_to < valid_from:
            valid_to = valid_from  # minimal validity
        e["valid_from"] = valid_from
        e["valid_to"] = valid_to

    return edges


# ═══════════════════════════════════════════════════════════════════════
# RELATIONSHIP CALIBRATION (v9+)
# ═══════════════════════════════════════════════════════════════════════

def _role_set(p) -> set:
    roles = p.get("roles", [])
    if roles is None:
        return set()
    if isinstance(roles, str):
        parts = [r.strip().lower() for r in roles.split(",") if r.strip()]
        return set(parts)
    if isinstance(roles, (list, tuple)):
        return set(str(r).strip().lower() for r in roles if str(r).strip())
    return set()


def _stage_index(p) -> int:
    st = p.get("career_stage", "prime")
    try:
        return CAREER_STAGES.index(st)
    except Exception:
        return CAREER_STAGES.index("prime")


def _genre_set(p) -> set:
    gs = p.get("main_genres")
    if gs is None:
        gs = p.get("genre_affinity")
    if gs is None:
        return set()
    if isinstance(gs, str):
        return set(x.strip().lower() for x in gs.split(",") if x.strip())
    if isinstance(gs, (list, tuple)):
        return set(str(x).strip().lower() for x in gs if str(x).strip())
    return set()


def _style_vec(latent_map: dict, pid: int) -> list:
    lv = latent_map.get(pid) or {}
    v = lv.get("creative_style_vector")
    if isinstance(v, list) and len(v) >= 8:
        return [float(x) for x in v[:8]]
    return [0.0] * 8


def _safe01(x, default=0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v != v:
        return float(default)
    return max(0.0, min(1.0, v))




def _coerce_year(raw, default: int) -> int:
    try:
        y = int(raw)
        if y <= 0:
            raise ValueError
        return y
    except Exception:
        return int(default)


def _build_person_ranges(persons: list[dict]) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    for p in persons:
        try:
            pid = int(p.get("person_id"))
        except Exception:
            continue
        debut = _coerce_year(p.get("debut_year"), 1975)
        retire = _coerce_year(p.get("retirement_year"), 2060)
        if retire < debut:
            retire = debut
        ranges[pid] = (debut, retire)
    return ranges


def _build_company_ranges(companies: list[dict]) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    for c in companies:
        try:
            cid = int(c.get("company_id"))
        except Exception:
            continue
        founded = _coerce_year(c.get("founded_year"), 1970)
        defunct = _coerce_year(c.get("defunct_year"), 2065)
        if defunct < founded:
            defunct = founded
        ranges[cid] = (founded, defunct)
    return ranges


def _apply_validity_window(edge: dict, src_range: tuple[int, int], dst_range: tuple[int, int]):
    vf = max(int(src_range[0]), int(dst_range[0]))
    vt = min(int(src_range[1]), int(dst_range[1]))
    if vt < vf:
        vt = vf
    edge["valid_from"] = int(vf)
    edge["valid_to"] = int(vt)


def add_temporal_to_non_person_edges(
    pc_edges: list[dict],
    cc_edges: list[dict],
    persons: list[dict],
    companies: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Ensure person-company/company-company edges also have non-null validity windows."""
    person_ranges = _build_person_ranges(persons)
    company_ranges = _build_company_ranges(companies)

    default_person = (1975, 2060)
    default_company = (1970, 2065)

    for e in pc_edges:
        try:
            src = int(e.get("src_id"))
            dst = int(e.get("dst_id"))
        except Exception:
            continue
        src_person = person_ranges.get(src)
        dst_person = person_ranges.get(dst)
        src_company = company_ranges.get(src)
        dst_company = company_ranges.get(dst)

        person_range = src_person or dst_person or default_person
        company_range = src_company or dst_company or default_company
        _apply_validity_window(e, person_range, company_range)

    for e in cc_edges:
        try:
            src = int(e.get("src_id"))
            dst = int(e.get("dst_id"))
        except Exception:
            continue
        src_range = company_ranges.get(src, default_company)
        dst_range = company_ranges.get(dst, default_company)
        _apply_validity_window(e, src_range, dst_range)

    return pc_edges, cc_edges


def dedupe_edges(edges: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate edges deterministically, preserving strongest weight per key."""
    best: dict[tuple[str, str, int, int], dict] = {}
    replaced = 0

    for raw in edges:
        e = dict(raw)
        try:
            src = int(e.get("src_id"))
            dst = int(e.get("dst_id"))
        except Exception:
            continue
        if src == dst:
            continue

        et = str(e.get("edge_type", "")).strip().lower()
        sign = str(e.get("sign", "+")).strip()

        if et in _UNDIRECTED_EDGE_TYPES and src > dst:
            src, dst = dst, src
            sn = e.get("src_name")
            dn = e.get("dst_name")
            e["src_name"] = dn
            e["dst_name"] = sn

        e["src_id"] = src
        e["dst_id"] = dst

        try:
            w = float(e.get("weight", 0.0))
        except Exception:
            w = 0.0
        e["weight"] = round(max(0.0, min(1.0, w)), 4)

        vf = _coerce_year(e.get("valid_from"), 1975)
        vt = _coerce_year(e.get("valid_to"), 2065)
        if vt < vf:
            vt = vf
        e["valid_from"] = vf
        e["valid_to"] = vt

        key = (et, sign, src, dst)
        old = best.get(key)
        if old is None:
            best[key] = e
            continue

        if float(e.get("weight", 0.0)) > float(old.get("weight", 0.0)):
            best[key] = e
            replaced += 1
            continue

        old_vf = _coerce_year(old.get("valid_from"), 1975)
        old_vt = _coerce_year(old.get("valid_to"), 2065)
        overlap_from = max(old_vf, vf)
        overlap_to = min(old_vt, vt)
        if overlap_to < overlap_from:
            overlap_to = overlap_from
        old["valid_from"] = overlap_from
        old["valid_to"] = overlap_to
        replaced += 1

    return list(best.values()), replaced


def complete_community_assignments(communities: dict[int, int], persons: list[dict]) -> tuple[dict[int, int], int]:
    """Assign a deterministic community to every person, including isolates."""
    all_pids = []
    for p in persons:
        try:
            all_pids.append(int(p.get("person_id")))
        except Exception:
            continue

    out = {int(pid): int(comm) for pid, comm in communities.items()}
    existing = sorted(set(out.values()))
    added = 0

    if not existing:
        n = max(8, min(64, int(math.sqrt(max(1, len(all_pids))))))
        for pid in all_pids:
            bucket = int(hashlib.md5(f"community|{pid}".encode("utf-8")).hexdigest(), 16) % n
            out[pid] = bucket + 1
            added += 1
        return out, added

    for pid in all_pids:
        if pid in out:
            continue
        h = int(hashlib.md5(f"community|{pid}".encode("utf-8")).hexdigest(), 16)
        out[pid] = existing[h % len(existing)]
        added += 1

    return out, added

def calibrate_relationship_targets(pp_edges: list, persons: list, latent_map: dict, rng,
                                   targets: dict = RELATIONSHIP_TARGETS,
                                   seed: int = SEED) -> list:
    """Post-process person<->person edges to match coarse relationship targets.

    Targets come from contracts.RELATIONSHIP_TARGETS.

    Implemented (approx.):
      - best_friend_rate: fraction of actors with >=1 friendship (actor<->actor)
      - rival_rate: fraction of actors with >=1 rivalry (actor<->actor)
      - bf_same_community_rate: fraction of actor best-friend links within community
      - director_preferred_actors: ~N outgoing mentorship edges per director (-> actors)
      - director_avoided_actors: ~N outgoing avoid edges per director (-> actors)

    Notes:
      - Deterministic by using the seeded RNG.
      - Calibration operates on the *static* graph; activation remains probabilistic
        at movie-assembly time (generate_movies.py).
    """

    if not persons:
        return pp_edges

    # ─── Precompute metadata ─────────────────────────────────────────
    name_map = {int(p["person_id"]): p.get("name", "") for p in persons}
    roles_map = {int(p["person_id"]): _role_set(p) for p in persons}
    stage_map = {int(p["person_id"]): _stage_index(p) for p in persons}
    genres_map = {int(p["person_id"]): _genre_set(p) for p in persons}
    style_map = {int(p["person_id"]): _style_vec(latent_map, int(p["person_id"])) for p in persons}

    actor_ids = [pid for pid, rs in roles_map.items() if "actor" in rs]
    director_ids = [pid for pid, rs in roles_map.items() if "director" in rs]

    if not actor_ids:
        return pp_edges

    # Edge index for updates (avoid duplicates)
    def _edge_key(et: str, sign: str, src: int, dst: int):
        # Normalize ALL types to min/max so (A->B) and (B->A) are the same key.
        # Mentorship direction is stored in src/dst, but for dedup purposes
        # we only want one edge per (pair, type) regardless of which way it points.
        a, b = (min(src, dst), max(src, dst))
        return (et, sign, a, b)

    edge_pos = {}
    for i, e in enumerate(pp_edges):
        try:
            k = _edge_key(e.get("edge_type"), e.get("sign"), int(e.get("src_id")), int(e.get("dst_id")))
        except Exception:
            continue
        if k not in edge_pos:
            edge_pos[k] = i
        else:
            old_i = edge_pos[k]
            if float(e.get("weight", 0)) > float(pp_edges[old_i].get("weight", 0)):
                edge_pos[k] = i

    def _upsert_edge(et: str, sign: str, src: int, dst: int, weight: float, reason: str):
        w = float(max(0.1, min(0.95, weight)))
        k = _edge_key(et, sign, src, dst)
        if k in edge_pos:
            e = pp_edges[edge_pos[k]]
            if w > float(e.get("weight", 0.0)):
                e["weight"] = w
                e["reason"] = reason
                e["source_kind"] = "latent_hybrid"
            return

        if et in {"friendship", "rivalry"}:
            src2, dst2 = (min(src, dst), max(src, dst))
        else:
            src2, dst2 = int(src), int(dst)

        new_e = {
            "src_id": int(src2),
            "dst_id": int(dst2),
            "src_name": name_map.get(int(src2), ""),
            "dst_name": name_map.get(int(dst2), ""),
            "edge_type": et,
            "sign": sign,
            "weight": w,
            "source_kind": "latent_hybrid",
            "reason": reason,
        }
        pp_edges.append(new_e)
        edge_pos[k] = len(pp_edges) - 1

    # Build adjacency for actor<->actor friendships/rivalries
    friends = defaultdict(dict)
    rivals_adj = defaultdict(dict)

    for e in pp_edges:
        et = e.get("edge_type")
        if et not in {"friendship", "rivalry"}:
            continue
        a = int(e.get("src_id"))
        b = int(e.get("dst_id"))
        if a not in actor_ids or b not in actor_ids:
            continue
        w = float(e.get("weight", 0.0))
        if et == "friendship" and e.get("sign") == "+":
            if w > friends[a].get(b, 0.0):
                friends[a][b] = w
                friends[b][a] = w
        elif et == "rivalry":
            if w > rivals_adj[a].get(b, 0.0):
                rivals_adj[a][b] = w
                rivals_adj[b][a] = w

    # Similarity helpers
    def jaccard(sa: set, sb: set) -> float:
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        uni = len(sa | sb)
        return inter / uni if uni else 0.0

    def style_sim(a: int, b: int) -> float:
        return 0.5 * (cosine_similarity(style_map[a], style_map[b]) + 1.0)

    def stage_sim(a: int, b: int) -> float:
        da = abs(stage_map[a] - stage_map[b])
        denom = max(1, len(CAREER_STAGES) - 1)
        return 1.0 - (da / denom)

    def friendship_score(a: int, b: int) -> float:
        return 0.55 * style_sim(a, b) + 0.35 * jaccard(genres_map[a], genres_map[b]) + 0.10 * stage_sim(a, b)

    def rivalry_score(a: int, b: int) -> float:
        return 0.60 * jaccard(genres_map[a], genres_map[b]) + 0.20 * (1.0 - style_sim(a, b)) + 0.20 * stage_sim(a, b)

    # ─── Calibrate: best_friend_rate ─────────────────────────────────
    target_bf = float(targets.get("best_friend_rate", 0.0))
    want_bf = int(round(target_bf * len(actor_ids)))
    have_bf = sum(1 for pid in actor_ids if friends.get(pid))

    if have_bf < want_bf:
        missing = [pid for pid in actor_ids if not friends.get(pid)]
        rng.shuffle(missing)
        added = 0
        for pid in missing:
            if have_bf + added >= want_bf:
                break

            best = None
            best_sc = -1.0
            for cand in actor_ids:
                if cand == pid:
                    continue
                if cand in friends.get(pid, {}):
                    continue
                if cand in rivals_adj.get(pid, {}):
                    continue
                sc = friendship_score(pid, cand)
                # v11 P1: Preferential attachment -- stars attract friends
                cand_degree = len(friends.get(cand, {}))
                sc *= (1.0 + math.log(1.0 + cand_degree))
                if sc > best_sc:
                    best_sc = sc
                    best = cand

            if best is None:
                continue

            # WEIGHT FIX: lower calibration ceiling to match main generator
            w = 0.35 + 0.48 * best_sc + 0.05 * rng.rand()  # max ~0.88
            _upsert_edge(
                "friendship", "+", pid, best, w,
                reason=f"CAL(best_friend_rate): friend_score={best_sc:.3f}",
            )
            friends[pid][best] = w
            friends[best][pid] = w
            added += 1

        print(f"[Calibrate] Added {added} friendships to reach best_friend_rate")

    # ─── Calibrate: rival_rate ───────────────────────────────────────
    target_r = float(targets.get("rival_rate", 0.0))
    want_r = int(round(target_r * len(actor_ids)))
    have_r = sum(1 for pid in actor_ids if rivals_adj.get(pid))

    if have_r < want_r:
        missing = [pid for pid in actor_ids if not rivals_adj.get(pid)]
        rng.shuffle(missing)
        added = 0
        for pid in missing:
            if have_r + added >= want_r:
                break

            best = None
            best_sc = -1.0
            for cand in actor_ids:
                if cand == pid:
                    continue
                if cand in rivals_adj.get(pid, {}):
                    continue
                if cand in friends.get(pid, {}):
                    continue
                sc = rivalry_score(pid, cand)
                if sc > best_sc:
                    best_sc = sc
                    best = cand

            if best is None:
                continue

            w = 0.45 + 0.45 * best_sc + 0.05 * rng.rand()
            _upsert_edge(
                "rivalry", "-", pid, best, w,
                reason=f"CAL(rival_rate): rival_score={best_sc:.3f}",
            )
            rivals_adj[pid][best] = w
            rivals_adj[best][pid] = w
            added += 1

        print(f"[Calibrate] Added {added} rivalries to reach rival_rate")

    # ─── Calibrate: director preferred/avoided actors ─────────────────
    # V15 FIX: pre-index actors by genre so each director only scores
    # genre-overlapping actors + DIRECTOR_ACTOR_SUPPLEMENT random ones.
    # Old: O(d x a) = 2,833 x 14,473 = 41M pairs
    # New: O(d x avg_match) ~= 2,833 x ~630 = ~1.8M pairs (23x speedup)
    DIRECTOR_ACTOR_SUPPLEMENT = 200

    actor_genre_index: dict = defaultdict(list)
    actor_ids_set = set(actor_ids)
    for aid in actor_ids:
        for g in genres_map.get(aid, set()):
            actor_genre_index[g].append(aid)
    actor_ids_arr = np.array(actor_ids)

    def _director_candidate_actors(did: int) -> list:
        """Genre-overlapping actors for director did + random supplement."""
        cands: set = set()
        for g in genres_map.get(did, set()):
            cands.update(actor_genre_index.get(g, []))
        # Random supplement -- prevents complete genre blindness
        n_supp = min(DIRECTOR_ACTOR_SUPPLEMENT, len(actor_ids))
        for aid in rng.choice(actor_ids_arr, size=n_supp, replace=False).tolist():
            cands.add(int(aid))
        cands.discard(did)
        return list(cands)

    def current_outgoing(et: str, src: int) -> set:
        out = set()
        for e in pp_edges:
            if e.get("edge_type") != et:
                continue
            if int(e.get("src_id")) != int(src):
                continue
            dst = int(e.get("dst_id"))
            if dst in actor_ids_set:
                out.add(dst)
        return out

    pref_target_base = int(targets.get("director_preferred_actors", 0))
    avoid_target_base = int(targets.get("director_avoided_actors", 0))

    actor_contro = {pid: _safe01(latent_map.get(pid, {}).get("controversy_score"), 0.15) for pid in actor_ids}

    for did in director_ids:
        senior      = stage_map.get(did, CAREER_STAGES.index("prime"))
        senior_norm = senior / max(1, len(CAREER_STAGES) - 1)
        pref_target  = max(0, int(round(pref_target_base  * (0.8 + 0.4 * senior_norm))))
        avoid_target = max(0, int(round(avoid_target_base * (0.8 + 0.4 * senior_norm))))

        prefs  = current_outgoing("mentorship", did)
        avoids = current_outgoing("avoid",      did)

        # Genre-indexed candidate actors for this director
        cand_actors = _director_candidate_actors(did)

        need = max(0, pref_target - len(prefs))
        if need > 0:
            scored = []
            for aid in cand_actors:
                if aid in prefs or aid in avoids:
                    continue
                sc = (0.65 * style_sim(did, aid)
                      + 0.25 * jaccard(genres_map[did], genres_map[aid])
                      + 0.10 * stage_sim(did, aid))
                scored.append((sc, aid))
            scored.sort(reverse=True)
            for sc, aid in scored[:need]:
                w = 0.55 + 0.40 * sc + 0.05 * rng.rand()
                _upsert_edge("mentorship", "+", did, aid, w,
                             reason=f"CAL(director_pref): score={sc:.3f}")
                prefs.add(aid)

        need = max(0, avoid_target - len(avoids))
        if need > 0:
            d_rep  = _safe01(latent_map.get(did, {}).get("public_reputation"), 0.4)
            scored = []
            for aid in cand_actors:
                if aid in avoids or aid in prefs:
                    continue
                sc = (0.50 * (1.0 - style_sim(did, aid))
                      + 0.20 * (1.0 - jaccard(genres_map[did], genres_map[aid]))
                      + 0.30 * (actor_contro[aid] * d_rep))
                scored.append((sc, aid))
            scored.sort(reverse=True)
            for sc, aid in scored[:need]:
                w = 0.55 + 0.35 * sc + 0.05 * rng.rand()
                _upsert_edge("avoid", "-", did, aid, w,
                             reason=f"CAL(director_avoid): score={sc:.3f}")
                avoids.add(aid)


    # ─── Calibrate: bf_same_community_rate ───────────────────────────
    target_same = float(targets.get("bf_same_community_rate", 0.0))

    def best_friend_map(communities: dict) -> tuple:
        bf = {}
        total = 0
        within = 0
        for pid in actor_ids:
            neigh = [(nbr, w) for nbr, w in friends.get(pid, {}).items() if nbr in actor_ids]
            if not neigh:
                continue
            nbr, w = max(neigh, key=lambda t: t[1])
            bf[pid] = (nbr, w)
            total += 1
            if communities.get(pid) is not None and communities.get(pid) == communities.get(nbr):
                within += 1
        ratio = (within / total) if total else 1.0
        return bf, ratio

    if target_same > 0 and len(actor_ids) >= 5:
        for it in range(3):
            communities = detect_communities_louvain(pp_edges, seed=seed)
            bf_map, ratio = best_friend_map(communities)
            if ratio >= target_same or not bf_map:
                break

            comm_members = defaultdict(list)
            for pid in actor_ids:
                c = communities.get(pid)
                if c is not None:
                    comm_members[c].append(pid)

            offenders = [pid for pid, (nbr, _) in bf_map.items()
                         if communities.get(pid) is not None and communities.get(nbr) != communities.get(pid)]
            rng.shuffle(offenders)

            target_fix = int(round((target_same - ratio) * len(bf_map)))
            fixed = 0

            for pid in offenders:
                if fixed >= target_fix:
                    break
                c = communities.get(pid)
                pool = [x for x in comm_members.get(c, []) if x != pid]
                if not pool:
                    continue

                best = None
                best_sc = -1.0
                for cand in pool:
                    if cand in rivals_adj.get(pid, {}):
                        continue
                    sc = friendship_score(pid, cand)
                    if sc > best_sc:
                        best_sc = sc
                        best = cand

                if best is None:
                    continue

                cur_best_w = bf_map[pid][1]
                w = min(0.95, max(cur_best_w + 0.05, 0.85))
                _upsert_edge(
                    "friendship", "+", pid, best, w,
                    reason=f"CAL(bf_same_comm): boost_to={w:.2f}",
                )
                friends[pid][best] = w
                friends[best][pid] = w
                fixed += 1

            if fixed == 0:
                break

        communities = detect_communities_louvain(pp_edges, seed=seed)
        _, ratio = best_friend_map(communities)
        print(f"[Calibrate] bf_same_community_rate={ratio:.3f} (target {target_same})")

    return pp_edges


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.RandomState(SEED)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    
    # Load data
    # V15: ALL persons are fully LLM-generated -- no is_extra split needed.
    # The old core/extras distinction (v9-v14) is dead; every person in
    # persons.json has a full bio, latent vars, and graph representation.
    persons = load_json_batch(ENTITY_DIR / "persons.json")
    graph_persons = persons
    # Keep stable IDs; only backfill if missing (shouldn't happen in v15)
    for i, p in enumerate(graph_persons):
        if "person_id" not in p:
            p["person_id"] = i + 1

    companies = load_json_batch(ENTITY_DIR / "companies.json")
    for i, c in enumerate(companies):
        if "company_id" not in c:
            c["company_id"] = i + 1

    print(f"Loaded {len(graph_persons)} persons (all LLM-generated), {len(companies)} companies")
    
    # Load latent variables
    person_latent_path = ENTITY_DIR / "persons_latent.json"
    company_latent_path = ENTITY_DIR / "companies_latent.json"
    
    if not person_latent_path.exists():
        print("ERROR: persons_latent.json not found. Run generate_latent_vars_api.py first.")
        return
    
    person_latent = json.loads(person_latent_path.read_text(encoding="utf-8"))
    p_latent_map = {lv["person_id"]: lv for lv in person_latent}
    print(f"Loaded {len(person_latent)} person latent vars")
    
    company_latent = []
    if company_latent_path.exists():
        company_latent = json.loads(company_latent_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(company_latent)} company latent vars")
    else:
        print("No company latent vars found -- skipping company edges")
    
    # ═══ Generate person<->person edges ═════════════════════════════════
    # Sort persons: legends/veterans first so they bootstrap high degree before
    # rising actors are encountered. With adaptive threshold, legend vs legend
    # connects freely (thresh~=0.21). Rising vs rising (processed last, still
    # deg=0) faces thresh=0.58 and average pairs (p_edge~=0.41) don't connect.
    _STAGE_PRIORITY = {"legend": 0, "veteran": 1, "prime": 2, "retired": 3, "rising": 4}
    graph_persons.sort(key=lambda p: _STAGE_PRIORITY.get(p.get("career_stage", "prime"), 2))
    stage_counts = Counter(p.get("career_stage", "prime") for p in graph_persons)
    print(f"  Stage order: {dict(stage_counts)}")

    print("\nGenerating person<->person edges...")
    pp_edges = generate_person_person_edges(graph_persons, p_latent_map, rng)
    pp_types = Counter(e["edge_type"] for e in pp_edges)
    print(f"  Person<->Person: {len(pp_edges)} edges | {dict(pp_types)}")

    # ─── Calibration: match coarse relationship targets ─────────────
    print("Calibrating relationships to targets...")
    pp_edges = calibrate_relationship_targets(pp_edges, graph_persons, p_latent_map, rng)
    pp_types = Counter(e["edge_type"] for e in pp_edges)
    print(f"  Person<->Person (calibrated): {len(pp_edges)} edges | {dict(pp_types)}")

    # ─── Social graph enrichment: serendipitous + triadic closure ───
    print("Adding serendipitous cross-community edges...")
    pp_edges = _add_serendipitous_edges(pp_edges, graph_persons, p_latent_map, rng)

    print("Running triadic closure (stage-gated)...")
    pp_edges = _add_triadic_closure(pp_edges, graph_persons, rng)

    pp_types = Counter(e["edge_type"] for e in pp_edges)
    print(f"  Person<->Person (enriched): {len(pp_edges)} edges | {dict(pp_types)}")

    
    # ═══ Generate person<->company edges ════════════════════════════════
    pc_edges = []
    if company_latent:
        print("Generating person<->company edges...")
        pc_edges = generate_person_company_edges(
            graph_persons, companies, person_latent, company_latent, rng
        )
        pc_types = Counter(e["edge_type"] for e in pc_edges)
        print(f"  Person<->Company: {len(pc_edges)} edges | {dict(pc_types)}")
    
    # ═══ Generate company<->company edges ═══════════════════════════════
    cc_edges = []
    if company_latent:
        print("Generating company<->company edges...")
        cc_edges = generate_company_company_edges(companies, company_latent, rng)
        cc_types = Counter(e["edge_type"] for e in cc_edges)
        print(f"  Company<->Company: {len(cc_edges)} edges | {dict(cc_types)}")
    
    # Add temporal dimension to person<->person edges BEFORE merge
    print("Adding temporal dimension...")
    pp_edges = add_temporal_edges(pp_edges, graph_persons)
    pc_edges, cc_edges = add_temporal_to_non_person_edges(pc_edges, cc_edges, graph_persons, companies)

    # Merge and re-deduplicate all edges with stable validity windows.
    all_edges = pp_edges + pc_edges + cc_edges
    all_edges, n_dedup = dedupe_edges(all_edges)
    
    # ═══ Community detection ══════════════════════════════════════════
    print("Running community detection...")
    communities = detect_communities_louvain(pp_edges, seed=SEED)
    communities, n_completed = complete_community_assignments(communities, graph_persons)
    n_communities = len(set(communities.values())) if communities else 0
    print(f"  Detected {n_communities} communities across {len(communities)} nodes")
    if n_completed > 0:
        print(f"  Filled {n_completed} isolated persons with deterministic community fallback")
    
    # ═══ Save outputs ═════════════════════════════════════════════════
    
    # Edge graph CSV
    keys = ["src_id", "dst_id", "src_name", "dst_name", "edge_type", "sign",
            "weight", "source_kind", "reason", "valid_from", "valid_to"]
    edge_path = GRAPH_DIR / "edge_graph.csv"
    with open(edge_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_edges)
    print(f"\nSaved {len(all_edges)} edges -> {edge_path}")
    
    # Communities CSV
    comm_path = GRAPH_DIR / "communities.csv"
    with open(comm_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["person_id", "community"])
        writer.writeheader()
        for pid, comm in sorted(communities.items()):
            writer.writerow({"person_id": pid, "community": comm})
    print(f"Saved {len(communities)} community assignments -> {comm_path}")
    
    # Edge JSON for debugging
    with open(GRAPH_DIR / "edges_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(all_edges, f, indent=2, ensure_ascii=False)
    
    # ═══ Summary ══════════════════════════════════════════════════════
    all_types = Counter(e["edge_type"] for e in all_edges)
    print(f"\n{'='*60}")
    print(f"  HYBRID GRAPH COMPLETE")
    print(f"{'='*60}")
    print(f"  Total edges:      {len(all_edges)}")
    print(f"  Person<->Person:    {len(pp_edges)}")
    print(f"  Person<->Company:   {len(pc_edges)}")
    print(f"  Company<->Company:  {len(cc_edges)}")
    print(f"  Dedupe merges:      {n_dedup}")
    print(f"  Communities:      {n_communities}")
    print(f"  Edge types:       {dict(all_types)}")
    print(f"  Temporal edges:   {sum(1 for e in all_edges if 'valid_from' in e)}")
    print(f"  Source:           100% latent_hybrid (deterministic, seeded)")
    print(f"  Cost:             $0.00 (pure procedural)")


if __name__ == "__main__":
    main()


