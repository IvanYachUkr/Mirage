from __future__ import annotations

"""
Rewritten world_state.py
========================

This module owns all persistent in-memory state used by movie assembly:
entities, graph-derived affinity lookups, latent-variable caches, hidden
confounders, career timelines, franchise slots, and year-filtered actor caches.

Design goals
------------
- keep the public API stable for assembly.py / generate_movies.py /
  temporal_evolution_api.py / financials.py
- make load-time behaviour deterministic and easier to reason about
- keep hot-path lookups O(1) after load()
- provide sane fallbacks when optional artifacts are missing
"""

import json
import hashlib
import logging
import os
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import sys
sys.path.insert(0, os.path.dirname(__file__))

from contracts import (
    AWARD_CAMPAIGN_GENRES,
    BUDGET_RANGES,
    CAST_SIZE_RANGES,
    CERTIFICATIONS,
    CERT_DISTS,
    COUNTRIES,
    COUNTRY_LANGUAGE,
    COUNTRY_WEIGHTS,
    CREW_DEPARTMENTS,
    DECADE_WEIGHTS,
    DIRECTOR_STYLES,
    ENTITY_COUNTS,
    FRANCHISE_CONFIG,
    GENRES,
    GENRE_WEIGHTS,
    N_AGENCIES,
    N_COMPANY_CLIQUES,
    PRODUCTION_TIERS,
    SNAPSHOT_CONFIG,
    STYLE_TAGS,
    TIER_WEIGHTS,
    YEAR_RANGE,
    ARCHETYPES,
    generate_compositional_title,
)
from utils import _safe_float, _clip01, normalize_weights

log = logging.getLogger(__name__)


# ============================================================================
# Module-level helpers
# ============================================================================

# C1-FIX: _clip01 now imported from utils.py


def _split_multi_value(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
    return []


def _stable_unit_interval(*parts: Any) -> float:
    key = "||".join(str(p) for p in parts).encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float((1 << 64) - 1)


def _normalize_distribution(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    arr = np.clip(arr, 1e-9, None)
    s = float(arr.sum())
    if s <= 0:
        return [1.0 / len(arr)] * len(arr)
    return (arr / s).tolist()


def _parse_tag_set(raw: str) -> frozenset[str]:
    raw = str(raw or "").lower().replace("[", "").replace("]", "").replace("'", "")
    return frozenset(t.strip() for t in raw.replace(";", ",").split(",") if t.strip())


def _normalize_vec(vec: Any, dim: int) -> np.ndarray:
    a = np.asarray(vec, dtype=np.float32)
    if a.ndim != 1 or len(a) != dim:
        a = np.full(dim, 0.5, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-10 else a


def _cosine_sim_cached(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        return 0.0
    return float(max(0.0, min(1.0, np.dot(aa, bb))))


# ============================================================================
# Latent fallback getters
# ============================================================================

def get_person_latent(world: "WorldState", person_id: int) -> Dict[str, Any]:
    pid = int(person_id)
    lv = world.person_latent.get(pid)
    if isinstance(lv, dict):
        return lv

    pop = _safe_float(world.person_pop_weight.get(pid, 0.1), default=0.1)
    return {
        "person_id": pid,
        "creative_style_vector": [0.0] * 8,
        "risk_tolerance": 0.5,
        "collaboration_style": "ensemble",
        "controversy_score": 0.15,
        "public_reputation": float(np.clip(pop, 0.0, 1.0)),
        "budget_band_pref": [0.5] * 5,
        "artistic_ambition": 0.5,
        "volatility": 0.4,
    }


def get_company_latent(world: "WorldState", company_id: int) -> Dict[str, Any]:
    cid = int(company_id)
    lv = world.company_latent.get(cid)
    if isinstance(lv, dict):
        return lv

    pop = _safe_float(world.company_pop_weight.get(cid, 0.5), default=0.5)
    tier_to_focus = {
        "Global": [0.60, 0.25, 0.10, 0.03, 0.02],
        "Major": [0.20, 0.50, 0.20, 0.07, 0.03],
        "Mid-Budget": [0.05, 0.20, 0.50, 0.20, 0.05],
        "Indie": [0.02, 0.05, 0.20, 0.55, 0.18],
        "Micro": [0.01, 0.02, 0.10, 0.35, 0.52],
    }
    tier_str = getattr(world, "_company_tier_map", {}).get(cid, "Mid-Budget")
    return {
        "company_id": cid,
        "risk_appetite": 0.5,
        "prestige_score": float(np.clip(pop, 0.0, 1.0)),
        "genre_portfolio": [1.0 / 12] * 12,
        "budget_tier_focus": tier_to_focus.get(tier_str, tier_to_focus["Mid-Budget"]),
        "controversy_tolerance": 0.5,
        "market_trend_sensitivity": 0.5,
    }


# ============================================================================
# Similarity
# ============================================================================

def latent_similarity(world: "WorldState", pid_a: int, pid_b: int) -> float:
    ca = world._person_sim_cache.get(int(pid_a))
    cb = world._person_sim_cache.get(int(pid_b))
    if ca is None or cb is None:
        return 0.0

    csv_a = ca.get("csv_normed")
    csv_b = cb.get("csv_normed")
    csv_sim = float(max(0.0, min(1.0, np.dot(csv_a, csv_b)))) if csv_a is not None and csv_b is not None else _cosine_sim_cached(
        ca.get("creative_style_vector", [0.0] * 8),
        cb.get("creative_style_vector", [0.0] * 8),
    )

    ga_a, ga_b = ca["genre_set"], cb["genre_set"]
    genre_sim = len(ga_a & ga_b) / max(1, len(ga_a | ga_b))

    st_a, st_b = ca["style_set"], cb["style_set"]
    style_sim = len(st_a & st_b) / max(1, len(st_a | st_b))

    lat_dist = (abs(ca["risk_tolerance"] - cb["risk_tolerance"]) + abs(ca["artistic_ambition"] - cb["artistic_ambition"])) / 2.0
    scalar_sim = max(0.0, 1.0 - lat_dist)

    bbp_a = ca.get("bbp_normed")
    bbp_b = cb.get("bbp_normed")
    bbp_sim = float(max(0.0, min(1.0, np.dot(bbp_a, bbp_b)))) if bbp_a is not None and bbp_b is not None else _cosine_sim_cached(
        ca.get("budget_band_pref", [0.5] * 5),
        cb.get("budget_band_pref", [0.5] * 5),
    )

    collab_match = 1.0 if ca.get("collaboration_style") == cb.get("collaboration_style") else 0.0

    return float(
        0.35 * csv_sim
        + 0.20 * genre_sim
        + 0.15 * style_sim
        + 0.15 * scalar_sim
        + 0.10 * bbp_sim
        + 0.05 * collab_match
    )


# Popcount lookup table for 16-bit integers (used by vectorized Jaccard)
_POPCOUNT16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.int8)

def _popcount32(arr: np.ndarray) -> np.ndarray:
    """Vectorized popcount for uint32 arrays via 16-bit table lookup."""
    a = arr.astype(np.uint32)
    lo = (a & 0xFFFF).astype(np.uint16)
    hi = ((a >> 16) & 0xFFFF).astype(np.uint16)
    return _POPCOUNT16[lo].astype(np.int32) + _POPCOUNT16[hi].astype(np.int32)


def latent_similarity_batch(
    world: "WorldState",
    cand_latent_indices: np.ndarray,
    cast_latent_indices: np.ndarray,
) -> np.ndarray:
    """Vectorized full 6-component similarity: candidates (m) vs cast (k).

    Returns shape (m,) — max similarity of each candidate across all cast members.
    Uses pre-built arrays: CSV, BBP (matrix multiply), genre/style (bit-vector
    Jaccard via popcount), risk/ambition (arithmetic), collab (int comparison).
    """
    m = len(cand_latent_indices)
    k = len(cast_latent_indices)
    if m == 0 or k == 0:
        return np.zeros(m, dtype=np.float32)

    # 1. CSV similarity (weight 0.35) — matrix multiply
    cand_csv = world._latent_csv_normed[cand_latent_indices]   # (m, 8)
    cast_csv = world._latent_csv_normed[cast_latent_indices]   # (k, 8)
    csv_sim = np.clip(cand_csv @ cast_csv.T, 0.0, 1.0)        # (m, k)

    # 2. BBP similarity (weight 0.10) — matrix multiply
    cand_bbp = world._latent_bbp_normed[cand_latent_indices]   # (m, 5)
    cast_bbp = world._latent_bbp_normed[cast_latent_indices]   # (k, 5)
    bbp_sim = np.clip(cand_bbp @ cast_bbp.T, 0.0, 1.0)        # (m, k)

    # 3. Genre Jaccard (weight 0.20) — bit-vector popcount
    cand_gbits = world._latent_genre_bits[cand_latent_indices] # (m,) uint32
    cast_gbits = world._latent_genre_bits[cast_latent_indices] # (k,) uint32
    # Broadcast: (m,1) op (1,k) -> (m,k)
    g_inter = _popcount32((cand_gbits[:, None] & cast_gbits[None, :]).ravel()).reshape(m, k)
    g_union = _popcount32((cand_gbits[:, None] | cast_gbits[None, :]).ravel()).reshape(m, k)
    genre_sim = np.where(g_union > 0, g_inter.astype(np.float32) / g_union.astype(np.float32), 0.0)

    # 4. Style Jaccard (weight 0.15) — bit-vector popcount
    cand_sbits = world._latent_style_bits[cand_latent_indices] # (m,) uint32
    cast_sbits = world._latent_style_bits[cast_latent_indices] # (k,) uint32
    s_inter = _popcount32((cand_sbits[:, None] & cast_sbits[None, :]).ravel()).reshape(m, k)
    s_union = _popcount32((cand_sbits[:, None] | cast_sbits[None, :]).ravel()).reshape(m, k)
    style_sim = np.where(s_union > 0, s_inter.astype(np.float32) / s_union.astype(np.float32), 0.0)

    # 5. Scalar similarity (weight 0.15) — risk + ambition distance
    cand_risk = world._latent_risk[cand_latent_indices]        # (m,)
    cast_risk = world._latent_risk[cast_latent_indices]        # (k,)
    cand_amb  = world._latent_ambition[cand_latent_indices]    # (m,)
    cast_amb  = world._latent_ambition[cast_latent_indices]    # (k,)
    lat_dist = (np.abs(cand_risk[:, None] - cast_risk[None, :]) +
                np.abs(cand_amb[:, None] - cast_amb[None, :])) / 2.0
    scalar_sim = np.clip(1.0 - lat_dist, 0.0, 1.0)            # (m, k)

    # 6. Collab match (weight 0.05) — integer comparison
    cand_collab = world._latent_collab_code[cand_latent_indices]  # (m,) int8
    cast_collab = world._latent_collab_code[cast_latent_indices]  # (k,) int8
    collab_sim = (cand_collab[:, None] == cast_collab[None, :]).astype(np.float32)  # (m, k)

    # Weighted combination
    total = (0.35 * csv_sim + 0.20 * genre_sim + 0.15 * style_sim +
             0.15 * scalar_sim + 0.10 * bbp_sim + 0.05 * collab_sim)  # (m, k)

    return total.max(axis=1).astype(np.float32)  # (m,)


# ============================================================================
# WorldState
# ============================================================================

class WorldState:
    """Persistent state container used by generation, evolution, and post-processing."""

    def __init__(self, base_dir: str, seed: int = 42):
        self.base_dir = Path(base_dir)
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        self.seed = int(seed)

        # Core entities / graph
        self.persons: Optional[pd.DataFrame] = None
        self.person_roles: Optional[pd.DataFrame] = None
        self.actors: Optional[pd.DataFrame] = None
        self.directors: Optional[pd.DataFrame] = None
        self.crew_pools: Dict[str, pd.DataFrame] = {}
        self.companies: Optional[pd.DataFrame] = None
        self.keywords: Optional[pd.DataFrame] = None
        self.title_bank: Optional[pd.DataFrame] = None
        self.character_bank: Optional[pd.DataFrame] = None
        self.edge_graph = None
        self.affinity_index: Optional[dict] = None
        self.communities: Dict[int, int] = {}

        # Confounders / structural assignments
        self.person_agency: Dict[int, int] = {}
        self.company_clique: Dict[int, int] = {}
        self.company_family: Dict[int, set] = {}
        self._merge_families: Dict[int, set] = {}

        # Latents and lookups
        self.person_latent: Dict[int, dict] = {}
        self.company_latent: Dict[int, dict] = {}
        self.person_pop_weight: Dict[int, float] = {}
        self.company_pop_weight: Dict[int, float] = {}
        self.company_financial_profile: Dict[int, dict] = {}
        self._company_tier_map: Dict[int, str] = {}
        self._person_sim_cache: Dict[int, dict] = {}
        self._latent_pid_to_idx: Dict[int, int] = {}
        self._latent_csv_normed: Optional[np.ndarray] = None
        self._latent_bbp_normed: Optional[np.ndarray] = None
        self._latent_controversy: Optional[np.ndarray] = None
        self._latent_volatility: Optional[np.ndarray] = None
        self._latent_collab: Optional[np.ndarray] = None
        self._latent_avoid_genres: Dict[int, set] = {}
        self.edge_weights: Dict[tuple, float] = {}

        # Node-keyed edge adjacency (built once at load time)
        self._friend_adj_all: Dict[int, list] = {}
        self._rival_adj_all: Dict[int, list] = {}

        # Hot-path caches
        self._year_cache: Dict[int, pd.DataFrame] = {}
        self._company_by_tier_genre: Dict[tuple, set] = {}
        self._yearly_workload = Counter()
        self._used_char_names_global = set()
        self._crew_fallback_warned = set()

        # Runtime generation state
        self.person_award_wins: Dict[int, int] = {}
        self.director_quality_offset: Dict[int, float] = {}
        self.person_film_count = Counter()
        self.person_recent = defaultdict(list)
        self.director_recent = defaultdict(list)
        self.company_recent = defaultdict(list)
        self.director_writer_history: Dict[int, set] = {}
        self.director_film_count = Counter()
        self.company_film_count = Counter()
        self.used_titles = set()
        self.franchises: List[dict] = []
        self.movie_franchise_map: Dict[int, dict] = {}

        # Event / temporal state
        self.active_effects: List[dict] = []
        self.world_events: List[dict] = []
        self.genre_weight_overrides: Dict[str, float] = {}
        self.country_weight_overrides: Dict[str, float] = {}
        self.award_prestige: Dict[str, float] = {}
        self.paused_persons: Dict[int, dict] = {}

        # Financial momentum state
        self.director_recent_outcomes = defaultdict(list)
        self.company_recent_outcomes = defaultdict(list)
        self.genre_recent_outcomes = defaultdict(list)
        self._financial_regime_cache: Dict[int, dict] = {}

        # Misc temporal-spawn state
        self._chemistry_pairs = set()
        self._yearly_friendship_spawns: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public load
    # ------------------------------------------------------------------

    def load(self):
        edir = self.base_dir / "entities"
        if not edir.exists():
            raise FileNotFoundError(f"Entities directory not found: {edir}")

        self.persons = self._load_csv_with_id(edir / "person.csv", "person_id")
        self._assign_career_timelines()
        self._assign_pop_weights()
        self.person_roles = self._load_person_roles(edir)
        self._build_person_role_views()

        self.companies = self._load_csv_with_id(edir / "company.csv", "company_id")
        self.companies["pop_weight"] = pd.to_numeric(self.companies.get("pop_weight", 0.5), errors="coerce").fillna(0.5)
        self._assign_company_pop_weights()

        self.keywords = self._load_csv_with_id(edir / "keyword.csv", "keyword_id")

        tb_path = edir / "title_bank.csv"
        self.title_bank = pd.read_csv(tb_path) if tb_path.exists() else pd.DataFrame(columns=["title", "tagline", "genre_hint"])

        cb_path = edir / "character_bank.csv"
        if not cb_path.exists():
            raise FileNotFoundError(f"character_bank.csv not found at {cb_path}")
        self.character_bank = pd.read_csv(cb_path)

        self._load_edge_graph()
        self._build_edge_adjacency()
        self._load_latents(edir)
        self._load_company_financial_profiles(edir)
        self._build_lookup_dicts()
        self._init_director_quality_offsets()
        self._load_communities()
        self._assign_agencies()
        self._assign_company_cliques()
        self._setup_franchises()
        self._build_person_sim_cache()
        self._prewarm_year_cache()

        print(
            f"World loaded: {len(self.persons)} persons, {len(self.companies)} companies, "
            f"{len(self.keywords)} keywords, {len(self.title_bank)} titles, {len(self.character_bank)} characters"
        )

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_csv_with_id(self, path: Path, id_col: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if id_col not in df.columns:
            df[id_col] = np.arange(1, len(df) + 1, dtype=int)
        else:
            df[id_col] = pd.to_numeric(df[id_col], errors="coerce").fillna(0).astype(int)
            if (df[id_col] <= 0).any() or df[id_col].duplicated().any():
                print(f"  WARNING: invalid {id_col} values in {path.name} -- reassigning sequential IDs")
                df[id_col] = np.arange(1, len(df) + 1, dtype=int)
        return df

    def _load_person_roles(self, edir: Path) -> pd.DataFrame:
        path = edir / "person_roles.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "person_id" in df.columns:
                df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").fillna(0).astype(int)
            return df

        assert self.persons is not None
        pids = self.persons["person_id"].astype(int).tolist()
        roles_raw = self.persons["roles"].fillna("actor").astype(str).str.lower().tolist() if "roles" in self.persons.columns else ["actor"] * len(self.persons)
        rows = []
        for pid, raw in zip(pids, roles_raw):
            rows.append({"person_id": pid, "role_type": "actor"})
            if "director" in raw:
                rows.append({"person_id": pid, "role_type": "director"})
            for crew_role in CREW_DEPARTMENTS:
                if crew_role.lower() in raw:
                    rows.append({"person_id": pid, "role_type": crew_role})
        return pd.DataFrame(rows)

    def _build_person_role_views(self):
        assert self.persons is not None and self.person_roles is not None
        role_to_ids: Dict[str, set] = {}
        for role, g in self.person_roles.groupby("role_type"):
            role_to_ids[str(role)] = set(g["person_id"].astype(int).tolist())

        actor_ids = role_to_ids.get("actor", set())
        director_ids = role_to_ids.get("director", set())
        self.actors = self.persons[self.persons["person_id"].isin(actor_ids)].copy()
        self.directors = self.persons[self.persons["person_id"].isin(director_ids)].copy()

        self.crew_pools = {}
        for crew_role in CREW_DEPARTMENTS:
            ids = role_to_ids.get(crew_role, set())
            self.crew_pools[crew_role] = self.persons[self.persons["person_id"].isin(ids)].copy() if ids else pd.DataFrame(columns=self.persons.columns)

        self.writers = self.crew_pools.get("writer", pd.DataFrame())
        self.cinematographers = self.crew_pools.get("cinematographer", pd.DataFrame())
        self.editors = self.crew_pools.get("editor", pd.DataFrame())
        self.composers = self.crew_pools.get("composer", pd.DataFrame())

        crew_counts = ", ".join(f"{len(self.crew_pools[r])} {r}s" for r in CREW_DEPARTMENTS if len(self.crew_pools[r]) > 0)
        print(f"Loaded {len(self.persons)} persons ({len(self.actors)} actors, {len(self.directors)} directors, {crew_counts or 'no crew pools'})")
        print(f"Loaded {len(self.companies) if self.companies is not None else 0} companies")
        print(f"Loaded {len(self.keywords) if self.keywords is not None else 0} keywords")
        print(f"Loaded {len(self.title_bank) if self.title_bank is not None else 0} curated titles")
        print(f"Loaded {len(self.character_bank) if self.character_bank is not None else 0} character names")

    def _load_edge_graph(self):
        edge_path = self.base_dir / "graph" / "edge_graph.csv"
        if not edge_path.exists():
            self.edge_graph = None
            self.affinity_index = {
                "friendships": {},
                "rivalries": {},
                "director_prefs": defaultdict(list),
                "director_avoids": defaultdict(list),
                "company_affinity": {},
                "company_rivalry": {},
                "person_company_affinity": defaultdict(list),
            }
            return

        from graph_stitching import EdgeGraph

        df = pd.read_csv(edge_path)
        eg = EdgeGraph()
        eg.edges = df.to_dict("records")
        for row in df.itertuples(index=False):
            sid = int(getattr(row, "src_id", 0))
            did = int(getattr(row, "dst_id", 0))
            if sid > 0:
                eg.id_to_name[sid] = getattr(row, "src_name", "") if hasattr(row, "src_name") else ""
            if did > 0:
                eg.id_to_name[did] = getattr(row, "dst_name", "") if hasattr(row, "dst_name") else ""

        self.edge_graph = eg
        self.affinity_index = eg.build_affinity_index()
        self._populate_edge_weights()

    def _load_latents(self, edir: Path):
        self.person_latent = self._load_person_latent_json(edir / "persons_latent.json")
        self.company_latent = self._load_company_latent_json(edir / "companies_latent.json")

    def _load_person_latent_json(self, path: Path) -> Dict[int, dict]:
        if not path.exists():
            warnings.warn(
                "persons_latent.json not found. Person latent variables will use fallback values.",
                UserWarning,
                stacklevel=2,
            )
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[int, dict] = {}
        for lv in data:
            if not isinstance(lv, dict):
                continue
            try:
                pid = int(lv.get("person_id", 0))
            except Exception:
                continue
            if pid <= 0:
                continue
            item = dict(lv)
            item["person_id"] = pid
            for k, d in (("public_reputation", 0.5), ("controversy_score", 0.15), ("risk_tolerance", 0.5), ("artistic_ambition", 0.5), ("volatility", 0.4)):
                item[k] = _clip01(item.get(k, d), d)
            out[pid] = item
        print(f"Loaded {len(out)} person latent vars")
        return out

    def _load_company_latent_json(self, path: Path) -> Dict[int, dict]:
        if not path.exists():
            warnings.warn(
                "companies_latent.json not found. Company latent variables will use fallback values.",
                UserWarning,
                stacklevel=2,
            )
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[int, dict] = {}
        for lv in data:
            if not isinstance(lv, dict):
                continue
            try:
                cid = int(lv.get("company_id", 0))
            except Exception:
                continue
            if cid <= 0:
                continue
            item = dict(lv)
            item["company_id"] = cid
            for k, d in (("risk_appetite", 0.5), ("prestige_score", 0.5), ("controversy_tolerance", 0.5), ("market_trend_sensitivity", 0.5)):
                item[k] = _clip01(item.get(k, d), d)
            out[cid] = item
        print(f"Loaded {len(out)} company latent vars")
        return out

    def _load_company_financial_profiles(self, edir: Path):
        path = edir / "company_financial_profile.csv"
        if not path.exists():
            self.company_financial_profile = {}
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"WARNING: could not load company_financial_profile.csv: {exc}")
            self.company_financial_profile = {}
            return

        if "company_id" not in df.columns:
            self.company_financial_profile = {}
            return
        df["company_id"] = pd.to_numeric(df["company_id"], errors="coerce").fillna(0).astype(int)
        self.company_financial_profile = {
            int(row["company_id"]): {k: row[k] for k in df.columns if k != "company_id"}
            for _, row in df.iterrows()
            if int(row["company_id"]) > 0
        }
        print(f"Loaded {len(self.company_financial_profile)} company financial profiles")

    def _build_lookup_dicts(self):
        assert self.persons is not None and self.companies is not None
        self.person_pop_weight = dict(zip(self.persons["person_id"].astype(int), self.persons["pop_weight"].astype(float)))
        self.company_pop_weight = dict(zip(self.companies["company_id"].astype(int), self.companies["pop_weight"].astype(float)))
        self._company_tier_map = dict(zip(self.companies["company_id"].astype(int), self.companies.get("tier", pd.Series(["Mid-Budget"] * len(self.companies))).astype(str)))

    def _load_communities(self):
        path = self.base_dir / "graph" / "communities.csv"
        all_pids = set(self.persons["person_id"].astype(int).tolist()) if self.persons is not None else set()
        communities: Dict[int, int] = {}
        if path.exists():
            try:
                df = pd.read_csv(path)
                if {"person_id", "community"}.issubset(df.columns):
                    df = df[["person_id", "community"]].copy()
                    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").fillna(0).astype(int)
                    df["community"] = pd.to_numeric(df["community"], errors="coerce").fillna(0).astype(int)
                    df = df[(df["person_id"] > 0) & (df["community"] > 0)]
                    communities = dict(zip(df["person_id"], df["community"]))
            except Exception as exc:
                print(f"WARNING: failed to load communities.csv: {exc}")

        if not communities:
            n = max(8, min(64, int(np.sqrt(max(1, len(all_pids))))))
            for pid in all_pids:
                h = int(hashlib.md5(f"community|{pid}".encode("utf-8")).hexdigest(), 16)
                communities[int(pid)] = int(h % n) + 1
        else:
            existing = sorted(set(communities.values()))
            for pid in sorted(all_pids):
                if pid not in communities:
                    h = int(hashlib.md5(f"community|{pid}".encode("utf-8")).hexdigest(), 16)
                    communities[pid] = existing[h % len(existing)]

        self.communities = communities
        print(f"Loaded {len(self.communities)} community assignments")

    # ------------------------------------------------------------------
    # Structural assignments
    # ------------------------------------------------------------------

    def _assign_agencies(self):
        assert self.actors is not None
        self.person_agency = {}
        actor_ids = set(self.actors["person_id"].astype(int).tolist())

        entities_dir = self.base_dir / "entities"
        agencies_path = entities_dir / "agencies.json"
        persons_json_path = entities_dir / "persons.json"

        agency_name_to_id: Dict[str, int] = {}
        if agencies_path.exists():
            try:
                agencies = json.loads(agencies_path.read_text(encoding="utf-8"))
                if isinstance(agencies, list):
                    for i, a in enumerate(agencies, start=1):
                        if not isinstance(a, dict):
                            continue
                        name = str(a.get("name", "")).strip().lower()
                        if name and name not in agency_name_to_id:
                            agency_name_to_id[name] = i
            except Exception as exc:
                print(f"WARNING: failed to parse agencies.json: {exc}")

        assigned_from_generated = 0
        if persons_json_path.exists():
            try:
                persons_json = json.loads(persons_json_path.read_text(encoding="utf-8"))
                if isinstance(persons_json, list):
                    for row in persons_json:
                        if not isinstance(row, dict):
                            continue
                        try:
                            pid = int(row.get("person_id", 0) or 0)
                        except Exception:
                            pid = 0
                        if pid <= 0 or pid not in actor_ids:
                            continue
                        aid = None
                        raw_aid = row.get("agency_id")
                        if raw_aid is not None:
                            try:
                                q = int(raw_aid)
                                if q > 0:
                                    aid = q
                            except Exception:
                                pass
                        if aid is None:
                            aname = str(row.get("agency", "")).strip().lower()
                            if aname:
                                aid = agency_name_to_id.get(aname)
                                if aid is None and not agency_name_to_id:
                                    agency_name_to_id[aname] = len(agency_name_to_id) + 1
                                    aid = agency_name_to_id[aname]
                        if aid is not None and int(aid) > 0:
                            self.person_agency[pid] = int(aid)
                            assigned_from_generated += 1
            except Exception as exc:
                print(f"WARNING: failed to parse persons.json agency assignments: {exc}")

        n_agencies = max(int(N_AGENCIES), len(agency_name_to_id), max(self.person_agency.values(), default=0), 1)
        missing = [pid for pid in sorted(actor_ids) if pid not in self.person_agency]
        if missing:
            assert self.actors is not None
            actor_df = self.actors.set_index("person_id")
            for pid in missing:
                tags = str(actor_df.loc[pid].get("style_tags", "")) if pid in actor_df.index else ""
                nat = str(actor_df.loc[pid].get("nationality", "")) if pid in actor_df.index else ""
                key = f"agency|{pid}|{tags}|{nat}"
                bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % n_agencies
                self.person_agency[pid] = bucket + 1

        sizes = sorted(Counter(self.person_agency.values()).values(), reverse=True)[:10]
        print(f"Assigned {len(self.person_agency)} actors to {n_agencies} agencies (generated={assigned_from_generated}, sizes={sizes})")

    def _assign_company_cliques(self):
        assert self.companies is not None
        self.company_clique = {}
        entities_dir = self.base_dir / "entities"
        cliques_path = entities_dir / "cliques.json"
        companies_json_path = entities_dir / "companies.json"

        clique_name_to_id: Dict[str, int] = {}
        clique_ids: List[int] = []
        if cliques_path.exists():
            try:
                cliques = json.loads(cliques_path.read_text(encoding="utf-8"))
                if isinstance(cliques, list):
                    for i, c in enumerate(cliques, start=1):
                        if not isinstance(c, dict):
                            continue
                        try:
                            cid = int(c.get("clique_id", i))
                        except Exception:
                            cid = i
                        cid = max(cid, 1)
                        name = str(c.get("name", "")).strip().lower()
                        if name:
                            clique_name_to_id[name] = cid
                        clique_ids.append(cid)
            except Exception as exc:
                print(f"WARNING: failed to parse cliques.json: {exc}")

        n_cliques = max(max(clique_ids) if clique_ids else 0, int(N_COMPANY_CLIQUES), 1)
        assigned_from_generated = 0
        if companies_json_path.exists():
            try:
                companies_json = json.loads(companies_json_path.read_text(encoding="utf-8"))
                if isinstance(companies_json, list):
                    for row in companies_json:
                        if not isinstance(row, dict):
                            continue
                        try:
                            company_id = int(row.get("company_id", 0) or 0)
                        except Exception:
                            continue
                        if company_id <= 0:
                            continue
                        target_id = None
                        raw_id = row.get("clique_id")
                        raw_name = row.get("clique")
                        if raw_id is not None:
                            try:
                                q = int(raw_id)
                                if q > 0:
                                    target_id = q
                            except Exception:
                                pass
                        if target_id is None and raw_name is not None:
                            if isinstance(raw_name, (int, float)):
                                try:
                                    q = int(raw_name)
                                    if q > 0:
                                        target_id = q
                                except Exception:
                                    pass
                            else:
                                target_id = clique_name_to_id.get(str(raw_name).strip().lower())
                        if target_id is not None and target_id > 0:
                            self.company_clique[company_id] = int(target_id)
                            assigned_from_generated += 1
            except Exception as exc:
                print(f"WARNING: failed to parse companies.json clique assignments: {exc}")

        tier_order = {"Global": 0, "Major": 1, "Mid-Budget": 2, "Indie": 3, "Micro": 4}
        for row in self.companies.itertuples(index=False):
            cid = int(getattr(row, "company_id"))
            if cid in self.company_clique:
                continue
            tier = str(getattr(row, "tier", "Mid-Budget"))
            specialty = str(getattr(row, "specialty_genres", ""))
            tier_idx = tier_order.get(tier, 2)
            first_genre = specialty.split(";")[0].strip() if specialty else "Drama"
            try:
                genre_idx = GENRES.index(first_genre)
            except ValueError:
                genre_idx = int(hashlib.md5(first_genre.encode("utf-8")).hexdigest(), 16) % max(1, len(GENRES))
            clique = (tier_idx * 7 + genre_idx) % n_cliques + 1
            self.company_clique[cid] = int(clique)

        sizes = sorted(Counter(self.company_clique.values()).values(), reverse=True)[:10]
        print(f"Assigned {len(self.company_clique)} companies to {n_cliques} cliques (generated={assigned_from_generated}, sizes={sizes})")

    # ------------------------------------------------------------------
    # Pop weights / careers / franchises
    # ------------------------------------------------------------------

    def _assign_career_timelines(self):
        assert self.persons is not None
        n = len(self.persons)
        stage = self.persons.get("career_stage", pd.Series(["prime"] * n)).fillna("prime").astype(str)

        yr = YEAR_RANGE
        yr_lo = yr[0] if yr else 1975
        yr_hi = yr[1] if yr else 2025

        debut, peak_s, peak_e, retire, yearly_max = [], [], [], [], []
        for cs in stage.tolist():
            cs = str(cs).lower()
            # B1-FIX: always use yr_lo/yr_hi (which default to 1975/2025
            # when YEAR_RANGE is None). The old code had a separate branch
            # for YEAR_RANGE=None with hardcoded decades that was never
            # reached by the intended YEAR_RANGE-aware logic.
            if cs == "legend":
                d = self.rng.randint(yr_lo - 35, yr_lo - 15)
                ps = d + self.rng.randint(5, 10)
                pe = ps + self.rng.randint(5, 8)
                r = max(yr_hi + self.rng.randint(0, 8), pe + 2)
                ym = self.rng.randint(5, 9)
            elif cs == "veteran":
                d = self.rng.randint(yr_lo - 25, yr_lo - 8)
                ps = d + self.rng.randint(5, 10)
                pe = ps + self.rng.randint(4, 7)
                r = max(yr_hi + self.rng.randint(0, 10), pe + 3)
                ym = self.rng.randint(3, 6)
            elif cs == "prime":
                d = self.rng.randint(yr_lo - 15, yr_lo - 2)
                ps = d + self.rng.randint(3, 7)
                pe = ps + self.rng.randint(3, 6)
                r = max(yr_hi + self.rng.randint(5, 15), pe + 5)
                ym = self.rng.randint(4, 7)
            elif cs == "rising":
                d = self.rng.randint(yr_lo - 5, yr_hi - 1)
                ps = d + self.rng.randint(2, 5)
                pe = ps + self.rng.randint(3, 6)
                r = 2100
                ym = self.rng.randint(2, 5)
            else:
                if self.rng.random_sample() < 0.7:
                    d = self.rng.randint(yr_lo - 30, yr_lo - 10)
                    ps = d + self.rng.randint(5, 10)
                    pe = ps + self.rng.randint(3, 6)
                    r = self.rng.randint(yr_lo - 3, yr_lo + 1)
                else:
                    d = self.rng.randint(yr_lo - 40, yr_lo - 20)
                    ps = d + self.rng.randint(5, 10)
                    pe = ps + self.rng.randint(3, 6)
                    r = pe + self.rng.randint(0, 5)
                ym = self.rng.randint(1, 2)

            debut.append(int(d))
            peak_s.append(int(ps))
            peak_e.append(int(pe))
            retire.append(int(min(r, 2100)))
            yearly_max.append(int(ym))

        self.persons["debut_year"] = debut
        self.persons["peak_start"] = peak_s
        self.persons["peak_end"] = peak_e
        self.persons["retirement_year"] = retire
        self.persons["yearly_max"] = yearly_max

    def _assign_pop_weights(self):
        assert self.persons is not None
        n = len(self.persons)
        if n <= 0:
            self.persons["pop_weight"] = []
            return

        def _gini(x: np.ndarray) -> float:
            arr = np.asarray(x, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return 0.0
            arr = np.clip(arr, 1e-12, None)
            arr = np.sort(arr)
            idx = np.arange(1, arr.size + 1, dtype=float)
            return float((2.0 * np.sum(idx * arr) - (arr.size + 1.0) * np.sum(arr)) / (arr.size * np.sum(arr)))

        raw = None
        if "pop_weight" in self.persons.columns:
            pw = pd.to_numeric(self.persons["pop_weight"], errors="coerce")
            if pw.notna().sum() > n * 0.6:
                raw = pw.fillna(float(pw.median()) if pw.notna().any() else 0.1).values.astype(float)
        if raw is None:
            raw = (self.rng.pareto(1.45, size=n) + 1.0).astype(float)

        stage_col = self.persons.get("career_stage", pd.Series(["prime"] * n)).fillna("prime").astype(str).str.lower().values
        stage_mult_map = {"legend": 2.6, "prime": 1.55, "veteran": 1.25, "rising": 0.85, "retired": 0.35}
        stage_mult = np.array([stage_mult_map.get(s, 1.0) for s in stage_col], dtype=float)

        signal = np.clip(raw * stage_mult, 1e-12, None)
        signal = signal * (self.rng.pareto(1.20, size=n) + 1.0)
        order = np.argsort(-signal)
        stage_norm = stage_mult / max(float(stage_mult.max()), 1e-12)

        # A1-FIX: scale Gini target with dataset size so top-k concentration
        # stays realistic. log10 scaling: N=500→0.62, N=5000→0.55, N=24000→0.50
        target_mid = float(np.clip(0.72 - 0.025 * np.log10(max(n, 100)), 0.45, 0.65))
        target_lo = target_mid - 0.04
        target_hi = target_mid + 0.04

        def _weights_for_alpha(alpha: float) -> np.ndarray:
            ranks = np.arange(1, n + 1, dtype=float)
            base = 1.0 / np.power(ranks, alpha)
            w = np.zeros(n, dtype=float)
            w[order] = base
            w = w * (0.70 + 0.30 * stage_norm)
            w = w / max(float(w.max()), 1e-12)
            return np.clip(1e-6 + (1.0 - 1e-6) * w, 1e-6, 1.0)

        lo_a, hi_a = 0.15, 3.50
        best_w = _weights_for_alpha(1.0)
        best_gap = abs(_gini(best_w) - target_mid)
        # A8-FIX: add tolerance-based early exit instead of always running 36 iterations
        for _ in range(36):
            mid = (lo_a + hi_a) / 2.0
            cand = _weights_for_alpha(mid)
            g = _gini(cand)
            gap = abs(g - target_mid)
            if gap < best_gap:
                best_gap = gap
                best_w = cand
            if target_lo <= g <= target_hi:
                best_w = cand
                break
            elif g < target_lo:
                lo_a = mid
            else:
                hi_a = mid
            # A8-FIX: exit early if converged (gap < 0.001)
            if best_gap < 0.001:
                break

        self.persons["pop_weight"] = best_w
        print(f"  pop_weight: calibrated Gini={_gini(best_w):.3f} (target {target_lo:.2f}-{target_hi:.2f})")

    def _assign_company_pop_weights(self):
        assert self.companies is not None
        n = len(self.companies)
        if n <= 0:
            self.companies["pop_weight"] = []
            return

        # A2-FIX: sort companies by tier before weight assignment so that
        # Global/Major companies always outweigh Indie/Micro companies.
        # Within each tier, jitter provides natural variation.
        _TIER_ORDER = {"Global": 0, "Major": 1, "A-List": 1, "A": 1,
                       "Mid-Budget": 2, "Mid": 2, "Indie": 3,
                       "Micro": 4, "Micro-Budget": 4}
        # Weight bands per tier: (low, high)
        _TIER_BANDS = {
            0: (0.82, 0.99),   # Global
            1: (0.55, 0.84),   # Major / A-List
            2: (0.20, 0.54),   # Mid-Budget
            3: (0.08, 0.22),   # Indie
            4: (0.01, 0.09),   # Micro
        }

        tiers = self.companies["tier"].astype(str).tolist() if "tier" in self.companies.columns else ["Mid"] * n
        tier_codes = np.array([_TIER_ORDER.get(t, 3) for t in tiers], dtype=int)

        weights = np.zeros(n, dtype=float)
        for i in range(n):
            lo, hi = _TIER_BANDS.get(int(tier_codes[i]), (0.08, 0.22))
            weights[i] = self.rng.uniform(lo, hi)

        self.companies["pop_weight"] = weights

    def _init_director_quality_offsets(self):
        assert self.directors is not None
        self.director_quality_offset = {}
        for did in self.directors["person_id"].astype(int).tolist():
            lv = get_person_latent(self, did)
            rep = _safe_float(lv.get("public_reputation"), 0.4)
            base = (rep - 0.5) * 0.9
            noise = float(self.rng.normal(0, 0.55))
            self.director_quality_offset[did] = float(np.clip(base + noise, -1.6, 1.6))

    def _setup_franchises(self):
        cfg = FRANCHISE_CONFIG
        total_movies = int(ENTITY_COUNTS["movies"])
        scale = total_movies / 7500.0
        base_lo, base_hi = cfg["count_range"]
        lo = max(base_lo, int(base_lo * scale))
        hi = max(base_hi, int(base_hi * scale))
        n_franchises = self.rng.randint(lo, hi + 1)
        target_frac = self.rng.uniform(*cfg["target_pct_of_total"])
        target_franchise_movies = int(total_movies * target_frac)

        # A3-FIX: assign each franchise a start_year in the configured range
        # and pre-space installments 2-3 years apart so they cluster temporally.
        yr = YEAR_RANGE
        yr_lo = yr[0] if yr else 1975
        yr_hi = yr[1] if yr else 2025
        span = max(1, yr_hi - yr_lo)

        self.franchises = []
        remaining = target_franchise_movies
        for i in range(n_franchises):
            if remaining <= 0:
                break
            n_movies = self.rng.randint(*cfg["movies_per_franchise"])
            n_movies = min(n_movies, remaining)

            # A3-FIX: franchise start_year leaves room for installments
            max_start = yr_hi - max(1, n_movies - 1) * 2
            start_year = int(self.rng.randint(yr_lo, max(yr_lo + 1, max_start + 1)))
            # Pre-compute installment years with 2-3 year gaps
            inst_years = [start_year]
            for j in range(1, n_movies):
                gap = self.rng.randint(2, 4)  # 2-3 years between installments
                next_yr = min(inst_years[-1] + gap, yr_hi)
                inst_years.append(next_yr)

            self.franchises.append(
                {
                    "franchise_id": i + 1,
                    "name": f"Franchise_{i+1}",
                    "n_movies": int(n_movies),
                    "genre": self.py_rng.choice(GENRES),
                    "tier": self.py_rng.choice(PRODUCTION_TIERS[:3]),
                    "movies_generated": 0,
                    "cast_pool": [],
                    "director_id": None,
                    "company_ids": [],
                    "installment_years": inst_years,  # A3: pre-assigned years
                }
            )
            remaining -= n_movies

        total_franchise_movies = sum(f["n_movies"] for f in self.franchises)

        # A3-FIX: Build year→movie_id bucket for temporal locality.
        # Movies per year follows a roughly uniform distribution across the range.
        movies_per_year = max(1, total_movies // max(1, span))
        year_buckets: dict[int, list[int]] = {}
        for mid in range(1, total_movies + 1):
            approx_year = yr_lo + (mid - 1) // movies_per_year
            approx_year = min(approx_year, yr_hi)
            year_buckets.setdefault(approx_year, []).append(mid)

        # Assign franchise movies from appropriate year buckets
        self.movie_franchise_map = {}
        used_ids: set[int] = set()

        for franchise in self.franchises:
            for inst_idx in range(franchise["n_movies"]):
                target_year = franchise["installment_years"][inst_idx]
                # Search target year ±2 for available IDs
                candidates = []
                for dy in range(6):  # search radius: 0, ±1, ±2, ±3
                    for sign in ([0] if dy == 0 else [-1, 1]):
                        y = target_year + sign * dy
                        for mid in year_buckets.get(y, []):
                            if mid not in used_ids:
                                candidates.append(mid)
                    if candidates:
                        break
                if candidates:
                    chosen = candidates[int(self.rng.randint(0, len(candidates)))]
                    used_ids.add(chosen)
                    self.movie_franchise_map[chosen] = franchise

        print(f"Setup {len(self.franchises)} franchises ({total_franchise_movies} movies, {100.0 * total_franchise_movies / max(1, total_movies):.1f}%)")

    # ------------------------------------------------------------------
    # Graph-derived lookup helpers
    # ------------------------------------------------------------------

    def _populate_edge_weights(self):
        aff = self.affinity_index or {}
        ew: Dict[tuple, float] = {}
        friendships = aff.get("friendships", {})
        for key, val in friendships.items():
            if isinstance(key, tuple) and len(key) == 2:
                ew[key] = float(val.get("weight", 0.5) if isinstance(val, dict) else (val or 0.5))
        rivalries = aff.get("rivalries", {})
        if isinstance(rivalries, dict):
            for key, val in rivalries.items():
                if isinstance(key, tuple) and len(key) == 2:
                    ew[key] = float(val.get("weight", 0.8) if isinstance(val, dict) else (val or 0.8))
        self.edge_weights = ew
        if ew:
            print(f"  Populated edge_weights: {len(ew)} entries")

    def _build_edge_adjacency(self):
        """Pre-build node-keyed adjacency lists for O(degree) traversal in cast selection.

        Each entry is (neighbor_id, weight, valid_from, valid_to).
        Built once at load time; avoids O(n_edges) scan per movie.

        P5-FIX: called lazily via _ensure_edge_adjacency() instead of eagerly
        after every year step. Use _mark_adjacency_dirty() to flag for rebuild.
        """
        aff = self.affinity_index or {}
        friend_adj: Dict[int, list] = defaultdict(list)
        rival_adj: Dict[int, list] = defaultdict(list)

        friendships = aff.get("friendships", {})
        if isinstance(friendships, dict):
            for (a, b), entry in friendships.items():
                if isinstance(entry, dict):
                    w = float(entry.get("weight", 0.0) or 0.0)
                    vf = entry.get("valid_from")
                    vt = entry.get("valid_to")
                else:
                    w = float(entry) if entry else 0.0
                    vf = vt = None
                if w > 0:
                    friend_adj[int(a)].append((int(b), w, vf, vt))
                    friend_adj[int(b)].append((int(a), w, vf, vt))

        rivalries = aff.get("rivalries", {})
        if isinstance(rivalries, dict):
            for (a, b), entry in rivalries.items():
                if isinstance(entry, dict):
                    w = float(entry.get("weight", 0.9) or 0.9)
                    vf = entry.get("valid_from")
                    vt = entry.get("valid_to")
                else:
                    w = 0.9
                    vf = vt = None
                rival_adj[int(a)].append((int(b), w, vf, vt))
                rival_adj[int(b)].append((int(a), w, vf, vt))

        self._friend_adj_all = dict(friend_adj)
        self._rival_adj_all = dict(rival_adj)
        self._adjacency_dirty = False
        n_f = sum(len(v) for v in self._friend_adj_all.values()) // 2
        n_r = sum(len(v) for v in self._rival_adj_all.values()) // 2
        print(f"  Built edge adjacency: {n_f} friendships, {n_r} rivalries")

    def _mark_adjacency_dirty(self):
        """P5-FIX: flag adjacency for lazy rebuild instead of rebuilding immediately."""
        self._adjacency_dirty = True

    def _ensure_edge_adjacency(self):
        """P5-FIX: rebuild adjacency only when dirty (lazy evaluation)."""
        if getattr(self, "_adjacency_dirty", True):
            self._build_edge_adjacency()

    def _build_person_sim_cache(self):
        assert self.persons is not None
        pids = self.persons["person_id"].astype(int).values
        ga_col = self.persons["genre_affinity"].fillna("").astype(str).values if "genre_affinity" in self.persons.columns else np.full(len(self.persons), "")
        st_col = self.persons["style_tags"].fillna("").astype(str).values if "style_tags" in self.persons.columns else np.full(len(self.persons), "")

        n = len(pids)
        csv_all = np.zeros((n, 8), dtype=np.float32)
        bbp_all = np.zeros((n, 5), dtype=np.float32)
        controversy_all = np.full(n, 0.15, dtype=np.float32)
        volatility_all = np.full(n, 0.4, dtype=np.float32)
        collab_all = np.empty(n, dtype=object)
        avoid_genres_sparse: Dict[int, set] = {}
        cache: Dict[int, dict] = {}

        # A3-FIX: bit-vector arrays built in the SAME loop as cache (was two passes)
        from contracts import GENRES as _GENRES, STYLE_TAGS as _STYLE_TAGS
        genre_to_bit = {g.lower(): (1 << i) for i, g in enumerate(_GENRES)}
        style_to_bit = {s.lower(): (1 << i) for i, s in enumerate(_STYLE_TAGS)}
        genre_bits = np.zeros(n, dtype=np.uint32)
        style_bits = np.zeros(n, dtype=np.uint32)
        risk_arr = np.full(n, 0.5, dtype=np.float32)
        ambition_arr = np.full(n, 0.5, dtype=np.float32)
        collab_labels = {"solo": 0, "ensemble": 1, "chameleon": 2, "mentorship": 3}
        collab_code_arr = np.full(n, 2, dtype=np.int8)  # default chameleon

        for i, pid in enumerate(pids):
            lv = get_person_latent(self, int(pid))
            csv_normed = _normalize_vec(lv.get("creative_style_vector", [0.5] * 8), 8)
            bbp_normed = _normalize_vec(lv.get("budget_band_pref", [0.5] * 5), 5)
            ga_parsed = _parse_tag_set(ga_col[i])
            st_parsed = _parse_tag_set(st_col[i])
            risk_val = float(lv.get("risk_tolerance", 0.5))
            ambition_val = float(lv.get("artistic_ambition", 0.5))
            collab_style = str(lv.get("collaboration_style", "chameleon"))
            cache[int(pid)] = {
                "genre_set": ga_parsed,
                "genre_affinities": list(ga_parsed),
                "style_set": st_parsed,
                "risk_tolerance": risk_val,
                "artistic_ambition": ambition_val,
                "csv_normed": csv_normed,
                "bbp_normed": bbp_normed,
                "controversy_score": float(lv.get("controversy_score", 0.15)),
                "public_reputation": float(lv.get("public_reputation", 0.5)),
                "collaboration_style": collab_style,
            }
            csv_all[i] = csv_normed
            bbp_all[i] = bbp_normed
            controversy_all[i] = float(lv.get("controversy_score", 0.15))
            volatility_all[i] = float(lv.get("volatility", 0.4))
            collab_all[i] = collab_style
            ag = lv.get("avoid_genres")
            if isinstance(ag, list) and ag:
                avoid_genres_sparse[int(pid)] = set(ag)

            # A3-FIX: build bit-vectors in the same loop
            for g in ga_parsed:
                genre_bits[i] |= genre_to_bit.get(str(g).lower(), 0)
            for s in st_parsed:
                style_bits[i] |= style_to_bit.get(str(s).lower(), 0)
            risk_arr[i] = risk_val
            ambition_arr[i] = ambition_val
            collab_code_arr[i] = collab_labels.get(collab_style, 2)

        self._person_sim_cache = cache
        self._latent_pid_to_idx = {int(pid): i for i, pid in enumerate(pids)}
        self._latent_csv_normed = csv_all
        self._latent_bbp_normed = bbp_all
        self._latent_controversy = controversy_all
        self._latent_volatility = volatility_all
        self._latent_collab = collab_all
        self._latent_avoid_genres = avoid_genres_sparse

        self._latent_genre_bits = genre_bits
        self._latent_style_bits = style_bits
        self._latent_risk = risk_arr
        self._latent_ambition = ambition_arr
        self._latent_collab_code = collab_code_arr
        print(f"  Built person similarity cache ({len(cache)} entries, {int(genre_bits.astype(bool).sum())} with genre bits, {int(style_bits.astype(bool).sum())} with style bits)")

    def _prewarm_year_cache(self):
        assert self.actors is not None
        actors = self.actors
        if actors.empty:
            self._year_cache = {}
            return

        has_debut = "debut_year" in actors.columns
        has_retire = "retirement_year" in actors.columns
        if YEAR_RANGE is not None:
            yr_lo, yr_hi = YEAR_RANGE
        elif has_debut:
            yr_lo = max(1970, int(pd.to_numeric(actors["debut_year"], errors="coerce").fillna(1970).min()))
            yr_hi = min(2035, int(pd.to_numeric(actors["retirement_year"], errors="coerce").fillna(2035).max())) if has_retire else 2030
        else:
            return

        # P2-FIX: pre-compute lowered string columns ONCE on master DataFrame
        # instead of per-year on each copy.
        if "genre_affinity" in actors.columns:
            actors["_ga_lower"] = actors["genre_affinity"].fillna("").astype(str).str.lower()
        if "style_tags" in actors.columns:
            actors["_st_lower"] = actors["style_tags"].fillna("").astype(str).str.lower()

        # P2-FIX: store boolean masks instead of full DataFrame copies.
        # Each mask is ~N bytes vs ~N×500 bytes per copy.
        # Cache access: `self.actors[self._year_cache[year]]`
        self._year_cache = {}
        debut_arr = pd.to_numeric(actors["debut_year"], errors="coerce").fillna(1970).astype(int).to_numpy() if has_debut else None
        retire_arr = pd.to_numeric(actors["retirement_year"], errors="coerce").fillna(2100).astype(int).to_numpy() if has_retire else None

        full_mask = np.ones(len(actors), dtype=bool)
        for year in range(int(yr_lo), int(yr_hi) + 1):
            if debut_arr is not None and retire_arr is not None:
                mask = (debut_arr <= year) & (retire_arr >= year)
                if mask.sum() < 10:
                    mask = full_mask
            else:
                mask = full_mask
            self._year_cache[year] = mask

        print(f"  Pre-warmed year cache for {len(self._year_cache)} years ({yr_lo}-{yr_hi})")
