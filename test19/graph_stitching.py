from __future__ import annotations

"""
Rewritten graph_stitching.py
============================

Unified typed edge graph with:
- batch ingestion from raw LLM outputs
- contradiction marking / deduplication
- percentile-based weight calibration
- optional weak-edge inference
- community detection
- fast affinity-index construction and incremental updates

The public API is kept compatible with the previous pipeline.
"""

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from contracts import (
    EDGE_SIGNS,
    EDGE_SOURCE_KINDS,
    EDGE_TYPES,
    GENRES,
    STYLE_TAGS,
    levenshtein,
    load_json_batch,
    normalize_name,
    save_json_batch,
    validate_edge,
)


# ============================================================================
# EdgeGraph
# ============================================================================

class EdgeGraph:
    """Unified typed edge table with merge, calibration, inference, and indexing."""

    UNDIRECTED_TYPES = {"friendship", "rivalry", "avoid", "collaboration", "chemistry", "market_rival", "co_production", "subsidiary"}

    _EDGE_TYPE_TO_BUCKET = {
        "friendship": "friendship",
        "collaboration": "friendship",
        "chemistry": "friendship",
        "clique": "friendship",
        "former_collaborator": "friendship",
        "rivalry": "rivalry",
        "mentorship": "mentorship",
        "preferred": "mentorship",
        "avoid": "avoid",
        "blacklist": "avoid",
        "employment": "person_company_affinity",
        "brand_fit": "person_company_affinity",
        "exclusive_deal": "person_company_affinity",
        "co_production": "company_affinity",
        "subsidiary": "company_affinity",
        "market_rival": "company_rivalry",
    }

    def __init__(self):
        self.edges: List[dict] = []
        self.name_to_id: Dict[str, int] = {}
        self.id_to_name: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Name registry
    # ------------------------------------------------------------------

    def register_name(self, name: str) -> int:
        norm = normalize_name(name)
        if norm not in self.name_to_id:
            new_id = len(self.name_to_id) + 1
            self.name_to_id[norm] = new_id
            self.id_to_name[new_id] = norm
        return self.name_to_id[norm]

    def resolve_name(self, name: str) -> Optional[int]:
        norm = normalize_name(name)
        if norm in self.name_to_id:
            return self.name_to_id[norm]
        best_match, best_dist = None, 999
        for registered in self.name_to_id:
            d = levenshtein(norm.lower(), registered.lower())
            if d < best_dist:
                best_dist = d
                best_match = registered
        if best_match is not None and best_dist <= 3:
            return self.name_to_id[best_match]
        return None

    # ------------------------------------------------------------------
    # Canonicalization / helpers
    # ------------------------------------------------------------------

    @classmethod
    def canonical_pair(cls, src_id: int, dst_id: int, edge_type: str) -> Tuple[int, int]:
        if edge_type in cls.UNDIRECTED_TYPES:
            return (min(src_id, dst_id), max(src_id, dst_id))
        return (src_id, dst_id)

    @classmethod
    def edge_key(cls, edge: dict) -> Tuple[int, int, str, str]:
        src, dst = cls.canonical_pair(int(edge.get("src_id", 0)), int(edge.get("dst_id", 0)), str(edge.get("edge_type", "")))
        return (src, dst, str(edge.get("edge_type", "")), str(edge.get("sign", "+")))

    def _deduplicate_in_place(self):
        best: Dict[Tuple[int, int, str, str], dict] = {}
        for e in self.edges:
            key = self.edge_key(e)
            cur = best.get(key)
            if cur is None or float(e.get("weight", 0.0) or 0.0) > float(cur.get("weight", 0.0) or 0.0):
                src, dst = self.canonical_pair(int(e.get("src_id", 0)), int(e.get("dst_id", 0)), str(e.get("edge_type", "")))
                e["src_id"] = src
                e["dst_id"] = dst
                best[key] = e
        removed = max(0, len(self.edges) - len(best))
        self.edges = list(best.values())
        if removed:
            print(f"[Dedup] removed {removed} duplicate edges")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_batch(self, edges: List[dict], batch_id: str, source_kind: str = "llm"):
        valid = 0
        invalid = 0
        for raw in edges:
            errs = validate_edge(raw)
            if errs:
                invalid += 1
                continue
            src_id = self.register_name(raw["src"])
            dst_id = self.register_name(raw["dst"])
            if src_id == dst_id:
                invalid += 1
                continue
            edge_type = str(raw["edge_type"])
            src_id, dst_id = self.canonical_pair(src_id, dst_id, edge_type)
            self.edges.append(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "edge_type": edge_type,
                    "sign": raw.get("sign", "+" if edge_type == "friendship" else "-"),
                    "raw_weight": float(raw.get("weight", 0.5)),
                    "weight": float(raw.get("weight", 0.5)),
                    "reason": raw.get("reason", ""),
                    "source_batch": batch_id,
                    "source_kind": source_kind,
                    "valid_from": raw.get("valid_from"),
                    "valid_to": raw.get("valid_to"),
                }
            )
            valid += 1
        print(f"[Batch {batch_id}] Ingested {valid} edges ({invalid} rejected)")

    # ------------------------------------------------------------------
    # Contradictions / calibration
    # ------------------------------------------------------------------

    def resolve_contradictions(self):
        pair_edges: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
        for e in self.edges:
            src = int(e.get("src_id", 0))
            dst = int(e.get("dst_id", 0))
            pair_edges[(min(src, dst), max(src, dst))].append(e)

        contradictions = 0
        for edges in pair_edges.values():
            types = {e.get("edge_type") for e in edges}
            if "friendship" in types and "rivalry" in types:
                contradictions += 1
                for e in edges:
                    if e.get("edge_type") == "friendship":
                        e["_overridden_by_rivalry"] = True
        print(f"[Contradiction resolution] {contradictions} friend/rival conflicts resolved")

    def calibrate_weights(self, global_range: Tuple[float, float] = (0.1, 0.95)):
        lo, hi = global_range
        groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for e in self.edges:
            groups[(str(e.get("source_batch", "")), str(e.get("edge_type", "")))].append(e)

        for _, group_edges in groups.items():
            if len(group_edges) <= 1:
                for e in group_edges:
                    e["weight"] = (lo + hi) / 2.0
                continue
            sorted_edges = sorted(group_edges, key=lambda x: float(x.get("raw_weight", 0.0) or 0.0))
            n = len(sorted_edges)
            for rank, e in enumerate(sorted_edges):
                pct = rank / max(1, n - 1)
                e["weight"] = float(lo + pct * (hi - lo))
        print(f"[Calibration] Normalized {len(self.edges)} edges across {len(groups)} (batch, type) groups")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer_tag_similarity_edges(self, persons: List[dict], threshold: int = 3, max_edges: Optional[int] = None):
        import hashlib

        pid_map: Dict[int, int] = {}
        debut_map: Dict[int, Optional[int]] = {}
        for i, p in enumerate(persons):
            pid = self.resolve_name(p.get("name", ""))
            if pid is not None:
                pid_map[i] = pid
                debut_map[pid] = p.get("debut_year") or p.get("year_debut")

        tag_index: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(persons):
            pid = pid_map.get(i)
            if pid is None:
                continue
            tags = p.get("style_tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.replace(";", ",").split(",") if t.strip()]
            for tag in set(tags):
                tag_index[str(tag)].append(pid)

        existing = set()
        for e in self.edges:
            if str(e.get("edge_type")) in {"friendship", "rivalry"}:
                existing.add((min(int(e["src_id"]), int(e["dst_id"])), max(int(e["src_id"]), int(e["dst_id"]))))

        pair_shared: Dict[Tuple[int, int], set] = defaultdict(set)
        for tag, pids in tag_index.items():
            if len(pids) < 2:
                continue
            bucket = pids[:200]
            for ai in range(len(bucket)):
                for bi in range(ai + 1, len(bucket)):
                    a, b = bucket[ai], bucket[bi]
                    key = (min(a, b), max(a, b))
                    if key not in existing:
                        pair_shared[key].add(tag)

        qualifying = [(pair, shared) for pair, shared in pair_shared.items() if len(shared) >= threshold]
        qualifying.sort(key=lambda x: -len(x[1]))
        if max_edges is not None:
            qualifying = qualifying[:max_edges]

        count = 0
        for (src, dst), shared in qualifying:
            shared_sample = list(shared)[:3]
            debut_i = debut_map.get(src)
            debut_j = debut_map.get(dst)
            age_proximate = debut_i is not None and debut_j is not None and abs(int(debut_i) - int(debut_j)) <= 10
            w = 0.25 if age_proximate else 0.20
            reasons = [
                f"Shared aesthetic: {', '.join(shared_sample)}",
                f"Overlapping style profile ({len(shared)} common tags)",
                f"Affinity via {shared_sample[0]} style similarity",
                f"Similar creative sensibility: {', '.join(shared_sample)}",
            ]
            ridx = int(hashlib.md5(f"{src}_{dst}".encode()).hexdigest(), 16) % len(reasons)
            self.edges.append(
                {
                    "src_id": src,
                    "dst_id": dst,
                    "edge_type": "friendship",
                    "sign": "+",
                    "raw_weight": w,
                    "weight": w,
                    "reason": reasons[ridx],
                    "source_batch": "inferred",
                    "source_kind": "inferred_tag",
                    "valid_from": None,
                    "valid_to": None,
                }
            )
            existing.add((src, dst))
            count += 1
        print(f"[Tag inference] Added {count} weak friendship edges")

    def infer_transitive_edges(self, max_weight: float = 0.15):
        friendships: Dict[int, set] = defaultdict(set)
        existing = set()
        for e in self.edges:
            if str(e.get("edge_type")) == "friendship" and not e.get("_overridden_by_rivalry"):
                src, dst = int(e["src_id"]), int(e["dst_id"])
                friendships[src].add(dst)
                friendships[dst].add(src)
                existing.add((min(src, dst), max(src, dst)))

        count = 0
        for node, friends in friendships.items():
            for f1 in friends:
                for f2 in friendships.get(f1, set()):
                    if f2 == node:
                        continue
                    src, dst = min(node, f2), max(node, f2)
                    if (src, dst) in existing:
                        continue
                    self.edges.append(
                        {
                            "src_id": src,
                            "dst_id": dst,
                            "edge_type": "friendship",
                            "sign": "+",
                            "raw_weight": max_weight,
                            "weight": max_weight,
                            "reason": f"Transitive via {self.id_to_name.get(f1, f1)}",
                            "source_batch": "inferred",
                            "source_kind": "inferred_transitive",
                            "valid_from": None,
                            "valid_to": None,
                        }
                    )
                    existing.add((src, dst))
                    count += 1
        print(f"[Transitive inference] Added {count} weak edges via transitive closure")

    def ingest_tv_coappearances(self, episode_cast: List[dict], min_coappearances: int = 2, max_edges: int = 3000) -> int:
        episode_to_persons: Dict[Any, List[int]] = defaultdict(list)
        for row in episode_cast:
            ep_id = row.get("episode_id")
            pid = row.get("person_id")
            if ep_id is None or pid is None:
                continue
            try:
                episode_to_persons[ep_id].append(int(pid))
            except Exception:
                continue

        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for persons in episode_to_persons.values():
            pids = list(set(persons))
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    key = (min(pids[i], pids[j]), max(pids[i], pids[j]))
                    pair_counts[key] += 1

        existing = set()
        for e in self.edges:
            if str(e.get("edge_type")) in {"friendship", "rivalry"}:
                existing.add((min(int(e["src_id"]), int(e["dst_id"])), max(int(e["src_id"]), int(e["dst_id"]))))

        qualifying = [(cnt, pair) for pair, cnt in pair_counts.items() if cnt >= min_coappearances and pair not in existing]
        qualifying.sort(reverse=True)
        added = 0
        for count, (src, dst) in qualifying[:max_edges]:
            if count >= 10:
                weight, reason = 0.40, f"TV series regulars: {count} shared episodes"
            elif count >= 4:
                weight, reason = 0.25, f"Recurring TV co-stars: {count} shared episodes"
            else:
                weight, reason = 0.15, f"TV co-appearance: {count} shared episodes"
            self.edges.append(
                {
                    "src_id": src,
                    "dst_id": dst,
                    "edge_type": "friendship",
                    "sign": "+",
                    "raw_weight": weight,
                    "weight": weight,
                    "reason": reason,
                    "source_batch": "tv_coappearance",
                    "source_kind": "procedural_tv",
                    "valid_from": None,
                    "valid_to": None,
                }
            )
            existing.add((src, dst))
            added += 1
        print(f"[D3 TV co-appearance] {added} friendship edges added")
        return added

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------

    def detect_communities(self, min_communities: int = 8, max_communities: int = 14, resolution: float = 1.0) -> Dict[int, int]:
        all_nodes = set(self.id_to_name.keys())
        friendship_edges = []
        for e in self.edges:
            if str(e.get("edge_type")) == "friendship" and not e.get("_overridden_by_rivalry"):
                friendship_edges.append(e)
                all_nodes.add(int(e["src_id"]))
                all_nodes.add(int(e["dst_id"]))

        labels: Dict[int, int]
        used_louvain = False
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities
            G = nx.Graph()
            G.add_nodes_from(all_nodes)
            for e in friendship_edges:
                u, v = int(e["src_id"]), int(e["dst_id"])
                w = float(e.get("weight", 1.0) or 1.0)
                if G.has_edge(u, v):
                    G[u][v]["weight"] += w
                else:
                    G.add_edge(u, v, weight=w)
            part = louvain_communities(G, seed=42, resolution=resolution)
            labels = {}
            for cid, members in enumerate(part):
                for pid in members:
                    labels[int(pid)] = cid
            for node in all_nodes:
                labels.setdefault(int(node), 0)
            used_louvain = True
        except Exception:
            adj: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
            for e in friendship_edges:
                u, v = int(e["src_id"]), int(e["dst_id"])
                w = float(e.get("weight", 1.0) or 1.0)
                adj[u][v] += w
                adj[v][u] += w
            labels = {int(n): int(n) for n in all_nodes}
            rng = np.random.RandomState(42)
            nodes = list(all_nodes)
            for _ in range(50):
                rng.shuffle(nodes)
                changed = 0
                for node in nodes:
                    if node not in adj or len(adj[node]) == 0:
                        continue
                    votes = defaultdict(float)
                    for nb, w in adj[node].items():
                        votes[labels[nb]] += w
                    if votes:
                        best = max(votes, key=votes.get)
                        if labels[node] != best:
                            labels[node] = best
                            changed += 1
                if changed == 0:
                    break
            remap = {old: i for i, old in enumerate(sorted(set(labels.values())))}
            labels = {n: remap[c] for n, c in labels.items()}

        sizes = Counter(labels.values())
        method = "Louvain" if used_louvain else "label-propagation"
        print(f"[Community detection] {len(sizes)} communities [{method}]")
        for comm_id, cnt in sizes.most_common(10):
            pct = 100.0 * cnt / max(1, len(all_nodes))
            print(f"  Community {comm_id}: {cnt} members ({pct:.1f}%)")
        return labels

    # ------------------------------------------------------------------
    # Diagnostics / export
    # ------------------------------------------------------------------

    def diagnostics(self):
        print("\n" + "=" * 60)
        print("  GRAPH DIAGNOSTICS")
        print("=" * 60)
        print(f"  Total edges: {len(self.edges)}")
        print(f"  Total nodes: {len(self.id_to_name)}")
        by_type = Counter(str(e.get("edge_type")) for e in self.edges)
        for et, cnt in by_type.most_common():
            print(f"  {et}: {cnt}")
        by_source = Counter(str(e.get("source_kind")) for e in self.edges)
        print("\n  By source:")
        for src, cnt in by_source.most_common():
            print(f"    {src}: {cnt}")

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for e in self.edges:
            rows.append(
                {
                    "src_id": e.get("src_id"),
                    "src_name": self.id_to_name.get(int(e.get("src_id", 0)), "?"),
                    "dst_id": e.get("dst_id"),
                    "dst_name": self.id_to_name.get(int(e.get("dst_id", 0)), "?"),
                    "edge_type": e.get("edge_type"),
                    "sign": e.get("sign"),
                    "weight": e.get("weight"),
                    "raw_weight": e.get("raw_weight"),
                    "reason": e.get("reason"),
                    "source_batch": e.get("source_batch"),
                    "source_kind": e.get("source_kind"),
                    "valid_from": e.get("valid_from"),
                    "valid_to": e.get("valid_to"),
                }
            )
        return pd.DataFrame(rows)

    def save_csv(self, filepath: str):
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} edges to {filepath}")

    # ------------------------------------------------------------------
    # Legacy lookup helpers
    # ------------------------------------------------------------------

    def get_friendship_weight(self, id_a: int, id_b: int) -> float:
        key = (min(int(id_a), int(id_b)), max(int(id_a), int(id_b)))
        for e in self.edges:
            if (int(e.get("src_id", 0)), int(e.get("dst_id", 0))) == key and str(e.get("edge_type")) == "friendship" and not e.get("_overridden_by_rivalry"):
                return float(e.get("weight", 0.0) or 0.0)
        return 0.0

    def is_rival(self, id_a: int, id_b: int) -> bool:
        key = (min(int(id_a), int(id_b)), max(int(id_a), int(id_b)))
        return any((int(e.get("src_id", 0)), int(e.get("dst_id", 0))) == key and str(e.get("edge_type")) == "rivalry" for e in self.edges)

    def get_director_preferred_actors(self, director_id: int) -> List[Tuple[int, float]]:
        out = []
        did = int(director_id)
        for e in self.edges:
            if str(e.get("edge_type")) == "mentorship" and str(e.get("sign", "+")) == "+":
                if int(e.get("src_id", 0)) == did:
                    out.append((int(e.get("dst_id", 0)), float(e.get("weight", 0.0) or 0.0)))
                elif int(e.get("dst_id", 0)) == did:
                    out.append((int(e.get("src_id", 0)), float(e.get("weight", 0.0) or 0.0)))
        return out

    def get_director_avoided_actors(self, director_id: int) -> List[int]:
        out = []
        did = int(director_id)
        for e in self.edges:
            if str(e.get("edge_type")) == "avoid":
                if int(e.get("src_id", 0)) == did:
                    out.append(int(e.get("dst_id", 0)))
                elif int(e.get("dst_id", 0)) == did:
                    out.append(int(e.get("src_id", 0)))
        return out

    # ------------------------------------------------------------------
    # Affinity index / incremental updates
    # ------------------------------------------------------------------

    def build_affinity_index(self) -> dict:
        index = {
            "friendships": {},
            "rivalries": {},
            "director_prefs": defaultdict(list),
            "director_avoids": defaultdict(list),
            "company_affinity": {},
            "company_rivalry": {},
            "person_company_affinity": defaultdict(list),
        }

        for e in self.edges:
            if e.get("_scd2_retired", False):
                continue
            edge_type = str(e.get("edge_type", ""))
            bucket = self._EDGE_TYPE_TO_BUCKET.get(edge_type)
            if bucket is None:
                continue
            src = int(e.get("src_id", 0))
            dst = int(e.get("dst_id", 0))
            payload = {
                "weight": float(e.get("weight", 0.0) or 0.0),
                "valid_from": e.get("valid_from"),
                "valid_to": e.get("valid_to"),
            }

            if bucket == "friendship" and not e.get("_overridden_by_rivalry"):
                key = (min(src, dst), max(src, dst))
                old = index["friendships"].get(key)
                if old is None or payload["weight"] > float(old.get("weight", 0.0)):
                    index["friendships"][key] = payload
            elif bucket == "rivalry":
                key = (min(src, dst), max(src, dst))
                old = index["rivalries"].get(key)
                if old is None or payload["weight"] > float(old.get("weight", 0.0)):
                    index["rivalries"][key] = payload
            elif bucket == "mentorship" and str(e.get("sign", "+")) != "-":
                index["director_prefs"][src].append({"actor_id": dst, **payload})
                index["director_prefs"][dst].append({"actor_id": src, **payload})
            elif bucket == "avoid":
                index["director_avoids"][src].append({"actor_id": dst, "valid_from": payload["valid_from"], "valid_to": payload["valid_to"]})
                index["director_avoids"][dst].append({"actor_id": src, "valid_from": payload["valid_from"], "valid_to": payload["valid_to"]})
            elif bucket == "company_affinity":
                key = (min(src, dst), max(src, dst))
                old = index["company_affinity"].get(key)
                if old is None or payload["weight"] > float(old.get("weight", 0.0)):
                    index["company_affinity"][key] = payload
            elif bucket == "company_rivalry":
                key = (min(src, dst), max(src, dst))
                old = index["company_rivalry"].get(key)
                if old is None or payload["weight"] > float(old.get("weight", 0.0)):
                    index["company_rivalry"][key] = payload
            elif bucket == "person_company_affinity":
                index["person_company_affinity"][src].append({"company_id": dst, **payload})
                index["person_company_affinity"][dst].append({"company_id": src, **payload})

        print(
            f"[Affinity index] {len(index['friendships'])} friendships, {len(index['rivalries'])} rivalries, "
            f"{len(index['director_prefs'])} directors with prefs, {len(index['company_affinity'])} company affinities, "
            f"{len(index['company_rivalry'])} company rivalries, {len(index['person_company_affinity'])} persons with company edges"
        )
        return index

    def index_add_edge(self, index: dict, edge: dict) -> None:
        if index is None:
            return
        edge_type = str(edge.get("edge_type", ""))
        bucket = self._EDGE_TYPE_TO_BUCKET.get(edge_type)
        if bucket is None:
            return
        src = int(edge.get("src_id", 0))
        dst = int(edge.get("dst_id", 0))
        payload = {"weight": float(edge.get("weight", 0.0) or 0.0), "valid_from": edge.get("valid_from"), "valid_to": edge.get("valid_to")}

        if bucket == "friendship" and not edge.get("_overridden_by_rivalry"):
            key = (min(src, dst), max(src, dst))
            old = index["friendships"].get(key)
            if old is None or payload["weight"] > float(old.get("weight", 0.0)):
                index["friendships"][key] = payload
        elif bucket == "rivalry":
            key = (min(src, dst), max(src, dst))
            old = index["rivalries"].get(key)
            if old is None or payload["weight"] > float(old.get("weight", 0.0)):
                index["rivalries"][key] = payload
        elif bucket == "mentorship" and str(edge.get("sign", "+")) != "-":
            index["director_prefs"][src].append({"actor_id": dst, **payload})
            index["director_prefs"][dst].append({"actor_id": src, **payload})
        elif bucket == "avoid":
            index["director_avoids"][src].append({"actor_id": dst, "valid_from": payload["valid_from"], "valid_to": payload["valid_to"]})
            index["director_avoids"][dst].append({"actor_id": src, "valid_from": payload["valid_from"], "valid_to": payload["valid_to"]})
        elif bucket == "company_affinity":
            key = (min(src, dst), max(src, dst))
            old = index["company_affinity"].get(key)
            if old is None or payload["weight"] > float(old.get("weight", 0.0)):
                index["company_affinity"][key] = payload
        elif bucket == "company_rivalry":
            key = (min(src, dst), max(src, dst))
            old = index["company_rivalry"].get(key)
            if old is None or payload["weight"] > float(old.get("weight", 0.0)):
                index["company_rivalry"][key] = payload
        elif bucket == "person_company_affinity":
            index["person_company_affinity"][src].append({"company_id": dst, **payload})
            index["person_company_affinity"][dst].append({"company_id": src, **payload})

    def index_expire_edge(self, index: dict, src_id: int, dst_id: int, edge_type: str) -> None:
        if index is None:
            return
        bucket = self._EDGE_TYPE_TO_BUCKET.get(edge_type)
        if bucket is None:
            return
        src = int(src_id)
        dst = int(dst_id)
        if bucket == "friendship":
            index["friendships"].pop((min(src, dst), max(src, dst)), None)
        elif bucket == "rivalry":
            index["rivalries"].pop((min(src, dst), max(src, dst)), None)
        elif bucket == "mentorship":
            for node, other in ((src, dst), (dst, src)):
                lst = index["director_prefs"].get(node)
                if lst:
                    index["director_prefs"][node] = [e for e in lst if int(e.get("actor_id", -1)) != other]
        elif bucket == "avoid":
            for node, other in ((src, dst), (dst, src)):
                lst = index["director_avoids"].get(node)
                if lst:
                    index["director_avoids"][node] = [e for e in lst if int(e.get("actor_id", -1)) != other]
        elif bucket == "company_affinity":
            index["company_affinity"].pop((min(src, dst), max(src, dst)), None)
        elif bucket == "company_rivalry":
            index["company_rivalry"].pop((min(src, dst), max(src, dst)), None)
        elif bucket == "person_company_affinity":
            for node, other in ((src, dst), (dst, src)):
                lst = index["person_company_affinity"].get(node)
                if lst:
                    index["person_company_affinity"][node] = [e for e in lst if int(e.get("company_id", -1)) != other]


# ============================================================================
# Pipeline helper
# ============================================================================

def stitch_graph(batch_files: List[str], persons: Optional[List[dict]] = None, episode_cast: Optional[List[dict]] = None) -> EdgeGraph:
    graph = EdgeGraph()
    if persons:
        for p in persons:
            if isinstance(p, dict) and p.get("name"):
                graph.register_name(str(p["name"]))

    for bf in batch_files:
        edges = load_json_batch(bf)
        graph.ingest_batch(edges, batch_id=Path(bf).stem)

    graph.resolve_contradictions()
    graph.calibrate_weights()
    graph._deduplicate_in_place()

    if episode_cast:
        graph.ingest_tv_coappearances(episode_cast)
    if persons:
        graph.infer_tag_similarity_edges(persons)
        graph.infer_transitive_edges()

    graph.diagnostics()
    return graph
