from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from llm_provider import get_llm_client
load_dotenv(Path(__file__).parent.parent / ".env")


RUNNER = "overnight_10k_runner"
CKPT_SUBDIR = Path("checkpoints") / "overnight_10k"
STATE_FILE = "state.json"
REPORT_FILE = "run_report.json"

DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite-preview"

COST_PATTERNS = [
    re.compile(r"Total\s+cost\s*[:=]\s*\$\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\bCost\s*[:=]\s*\$\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\brun\s+cost\s*\$\s*([0-9]+(?:\.[0-9]+)?)", re.I),
]

PERSON_EDGE_TYPES = {
    "friendship",
    "rivalry",
    "mentorship",
    "avoid",
    "collaboration",
    "chemistry",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def count_json_items(path: Path) -> int:
    data = read_json(path, [])
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return len(data)
    return 0


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def parse_json_loose(text: str) -> Any:
    t = str(text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) > 1:
            t = "\n".join(lines[1:])
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    t = t.strip()
    for x in (t, t[t.find("[") : t.rfind("]") + 1] if ("[" in t and "]" in t) else ""):
        if not x:
            continue
        try:
            return json.loads(x)
        except Exception:
            pass
    raise ValueError("cannot parse JSON")


def gini(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = np.clip(arr, 1e-12, None)
    arr = np.sort(arr)
    idx = np.arange(1, arr.size + 1, dtype=float)
    return float((2 * np.sum(idx * arr) - (arr.size + 1) * np.sum(arr)) / (arr.size * np.sum(arr)))


def _stable_uniform(*parts: Any) -> float:
    h = 2166136261
    for p in parts:
        s = str(p)
        for ch in s:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
    return (h % 1_000_000) / 1_000_000.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base = Path(args.base_dir).resolve()
        self.ent = self.base / "entities"
        self.graph = self.base / "graph"
        self.ckpt = self.base / CKPT_SUBDIR
        self.state_path = self.ckpt / STATE_FILE
        self.report_path = self.ckpt / REPORT_FILE
        self.stage_dir = self.ckpt / "stages"

        self.runtime = {
            "include_plots": bool(args.include_plots),
            "enable_llm_temporal_evolution": bool(args.enable_llm_temporal_evolution),
            "quality_autofix_attempted": False,
        }
        self.stage_cost: dict[str, float] = {}
        self.state: dict[str, Any] = {}
        self.inventory_before: dict[str, Any] = {}

    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    def init_state(self):
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

        if self.args.resume and self.state_path.exists():
            self.state = read_json(self.state_path, {})
            if not self.state:
                raise RuntimeError("resume requested but state.json unreadable")
            self.runtime.update(self.state.get("runtime", {}))

            art = self.state.get("artifacts", {}) if isinstance(self.state, dict) else {}
            inv0 = art.get("inventory", {}) if isinstance(art, dict) else {}
            if isinstance(inv0, dict):
                self.inventory_before = dict(inv0)

            self.stage_cost = dict(self.state.get("stage_cost_estimates", {}) or {})
            if not self.stage_cost:
                bp = art.get("budget_plan") if isinstance(art, dict) else None
                if bp:
                    bpj = read_json(Path(bp), {})
                    if isinstance(bpj, dict):
                        self.stage_cost = dict(bpj.get("stage_cost_estimates", {}) or {})

            return

        self.state = {
            "runner": RUNNER,
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": now_utc(),
            "updated_at": now_utc(),
            "status": "running",
            "final_status": None,
            "config": {
                "base_dir": str(self.base),
                "target_movies": int(self.args.target_movies),
                "budget_usd": float(self.args.budget_usd),
                "price_per_million": float(self.args.price_per_million),
                "strict_job_export": bool(self.args.strict_job_export),
                "include_extras": bool(self.args.include_extras),
                "preflight_only": bool(self.args.preflight_only),
                "llm_model": str(self.args.llm_model),
            },
            "runtime": dict(self.runtime),
            "completed": [],
            "stages": {},
            "budget": {
                "cap_usd": float(self.args.budget_usd),
                "spent_estimated_usd": 0.0,
                "spent_observed_usd": 0.0,
                "core_projected_usd": None,
                "optional_projected_usd": None,
                "fallback_policy": "core_only",
            },
            "artifacts": {},
            "restart_from_stage": 1,
        }
        self.save_state()

    def save_state(self):
        self.state["updated_at"] = now_utc()
        self.state["runtime"] = dict(self.runtime)
        write_json(self.state_path, self.state)

    def marker(self, idx: int, slug: str) -> Path:
        return self.stage_dir / f"{idx:02d}_{slug}.done.json"

    def done(self, idx: int, slug: str) -> bool:
        return self.marker(idx, slug).exists()

    def mark_start(self, idx: int, slug: str, name: str):
        self.state["stages"].setdefault(slug, {})
        self.state["stages"][slug].update({
            "idx": idx,
            "name": name,
            "status": "in_progress",
            "started_at": now_utc(),
        })
        self.state["restart_from_stage"] = idx
        self.save_state()

    def mark_done(self, idx: int, slug: str, result: dict[str, Any], elapsed: float):
        self.state["stages"].setdefault(slug, {})
        self.state["stages"][slug].update({
            "status": "completed",
            "finished_at": now_utc(),
            "elapsed_sec": round(elapsed, 3),
            "result": result,
        })
        if idx not in self.state["completed"]:
            self.state["completed"].append(idx)
            self.state["completed"] = sorted(self.state["completed"])
        self.state["restart_from_stage"] = idx + 1
        write_json(self.marker(idx, slug), {"result": result, "finished_at": now_utc()})
        self.save_state()

    def mark_fail(self, idx: int, slug: str, err: Exception, elapsed: float):
        self.state["stages"].setdefault(slug, {})
        self.state["stages"][slug].update({
            "status": "failed",
            "finished_at": now_utc(),
            "elapsed_sec": round(elapsed, 3),
            "error": str(err),
        })
        self.state["status"] = "failed"
        self.state["final_status"] = f"failed_at_{idx:02d}_{slug}"
        self.state["restart_from_stage"] = idx
        self.save_state()

    def py(self, script: str, *args: str) -> list[str]:
        return [sys.executable, str(self.base / script), *[str(x) for x in args]]

    def run_cmd(self, cmd: list[str], stage_slug: str, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items()})

        self.log("CMD> " + " ".join(cmd))
        start = time.time()
        cost_max = 0.0
        tail: list[str] = []

        p = subprocess.Popen(
            cmd,
            cwd=str(self.base),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            while True:
                line = p.stdout.readline() if p.stdout is not None else ""
                if line == "" and p.poll() is not None:
                    break
                if not line:
                    continue
                line = line.rstrip("\n")
                print(line, flush=True)
                tail.append(line)
                if len(tail) > 250:
                    tail = tail[-250:]
                for rx in COST_PATTERNS:
                    m = rx.search(line)
                    if m:
                        try:
                            cost_max = max(cost_max, float(m.group(1)))
                        except Exception:
                            pass

            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"command failed exit={rc}: {' '.join(cmd)}")

            return {
                "elapsed_sec": round(time.time() - start, 3),
                "observed_cost_usd": round(cost_max, 4) if cost_max > 0 else None,
                "output_tail": tail[-30:],
            }
        finally:
            if p.poll() is None:
                p.kill()

    def inventory(self) -> dict[str, Any]:
        role = read_csv(self.ent / "person_roles.csv")
        person_df = read_csv(self.ent / "person.csv")

        actors = 0
        directors = 0

        if not role.empty and {"person_id", "role_type"}.issubset(role.columns):
            actors = int(role[role["role_type"] == "actor"]["person_id"].nunique())
            directors = int(role[role["role_type"] == "director"]["person_id"].nunique())

        # Fallback/consistency check from person.csv roles in case person_roles.csv is stale.
        if not person_df.empty and {"person_id", "roles"}.issubset(person_df.columns):
            tmp = person_df[["person_id", "roles"]].copy()
            tmp["person_id"] = pd.to_numeric(tmp["person_id"], errors="coerce").fillna(0).astype(int)
            tmp["roles"] = tmp["roles"].astype(str).str.lower()
            actors_csv = int(tmp[tmp["roles"].str.contains(r"\bactor\b", regex=True)]["person_id"].nunique())
            directors_csv = int(tmp[tmp["roles"].str.contains(r"\bdirector\b", regex=True)]["person_id"].nunique())
            actors = max(actors, actors_csv)
            directors = max(directors, directors_csv)

        return {
            "persons_json": count_json_items(self.ent / "persons.json"),
            "companies_json": count_json_items(self.ent / "companies.json"),
            "persons_latent_json": count_json_items(self.ent / "persons_latent.json"),
            "companies_latent_json": count_json_items(self.ent / "companies_latent.json"),
            "person_csv": count_csv_rows(self.ent / "person.csv"),
            "company_csv": count_csv_rows(self.ent / "company.csv"),
            "keyword_csv": count_csv_rows(self.ent / "keyword.csv"),
            "title_bank_csv": count_csv_rows(self.ent / "title_bank.csv"),
            "character_bank_csv": count_csv_rows(self.ent / "character_bank.csv"),
            "edge_graph_csv": count_csv_rows(self.graph / "edge_graph.csv"),
            "agencies_json": count_json_items(self.ent / "agencies.json"),
            "cliques_json": count_json_items(self.ent / "cliques.json"),
            "actors_available": actors,
            "directors_available": directors,
            "movie_rows": count_csv_rows(self.base / "movie.csv"),
            "cast_rows": count_csv_rows(self.base / "cast_info.csv"),
        }

    def stage_cost_est(self, slug: str) -> float:
        return float(self.stage_cost.get(slug, 0.0) or 0.0)

    def budget_guard(self, slug: str, optional: bool = False) -> bool:
        spent = float(self.state["budget"]["spent_estimated_usd"])
        cap = float(self.state["budget"]["cap_usd"])
        est = self.stage_cost_est(slug)
        if spent + est <= cap + 1e-9:
            return True
        if optional:
            self.log(f"Budget guard skipped optional stage '{slug}' (spent={spent:.2f}, est={est:.2f}, cap={cap:.2f})")
            return False
        raise RuntimeError(f"Budget guard blocked mandatory stage '{slug}' (spent={spent:.2f}, est={est:.2f}, cap={cap:.2f})")

    def register_spend(self, slug: str, observed: float | None = None):
        self.state["budget"]["spent_estimated_usd"] = round(
            float(self.state["budget"]["spent_estimated_usd"]) + self.stage_cost_est(slug),
            4,
        )
        if observed is not None and observed >= 0:
            self.state["budget"]["spent_observed_usd"] = round(
                float(self.state["budget"]["spent_observed_usd"]) + float(observed),
                4,
            )
        self.save_state()


def _largest_component_ratio(edges_df: pd.DataFrame) -> float:
    if edges_df.empty or not {"src_id", "dst_id"}.issubset(edges_df.columns):
        return 0.0

    adj: dict[int, set[int]] = defaultdict(set)
    for r in edges_df.itertuples(index=False):
        try:
            a = int(getattr(r, "src_id"))
            b = int(getattr(r, "dst_id"))
        except Exception:
            continue
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)

    nodes = list(adj.keys())
    if not nodes:
        return 0.0

    seen: set[int] = set()
    best = 0
    for s in nodes:
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        size = 0
        while q:
            x = q.popleft()
            size += 1
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        best = max(best, size)

    return float(best / max(1, len(nodes)))


def compute_realism_metrics(ctx: Ctx) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    cast = read_csv(ctx.base / "cast_info.csv")
    persons = read_csv(ctx.ent / "person.csv")
    movies = read_csv(ctx.base / "movie.csv")
    edges = read_csv(ctx.graph / "edge_graph.csv")

    if not cast.empty and {"title_id", "person_id"}.issubset(cast.columns):
        cast_counts = cast.groupby("person_id").size().astype(float)
        per_title = cast.groupby("title_id").size().astype(float)
        metrics["cast_rows"] = int(len(cast))
        metrics["cast_unique_persons"] = int(cast_counts.shape[0])
        metrics["cast_gini"] = round(gini(cast_counts.tolist()), 4)
        metrics["max_cast_per_title"] = int(per_title.max()) if len(per_title) else 0
        metrics["p95_cast_per_title"] = float(np.quantile(per_title, 0.95)) if len(per_title) else 0.0

        if not movies.empty and {"title_id", "production_tier"}.issubset(movies.columns):
            tier_map = dict(zip(movies["title_id"], movies["production_tier"].astype(str)))
            epic_counts = [int(n) for tid, n in per_title.items() if tier_map.get(tid) == "Epic"]
            metrics["max_epic_cast"] = int(max(epic_counts)) if epic_counts else 0
        else:
            metrics["max_epic_cast"] = 0

        if not persons.empty and {"person_id", "pop_weight"}.issubset(persons.columns):
            p = persons[["person_id", "pop_weight"]].copy()
            p["person_id"] = pd.to_numeric(p["person_id"], errors="coerce").fillna(0).astype(int)
            p["pop_weight"] = pd.to_numeric(p["pop_weight"], errors="coerce").fillna(0.0)
            p["cast_count"] = p["person_id"].map(cast_counts).fillna(0.0)
            if len(p) >= 4:
                metrics["pop_cast_corr"] = round(float(p[["pop_weight", "cast_count"]].corr(method="spearman").iloc[0, 1]), 4)
            else:
                metrics["pop_cast_corr"] = 0.0

            p = p.sort_values("cast_count", ascending=False)
            top_k = max(1, int(math.ceil(len(p) * 0.01)))
            top_share = float(p.head(top_k)["cast_count"].sum() / max(1.0, p["cast_count"].sum()))
            metrics["top1pct_cast_share"] = round(top_share, 4)
    else:
        metrics.update({
            "cast_rows": 0,
            "cast_unique_persons": 0,
            "cast_gini": 0.0,
            "max_cast_per_title": 0,
            "p95_cast_per_title": 0.0,
            "max_epic_cast": 0,
            "pop_cast_corr": 0.0,
            "top1pct_cast_share": 0.0,
        })

    if not edges.empty and {"src_id", "dst_id"}.issubset(edges.columns):
        metrics["edge_rows_total"] = int(len(edges))

        person_ids = set()
        if not persons.empty and "person_id" in persons.columns:
            person_ids = set(pd.to_numeric(persons["person_id"], errors="coerce").dropna().astype(int).tolist())

        if person_ids:
            e = edges.copy()
            e["src_id"] = pd.to_numeric(e["src_id"], errors="coerce").fillna(-1).astype(int)
            e["dst_id"] = pd.to_numeric(e["dst_id"], errors="coerce").fillna(-1).astype(int)
            mask = e["src_id"].isin(person_ids) & e["dst_id"].isin(person_ids)
            if "edge_type" in e.columns:
                et = e["edge_type"].astype(str).str.lower()
                mask = mask & et.isin(PERSON_EDGE_TYPES)
            e_person = e.loc[mask].copy()
        else:
            e_person = edges.copy()

        deg = Counter()
        for r in e_person.itertuples(index=False):
            try:
                a = int(getattr(r, "src_id"))
                b = int(getattr(r, "dst_id"))
            except Exception:
                continue
            if a == b:
                continue
            deg[a] += 1
            deg[b] += 1

        degree_vals = [float(v) for v in deg.values()]
        metrics["edge_rows"] = int(len(e_person))
        metrics["edge_nodes"] = int(len(deg))
        metrics["edge_degree_gini"] = round(gini(degree_vals), 4) if degree_vals else 0.0
        metrics["edge_giant_component_ratio"] = round(_largest_component_ratio(e_person), 4) if len(e_person) > 0 else 0.0
    else:
        metrics.update({
            "edge_rows_total": 0,
            "edge_rows": 0,
            "edge_nodes": 0,
            "edge_degree_gini": 0.0,
            "edge_giant_component_ratio": 0.0,
        })

    return metrics


def evaluate_quality(metrics: dict[str, Any], strict: bool) -> dict[str, Any]:
    checks = []

    g = float(metrics.get("cast_gini", 0.0))
    corr = float(metrics.get("pop_cast_corr", 0.0))
    mx_any = int(metrics.get("max_cast_per_title", 0))
    mx_epic = int(metrics.get("max_epic_cast", 0))
    gc = float(metrics.get("edge_giant_component_ratio", 0.0))

    if strict:
        checks.append({"name": "cast_gini_range", "ok": 0.50 <= g <= 0.60, "value": g, "target": "0.50..0.60"})
        checks.append({"name": "pop_cast_corr", "ok": corr >= 0.0, "value": corr, "target": ">=0.0"})
        tail_ok = (mx_epic >= 100) or (mx_any >= 110)
        checks.append({"name": "blockbuster_tail", "ok": tail_ok, "value": {"max_epic_cast": mx_epic, "max_any_cast": mx_any}, "target": "epic>=100 or any>=110"})
        checks.append({"name": "edge_connectivity", "ok": gc >= 0.85, "value": gc, "target": ">=0.85"})
    else:
        checks.append({"name": "cast_presence", "ok": int(metrics.get("cast_rows", 0)) > 0, "value": int(metrics.get("cast_rows", 0)), "target": ">0"})
        checks.append({"name": "edges_presence", "ok": int(metrics.get("edge_rows", 0)) > 0, "value": int(metrics.get("edge_rows", 0)), "target": ">0"})

    ok_all = all(bool(c["ok"]) for c in checks)
    return {"pass": bool(ok_all), "checks": checks, "metrics": metrics}


def stage_01_inventory_contracts(ctx: Ctx) -> dict[str, Any]:
    inv = ctx.inventory()
    req = [
        ctx.ent / "persons.json",
        ctx.ent / "companies.json",
        ctx.ent / "person.csv",
        ctx.ent / "company.csv",
        ctx.ent / "keyword.csv",
        ctx.ent / "title_bank.csv",
        ctx.ent / "character_bank.csv",
    ]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError("missing files: " + ", ".join(missing))

    pcols = set(read_csv(ctx.ent / "person.csv").columns)
    ccols = set(read_csv(ctx.ent / "company.csv").columns)
    if not {"person_id", "name", "bio", "roles"}.issubset(pcols):
        raise RuntimeError("person.csv missing required columns")
    if not {"company_id", "name", "country", "tier", "description"}.issubset(ccols):
        raise RuntimeError("company.csv missing required columns")

    baseline = evaluate_quality(compute_realism_metrics(ctx), strict=False)

    ctx.inventory_before = inv
    ctx.state["artifacts"]["inventory"] = inv
    ctx.state["artifacts"]["quality_baseline"] = baseline
    ctx.save_state()

    return {"inventory": inv, "missing": missing, "quality_baseline": baseline}


def _build_demand_plan(inv: dict[str, Any], target: int, runtime_cfg: Any = None) -> dict[str, Any]:
    """V17: Extracted demand plan computation for testability and config-driven scaling.

    Parameters
    ----------
    inv : dict  — output of Ctx.inventory()
    target : int — target movie count
    runtime_cfg : Any — optional object with target_actor_load, target_director_load, etc.
    """
    tier_w = {"Epic": 0.05, "A": 0.15, "Mid": 0.40, "Indie": 0.30, "Micro": 0.10}
    # V17: Updated cast sizes to calibrated values
    tier_cast = {"Epic": 42.0, "A": 19.5, "Mid": 8.2, "Indie": 4.6, "Micro": 2.2}
    avg_cast = float(sum(tier_w[k] * tier_cast[k] for k in tier_w))
    cast_rows = int(round(target * avg_cast))

    cur_persons = int(inv.get("persons_json", 0) or inv.get("person_csv", 0))
    cur_companies = int(inv.get("companies_json", 0) or inv.get("company_csv", 0))
    cur_keywords = int(inv.get("keyword_csv", 0))
    cur_titles = int(inv.get("title_bank_csv", 0))
    cur_characters = int(inv.get("character_bank_csv", 0))
    cur_actors = int(inv.get("actors_available", 0))
    cur_directors = int(inv.get("directors_available", 0))

    # Data-aware sizing: prefer reusing existing inventory and preserve actor concentration.
    actor_share_cur = (cur_actors / max(1, cur_persons)) if cur_persons > 0 else 0.62
    actor_share_target = float(min(0.78, max(0.52, actor_share_cur or 0.62)))

    # V17: config-driven load targets
    target_actor_load = max(1.0, float(getattr(runtime_cfg, 'target_actor_load', 6.2)))
    target_director_load = max(1.0, float(getattr(runtime_cfg, 'target_director_load', 7.5)))
    target_company_load = max(1.0, float(getattr(runtime_cfg, 'target_company_load', 10.5)))

    # V17: flexible min values that scale with target for <10k runs
    req_actors = max(max(1200, int(target * 0.78)), int(math.ceil(cast_rows / target_actor_load)))
    req_directors = max(max(160, int(target * 0.06)), int(math.ceil(target / target_director_load)))
    req_persons = int(math.ceil(req_actors / actor_share_target))

    if target >= 10000:
        req_actors = max(10000, req_actors)
        req_directors = max(900, req_directors)

    company_assignments = int(round(target * 1.7))
    req_companies = max(max(140, int(target * 0.09)), int(math.ceil(company_assignments / target_company_load)))
    if target >= 10000:
        req_companies = max(900, req_companies)

    req_keywords = max(900, int(math.ceil(target * 0.24)))
    req_titles = target
    req_characters = max(max(2200, int(target * 2.5)), int(math.ceil(cast_rows * 0.42)))
    req_agencies = max(18, int(math.ceil(req_persons / 480.0)))
    req_cliques = max(8, int(math.ceil(req_companies / 85.0)))

    deficits = {
        "persons": max(0, req_persons - cur_persons),
        "actors": max(0, req_actors - cur_actors),
        "directors": max(0, req_directors - cur_directors),
        "companies": max(0, req_companies - cur_companies),
        "keywords": max(0, req_keywords - cur_keywords),
        "titles": max(0, req_titles - cur_titles),
        "characters": max(0, req_characters - cur_characters),
        "agencies": max(0, req_agencies - int(inv.get("agencies_json", 0))),
        "cliques": max(0, req_cliques - int(inv.get("cliques_json", 0))),
    }

    return {
        "target_movies": target,
        "requirements": {
            "persons": req_persons,
            "actors": req_actors,
            "directors": req_directors,
            "companies": req_companies,
            "keywords": req_keywords,
            "titles": req_titles,
            "characters": req_characters,
            "agencies": req_agencies,
            "cliques": req_cliques,
        },
        "current": {
            "persons": cur_persons,
            "actors": cur_actors,
            "directors": cur_directors,
            "companies": cur_companies,
            "keywords": cur_keywords,
            "titles": cur_titles,
            "characters": cur_characters,
            "agencies": int(inv.get("agencies_json", 0)),
            "cliques": int(inv.get("cliques_json", 0)),
        },
        "deficits": deficits,
        "hardness": {
            "projected_avg_cast_per_movie": round(avg_cast, 3),
            "projected_cast_rows": cast_rows,
            "actor_load_ratio": round(cast_rows / max(1, cur_actors), 3),
            "company_load_ratio": round(company_assignments / max(1, cur_companies), 3),
            "title_coverage": round(cur_titles / max(1, target), 3),
            "modeled_load_targets": {
                "actor": round(target_actor_load, 3),
                "director": round(target_director_load, 3),
                "company": round(target_company_load, 3),
            },
            "saturation_risk": {
                "actor_pool": "high" if cur_actors < req_actors * 0.8 else "ok",
                "company_pool": "high" if cur_companies < req_companies * 0.8 else "ok",
                "title_pool": "high" if cur_titles < target else "ok",
            },
        },
        "modeling_priors": {
            "target_actor_load": round(target_actor_load, 3),
            "target_director_load": round(target_director_load, 3),
            "target_company_load": round(target_company_load, 3),
        },
    }


def stage_02_demand_plan(ctx: Ctx) -> dict[str, Any]:
    inv = ctx.inventory()
    target = int(ctx.args.target_movies)
    # V17: delegate to extracted _build_demand_plan for testability
    runtime_cfg = getattr(getattr(ctx, 'workspace', None), 'config', None)
    runtime_cfg = getattr(runtime_cfg, 'priors', runtime_cfg)
    plan = _build_demand_plan(inv, target, runtime_cfg)

    ctx.state["demand"] = plan
    write_json(ctx.ckpt / "demand_plan.json", plan)
    ctx.state["artifacts"]["demand_plan"] = str(ctx.ckpt / "demand_plan.json")
    ctx.save_state()
    return plan


def stage_03_budget_preflight(ctx: Ctx) -> dict[str, Any]:
    d = ctx.state.get("demand", {})
    deficits = d.get("deficits", {})
    target = int(ctx.args.target_movies)
    ppm = float(ctx.args.price_per_million)

    person_gap = int(deficits.get("persons", 0))
    company_gap = int(deficits.get("companies", 0))

    person_tokens = int(person_gap * 340)
    company_tokens = int(company_gap * 520)
    remaining_tokens = int((person_gap + company_gap) * 600)
    temporal_tokens = int(target * 220) if ctx.runtime["enable_llm_temporal_evolution"] else 0
    plot_tokens = int(target * 340) if ctx.runtime["include_plots"] else 0

    core_cost = (person_tokens + company_tokens + remaining_tokens + temporal_tokens) / 1_000_000 * ppm
    optional_cost = plot_tokens / 1_000_000 * ppm

    cap = float(ctx.args.budget_usd)
    if core_cost > cap:
        raise RuntimeError(f"core projected cost ${core_cost:.2f} exceeds cap ${cap:.2f}")

    if ctx.runtime["include_plots"] and core_cost + optional_cost > cap:
        ctx.log("budget fallback: switch to core-only (skip plots)")
        ctx.runtime["include_plots"] = False

    ctx.stage_cost = {
        "llm_person_fill": round(person_tokens / 1_000_000 * ppm, 4),
        "llm_company_fill": round(company_tokens / 1_000_000 * ppm, 4),
        "llm_remaining": round(remaining_tokens / 1_000_000 * ppm, 4),
        "pipeline_core": round(temporal_tokens / 1_000_000 * ppm, 4),
        "plot_summaries": round(plot_tokens / 1_000_000 * ppm, 4),
    }

    rep = {
        "budget_cap_usd": cap,
        "projected_cost_usd": {
            "core": round(core_cost, 4),
            "optional": round(optional_cost, 4),
            "core_plus_optional": round(core_cost + optional_cost, 4),
        },
        "runtime": dict(ctx.runtime),
        "stage_cost_estimates": dict(ctx.stage_cost),
        "fallback_policy": "core_only",
    }

    ctx.state["stage_cost_estimates"] = dict(ctx.stage_cost)
    ctx.state["budget"]["core_projected_usd"] = rep["projected_cost_usd"]["core"]
    ctx.state["budget"]["optional_projected_usd"] = rep["projected_cost_usd"]["optional"]
    ctx.state["budget"]["fallback_policy"] = "core_only"
    write_json(ctx.ckpt / "budget_plan.json", rep)
    ctx.state["artifacts"]["budget_plan"] = str(ctx.ckpt / "budget_plan.json")
    ctx.save_state()
    return rep

def _build_name_pools(persons: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    first = []
    last = []
    for p in persons:
        parts = re.findall(r"[A-Za-z][A-Za-z'-]+", str(p.get("name", "")))
        if len(parts) >= 2:
            first.append(parts[0])
            last.append(parts[-1])
    cf = Counter([x for x in first if len(x) >= 3])
    cl = Counter([x for x in last if len(x) >= 3])
    first_pool = [w for w, _ in cf.most_common(1200)] or ["Aiden", "Mila", "Noah", "Leila", "Nora", "Aria", "Kai", "Yara"]
    last_pool = [w for w, _ in cl.most_common(1200)] or ["Kovacs", "Rahman", "Ibarra", "Sato", "Dubois", "Navarro", "Petrov", "Haddad"]
    return first_pool, last_pool


def _mk_name(rng: random.Random, fp: list[str], lp: list[str]) -> str:
    a = ["Al", "Be", "Ca", "De", "El", "Fa", "Gi", "Ha", "Io", "Ka", "Lu", "Mi", "Na", "Or", "Pa", "Ra", "Sa", "Te", "Vi", "Ya", "Zo"]
    b = ["den", "lia", "tor", "mir", "ron", "vian", "sel", "kar", "dri", "mara", "tan", "dell", "reza", "nori", "vera"]
    r = rng.random()
    if r < 0.65:
        x = f"{rng.choice(fp)} {rng.choice(lp)}"
    elif r < 0.85:
        x = f"{rng.choice(a)}{rng.choice(b)} {rng.choice(lp)}"
    else:
        x = f"{rng.choice(fp)} {rng.choice(a)}{rng.choice(b)}"
    if rng.random() < 0.12:
        x = x.replace(" ", "-", 1)
    return x


def topup_person_stubs(ctx: Ctx, n_add: int) -> dict[str, Any]:
    if n_add <= 0:
        return {"added": 0, "new_person_ids": []}

    from contracts import NATIONALITIES, GENRES

    path = ctx.ent / "persons.json"
    rows = read_json(path, [])
    if not isinstance(rows, list):
        rows = []

    used = {str(x.get("name", "")).strip().lower() for x in rows}
    fp, lp = _build_name_pools(rows)
    nat_cnt = Counter([str(x.get("nationality", "American")) for x in rows])
    nat_pool = list(nat_cnt.keys()) if nat_cnt else list(NATIONALITIES)
    nat_w = [nat_cnt.get(n, 1) for n in nat_pool] if nat_cnt else [1] * len(nat_pool)

    rng = random.Random(20260320 + len(rows))
    next_id = 1 + max([int(x.get("person_id", 0) or 0) for x in rows], default=0)

    role_options = [["actor"], ["actor", "director"], ["director"], ["producer"], ["writer"], ["cinematographer"], ["editor"], ["composer"]]
    role_w = [0.62, 0.06, 0.06, 0.07, 0.06, 0.05, 0.04, 0.04]
    stg = ["rising", "prime", "veteran", "legend", "retired"]
    stg_w = [0.34, 0.34, 0.19, 0.09, 0.04]
    g_pool = ["M", "F", "NB"]
    g_w = [0.52, 0.44, 0.04]

    new_rows = []
    for i in range(n_add):
        for _ in range(80):
            nm = _mk_name(rng, fp, lp)
            if nm.lower() not in used:
                used.add(nm.lower())
                break
        else:
            nm = f"Person_{next_id + i}"

        new_rows.append({
            "person_id": next_id + i,
            "name": nm,
            "nationality": rng.choices(nat_pool, weights=nat_w, k=1)[0],
            "gender": rng.choices(g_pool, weights=g_w, k=1)[0],
            "bio": "",
            "style_tags": [],
            "genre_affinity": [rng.choice(GENRES)],
            "career_stage": rng.choices(stg, weights=stg_w, k=1)[0],
            "roles": rng.choices(role_options, weights=role_w, k=1)[0],
            "market_fit": ["Regional"],
        })

    rows.extend(new_rows)
    write_json(path, rows)
    return {"added": len(new_rows), "new_person_ids": [int(x["person_id"]) for x in new_rows]}


def topup_keywords(ctx: Ctx, n_add: int) -> dict[str, Any]:
    if n_add <= 0:
        return {"added": 0}

    from contracts import GENRES

    path = ctx.ent / "keyword.csv"
    df = read_csv(path)
    if df.empty:
        df = pd.DataFrame(columns=["keyword_id", "keyword", "topic_genre", "pop_weight"])

    used = set(df.get("keyword", pd.Series(dtype=str)).astype(str).str.lower().tolist())
    next_id = int(pd.to_numeric(df.get("keyword_id", pd.Series([0])), errors="coerce").fillna(0).max()) + 1

    stems = [
        "uprising", "inheritance", "reckoning", "afterglow", "faultline", "undertow", "crossfire", "echo", "fracture", "redemption",
        "hinterland", "detour", "finale", "origin", "ascent", "collapse", "voyage", "paradox", "cataclysm", "rebirth",
    ]
    mods = ["urban", "silent", "neon", "winter", "royal", "digital", "deep", "parallel", "shadow", "solar", "mythic", "forgotten"]
    rng = random.Random(20260331 + len(df))

    rows = []
    attempts = 0
    while len(rows) < n_add and attempts < max(2000, n_add * 40):
        attempts += 1
        g = rng.choice(GENRES)
        kw = f"{rng.choice(mods)} {rng.choice(stems)}"
        if rng.random() < 0.18:
            kw = f"{kw} {rng.choice(['protocol', 'case', 'chronicle', 'project', 'incident'])}"
        key = kw.lower().strip()
        if key in used:
            continue
        used.add(key)
        rows.append({
            "keyword_id": next_id + len(rows),
            "keyword": kw,
            "topic_genre": g,
            "pop_weight": round(0.02 + 0.10 * _stable_uniform(kw, g), 3),
        })

    if rows:
        out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(path, index=False)

    return {"added": len(rows)}


def topup_character_bank(ctx: Ctx, n_add: int) -> dict[str, Any]:
    if n_add <= 0:
        return {"added": 0}

    from contracts import ARCHETYPES

    path = ctx.ent / "character_bank.csv"
    df = read_csv(path)
    if df.empty:
        df = pd.DataFrame(columns=["character_name", "archetype"])

    used = set(df.get("character_name", pd.Series(dtype=str)).astype(str).str.lower().tolist())

    # Expanded pools: ~150 firsts × ~150 lasts = ~22,500 base combos.
    # At 44k characters needed, collision probability is very low and
    # the suffix fallback will almost never be reached.
    first = [
        # Western / English
        "Arin", "Selma", "Kaito", "Rhea", "Nico", "Imani", "Levin", "Tara", "Marek", "Yuna",
        "Soren", "Mina", "Vera", "Noel", "Dax", "Kira", "Theo", "Zara", "Finn", "Liora",
        "Caden", "Rowan", "Elara", "Saya", "Declan", "Niamh", "Isak", "Camille", "Remy", "Jade",
        "Orion", "Lyra", "Silas", "Nadia", "Felix", "Emre", "Omar", "Freya", "Ingrid", "Wren",
        "Tobias", "Cassius", "Brendan", "Lukas", "Mateo", "Ezra", "Celeste", "Sabine", "Astrid", "Paz",
        "Lena", "Hana", "Dara", "Anais", "Isolde", "Chiara", "Sol\u00e8ne", "Zo\u00eb", "S\u00f8ren", "Cyprian",
        # African / Afro-Caribbean
        "Ade", "Fatou", "Kwame", "Amara", "Idris", "Tariq", "Malik", "Ifeoma", "Chidi", "Adaeze",
        "Emeka", "Ngozi", "Obinna", "Seun", "Tunde", "Yemi", "Kofi", "Abena", "Akosua", "Nana",
        "Fatima", "Aminata", "Moussa", "Oumar", "Bintu", "Mariama", "Seydou", "Rokia", "Bintou", "Cheikh",
        # South Asian
        "Priya", "Ravi", "Hamid", "Yosef", "Irina", "Anaya", "Arjun", "Devi", "Kavya", "Shreya",
        "Rohan", "Meera", "Vikram", "Layla", "Zainab", "Samira", "Farrukh", "Dilnoza", "Jasur", "Nilufar",
        # East / Southeast Asian
        "Kenji", "Takeshi", "Elif", "Mirela", "Selin", "Lior", "Yara", "Caio", "Kenzo", "Haruki",
        "Yuki", "Aiko", "Ren", "Hiro", "Mei", "Jian", "Wei", "Xiao", "Hyun", "Jisoo",
        "Minjun", "Seojun", "Dahyun", "Subin", "Linh", "Minh", "Thanh", "Ngan", "Kiet", "Bao",
        # Latin American
        "Ximena", "Santiago", "Valentina", "Emiliano", "Catalina", "Renata", "Lucia", "Felipe", "Isabela", "Tomas",
        # Middle Eastern
        "Yasmeen", "Khalid", "Leila", "Nour", "Rania", "Kareem", "Amir", "Cyrus", "Shirin", "Dariush",
    ]
    last = [
        # Western / European
        "Vale", "Ishida", "Khan", "Morozov", "Navarre", "Bellamy", "Sato", "Mendez", "Rahim", "Byrne",
        "Kepler", "Rossi", "Noor", "Popov", "Dawes", "Strand", "Varga", "Larue", "Inoue", "Brennan",
        "Kaya", "Dubois", "Volkov", "Lindqvist", "Reyes", "Petrov", "Holt", "Nakamura", "Johansson", "Rahman",
        "Cortez", "Vasquez", "Magnusson", "Larsen", "Ferretti", "Svensson", "Andrade", "Ozturk", "Kimura",
        "Haugen", "Mirza", "Bonnet", "Park", "Salazar", "Iwata", "Beaumont", "Chung", "Eriksson",
        "Blanc", "Hashimoto", "Graves", "Szabo", "Watanabe", "Delacroix", "Gomes", "S\u00f6derberg",
        "Koch", "Bauer", "Richter", "Hoffman", "Schneider", "Weber", "Fischer", "Meyer", "Wagner", "Braun",
        "Leroy", "Moreau", "Simon", "Laurent", "Dumont", "Girard", "Renard", "Fabre", "Dupont", "Lemaire",
        "Esposito", "Romano", "Colombo", "Ricci", "Marino", "Greco", "Bruno", "Gallo", "Conti", "Leone",
        "Novak", "Horvat", "Kovacs", "Nemeth", "Szabo", "Toth", "Varga", "Kiss", "Fekete", "Balogh",
        # African / Nollywood
        "Okafor", "Abara", "Chowdhury", "Osei", "Miele", "Boateng", "Achebe", "Okonkwo", "Nduka",
        "Abubakar", "Fonseca", "Mwangi", "Nkosi", "Diallo", "Patel", "Uchenna", "Mbeki", "Adichie",
        "Obi", "Eze", "Nwosu", "Okeke", "Amadi", "Nzeogwu", "Ijele", "Onyeka", "Uzoma", "Chukwu",
        "Mensah", "Asante", "Appiah", "Acheampong", "Owusu", "Amoah", "Nyarko", "Asare", "Ofori", "Boadu",
        # South / Central Asian
        "Tanaka", "Ferreira", "Celik", "Tcherepnin", "Hashimoto", "Mwangi", "Noor",
        "Karimov", "Nazarov", "Umarov", "Tashkentov", "Rakhimov", "Yusupov", "Mirzaev", "Holiqov",
        # East Asian
        "Chen", "Zhang", "Wang", "Liu", "Yang", "Huang", "Zhao", "Wu", "Zhou", "Sun",
        "Li", "Guo", "Lin", "Luo", "Ma", "Zhu", "He", "Hu", "Deng", "Xu",
        "Kim", "Lee", "Choi", "Jung", "Kang", "Cho", "Yoon", "Lim", "Oh", "Shin",
        # Latin American
        "Herrera", "Rojas", "Castillo", "Torres", "Morales", "Jimenez", "Ortiz", "Vargas", "Flores", "Perez",
        # Middle Eastern
        "Al-Rashid", "Mansouri", "Tehrani", "Shirazi", "Tabriz", "Sadeghi", "Ahmadi", "Hosseini", "Karimi", "Rezaei",
    ]
    titles = ["Dr.", "Capt.", "Professor", "Commander", "Inspector", "Agent", "Sister", "Brother", "Judge"]
    rng = random.Random(20260401 + len(df))

    rows = []
    attempts = 0
    max_attempts = max(n_add * 6, 10000)  # conservative: 6 tries per desired name
    while len(rows) < n_add and attempts < max_attempts:
        attempts += 1
        base = f"{rng.choice(first)} {rng.choice(last)}"
        if rng.random() < 0.25:
            base = f"{rng.choice(titles)} {base}"
        if rng.random() < 0.10:
            base = base.replace(" ", "-", 1)

        key = base.lower().strip()
        if key in used:
            continue
        used.add(key)
        rows.append({"character_name": base, "archetype": rng.choice(ARCHETYPES)})

    # Guaranteed suffix fallback: if pool exhausted, append a numeric suffix to guarantee uniqueness
    suffix_i = 1
    while len(rows) < n_add:
        base = f"{rng.choice(first)} {rng.choice(last)} {suffix_i}"
        key = base.lower().strip()
        if key not in used:
            used.add(key)
            rows.append({"character_name": base, "archetype": rng.choice(ARCHETYPES)})
        suffix_i += 1

    if rows:
        out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(path, index=False)

    return {"added": len(rows)}


def stage_04_topup_ids(ctx: Ctx) -> dict[str, Any]:
    deficits = ctx.state.get("demand", {}).get("deficits", {})

    p = topup_person_stubs(ctx, int(deficits.get("persons", 0)))
    k = topup_keywords(ctx, int(deficits.get("keywords", 0)))
    ch = topup_character_bank(ctx, int(deficits.get("characters", 0)))

    t = ctx.run_cmd(
        ctx.py("generate_titles_llm.py", "--base-dir", str(ctx.base), "--target-count", str(ctx.args.target_movies)),
        stage_slug="procedural_topup_ids",
    )
    c = ctx.run_cmd(ctx.py("entities_to_csv.py", str(ctx.base), "convert"), stage_slug="procedural_topup_ids")

    ctx.state["artifacts"]["new_person_ids"] = p.get("new_person_ids", [])
    ctx.save_state()
    return {"persons_topup": p, "keyword_topup": k, "character_topup": ch, "title_topup": t, "entities_to_csv": c}


def stage_05_llm_person_fill(ctx: Ctx) -> dict[str, Any]:
    if not ctx.budget_guard("llm_person_fill", optional=False):
        return {"skipped": True}

    rows = read_json(ctx.ent / "persons.json", [])
    if not isinstance(rows, list):
        rows = []

    new_ids = set(ctx.state.get("artifacts", {}).get("new_person_ids", []) or [])
    todo_mandatory = []
    todo_optional = []

    for x in rows:
        pid = int(x.get("person_id", 0) or 0)
        bio = str(x.get("bio", "")).strip()
        st = x.get("style_tags", [])
        if isinstance(st, str):
            st = [z.strip() for z in st.replace(";", ",").split(",") if z.strip()]

        if pid in new_ids:
            todo_mandatory.append(x)
        elif len(bio) < 35 or len(st) == 0:
            todo_optional.append(x)

    # Keep budget-safe behavior: mandatory fill for newly created persons first,
    # then only a bounded optional repair slice from legacy rows.
    max_optional = 0
    if len(new_ids) > 0:
        max_optional = min(len(todo_optional), max(120, int(len(new_ids) * 0.20)))
    todo_optional = sorted(todo_optional, key=lambda z: int(z.get("person_id", 0) or 0))[:max_optional]
    todo = todo_mandatory + todo_optional

    if not todo:
        ctx.register_spend("llm_person_fill", observed=0.0)
        return {"updated": 0, "cost_usd": 0.0}

    from contracts import GENRES, STYLE_TAGS, MARKETS

    llm = get_llm_client()
    by_id = {int(x.get("person_id", 0) or 0): x for x in rows}
    rng = random.Random(20260321)

    tin = tout = 0
    cost = 0.0
    retries = timeout_r = service_r = 0
    updated = fallback = 0

    def fill_fallback(x: dict[str, Any]) -> bool:
        nonlocal fallback
        changed = False

        if len(str(x.get("bio", "")).strip()) < 40:
            x["bio"] = f"{x.get('name', 'This person')} is a {x.get('career_stage', 'prime')} {', '.join(x.get('roles', ['actor']))} with a collaborative track record across modern productions."
            changed = True

        stx = x.get("style_tags", [])
        if isinstance(stx, str):
            stx = [z.strip() for z in stx.split(",") if z.strip()]
        if not stx:
            x["style_tags"] = rng.sample(STYLE_TAGS, k=min(3, len(STYLE_TAGS)))
            changed = True

        gx = x.get("genre_affinity", [])
        if isinstance(gx, str):
            gx = [z.strip() for z in gx.split(",") if z.strip()]
        if not gx:
            x["genre_affinity"] = [rng.choice(GENRES), rng.choice(GENRES)]
            changed = True

        mx = x.get("market_fit", [])
        if isinstance(mx, str):
            mx = [z.strip() for z in mx.split(",") if z.strip()]
        if not mx:
            x["market_fit"] = [rng.choice(MARKETS)]
            changed = True

        if changed:
            fallback += 1
        return changed

    for s in range(0, len(todo), 60):
        batch = todo[s : s + 60]
        lines = []
        for x in batch:
            roles = x.get("roles", ["actor"])
            if isinstance(roles, str):
                roles = [z.strip() for z in roles.split(",") if z.strip()]
            lines.append(f"[{x.get('person_id')}] {x.get('name')} | {x.get('nationality')} | {x.get('gender')} | {','.join(roles)} | {x.get('career_stage', 'prime')}")

        prompt = (
            f"Complete details for {len(batch)} movie-industry persons. Keep identity fields unchanged.\n"
            f"Return JSON array of objects with: person_id, bio, style_tags, genre_affinity, market_fit.\n"
            f"style_tags from {STYLE_TAGS[:20]}\n"
            f"genre_affinity from {GENRES}\n"
            f"market_fit from {MARKETS}\n"
            "JSON only.\n\nPERSONS:\n" + "\n".join(lines)
        )

        try:
            resp = llm.generate(
                prompt,
                model=str(ctx.args.llm_model),
                temperature=0.7,
                max_tokens=len(batch) * 260,
                timeout_sec=70,
                max_attempts=5,
            )

            tin += resp.input_tokens
            tout += resp.output_tokens
            cost += resp.cost_usd

            parsed = parse_json_loose(resp.text)
            if isinstance(parsed, dict):
                parsed = parsed.get("persons") or parsed.get("data") or []
            if not isinstance(parsed, list):
                parsed = []

            out_by_id = {}
            for z in parsed:
                if isinstance(z, dict):
                    pid = int(z.get("person_id", 0) or 0)
                    if pid > 0:
                        out_by_id[pid] = z

            for x in batch:
                pid = int(x.get("person_id", 0) or 0)
                out = out_by_id.get(pid)
                if isinstance(out, dict):
                    bio = str(out.get("bio", "")).strip()
                    stx = out.get("style_tags", [])
                    gx = out.get("genre_affinity", [])
                    mx = out.get("market_fit", [])
                    if isinstance(stx, str):
                        stx = [z.strip() for z in stx.split(",") if z.strip()]
                    if isinstance(gx, str):
                        gx = [z.strip() for z in gx.split(",") if z.strip()]
                    if isinstance(mx, str):
                        mx = [z.strip() for z in mx.split(",") if z.strip()]

                    if len(bio) >= 40:
                        x["bio"] = bio
                    x["style_tags"] = [z for z in stx if z in STYLE_TAGS][:4] or x.get("style_tags", [])
                    x["genre_affinity"] = [z for z in gx if z in GENRES][:3] or x.get("genre_affinity", [])
                    x["market_fit"] = [z for z in mx if z in MARKETS][:2] or x.get("market_fit", [])

                fill_fallback(x)
                by_id[pid] = x
                updated += 1
        except Exception as e:
            ctx.log(f"person detail fill fallback batch due to error: {e}")
            for x in batch:
                fill_fallback(x)
                pid = int(x.get("person_id", 0) or 0)
                by_id[pid] = x
                updated += 1

    out_rows = sorted(by_id.values(), key=lambda z: int(z.get("person_id", 0) or 0))
    write_json(ctx.ent / "persons.json", out_rows)
    ctx.run_cmd(ctx.py("entities_to_csv.py", str(ctx.base), "convert"), stage_slug="llm_person_fill")

    ctx.register_spend("llm_person_fill", observed=cost)
    return {
        "updated": updated,
        "fallback_fills": fallback,
        "tokens_in": tin,
        "tokens_out": tout,
        "cost_usd": round(cost, 4),
        "retries": retries,
        "timeout_retries": timeout_r,
        "service_retries": service_r,
    }


def stage_06_llm_company_fill(ctx: Ctx) -> dict[str, Any]:
    if not ctx.budget_guard("llm_company_fill", optional=False):
        return {"skipped": True}

    req_companies = int(ctx.state.get("demand", {}).get("requirements", {}).get("companies", 0))
    inv_before = ctx.inventory()
    cur_companies = int(inv_before.get("companies_json", 0) or inv_before.get("company_csv", 0))

    if cur_companies >= req_companies:
        ctx.register_spend("llm_company_fill", observed=0.0)
        return {"skipped": True, "reason": "already_at_target", "current": cur_companies, "target": req_companies}

    r = ctx.run_cmd(
        ctx.py(
            "generate_companies_llm.py",
            "--target",
            str(req_companies),
            "--timeout",
            "70",
            "--batch-size",
            "50",
        ),
        stage_slug="llm_company_fill",
    )
    c = ctx.run_cmd(ctx.py("entities_to_csv.py", str(ctx.base), "convert"), stage_slug="llm_company_fill")

    inv_after = ctx.inventory()
    added = int(inv_after.get("companies_json", 0)) - int(inv_before.get("companies_json", 0))

    ctx.register_spend("llm_company_fill", observed=r.get("observed_cost_usd"))
    return {"generate_companies": r, "entities_to_csv": c, "companies_added": added}


def stage_07_remaining_entities(ctx: Ctx) -> dict[str, Any]:
    if not ctx.budget_guard("llm_remaining", optional=False):
        return {"skipped": True}

    deficits = ctx.state.get("demand", {}).get("deficits", {})
    req = ctx.state.get("demand", {}).get("requirements", {})
    inv = ctx.inventory()

    observed_cost = 0.0
    actions: dict[str, Any] = {}

    new_person_ids = ctx.state.get("artifacts", {}).get("new_person_ids", []) or []
    need_ag_step1 = int(deficits.get("agencies", 0)) > 0 or int(inv.get("agencies_json", 0)) == 0
    need_ag_assign = need_ag_step1 or len(new_person_ids) > 0

    if need_ag_step1:
        s1 = ctx.run_cmd(ctx.py("generate_agencies.py", "--step", "1", "--auto"), stage_slug="llm_remaining")
        actions["agencies_step1"] = s1
        observed_cost += float(s1.get("observed_cost_usd") or 0.0)
    else:
        actions["agencies_step1"] = {"skipped": True, "reason": "already_sufficient"}

    if need_ag_assign:
        s2 = ctx.run_cmd(ctx.py("generate_agencies.py", "--step", "2", "--auto", "--batch-size", "180"), stage_slug="llm_remaining")
        actions["agencies_step2"] = s2
        observed_cost += float(s2.get("observed_cost_usd") or 0.0)
    else:
        actions["agencies_step2"] = {"skipped": True, "reason": "no_new_persons"}

    cur_cliques = int(inv.get("cliques_json", 0))
    req_cliques = int(req.get("cliques", 18) or 18)
    need_force = cur_cliques < req_cliques

    req_companies = int(req.get("companies", 0) or 0)
    cur_companies_now = int(inv.get("companies_json", 0) or inv.get("company_csv", 0))
    company_gap_now = max(0, req_companies - cur_companies_now)
    need_clique_assign = need_force or company_gap_now > 0

    if need_clique_assign:
        c1_cmd = ["generate_cliques.py", "--step", "1", "--num-cliques", str(req_cliques), "--timeout", "70"]
        if need_force:
            c1_cmd.append("--force")
        c1 = ctx.run_cmd(ctx.py(*c1_cmd), stage_slug="llm_remaining")
        c2 = ctx.run_cmd(ctx.py("generate_cliques.py", "--step", "2", "--batch-size", "100", "--timeout", "70"), stage_slug="llm_remaining")
        actions["cliques_step1"] = c1
        actions["cliques_step2"] = c2
        observed_cost += float(c1.get("observed_cost_usd") or 0.0)
        observed_cost += float(c2.get("observed_cost_usd") or 0.0)
    else:
        actions["cliques_step1"] = {"skipped": True, "reason": "already_sufficient"}
        actions["cliques_step2"] = {"skipped": True, "reason": "no_new_companies"}

    lat = ctx.run_cmd(
        ctx.py(
            "generate_latent_vars_api.py",
            "--auto",
            "--model",
            str(ctx.args.llm_model),
            "--batch-size",
            "60",
        ),
        stage_slug="llm_remaining",
    )
    observed_cost += float(lat.get("observed_cost_usd") or 0.0)
    actions["latent_generation"] = lat

    c = ctx.run_cmd(ctx.py("entities_to_csv.py", str(ctx.base), "convert"), stage_slug="llm_remaining")
    actions["entities_to_csv"] = c

    ctx.register_spend("llm_remaining", observed=observed_cost)
    return actions


def _calibrate_edges_file(ctx: Ctx) -> dict[str, Any]:
    path = ctx.graph / "edge_graph.csv"
    if not path.exists():
        return {"edges": 0, "added_bridges": 0, "added_hub_edges": 0}

    df = pd.read_csv(path, low_memory=False)
    if df.empty or not {"src_id", "dst_id", "weight"}.issubset(df.columns):
        return {"edges": int(len(df)), "added_bridges": 0, "added_hub_edges": 0}

    # Keep temporal windows total-order and non-null across all edge types.
    if "valid_from" not in df.columns:
        df["valid_from"] = 1975
    if "valid_to" not in df.columns:
        df["valid_to"] = 2065
    df["valid_from"] = pd.to_numeric(df["valid_from"], errors="coerce").fillna(1975).astype(int)
    df["valid_to"] = pd.to_numeric(df["valid_to"], errors="coerce").fillna(2065).astype(int)
    bad_window = df["valid_to"] < df["valid_from"]
    if bad_window.any():
        df.loc[bad_window, "valid_to"] = df.loc[bad_window, "valid_from"]

    name_map: dict[int, str] = {}
    if "src_name" in df.columns:
        for sid, sname in zip(df["src_id"], df["src_name"]):
            try:
                name_map[int(sid)] = str(sname)
            except Exception:
                continue
    if "dst_name" in df.columns:
        for did, dname in zip(df["dst_id"], df["dst_name"]):
            try:
                name_map[int(did)] = str(dname)
            except Exception:
                continue

    persons = read_csv(ctx.ent / "person.csv")
    person_ids: set[int] = set()
    if not persons.empty and "person_id" in persons.columns:
        person_ids = set(pd.to_numeric(persons["person_id"], errors="coerce").dropna().astype(int).tolist())

    if not persons.empty and {"person_id", "name"}.issubset(persons.columns):
        for r in persons[["person_id", "name"]].itertuples(index=False):
            try:
                name_map[int(getattr(r, "person_id"))] = str(getattr(r, "name"))
            except Exception:
                continue

    if not person_ids:
        return {"edges": int(len(df)), "added_bridges": 0, "added_hub_edges": 0, "note": "no_person_ids"}

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).clip(0.05, 0.98)

    deg = Counter()
    adj: dict[int, set[int]] = defaultdict(set)
    pair_keys = set()

    def pair_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    for r in df.itertuples(index=False):
        try:
            a = int(getattr(r, "src_id"))
            b = int(getattr(r, "dst_id"))
            et = str(getattr(r, "edge_type", "")).lower()
        except Exception:
            continue
        if a == b:
            continue
        if a not in person_ids or b not in person_ids:
            continue
        if et and et not in PERSON_EDGE_TYPES:
            continue
        deg[a] += 1
        deg[b] += 1
        adj[a].add(b)
        adj[b].add(a)
        pair_keys.add(pair_key(a, b))

    nodes = list(adj.keys())
    comps: list[list[int]] = []
    seen = set()
    for s in nodes:
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        cur = []
        while q:
            x = q.popleft()
            cur.append(x)
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        comps.append(cur)

    comps.sort(key=len, reverse=True)
    main_comp = comps[0] if comps else []
    main_hub = max(main_comp, key=lambda x: deg.get(x, 0)) if main_comp else None

    new_rows = []
    bridge_added = 0
    if main_hub is not None:
        for comp in comps[1:]:
            src = max(comp, key=lambda x: deg.get(x, 0))
            dst = int(main_hub)
            k = pair_key(src, dst)
            if k in pair_keys:
                continue
            pair_keys.add(k)
            bridge_added += 1
            new_rows.append({
                "src_id": src,
                "dst_id": dst,
                "src_name": name_map.get(src, ""),
                "dst_name": name_map.get(dst, ""),
                "edge_type": "collaboration",
                "sign": "+",
                "weight": 0.56,
                "source_kind": "runner_calibration",
                "reason": "component_bridge",
                "valid_from": 1975,
                "valid_to": 2035,
            })

    hub_added = 0
    if not persons.empty and {"person_id", "pop_weight"}.issubset(persons.columns):
        p = persons[["person_id", "name", "pop_weight"]].copy()
        p["person_id"] = pd.to_numeric(p["person_id"], errors="coerce").fillna(0).astype(int)
        p["pop_weight"] = pd.to_numeric(p["pop_weight"], errors="coerce").fillna(0.0)
        p = p.sort_values("pop_weight", ascending=False)
        top = p.head(min(80, len(p)))
        top_ids = [int(x) for x in top["person_id"].tolist() if int(x) > 0 and int(x) in person_ids]

        for i in range(min(len(top_ids), 60)):
            a = int(top_ids[i])
            b = int(top_ids[(i + 7) % len(top_ids)])
            if a == b:
                continue
            k = pair_key(a, b)
            if k in pair_keys:
                continue
            pair_keys.add(k)
            hub_added += 1
            new_rows.append({
                "src_id": a,
                "dst_id": b,
                "src_name": name_map.get(a, ""),
                "dst_name": name_map.get(b, ""),
                "edge_type": "friendship",
                "sign": "+",
                "weight": round(0.42 + 0.25 * _stable_uniform(a, b), 2),
                "source_kind": "runner_calibration",
                "reason": "star_reinforcement",
                "valid_from": 1975,
                "valid_to": 2035,
            })

    if new_rows:
        add_df = pd.DataFrame(new_rows)
        for c in df.columns:
            if c not in add_df.columns:
                add_df[c] = None
        for c in add_df.columns:
            if c not in df.columns:
                df[c] = None
        df = pd.concat([df[df.columns], add_df[df.columns]], ignore_index=True)

    df.to_csv(path, index=False)
    return {"edges": int(len(df)), "added_bridges": int(bridge_added), "added_hub_edges": int(hub_added)}


def stage_08_rebuild_edges(ctx: Ctx) -> dict[str, Any]:
    r = ctx.run_cmd(ctx.py("generate_edges_hybrid.py"), stage_slug="edge_rebuild")
    cal = _calibrate_edges_file(ctx)
    metrics = compute_realism_metrics(ctx)
    return {"generate_edges": r, "calibration": cal, "edge_metrics": {k: metrics.get(k) for k in ["edge_rows", "edge_nodes", "edge_degree_gini", "edge_giant_component_ratio"]}}


def _synthesize_company_financial_profile(ctx: Ctx) -> dict[str, Any]:
    company_path = ctx.ent / "company.csv"
    if not company_path.exists():
        raise FileNotFoundError(f"missing {company_path}")

    cdf = pd.read_csv(company_path, low_memory=False)
    if cdf.empty or "company_id" not in cdf.columns:
        raise RuntimeError("company.csv missing company_id")

    cdf["company_id"] = pd.to_numeric(cdf["company_id"], errors="coerce").fillna(0).astype(int)

    mcomp = read_csv(ctx.base / "movie_companies.csv")
    movie = read_csv(ctx.base / "movie.csv")
    clink = read_csv(ctx.base / "company_links.csv")

    title_year = {}
    franchise_flag = {}
    if not movie.empty and "title_id" in movie.columns:
        y = pd.to_numeric(movie.get("year", pd.Series(dtype=float)), errors="coerce").fillna(2005).astype(int)
        title_year = dict(zip(movie["title_id"], y))
        if "franchise_id" in movie.columns:
            franchise_flag = {tid: int(not pd.isna(fid) and str(fid).strip() != "") for tid, fid in zip(movie["title_id"], movie["franchise_id"]) }

    hist_titles = Counter()
    recent_titles = Counter()
    franchise_titles = Counter()

    if not mcomp.empty and {"company_id", "title_id"}.issubset(mcomp.columns):
        for r in mcomp.itertuples(index=False):
            try:
                cid = int(getattr(r, "company_id"))
                tid = getattr(r, "title_id")
                yy = int(title_year.get(tid, 2005))
            except Exception:
                continue
            hist_titles[cid] += 1
            if yy >= 2015:
                recent_titles[cid] += 1
            if franchise_flag.get(tid, 0) == 1:
                franchise_titles[cid] += 1

    collab_deg = Counter()
    if not clink.empty and {"company_id_1", "company_id_2"}.issubset(clink.columns):
        for r in clink.itertuples(index=False):
            try:
                a = int(getattr(r, "company_id_1"))
                b = int(getattr(r, "company_id_2"))
            except Exception:
                continue
            if a == b:
                continue
            collab_deg[a] += 1
            collab_deg[b] += 1

    tier_prior = {
        "Global": {"capital": 0.88, "margin": 0.24, "debt": 0.44, "slate": 0.88, "buffer": 0.74, "growth": 0.44, "eff": 0.70},
        "Major": {"capital": 0.76, "margin": 0.21, "debt": 0.47, "slate": 0.74, "buffer": 0.62, "growth": 0.48, "eff": 0.63},
        "Mid-Budget": {"capital": 0.58, "margin": 0.18, "debt": 0.43, "slate": 0.54, "buffer": 0.50, "growth": 0.54, "eff": 0.56},
        "Indie": {"capital": 0.40, "margin": 0.15, "debt": 0.35, "slate": 0.34, "buffer": 0.40, "growth": 0.58, "eff": 0.51},
        "Micro": {"capital": 0.26, "margin": 0.11, "debt": 0.29, "slate": 0.22, "buffer": 0.30, "growth": 0.52, "eff": 0.45},
    }

    rows = []
    for r in cdf.itertuples(index=False):
        cid = int(getattr(r, "company_id"))
        tier = str(getattr(r, "tier", "Mid-Budget"))
        base = tier_prior.get(tier, tier_prior["Mid-Budget"])

        spec = str(getattr(r, "specialty_genres", ""))
        spec_cnt = max(1, len([x for x in re.split(r"[;,|]", spec) if str(x).strip()]))

        n_hist = int(hist_titles.get(cid, 0))
        n_recent = int(recent_titles.get(cid, 0))
        n_fr = int(franchise_titles.get(cid, 0))
        deg = int(collab_deg.get(cid, 0))

        hist_norm = min(1.0, math.log1p(n_hist) / math.log1p(120.0))
        recent_norm = min(1.0, math.log1p(n_recent) / math.log1p(35.0))
        collab_norm = min(1.0, math.log1p(deg) / math.log1p(80.0))
        franchise_norm = min(1.0, n_fr / max(1.0, n_hist))
        focus_pen = min(0.25, 0.05 * max(0, spec_cnt - 3))

        eps = (_stable_uniform("fin", cid) - 0.5) * 0.08

        capital = np.clip(base["capital"] + 0.20 * hist_norm + 0.12 * collab_norm + eps, 0.12, 0.98)
        margin = np.clip(base["margin"] + 0.08 * recent_norm + 0.04 * franchise_norm + eps * 0.5, 0.04, 0.45)
        debt = np.clip(base["debt"] + 0.10 * (1.0 - capital) - 0.05 * margin + 0.03 * focus_pen + eps * 0.4, 0.08, 0.88)
        slate = np.clip(base["slate"] + 0.25 * hist_norm + 0.15 * recent_norm - 0.06 * focus_pen + eps, 0.08, 0.99)
        risk_buffer = np.clip(base["buffer"] + 0.18 * capital + 0.10 * margin - 0.14 * debt + eps, 0.05, 0.95)
        growth = np.clip(base["growth"] + 0.16 * recent_norm + 0.10 * collab_norm - 0.08 * franchise_norm + eps, 0.04, 0.96)
        eff = np.clip(base["eff"] + 0.10 * margin + 0.08 * collab_norm + 0.04 * franchise_norm - 0.05 * focus_pen + eps, 0.08, 0.95)

        rows.append({
            "company_id": cid,
            "capital_score": round(float(capital), 4),
            "operating_margin": round(float(margin), 4),
            "debt_ratio": round(float(debt), 4),
            "slate_capacity": round(float(slate), 4),
            "risk_buffer": round(float(risk_buffer), 4),
            "growth_bias": round(float(growth), 4),
            "revenue_efficiency": round(float(eff), 4),
            "profile_bucket": tier,
            "updated_at": now_utc(),
        })

    out = pd.DataFrame(rows).sort_values("company_id")
    out_path = ctx.ent / "company_financial_profile.csv"
    out.to_csv(out_path, index=False)

    return {
        "rows": int(len(out)),
        "path": str(out_path),
        "capital_mean": round(float(out["capital_score"].mean()), 4),
        "margin_mean": round(float(out["operating_margin"].mean()), 4),
        "debt_mean": round(float(out["debt_ratio"].mean()), 4),
    }


def stage_09_financial_synthesis(ctx: Ctx) -> dict[str, Any]:
    rep = _synthesize_company_financial_profile(ctx)
    return rep

def _run_generation_core(ctx: Ctx, stage_slug: str, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    gen = ctx.run_cmd(
        ctx.py(
            "generate_movies.py",
            "--base_dir",
            str(ctx.base),
            "--n_movies",
            str(ctx.args.target_movies),
            "--llm_model",
            str(ctx.args.llm_model),
            "--enable_llm_evolution" if ctx.runtime["enable_llm_temporal_evolution"] else "--disable_llm_evolution",
        ),
        stage_slug=stage_slug,
        extra_env=extra_env,
    )
    return gen


def stage_10_full_generation(ctx: Ctx) -> dict[str, Any]:
    if not ctx.budget_guard("pipeline_core", optional=False):
        return {"skipped": True}

    core = _run_generation_core(ctx, stage_slug="pipeline_generation")
    ctx.register_spend("pipeline_core", observed=core.get("observed_cost_usd"))

    plot_rep: dict[str, Any] = {"skipped": True, "reason": "disabled"}
    if ctx.runtime["include_plots"]:
        if ctx.budget_guard("plot_summaries", optional=True):
            p = ctx.run_cmd(
                ctx.py("generate_plot_summaries_api.py", "--auto", "--model", str(ctx.args.llm_model)),
                stage_slug="pipeline_generation",
            )
            ctx.register_spend("plot_summaries", observed=p.get("observed_cost_usd"))
            plot_rep = p
        else:
            ctx.runtime["include_plots"] = False
            ctx.save_state()
            plot_rep = {"skipped": True, "reason": "budget_guard"}

    return {"generate_movies": core, "plot_stage": plot_rep}


def _autofix_pass(ctx: Ctx, reasons: list[str]) -> dict[str, Any]:
    ctx.log("Quality gate failed; running deterministic one-pass auto-fix")

    edges = stage_08_rebuild_edges(ctx)
    fin = stage_09_financial_synthesis(ctx)

    env = {
        "V16_LEGEND_MULT": "18.0",
        "V16_PRIME_MULT": "6.2",
        "V16_VETERAN_MULT": "2.8",
        "V16_TOP_STAR_SLOT_PENALTY": "0.68",
        "V16_EPIC_TAIL_SCALE": "1.30",
        "V16_A_TAIL_PROB": "0.10",
        "V16_CAST_MAX": "140",
    }

    if not ctx.budget_guard("pipeline_core", optional=False):
        raise RuntimeError("autofix cannot run generation: budget guard")

    core = _run_generation_core(ctx, stage_slug="quality_gate", extra_env=env)
    ctx.register_spend("pipeline_core", observed=core.get("observed_cost_usd"))

    plot_rep: dict[str, Any] = {"skipped": True}
    if ctx.runtime["include_plots"] and ctx.budget_guard("plot_summaries", optional=True):
        p = ctx.run_cmd(
            ctx.py("generate_plot_summaries_api.py", "--auto", "--model", str(ctx.args.llm_model)),
            stage_slug="quality_gate",
        )
        ctx.register_spend("plot_summaries", observed=p.get("observed_cost_usd"))
        plot_rep = p

    return {
        "reasons": reasons,
        "edge_rebuild": edges,
        "financial_refresh": fin,
        "regenerate": core,
        "plots": plot_rep,
        "env_overrides": env,
    }


def stage_11_quality_gate(ctx: Ctx) -> dict[str, Any]:
    before = evaluate_quality(compute_realism_metrics(ctx), strict=True)
    fail_reasons = [f"{c['name']}={c['value']} (target {c['target']})" for c in before["checks"] if not c["ok"]]

    if before["pass"]:
        ctx.state["artifacts"]["quality_post"] = before
        ctx.save_state()
        return {"before": before, "auto_fix": None, "after": before}

    if ctx.runtime.get("quality_autofix_attempted", False):
        raise RuntimeError("quality gate failed after already using auto-fix pass: " + "; ".join(fail_reasons))

    ctx.runtime["quality_autofix_attempted"] = True
    ctx.save_state()

    fix = _autofix_pass(ctx, fail_reasons)
    after = evaluate_quality(compute_realism_metrics(ctx), strict=True)

    ctx.state["artifacts"]["quality_post"] = after
    ctx.state["artifacts"]["quality_autofix"] = fix
    ctx.save_state()

    if not after["pass"]:
        fail_after = [f"{c['name']}={c['value']} (target {c['target']})" for c in after["checks"] if not c["ok"]]
        raise RuntimeError("quality gate failed after deterministic auto-fix: " + "; ".join(fail_after))

    return {"before": before, "auto_fix": fix, "after": after}


def stage_12_job_export(ctx: Ctx) -> dict[str, Any]:
    out_dir = (ctx.base / "imdb_schema_v16" / f"overnight_{ctx.state['run_id']}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        *ctx.py("v15_to_imdb_schema.py"),
        "--base-dir",
        str(ctx.base),
        "--out-dir",
        str(out_dir),
        "--strict-job" if bool(ctx.args.strict_job_export) else "--no-strict-job",
        "--include-extras" if bool(ctx.args.include_extras) else "--no-include-extras",
    ]
    conv = ctx.run_cmd(cmd, stage_slug="job_export")

    val = None
    if bool(ctx.args.strict_job_export):
        val = ctx.run_cmd(
            ctx.py("validate_imdb_schema.py", "--schema-dir", str(out_dir)),
            stage_slug="job_export",
        )

    extras_dir = out_dir / "extras"
    extras_dir.mkdir(parents=True, exist_ok=True)
    edge_src = ctx.graph / "edge_graph.csv"
    if edge_src.exists():
        shutil.copy2(edge_src, extras_dir / "edge_graph.csv")

    ctx.state["artifacts"]["job_export_dir"] = str(out_dir)
    ctx.save_state()
    return {
        "convert": conv,
        "validate": val,
        "export_dir": str(out_dir),
        "edge_extra": str(extras_dir / "edge_graph.csv") if edge_src.exists() else None,
    }


def build_run_report(ctx: Ctx) -> dict[str, Any]:
    after = ctx.inventory()
    before = ctx.inventory_before or {}

    delta = {}
    keys = sorted(set(before.keys()) | set(after.keys()))
    for k in keys:
        try:
            delta[k] = int(after.get(k, 0)) - int(before.get(k, 0))
        except Exception:
            pass

    quality_before = ctx.state.get("artifacts", {}).get("quality_baseline")
    quality_after = ctx.state.get("artifacts", {}).get("quality_post")

    retry_stats = {
        "person_fill": ctx.state.get("stages", {}).get("llm_person_fill", {}).get("result", {}),
    }

    report = {
        "runner": RUNNER,
        "run_id": ctx.state.get("run_id"),
        "status": ctx.state.get("status"),
        "final_status": ctx.state.get("final_status"),
        "started_at": ctx.state.get("started_at"),
        "updated_at": ctx.state.get("updated_at"),
        "restart_from_stage": ctx.state.get("restart_from_stage"),
        "config": ctx.state.get("config", {}),
        "runtime": dict(ctx.runtime),
        "entity_inventory_before": before,
        "entity_inventory_after": after,
        "entity_deltas": delta,
        "budget": ctx.state.get("budget", {}),
        "spend_by_stage_estimate": dict(ctx.stage_cost),
        "retry_timeout_stats": retry_stats,
        "quality_before": quality_before,
        "quality_after": quality_after,
        "stages": ctx.state.get("stages", {}),
        "artifacts": ctx.state.get("artifacts", {}),
    }

    write_json(ctx.report_path, report)
    ctx.state["artifacts"]["run_report"] = str(ctx.report_path)
    ctx.save_state()
    return report


STAGES: list[tuple[int, str, str, Callable[[Ctx], dict[str, Any]]]] = [
    (1, "inventory_contracts", "Baseline Inventory + Schema Contracts", stage_01_inventory_contracts),
    (2, "demand_plan", "10k Demand Planning", stage_02_demand_plan),
    (3, "budget_preflight", "Budget Preflight", stage_03_budget_preflight),
    (4, "procedural_topup_ids", "Procedural Top-Up for Dedupe-Sensitive IDs", stage_04_topup_ids),
    (5, "llm_person_fill", "LLM Person Detail Fill", stage_05_llm_person_fill),
    (6, "llm_company_fill", "LLM Company Names + Details", stage_06_llm_company_fill),
    (7, "llm_remaining", "Top-Up Remaining Entities in Dependency Order", stage_07_remaining_entities),
    (8, "edge_rebuild", "Rebuild and Calibrate Dependency Edges", stage_08_rebuild_edges),
    (9, "financial_synthesis", "Company Financial Synthesis Pass", stage_09_financial_synthesis),
    (10, "pipeline_generation", "Full Generation Pipeline (Temporal Evolution ON)", stage_10_full_generation),
    (11, "quality_gate", "Post-Run Quality Gate with One Auto-Fix Pass", stage_11_quality_gate),
    (12, "job_export", "Strict JOB Export + Extras", stage_12_job_export),
]


def run_stage(ctx: Ctx, idx: int, slug: str, name: str, fn: Callable[[Ctx], dict[str, Any]]):
    if ctx.args.resume and ctx.done(idx, slug):
        ctx.log(f"SKIP stage {idx:02d} {slug}: completion marker exists")
        mk = read_json(ctx.marker(idx, slug), {})
        res = mk.get("result", {}) if isinstance(mk, dict) else {}
        ctx.state["stages"].setdefault(slug, {})
        ctx.state["stages"][slug].update({"idx": idx, "name": name, "status": "completed", "result": res})
        if idx not in ctx.state.get("completed", []):
            ctx.state.setdefault("completed", []).append(idx)
            ctx.state["completed"] = sorted(ctx.state["completed"])
        ctx.state["restart_from_stage"] = idx + 1
        ctx.save_state()
        return

    start = time.time()
    ctx.mark_start(idx, slug, name)
    try:
        result = fn(ctx)
        elapsed = time.time() - start
        ctx.mark_done(idx, slug, result, elapsed)
    except Exception as e:
        elapsed = time.time() - start
        ctx.mark_fail(idx, slug, e, elapsed)
        raise


def _add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool):
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        parser.add_argument(name, action=action, default=default)
        return

    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true")
    parser.add_argument(f"--no-{dest.replace('_', '-')}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: bool(default)})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overnight 10k movie runner with budget guards, resume, quality gates, and JOB export")
    p.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    p.add_argument("--target-movies", type=int, default=10000)
    p.add_argument("--budget-usd", type=float, default=50.0)
    p.add_argument("--price-per-million", type=float, default=1.50)
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    _add_bool_flag(p, "--enable-llm-temporal-evolution", True)
    _add_bool_flag(p, "--include-plots", True)
    _add_bool_flag(p, "--strict-job-export", True)
    _add_bool_flag(p, "--include-extras", True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ctx = Ctx(args)
    ctx.init_state()

    stages = STAGES[:3] if args.preflight_only else STAGES

    try:
        for idx, slug, name, fn in stages:
            run_stage(ctx, idx, slug, name, fn)

        ctx.state["status"] = "completed"
        ctx.state["final_status"] = "preflight_only_completed" if args.preflight_only else "completed"
        ctx.state["restart_from_stage"] = (3 if args.preflight_only else STAGES[-1][0]) + 1
        ctx.save_state()

    except Exception as e:
        ctx.log(f"RUN FAILED: {e}")

    report = build_run_report(ctx)

    print("=" * 72)
    print("OVERNIGHT RUN REPORT")
    print("=" * 72)
    print(f"Status: {report['status']} ({report['final_status']})")
    print(f"Run ID: {report['run_id']}")
    print(f"Restart from stage: {report.get('restart_from_stage')}")
    print(f"Estimated spend: ${report['budget'].get('spent_estimated_usd', 0.0):.4f}")
    print(f"Observed spend:  ${report['budget'].get('spent_observed_usd', 0.0):.4f}")
    if report.get("quality_after"):
        qa = report["quality_after"]
        print(f"Quality pass: {qa.get('pass')}")
    print(f"Report path: {ctx.report_path}")

    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()



















