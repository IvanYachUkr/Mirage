"""
Generate baseline company financial profiles for movie generation.

This is a fresh-bootstrap generator, not a post-run analytics pass.
It derives stable finance signals from company tier, specialties, and
optionally company latent variables so WorldState/financials.py can use
them during movie generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ENTITY_DIR = BASE_DIR / "entities"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_uniform(*parts: object) -> float:
    raw = "|".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float((1 << 64) - 1)


def _safe_float(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _load_company_latent_map(entities_dir: Path) -> dict[int, dict]:
    path = entities_dir / "companies_latent.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    out: dict[int, dict] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        try:
            cid = int(row.get("company_id", 0))
        except Exception:
            continue
        if cid > 0:
            out[cid] = row
    return out


def synthesize_company_financial_profiles(base_dir: Path) -> Path:
    entities_dir = base_dir / "entities"
    company_csv = entities_dir / "company.csv"
    if not company_csv.exists():
        raise FileNotFoundError(f"company.csv not found at {company_csv}")

    cdf = pd.read_csv(company_csv, low_memory=False)
    if cdf.empty or "company_id" not in cdf.columns:
        raise RuntimeError("company.csv is empty or missing company_id")

    cdf["company_id"] = pd.to_numeric(cdf["company_id"], errors="coerce").fillna(0).astype(int)
    cdf = cdf[cdf["company_id"] > 0].copy()
    if cdf.empty:
        raise RuntimeError("company.csv contains no valid company rows")

    latent_map = _load_company_latent_map(entities_dir)

    tier_prior = {
        "Global": {"capital": 0.88, "margin": 0.24, "debt": 0.44, "slate": 0.88, "buffer": 0.74, "growth": 0.44, "eff": 0.70},
        "Major": {"capital": 0.76, "margin": 0.21, "debt": 0.47, "slate": 0.74, "buffer": 0.62, "growth": 0.48, "eff": 0.63},
        "Mid-Budget": {"capital": 0.58, "margin": 0.18, "debt": 0.43, "slate": 0.54, "buffer": 0.50, "growth": 0.54, "eff": 0.56},
        "Indie": {"capital": 0.40, "margin": 0.15, "debt": 0.35, "slate": 0.34, "buffer": 0.40, "growth": 0.58, "eff": 0.51},
        "Micro": {"capital": 0.26, "margin": 0.11, "debt": 0.29, "slate": 0.22, "buffer": 0.30, "growth": 0.52, "eff": 0.45},
    }
    budget_weights = np.array([0.18, 0.34, 0.56, 0.78, 1.00], dtype=float)

    rows: list[dict] = []
    for rec in cdf.itertuples(index=False):
        cid = int(getattr(rec, "company_id"))
        tier = str(getattr(rec, "tier", "Mid-Budget") or "Mid-Budget")
        base = tier_prior.get(tier, tier_prior["Mid-Budget"])

        spec_raw = str(getattr(rec, "specialty_genres", "") or "")
        spec_cnt = max(1, len([chunk for chunk in re.split(r"[;,|]", spec_raw) if str(chunk).strip()]))
        focus_pen = min(0.25, 0.05 * max(0, spec_cnt - 3))

        latent = latent_map.get(cid, {})
        prestige = _safe_float(latent.get("prestige_score"), _safe_float(getattr(rec, "pop_weight", 0.50), 0.50))
        risk_appetite = _safe_float(latent.get("risk_appetite"), 0.50)
        controversy_tol = _safe_float(latent.get("controversy_tolerance"), 0.50)
        trend = _safe_float(latent.get("market_trend_sensitivity"), 0.50)
        budget_focus = latent.get("budget_tier_focus", [0.5] * 5)
        if not isinstance(budget_focus, list):
            budget_focus = [0.5] * 5
        if len(budget_focus) != 5:
            budget_focus = (list(budget_focus) + [0.5] * 5)[:5]
        budget_focus_arr = np.clip(np.asarray(budget_focus, dtype=float), 0.0, 1.0)
        focus_strength = float(np.dot(budget_focus_arr, budget_weights) / float(budget_weights.sum()))

        eps = (_stable_uniform("finance-profile", cid) - 0.5) * 0.08

        capital = np.clip(base["capital"] + 0.18 * prestige + 0.12 * focus_strength - 0.05 * focus_pen + eps, 0.12, 0.98)
        margin = np.clip(base["margin"] + 0.08 * prestige + 0.05 * (1.0 - risk_appetite) + 0.03 * (1.0 - controversy_tol) + eps * 0.5, 0.04, 0.45)
        debt = np.clip(base["debt"] + 0.12 * (1.0 - capital) + 0.10 * risk_appetite - 0.04 * margin + 0.03 * focus_pen + eps * 0.4, 0.08, 0.88)
        slate = np.clip(base["slate"] + 0.18 * focus_strength + 0.10 * trend + 0.08 * prestige - 0.06 * focus_pen + eps, 0.08, 0.99)
        risk_buffer = np.clip(base["buffer"] + 0.18 * capital + 0.10 * margin - 0.12 * debt - 0.08 * risk_appetite + eps, 0.05, 0.95)
        growth = np.clip(base["growth"] + 0.14 * trend + 0.08 * risk_appetite + 0.06 * prestige - 0.05 * focus_pen + eps, 0.04, 0.96)
        efficiency = np.clip(base["eff"] + 0.10 * margin + 0.08 * prestige + 0.06 * (1.0 - debt) + 0.05 * focus_strength - 0.03 * focus_pen + eps, 0.08, 0.95)

        rows.append({
            "company_id": cid,
            "capital_score": round(float(capital), 4),
            "operating_margin": round(float(margin), 4),
            "debt_ratio": round(float(debt), 4),
            "slate_capacity": round(float(slate), 4),
            "risk_buffer": round(float(risk_buffer), 4),
            "growth_bias": round(float(growth), 4),
            "revenue_efficiency": round(float(efficiency), 4),
            "profile_bucket": tier,
            "updated_at": _now_utc(),
        })

    out = pd.DataFrame(rows).sort_values("company_id")
    out_path = entities_dir / "company_financial_profile.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved company_financial_profile.csv ({len(out):,} rows)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline company financial profiles")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    synthesize_company_financial_profiles(base_dir)


if __name__ == "__main__":
    main()
