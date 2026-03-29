from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from contracts import ENTITY_COUNTS, MODEL_TIERS, SNAPSHOT_CONFIG
from llm_provider import get_llm_client, safe_json_parse
from policy_runtime import (
    franchise_bibles_path,
    normalize_franchise_bibles,
    safe_load_json,
    write_json,
)
from world_state import WorldState

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = MODEL_TIERS.get("entity_gen", "gemini-3.1-flash-lite-preview")
BATCH_SIZE = 18


def _is_local_like(model: str | None = None) -> bool:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_name = str(model or "").strip().lower()
    return provider in {"local", "ollama", "vllm", "openai", "tgi", "litellm"} or "qwen" in model_name


def _franchise_batch_size(model: str | None = None) -> int:
    return 4 if _is_local_like(model) else BATCH_SIZE


def _extract_bible_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("bibles", "rows", "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if payload.get("franchise_id"):
            return [payload]
    return []


def _build_prompt(batch: list[dict[str, Any]], context: dict[str, Any]) -> str:
    return (
        "You are generating franchise continuity bibles for a synthetic film database.\n"
        "Return JSON only.\n\n"
        "Requirements:\n"
        '1. Top-level key must be "bibles".\n'
        "2. Return one bible per franchise.\n"
        "3. Each bible must include: franchise_id, franchise_name, genre, tier, installments,\n"
        "   continuity_anchors, recurring_motifs, keyword_families, title_style, subtitle_tokens,\n"
        "   release_season_bias, company_strategy_tag, cast_chemistry_target,\n"
        "   carryover_director_bias, carryover_cast_bias.\n"
        "4. Keep lists short and machine-friendly.\n"
        "5. carryover_director_bias and carryover_cast_bias must be 0..1.\n\n"
        f"Context:\n{json.dumps(context, ensure_ascii=True)}\n\n"
        f"Franchises:\n{json.dumps(batch, ensure_ascii=True)}\n"
    )


def _load_world(base_dir: Path, n_movies: int) -> WorldState:
    ENTITY_COUNTS["movies"] = int(n_movies)
    world = WorldState(
        str(base_dir),
        seed=SNAPSHOT_CONFIG["seed"],
        config_path=None,
        workspace=None,
    )
    world.load()
    return world


def generate_franchise_bibles(base_dir: Path, *, n_movies: int, model: str | None = None) -> dict[str, Any]:
    world = _load_world(base_dir, n_movies)
    context = {
        "world_policy": safe_load_json(base_dir / "world_policy.json", default={}),
        "year_slate_count": len((safe_load_json(base_dir / "year_slate_plan.json", default={}) or {}).get("slates", [])),
        "concept_pack_count": len((safe_load_json(base_dir / "concept_packs.json", default={}) or {}).get("packs", [])),
    }
    franchises = [
        {
            "franchise_id": int(row.get("franchise_id", 0) or 0),
            "name": str(row.get("name", "")),
            "genre": str(row.get("genre", "")),
            "tier": str(row.get("tier", "")),
            "n_movies": int(row.get("n_movies", 0) or 0),
            "installment_years": list(row.get("installment_years", [])),
        }
        for row in getattr(world, "franchises", [])
        if isinstance(row, dict)
    ]
    parsed_rows: list[dict[str, Any]] = []
    client = None
    try:
        client = get_llm_client()
    except Exception:
        client = None
    batch_size = _franchise_batch_size(model)
    n_batches = math.ceil(len(franchises) / batch_size) if franchises else 0
    for batch_idx in range(n_batches):
        batch = franchises[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        if client is None:
            continue
        try:
            response = client.generate(
                _build_prompt(batch, context),
                model=model or DEFAULT_MODEL,
                json_mode=True,
                temperature=0.2,
                max_tokens=4096,
                timeout_sec=90.0,
                max_attempts=4,
            )
            parsed = safe_json_parse(response.text)
            parsed_rows.extend(_extract_bible_rows(parsed))
        except Exception as exc:
            print(f"  Franchise bible batch {batch_idx + 1}/{max(1, n_batches)} fallback: {exc}")
    payload = normalize_franchise_bibles({"bibles": parsed_rows}, franchises=franchises)
    write_json(franchise_bibles_path(base_dir), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate franchise continuity bibles for test27.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--n-movies", type=int, required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    payload = generate_franchise_bibles(base_dir, n_movies=int(args.n_movies), model=args.model)
    print(
        "  Saved franchise bibles:",
        franchise_bibles_path(base_dir),
        f"({len(payload.get('bibles', []))} bibles)",
    )


if __name__ == "__main__":
    main()
