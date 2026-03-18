"""
Generate keyword-to-genre assignments using LLM.
==================================================
Reads keyword.csv, identifies keywords with missing or default topic_genre
and pop_weight, and uses the LLM to assign proper genre associations.

Usage:
    python generate_keyword_genres.py --auto
    python generate_keyword_genres.py --model gemini-3.1-flash-lite-preview
    python generate_keyword_genres.py --force   # re-assign ALL keywords
"""
import os
import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

try:
    from contracts import GENRES, MODEL_TIERS
except ImportError:
    GENRES = ["Action", "Drama", "Comedy", "Horror", "Sci-Fi", "Thriller",
              "Romance", "Animation", "Documentary", "Fantasy", "Crime",
              "Mystery", "Western", "Film-Noir", "Sport", "Family"]
    MODEL_TIERS = {}

try:
    from google import genai
except ImportError:
    genai = None

try:
    from api_resilience import generate_content_resilient
except ImportError:
    generate_content_resilient = None

# ═══════════════════════════════════════════════════════════════════════
# Default model
# ═══════════════════════════════════════════════════════════════════════
MODEL = MODEL_TIERS.get("entity_gen", "gemini-3.1-flash-lite-preview")

BATCH_SIZE = 80  # keywords per LLM call

PRICING = {
    "gemini-3.1-flash-lite-preview": {"in": 0.02, "out": 0.08},
    "gemini-2.5-flash": {"in": 0.15, "out": 0.60},
}


def _build_prompt(keywords: list[str], genres: list[str]) -> str:
    """Build LLM prompt for keyword→genre assignment."""
    return f"""You are a film-industry metadata expert.

For each keyword below, assign:
1. "topic_genre": the single most relevant genre from this list: {json.dumps(genres)}
2. "pop_weight": a float 0.005-0.12 indicating how commonly this keyword appears in movies.
   - Very common keywords (love, death, murder, police): 0.06-0.12
   - Moderately common (hospital, robot, time-travel): 0.02-0.05
   - Niche keywords (zeppelin, tundra): 0.005-0.015

Return ONLY a JSON array. No markdown. No explanation.
Each element: {{"keyword": "...", "topic_genre": "...", "pop_weight": 0.5}}

Keywords:
{json.dumps(keywords)}

Output JSON array now."""


def assign_genres_batch(
    client, model: str, keywords: list[str], genres: list[str]
) -> list[dict]:
    """Call LLM to assign genre + pop_weight for a batch of keywords."""
    prompt = _build_prompt(keywords, genres)
    
    response, _ = generate_content_resilient(
        client=client,
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.3,
            "thinking_config": {"thinking_budget": 0},
        },
        timeout_sec=60,
        max_attempts=5,
        base_delay_sec=3,
        max_delay_sec=15,
    )
    
    raw = (response.text or "").strip()
    parsed = json.loads(raw)
    
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed)}")
    
    # Validate and clean
    valid_genres = set(genres)
    results = []
    for item in parsed:
        kw = item.get("keyword", "")
        tg = item.get("topic_genre", "Drama")
        pw = item.get("pop_weight", 0.5)
        
        if tg not in valid_genres:
            tg = "Drama"  # safe fallback
        pw = max(0.005, min(0.12, float(pw)))
        
        results.append({"keyword": kw, "topic_genre": tg, "pop_weight": pw})
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Assign topic_genre and pop_weight to keywords using LLM"
    )
    parser.add_argument("--auto", action="store_true", help="Skip prompts")
    parser.add_argument("--model", default=None, help=f"Override model (default: {MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="Re-assign ALL keywords, not just missing ones")
    parser.add_argument("--base-dir", default=None, help="Base directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else BASE_DIR
    edir = base_dir / "entities"
    csv_path = edir / "keyword.csv"
    
    if not csv_path.exists():
        print(f"  keyword.csv not found at {csv_path}")
        sys.exit(1)
    
    model = args.model or MODEL
    
    # Load keywords
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} keywords from {csv_path}")
    
    # Find keywords needing assignment
    if args.force:
        needs_assignment = df.index.tolist()
    else:
        mask = df["topic_genre"].isna() | df["pop_weight"].isna()
        needs_assignment = df[mask].index.tolist()
    
    if not needs_assignment:
        print("  All keywords already have topic_genre and pop_weight. Nothing to do.")
        return
    
    print(f"  {len(needs_assignment)} keywords need genre assignment")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  ERROR: No GOOGLE_API_KEY or GEMINI_API_KEY found.")
        sys.exit(1)
    
    if genai is None:
        print("  ERROR: google-genai not installed")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    
    # Process in batches
    keywords_to_process = df.loc[needs_assignment, "keyword"].tolist()
    n_batches = math.ceil(len(keywords_to_process) / BATCH_SIZE)
    total_in, total_out = 0, 0
    
    print(f"  Processing {len(keywords_to_process)} keywords in {n_batches} batches "
          f"using {model}")
    
    all_results = {}
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(keywords_to_process))
        batch_kws = keywords_to_process[start:end]
        
        print(f"  Batch {batch_idx + 1}/{n_batches} ({len(batch_kws)} keywords)...",
              end="", flush=True)
        
        try:
            results = assign_genres_batch(client, model, batch_kws, GENRES)
            # Map results back by keyword name
            for r in results:
                all_results[r["keyword"]] = r
            print(f" OK ({len(results)} assigned)")
        except Exception as exc:
            print(f" ERROR: {exc}")
            # Fallback: assign Drama + 0.3 for failed batch
            for kw in batch_kws:
                if kw not in all_results:
                    all_results[kw] = {
                        "keyword": kw,
                        "topic_genre": "Drama",
                        "pop_weight": 0.3,
                    }
    
    # Apply results back to DataFrame
    updated = 0
    for idx in needs_assignment:
        kw_name = df.at[idx, "keyword"]
        if kw_name in all_results:
            r = all_results[kw_name]
            df.at[idx, "topic_genre"] = r["topic_genre"]
            df.at[idx, "pop_weight"] = r["pop_weight"]
            updated += 1
    
    # Save
    df.to_csv(csv_path, index=False)
    print(f"  Updated {updated} keywords in {csv_path}")
    
    # Also update keywords.json if it exists
    json_path = edir / "keywords.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                kw_json = json.load(f)
            
            json_updated = 0
            for item in kw_json:
                kw_name = item.get("keyword", item.get("name", ""))
                if kw_name in all_results:
                    r = all_results[kw_name]
                    item["topic_genre"] = r["topic_genre"]
                    item["pop_weight"] = r["pop_weight"]
                    json_updated += 1
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(kw_json, f, ensure_ascii=False, indent=2)
            print(f"  Updated {json_updated} keywords in {json_path}")
        except Exception as exc:
            print(f"  Warning: could not update keywords.json: {exc}")
    
    # Summary
    remaining = df["topic_genre"].isna().sum()
    print(f"\n  Summary: {updated} assigned, {remaining} still missing")
    if remaining > 0:
        print(f"  WARNING: {remaining} keywords still have no topic_genre")


if __name__ == "__main__":
    main()
