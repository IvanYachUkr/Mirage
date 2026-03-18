# -*- coding: utf-8 -*-
"""
V17: Backfill avoid_genres for existing person latent records.
==============================================================
Reads persons.json + persons_latent.json, finds persons missing avoid_genres,
batches them, asks the LLM for just avoid_genres, and patches the latent file.

Usage:
    python backfill_avoid_genres.py          # interactive
    python backfill_avoid_genres.py --auto   # skip prompts
    python backfill_avoid_genres.py --model gemini-2.5-flash-lite
"""
import argparse, os, json, sys, time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from api_resilience import generate_content_resilient
from contracts import GENRES

load_dotenv()

MODEL = "gemini-3.1-flash-lite-preview"
BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"
BATCH_SIZE = 80  # larger batches ok since we only ask for 1 field
MAX_RETRIES = 5
API_TIMEOUT = 60

def build_avoid_genres_prompt(batch):
    """Build a focused prompt asking only for avoid_genres."""
    lines = []
    for p in batch:
        pid = p["person_id"]
        name = p.get("name", "?")
        bio = str(p.get("bio", ""))[:150]
        styles = p.get("style_tags", [])
        if isinstance(styles, str):
            styles = [s.strip() for s in styles.split(",")]
        genres_aff = p.get("genre_affinity", [])
        if isinstance(genres_aff, str):
            genres_aff = [g.strip() for g in genres_aff.split(",")]
        lines.append(
            f'[{pid}] {name} | styles: {",".join(styles[:4])} '
            f'| genre_affinity: {",".join(genres_aff)} | bio: {bio}'
        )

    persons_block = "\n".join(lines)

    return f"""You are assigning genre avoidance data to movie industry persons based on their profiles.

For each person below, pick 1-3 genres from this EXACT list that the person would AVOID appearing in:
{GENRES}

Rules:
- Pick genres that CONFLICT with the person's style, genre_affinity, and bio
- Family-friendly actors avoid Horror/Thriller; cerebral actors avoid Action/Superhero
- Comedy specialists may avoid Documentary or War; action stars may avoid Musical or Romance
- Always pick at LEAST 1 genre. Maximum 3.
- Use EXACT genre names from the list above

=== PERSONS ({len(batch)}) ===
{persons_block}

Output ONLY a JSON array. Each object: {{"person_id": <id>, "avoid_genres": ["Genre1", "Genre2"]}}
No markdown fences, no explanation."""


def parse_json_response(text):
    """Extract JSON array from response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Backfill avoid_genres for person latent records")
    parser.add_argument("--auto", action="store_true", help="Skip prompts")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without calling API")
    args = parser.parse_args()

    model = args.model or MODEL

    # Load data
    persons_path = ENTITY_DIR / "persons.json"
    latent_path = ENTITY_DIR / "persons_latent.json"

    with open(persons_path, "r", encoding="utf-8") as f:
        persons = json.load(f)
    with open(latent_path, "r", encoding="utf-8") as f:
        latent_data = json.load(f)

    # Build lookup
    person_by_id = {p["person_id"]: p for p in persons}
    latent_by_id = {lv["person_id"]: lv for lv in latent_data}

    # Find persons missing avoid_genres
    missing = []
    for lv in latent_data:
        pid = lv["person_id"]
        if "avoid_genres" not in lv or not lv["avoid_genres"]:
            p = person_by_id.get(pid)
            if p:
                missing.append(p)

    print(f"Total person latent records: {len(latent_data)}")
    print(f"Missing avoid_genres: {len(missing)}")
    print(f"Batches needed: {(len(missing) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"Model: {model}")

    if not missing:
        print("Nothing to backfill!")
        return

    if args.dry_run:
        print("Dry-run: would process above batches. Exiting.")
        return

    if not args.auto:
        resp = input("\nProceed? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    # Initialize API client
    client = genai.Client()
    config = types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )

    total_filled = 0
    total_errors = 0
    batches = [missing[i:i+BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]

    for bi, batch in enumerate(batches):
        prompt = build_avoid_genres_prompt(batch)
        print(f"\n  Batch {bi+1}/{len(batches)} ({len(batch)} persons)...", end="", flush=True)

        for attempt in range(MAX_RETRIES):
            try:
                response, _stats = generate_content_resilient(
                    client=client, model=model, contents=prompt, config=config,
                    max_attempts=3, timeout_sec=API_TIMEOUT,
                )
                results = parse_json_response(response.text)

                filled = 0
                for r in results:
                    pid = r.get("person_id")
                    ag = r.get("avoid_genres", [])
                    if pid and isinstance(ag, list) and len(ag) > 0:
                        # Validate genres
                        valid_ag = [g for g in ag if g in GENRES]
                        if valid_ag and pid in latent_by_id:
                            latent_by_id[pid]["avoid_genres"] = valid_ag
                            filled += 1

                total_filled += filled
                print(f" OK ({filled}/{len(batch)} filled)", flush=True)
                break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(f" retry({attempt+1}, {e})...", end="", flush=True)
                    time.sleep(wait)
                else:
                    print(f" FAILED after {MAX_RETRIES} attempts: {e}", flush=True)
                    total_errors += 1

        # Save checkpoint every 10 batches
        if (bi + 1) % 10 == 0:
            with open(latent_path, "w", encoding="utf-8") as f:
                json.dump(latent_data, f, indent=2, ensure_ascii=False)
            print(f"  [Checkpoint] Saved after batch {bi+1}", flush=True)

    # Final save
    with open(latent_path, "w", encoding="utf-8") as f:
        json.dump(latent_data, f, indent=2, ensure_ascii=False)

    # Verify
    final_missing = sum(1 for lv in latent_data if "avoid_genres" not in lv or not lv["avoid_genres"])
    print(f"\n{'='*60}")
    print(f"  BACKFILL COMPLETE")
    print(f"  Filled: {total_filled}")
    print(f"  Errors: {total_errors}")
    print(f"  Still missing: {final_missing}/{len(latent_data)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
