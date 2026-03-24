"""
LLM enrichment for procedurally generated persons.

Reads entities/persons.json produced by generate_persons_procedural.py and fills:
  - bio
  - style_tags
  - genre_affinity
  - market_fit

Names, nationality, gender, roles, and career stage stay procedural.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from contracts import (
    DIRECTOR_STYLES,
    GENRES,
    MARKETS,
    MODEL_TIERS,
    STYLE_TAGS,
    load_json_batch,
    normalize_style_tag,
    save_json_batch,
    validate_batch,
    validate_person,
)
from entities_to_csv import normalize_person_record
from llm_provider import get_llm_client

BASE_DIR = Path(__file__).resolve().parent
ENTITY_DIR = BASE_DIR / "entities"
DEFAULT_MODEL = MODEL_TIERS.get("entity_gen", "gemini-3.1-flash-lite-preview")
DEFAULT_BATCH_SIZE = 40


def _parse_json_response(text: str) -> list[dict]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:])
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed)}")
    return parsed


def _normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.replace("|", ",").replace(";", ",")
        return [part.strip() for part in text.split(",") if part.strip()]
    return [str(value).strip()]


def _fallback_fields(person: dict) -> dict:
    roles = [str(r).strip() for r in person.get("roles", []) if str(r).strip()]
    primary = roles[0] if roles else "actor"
    stage = str(person.get("career_stage", "prime"))
    nationality = str(person.get("nationality", "American"))
    name = str(person.get("name", f"Person {person.get('person_id', '?')}"))
    market_fit = _normalize_string_list(person.get("market_fit", [])) or ["Regional"]

    if "director" in roles:
        styles = ["atmospheric", "intimate"]
        genres = ["Drama", "Thriller"]
    elif "writer" in roles:
        styles = ["cerebral", "understated"]
        genres = ["Drama", "Mystery"]
    elif "composer" in roles:
        styles = ["lyrical", "theatrical"]
        genres = ["Drama", "Fantasy"]
    elif "cinematographer" in roles:
        styles = ["visual-spectacle", "naturalistic"]
        genres = ["Drama", "Thriller"]
    else:
        styles = ["naturalistic", "magnetic"]
        genres = ["Drama", "Comedy"]

    bio = (
        f"{name} is a {nationality.lower()} {primary} in the {stage} stage of their career, "
        f"known for disciplined collaboration and a clear on-screen identity. Their work tends "
        f"to fit {', '.join(g.lower() for g in genres[:2])} projects with strong ensemble energy."
    )
    return {
        "bio": bio,
        "style_tags": styles,
        "genre_affinity": genres,
        "market_fit": market_fit[:2],
    }


def _sanitize_enrichment(person: dict, patch: dict) -> dict:
    combined_styles = list(STYLE_TAGS) + list(DIRECTOR_STYLES)
    fallback = _fallback_fields(person)

    bio = str(patch.get("bio", "") or "").strip()
    if len(bio) < 40:
        bio = fallback["bio"]

    styles: list[str] = []
    for tag in _normalize_string_list(patch.get("style_tags", [])):
        norm = normalize_style_tag(tag, vocab=combined_styles)
        if norm and norm not in styles:
            styles.append(norm)
    if not styles:
        styles = fallback["style_tags"]
    styles = styles[:4]

    genres: list[str] = []
    for genre in _normalize_string_list(patch.get("genre_affinity", [])):
        for valid in GENRES:
            if genre.strip().lower() == valid.lower() and valid not in genres:
                genres.append(valid)
                break
    if not genres:
        genres = fallback["genre_affinity"]
    genres = genres[:3]

    market_fit: list[str] = []
    for market in _normalize_string_list(patch.get("market_fit", [])):
        for valid in MARKETS:
            if market.strip().lower() == valid.lower() and valid not in market_fit:
                market_fit.append(valid)
                break
    if not market_fit:
        market_fit = _normalize_string_list(person.get("market_fit", [])) or fallback["market_fit"]
    market_fit = market_fit[:2]

    return {
        "bio": bio,
        "style_tags": styles,
        "genre_affinity": genres,
        "market_fit": market_fit,
    }


def _build_prompt(batch: list[dict]) -> str:
    style_vocab = ", ".join(list(STYLE_TAGS) + list(DIRECTOR_STYLES))
    genre_vocab = ", ".join(GENRES)
    market_vocab = ", ".join(MARKETS)

    lines = []
    for person in batch:
        lines.append(
            f"[{person['person_id']}] {person['name']} | nationality: {person['nationality']} | "
            f"gender: {person['gender']} | roles: {', '.join(person.get('roles', []))} | "
            f"career_stage: {person.get('career_stage', 'prime')} | "
            f"current_market_fit: {', '.join(_normalize_string_list(person.get('market_fit', [])))}"
        )

    people_block = "\n".join(lines)
    return f"""You are enriching a synthetic film-industry person roster.

For each person below, keep the existing identity fixed and add four fields:
1. bio: 2-3 vivid Wikipedia-style sentences, 45-95 words total
2. style_tags: 2-4 tags chosen only from this list: {style_vocab}
3. genre_affinity: 1-3 genres chosen only from this list: {genre_vocab}
4. market_fit: 1-2 markets chosen only from this list: {market_vocab}

Important rules:
- Do NOT invent or change names, nationality, gender, roles, or career_stage.
- Match the bio to the person's role mix and career stage.
- Make the bios varied and specific rather than templated.
- Director / writer / composer people can use behind-the-camera style tags when appropriate.
- Output ONLY a JSON array.

People:
{people_block}

Return JSON array of:
{{
  "person_id": 123,
  "bio": "...",
  "style_tags": ["..."],
  "genre_affinity": ["..."],
  "market_fit": ["..."]
}}
"""


def enrich_persons(
    base_dir: Path,
    *,
    model: str | None,
    batch_size: int,
    force: bool,
) -> None:
    persons_path = base_dir / "entities" / "persons.json"
    if not persons_path.exists():
        raise FileNotFoundError(f"persons.json not found at {persons_path}")

    persons = load_json_batch(persons_path)
    by_id = {int(p["person_id"]): p for p in persons if "person_id" in p}
    pending = [
        p for p in persons
        if force
        or not str(p.get("bio", "") or "").strip()
        or not _normalize_string_list(p.get("style_tags", []))
        or not _normalize_string_list(p.get("genre_affinity", []))
    ]

    def _normalize_and_validate(current_persons: list[dict]) -> None:
        normalized = [normalize_person_record(person) for person in current_persons]
        save_json_batch(normalized, persons_path)
        result = validate_batch(normalized, validate_person, "persons")
        if result["invalid"]:
            raise ValueError(
                f"Person enrichment finished with {result['invalid']} invalid records; "
                "inspect persons.json before continuing."
            )
        print(f"Validated {result['valid']} persons")

    print(f"Loaded {len(persons)} persons")
    print(f"Need enrichment: {len(pending)}")
    if not pending:
        print("All persons already enriched.")
        _normalize_and_validate(persons)
        return

    llm = get_llm_client()
    effective_model = model or DEFAULT_MODEL
    total_batches = (len(pending) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch = pending[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch)} persons)...", flush=True)
        prompt = _build_prompt(batch)

        parsed: list[dict] | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = llm.generate(
                    prompt,
                    model=effective_model,
                    json_mode=True,
                    temperature=0.8,
                    max_tokens=16384,
                    timeout_sec=90,
                    max_attempts=5,
                )
                parsed = _parse_json_response(response.text)
                break
            except Exception as exc:
                last_error = exc
                print(f"    Retry {attempt + 1}/3 after error: {exc}")
                time.sleep(2 + attempt * 2)

        patches_by_id: dict[int, dict] = {}
        if parsed is not None:
            for item in parsed:
                try:
                    pid = int(item.get("person_id"))
                except Exception:
                    continue
                if pid in by_id:
                    patches_by_id[pid] = item
        elif last_error is not None:
            print(f"    Falling back for batch after repeated failure: {last_error}")

        for person in batch:
            pid = int(person["person_id"])
            patch = patches_by_id.get(pid, {})
            enriched = _sanitize_enrichment(person, patch)
            person.update(enriched)

        persons = [normalize_person_record(person) for person in persons]
        save_json_batch(persons, persons_path)

    _normalize_and_validate(persons)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-enrich procedural persons")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--auto", action="store_true", help="Accepted for runner compatibility")
    args = parser.parse_args()

    load_dotenv(BASE_DIR.parent / ".env")
    enrich_persons(
        Path(args.base_dir).resolve(),
        model=args.model,
        batch_size=max(1, int(args.batch_size)),
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
