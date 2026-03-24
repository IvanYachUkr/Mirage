#!/usr/bin/env python3
"""Procedural person stub generator — 400K+ unique names, zero LLM calls.

Outputs entities/person_stubs.json with (name, nationality, gender, roles, career_stage).
LLM enrichment (bios, style_tags) is done separately by generate_persons_llm.py.

Usage:
    python generate_persons_procedural.py --target 400000
    python generate_persons_procedural.py --target 400000 --seed 42
"""
import json, argparse, random, os, sys, time
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

sys.path.insert(0, str(BASE_DIR))
from contracts import CAREER_STAGES, GENRES, MARKETS
from name_banks import NAME_BANKS

# ── Distributions ────────────────────────────────────────────────────
ROLE_DISTRIBUTION = {
    "actor": 0.60, "director": 0.06, "actor,director": 0.04,
    "producer": 0.06, "writer": 0.05, "writer,director": 0.02,
    "cinematographer": 0.05, "editor": 0.05, "composer": 0.04,
    "production_designer": 0.03,
}
STAGE_WEIGHTS = {"rising": 0.30, "prime": 0.35, "veteran": 0.20, "legend": 0.10, "retired": 0.05}
GENDER_WEIGHTS = {"M": 0.45, "F": 0.45, "NB": 0.10}

# Suffixes applied with low probability for richness
SUFFIXES = ["Jr.", "Sr.", "II", "III", "IV"]
SUFFIX_PROB = 0.03  # 3% of persons get a suffix

# Double-surname probability (culture-dependent)
DOUBLE_SURNAME_CULTURES = {
    "Spanish", "Mexican", "Colombian", "Venezuelan", "Peruvian", "Chilean",
    "Argentine", "Cuban", "Puerto Rican", "Ecuadorian", "Bolivian",
    "Portuguese", "Brazilian", "Filipino", "Catalan", "Basque", "Galician",
}
DOUBLE_SURNAME_PROB_CULTURE = 0.70  # 70% for Hispanic/Lusophone cultures
DOUBLE_SURNAME_PROB_OTHER = 0.06    # 6% for others (married names, etc.)

# Middle name/initial probability
MIDDLE_NAME_PROB = 0.15
MIDDLE_INITIAL_PROB = 0.10

# Market fit by region
REGION_MARKETS = {
    "north_america": ["North America", "Global"],
    "europe": ["Europe", "Global"],
    "east_asia": ["Asia", "Global"],
    "south_asia": ["Asia", "Regional"],
    "southeast_asia": ["Asia", "Regional"],
    "middle_east": ["Middle East", "Regional"],
    "africa": ["Africa", "Regional"],
    "latin_america": ["Latin America", "South America", "Regional"],
    "oceania": ["Oceania", "Regional"],
}


def _weighted_choice(rng, items_weights):
    """Pick from {item: weight} dict."""
    items = list(items_weights.keys())
    weights = list(items_weights.values())
    return rng.choices(items, weights=weights, k=1)[0]


def generate_name(rng, bank, gender, used_names):
    """Generate a unique full name from a nationality's name bank."""
    first_pool = bank["first_m"] if gender == "M" else bank["first_f"]
    if gender == "NB":
        first_pool = bank.get("first_nb", bank["first_m"] + bank["first_f"])
    surname_pool = bank["surnames"]
    nat = bank.get("nationality", "")
    region = bank.get("region", "europe")

    for _ in range(200):  # max attempts per name
        first = rng.choice(first_pool)
        last = rng.choice(surname_pool)

        # Double surname
        is_double_culture = nat in DOUBLE_SURNAME_CULTURES
        double_prob = DOUBLE_SURNAME_PROB_CULTURE if is_double_culture else DOUBLE_SURNAME_PROB_OTHER
        if rng.random() < double_prob:
            last2 = rng.choice(surname_pool)
            while last2 == last:
                last2 = rng.choice(surname_pool)
            sep = " " if is_double_culture else "-"
            last = f"{last}{sep}{last2}"

        # Middle name or initial
        r = rng.random()
        middle = ""
        if r < MIDDLE_NAME_PROB:
            middle = rng.choice(first_pool)
            while middle == first:
                middle = rng.choice(first_pool)
            middle = f" {middle}"
        elif r < MIDDLE_NAME_PROB + MIDDLE_INITIAL_PROB:
            letter = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            middle = f" {letter}."

        # Suffix
        suffix = ""
        if rng.random() < SUFFIX_PROB:
            suffix = f" {rng.choice(SUFFIXES)}"

        full_name = f"{first}{middle} {last}{suffix}"
        key = full_name.lower()
        if key not in used_names:
            used_names.add(key)
            return full_name

    return None  # exhausted attempts


def generate_persons(target, seed=42):
    """Generate target number of unique person stubs."""
    rng = random.Random(seed)
    used_names = set()
    persons = []

    # Build weighted nationality list from name banks
    nat_weights = {b["nationality"]: b.get("weight", 1.0) for b in NAME_BANKS}
    nat_index = {b["nationality"]: b for b in NAME_BANKS}
    nat_list = list(nat_weights.keys())
    nat_w = list(nat_weights.values())

    t0 = time.time()

    for i in range(target):
        nat = rng.choices(nat_list, weights=nat_w, k=1)[0]
        bank = nat_index[nat]
        gender = _weighted_choice(rng, GENDER_WEIGHTS)
        role_combo = _weighted_choice(rng, ROLE_DISTRIBUTION)
        roles = [r.strip() for r in role_combo.split(",")]
        career_stage = _weighted_choice(rng, STAGE_WEIGHTS)

        name = generate_name(rng, bank, gender, used_names)
        if name is None:
            # Pool exhausted for this nationality — try another
            for _ in range(10):
                nat = rng.choices(nat_list, weights=nat_w, k=1)[0]
                bank = nat_index[nat]
                name = generate_name(rng, bank, gender, used_names)
                if name:
                    break
            if name is None:
                print(f"  WARNING: Could not generate unique name at {i}, stopping")
                break

        region = bank.get("region", "europe")
        markets = REGION_MARKETS.get(region, ["Regional"])
        market_fit = [rng.choice(markets)]
        if rng.random() < 0.15:
            market_fit.append("Global")

        persons.append({
            "person_id": i + 1,
            "name": name,
            "nationality": nat,
            "gender": gender,
            "bio": "",
            "style_tags": [],
            "genre_affinity": [],
            "roles": roles,
            "career_stage": career_stage,
            "market_fit": list(set(market_fit)),
        })

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1:>8,} / {target:,} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Generated {len(persons):,} unique persons in {elapsed:.1f}s")
    return persons


def main():
    parser = argparse.ArgumentParser(description="Procedural person stub generator")
    parser.add_argument("--target", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ENTITY_DIR / "persons.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("  PROCEDURAL PERSON GENERATOR")
    print("=" * 60)
    print(f"  Target:  {args.target:,}")
    print(f"  Seed:    {args.seed}")
    print(f"  Banks:   {len(NAME_BANKS)} nationalities")

    # Show capacity
    total_capacity = 0
    for b in NAME_BANKS:
        fm = len(b.get("first_m", []))
        ff = len(b.get("first_f", []))
        s = len(b["surnames"])
        total_capacity += (fm + ff) * s
    print(f"  Capacity: ~{total_capacity:,} base combos (before modifiers)")
    print("=" * 60)

    persons = generate_persons(args.target, args.seed)

    # Stats
    genders = Counter(p["gender"] for p in persons)
    stages = Counter(p["career_stage"] for p in persons)
    roles = Counter()
    for p in persons:
        for r in p["roles"]:
            roles[r] += 1
    nats = Counter(p["nationality"] for p in persons)
    doubles = sum(1 for p in persons if "-" in p["name"] or p["name"].count(" ") > 1)

    print(f"\n  Gender:  {dict(genders.most_common())}")
    print(f"  Stages:  {dict(stages.most_common())}")
    print(f"  Roles:   {dict(roles.most_common(10))}")
    print(f"  Top nats: {dict(nats.most_common(10))}")
    print(f"  Double/compound names: {doubles:,} ({doubles/len(persons)*100:.1f}%)")
    print(f"  Unique names: {len(set(p['name'] for p in persons)):,}")

    # Save
    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(persons, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
