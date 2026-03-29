#!/usr/bin/env python3
"""Procedural character name bank generator — 1M+ unique character names.

Generates entities/character_bank.csv with (character_name, archetype).
Reuses nationality name pools from name_banks.py for cultural authenticity.

Usage:
    python generate_character_bank.py --target 500000
    python generate_character_bank.py --target 1000000 --seed 42
"""
import csv, argparse, random, os, sys, time
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

sys.path.insert(0, str(BASE_DIR))
from name_banks import NAME_BANKS

# ── Archetypes with weights ──────────────────────────────────────────
ARCHETYPES = {
    "Lead Hero": 0.12, "Lead Villain": 0.10, "Love Interest": 0.06,
    "Sidekick": 0.06, "Mentor": 0.07, "Supporting": 0.18,
    "Extra": 0.13, "Authority Figure": 0.07, "Henchman": 0.06,
    "Comic Relief": 0.05, "Victim": 0.05, "Mysterious Stranger": 0.05,
}

# ── Title prefixes (applied ~15% of the time) ────────────────────────
TITLES_M = ["Dr.", "Prof.", "Sgt.", "Capt.", "Det.", "Officer", "Agent",
            "Brother", "Father", "Judge", "Chief", "General", "Col.",
            "Commissioner", "Mayor", "Senator", "King", "Prince", "Lord"]
TITLES_F = ["Dr.", "Prof.", "Sgt.", "Capt.", "Det.", "Officer", "Agent",
            "Sister", "Mother", "Judge", "Chief", "General", "Col.",
            "Commissioner", "Mayor", "Senator", "Queen", "Princess", "Lady"]
TITLES_N = ["Dr.", "Prof.", "Sgt.", "Capt.", "Det.", "Officer", "Agent",
            "Judge", "Chief", "Commissioner", "Mayor", "Senator"]
TITLE_PROB = 0.15

# ── Nicknames (applied ~8% of the time) ──────────────────────────────
NICKNAMES = [
    "The Hammer", "The Ghost", "The Fox", "The Shadow", "The Blade",
    "The Viper", "The Jackal", "The Raven", "The Wolf", "The Shark",
    "The Phoenix", "The Iron", "The Storm", "The Silence", "The Ace",
    "The Oracle", "The Reaper", "The Nomad", "The Hawk", "The Frost",
    "The Zodiac", "The Spider", "The Flash", "The Torch", "The Serpent",
    "The Panther", "The Eagle", "The Bull", "The Tiger", "The Bear",
    "The Wraith", "The Cipher", "The Drifter", "The Catalyst", "The Aegis",
    "The Maven", "The Pariah", "The Specter", "The Sentinel", "The Warden",
]
NICKNAME_PROB = 0.08

# ── "The [X]" solo names for mysterious/villain types (~3%) ──────────
THE_NAMES = [
    "The Collector", "The Architect", "The Butcher", "The Chemist",
    "The Doctor", "The Engineer", "The Fixer", "The Handler",
    "The Informant", "The Janitor", "The Keeper", "The Librarian",
    "The Mechanic", "The Navigator", "The Operator", "The Professor",
    "The Quartermaster", "The Ringmaster", "The Surgeon", "The Watchmaker",
    "The Alchemist", "The Broker", "The Cardinal", "The Diplomat",
    "The Executioner", "The Forger", "The Governor", "The Harbinger",
    "The Inquisitor", "The Judge", "The Knight", "The Linguist",
    "The Marquis", "The Nobleman", "The Oracle", "The Patriarch",
    "The Queen", "The Regent", "The Shepherd", "The Treasurer",
]
THE_NAME_ARCHETYPES = {"Lead Villain", "Mysterious Stranger", "Mentor", "Authority Figure"}

# ── Suffixes ─────────────────────────────────────────────────────────
SUFFIXES = ["Jr.", "Sr.", "II", "III"]
SUFFIX_PROB = 0.02


def generate_character_name(rng, bank, gender, archetype, used_names):
    """Generate a unique character name."""
    first_pool = bank["first_m"] if gender == "M" else bank["first_f"]
    surname_pool = bank["surnames"]

    for _ in range(100):
        first = rng.choice(first_pool)
        last = rng.choice(surname_pool)

        # Title prefix
        title = ""
        if rng.random() < TITLE_PROB:
            pool = TITLES_M if gender == "M" else (TITLES_F if gender == "F" else TITLES_N)
            title = rng.choice(pool) + " "

        # Nickname insertion
        nick = ""
        if rng.random() < NICKNAME_PROB:
            nick = f" '{rng.choice(NICKNAMES)}'"

        # Suffix
        suffix = ""
        if rng.random() < SUFFIX_PROB:
            suffix = f" {rng.choice(SUFFIXES)}"

        # Double surname (less common than person names)
        if rng.random() < 0.08:
            last2 = rng.choice(surname_pool)
            if last2 != last:
                last = f"{last}-{last2}"

        name = f"{title}{first}{nick} {last}{suffix}"
        key = name.lower()

        if key not in used_names:
            used_names.add(key)
            return name

    return None


def generate_characters(target, seed=42):
    """Generate target number of unique character names with archetypes."""
    rng = random.Random(seed)
    used_names = set()
    characters = []

    arch_list = list(ARCHETYPES.keys())
    arch_weights = list(ARCHETYPES.values())

    # Build flat pools from all banks for "The X" names
    the_name_used = set()

    t0 = time.time()

    for i in range(target):
        archetype = rng.choices(arch_list, weights=arch_weights, k=1)[0]
        gender = rng.choices(["M", "F", "NB"], weights=[0.50, 0.42, 0.08], k=1)[0]

        # Small chance of "The X" solo name for villain/mystery archetypes
        if archetype in THE_NAME_ARCHETYPES and rng.random() < 0.03:
            available = [n for n in THE_NAMES if n.lower() not in the_name_used]
            if available:
                name = rng.choice(available)
                the_name_used.add(name.lower())
                characters.append({"character_name": name, "archetype": archetype})
                if (i + 1) % 100000 == 0:
                    print(f"  {i+1:>10,} / {target:,} ({time.time()-t0:.1f}s)")
                continue

        # Pick random nationality bank
        bank = rng.choice(NAME_BANKS)
        name = generate_character_name(rng, bank, gender, archetype, used_names)

        if name is None:
            # Try other banks
            for _ in range(5):
                bank = rng.choice(NAME_BANKS)
                name = generate_character_name(rng, bank, gender, archetype, used_names)
                if name:
                    break
            if name is None:
                print(f"  WARNING: Exhausted at {i}, stopping")
                break

        characters.append({"character_name": name, "archetype": archetype})

        if (i + 1) % 100000 == 0:
            print(f"  {i+1:>10,} / {target:,} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Generated {len(characters):,} unique characters in {elapsed:.1f}s")
    return characters


def main():
    parser = argparse.ArgumentParser(description="Procedural character bank generator")
    parser.add_argument("--target", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ENTITY_DIR / "character_bank.csv"))
    args = parser.parse_args()

    print("=" * 60)
    print("  PROCEDURAL CHARACTER BANK GENERATOR")
    print("=" * 60)
    print(f"  Target:  {args.target:,}")
    print(f"  Seed:    {args.seed}")
    print("=" * 60)

    characters = generate_characters(args.target, args.seed)

    # Stats
    archetypes = Counter(c["archetype"] for c in characters)
    print(f"\n  Archetypes:")
    for a, count in archetypes.most_common():
        print(f"    {a:<22} {count:>8,} ({count/len(characters)*100:.1f}%)")

    titled = sum(1 for c in characters if any(c["character_name"].startswith(t)
                 for t in ["Dr.", "Prof.", "Sgt.", "Capt.", "Det.", "Officer",
                           "Agent", "Judge", "Chief", "General", "Col."]))
    nicked = sum(1 for c in characters if "'" in c["character_name"])
    print(f"\n  Titled names: {titled:,} ({titled/len(characters)*100:.1f}%)")
    print(f"  Nicknamed:    {nicked:,} ({nicked/len(characters)*100:.1f}%)")

    # Save
    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["character_name", "archetype"])
        writer.writeheader()
        writer.writerows(characters)
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
