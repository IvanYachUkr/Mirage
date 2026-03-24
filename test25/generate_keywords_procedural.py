#!/usr/bin/env python3
"""Procedural keyword generator — creates entities/keywords.json.

Usage:
    python generate_keywords_procedural.py --target 900
    python generate_keywords_procedural.py --target 900 --seed 42
"""
import json, argparse, random, os, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

sys.path.insert(0, str(BASE_DIR))
from contracts import GENRES

# ── Keyword pools by genre ───────────────────────────────────────────
KEYWORD_POOLS = {
    "Drama": ["betrayal", "family-secrets", "coming-of-age", "grief", "forbidden-love",
              "class-struggle", "immigrant-experience", "addiction", "divorce", "inheritance",
              "terminal-illness", "sibling-rivalry", "small-town", "working-class", "custody-battle",
              "poverty", "ambition", "sacrifice", "loneliness", "redemption"],
    "Action": ["heist", "car-chase", "explosion", "martial-arts", "hostage",
               "mercenary", "arms-dealer", "bounty-hunter", "prison-break", "undercover",
               "bank-robbery", "assassination", "smuggling", "escape", "rescue-mission",
               "gunfight", "demolition", "vigilante", "combat", "warfare"],
    "Comedy": ["mistaken-identity", "road-trip", "wedding-disaster", "odd-couple", "prank",
               "fish-out-of-water", "workplace-comedy", "family-reunion", "blind-date", "slapstick",
               "satire", "parody", "farce", "situational-comedy", "dark-humor",
               "culture-clash", "bachelor-party", "holiday-chaos", "school-reunion", "neighbors"],
    "Thriller": ["conspiracy", "serial-killer", "surveillance", "double-cross", "kidnapping",
                 "witness-protection", "psychological-manipulation", "stalker", "blackmail", "espionage",
                 "bombing", "cyberattack", "corruption", "cover-up", "manhunt",
                 "interrogation", "paranoia", "obsession", "sabotage", "infiltration"],
    "Horror": ["haunted-house", "possession", "slasher", "zombie", "supernatural",
               "cult", "isolated-cabin", "cursed-object", "body-horror", "found-footage",
               "demonic", "vampire", "werewolf", "ghost", "nightmare",
               "outbreak", "experiment-gone-wrong", "ritual", "asylum", "cemetery"],
    "Sci-Fi": ["time-travel", "artificial-intelligence", "alien-invasion", "parallel-universe", "space-exploration",
               "dystopia", "clone", "robot", "virtual-reality", "terraforming",
               "first-contact", "genetic-engineering", "nanotechnology", "singularity", "wormhole",
               "cyberpunk", "post-apocalyptic", "android", "space-station", "quantum"],
    "Romance": ["love-triangle", "unrequited-love", "long-distance", "second-chance", "enemies-to-lovers",
                "forbidden-romance", "holiday-romance", "workplace-romance", "childhood-friends", "arranged-marriage",
                "summer-love", "pen-pal", "cross-cultural", "age-gap", "rekindled-flame"],
    "Fantasy": ["prophecy", "quest", "chosen-one", "magic", "dragon",
                "kingdom", "enchantment", "dark-lord", "sword-and-sorcery", "mythological",
                "fairy-tale", "portal", "ancient-artifact", "elemental-magic", "shapeshifter"],
    "Mystery": ["whodunit", "locked-room", "cold-case", "detective", "missing-person",
                "crime-scene", "alibi", "forensic", "puzzle", "red-herring",
                "amateur-sleuth", "police-procedural", "unsolved-crime", "witness", "evidence"],
    "Documentary": ["nature", "true-crime", "political", "social-justice", "environmental",
                    "music-documentary", "sports-documentary", "war-documentary", "biographical", "investigation",
                    "cultural", "historical", "science", "technology", "art"],
    "Crime": ["organized-crime", "drug-cartel", "money-laundering", "murder", "robbery",
              "gang-war", "dirty-cop", "witness-intimidation", "crime-boss", "evidence-tampering",
              "prison", "trial", "defense-attorney", "prosecutor", "informant"],
    "Animation": ["talking-animals", "magical-world", "friendship", "adventure-quest", "transformation",
                  "forest-spirit", "ocean-voyage", "toy-world", "dream-world", "flying"],
    "War": ["trenches", "resistance", "d-day", "prisoner-of-war", "civil-war",
            "guerrilla", "battlefield", "veterans", "occupation", "liberation"],
    "Western": ["outlaw", "gold-rush", "frontier", "sheriff", "saloon",
                "train-robbery", "cattle-drive", "duel", "bounty", "homestead"],
}

# ── Generic cross-genre keywords ─────────────────────────────────────
GENERIC_KEYWORDS = [
    "revenge", "survival", "justice", "deception", "identity", "transformation",
    "fate", "destiny", "courage", "betrayal", "discovery", "legacy", "honor",
    "freedom", "power", "greed", "jealousy", "trust", "loyalty", "duty",
    "obsession", "regret", "forgiveness", "guilt", "pride", "hope", "despair",
    "truth", "lies", "violence", "peace", "chaos", "order", "rebellion",
    "tradition", "progress", "nature", "technology", "memory", "time",
]


def generate_keywords(target, seed=42):
    rng = random.Random(seed)
    keywords = []
    used = set()

    # First: one from each genre pool
    for genre, pool in KEYWORD_POOLS.items():
        for kw in pool:
            if len(keywords) >= target:
                break
            if kw not in used:
                used.add(kw)
                keywords.append({
                    "keyword_id": len(keywords) + 1,
                    "keyword": kw,
                    "topic_genre": genre,
                    "pop_weight": round(rng.uniform(0.3, 1.0), 3),
                })

    # Then: generic keywords
    for kw in GENERIC_KEYWORDS:
        if len(keywords) >= target:
            break
        if kw not in used:
            used.add(kw)
            keywords.append({
                "keyword_id": len(keywords) + 1,
                "keyword": kw,
                "topic_genre": rng.choice(GENRES),
                "pop_weight": round(rng.uniform(0.3, 1.0), 3),
            })

    # Fill remaining with combinations
    modifiers = ["dark", "epic", "secret", "lost", "final", "ancient", "modern",
                 "urban", "rural", "coastal", "mountain", "underground", "royal"]
    bases = ["conflict", "journey", "mystery", "pursuit", "alliance", "rivalry",
             "crisis", "encounter", "mission", "ceremony", "competition", "trial"]

    while len(keywords) < target:
        kw = f"{rng.choice(modifiers)}-{rng.choice(bases)}"
        if kw not in used:
            used.add(kw)
            keywords.append({
                "keyword_id": len(keywords) + 1,
                "keyword": kw,
                "topic_genre": rng.choice(GENRES),
                "pop_weight": round(rng.uniform(0.2, 0.8), 3),
            })

    return keywords


def main():
    parser = argparse.ArgumentParser(description="Procedural keyword generator")
    parser.add_argument("--target", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ENTITY_DIR / "keywords.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("  PROCEDURAL KEYWORD GENERATOR")
    print("=" * 60)
    print(f"  Target:  {args.target}")
    print("=" * 60)

    keywords = generate_keywords(args.target, args.seed)

    from collections import Counter
    genres = Counter(k["topic_genre"] for k in keywords)
    print(f"  Generated {len(keywords)} keywords")
    print(f"  Genre distribution: {dict(genres.most_common(10))}")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {args.out}")


if __name__ == "__main__":
    main()
