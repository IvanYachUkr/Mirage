#!/usr/bin/env python3
"""Procedural keyword generator that creates entities/keywords.json."""

import argparse
import json
import os
import random
import sys
from collections import Counter, deque
from pathlib import Path

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

sys.path.insert(0, str(BASE_DIR))
from contracts import GENRES


KEYWORD_POOLS = {
    "Drama": [
        "betrayal", "family-secrets", "coming-of-age", "grief", "forbidden-love",
        "class-struggle", "immigrant-experience", "addiction", "divorce", "inheritance",
        "terminal-illness", "sibling-rivalry", "small-town", "working-class", "custody-battle",
        "poverty", "ambition", "sacrifice", "loneliness", "redemption",
    ],
    "Action": [
        "heist", "car-chase", "explosion", "martial-arts", "hostage",
        "mercenary", "arms-dealer", "bounty-hunter", "prison-break", "undercover",
        "bank-robbery", "assassination", "smuggling", "escape", "rescue-mission",
        "gunfight", "demolition", "vigilante", "combat", "warfare",
    ],
    "Comedy": [
        "mistaken-identity", "road-trip", "wedding-disaster", "odd-couple", "prank",
        "fish-out-of-water", "workplace-comedy", "family-reunion", "blind-date", "slapstick",
        "satire", "parody", "farce", "situational-comedy", "dark-humor",
        "culture-clash", "bachelor-party", "holiday-chaos", "school-reunion", "neighbors",
    ],
    "Thriller": [
        "conspiracy", "serial-killer", "surveillance", "double-cross", "kidnapping",
        "witness-protection", "psychological-manipulation", "stalker", "blackmail", "espionage",
        "bombing", "cyberattack", "corruption", "cover-up", "manhunt",
        "interrogation", "paranoia", "obsession", "sabotage", "infiltration",
    ],
    "Horror": [
        "haunted-house", "possession", "slasher", "zombie", "supernatural",
        "cult", "isolated-cabin", "cursed-object", "body-horror", "found-footage",
        "demonic", "vampire", "werewolf", "ghost", "nightmare",
        "outbreak", "experiment-gone-wrong", "ritual", "asylum", "cemetery",
    ],
    "Sci-Fi": [
        "time-travel", "artificial-intelligence", "alien-invasion", "parallel-universe", "space-exploration",
        "dystopia", "clone", "robot", "virtual-reality", "terraforming",
        "first-contact", "genetic-engineering", "nanotechnology", "singularity", "wormhole",
        "cyberpunk", "post-apocalyptic", "android", "space-station", "quantum",
    ],
    "Romance": [
        "love-triangle", "unrequited-love", "long-distance", "second-chance", "enemies-to-lovers",
        "forbidden-romance", "holiday-romance", "workplace-romance", "childhood-friends", "arranged-marriage",
        "summer-love", "pen-pal", "cross-cultural", "age-gap", "rekindled-flame",
    ],
    "Fantasy": [
        "prophecy", "quest", "chosen-one", "magic", "dragon",
        "kingdom", "enchantment", "dark-lord", "sword-and-sorcery", "mythological",
        "fairy-tale", "portal", "ancient-artifact", "elemental-magic", "shapeshifter",
    ],
    "Mystery": [
        "whodunit", "locked-room", "cold-case", "detective", "missing-person",
        "crime-scene", "alibi", "forensic", "puzzle", "red-herring",
        "amateur-sleuth", "police-procedural", "unsolved-crime", "witness", "evidence",
    ],
    "Documentary": [
        "nature", "true-crime", "political", "social-justice", "environmental",
        "music-documentary", "sports-documentary", "war-documentary", "biographical", "investigation",
        "cultural", "historical", "science", "technology", "art",
    ],
    "Crime": [
        "organized-crime", "drug-cartel", "money-laundering", "murder", "robbery",
        "gang-war", "dirty-cop", "witness-intimidation", "crime-boss", "evidence-tampering",
        "prison", "trial", "defense-attorney", "prosecutor", "informant",
    ],
    "Animation": [
        "talking-animals", "magical-world", "friendship", "adventure-quest", "transformation",
        "forest-spirit", "ocean-voyage", "toy-world", "dream-world", "flying",
    ],
    "War": [
        "trenches", "resistance", "d-day", "prisoner-of-war", "civil-war",
        "guerrilla", "battlefield", "veterans", "occupation", "liberation",
    ],
    "Western": [
        "outlaw", "gold-rush", "frontier", "sheriff", "saloon",
        "train-robbery", "cattle-drive", "duel", "bounty", "homestead",
    ],
}

GENERIC_KEYWORDS = [
    "revenge", "survival", "justice", "deception", "identity", "transformation",
    "fate", "destiny", "courage", "betrayal", "discovery", "legacy", "honor",
    "freedom", "power", "greed", "jealousy", "trust", "loyalty", "duty",
    "obsession", "regret", "forgiveness", "guilt", "pride", "hope", "despair",
    "truth", "lies", "violence", "peace", "chaos", "order", "rebellion",
    "tradition", "progress", "nature", "technology", "memory", "time",
]

UNIVERSAL_QUALIFIERS = [
    "dark", "epic", "secret", "lost", "final", "ancient", "modern", "urban",
    "rural", "coastal", "mountain", "underground", "royal", "hidden", "forgotten",
    "midnight", "neon", "fractured", "forbidden", "shifting",
]

UNIVERSAL_CONTEXTS = [
    "conflict", "journey", "mystery", "pursuit", "alliance", "rivalry", "crisis",
    "encounter", "mission", "ceremony", "competition", "trial", "reckoning",
    "uprising", "aftermath", "legacy",
]

GENRE_CONTEXTS = {
    "Drama": ["household", "inheritance", "homecoming", "reckoning", "upbringing", "intimacy"],
    "Action": ["standoff", "raid", "siege", "extraction", "chase", "strike-team"],
    "Comedy": ["mishap", "mix-up", "celebration", "reunion", "scheme", "detour"],
    "Thriller": ["cover-up", "surveillance", "blackout", "ultimatum", "panic", "double-life"],
    "Horror": ["ritual", "curse", "nightfall", "haunting", "infection", "awakening"],
    "Sci-Fi": ["protocol", "anomaly", "signal", "frontier", "simulation", "catastrophe"],
    "Romance": ["courtship", "heartbreak", "reunion", "promise", "longing", "destiny"],
    "Fantasy": ["kingdom", "prophecy", "artifact", "realm", "sorcery", "awakening"],
    "Mystery": ["clue", "vanishing", "investigation", "alibi", "breakthrough", "cold-trail"],
    "Documentary": ["movement", "portrait", "chronicle", "archive", "witness", "debate"],
    "Crime": ["syndicate", "cartel", "informant", "getaway", "takedown", "underworld"],
    "Animation": ["wonder", "playground", "friendship", "voyage", "dreamscape", "discovery"],
    "War": ["frontline", "occupation", "campaign", "retreat", "resistance", "liberation"],
    "Western": ["frontier", "showdown", "homestead", "outpost", "dust-trail", "cattle-run"],
}

GENRE_QUALIFIERS = {
    "Drama": ["domestic", "working-class", "quiet", "aching", "intergenerational"],
    "Action": ["armed", "high-speed", "explosive", "rogue", "tactical"],
    "Comedy": ["awkward", "chaotic", "offbeat", "absurd", "playful"],
    "Thriller": ["paranoid", "volatile", "covert", "high-stakes", "shadow"],
    "Horror": ["cursed", "unnerving", "blood-soaked", "occult", "eerie"],
    "Sci-Fi": ["orbital", "synthetic", "posthuman", "quantum", "interstellar"],
    "Romance": ["bittersweet", "sweeping", "tender", "star-crossed", "intimate"],
    "Fantasy": ["mythic", "enchanted", "ancient", "celestial", "runic"],
    "Mystery": ["buried", "cryptic", "silent", "twisted", "unsolved"],
    "Documentary": ["investigative", "observational", "cultural", "historic", "grassroots"],
    "Crime": ["street-level", "criminal", "illicit", "hardened", "dirty"],
    "Animation": ["whimsical", "colorful", "storybook", "playful", "uplifting"],
    "War": ["wartime", "battle-scarred", "occupied", "heroic", "fractured"],
    "Western": ["dusty", "lawless", "lonesome", "frontier", "weathered"],
}


def _append_keyword(keywords, used, keyword, topic_genre, rng, low=0.2, high=0.8):
    if keyword in used:
        return False
    used.add(keyword)
    keywords.append(
        {
            "keyword_id": len(keywords) + 1,
            "keyword": keyword,
            "topic_genre": topic_genre,
            "pop_weight": round(rng.uniform(low, high), 3),
        }
    )
    return True


def _genre_expansion_candidates(genre, pool, rng):
    qualifiers = list(dict.fromkeys(GENRE_QUALIFIERS.get(genre, []) + UNIVERSAL_QUALIFIERS))
    contexts = list(dict.fromkeys(GENRE_CONTEXTS.get(genre, []) + UNIVERSAL_CONTEXTS))
    seeds = list(pool)

    rng.shuffle(seeds)
    rng.shuffle(qualifiers)
    rng.shuffle(contexts)

    candidates = []
    for seed in seeds:
        for qualifier in qualifiers[:12]:
            candidates.append(f"{qualifier}-{seed}")
        for context in contexts[:10]:
            candidates.append(f"{seed}-{context}")
        for qualifier in qualifiers[:8]:
            for context in contexts[:6]:
                candidates.append(f"{qualifier}-{context}-{seed}")

    rng.shuffle(candidates)
    return deque(candidates)


def generate_keywords(target, seed=42):
    rng = random.Random(seed)
    keywords = []
    used = set()

    for genre, pool in KEYWORD_POOLS.items():
        for kw in pool:
            if len(keywords) >= target:
                break
            _append_keyword(keywords, used, kw, genre, rng, low=0.3, high=1.0)

    for kw in GENERIC_KEYWORDS:
        if len(keywords) >= target:
            break
        _append_keyword(keywords, used, kw, rng.choice(GENRES), rng, low=0.3, high=1.0)

    expansion_queues = {
        genre: _genre_expansion_candidates(genre, pool, rng)
        for genre, pool in KEYWORD_POOLS.items()
    }

    while len(keywords) < target and any(expansion_queues.values()):
        progressed = False
        for genre in GENRES:
            queue = expansion_queues.get(genre)
            if not queue:
                continue
            while queue:
                kw = queue.popleft()
                if _append_keyword(keywords, used, kw, genre, rng):
                    progressed = True
                    break
            if len(keywords) >= target:
                break
        if not progressed:
            break

    overflow = 1
    while len(keywords) < target:
        genre = GENRES[(overflow - 1) % len(GENRES)]
        qualifier = UNIVERSAL_QUALIFIERS[(overflow - 1) % len(UNIVERSAL_QUALIFIERS)]
        context = UNIVERSAL_CONTEXTS[(overflow * 3 - 1) % len(UNIVERSAL_CONTEXTS)]
        kw = f"{qualifier}-{genre.lower().replace(' ', '-')}-{context}-{overflow:04d}"
        _append_keyword(keywords, used, kw, genre, rng)
        overflow += 1

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

    genres = Counter(k["topic_genre"] for k in keywords)
    print(f"  Generated {len(keywords)} keywords")
    print(f"  Genre distribution: {dict(genres.most_common(10))}")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {args.out}")


if __name__ == "__main__":
    main()
