#!/usr/bin/env python3
"""Procedural company generator — creates entities/companies.json.

Usage:
    python generate_companies_procedural.py --target 500
    python generate_companies_procedural.py --target 500 --seed 42
"""
import json, argparse, random, os, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

sys.path.insert(0, str(BASE_DIR))
from contracts import GENRES, COUNTRIES, COMPANY_TIERS, STYLE_TAGS, DIRECTOR_STYLES

# ── Company name components ──────────────────────────────────────────
PREFIXES = [
    "Apex", "Nova", "Zenith", "Eclipse", "Prism", "Titan", "Nebula", "Solaris",
    "Vortex", "Atlas", "Phoenix", "Onyx", "Stellar", "Crimson", "Azure", "Polar",
    "Emerald", "Silver", "Golden", "Iron", "Sapphire", "Obsidian", "Ivory", "Amber",
    "Cobalt", "Granite", "Velvet", "Diamond", "Platinum", "Mercury", "Neptune",
    "Horizon", "Summit", "Pinnacle", "Vertex", "Nexus", "Parallax", "Meridian",
    "Cascade", "Beacon", "Compass", "Lantern", "Voyager", "Pioneer", "Frontier",
    "Monarch", "Sentinel", "Citadel", "Bastion", "Aegis", "Arcane", "Mystic",
    "Tempest", "Avalon", "Elysium", "Olympus", "Pantheon", "Mirage", "Oasis",
    "Coral", "Safari", "Monsoon", "Bamboo", "Lotus", "Pagoda", "Silk", "Jade",
    "Dragon", "Tiger", "Falcon", "Raven", "Wolf", "Lion", "Eagle", "Hawk",
    "Metro", "Urban", "Skyline", "Coast", "Harbor", "River", "Valley", "Ridge",
    "Thunder", "Lightning", "Blaze", "Storm", "Frost", "Aurora", "Dawn", "Dusk",
    "Quantum", "Vector", "Matrix", "Cipher", "Binary", "Pixel", "Neon", "Retro",
]
SUFFIXES = [
    "Studios", "Pictures", "Films", "Entertainment", "Media", "Productions",
    "Cinema", "Motion Pictures", "Releasing", "International", "Worldwide",
    "Creative", "Vision", "Arts", "Works", "Group", "Company", "Collective",
]

# ── Country-region mapping for localized names ───────────────────────
REGION_COUNTRIES = {
    "north_america": ["USA", "Canada"],
    "europe": ["UK", "France", "Germany", "Italy", "Spain", "Sweden", "Norway",
               "Denmark", "Netherlands", "Belgium", "Switzerland", "Austria",
               "Ireland", "Poland", "Czech Republic", "Hungary", "Romania",
               "Greece", "Portugal", "Finland"],
    "east_asia": ["Japan", "South Korea", "China", "Taiwan", "Hong Kong"],
    "south_asia": ["India", "Pakistan", "Bangladesh", "Sri Lanka"],
    "southeast_asia": ["Thailand", "Vietnam", "Philippines", "Indonesia", "Malaysia"],
    "middle_east": ["Turkey", "Iran", "Egypt", "Israel", "Lebanon", "UAE", "Saudi Arabia"],
    "africa": ["Nigeria", "South Africa", "Kenya", "Ghana", "Ethiopia", "Morocco",
               "Senegal", "Congo", "Tanzania", "Uganda"],
    "latin_america": ["Mexico", "Brazil", "Argentina", "Colombia", "Chile", "Peru",
                      "Venezuela", "Cuba", "Dominican Republic"],
    "oceania": ["Australia", "New Zealand"],
}

TIER_WEIGHTS = {"Global": 0.05, "Major": 0.15, "Mid-Budget": 0.35, "Indie": 0.30, "Micro": 0.15}


def generate_companies(target, seed=42):
    rng = random.Random(seed)
    companies = []
    used_names = set()

    # Flatten country list
    all_countries = []
    for region, countries in REGION_COUNTRIES.items():
        for c in countries:
            if c in COUNTRIES:
                all_countries.append(c)
    if not all_countries:
        all_countries = list(COUNTRIES)

    tier_list = list(TIER_WEIGHTS.keys())
    tier_w = list(TIER_WEIGHTS.values())

    t0 = time.time()

    for i in range(target):
        # Generate unique name
        for _ in range(100):
            prefix = rng.choice(PREFIXES)
            suffix = rng.choice(SUFFIXES)
            name = f"{prefix} {suffix}"
            if name not in used_names:
                used_names.add(name)
                break
        else:
            # Add number to make unique
            name = f"{rng.choice(PREFIXES)} {rng.choice(SUFFIXES)} {i}"
            used_names.add(name)

        tier = rng.choices(tier_list, weights=tier_w, k=1)[0]
        country = rng.choice(all_countries)
        n_genres = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)[0]
        specialty = rng.sample(GENRES, min(n_genres, len(GENRES)))

        companies.append({
            "company_id": i + 1,
            "name": name,
            "country": country,
            "description": f"A {tier.lower()} production company specializing in {', '.join(g.lower() for g in specialty)} films.",
            "specialty_genres": specialty,
            "tier": tier,
            "preferred_actor_styles": rng.sample(STYLE_TAGS, min(rng.randint(1, 3), len(STYLE_TAGS))),
            "preferred_director_styles": rng.sample(DIRECTOR_STYLES, min(rng.randint(1, 3), len(DIRECTOR_STYLES))),
        })

    elapsed = time.time() - t0
    print(f"  Generated {len(companies)} companies in {elapsed:.1f}s")
    return companies


def main():
    parser = argparse.ArgumentParser(description="Procedural company generator")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ENTITY_DIR / "companies.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("  PROCEDURAL COMPANY GENERATOR")
    print("=" * 60)
    print(f"  Target:  {args.target}")
    print("=" * 60)

    companies = generate_companies(args.target, args.seed)

    from collections import Counter
    tiers = Counter(c["tier"] for c in companies)
    print(f"  Tiers: {dict(tiers.most_common())}")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(companies, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {args.out}")


if __name__ == "__main__":
    main()
