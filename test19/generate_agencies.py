"""
V15 -- Generate and expand talent agencies, then assign all persons.
====================================================================
Two-step:
  Step 1: LLM generates new agencies covering underrepresented regions
           (keeps all existing 10, adds ~50 more)
  Step 2: LLM assigns every person to one agency based on nationality,
           career stage, and style -- in batches of 150

Existing 10 agencies are Hollywood-centric. We expand with regional
agencies covering Bollywood, Nollywood, East Asian, Latin American,
MENA, European, and Southeast Asian film industries.

Target: ~60 agencies for 23,914 persons ~= ~400 persons/agency (realistic:
  powerhouse agencies handle 1000+, boutiques handle 50-200)

Fixes vs any prior script:
  - gemini-3.1-flash-lite-preview + $0.25/$1.50 pricing
  - 70s timeout + ThreadPoolExecutor
  - 503 detection + exponential backoff
  - Incremental save every batch
  - Full resume support

Usage:
    python test15/generate_agencies.py               # both steps
    python test15/generate_agencies.py --step 1      # generate only
    python test15/generate_agencies.py --step 2      # assign only
    python test15/generate_agencies.py --auto        # no prompts
"""
import json, os, sys, time, argparse, concurrent.futures, random
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR       = Path(__file__).parent
ENTITY_DIR     = BASE_DIR / "entities"
AGENCIES_PATH  = ENTITY_DIR / "agencies.json"
PERSONS_PATH   = ENTITY_DIR / "persons.json"

MODEL   = "gemini-3.1-flash-lite-preview"
PRICING = {"input": 0.25, "output": 1.50}
API_TIMEOUT = 70

# ── API helpers ───────────────────────────────────────────────────────────────

def _timeout_call(client, prompt, config):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    f = pool.submit(client.models.generate_content,
                    model=MODEL, contents=prompt, config=config)
    try:
        result = f.result(timeout=API_TIMEOUT)
        pool.shutdown(wait=False, cancel_futures=True)
        return result
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"API exceeded {API_TIMEOUT}s")


def call_api(client, prompt, config, max_retries=5):
    for retry in range(max_retries):
        try:
            resp = _timeout_call(client, prompt, config)
        except TimeoutError as e:
            print(f"  TIMEOUT (retry {retry+1}): {e}")
            time.sleep(15 * (retry + 1))
            continue
        except Exception as e:
            is_503 = "503" in str(e) or "unavailable" in str(e).lower()
            print(f"  {'503 ' if is_503 else ''}API ERROR (retry {retry+1}): {e}")
            time.sleep((70 if is_503 else 5) * (retry + 1))
            continue

        usage = resp.usage_metadata
        inp = getattr(usage, "prompt_token_count", 0) or 0
        out = getattr(usage, "candidates_token_count", 0) or 0
        cost = inp / 1e6 * PRICING["input"] + out / 1e6 * PRICING["output"]
        print(f"  {inp:,}in+{out:,}out=${cost:.4f}")

        text = resp.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break
            if isinstance(data, list):
                return data, cost
        except json.JSONDecodeError:
            pass

        last = text.rfind("}")
        if last > 0:
            candidate = text[:last+1].rstrip().rstrip(",") + "]"
            if not candidate.startswith("["):
                candidate = "[" + candidate
            try:
                data = json.loads(candidate)
                print(f"  (recovered {len(data)} from truncated JSON)")
                return data, cost
            except json.JSONDecodeError:
                pass

        print(f"  PARSE ERROR (retry {retry+1}): bad JSON ({len(text)} chars)")
        time.sleep(2)

    raise RuntimeError(f"All {max_retries} retries failed")


def safe(s):
    return str(s).encode("ascii", "replace").decode()


# ── Step 1: generate new regional agencies ────────────────────────────────────

REGIONAL_BATCHES = [
    {
        "label": "Bollywood & Indian regional film agencies",
        "n": 12,
        "prompt_extra": (
            "Indian talent agencies covering:\n"
            "  - Mumbai Bollywood powerhouses (Hindi film industry)\n"
            "  - Hyderabad/Chennai agencies for Tollywood (Telugu) and Kollywood (Tamil)\n"
            "  - Kerala-based agencies for Malayalam cinema\n"
            "  - Pan-Indian agencies managing crossover stars\n"
            "Market reach should include India (regional) and Global (for crossover talent)."
        ),
    },
    {
        "label": "African and Nollywood agencies",
        "n": 8,
        "prompt_extra": (
            "African talent agencies covering:\n"
            "  - Lagos-based Nollywood powerhouses (Nigerian film)\n"
            "  - Pan-African agencies bridging Nollywood and international markets\n"
            "  - South African agencies (Johannesburg)\n"
            "  - Ghanaian (Ghallywood) and East African (Nairobi) agencies\n"
            "Market reach: Africa (regional), some Global crossover."
        ),
    },
    {
        "label": "East Asian agencies (Chinese, Korean, Japanese)",
        "n": 10,
        "prompt_extra": (
            "East Asian talent agencies covering:\n"
            "  - South Korean K-Wave (Hallyu) agencies -- very important, rivalling Hollywood\n"
            "  - Japanese major talent agencies (known for exclusive contracts)\n"
            "  - Chinese mainland agencies (large-scale, state-connected or private)\n"
            "  - Hong Kong and Taiwan boutique agencies\n"
            "  - Pan-Asian crossover agencies bridging Asian and Hollywood markets\n"
            "Market reach should vary: some Local/Regional, some Global."
        ),
    },
    {
        "label": "Latin American agencies",
        "n": 7,
        "prompt_extra": (
            "Latin American talent agencies covering:\n"
            "  - Brazilian agencies (São Paulo / Rio) for telenovela and cinema talent\n"
            "  - Mexican agencies (CDMX) linking to Hollywood crossover\n"
            "  - Argentine agencies (Buenos Aires) for Palme d'Or/arthouse talent\n"
            "  - Pan-Latin agencies bridging Spanish and Portuguese-language markets\n"
            "Market reach: Latin America (regional), some North America crossover."
        ),
    },
    {
        "label": "European film agencies",
        "n": 8,
        "prompt_extra": (
            "European talent agencies covering:\n"
            "  - French agencies (Paris) for arthouse and Cannes-circuit talent\n"
            "  - German and Scandinavian agencies (Berlin, Copenhagen)\n"
            "  - Pan-European co-production agencies (Eurimages-connected)\n"
            "  - Eastern European agencies (Warsaw, Prague) for festival darlings\n"
            "  - Boutique European agencies handling local-language stars\n"
            "Market reach: Europe (regional), some Global."
        ),
    },
    {
        "label": "MENA and Southeast Asian agencies",
        "n": 7,
        "prompt_extra": (
            "Middle East/North Africa and Southeast Asian agencies:\n"
            "  - Turkish mega-agencies (Istanbul) for drama export talent\n"
            "  - Iranian arthouse agencies for festival circuit directors/actors\n"
            "  - Gulf state (UAE, Saudi) new-wave agencies post-Vision 2030\n"
            "  - Thai and Filipino agencies for genre and commercial cinema\n"
            "  - Indonesian and Vietnamese agencies for growing local industries\n"
            "Market reach: Regional with selective Global."
        ),
    },
]


def step1_generate(client, force=False):
    print("=" * 60)
    print("  STEP 1: Expanding Agencies")
    print("=" * 60)

    existing = []
    if AGENCIES_PATH.exists():
        existing = json.loads(AGENCIES_PATH.read_text(encoding="utf-8"))
        print(f"  Existing: {len(existing)} agencies")
        if len(existing) >= 50 and not force:
            print("  Already have 50+ agencies. Use --force to regenerate.")
            return existing

    existing_names = {a["name"].lower() for a in existing}
    new_agencies = []
    total_cost = 0.0

    # Define the field structure based on existing agencies
    field_spec = (
        "name, tier (one of: powerhouse/major/mid/boutique/regional), "
        "specialty (2-3 sentence description of identity and market position), "
        "preferred_styles (2-4 acting styles from: magnetic, intense, chameleon, physical, naturalistic, "
        "cerebral, vulnerable, theatrical, understated, comedic, improvisational, deadpan, stoic, "
        "musical, provocative, voice-artist, acrobatic, explosive, menacing, minimalist), "
        "genre_focus (2-4 genres), "
        "market_reach (list: Global/North America/Europe/Asia/Latin America/Africa/MENA/Local/Regional), "
        "career_preference (list from: rising/prime/veteran/legend), "
        "avoids (1 sentence on what talent they don't represent), "
        "packaging_style (1 sentence on their deal-making approach), "
        "rivalries (1-2 rival agency names from the same batch)"
    )

    for batch in REGIONAL_BATCHES:
        n = batch["n"]
        print(f"\n  Generating {n} {batch['label']}...")

        prompt = f"""Generate exactly {n} FICTIONAL talent agencies for a film industry database.

REGION FOCUS:
{batch['prompt_extra']}

For each agency provide these fields:
{field_spec}

Requirements:
- Every name must be COMPLETELY FICTIONAL (no CAA, WME, ICM, etc.)
- Names should feel authentic to the region (can include local-language words)
- Each agency should feel distinct -- vary tiers, specialties, sizes
- Rivalries can reference other agencies IN THIS BATCH (use their exact names)
- Specialty descriptions must reference the specific region/culture authentically

Output a JSON array of exactly {n} objects. No markdown. Start with ["""

        try:
            data, cost = call_api(client, prompt, {
                "temperature": 0.9,
                "max_output_tokens": n * 300,
                "thinking_config": {"thinking_budget": 0},
            })
            total_cost += cost
        except RuntimeError as e:
            print(f"  FAILED: {e}")
            continue

        added = 0
        for a in data:
            name = str(a.get("name", "")).strip()
            if not name or name.lower() in existing_names:
                continue
            existing_names.add(name.lower())
            new_agencies.append(a)
            added += 1

        print(f"  +{added} agencies -> {len(new_agencies)} new total")
        time.sleep(0.3)

    all_agencies = existing + new_agencies
    AGENCIES_PATH.write_text(
        json.dumps(all_agencies, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n  Total agencies: {len(all_agencies)}")
    print(f"  New agencies added: {len(new_agencies)}")
    print(f"  Step 1 cost: ${total_cost:.4f}")
    print(f"  Saved: {AGENCIES_PATH}")
    return all_agencies


# ── Step 2: assign all persons to agencies ────────────────────────────────────

def build_agency_assign_prompt(agencies, person_batch):
    # Compact agency summaries
    lines = []
    for a in agencies:
        reach = ", ".join(a.get("market_reach", []))
        career = ", ".join(a.get("career_preference", []))
        lines.append(
            f"- {a['name']} [{a.get('tier','?')}]: {str(a.get('specialty',''))[:80]}... "
            f"reach=[{reach}], careers=[{career}], "
            f"genres={a.get('genre_focus',[])}"
        )
    agencies_text = "\n".join(lines)
    persons_json = json.dumps(person_batch, indent=1, ensure_ascii=False)

    return f"""Assign each person to ONE talent agency. Output exactly {len(person_batch)} objects.

AGENCIES (use EXACT names):
{agencies_text}

PERSONS:
{persons_json}

For each person output: {{"person_id": N, "agency": "EXACT agency name"}}

RULES:
- Match by nationality/region first (Indian -> Indian agencies, Korean -> Korean agencies, etc.)
- Then by career stage: rising stars -> mid/boutique, legends -> powerhouse/major
- Then by genre/style fit
- Powerhouse agencies can have up to 1000+ clients; boutiques 50-150
- Distribution should be UNEVEN and realistic

Output a JSON array of exactly {len(person_batch)} objects. No markdown. Start with ["""


def step2_assign(client, agencies, batch_size=150):
    print("\n" + "=" * 60)
    print("  STEP 2: Assigning Persons to Agencies")
    print("=" * 60)

    persons = json.loads(PERSONS_PATH.read_text(encoding="utf-8"))
    agency_names = {a["name"] for a in agencies}
    lower_map = {n.lower(): n for n in agency_names}

    # Find unassigned
    unassigned = [p for p in persons if not p.get("agency")]
    print(f"  Total persons: {len(persons)}")
    print(f"  Already assigned: {len(persons) - len(unassigned)}")
    print(f"  Needing assignment: {len(unassigned)}")

    if not unassigned:
        print("  All persons already assigned!")
        return

    total_cost = 0.0
    assigned_map = {}  # person_id -> agency
    n_batches = (len(unassigned) + batch_size - 1) // batch_size

    for i in range(0, len(unassigned), batch_size):
        batch = unassigned[i: i + batch_size]
        b_num = i // batch_size + 1

        # Compact person summaries
        person_batch = [
            {
                "person_id": p["person_id"],
                "name":        p.get("name", "?"),
                "nationality": p.get("nationality", "?"),
                "career_stage": p.get("career_stage", p.get("stage", "prime")),
                "role":        p.get("role", "actor"),
            }
            for p in batch
        ]

        print(f"\n  Batch {b_num}/{n_batches} ({len(batch)} persons)")
        prompt = build_agency_assign_prompt(agencies, person_batch)

        try:
            data, cost = call_api(client, prompt, {
                "temperature": 0.5,
                "max_output_tokens": max(8000, len(batch) * 40),
                "thinking_config": {"thinking_budget": 0},
            })
            total_cost += cost
        except RuntimeError as e:
            print(f"  FAILED: {e} -- skipping batch")
            continue

        matched = 0
        fallbacks = 0
        rng = random.Random(i)
        for item in data:
            pid    = item.get("person_id")
            agency = str(item.get("agency", "")).strip()
            if pid is None or pid in assigned_map:
                continue
            if agency not in agency_names:
                canonical = lower_map.get(agency.lower())
                if not canonical:
                    for real in agency_names:
                        if real.lower() in agency.lower() or agency.lower() in real.lower():
                            canonical = real
                            break
                if canonical:
                    agency = canonical
                else:
                    agency = rng.choice(list(agency_names))
                    fallbacks += 1
            assigned_map[pid] = agency
            matched += 1

        print(f"  +{matched} assigned ({fallbacks} fallback) -> {len(assigned_map)} total")

        # Apply to persons in memory
        pid_to_agency = dict(assigned_map)
        for p in persons:
            if p["person_id"] in pid_to_agency:
                p["agency"] = pid_to_agency[p["person_id"]]

        # Incremental save
        PERSONS_PATH.write_text(
            json.dumps(persons, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        time.sleep(0.3)

    # Final report
    dist = Counter(assigned_map.values())
    print(f"\n{'='*60}")
    print(f"  AGENCY ASSIGNMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Assigned: {len(assigned_map)}")
    print(f"  Cost: ${total_cost:.4f}")
    print(f"\n  Top agencies by client count:")
    for ag, cnt in dist.most_common(12):
        print(f"    {safe(ag):40} {cnt:5d}")
    print(f"\n  Saved: {PERSONS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate and assign V15 talent agencies")
    parser.add_argument("--step",     type=int, choices=[1, 2])
    parser.add_argument("--auto",     action="store_true")
    parser.add_argument("--force",    action="store_true", help="Regenerate agencies even if 50+ exist")
    parser.add_argument("--batch-size", type=int, default=150)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    if args.step in (None, 1):
        agencies = step1_generate(client, force=args.force)
    else:
        agencies = json.loads(AGENCIES_PATH.read_text(encoding="utf-8"))
        print(f"  Loaded {len(agencies)} existing agencies")

    if args.step in (None, 2):
        step2_assign(client, agencies, batch_size=args.batch_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
