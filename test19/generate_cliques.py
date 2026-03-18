"""
V15 -- Generate company cliques and assign all companies.
=========================================================
Two-step:
  Step 1: LLM generates N clique profiles (--num-cliques, default 18)
  Step 2: LLM assigns every company in companies.json to one clique

Why 18? With 1,024 companies we want avg ~57/clique (range 20-120) for
meaningful graph clustering. The existing 5 cliques were too coarse --
Ivory Tower alone contained 35% of all companies.

Mix of philosophical + regional clusters:
  Philosophical (8): blockbuster, prestige, horror/genre, arthouse,
    streaming-native, documentary, animation, experimental
  Regional/cultural (10): Bollywood, Nollywood, East Asian, SE Asian,
    Latin American, MENA, European heritage, Slavic/Eastern European,
    Anglophone indie, Pacific/Oceanian

Fixes vs V12:
  - Self-contained (no _common dependency)
  - gemini-3.1-flash-lite-preview + correct $0.25/$1.50 pricing
  - 70s API hang timeout (ThreadPoolExecutor)
  - 503 detection + 5 retries with backoff
  - --force flag to replace existing cliques
  - Incremental save every batch
  - All ASCII-safe output (no unicode encoding errors)

Usage:
    python test15/generate_cliques.py                   # both steps
    python test15/generate_cliques.py --step 1          # generate cliques only
    python test15/generate_cliques.py --step 2          # assign only (use existing cliques)
    python test15/generate_cliques.py --force           # overwrite existing cliques.json
    python test15/generate_cliques.py --num-cliques 20  # custom count
"""
import json, os, sys, time, argparse, concurrent.futures
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, os.path.dirname(__file__))
from contracts import GENRES, COMPANY_TIERS

BASE_DIR       = Path(__file__).parent
ENTITY_DIR     = BASE_DIR / "entities"
COMPANIES_PATH = ENTITY_DIR / "companies.json"
CLIQUES_PATH   = ENTITY_DIR / "cliques.json"

MODEL   = "gemini-3.1-flash-lite-preview"
PRICING = {"input": 0.25, "output": 1.50}

# â”€â”€ API helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def call_with_timeout(client, prompt, config, timeout=70):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    f = pool.submit(client.models.generate_content,
                    model=MODEL, contents=prompt, config=config)
    try:
        result = f.result(timeout=timeout)
        pool.shutdown(wait=False, cancel_futures=True)
        return result
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"API exceeded {timeout}s")


def call_api(client, prompt, config, max_retries=5, timeout=70):
    """Call API with 503 detection + exponential backoff. Returns (parsed_list, cost)."""
    for retry in range(max_retries):
        try:
            resp = call_with_timeout(client, prompt, config, timeout=timeout)
        except TimeoutError as e:
            print(f"  TIMEOUT (retry {retry+1}): {e}")
            time.sleep(10 * (retry + 1))
            continue
        except Exception as e:
            is_503 = "503" in str(e) or "unavailable" in str(e).lower()
            print(f"  {'503 ' if is_503 else ''}API ERROR (retry {retry+1}): {e}")
            time.sleep(5 * (retry + 1))
            continue

        usage = resp.usage_metadata
        inp = getattr(usage, "prompt_token_count", 0) or 0
        out = getattr(usage, "candidates_token_count", 0) or 0
        cost = inp / 1e6 * PRICING["input"] + out / 1e6 * PRICING["output"]
        print(f"  {inp:,}in+{out:,}out=${cost:.4f}")

        text = resp.text.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # Wrapped in a single object -- try to extract list
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break
            if isinstance(data, list):
                return data, cost
        except json.JSONDecodeError:
            pass

        # Try truncation recovery
        import re
        last = text.rfind("}")
        if last > 0:
            candidate = text[:last+1].rstrip().rstrip(",") + "]"
            try:
                data = json.loads("[" + candidate if not candidate.startswith("[") else candidate)
                print(f"  (recovered {len(data)} from truncated JSON)")
                return data, cost
            except json.JSONDecodeError:
                pass

        print(f"  PARSE ERROR (retry {retry+1}): bad JSON ({len(text)} chars)")
        time.sleep(2)

    raise RuntimeError(f"All {max_retries} retries failed")


def safe_print(s):
    print(s.encode("ascii", "replace").decode())


# â”€â”€ Step 1: Generate clique profiles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def step1_generate(client, num_cliques, force=False):
    print("=" * 60)
    print(f"  STEP 1: Generating {num_cliques} Clique Profiles")
    print("=" * 60)

    if CLIQUES_PATH.exists() and not force:
        existing = json.loads(CLIQUES_PATH.read_text(encoding="utf-8"))
        print(f"  Already have {len(existing)} cliques. Use --force to regenerate.")
        return existing

    # Design hint to guide the LLM toward a good mix
    design_hint = f"""
Design EXACTLY {num_cliques} cliques with this approximate split:
  - 8 philosophical/aesthetic cliques (cross-regional, defined by filmmaking approach):
      e.g., blockbuster spectacle, prestige/awards, horror+genre, arthouse/experimental,
      streaming-native, documentary advocacy, animation specialists, exploitation/grindhouse
  - 10 regional/cultural cliques (geographically anchored):
      e.g., Bollywood + Indian regional cinemas, Nollywood + pan-African,
      East Asian (Chinese/Japanese/Korean), Southeast Asian, Latin American,
      MENA (Middle East + North Africa), Eastern European + Russian,
      Western European heritage, Anglophone indie (UK/Canada/Ireland/Australia),
      Pacific + Oceanian

IMPORTANT for realistic distribution:
  - Philosophical cliques will naturally be larger (60-120 companies each)
  - Regional cliques will be smaller (25-60 each) matching actual industry sizes
  - Nollywood should be larger than, e.g., Pacific/Oceanian
  - All {num_cliques} cliques should be DISTINCT -- no overlap
  - Together they must cover all company types and regions"""

    prompt = f"""You are a film industry analyst. Generate {num_cliques} PRODUCTION CLIQUES --
informal alliances of production companies sharing a house style or cultural/regional identity.
{design_hint}

For each clique provide:
- name: short evocative label (2-5 words)
- philosophy: 2 sentences on identity, market position, and filmmaking worldview
- budget_preference: "high", "medium", "low", or "mixed"
- genre_focus: 3-5 genres from {json.dumps(GENRES[:20])}
- rating_target: "critical" (7+), "crowd-pleaser" (5.5-7), or "niche" (variable)
- risk_appetite: "conservative", "moderate", or "aggressive"
- market_strategy: one of "global_theatrical", "streaming_first", "festival_circuit", "regional", "hybrid"
- preferred_company_tiers: subset of {json.dumps(COMPANY_TIERS)}
- rivalries: 1-2 other clique names this clique competes with (use exact names from your list)

Output a JSON array of exactly {num_cliques} objects. No markdown. Start with ["""

    data, cost = call_api(client, prompt,
                          {"temperature": 0.9,
                           "max_output_tokens": num_cliques * 200,
                           "thinking_config": {"thinking_budget": 0}})

    print(f"  Generated {len(data)} cliques")
    if len(data) != num_cliques:
        print(f"  WARNING: wanted {num_cliques}, got {len(data)}")

    for i, c in enumerate(data):
        c["clique_id"] = i + 1

    CLIQUES_PATH.parent.mkdir(exist_ok=True)
    CLIQUES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    for c in data:
        safe_print(f"  [{c.get('budget_preference','?'):6}] {c.get('name','?'):35} "
                   f"risk={c.get('risk_appetite','?')}")
    print(f"\n  Saved: {CLIQUES_PATH}")
    print(f"  Total cost: ${cost:.4f}")
    return data


# â”€â”€ Step 2: Assign companies to cliques â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_assign_prompt(cliques, company_batch):
    clique_lines = []
    for c in cliques:
        genres_str = ", ".join(c.get("genre_focus", []))
        tiers_str  = ", ".join(c.get("preferred_company_tiers", []))
        phil = c.get("philosophy", "")[:90]
        clique_lines.append(
            f"- {c['name']}: {phil}... "
            f"budget={c.get('budget_preference','?')}, "
            f"risk={c.get('risk_appetite','?')}, "
            f"strategy={c.get('market_strategy','?')}, "
            f"genres=[{genres_str}], tiers=[{tiers_str}]"
        )
    cliques_text = "\n".join(clique_lines)
    companies_json = json.dumps(company_batch, indent=1, ensure_ascii=False)

    return f"""Assign each company to ONE clique. You MUST output exactly {len(company_batch)} objects.

CLIQUES (use these EXACT names -- copy character-for-character):
{cliques_text}

COMPANIES TO ASSIGN:
{companies_json}

For EVERY company output: {{"company_id": N, "clique": "EXACT clique name", "reason": "1 sentence"}}

RULES:
- Match by genre overlap, tier alignment, philosophy fit, AND country/cultural origin
- Regional cliques should get companies from their geographic area
- Distribution should be UNEVEN -- popular cliques get more members than niche ones
- Every company_id must appear exactly once

Output a JSON array of exactly {len(company_batch)} objects. No markdown. Start with ["""


def step2_assign(client, cliques, batch_size=80, timeout=70):
    print("\n" + "=" * 60)
    print("  STEP 2: Assigning Companies to Cliques")
    print("=" * 60)

    companies = json.loads(COMPANIES_PATH.read_text(encoding="utf-8"))
    clique_names = {c["name"] for c in cliques}
    lower_map = {n.lower(): n for n in clique_names}

    print(f"  Companies total: {len(companies)}")
    print(f"  Cliques: {len(cliques)}")

    # Reassign ALL companies (since clique definitions changed)
    # Clear existing clique assignments so we start fresh
    for c in companies:
        c.pop("clique", None)
        c.pop("clique_reason", None)

    total_cost  = 0.0
    assigned    = {}   # company_id -> clique name
    reason_map  = {}   # company_id -> reason

    n_batches = (len(companies) + batch_size - 1) // batch_size

    for i in range(0, len(companies), batch_size):
        batch       = companies[i: i + batch_size]
        batch_num   = i // batch_size + 1

        company_batch = [
            {
                "company_id":       c["company_id"],
                "name":             c["name"],
                "country":          c.get("country", "?"),
                "specialty_genres": c.get("specialty_genres", []),
                "tier":             c.get("tier", "Indie"),
                "description":      c.get("description", "")[:90],
            }
            for c in batch
        ]

        print(f"\n  Batch {batch_num}/{n_batches} ({len(batch)} companies)")
        prompt = build_assign_prompt(cliques, company_batch)

        try:
            data, cost = call_api(
                client, prompt,
                {"temperature": 0.6,
                 "max_output_tokens": max(6000, len(batch) * 80),
                 "thinking_config": {"thinking_budget": 0}},
                timeout=timeout,
            )
            total_cost += cost
        except RuntimeError as e:
            print(f"  FAILED: {e} -- skipping batch")
            continue

        matched = 0
        unknown = 0
        for item in data:
            cid    = item.get("company_id")
            clique = item.get("clique", "").strip()
            reason = item.get("reason", "")

            if cid is None or cid in assigned:
                continue

            # Exact match
            if clique not in clique_names:
                # Case-insensitive
                canonical = lower_map.get(clique.lower())
                if not canonical:
                    # Substring
                    for real in clique_names:
                        if real.lower() in clique.lower() or clique.lower() in real.lower():
                            canonical = real
                            break
                if canonical:
                    clique = canonical
                else:
                    unknown += 1
                    clique = cliques[0]["name"]  # fallback

            assigned[cid]   = clique
            reason_map[cid] = reason
            matched += 1

        print(f"  +{matched} assigned, {unknown} fallback -> {len(assigned)}/{len(companies)} total")

        # Apply to in-memory companies list
        for c in companies:
            cid = c["company_id"]
            if cid in assigned:
                c["clique"]        = assigned[cid]
                c["clique_reason"] = reason_map.get(cid, "")

        # Incremental save
        COMPANIES_PATH.write_text(
            json.dumps(companies, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if data:
            first = data[0]
            first_name = next((c["name"] for c in batch
                               if c.get("company_id") == first.get("company_id")), "?")
            safe_print(f"    e.g. {first_name} -> {first.get('clique','?')}: "
                       f"{first.get('reason','')[:70]}")

        time.sleep(0.3)

    # Final report
    dist = Counter(assigned.values())
    print(f"\n{'='*60}")
    print(f"  CLIQUE ASSIGNMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Assigned: {len(assigned)}/{len(companies)}")
    print(f"  Run cost: ${total_cost:.4f}")
    print(f"\n  Distribution ({len(dist)} cliques):")
    for cliq, cnt in dist.most_common():
        pct = cnt / max(len(assigned), 1) * 100
        bar = "#" * (cnt // 5)
        safe_print(f"    {cliq:35} {cnt:4d} ({pct:4.1f}%) {bar}")
    print(f"\n  Saved: {COMPANIES_PATH}")


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(description="Generate + assign V15 company cliques")
    parser.add_argument("--step",        type=int, choices=[1, 2],
                        help="Run only step 1 (generate) or step 2 (assign)")
    parser.add_argument("--num-cliques", type=int, default=18,
                        help="How many cliques to generate (default: 18)")
    parser.add_argument("--batch-size",  type=int, default=80,
                        help="Companies per assignment batch (default: 80)")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing cliques.json")
    parser.add_argument("--timeout",     type=int, default=70,
                        help="API timeout in seconds (default: 90)")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    if args.step in (None, 1):
        cliques = step1_generate(client, args.num_cliques, force=args.force)
    else:
        if not CLIQUES_PATH.exists():
            raise RuntimeError(f"No {CLIQUES_PATH} -- run step 1 first")
        cliques = json.loads(CLIQUES_PATH.read_text(encoding="utf-8"))
        print(f"  Loaded {len(cliques)} existing cliques")

    if args.step in (None, 2):
        step2_assign(client, cliques, batch_size=args.batch_size, timeout=args.timeout)

    print("\nDone!")


if __name__ == "__main__":
    main()


