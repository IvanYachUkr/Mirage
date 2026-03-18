"""
V15 -- Generate production companies via Gemini Flash Lite.

Two-phase:
  Phase 1: LLM generates company names + countries in regional batches
  Phase 2: LLM fills full profiles for unique names

Fixes vs V12:
  - Correct model: gemini-3.1-flash-lite-preview
  - Correct pricing: $0.25/$1.50 per 1M tokens
  - 70s API hang timeout + 5 retries with 503 detection
  - Rotating regional prompts tuned to reflect real film industry output volumes
  - founded_year generated (realistic 1920-2022 range per tier)
  - --target CLI arg, resumes from existing companies.json
  - Paths wired to test15/entities/

Usage:
    python test15/generate_companies_llm.py --target 1001
"""
import json, re, os, time, sys, argparse, random, concurrent.futures
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, os.path.dirname(__file__))
from contracts import GENRES, STYLE_TAGS, DIRECTOR_STYLES, COMPANY_TIERS, COUNTRIES

BASE_DIR       = Path(__file__).parent
ENTITY_DIR     = BASE_DIR / "entities"
COMPANIES_PATH = ENTITY_DIR / "companies.json"

MODEL   = "gemini-3.1-flash-lite-preview"
PRICING = {"input": 0.075, "output": 0.30}  # per 1M tokens (lite-preview)

# ── Tier distribution (realistic for a movie database) ────────────────────────
# Global = major international conglomerates (very few)
# Major  = national box-office players
# Mid    = consistently releasing mid-level features
# Indie  = limited release, festival circuit
# Micro  = one-off or tiny boutique
TIER_DIST = {
    "Global":     0.04,
    "Major":      0.13,
    "Mid-Budget": 0.28,
    "Indie":      0.38,
    "Micro":      0.17,
}

# ── Founded-year ranges by tier ───────────────────────────────────────────────
FOUNDED_RANGES = {
    "Global":     (1920, 1985),  # old institutions
    "Major":      (1940, 2000),
    "Mid-Budget": (1960, 2015),
    "Indie":      (1975, 2020),
    "Micro":      (1995, 2022),
}

# ── Regional prompt batches (tuned to real film industry output volumes) ──────
# Weights roughly reflect annual feature-film production volume by region.
# India and China are the world's highest-volume markets; include sub-regions.
# Africa (especially Nollywood) is the world's 2nd-largest by count.
# Southeast Asia is fast-growing and currently underrepresented.
REGION_BATCHES = [
    {
        "weight": 12,
        "label": "Hollywood / American indie",
        "prompt": (
            "American production companies -- mix of:\n"
            "  * Major Hollywood studios (slates of 5-20 features/year, LA/NY HQ)\n"
            "  * Sundance/A24-style prestige indie labels\n"
            "  * Genre specialists (horror, sci-fi, action)\n"
            "  * Black-led studios and culturally specific American production houses\n"
            "  * Documentary-focused companies\n"
            "Country: USA"
        ),
    },
    {
        "weight": 16,
        "label": "Indian cinema (Bollywood + regional industries)",
        "prompt": (
            "Indian production companies across ALL regional film industries:\n"
            "  * Bollywood (Hindi, Mumbai): masala, romance, action\n"
            "  * Tollywood (Telugu, Hyderabad): big-budget action spectacles\n"
            "  * Kollywood (Tamil, Chennai): stylised action, social drama\n"
            "  * Mollywood (Malayalam, Kerala): critically acclaimed realist drama\n"
            "  * Kannada, Odia, Bhojpuri, Marathi, Bengali regional studios\n"
            "  * OTT-first production houses (Amazon India, Netflix India partners)\n"
            "Use authentic regional studio naming conventions.\n"
            "Country: vary by industry (India, with city hint in description)"
        ),
    },
    {
        "weight": 10,
        "label": "Chinese cinema",
        "prompt": (
            "Chinese production companies:\n"
            "  * State-owned major studios (Beijing, Shanghai, Guangzhou)\n"
            "  * Private mid-budget companies (action, historical epics, romance)\n"
            "  * Hong Kong cinema tradition (crime, martial arts)\n"
            "  * Taiwanese art-house and commercial production\n"
            "  * Online video platform production arms (iQiyi, Youku style)\n"
            "Use authentic Mandarin/Cantonese naming conventions -- include romanisation.\n"
            "Country: China or Taiwan"
        ),
    },
    {
        "weight": 8,
        "label": "Japanese and South Korean cinema",
        "prompt": (
            "Japanese and South Korean production companies:\n"
            "  * Major Japanese studios (anime, live-action, kaiju/genre)\n"
            "  * South Korean Chaebol-linked and indie production houses\n"
            "  * Korean Wave (Hallyu) drama and film studios\n"
            "  * Japanese art-house and festival-circuit companies\n"
            "  * Animation-specialist studios (Japan)\n"
            "Use authentic Japanese/Korean naming conventions.\n"
            "Country: Japan or South Korea"
        ),
    },
    {
        "weight": 9,
        "label": "Western European cinema",
        "prompt": (
            "Western European production companies:\n"
            "  * French Cinema companies (art-house, heritage drama, comedy)\n"
            "  * German companies (auteur cinema, genre, TV-movie crossovers)\n"
            "  * Italian studios (genre, neorealist tradition, commercial)\n"
            "  * Spanish/Catalan companies (genre, social drama, horror)\n"
            "  * Scandinavian (Swedish, Norwegian, Danish, Finnish) moody dramas\n"
            "  * British indie and heritage companies\n"
            "Use name conventions of each country (French, German, Italian, Spanish names).\n"
            "Country: France / Germany / Italy / Spain / Sweden / Norway / UK / etc."
        ),
    },
    {
        "weight": 5,
        "label": "Eastern European and Russian cinema",
        "prompt": (
            "Eastern European and Russian/CIS production companies:\n"
            "  * Russian studios (action blockbusters, historical epics, arthouse)\n"
            "  * Polish, Czech, Romanian, Hungarian companies (festival darlings)\n"
            "  * Ukrainian, Georgian, Azerbaijani, Armenian cinema\n"
            "  * Ex-Yugoslav (Serbian, Croatian, Bosnian) companies\n"
            "  * Baltic (Estonian, Latvian, Lithuanian) micro-studios\n"
            "Use regional naming conventions.\n"
            "Country: vary per company"
        ),
    },
    {
        "weight": 12,
        "label": "African cinema -- Nollywood and beyond",
        "prompt": (
            "African production companies (critically underrepresented in databases):\n"
            "  * Nigerian Nollywood studios (world's 2nd-largest film industry by volume)\n"
            "    -- Lagos DTV, Igbo-language, Yoruba-language companies\n"
            "  * South African studios (commercial, genre, apartheid/post-apartheid drama)\n"
            "  * Kenyan, Ugandan, Tanzanian East African companies\n"
            "  * Ghanaian Ghallywood studios\n"
            "  * Ethiopian, Rwandan, Senegalese, Ivorian companies\n"
            "  * Francophone West African (Burkina Faso FESPACO tradition, Mali, Cameroon)\n"
            "  * Egyptian studio tradition (Arab world's biggest film industry)\n"
            "  * Moroccan, Tunisian, Algerian Maghreb companies\n"
            "Use authentic regional African studio names.\n"
            "Country: Nigeria / South Africa / Kenya / Ghana / Ethiopia / Egypt / Morocco / etc."
        ),
    },
    {
        "weight": 6,
        "label": "Middle East and Central Asia",
        "prompt": (
            "Middle Eastern and Central Asian production companies:\n"
            "  * Iranian cinema (art-house, rich festival tradition, Fajr winners)\n"
            "  * Turkish studios (world's largest TV drama exporter, plus film)\n"
            "  * Israeli companies (drama, genre, festival circuit)\n"
            "  * Saudi, UAE, and Gulf state new-wave studios\n"
            "  * Lebanese, Jordanian, Palestinian companies\n"
            "  * Uzbek, Kazakh, Kyrgyz Central Asian studios\n"
            "Use authentic naming conventions.\n"
            "Country: Iran / Turkey / Israel / Saudi Arabia / UAE / Kazakhstan / etc."
        ),
    },
    {
        "weight": 7,
        "label": "Latin America",
        "prompt": (
            "Latin American production companies:\n"
            "  * Brazilian studios (Cinema Novo tradition, commercial genre, Globo Filmes style)\n"
            "  * Mexican companies (genre, social realism, arthouse, mainstream comedy)\n"
            "  * Argentine companies (Buenos Aires art cinema, genre)\n"
            "  * Colombian, Chilean, Peruvian companies\n"
            "  * Cuban, Venezuelan, Bolivian, Ecuadorian micro-studios\n"
            "  * Caribbean (Jamaican, Haitian, Dominican) companies\n"
            "Use Spanish/Portuguese naming conventions as appropriate.\n"
            "Country: Brazil / Mexico / Argentina / Colombia / Chile / etc."
        ),
    },
    {
        "weight": 8,
        "label": "Southeast Asia and Oceania",
        "prompt": (
            "Southeast Asian and Oceanian production companies:\n"
            "  * Thai studios (horror specialists, action, royal drama)\n"
            "  * Filipino companies (commercial, indie, Cinemalaya-style)\n"
            "  * Indonesian studios (horror, action, Javanese/Batak stories)\n"
            "  * Vietnamese companies (war epics, romance, genre)\n"
            "  * Malaysian and Singaporean companies\n"
            "  * Burmese, Cambodian, Laotian micro-studios\n"
            "  * Australian companies (genre, indigenous storytelling, prestige)\n"
            "  * New Zealand companies (fantasy, adventure, indie)\n"
            "  * Pacific Islander (Fiji, Samoa, Maori-led) studios\n"
            "Use authentic names from each country.\n"
            "Country: Thailand / Philippines / Indonesia / Vietnam / Australia / New Zealand / etc."
        ),
    },
    {
        "weight": 5,
        "label": "Canadian, Irish, and small English-speaking markets",
        "prompt": (
            "Canadian, Irish, and smaller English-speaking market companies:\n"
            "  * Canadian English-language (Ontario, BC, Quebec-English)\n"
            "  * Quebecois French-language studios\n"
            "  * Indigenous Canadian (First Nations, Métis, Inuit) production houses\n"
            "  * Irish companies (literary adaptations, Gaeltacht content, genre)\n"
            "  * Scottish and Welsh devolved industry companies\n"
            "Country: Canada / Ireland / UK (Scotland) / UK (Wales)"
        ),
    },
]
# Validate weights sum
assert sum(r["weight"] for r in REGION_BATCHES) == 98  # allowed to not be 100

# ── Helpers ───────────────────────────────────────────────────────────────────

def call_api_with_timeout(client, model, prompt, config, timeout=70):
    """Wrap API call with hard timeout. Raises TimeoutError if exceeded."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(
        client.models.generate_content,
        model=model, contents=prompt, config=config,
    )
    try:
        result = future.result(timeout=timeout)
        pool.shutdown(wait=False, cancel_futures=True)
        return result
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"API call exceeded {timeout}s")



def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if text.startswith("{"):
        text = "[" + re.sub(r'}\s*{', '},{', text) + "]"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    last = text.rfind("}")
    if last > 0:
        candidate = text[:last+1].rstrip().rstrip(",") + "]"
        try:
            data = json.loads(candidate)
            print(f"  (Recovered {len(data)} from truncated JSON)")
            return data
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON ({len(text)} chars)")


def call_with_retry(client, model, prompt, config, max_retries=5, timeout=70):
    """Call API with retry on 503/timeout. Returns (resp, cost_info) or raises."""
    for retry in range(max_retries):
        try:
            resp = call_api_with_timeout(client, model, prompt, config, timeout=timeout)
            usage = resp.usage_metadata
            inp = getattr(usage, "prompt_token_count", 0) or 0
            out = getattr(usage, "candidates_token_count", 0) or 0
            cost = inp / 1e6 * PRICING["input"] + out / 1e6 * PRICING["output"]
            return resp, inp, out, cost
        except TimeoutError as e:
            print(f"  TIMEOUT (retry {retry+1}/{max_retries}): {e}")
            time.sleep(10 * (retry + 1))
        except Exception as e:
            err = str(e)
            is_503 = "503" in err or "unavailable" in err.lower()
            print(f"  {'503 ' if is_503 else ''}API ERROR (retry {retry+1}/{max_retries}): {e}")
            time.sleep(5 * (retry + 1))
    raise RuntimeError(f"All {max_retries} retries failed")


def pick_region(rng):
    """Weighted-random region pick using REGION_BATCHES weights."""
    weights = [r["weight"] for r in REGION_BATCHES]
    return rng.choices(REGION_BATCHES, weights=weights, k=1)[0]


def pick_founded_year(tier, rng):
    lo, hi = FOUNDED_RANGES.get(tier, (1970, 2018))
    return rng.randint(lo, hi)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1001,
                        help="Total companies to reach (default: 1001)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Profiles per Phase-2 batch (default: 50)")
    parser.add_argument("--timeout", type=int, default=70,
                        help="API hang timeout seconds (default: 70)")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    # Load existing -- ALWAYS resume, never delete
    companies = []
    existing_names = set()
    if COMPANIES_PATH.exists():
        with open(COMPANIES_PATH, "r", encoding="utf-8") as f:
            companies = json.load(f)
        existing_names = {c["name"].lower() for c in companies}
        print(f"Resuming from {len(companies)} existing companies")

    need = args.target - len(companies)
    if need <= 0:
        print(f"Already at target ({len(companies)} >= {args.target}). Done!")
        return

    rng = random.Random(42)
    total_input = 0
    total_output = 0
    total_cost = 0.0

    # ── PHASE 1: Name bank ───────────────────────────────────────────────────
    print(f"\n=== PHASE 1: Generate {need} company names ===")
    namebank = []
    batch_num = 0

    while len(namebank) < need:
        batch_num += 1
        region = pick_region(rng)
        batch_need = min(80, need - len(namebank))
        # Always ask for at least 30 -- model returns garbage/empty for tiny requests
        ask_n = max(30, batch_need)

        print(f"\n  Name batch {batch_num} [{region['label']}] need={batch_need}")

        prompt = f"""Generate EXACTLY {ask_n} unique FICTIONAL movie/TV production company names.

REGION FOCUS:
{region['prompt']}

Requirements:
- Every name must be COMPLETELY FICTIONAL (no Disney, Warner, A24, Netflix, etc.)
- Mix styles: classic studio names, modern upstarts, local-language names, boutique labels
- Names should feel authentic to the region
- No two names should be similar to each other
- IMPORTANT: distribute companies REALISTICALLY within this region -- dominant film industries
  get more companies than minor ones. E.g. for Africa: Nigeria gets 35-40% of slots,
  South Africa 15-20%, Egypt 10-15%, others get fewer. For India: Bollywood + Telugu 
  get the majority. Do NOT split evenly across all sub-regions.

Return a JSON array. Each object: {{"name": "...", "country": "..."}}
No markdown. Start with ["""

        try:
            resp, inp, out, cost = call_with_retry(
                client, MODEL, prompt,
                {"temperature": 1.0, "max_output_tokens": ask_n * 30,
                 "thinking_config": {"thinking_budget": 0}},
                timeout=args.timeout,
            )
            total_input += inp; total_output += out; total_cost += cost
            print(f"  {inp:,}in+{out:,}out=${cost:.4f}")
        except RuntimeError as e:
            print(f"  Skipping batch: {e}")
            continue

        try:
            data = parse_json(resp.text)
        except Exception as e:
            print(f"  PARSE ERROR: {e}")
            continue

        added = 0
        for c in data:
            n = str(c.get("name", "")).strip().lower()
            if n and len(n) >= 3 and n not in existing_names:
                existing_names.add(n)
                namebank.append({
                    "name": c["name"].strip(),
                    "country": c.get("country", "Unknown"),
                })
                added += 1
        print(f"  +{added} names -> {len(namebank)}/{need}")

    print(f"\nName bank complete: {len(namebank)} companies ready for profiling")

    # ── PHASE 2: Full profiles ───────────────────────────────────────────────
    print(f"\n=== PHASE 2: Generate full profiles ===")
    BATCH_SIZE = args.batch_size

    for i in range(0, len(namebank), BATCH_SIZE):
        batch = namebank[i: i + BATCH_SIZE]
        b_num = i // BATCH_SIZE + 1
        b_total = (len(namebank) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n  Profile batch {b_num}/{b_total} ({len(batch)} companies)")

        names_json = json.dumps(batch, ensure_ascii=False)
        # Show realistic tier distribution hint
        tier_hint = ", ".join(
            f"{int(p*100)}% {t}" for t, p in TIER_DIST.items()
        )

        prompt = f"""Generate full profiles for these production companies.
Keep name and country EXACTLY as given. Add these fields to each:

{names_json}

Fields to add per company:
- description: 2-3 sentences. Must reference the company's actual country/culture authentically.
  Include founding ethos, signature genre/style, and current reputation.
- specialty_genres: 2-3 from {json.dumps(GENRES)}
- tier: one of {json.dumps(COMPANY_TIERS)}
  Target distribution across the batch: {tier_hint}
- preferred_actor_styles: 2-3 from {json.dumps(STYLE_TAGS[:18])}
- preferred_director_styles: 2-3 from {json.dumps(DIRECTOR_STYLES[:15])}
- avoid_actor_styles: 1-2 from {json.dumps(STYLE_TAGS[:18])} (styles they actively avoid)
- avoid_director_styles: 1-2 from {json.dumps(DIRECTOR_STYLES[:15])}

Descriptions must feel real and specific -- mention the city/region, what kind of films they champion,
and one distinguishing characteristic (e.g., 'known for nurturing first-time directors',
'favors practical effects over CGI', 'specialises in festival circuit drama').

JSON array. No markdown. Start with ["""

        try:
            resp, inp, out, cost = call_with_retry(
                client, MODEL, prompt,
                {"temperature": 0.9, "max_output_tokens": BATCH_SIZE * 250,
                 "thinking_config": {"thinking_budget": 0}},
                timeout=args.timeout,
            )
            total_input += inp; total_output += out; total_cost += cost
            print(f"  {inp:,}in+{out:,}out=${cost:.4f}")
        except RuntimeError as e:
            print(f"  Skipping batch: {e}")
            continue

        try:
            data = parse_json(resp.text)
        except Exception as e:
            print(f"  PARSE ERROR: {e}")
            dev_path = BASE_DIR / "_dev" / f"companies_batch{b_num}_raw.txt"
            os.makedirs(dev_path.parent, exist_ok=True)
            dev_path.write_text(resp.text, encoding="utf-8")
            continue

        added = 0
        for c in data:
            name = str(c.get("name", "")).strip()
            desc = str(c.get("description", "")).strip()
            if not name or len(desc) < 20:
                continue
            # Normalize tier
            tier = c.get("tier", "Indie")
            if tier not in COMPANY_TIERS:
                tier = "Indie"
            c["tier"] = tier
            # Add founded_year if missing
            if "founded_year" not in c:
                c["founded_year"] = pick_founded_year(tier, rng)
            # Normalize list fields
            for field in ["specialty_genres", "preferred_actor_styles",
                          "preferred_director_styles", "avoid_actor_styles",
                          "avoid_director_styles"]:
                v = c.get(field, [])
                if isinstance(v, str):
                    v = [x.strip() for x in v.split(",")]
                c[field] = [x for x in v if x]
            companies.append(c)
            added += 1

        pct = added / len(batch) * 100 if batch else 0
        print(f"  +{added}/{len(batch)} ({pct:.0f}%) -> {len(companies)} total | run cost ${total_cost:.3f}")
        if added > 0:
            s = companies[-1]
            safe_n = s['name'].encode('ascii', 'replace').decode()
            safe_d = s.get('description', '')[:100].encode('ascii', 'replace').decode()
            print(f"    [{s['tier']:<10}] {safe_n} ({s['country']})")
            print(f"    {safe_d}...")

        # Incremental save
        os.makedirs(ENTITY_DIR, exist_ok=True)
        for idx, c in enumerate(companies):
            if "company_id" not in c:
                c["company_id"] = idx + 1
        with open(COMPANIES_PATH, "w", encoding="utf-8") as f:
            json.dump(companies, f, indent=2, ensure_ascii=False)

        time.sleep(0.3)

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPANY GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total:    {len(companies)}")
    print(f"  Tokens:   {total_input:,} in + {total_output:,} out")
    print(f"  Cost:     ${total_cost:.4f}")
    tiers = Counter(c.get("tier", "?") for c in companies)
    countries_c = Counter(c.get("country", "?") for c in companies)
    print(f"  Tiers:    {dict(tiers.most_common())}")
    print(f"  Top countries: {dict(countries_c.most_common(12))}")
    print(f"  Saved:    {COMPANIES_PATH}")


if __name__ == "__main__":
    main()
