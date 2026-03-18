# -*- coding: utf-8 -*-
"""
Test 11 -- Generate Latent Variables via Gemini API
===================================================
Step 1 of the Hybrid Graph Architecture:
Asks the LLM to read each person's/company's bio and assign numeric
latent variables that capture nuance a procedural rule can't.

These latent vectors are then consumed by generate_edges_hybrid.py
(Step 2) to build the graph procedurally.

Usage:
    python test11/generate_latent_vars_api.py          # interactive
    python test11/generate_latent_vars_api.py --auto    # skip prompts
    python test11/generate_latent_vars_api.py --model gemini-2.5-flash
"""
import argparse
import os, json, sys, time
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path

from llm_provider import get_llm_client

MODEL = "gemini-3.1-flash-lite-preview"

import random as _rnd
from contracts import GENRES  # V17: used in avoid_genres prompt

API_TIMEOUT = 70  # seconds; kills hung API calls


def _jitter_to_3dp(latent_vars: list[dict]) -> list[dict]:
    """Post-process latent vars to ensure 3 decimal precision.

    LLMs tend to output round numbers (0.3, 0.25) despite explicit
    prompt instructions. This adds a tiny seeded jitter (Â±0.005) to
    any float with fewer than 3 significant decimals, then rounds to 3dp.
    """
    for entry in latent_vars:
        seed_str = entry.get("name", str(entry.get("person_id", 0)))
        rng = _rnd.Random(seed_str)
        for k, v in entry.items():
            if not isinstance(v, float):
                if isinstance(v, list):
                    entry[k] = [
                        round(x + rng.uniform(-0.005, 0.005), 3)
                        if isinstance(x, float) else x
                        for x in v
                    ]
                continue
            # Check if already 3dp
            s = f"{v:.10f}".rstrip('0')
            dec_part = s.split('.')[-1] if '.' in s else ''
            if len(dec_part) < 3:
                jitter = rng.uniform(-0.005, 0.005)
                entry[k] = round(v + jitter, 3)
    return latent_vars

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

BATCH_SIZE = 50
MAX_RETRIES = 10

# -----------------------------------------------------------------------
# TOKEN TRACKING
# -----------------------------------------------------------------------

class TokenTracker:
    """Track cumulative token usage and cost across all API calls."""
    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.calls = 0
        self.errors = 0
    
    def record_llm_response(self, resp):
        """Record token usage from an LLMResponse object."""
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_cost += resp.cost_usd
        self.calls += 1
        return resp.input_tokens, resp.output_tokens, resp.cost_usd
    
    def record_error(self):
        self.errors += 1
    
    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  TOKEN USAGE SUMMARY\n"
            f"{'='*60}\n"
            f"  Model:         {self.model_name}\n"
            f"  API calls:     {self.calls} ({self.errors} errors)\n"
            f"  Input tokens:  {self.total_input_tokens:,}\n"
            f"  Output tokens: {self.total_output_tokens:,}\n"
            f"  Total tokens:  {self.total_input_tokens + self.total_output_tokens:,}\n"
            f"  Total cost:    ${self.total_cost:.4f}\n"
            f"{'='*60}"
        )


# -----------------------------------------------------------------------
# PROMPTS
# -----------------------------------------------------------------------

def build_person_latent_prompt(batch_persons):
    """Build prompt to assign latent variables to a batch of persons."""
    person_lines = []
    for p in batch_persons:
        pid = p.get("person_id", "?")
        name = p["name"]
        bio = str(p.get("bio", ""))[:200]
        styles = p.get("style_tags", [])
        if isinstance(styles, str):
            styles = [s.strip() for s in styles.split(",")]
        genres = p.get("genre_affinity", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(",")]
        stage = p.get("career_stage", "prime")
        roles = p.get("roles", ["actor"])
        if isinstance(roles, str):
            roles = [roles]
        
        person_lines.append(
            f'[{pid}] {name} | roles: {",".join(roles)} | stage: {stage} '
            f'| styles: {",".join(styles[:4])} | genres: {",".join(genres)} '
            f'| bio: {bio}'
        )
    
    persons_block = "\n".join(person_lines)
    
    return f"""You are assigning numeric latent variables to movie industry persons based on their profiles.
Read each person's bio, styles, genres, and career stage, then assign scores that capture the nuance of their profile.

IMPORTANT: All float values MUST use 3 decimal places (e.g., 0.347, not 0.3). Each person should
have meaningfully different values -- avoid rounding to 0.1 increments.

=== PERSONS ({len(batch_persons)}) ===
{persons_block}

For EACH person, output a JSON object with their person_id and these latent variables:

1. "creative_style_vector": array of 8 floats in [-1, 1] with 3 decimal places. Dimensions:
   [0] methodical (-1) vs spontaneous (+1)
   [1] minimalist (-1) vs maximalist (+1)
   [2] dark/edgy (-1) vs light/uplifting (+1)
   [3] realistic (-1) vs stylized (+1)
   [4] introverted (-1) vs extroverted (+1)
   [5] traditional (-1) vs experimental (+1)
   [6] character-driven (-1) vs plot-driven (+1)
   [7] intimate (-1) vs spectacle (+1)

2. "risk_tolerance": float [0, 1] with 3 decimal places. How willing to take unconventional roles.
   Bio mentions "controversial", "avant-garde", "experimental" -> high.
   "family-friendly", "mainstream", "blockbuster" -> low.

3. "collaboration_style": MUST be exactly one of: "solo", "ensemble", "chameleon", "mentorship".
   No other values allowed. Solo = prefers tight, small crews. Ensemble = thrives in big groups.
   Chameleon = adapts to any format. Mentorship = tends to pair with newcomers / proteges.

4. "controversy_score": float [0, 1] with 3 decimal places. How controversial the person is.
   Mentions of scandals, provocative work, political activism -> high.
   Clean reputation, family-friendly -> low.

5. "public_reputation": float [0, 1] with 3 decimal places. Star power / name recognition.
   Legends and mega-stars -> 0.8-1.0. Rising newcomers -> 0.1-0.3.

6. "budget_band_pref": array of 5 floats [0, 1] with 3 decimal places, one per budget tier [Micro, Indie, Mid, A, Epic].
   How comfortable this person is in each tier. Sum doesn't need to be 1.

7. "artistic_ambition": float [0, 1] with 3 decimal places. High = art-house / experimental / festival focus.
   Low = commercial / mainstream / crowd-pleasing focus.

8. "volatility": float [0, 1] with 3 decimal places. High = swings between hit and flop. Low = consistent performer.
   Use bio cues like "unpredictable", "polarizing", "erratic" -> high.

9. "avoid_genres": array of 1-3 genre strings from this list: {GENRES}.
   Genres this person would NOT want to appear in. Infer from bio + style:
   Family-friendly actors avoid Horror/Thriller; cerebral actors avoid Action/Superhero;
   comedy specialists may avoid Documentary. Pick 1-3 genres that conflict with their profile.

Output ONLY a JSON array of objects. Each object has "person_id" and the 9 variables above.
No markdown fences, no explanation, no extra text."""


def build_company_latent_prompt(batch_companies):
    """Build prompt to assign latent variables to a batch of companies."""
    company_lines = []
    for c in batch_companies:
        cid = c.get("company_id", "?")
        name = c["name"]
        desc = str(c.get("description", ""))[:150]
        genres = c.get("specialty_genres", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(";")]
        tier = c.get("tier", "Mid-Budget")
        
        company_lines.append(
            f'[{cid}] {name} | tier: {tier} | genres: {",".join(genres)} | {desc}'
        )
    
    companies_block = "\n".join(company_lines)
    
    return f"""You are assigning numeric latent variables to movie production companies based on their profiles.

=== COMPANIES ({len(batch_companies)}) ===
{companies_block}

For EACH company, output a JSON object with their company_id and these latent variables:

1. "risk_appetite": float [0, 1]. How willing to take risky, unconventional projects.
   Indie studios doing art-house films -> high. Big tentpole studios -> low.

2. "prestige_score": float [0, 1]. Studio prestige / brand recognition.
   Global studios with Oscar history -> 0.8-1.0. New micro studios -> 0.1-0.3.

3. "genre_portfolio": array of 12 floats [0, 1], one per genre
   [Action, Drama, Comedy, Sci-Fi, Horror, Romance, Thriller, Fantasy, Mystery, Documentary, Crime, Animation].
   How much of their slate is in each genre. Sum should be roughly 1.

4. "budget_tier_focus": array of 5 floats [0, 1], one per tier [Micro, Indie, Mid, A, Epic].
   How much they operate in each tier. Sum should be roughly 1.

5. "market_trend_sensitivity": float [0, 1]. High = chases trends / shifts genres over decades.
   Low = sticks to brand identity and signature slate.

6. "controversy_tolerance": float [0, 1]. How much controversy they tolerate.
   Studios known for provocative content -> high. Family studios -> low.

Output ONLY a JSON array of objects. Each has "company_id" and the 6 variables above.
No markdown fences, no explanation."""


def parse_json_response(text):
    """Extract JSON array from response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Generate latent variables via Gemini API")
    parser.add_argument("--auto",            action="store_true", help="Skip interactive prompts")
    parser.add_argument("--companies-only",  action="store_true", help="Only process companies (skip persons)")
    parser.add_argument("--persons-only",    action="store_true", help="Only process persons (skip companies)")
    parser.add_argument("--model",           default=None,
                        help=f"LLM model override (default: provider default)")
    parser.add_argument("--batch-size",      type=int, default=BATCH_SIZE,
                        help=f"Persons/companies per batch (default: {BATCH_SIZE})")
    args = parser.parse_args()
    _model = args.model  # None = use provider default
    _batch_size = args.batch_size
    
    llm = get_llm_client()
    tracker = TokenTracker(getattr(llm, '_default_model', MODEL))
    os.makedirs(ENTITY_DIR, exist_ok=True)
    
    # --- Load persons -------------------------------------------------
    with open(ENTITY_DIR / "persons.json", encoding="utf-8") as f:
        all_persons = json.load(f)
    
    # Assign latent vars to ALL persons (core + extras)
    # The incremental logic below skips already-processed persons automatically.
    all_process = all_persons
    
    # F4: Stable IDs -- persist person_id so it survives re-runs
    ids_changed = False
    for i, p in enumerate(all_process):
        if "person_id" not in p:
            p["person_id"] = i + 1
            ids_changed = True
    if ids_changed:
        with open(ENTITY_DIR / "persons.json", "w", encoding="utf-8") as f:
            json.dump(all_persons, f, indent=2, ensure_ascii=False)
        print("  Persisted person_id values to persons.json")
    
    core_count = sum(1 for p in all_persons if not p.get("is_extra", False))
    print(f"Loaded {len(all_persons)} total persons")

    # Load existing persons latent vars (always needed to find what's missing)
    latent_path = ENTITY_DIR / "persons_latent.json"
    existing_ids = set()
    existing_latent = []
    if latent_path.exists():
        existing_latent = json.loads(latent_path.read_text(encoding="utf-8"))
        existing_ids = {lv.get("person_id") for lv in existing_latent}
        print(f"  Found {len(existing_latent)} existing person latent vars")

    if args.companies_only:
        print("  --companies-only: skipping persons")
        persons_needing = []
    else:
        persons_needing = [p for p in all_process if p["person_id"] not in existing_ids]
        print(f"Persons needing latent vars: {len(persons_needing)}")
    
    # Cost estimate (gemini-3-flash-preview: $0.25 in / $1.50 out per 1M tokens)
    n_batches = (len(persons_needing) + BATCH_SIZE - 1) // BATCH_SIZE
    est_cost = n_batches * (5000 / 1e6 * 0.25 + 3000 / 1e6 * 1.50)
    print(f"\nBatches: {n_batches}, Est. cost: ${est_cost:.3f}")
    
    if len(persons_needing) == 0:
        print("All persons already have latent vars. Skipping.")
    else:
        if not args.auto:
            proceed = input("Proceed with person latent vars? [y/N]: ").strip().lower()
            if proceed != "y":
                print("Skipped persons.")
                persons_needing = []
        
        if persons_needing:
            all_person_latent = list(existing_latent)
            
            for batch_start in range(0, len(persons_needing), _batch_size):
                batch = persons_needing[batch_start:batch_start + _batch_size]
                batch_num = batch_start // _batch_size + 1
                print(f"\n  Batch {batch_num}/{n_batches} ({len(batch)} persons)...")
                
                prompt = build_person_latent_prompt(batch)
                
                for retry in range(MAX_RETRIES):
                    t0 = time.time()
                    try:
                        response = llm.generate(
                            prompt,
                            model=_model,
                            json_mode=True,
                            temperature=0.3,
                            max_tokens=16384,
                            timeout_sec=API_TIMEOUT,
                            max_attempts=1,
                        )
                        elapsed = time.time() - t0
                    except TimeoutError as e:
                        print(f"  TIMEOUT (retry {retry+1}/{MAX_RETRIES}): {e}")
                        tracker.record_error()
                        time.sleep(15 * (retry + 1))
                        continue
                    except Exception as e:
                        is_503 = "503" in str(e) or "unavailable" in str(e).lower()
                        print(f"  {'503 ' if is_503 else ''}API ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
                        tracker.record_error()
                        time.sleep((70 if is_503 else 5) * (retry + 1))
                        continue
                    
                    inp, out, cost = tracker.record_llm_response(response)
                    print(f"  Tokens: {inp:,} in + {out:,} out = ${cost:.4f} | {elapsed:.1f}s")
                    
                    try:
                        latent_vars = parse_json_response(response.text)

                        # Validate: check we got enough
                        batch_ids = {p["person_id"] for p in batch}
                        got_ids = {lv.get("person_id") for lv in latent_vars}
                        missing = batch_ids - got_ids
                        
                        if len(missing) > 0 and retry < MAX_RETRIES - 1:
                            print(f"  Missing {len(missing)} persons, retrying...")
                            continue
                        
                        all_person_latent.extend(_jitter_to_3dp(latent_vars))
                        print(f"  Got {len(latent_vars)} latent vars (3dp jittered)")
                        break
                        
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"  JSON PARSE ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
                        raw_path = ENTITY_DIR / f"latent_batch{batch_num}_raw.txt"
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(response.text)
                        print(f"  Raw response saved -> {raw_path}")
                        if retry < MAX_RETRIES - 1:
                            time.sleep(3)
                            continue
                
                # Incremental save after each batch
                with open(latent_path, "w", encoding="utf-8") as f:
                    json.dump(all_person_latent, f, indent=2, ensure_ascii=False)
                
                time.sleep(0.5)
            
            print(f"\nSaved {len(all_person_latent)} person latent vars -> {latent_path}")
    
    # --- Load companies -----------------------------------------------
    with open(ENTITY_DIR / "companies.json", encoding="utf-8") as f:
        companies = json.load(f)
    # F4: Stable company IDs
    cids_changed = False
    for i, c in enumerate(companies):
        if "company_id" not in c:
            c["company_id"] = i + 1
            cids_changed = True
    if cids_changed:
        with open(ENTITY_DIR / "companies.json", "w", encoding="utf-8") as f:
            json.dump(companies, f, indent=2, ensure_ascii=False)
        print("  Persisted company_id values to companies.json")
    
    print(f"\nLoaded {len(companies)} companies")
    
    company_latent_path = ENTITY_DIR / "companies_latent.json"
    existing_company_latent = []
    existing_cids = set()
    if company_latent_path.exists():
        existing_company_latent = json.loads(company_latent_path.read_text(encoding="utf-8"))
        existing_cids = {cl.get("company_id") for cl in existing_company_latent}
    
    companies_needing = [c for c in companies if c["company_id"] not in existing_cids]
    n_cbatches = (len(companies_needing) + BATCH_SIZE - 1) // BATCH_SIZE
    
    if len(companies_needing) == 0:
        print("All companies already have latent vars.")
    else:
        print(f"Companies needing latent vars: {len(companies_needing)} ({n_cbatches} batches)")
        
        if not args.auto:
            proceed = input("Proceed with company latent vars? [y/N]: ").strip().lower()
            if proceed != "y":
                print("Skipped companies.")
                companies_needing = []
        if args.persons_only:
            print("  --persons-only: skipping companies")
            companies_needing = []
        
        if companies_needing:
            all_company_latent = list(existing_company_latent)
            
            for batch_start in range(0, len(companies_needing), _batch_size):
                batch = companies_needing[batch_start:batch_start + _batch_size]
                batch_num = batch_start // _batch_size + 1
                print(f"\n  Company batch {batch_num}/{n_cbatches} ({len(batch)} companies)...")
                
                prompt = build_company_latent_prompt(batch)
                
                for retry in range(MAX_RETRIES):
                    t0 = time.time()
                    try:
                        response = llm.generate(
                            prompt,
                            model=_model,
                            json_mode=True,
                            temperature=0.3,
                            max_tokens=16384,
                            timeout_sec=API_TIMEOUT,
                            max_attempts=1,
                        )
                        elapsed = time.time() - t0
                    except TimeoutError as e:
                        print(f"  TIMEOUT (retry {retry+1}/{MAX_RETRIES}): {e}")
                        tracker.record_error()
                        time.sleep(15 * (retry + 1))
                        continue
                    except Exception as e:
                        is_503 = "503" in str(e) or "unavailable" in str(e).lower()
                        print(f"  {'503 ' if is_503 else ''}API ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
                        tracker.record_error()
                        time.sleep((70 if is_503 else 5) * (retry + 1))
                        continue
                    
                    inp, out, cost = tracker.record_llm_response(response)
                    print(f"  Tokens: {inp:,} in + {out:,} out = ${cost:.4f} | {elapsed:.1f}s")
                    
                    try:
                        latent_vars = parse_json_response(response.text)
                        all_company_latent.extend(latent_vars)
                        print(f"  Got {len(latent_vars)} latent vars")
                        break
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"  JSON PARSE ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
                        if retry < MAX_RETRIES - 1:
                            time.sleep(3)
                
                # Incremental save after each batch
                with open(company_latent_path, "w", encoding="utf-8") as f:
                    json.dump(all_company_latent, f, indent=2, ensure_ascii=False)
                
                time.sleep(0.5)
            
            print(f"\nSaved {len(all_company_latent)} company latent vars -> {company_latent_path}")
    
    # Final token summary
    print(tracker.summary())


if __name__ == "__main__":
    main()






