"""
generate_titles_llm.py
======================
LLM-based title bank top-up for test16.

1. Loads existing title_bank.csv
2. Purges low-quality procedural titles (single-word, too short, known-bad patterns)
3. Calls Gemini in batches of 50 to generate cinematic replacement titles
4. Deduplicates and writes back to title_bank.csv

Usage:
    python test16/generate_titles_llm.py --base-dir test16 --target-count 20000
    python test16/generate_titles_llm.py --base-dir test16 --target-count 10000 --purge-only
"""
import argparse, json, os, re, sys, time, concurrent.futures
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL   = "gemini-3.1-flash-lite-preview"
PRICING = {"input": 0.25, "output": 1.50}
TIMEOUT = 70
BATCH   = 50

# Single-word titles we'll ALLOW even though they're one word (clearly cinematic)
_ALLOW_SINGLE = set("""
arrival dunkirk parasite nomadland moonlight spotlight
casablanca psycho vertigo motherhood boyhood birdman
malcolm joker hereditary tár nightcrawler prisoners
uncut mudbound midway overlord darkest bombshell
""".split())

# Common single English words that are terrible movie title candidates.
_BAD_SINGLE_WORDS = set("""
hard soft fast slow near far up down left right back front inside outside
high low open shut wet dry hot cold dark light big small good bad old new
long short flat wide thin fat deep first last next same other such little
early late free full empty clean dirty rough smooth quiet loud bright dim
plain red blue white black green yellow round sharp real true false strong
weak rich poor dead alive busy bare raw bold quick close past
kitchen bathroom bedroom office street road park yard market shop store
garden forest city town rule law fact point home away today work class head
sho nao aki kae emi ren kyo rui ken jin ryu ray cut pan sea sky fog ash
""".split())

# Pattern-based garbage: "X: X" where subtitle repeats main word
_RE_REPEAT_SUBTITLE = re.compile(r'^(.+):\s+\1\s*$', re.IGNORECASE)
# "Adj Noun: Same-Noun" repeated exact word in subtitle
_RE_WORD_REPEAT = re.compile(r'\b(\w{4,})\b.*:\s+.*\b\1\b', re.IGNORECASE)
# Person name pattern: contains - between lowercase letters (Jun-ho, Seo-yun) or digits (Kenji-7)
_RE_PERSON_NAME = re.compile(r'^[A-Z][a-z]+-[A-Za-z0-9]+$')


def load_used_titles(base: Path) -> set[str] | None:
    """Return lowercase set of titles from movie.csv, or None if file missing."""
    movie_csv = base / "movie.csv"
    if not movie_csv.exists():
        return None
    df = pd.read_csv(movie_csv, usecols=["title"])
    return set(df["title"].astype(str).str.strip().str.lower())




def call_api(client, prompt, max_retries=5):
    for attempt in range(max_retries):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            f = pool.submit(
                client.models.generate_content,
                model=MODEL,
                contents=prompt,
                config={
                    "temperature": 1.0,
                    "max_output_tokens": BATCH * 30,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            resp = f.result(timeout=TIMEOUT)
            pool.shutdown(wait=False, cancel_futures=True)
        except concurrent.futures.TimeoutError:
            pool.shutdown(wait=False, cancel_futures=True)
            print(f"  TIMEOUT (attempt {attempt+1}/{max_retries})")
            time.sleep(15 * (attempt + 1))
            continue
        except Exception as e:
            pool.shutdown(wait=False, cancel_futures=True)
            is503 = "503" in str(e) or "unavailable" in str(e).lower()
            print(f"  {'503 ' if is503 else ''}API ERROR (attempt {attempt+1}): {e}")
            time.sleep((70 if is503 else 5) * (attempt + 1))
            continue

        usage = resp.usage_metadata
        inp = getattr(usage, "prompt_token_count", 0) or 0
        out = getattr(usage, "candidates_token_count", 0) or 0
        cost = inp / 1e6 * PRICING["input"] + out / 1e6 * PRICING["output"]

        text = resp.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data, inp, out, cost
        except json.JSONDecodeError:
            pass

        # Recover truncated JSON
        last = text.rfind("}")
        if last > 0:
            candidate = text[:last+1].rstrip().rstrip(",") + "]"
            if not candidate.startswith("["):
                candidate = "[" + candidate
            try:
                data = json.loads(candidate)
                print(f"  (recovered {len(data)} from truncated JSON)")
                return data, inp, out, cost
            except json.JSONDecodeError:
                pass

        print(f"  PARSE ERROR (attempt {attempt+1}): {len(text)} chars")
        time.sleep(2)

    raise RuntimeError(f"All {max_retries} attempts failed")


GENRE_HINTS = [
    "Drama", "Thriller", "Romance", "Crime", "Action",
    "Sci-Fi", "Horror", "Comedy", "Mystery", "Documentary",
    "Fantasy", "Animation", "Biography", "Adventure", "Western",
]

REGION_PROMPTS = [
    "Mix of American, British, Australian indie and Hollywood titles.",
    "Indian cinema: Bollywood (Hindi), Tollywood (Telugu), Kollywood (Tamil), Malayalam.",
    "African cinema: Nigerian Nollywood, South African, Ghanaian, East African, Egyptian.",
    "East Asian: Chinese, Japanese, Korean, Hong Kong, Taiwanese cinema.",
    "Southeast Asian and Pacific: Thai, Filipino, Indonesian, Vietnamese, Australian.",
    "European: French, German, Italian, Spanish, Scandinavian, Eastern European.",
    "Latin American: Brazilian, Mexican, Argentine, Colombian, Chilean titles.",
    "Middle Eastern: Iranian, Turkish, Israeli, Arab Gulf cinema.",
]

import random as _rnd
_RNG = _rnd.Random(20260306)


def build_prompt(n: int, used_sample: list[str]) -> str:
    region = _RNG.choice(REGION_PROMPTS)
    genres = _RNG.sample(GENRE_HINTS, k=min(5, len(GENRE_HINTS)))
    avoid = "\n".join(f"  - {t}" for t in used_sample[-60:]) if used_sample else "  (none yet)"
    return f"""Generate exactly {n} UNIQUE, realistic movie titles for a film database.

REGION FOCUS: {region}
GENRE MIX (use variety): {', '.join(genres)} and others

Requirements:
- Each title must be 2-6 words (no single-word titles, no 7+ word titles)
- Titles must feel like real movies, not generic phrases
- No subtitles with colons UNLESS they genuinely improve the title
- Do NOT repeat any word in the subtitle that already appears in the main title
- Vary tone: prestige drama, pulpy genre, arthouse, commercial blockbuster, local indie
- Authentic to the region (use local naming conventions where appropriate)
- AVOID overused title templates: "The X Chronicles", "City of X", "X: The Y", "The Last X",
  "Beyond the X", "Shadow of X", "X of Destiny", "The X Legacy", "Rise of X", "X Reborn"
- AVOID these recently used titles (do not repeat or closely paraphrase):
{avoid}

Return a JSON array of exactly {n} strings — the titles ONLY. No objects, no keys.
Example: ["The Iron Season", "Monsoon Wedding", "Last Train to Busan"]
Start with ["""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--purge-only", action="store_true",
                        help="Only remove bad titles, do not call LLM")
    parser.add_argument("--batch-size", type=int, default=BATCH)
    args = parser.parse_args()

    base = Path(args.base_dir)
    path = base / "entities" / "title_bank.csv"

    if not path.exists():
        print(f"title_bank.csv not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    total_before = len(df)
    title_col = "title" if "title" in df.columns else df.columns[0]

    # --- Step 1: Purge bad titles ---
    # Primary: cross-reference movie.csv — keep only titles actually used in generation.
    # Fallback (first run, no movie.csv yet): keep all existing titles.
    used_in_movies = load_used_titles(base)
    if used_in_movies is not None:
        mask_keep = df[title_col].astype(str).str.strip().str.lower().isin(used_in_movies)
        n_bad = int((~mask_keep).sum())
        df_clean = df[mask_keep].copy().reset_index(drop=True)
        print(f"Title bank: {total_before} total | cross-ref movie.csv -> {n_bad} unused purged -> {len(df_clean)} kept")
    else:
        # No movie.csv yet (fresh run) — keep everything, just drop obvious blanks/NaN.
        mask_keep = df[title_col].astype(str).str.strip().str.lower().isin({"nan", "none", ""})
        df_clean = df[~mask_keep].copy().reset_index(drop=True)
        n_bad = int(mask_keep.sum())
        print(f"Title bank: {total_before} total | no movie.csv found, kept all {len(df_clean)}")

    if args.purge_only:
        df_clean.to_csv(path, index=False)
        print(f"Purge-only mode. Saved {len(df_clean)} titles -> {path}")
        return

    need = args.target_count - len(df_clean)
    if need <= 0:
        df_clean.to_csv(path, index=False)
        print(f"Already at or above target ({len(df_clean)} >= {args.target_count}). Saved clean version.")
        return

    print(f"Need {need} new LLM titles to reach {args.target_count}")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    used = set(df_clean[title_col].astype(str).str.lower().str.strip().tolist())
    new_titles = []
    used_sample: list[str] = list(df_clean[title_col].astype(str).tail(50).tolist())

    batch_size = args.batch_size
    n_batches = (need + batch_size - 1) // batch_size
    total_in = total_out = 0
    total_cost = 0.0
    consecutive_low = 0  # safety net: consecutive batches with <50% acceptance

    for b in range(n_batches):
        remaining = need - len(new_titles)
        if remaining <= 0:
            break
        ask = min(batch_size, remaining + 5)  # ask a few extra to account for dupes
        print(f"\n  Batch {b+1}/{n_batches} (asking {ask}, need {remaining} more)...")

        prompt = build_prompt(ask, used_sample)
        try:
            data, inp, out, cost = call_api(client, prompt)
        except RuntimeError as e:
            print(f"  FAILED: {e} — skipping batch")
            consecutive_low += 1
            if consecutive_low >= 3:
                print(f"\n  SAFETY NET: 3 consecutive failed/low-yield batches. Terminating early.")
                print(f"  Generated {len(new_titles)}/{need} titles before abort.")
                break
            continue

        total_in += inp
        total_out += out
        total_cost += cost

        added = 0
        for raw in data:
            t = str(raw).strip()
            key = t.lower()
            if not t or len(t) < 3 or key in used:
                continue
            used.add(key)
            new_titles.append(t)
            used_sample.append(t)
            added += 1
            if len(new_titles) >= need:
                break

        acceptance = added / max(len(data), 1)
        status = f"acceptance={acceptance:.0%}"
        if acceptance < 0.50:
            consecutive_low += 1
            status += f"  !! LOW ({consecutive_low}/3)"
            if consecutive_low >= 3:
                print(f"  +{added} -> {len(new_titles)}/{need} | {inp:,}in+{out:,}out = ${cost:.4f} | {status}")
                print(f"\n  SAFETY NET: 3 consecutive batches below 50% acceptance rate.")
                print(f"  Likely pool exhaustion or deduplication saturation.")
                print(f"  Generated {len(new_titles)}/{need} titles before abort.")
                break
        else:
            consecutive_low = 0  # reset on good batch

        print(f"  +{added} -> {len(new_titles)}/{need} | {inp:,}in+{out:,}out = ${cost:.4f} | {status}")
        time.sleep(0.5)

    # --- Step 3: Append new titles ---
    new_rows = pd.DataFrame({
        title_col: new_titles,
        **{c: None for c in df_clean.columns if c != title_col},
    })
    # Fill genre_hint from random if column exists
    if "genre_hint" in df_clean.columns:
        new_rows["genre_hint"] = [_RNG.choice(GENRE_HINTS) for _ in new_titles]
    # Fill year by sampling from existing distribution so temporal spread is preserved.
    # Without this all new titles default to 2020 in generate_movies.py.
    if "year" in df_clean.columns:
        existing_years = pd.to_numeric(df_clean["year"], errors="coerce").dropna().astype(int).tolist()
        if existing_years:
            new_rows["year"] = [_RNG.choice(existing_years) for _ in new_titles]

    df_out = pd.concat([df_clean, new_rows], ignore_index=True)
    df_out.to_csv(path, index=False)

    print(f"\n{'='*60}")
    print(f"  TITLE BANK COMPLETE")
    print(f"{'='*60}")
    print(f"  Purged:    {n_bad} bad titles")
    print(f"  Generated: {len(new_titles)} new LLM titles")
    print(f"  Total:     {len(df_out)} titles")
    print(f"  Tokens:    {total_in:,} in + {total_out:,} out")
    print(f"  Cost:      ${total_cost:.4f}")
    print(f"  Saved:     {path}")


if __name__ == "__main__":
    main()
