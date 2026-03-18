"""
Test 11 -- Generate Plot Summaries via Gemini API
=================================================
Fills the `plot_summary` column in movie.csv using genre, cast,
director, and keywords as context.

Run AFTER generate_movies.py (needs assembled movie data).

Usage:
    python test11/generate_plot_summaries_api.py          # interactive
    python test11/generate_plot_summaries_api.py --auto    # skip prompts
    python test11/generate_plot_summaries_api.py --model gemini-2.5-pro
"""
import argparse
import os, json, sys, time, csv
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
from llm_provider import get_llm_client

MODEL = "gemini-3.1-flash-lite-preview"

BASE_DIR = Path(__file__).parent
BATCH_SIZE = 50
MAX_RETRIES = 5
_RETRY_DELAYS = [4, 4, 4, 15, 20]
_API_TIMEOUT = 70  # seconds -- hard limit per call


# ═══════════════════════════════════════════════════════════════════════
# TOKEN TRACKING
# ═══════════════════════════════════════════════════════════════════════

class TokenTracker:
    """Track cumulative token usage and cost across all API calls."""
    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.calls = 0
        self.errors = 0
    
    def record(self, response):
        """Extract and record token usage from LLMResponse.
        
        Called immediately after API response arrives, so even if 
        JSON parsing fails, we still have accurate token counts.
        """
        inp = getattr(response, "input_tokens", 0) or 0
        out = getattr(response, "output_tokens", 0) or 0
        cost = getattr(response, "cost_usd", 0.0) or 0.0
        
        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.total_cost += cost
        self.calls += 1
        
        return inp, out, cost
    
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


# ═══════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Simple Levenshtein distance for near-duplicate detection."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _extract_key_phrases(summary: str) -> list:
    """Extract 2-3 key phrases from a summary for the anti-duplication blacklist."""
    phrases = []
    sentences = summary.split('.')
    if sentences:
        # First sentence opening (up to 8 words)
        first = sentences[0].strip()
        words = first.split()[:8]
        if len(words) >= 3:
            phrases.append(' '.join(words))
    # Any distinctive multi-word fragment from later in the text
    if len(sentences) > 1:
        second = sentences[1].strip()
        words2 = second.split()[:6]
        if len(words2) >= 3:
            phrases.append(' '.join(words2))
    return phrases


def build_summary_prompt(movies_batch, blacklist=None):
    """Build prompt for generating plot summaries for a batch of movies."""
    movie_lines = []
    for m in movies_batch:
        mid = m["title_id"]
        title = m["title"]
        genre = m["genre"]
        year = m["year"]
        director = m.get("director", "Unknown")
        actors = m.get("actors", "")
        keywords = m.get("keywords", "")
        tier = m.get("production_tier", "Mid")
        
        movie_lines.append(
            f'[{mid}] "{title}" ({year}) | Genre: {genre} | Tier: {tier} '
            f'| Director: {director} | Cast: {actors[:150]} '
            f'| Keywords: {keywords}'
        )
    
    movies_block = "\n".join(movie_lines)
    
    blacklist_block = ""
    if blacklist:
        # Take last 40 phrases max to keep prompt manageable
        recent = blacklist[-40:]
        blacklist_block = f"""\n6. AVOID these phrasings used in prior batches (do NOT repeat them):\n   {'; '.join(recent)}\n7. Each summary MUST start with a DIFFERENT opening word/phrase than any other summary"""

    return f"""Write brief plot summaries for these {len(movies_batch)} movies. Each summary should be 2-3 sentences that describe the basic plot, tone, and themes -- as if it's the "Overview" on a movie database page.

=== MOVIES ===
{movies_block}

Rules:
1. Each summary is 2-3 sentences, 40-80 words
2. Match the genre and tone (horror = dark, comedy = light, etc.)
3. Use the cast/keywords/director to create coherent plot details
4. Don't just list the genre -- tell a mini-story
5. Each summary must be UNIQUE and specific to the movie{blacklist_block}

Output a JSON array of objects: {{"title_id": <id>, "plot_summary": "<summary>"}}
No markdown fences, no explanation."""


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Generate plot summaries via Gemini API")
    parser.add_argument("--auto", action="store_true", help="Skip interactive prompts")
    parser.add_argument("--model", default=MODEL,
                        help=f"Gemini model to use (default: {MODEL})")
    args = parser.parse_args()

    # Override module-level MODEL with CLI arg
    _model = args.model
    
    # Load API key from .env or environment
    load_dotenv(BASE_DIR.parent / ".env")
    llm = get_llm_client()
    tracker = TokenTracker()
    
    # Load movie + flat data
    movie_path = BASE_DIR / "movie.csv"
    flat_path = BASE_DIR / "movies_flat.csv"
    
    if not movie_path.exists():
        print("movie.csv not found -- run generate_movies.py first")
        return
    
    movies_df = pd.read_csv(movie_path)
    flat_df = pd.read_csv(flat_path) if flat_path.exists() else None
    
    # Build movie context for prompts
    movies_data = []
    flat_title_ids = set(flat_df["title_id"].values) if flat_df is not None else set()
    for _, row in movies_df.iterrows():
        mid = row["title_id"]
        m = {
            "title_id": mid,
            "title": row["title"],
            "genre": row["genre"],
            "year": row["year"],
            "production_tier": row.get("production_tier", "Mid"),
        }
        
        # Get director + actors + keywords from flat table
        if flat_df is not None and mid in flat_title_ids:
            flat_row = flat_df[flat_df["title_id"] == mid].iloc[0]
            m["director"] = str(flat_row.get("director", "Unknown"))
            m["actors"] = str(flat_row.get("actors", ""))
            m["keywords"] = str(flat_row.get("keywords", ""))
        else:
            m["director"] = "Unknown"
            m["actors"] = ""
            m["keywords"] = ""
        
        movies_data.append(m)
    
    # Check which movies already have summaries
    existing_summaries = {}
    if "plot_summary" in movies_df.columns:
        for _, row in movies_df.iterrows():
            ps = row.get("plot_summary")
            if pd.notna(ps) and str(ps).strip():
                existing_summaries[row["title_id"]] = str(ps)
    
    movies_needing = [m for m in movies_data if m["title_id"] not in existing_summaries]
    print(f"Total movies: {len(movies_data)}")
    print(f"Already have summary: {len(existing_summaries)}")
    print(f"Needing summary: {len(movies_needing)}")
    
    if not movies_needing:
        print("All movies already have summaries!")
        return
    
    n_batches = (len(movies_needing) + BATCH_SIZE - 1) // BATCH_SIZE
    est_cost = n_batches * (5000 / 1e6 * 0.25 + 5000 / 1e6 * 1.50)
    print(f"Batches: {n_batches}, Est. cost: ${est_cost:.3f}")
    
    if not args.auto:
        proceed = input("Proceed? [y/N]: ").strip().lower()
        if proceed != "y":
            print("Aborted.")
            return
    
    all_summaries = dict(existing_summaries)
    # v11 P1: Anti-duplication -- rolling phrase blacklist
    phrase_blacklist = []
    recent_summaries = []  # rolling window of last 30 summaries for Levenshtein check
    dup_warnings = 0

    for batch_start in range(0, len(movies_needing), BATCH_SIZE):
        batch = movies_needing[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"\n  Batch {batch_num}/{n_batches} ({len(batch)} movies)...")

        prompt = build_summary_prompt(batch, blacklist=phrase_blacklist if phrase_blacklist else None)
        
        for retry in range(MAX_RETRIES):
            t0 = time.time()
            try:
                # Threading-based hang detection (70s hard timeout)
                import threading as _thr
                _result = [None, None]  # [response, exception]
                def _do_call():
                    try:
                        _result[0] = llm.generate(
                            prompt,
                            model=_model,
                            temperature=0.7,
                            max_tokens=16384,
                        )
                    except Exception as _e:
                        _result[1] = _e
                _t = _thr.Thread(target=_do_call, daemon=True)
                _t.start()
                _t.join(timeout=_API_TIMEOUT)
                if _t.is_alive():
                    raise TimeoutError(f"API call hung for >{_API_TIMEOUT}s")
                if _result[1] is not None:
                    raise _result[1]
                response = _result[0]
                elapsed = time.time() - t0
            except Exception as e:
                err_str = str(e)
                is_503 = "503" in err_str or "UNAVAILABLE" in err_str.upper()
                is_hang = isinstance(e, TimeoutError)
                print(f"  API ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
                tracker.record_error()
                if retry < MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[retry]
                    print(f"  Retrying in {delay}s...")
                    time.sleep(delay)
                continue
            
            # Record tokens BEFORE parsing (never lose metadata)
            inp, out, cost = tracker.record(response)
            print(f"  Tokens: {inp:,} in + {out:,} out = ${cost:.4f} | {elapsed:.1f}s")
            
            try:
                summaries = parse_json_response(response.text)
                for s in summaries:
                    mid = s.get("title_id")
                    ps = s.get("plot_summary", "")
                    if mid and ps:
                        all_summaries[mid] = ps
                
                batch_ids = {m["title_id"] for m in batch}
                got_ids = {s.get("title_id") for s in summaries}
                missing = batch_ids - got_ids
                
                # Positional fallback: if LLM returned wrong title_ids but
                # correct count, map by position (common LLM failure mode)
                if len(missing) > 0 and len(summaries) == len(batch):
                    positional_mapped = 0
                    for s, m in zip(summaries, batch):
                        ps = s.get("plot_summary", "")
                        if ps and m["title_id"] not in all_summaries:
                            all_summaries[m["title_id"]] = ps
                            positional_mapped += 1
                    if positional_mapped > 0:
                        print(f"  Positional fallback: mapped {positional_mapped} summaries")
                        missing = batch_ids - set(all_summaries.keys())
                
                if len(missing) > 0 and retry < MAX_RETRIES - 1:
                    print(f"  Missing {len(missing)} summaries, retrying...")
                    continue
                
                print(f"  Got {len(summaries)} summaries")
                break
                
            except Exception as e:
                print(f"  JSON PARSE ERROR (retry {retry+1}): {e}")
                raw_path = BASE_DIR / f"_dev/plots_batch{batch_num}_raw.txt"
                os.makedirs(BASE_DIR / "_dev", exist_ok=True)
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"  Raw response saved -> {raw_path}")
                if retry < MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[retry]
                    time.sleep(delay)
        
        # Incremental save after each batch
        movies_df["plot_summary"] = movies_df["title_id"].map(all_summaries).fillna("")
        movies_df.to_csv(movie_path, index=False)

        # v11 P1: Update anti-duplication blacklist with this batch's phrases
        for mid in [m["title_id"] for m in batch]:
            ps = all_summaries.get(mid, "")
            if ps:
                phrase_blacklist.extend(_extract_key_phrases(ps))
                # Levenshtein near-duplicate check against recent summaries
                ps_lower = ps.lower().strip()
                for prev in recent_summaries:
                    dist = _levenshtein_distance(ps_lower[:80], prev[:80])
                    if dist < 15:
                        dup_warnings += 1
                        if dup_warnings <= 5:  # don't spam
                            print(f"  ⚠ Near-duplicate detected (Levenshtein={dist}): {ps[:60]}...")
                recent_summaries.append(ps_lower)
                if len(recent_summaries) > 30:
                    recent_summaries.pop(0)
        # Keep blacklist bounded
        if len(phrase_blacklist) > 100:
            phrase_blacklist = phrase_blacklist[-60:]
        
        time.sleep(2)
    
    # Final save: movie.csv
    movies_df["plot_summary"] = movies_df["title_id"].map(all_summaries).fillna("")
    movies_df.to_csv(movie_path, index=False)
    
    # Update movies_flat.csv description column
    if flat_df is not None:
        flat_df["description"] = flat_df["title_id"].map(all_summaries).fillna("")
        flat_df.to_csv(flat_path, index=False)
    
    filled = sum(1 for v in all_summaries.values() if v.strip())
    print(f"\n  Filled: {filled}/{len(movies_df)} plot summaries")
    print(f"  Saved -> {movie_path}")
    
    # Token summary
    print(tracker.summary())


if __name__ == "__main__":
    main()


