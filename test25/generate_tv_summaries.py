"""
Test 13 -- Generate TV Series & Episode Summaries via Gemini API
===============================================================
D4 fix: Fills `plot_summary` in tv_series.csv and `description` in
episodes.csv using a two-tier LLM approach.

Tier 1 -- gemini-2.0-flash (1 call per series):
    Generates a 2-3 paragraph series overview given all metadata
    (title, genre, cast, season count, status).

Tier 2 -- gemini-2.0-flash-lite (1 call per season):
    Generates all episode descriptions in one batch, using:
    - The series overview from Tier 1
    - All episode metadata for the season (titles, ratings, air dates)
    - Cast list for the season
    - The finale line from the previous season (continuity)

Run AFTER generate_movies.py has produced tv_series.csv, seasons.csv,
episodes.csv, episode_cast.csv, and person.csv.

Usage:
    py -3.12 test13/generate_tv_summaries.py                  # full run
    py -3.12 test13/generate_tv_summaries.py --dry-run        # no API calls
    py -3.12 test13/generate_tv_summaries.py --batch-verify 1 # 1 series only
    py -3.12 test13/generate_tv_summaries.py --series-limit N # first N series
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from feather_sink import df_to_arrow, read_table
from llm_provider import get_llm_client

BASE_DIR = Path(__file__).parent

# ── Model config ────────────────────────────────────────────────────────
try:
    from contracts import MODEL_TIERS
    _DEFAULT_MODEL = MODEL_TIERS.get("plot_summaries", "gemini-3.1-flash-lite-preview")
except ImportError:
    _DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
TIER1_MODEL = _DEFAULT_MODEL
TIER2_MODEL = _DEFAULT_MODEL

MAX_RETRIES = 5
_RETRY_DELAYS = [4, 4, 4, 15, 20]
_API_TIMEOUT = 70  # seconds -- hard limit per call
SLEEP_BETWEEN_CALLS = 1.0
SLEEP_ON_ERROR = 5.0

# Cost rates (per 1M tokens) -- conservative estimates
COST_RATE = {
    "gemini-3.1-flash-lite-preview": {"in": 0.02, "out": 0.08},
    "gemini-2.0-flash":        {"in": 0.10, "out": 0.40},
    "gemini-2.0-flash-lite":   {"in": 0.02, "out": 0.08},
}


# ═══════════════════════════════════════════════════════════════════════
# TOKEN / COST TRACKER
# ═══════════════════════════════════════════════════════════════════════

class TokenTracker:
    def __init__(self):
        self.by_model: dict[str, dict] = {}

    def record(self, model: str, response) -> tuple[int, int, float]:
        inp = getattr(response, "input_tokens", 0) or 0
        out = getattr(response, "output_tokens", 0) or 0
        cost = getattr(response, "cost_usd", 0.0) or 0.0
        if model not in self.by_model:
            self.by_model[model] = {"calls": 0, "inp": 0, "out": 0, "cost": 0.0, "errors": 0}
        m = self.by_model[model]
        m["calls"] += 1
        m["inp"] += inp
        m["out"] += out
        m["cost"] += cost
        return inp, out, cost

    def record_error(self, model: str):
        if model not in self.by_model:
            self.by_model[model] = {"calls": 0, "inp": 0, "out": 0, "cost": 0.0, "errors": 0}
        self.by_model[model]["errors"] += 1

    def summary(self) -> str:
        lines = ["\n" + "="*60, "  TOKEN USAGE SUMMARY", "="*60]
        total_cost = 0.0
        for model, m in sorted(self.by_model.items()):
            lines.append(f"  {model}")
            lines.append(f"    Calls:  {m['calls']} ({m['errors']} errors)")
            lines.append(f"    Tokens: {m['inp']:,} in + {m['out']:,} out")
            lines.append(f"    Cost:   ${m['cost']:.4f}")
            total_cost += m["cost"]
        lines += ["-"*60, f"  TOTAL COST: ${total_cost:.4f}", "="*60]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data(base_dir: Path) -> dict:
    """Load all TV-related tables into DataFrames, preferring Arrow."""
    def _load(name, table_name: str | None = None):
        df = read_table(str(base_dir / name), table_name)
        if df is None or df.empty:
            print(f"  WARNING: {name}.arrow/.csv not found or empty")
            return pd.DataFrame()
        return df

    series_df = _load("tv_series", "tv_series")
    seasons_df = _load("seasons", "seasons")
    episodes_df = _load("episodes", "episodes")
    ep_cast_df = _load("episode_cast", "episode_cast")
    persons_df = _load("person")
    if persons_df.empty:
        persons_df = _load("persons_enriched")

    # Build person name lookup
    name_map: dict[int, str] = {}
    if not persons_df.empty and "person_id" in persons_df.columns:
        for _, row in persons_df.iterrows():
            name_map[int(row["person_id"])] = str(row.get("name", f"Person {row['person_id']}"))

    # Build cast lists per series (from episode_cast)
    series_cast: dict[int, list[str]] = defaultdict(list)
    if not ep_cast_df.empty:
        for _, row in ep_cast_df.iterrows():
            sid = int(row.get("series_id", 0))
            pid = int(row.get("person_id", 0))
            name = name_map.get(pid, f"Actor {pid}")
            if name not in series_cast[sid]:
                series_cast[sid].append(name)

    # Build episode lists per season
    season_episodes: dict[int, list[dict]] = defaultdict(list)
    if not episodes_df.empty:
        for _, row in episodes_df.iterrows():
            sn_id = int(row.get("season_id", 0))
            season_episodes[sn_id].append({
                "episode_id": int(row["episode_id"]),
                "episode_number": int(row.get("episode_number", 0)),
                "title": str(row.get("title", "")),
                "runtime_minutes": int(row.get("runtime_minutes", 45)),
                "rating": float(row.get("rating", 7.0)),
                "air_date": str(row.get("air_date", "")),
                # skip if already has description
                "description": str(row.get("description", "")) if "description" in episodes_df.columns else "",
            })
        # Sort by episode number within each season
        for sn_id in season_episodes:
            season_episodes[sn_id].sort(key=lambda e: e["episode_number"])

    # Build seasons per series
    series_seasons: dict[int, list[dict]] = defaultdict(list)
    if not seasons_df.empty:
        for _, row in seasons_df.iterrows():
            sid = int(row.get("series_id", 0))
            series_seasons[sid].append({
                "season_id": int(row["season_id"]),
                "season_number": int(row.get("season_number", 1)),
                "year": int(row.get("year", 2020)),
                "episode_count": int(row.get("episode_count", 8)),
                "avg_rating": float(row.get("avg_rating", 7.0)),
            })
        for sid in series_seasons:
            series_seasons[sid].sort(key=lambda s: s["season_number"])

    return {
        "series_df": series_df,
        "seasons_df": seasons_df,
        "episodes_df": episodes_df,
        "series_cast": series_cast,
        "series_seasons": series_seasons,
        "season_episodes": season_episodes,
    }


# ═══════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def build_series_prompt(row: dict, cast_names: list[str], seasons: list[dict]) -> str:
    title = row.get("title", "Unknown")
    genre = row.get("genre", "Drama")
    country = row.get("country", "USA")
    language = row.get("language", "English")
    network = row.get("network", "")
    year_start = row.get("year_start", 2015)
    year_end = row.get("year_end", "")
    status = row.get("status", "Ended")
    n_seasons = row.get("total_seasons", len(seasons))
    overall_rating = row.get("overall_rating", 7.0)

    year_range = f"{year_start}-{year_end}" if year_end else f"{year_start}-present"
    cast_str = ", ".join(cast_names[:8]) if cast_names else "ensemble cast"
    season_summary = ", ".join(
        f"S{s['season_number']} ({s['episode_count']} eps, {s['year']}, {s['avg_rating']:.1f}★)"
        for s in seasons
    )

    return f"""Write a compelling 2-3 paragraph series overview for this TV show, as it would appear on a streaming platform's "About" page. This is SYNTHETIC FICTIONAL data.

SERIES METADATA:
  Title:    {title}
  Genre:    {genre}
  Country:  {country} | Language: {language}
  Network:  {network}
  Years:    {year_range} | Status: {status}
  Seasons:  {n_seasons} ({season_summary})
  Rating:   {overall_rating:.1f}/10
  Cast:     {cast_str}

Requirements:
- Paragraph 1 (3-4 sentences): Premise and setting. Who are the main characters? What world do they inhabit?
- Paragraph 2 (2-3 sentences): Themes and tone. What makes this show distinctive?
- Paragraph 3 (1-2 sentences): Any series arc or something that draws viewers back season after season.
- Match the genre tone: {genre} -- be authentic to that register.
- Use the cast names naturally (don't just list them).
- The series should feel like a real show with a coherent identity.
- Do NOT mention it is synthetic or fictional.
- 120-200 words total.

Output ONLY the series overview text, no JSON, no headings."""


def build_season_prompt(
    series_title: str,
    series_summary: str,
    genre: str,
    season: dict,
    episodes: list[dict],
    cast_names: list[str],
    previous_finale_line: str,
) -> str:
    sn = season["season_number"]
    year = season["year"]
    avg_rating = season["avg_rating"]
    cast_str = ", ".join(cast_names[:6]) if cast_names else "ensemble cast"

    prev_context = (
        f'\nContinuity from Season {sn - 1} finale: "{previous_finale_line}"'
        if previous_finale_line and sn > 1
        else ""
    )

    ep_lines = "\n".join(
        f'  [{ep["episode_id"]}] Ep{ep["episode_number"]}: "{ep["title"]}" | '
        f'{ep["runtime_minutes"]}min | {ep["air_date"]} | ⭐{ep["rating"]:.1f}'
        for ep in episodes
    )

    return f"""Write one-paragraph episode descriptions for every episode in this TV season. This is SYNTHETIC FICTIONAL data.

SERIES: "{series_title}" ({genre})
SERIES OVERVIEW:
{series_summary}
{prev_context}

SEASON {sn} ({year}, avg rating {avg_rating:.1f}★, cast: {cast_str}):
{ep_lines}

Requirements per episode description:
- 2-3 sentences, 35-60 words
- Must advance the season arc logically (earlier episodes set up later ones)
- Higher-rated episodes should feel like standout/pivotal moments
- The LAST episode of the season should end on a note that could be resolved or a cliffhanger
- Genre tone: {genre}
- Use character names from the cast list naturally
- Do NOT repeat the episode title verbatim as the first words
- Do NOT mention ratings or runtime
- Each description must be distinct -- no template repetition

Output ONLY a JSON object mapping episode_id (integer) to description (string), plus a special key "_finale_line" containing the last 1-2 sentences of the final episode's description for season continuity.

Example format:
{{"123": "Description...", "124": "Description...", "_finale_line": "The finale's closing lines..."}}

No markdown fences, no explanation."""


# ═══════════════════════════════════════════════════════════════════════
# JSON PARSING HELPERS
# ═══════════════════════════════════════════════════════════════════════

def parse_json_response(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text)


def call_api(llm, model: str, prompt: str, tracker: TokenTracker,
             dry_run: bool = False, label: str = "") -> str | None:
    """Single API call with 5-retry + 70s hang detection."""
    if dry_run:
        print(f"  [DRY-RUN] Would call {model} for: {label}")
        print(f"  [DRY-RUN] Prompt ({len(prompt)} chars): {prompt[:200]}...")
        return None

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            response = llm.generate(
                prompt,
                model=model,
                temperature=0.75,
                max_tokens=8192,
                timeout_sec=_API_TIMEOUT,
                max_attempts=1,
            )
            elapsed = time.time() - t0
            inp, out, cost = tracker.record(model, response)
            print(f"    {model}: {inp:,}in + {out:,}out = ${cost:.4f} ({elapsed:.1f}s)")
            return response.text
        except Exception as e:
            tracker.record_error(model)
            print(f"    API ERROR [{label}] attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[attempt]
                print(f"    Retrying in {delay}s...")
                time.sleep(delay)
    return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate TV series and episode summaries")
    parser.add_argument("--base-dir", default=str(BASE_DIR), help="Dataset directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts, make zero API calls")
    parser.add_argument("--series-limit", type=int, default=None,
                        help="Process only first N series (default: all)")
    parser.add_argument("--batch-verify", type=int, default=None,
                        help="Process exactly N series then exit (for inspection)")
    parser.add_argument("--tier1-model", default=TIER1_MODEL,
                        help=f"Model for series summaries (default: {TIER1_MODEL})")
    parser.add_argument("--tier2-model", default=TIER2_MODEL,
                        help=f"Model for episode batches (default: {TIER2_MODEL})")
    args = parser.parse_args()

    t1_model = args.tier1_model
    t2_model = args.tier2_model

    # Load .env
    base_dir = Path(args.base_dir).resolve()
    load_dotenv(BASE_DIR.parent / ".env")
    llm = get_llm_client() if not args.dry_run else None

    tracker = TokenTracker()

    # Load data
    print("Loading TV data...")
    data = load_data(base_dir)
    series_df = data["series_df"]
    episodes_df = data["episodes_df"]
    series_cast = data["series_cast"]
    series_seasons = data["series_seasons"]
    season_episodes = data["season_episodes"]

    if series_df.empty:
        print("tv_series.arrow / tv_series.csv not found or empty -- run generate_movies.py first")
        sys.exit(1)

    # Add columns if missing
    if "plot_summary" not in series_df.columns:
        series_df["plot_summary"] = ""
    if "description" not in episodes_df.columns:
        episodes_df["description"] = ""

    # Build existing summary maps for resume support
    existing_series_summaries: dict[int, str] = {}
    for _, row in series_df.iterrows():
        ps = row.get("plot_summary", "")
        if pd.notna(ps) and str(ps).strip():
            existing_series_summaries[int(row["series_id"])] = str(ps)

    existing_ep_descriptions: dict[int, str] = {}
    if not episodes_df.empty and "description" in episodes_df.columns:
        for _, row in episodes_df.iterrows():
            d = row.get("description", "")
            if pd.notna(d) and str(d).strip():
                existing_ep_descriptions[int(row["episode_id"])] = str(d)

    # Determine which series to process
    all_series = series_df.to_dict("records")
    limit = args.batch_verify or args.series_limit
    if limit:
        all_series = all_series[:limit]

    n_series = len(all_series)
    n_seasons_total = sum(len(series_seasons.get(int(s["series_id"]), [])) for s in all_series)
    print(f"  Series to process: {n_series} | Seasons: {n_seasons_total}")
    print(f"  Already have series summary: {len(existing_series_summaries)}")
    print(f"  Already have episode descriptions: {len(existing_ep_descriptions)}")
    if args.dry_run:
        print("  [DRY-RUN MODE -- no API calls will be made]")

    # ── TIER 1: Series summaries ─────────────────────────────────────────
    print(f"\n{'='*60}\nTIER 1: Series summaries ({t1_model})\n{'='*60}")

    all_series_summaries: dict[int, str] = dict(existing_series_summaries)

    for i, series_row in enumerate(all_series):
        sid = int(series_row["series_id"])
        title = series_row.get("title", f"Series {sid}")

        if sid in all_series_summaries:
            print(f"  [{i+1}/{n_series}] '{title}' -- already has summary, skipping")
            continue

        print(f"  [{i+1}/{n_series}] '{title}'...")

        cast_names = series_cast.get(sid, [])
        seasons = series_seasons.get(sid, [])
        prompt = build_series_prompt(series_row, cast_names, seasons)

        text = call_api(llm, t1_model, prompt, tracker,
                        dry_run=args.dry_run, label=f"series/{title}")
        if text:
            all_series_summaries[sid] = text.strip()
        elif not args.dry_run:
            print(f"    FAILED for series {sid}, using empty string")
            all_series_summaries[sid] = ""

        if not args.dry_run:
            time.sleep(SLEEP_BETWEEN_CALLS)

    # Write Tier 1 results back to series_df
        if not args.dry_run:
            series_df["plot_summary"] = series_df["series_id"].map(all_series_summaries).fillna("")
            series_path = base_dir / "tv_series.csv"
            df_to_arrow(series_df, str(base_dir / "tv_series.arrow"), table_name="tv_series")
            series_df.to_csv(series_path, index=False)
            print(f"\n  Saved series summaries -> {series_path}")

    # ── TIER 2: Episode descriptions ──────────────────────────────────────
    print(f"\n{'='*60}\nTIER 2: Episode descriptions ({t2_model})\n{'='*60}")

    all_ep_descriptions: dict[int, str] = dict(existing_ep_descriptions)

    for i, series_row in enumerate(all_series):
        sid = int(series_row["series_id"])
        title = series_row.get("title", f"Series {sid}")
        genre = str(series_row.get("genre", "Drama"))
        cast_names = series_cast.get(sid, [])
        series_summary = all_series_summaries.get(sid, f"A {genre} series titled '{title}'.")
        seasons = series_seasons.get(sid, [])

        print(f"\n  Series [{i+1}/{n_series}]: '{title}' ({len(seasons)} seasons)")

        previous_finale_line = ""

        for sn_idx, season in enumerate(seasons):
            sn_id = season["season_id"]
            sn_num = season["season_number"]
            episodes = season_episodes.get(sn_id, [])

            if not episodes:
                continue

            # Check if all episodes in this season already have descriptions
            ep_ids = [ep["episode_id"] for ep in episodes]
            already_done = sum(1 for eid in ep_ids if eid in all_ep_descriptions)
            if already_done == len(ep_ids):
                print(f"    S{sn_num}: all {len(episodes)} eps already described, skipping")
                # Still need to set continuity from last episode
                last_ep_id = ep_ids[-1]
                last_desc = all_ep_descriptions.get(last_ep_id, "")
                if last_desc:
                    sentences = [s.strip() for s in last_desc.split(".") if s.strip()]
                    previous_finale_line = ". ".join(sentences[-2:]) + "." if len(sentences) >= 2 else last_desc
                continue

            print(f"    S{sn_num} ({len(episodes)} eps)...", end=" ", flush=True)
            prompt = build_season_prompt(
                series_title=title,
                series_summary=series_summary,
                genre=genre,
                season=season,
                episodes=episodes,
                cast_names=cast_names,
                previous_finale_line=previous_finale_line,
            )

            text = call_api(llm, t2_model, prompt, tracker,
                            dry_run=args.dry_run, label=f"{title}/S{sn_num}")

            if text:
                try:
                    parsed = parse_json_response(text)
                    finale_line = str(parsed.pop("_finale_line", ""))
                    previous_finale_line = finale_line

                    # Map episode_id -> description (keys may be strings)
                    for ep_id_str, desc in parsed.items():
                        try:
                            eid = int(ep_id_str)
                            if str(desc).strip():
                                all_ep_descriptions[eid] = str(desc).strip()
                        except (ValueError, TypeError):
                            continue

                    got = sum(1 for eid in ep_ids if eid in all_ep_descriptions)
                    print(f"got {got}/{len(episodes)}")

                    # Positional fallback: if we got right count but wrong IDs
                    if got < len(episodes):
                        values = [v for v in parsed.values() if isinstance(v, str) and v.strip()]
                        if len(values) == len(episodes):
                            for ep, desc in zip(episodes, values):
                                eid = ep["episode_id"]
                                if eid not in all_ep_descriptions:
                                    all_ep_descriptions[eid] = desc.strip()
                            print(f"    Positional fallback applied for S{sn_num}")

                except json.JSONDecodeError as e:
                    print(f"    JSON PARSE ERROR for S{sn_num}: {e}")
                    # Save raw response for inspection
                    raw_dir = BASE_DIR / "_dev"
                    raw_dir.mkdir(exist_ok=True)
                    raw_path = raw_dir / f"tv_s{sid}_sn{sn_num}_raw.txt"
                    raw_path.write_text(text, encoding="utf-8")
                    print(f"    Raw saved -> {raw_path}")
            else:
                print("FAILED")

            if not args.dry_run:
                time.sleep(SLEEP_BETWEEN_CALLS)

        # Incremental save after each series
        if not args.dry_run:
            episodes_df["description"] = episodes_df["episode_id"].map(all_ep_descriptions).fillna("")
            df_to_arrow(episodes_df, str(base_dir / "episodes.arrow"), table_name="episodes")
            ep_path = base_dir / "episodes.csv"
            episodes_df.to_csv(ep_path, index=False)

    # ── FINAL SAVE ────────────────────────────────────────────────────────
    if not args.dry_run:
        print(f"\n{'='*60}")
        series_df["plot_summary"] = series_df["series_id"].map(all_series_summaries).fillna("")
        df_to_arrow(series_df, str(base_dir / "tv_series.arrow"), table_name="tv_series")
        series_df.to_csv(base_dir / "tv_series.csv", index=False)
        episodes_df["description"] = episodes_df["episode_id"].map(all_ep_descriptions).fillna("")
        df_to_arrow(episodes_df, str(base_dir / "episodes.arrow"), table_name="episodes")
        episodes_df.to_csv(base_dir / "episodes.csv", index=False)

        filled_series = (series_df["plot_summary"] != "").sum()
        filled_eps = (episodes_df["description"] != "").sum()
        print(f"  Series summaries: {filled_series}/{len(series_df)}")
        print(f"  Episode descriptions: {filled_eps}/{len(episodes_df)}")

    if args.batch_verify:
        print(f"\n[batch-verify] Processed {args.batch_verify} series. Inspect output before full run.")

    print(tracker.summary())


if __name__ == "__main__":
    main()
