"""
test25 end-to-end pipeline runner.

Design goals:
  - fresh-machine bootstrap from procedural entities + LLM enrichment
  - modern Arrow-aware completion checks
  - explicit year-span control via title-bank generation
  - one command works for both smoke tests and large runs
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from contracts import ENTITY_COUNTS
from feather_sink import read_table

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_FILE = BASE_DIR / ".pipeline_checkpoint.json"

ROOT_ARROW_TABLES = [
    "movie",
    "cast_info",
    "movie_directors",
    "movie_companies",
    "movie_keyword",
    "movie_crew",
    "release_dates",
    "box_office_weekly",
    "box_office_by_territory",
    "box_office_daily",
    "reviews",
    "awards",
    "locations",
    "alternate_titles",
    "ratings_breakdown",
    "movie_links",
    "person_demographics",
    "tv_series",
    "seasons",
    "episodes",
    "episode_cast",
    "company_links",
    "user_ratings",
    "production_timeline",
    "media_links",
    "person_contracts",
    "world_events",
    "movies_flat",
    "movies_analysis",
    "persons_enriched",
    "companies_enriched",
    "edges_temporal",
    "edges_final",
]

ROOT_COMPAT_FILES = [
    "movie.csv",
    "movies_flat.csv",
    "movies_analysis.csv",
    "tv_series.csv",
    "seasons.csv",
    "episodes.csv",
    "episode_cast.csv",
    "critic_report.json",
]

ENTITY_GENERATED_FILES = [
    "persons.json",
    "companies.json",
    "keywords.json",
    "persons_latent.json",
    "companies_latent.json",
    "character_bank.csv",
    "title_bank.csv",
    "person.csv",
    "person_roles.csv",
    "company.csv",
    "keyword.csv",
    "company_financial_profile.csv",
]

ENTITY_GENERATED_PATTERNS = [
    "latent_batch*_raw.txt",
    "person_enrich*_raw.txt",
]


def _load_table(name: str, table_name: str | None = None) -> pd.DataFrame:
    return read_table(str(BASE_DIR / name), table_name)


def _nonempty_text_fraction(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    values = series.fillna("").astype(str).str.strip()
    return float((values != "").mean())


def _check_persons_enriched(base_dir: Path) -> bool:
    path = base_dir / "entities" / "persons.json"
    if not path.exists():
        return False
    try:
        persons = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(persons, list) or not persons:
        return False
    ok = 0
    for person in persons:
        if (
            str(person.get("bio", "") or "").strip()
            and bool(person.get("style_tags"))
            and bool(person.get("genre_affinity"))
        ):
            ok += 1
    return ok == len(persons)


def _check_latent(base_dir: Path) -> bool:
    persons_path = base_dir / "entities" / "persons.json"
    companies_path = base_dir / "entities" / "companies.json"
    persons_latent_path = base_dir / "entities" / "persons_latent.json"
    companies_latent_path = base_dir / "entities" / "companies_latent.json"

    if not all(path.exists() for path in (persons_path, companies_path, persons_latent_path, companies_latent_path)):
        return False
    try:
        persons = json.loads(persons_path.read_text(encoding="utf-8"))
        companies = json.loads(companies_path.read_text(encoding="utf-8"))
        person_latent = json.loads(persons_latent_path.read_text(encoding="utf-8"))
        company_latent = json.loads(companies_latent_path.read_text(encoding="utf-8"))
        return len(person_latent) >= len(persons) and len(company_latent) >= len(companies)
    except Exception:
        return False


def _check_edges(base_dir: Path) -> bool:
    return (
        (base_dir / "graph" / "runtime_manifest.json").exists()
        or (base_dir / "graph" / "edge_graph.csv").exists()
    )


def _check_entities_csv(base_dir: Path) -> bool:
    edir = base_dir / "entities"
    return all(
        (edir / name).exists()
        for name in ("person.csv", "person_roles.csv", "company.csv", "keyword.csv")
    )


def _check_company_financial_profiles(base_dir: Path) -> bool:
    entities_dir = base_dir / "entities"
    profile_path = entities_dir / "company_financial_profile.csv"
    company_path = entities_dir / "company.csv"
    if not profile_path.exists() or not company_path.exists():
        return False
    try:
        profiles = pd.read_csv(profile_path, low_memory=False)
        companies = pd.read_csv(company_path, low_memory=False)
    except Exception:
        return False
    if profiles.empty or companies.empty:
        return False
    if "company_id" not in profiles.columns or "company_id" not in companies.columns:
        return False
    return int(profiles["company_id"].nunique()) >= int(companies["company_id"].nunique())


def _check_movies(base_dir: Path) -> bool:
    required_paths = [
        base_dir / "movie.arrow",
        base_dir / "movies_flat.arrow",
        base_dir / "persons_enriched.arrow",
        base_dir / "companies_enriched.arrow",
        base_dir / "edges_temporal.arrow",
        base_dir / "edges_final.arrow",
    ]
    if not all(path.exists() for path in required_paths):
        return False
    movies = read_table(str(base_dir / "movie"), "movie")
    flat = read_table(str(base_dir / "movies_flat"))
    return not movies.empty and not flat.empty


def _check_keyword_genres(base_dir: Path) -> bool:
    path = base_dir / "entities" / "keyword.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    if df.empty or "topic_genre" not in df.columns or "pop_weight" not in df.columns:
        return False
    return not df["topic_genre"].isna().any() and not df["pop_weight"].isna().any()


def _check_plots(base_dir: Path) -> bool:
    movies = read_table(str(base_dir / "movie"), "movie")
    return not movies.empty and "plot_summary" in movies.columns and _nonempty_text_fraction(movies["plot_summary"]) == 1.0


def _check_tv_summaries(base_dir: Path) -> bool:
    series = read_table(str(base_dir / "tv_series"), "tv_series")
    episodes = read_table(str(base_dir / "episodes"), "episodes")
    if series.empty or episodes.empty:
        return False
    if "plot_summary" not in series.columns or "description" not in episodes.columns:
        return False
    return (
        _nonempty_text_fraction(series["plot_summary"]) == 1.0
        and _nonempty_text_fraction(episodes["description"]) == 1.0
    )


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed_steps": [], "started_at": None}


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def _reset_generated_outputs() -> None:
    for dirname in ("graph", "checkpoints", "critic"):
        target = BASE_DIR / dirname
        if target.exists():
            shutil.rmtree(target)

    for name in ROOT_ARROW_TABLES:
        path = BASE_DIR / f"{name}.arrow"
        if path.exists():
            path.unlink()
    for name in ROOT_COMPAT_FILES:
        path = BASE_DIR / name
        if path.exists():
            path.unlink()

    entities_dir = BASE_DIR / "entities"
    if entities_dir.exists():
        for name in ENTITY_GENERATED_FILES:
            path = entities_dir / name
            if path.exists():
                path.unlink()
        for pattern in ENTITY_GENERATED_PATTERNS:
            for path in entities_dir.glob(pattern):
                if path.is_file():
                    path.unlink()

    _clear_checkpoint()


def _auto_counts(n_movies: int) -> dict[str, int]:
    movies = max(1, int(n_movies))
    if movies <= 250:
        return {
            "n_titles": movies,
            "n_persons": max(450, int(round(movies * 4.0))),
            "n_companies": max(60, int(round(movies * 0.25))),
            "n_keywords": max(220, int(round(movies * 2.2))),
            "n_characters": max(500, int(round(movies * 5.0))),
        }
    return {
        "n_titles": movies,
        "n_persons": max(1200, int(round(movies * 3.2))),
        "n_companies": max(120, int(round(movies * 0.065))),
        "n_keywords": max(350, int(round(movies * 0.14))),
        "n_characters": max(1500, int(round(movies * 4.0))),
    }


def _resolve_targets(args: argparse.Namespace) -> None:
    auto = _auto_counts(args.n_movies)
    args.n_titles = int(args.n_titles if args.n_titles is not None else auto["n_titles"])
    args.n_persons = int(args.n_persons if args.n_persons is not None else auto["n_persons"])
    args.n_companies = int(args.n_companies if args.n_companies is not None else auto["n_companies"])
    args.n_keywords = int(args.n_keywords if args.n_keywords is not None else auto["n_keywords"])
    args.n_characters = int(args.n_characters if args.n_characters is not None else auto["n_characters"])

    if args.start_year is not None and args.end_year is None:
        args.end_year = args.start_year
    if args.end_year is not None and args.start_year is None:
        args.start_year = args.end_year
    if args.start_year is not None and args.end_year is not None and args.end_year < args.start_year:
        raise ValueError("end-year must be >= start-year")


def _build_steps(args: argparse.Namespace) -> list[dict]:
    return [
        {
            "id": 10,
            "name": "Generate Persons (Procedural)",
            "script": "generate_persons_procedural.py",
            "args": ["--target", str(args.n_persons), "--seed", str(args.seed)],
            "check": lambda bd: (bd / "entities" / "persons.json").exists(),
            "requires_api": False,
            "description": "Generate dedupe-safe procedural person identities.",
        },
        {
            "id": 20,
            "name": "Generate Companies (Procedural)",
            "script": "generate_companies_procedural.py",
            "args": ["--target", str(args.n_companies), "--seed", str(args.seed)],
            "check": lambda bd: (bd / "entities" / "companies.json").exists(),
            "requires_api": False,
            "description": "Generate procedural companies and their base metadata.",
        },
        {
            "id": 30,
            "name": "Generate Keywords (Procedural)",
            "script": "generate_keywords_procedural.py",
            "args": ["--target", str(args.n_keywords), "--seed", str(args.seed)],
            "check": lambda bd: (bd / "entities" / "keywords.json").exists(),
            "requires_api": False,
            "description": "Generate the keyword pool procedurally.",
        },
        {
            "id": 40,
            "name": "Enrich Persons (LLM)",
            "script": "enrich_persons_api.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto"] + (["--model", args.model] if args.model else []),
            "check": _check_persons_enriched,
            "requires_api": True,
            "description": "LLM-enrich procedural persons with bios, styles, and genre affinities.",
        },
        {
            "id": 50,
            "name": "Generate Character Bank",
            "script": "generate_character_bank.py",
            "args": ["--target", str(args.n_characters), "--seed", str(args.seed)],
            "check": lambda bd: (bd / "entities" / "character_bank.csv").exists(),
            "requires_api": False,
            "description": "Generate the reusable character-name bank procedurally.",
        },
        {
            "id": 60,
            "name": "Generate Title Bank",
            "script": "topup_title_bank.py",
            "args": (
                ["--base-dir", str(BASE_DIR), "--target-count", str(args.n_titles), "--seed", str(args.seed)]
                + (["--start-year", str(args.start_year), "--end-year", str(args.end_year)] if args.start_year is not None and args.end_year is not None else [])
            ),
            "check": lambda bd: (bd / "entities" / "title_bank.csv").exists(),
            "requires_api": False,
            "description": "Generate titles and distribute them across the requested year span.",
        },
        {
            "id": 70,
            "name": "Generate Latent Variables",
            "script": "generate_latent_vars_api.py",
            "args": ["--auto"] + (["--model", args.model] if args.model else []),
            "check": _check_latent,
            "requires_api": True,
            "description": "LLM-generate latent variables for persons and companies.",
        },
        {
            "id": 80,
            "name": "Generate Edge Graph",
            "script": "generate_edges_hybrid.py",
            "args": ["--base-dir", str(BASE_DIR)],
            "check": _check_edges,
            "requires_api": False,
            "description": "Build the initial graph runtime from entities and latents.",
        },
        {
            "id": 90,
            "name": "Convert Entities to CSV",
            "script": "entities_to_csv.py",
            "args": [str(BASE_DIR), "convert"],
            "check": _check_entities_csv,
            "requires_api": False,
            "description": "Convert JSON entities into the CSV/Arrow inputs used by assembly.",
        },
        {
            "id": 92,
            "name": "Generate Company Financial Profiles",
            "script": "generate_company_financial_profiles.py",
            "args": ["--base-dir", str(BASE_DIR)],
            "check": _check_company_financial_profiles,
            "requires_api": False,
            "description": "Generate baseline company finance profiles used during movie generation.",
        },
        {
            "id": 95,
            "name": "Enrich Keyword Genres",
            "script": "generate_keyword_genres.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto"] + (["--model", args.model] if args.model else []),
            "check": _check_keyword_genres,
            "requires_api": True,
            "description": "Fill any missing keyword genre metadata.",
        },
        {
            "id": 100,
            "name": "Generate Movies",
            "script": "generate_movies.py",
            "args": (
                ["--base_dir", str(BASE_DIR), "--n_movies", str(args.n_movies)]
                + (["--enable_llm_evolution"] if args.enable_llm_evolution else [])
                + (["--enable_llm_critic"] if args.enable_llm_critic else [])
                + (["--llm_model", args.model] if args.model else [])
            ),
            "check": _check_movies,
            "requires_api": False,
            "description": "Run full movie assembly and export all primary tables.",
        },
        {
            "id": 110,
            "name": "Generate Plot Summaries",
            "script": "generate_plot_summaries_api.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto"] + (["--model", args.model] if args.model else []),
            "check": _check_plots,
            "requires_api": True,
            "description": "Generate plot summaries for the assembled movie table.",
        },
        {
            "id": 120,
            "name": "Generate TV Summaries",
            "script": "generate_tv_summaries.py",
            "args": ["--base-dir", str(BASE_DIR)] + (["--tier1-model", args.model, "--tier2-model", args.model] if args.model else []),
            "check": _check_tv_summaries,
            "requires_api": True,
            "description": "Generate TV series and episode summaries for the TV tables.",
        },
    ]


def _run_step(step: dict, args: argparse.Namespace) -> bool:
    script_path = BASE_DIR / step["script"]
    if not script_path.exists():
        print(f"  ERROR: missing script: {script_path.name}")
        return False

    cmd = [sys.executable, str(script_path)] + list(step["args"])
    print(f"\n{'=' * 72}")
    print(f"STEP {step['id']}: {step['name']}")
    print(step["description"])
    print("CMD:", " ".join(cmd))
    print(f"{'=' * 72}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {args.timeout}s")
        return False
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"  OK ({elapsed:.1f}s)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="test25 end-to-end pipeline runner")
    parser.add_argument("--n-movies", type=int, required=True, dest="n_movies", help="Number of movies to generate")
    parser.add_argument("--start-year", type=int, default=None, help="Start year for title-bank distribution")
    parser.add_argument("--end-year", type=int, default=None, help="End year for title-bank distribution")
    parser.add_argument("--n-persons", type=int, default=None, dest="n_persons")
    parser.add_argument("--n-companies", type=int, default=None, dest="n_companies")
    parser.add_argument("--n-keywords", type=int, default=None, dest="n_keywords")
    parser.add_argument("--n-characters", type=int, default=None, dest="n_characters")
    parser.add_argument("--n-titles", type=int, default=None, dest="n_titles")
    parser.add_argument("--model", default=None, help="Override model for LLM-backed steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-llm-evolution", action="store_true")
    parser.add_argument("--enable-llm-critic", action="store_true")
    parser.add_argument("--from-step", type=int, default=10)
    parser.add_argument("--until-step", type=int, default=None)
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Delete generated outputs before running")
    parser.add_argument("--clear-checkpoint", action="store_true")
    parser.add_argument("--timeout", type=int, default=None, help="Per-step timeout in seconds")
    args = parser.parse_args()

    _resolve_targets(args)

    print("=" * 72)
    print("TEST25 PIPELINE RUNNER")
    print("=" * 72)
    print(f"Base dir:      {BASE_DIR}")
    print(f"Movies:        {args.n_movies}")
    print(f"Year span:     {args.start_year if args.start_year is not None else '(default)'}"
          f"{' -> ' + str(args.end_year) if args.end_year is not None else ''}")
    print(f"Persons:       {args.n_persons}")
    print(f"Companies:     {args.n_companies}")
    print(f"Keywords:      {args.n_keywords}")
    print(f"Characters:    {args.n_characters}")
    print(f"Titles:        {args.n_titles}")
    print(f"Model:         {args.model or '(provider default)'}")
    print(f"LLM evolution: {args.enable_llm_evolution}")
    print(f"LLM critic:    {args.enable_llm_critic}")
    print("=" * 72)

    if args.fresh:
        print("Resetting generated outputs...")
        _reset_generated_outputs()
    elif args.clear_checkpoint:
        _clear_checkpoint()

    state = _load_checkpoint()
    completed = set(state.get("completed_steps", []))
    if state.get("started_at") is None:
        state["started_at"] = datetime.now().isoformat()

    steps = _build_steps(args)
    if args.only is not None:
        steps = [step for step in steps if step["id"] == args.only]
    else:
        steps = [step for step in steps if step["id"] >= args.from_step]
    if args.until_step is not None:
        steps = [step for step in steps if step["id"] <= args.until_step]

    succeeded = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for step in steps:
        sid = step["id"]
        if sid in completed and not args.force:
            print(f"\nStep {sid} ({step['name']}): checkpoint says complete, skipping")
            skipped += 1
            continue

        if not args.force and step["check"](BASE_DIR):
            print(f"\nStep {sid} ({step['name']}): outputs already present, skipping")
            completed.add(sid)
            state["completed_steps"] = sorted(completed)
            _save_checkpoint(state)
            skipped += 1
            continue

        ok = _run_step(step, args)
        if not ok:
            failed += 1
            print(f"\nPipeline halted at step {sid}. Resume with --from-step {sid}.")
            break

        completed.add(sid)
        state["completed_steps"] = sorted(completed)
        _save_checkpoint(state)
        succeeded += 1

    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 72}")
    print(f"Succeeded: {succeeded}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Time:      {elapsed:.1f}s")
    print(f"{'=' * 72}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
