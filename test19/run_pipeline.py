"""V13 Pipeline Orchestrator
============================

Runs the full data generation pipeline in correct order with:
  - Step-level checkpointing (skip completed steps)
  - Dry-run mode (mock entities, no LLM calls)
  - Model tier config from contracts.py

Usage:
    python test11/run_pipeline.py                  # full pipeline
    python test11/run_pipeline.py --dry-run         # mock entities, no API
    python test11/run_pipeline.py --from-step 4     # resume from step 4
    python test11/run_pipeline.py --only 6          # run only step 6
"""

import os
import sys
import json
import time
import argparse
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PIPELINE STEP REGISTRY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Each step: (id, name, script, args_func, check_func, requires_api)
#   - args_func(args) -> list of CLI args to pass
#   - check_func(base_dir) -> bool (True = output exists, skip)

def _check_entities(base_dir: Path) -> bool:
    """Check if entity JSON files exist."""
    return (base_dir / "entities" / "persons.json").exists() and \
           (base_dir / "entities" / "companies.json").exists()

def _check_latent(base_dir: Path) -> bool:
    """Check if latent variable files exist."""
    return (base_dir / "entities" / "persons_latent.json").exists() and \
           (base_dir / "entities" / "companies_latent.json").exists()

def _check_edges(base_dir: Path) -> bool:
    """Check if edge graph exists."""
    return (base_dir / "graph" / "edge_graph.csv").exists() or (base_dir / "graph" / "edges.json").exists()

def _check_extras(base_dir: Path) -> bool:
    """Check if extras have been generated (or if step should be skipped entirely).

    V15: person pool is expanded with generate_persons_llm.py (LLM-generated).
    If generate_extras.py doesn't exist, this step is not applicable -- skip it.
    """
    # If the extras script doesn't exist, step 2 is a no-op for this version
    if not (base_dir / "generate_extras.py").exists():
        return True  # skip

    ent = base_dir / "entities" / "persons.json"
    if not ent.exists():
        return False
    try:
        with open(ent, encoding="utf-8") as f:
            persons = json.load(f)
        return len(persons) > 500  # any meaningful pool means extras already ran
    except Exception:
        return False

def _check_movies(base_dir: Path) -> bool:
    """Check if movie CSVs exist."""
    return (base_dir / "movie.csv").exists() and \
           (base_dir / "movies_flat.csv").exists()

def _check_characters(base_dir: Path) -> bool:
    """Check if character bank exists."""
    return (base_dir / "entities" / "character_bank.csv").exists()

def _check_plots(base_dir: Path) -> bool:
    """Check if plot summaries have been added."""
    movie_csv = base_dir / "movie.csv"
    if not movie_csv.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(movie_csv, nrows=5)
        if "plot_summary" in df.columns:
            return df["plot_summary"].notna().any()
    except Exception:
        pass
    return False


PIPELINE_STEPS = [
    {
        "id": 1,
        "name": "Normalize Entities",
        "script": "normalize_entities.py",
        "args": lambda a: ["--auto"] if a.auto else [],
        "check": lambda bd: (bd / "entities" / "persons.json").exists(),
        "requires_api": False,
        "description": "Normalize raw entity JSON into clean persons.json / companies.json",
    },
    {
        "id": 2,
        "name": "Generate Extras",
        "script": "generate_extras.py",
        "args": lambda a: [],
        "check": _check_extras,
        "requires_api": False,
        "description": "Add procedural extra actors, directors, crew to reach ENTITY_COUNTS targets",
    },
    {
        "id": 3,
        "name": "Generate Character Descriptions",
        "script": "generate_character_descriptions.py",
        "args": lambda a: [],
        "check": _check_characters,
        "requires_api": False,
        "description": "Generate character name bank for cast assignment",
    },
    {
        "id": 4,
        "name": "Generate Latent Variables",
        "script": "generate_latent_vars_api.py",
        "args": lambda a: ["--auto"] + (["--model", a.model] if a.model else []),
        "check": _check_latent,
        "requires_api": True,
        "description": "LLM assigns numeric latent variables to persons and companies",
    },
    {
        "id": 5,
        "name": "Generate Edge Graph",
        "script": "generate_edges_hybrid.py",
        "args": lambda a: [],
        "check": _check_edges,
        "requires_api": False,
        "description": "Build social graph (friendships, mentorships, rivalries) from latent variables",
    },
    {
        "id": 6,
        "name": "Convert Entities to CSV",
        "script": "entities_to_csv.py",
        "args": lambda a: [],
        "check": lambda bd: (bd / "persons.csv").exists(),
        "requires_api": False,
        "description": "Convert JSON entities to CSV format for movie generation",
    },
    {
        "id": 65,
        "name": "Enrich Keyword Genres",
        "script": "generate_keyword_genres.py",
        "args": lambda a: ["--auto"] + (["--model", a.model] if a.model else []),
        "check": lambda bd: (
            (bd / "entities" / "keyword.csv").exists() and
            not __import__("pandas").read_csv(bd / "entities" / "keyword.csv")["topic_genre"].isna().any()
        ),
        "requires_api": True,
        "description": "LLM assigns topic_genre and pop_weight to keywords missing genre associations",
    },
    {
        "id": 7,
        "name": "Generate Movies",
        "script": "generate_movies.py",
        "args": lambda a: (
            (["--n_movies", str(a.n_movies)] if a.n_movies is not None else [])
          + (["--enable_llm_evolution"] if getattr(a, "enable_llm_evolution", False) else [])
          + (["--disable_llm_evolution"] if a.dry_run else [])
          + (["--enable_llm_critic"] if getattr(a, "enable_llm_critic", False) else [])
          + (["--llm_model", a.model] if a.model else [])
        ),
        "check": _check_movies,
        "requires_api": False,  # LLM evolution is optional within this step
        "description": "Assemble movies with temporal evolution, crew, financials, reviews, awards",
    },
    {
        "id": 8,
        "name": "Generate Plot Summaries",
        "script": "generate_plot_summaries_api.py",
        "args": lambda a: ["--auto"] + (["--model", a.model] if a.model else []),
        "check": _check_plots,
        "requires_api": True,
        "description": "LLM generates 2-3 sentence plot summaries for each movie",
    },
    {
        "id": 9,
        "name": "Generate TV Summaries",
        "script": "generate_tv_summaries.py",
        "args": lambda a: [] + (["--tier1-model", a.model, "--tier2-model", a.model] if a.model else []),
        "check": lambda bd: (
            (bd / "tv_series.csv").exists() and
            __import__("pandas").read_csv(bd / "tv_series.csv", nrows=3).get("plot_summary", __import__("pandas").Series()).notna().any()
        ) if (bd / "tv_series.csv").exists() else False,
        "requires_api": True,
        "description": "LLM generates TV series overviews and episode descriptions",
    },
]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DRY-RUN SUPPORT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def setup_dry_run(base_dir: Path):
    """Create mock entity files for dry-run testing (no LLM needed)."""
    # Import valid vocabularies to keep mock data in sync
    try:
        sys.path.insert(0, str(base_dir))
        from contracts import (CAREER_STAGES, MARKETS, COMPANY_TIERS,
                               GENRES, STYLE_TAGS, NATIONALITIES)
    except ImportError:
        CAREER_STAGES = ["rising", "prime", "veteran", "legend", "retired"]
        MARKETS = ["Local", "Regional", "Europe", "Asia", "North America", "Global"]
        COMPANY_TIERS = ["Global", "Major", "Mid-Budget", "Indie", "Micro"]
        GENRES = ["Drama", "Action", "Comedy", "Sci-Fi", "Horror", "Thriller"]
        STYLE_TAGS = ["physical", "comedic", "cerebral", "stoic", "intense"]
        NATIONALITIES = ["American", "British", "French", "Japanese", "Indian"]

    ent_dir = base_dir / "entities"
    graph_dir = base_dir / "graph"
    os.makedirs(ent_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    import numpy as np
    rng_py = __import__('random').Random(42)

    # Mock persons (small set)
    if not (ent_dir / "persons.json").exists():
        mock_persons = []
        roles_pool = [["actor"], ["actor", "director"], ["actor"], ["director"], ["actor"]]
        for i in range(1, 101):
            mock_persons.append({
                "person_id": i,
                "name": f"Person_{i:03d}",
                "nationality": NATIONALITIES[i % len(NATIONALITIES)],
                "gender": "M" if i % 2 == 0 else "F",
                "bio": f"A talented performer known for diverse roles. Person {i}.",
                "style_tags": [STYLE_TAGS[j % len(STYLE_TAGS)] for j in range(i, i + 2)],
                "genre_affinity": [GENRES[(i - 1) % len(GENRES)], GENRES[i % len(GENRES)]],
                "career_stage": CAREER_STAGES[i % len(CAREER_STAGES)],
                "roles": roles_pool[(i - 1) % len(roles_pool)],
                "market_fit": [MARKETS[i % len(MARKETS)]],
            })
        with open(ent_dir / "persons.json", "w", encoding="utf-8") as f:
            json.dump(mock_persons, f, indent=2)
        print(f"  Created {len(mock_persons)} mock persons")

    # Mock companies
    if not (ent_dir / "companies.json").exists():
        mock_companies = []
        for i in range(1, 51):
            mock_companies.append({
                "company_id": i,
                "name": f"Studio_{i:02d}",
                "description": f"A production company specializing in {GENRES[(i-1) % len(GENRES)]} films.",
                "tier": COMPANY_TIERS[(i - 1) % len(COMPANY_TIERS)],
                "specialty_genres": [GENRES[(i-1) % len(GENRES)]],
                "country": "USA" if i % 2 == 0 else "UK",
                "preferred_actor_styles": [STYLE_TAGS[i % len(STYLE_TAGS)]],
                "preferred_director_styles": ["slow-burn"],
            })
        with open(ent_dir / "companies.json", "w", encoding="utf-8") as f:
            json.dump(mock_companies, f, indent=2)
        print(f"  Created {len(mock_companies)} mock companies")

    # Mock latent variables (skip LLM step)
    rng = np.random.RandomState(42)

    if not (ent_dir / "persons_latent.json").exists():
        persons = json.loads((ent_dir / "persons.json").read_text(encoding="utf-8"))
        person_latent = []
        for p in persons:
            person_latent.append({
                "person_id": p["person_id"],
                "creative_style_vector": [round(float(x), 3) for x in rng.uniform(-1, 1, 8)],
                "risk_tolerance": round(float(rng.uniform(0.1, 0.9)), 3),
                "collaboration_style": str(rng.choice(["solo", "ensemble", "chameleon", "mentorship"])),
                "controversy_score": round(float(rng.uniform(0.0, 0.4)), 3),
                "public_reputation": round(float(rng.uniform(0.1, 0.9)), 3),
                "budget_band_pref": [round(float(x), 3) for x in rng.dirichlet([1]*5)],
                "artistic_ambition": round(float(rng.uniform(0.2, 0.8)), 3),
                "volatility": round(float(rng.uniform(0.1, 0.6)), 3),
            })
        with open(ent_dir / "persons_latent.json", "w", encoding="utf-8") as f:
            json.dump(person_latent, f, indent=2)
        print(f"  Created {len(person_latent)} mock person latent vars")

    if not (ent_dir / "companies_latent.json").exists():
        companies = json.loads((ent_dir / "companies.json").read_text(encoding="utf-8"))
        company_latent = []
        for c in companies:
            company_latent.append({
                "company_id": c["company_id"],
                "risk_appetite": round(float(rng.uniform(0.2, 0.8)), 3),
                "prestige_score": round(float(rng.uniform(0.1, 0.9)), 3),
                "genre_portfolio": [round(float(x), 3) for x in rng.dirichlet([1]*12)],
                "budget_tier_focus": [round(float(x), 3) for x in rng.dirichlet([1]*5)],
                "market_trend_sensitivity": round(float(rng.uniform(0.2, 0.8)), 3),
                "controversy_tolerance": round(float(rng.uniform(0.2, 0.7)), 3),
            })
        with open(ent_dir / "companies_latent.json", "w", encoding="utf-8") as f:
            json.dump(company_latent, f, indent=2)
        print(f"  Created {len(company_latent)} mock company latent vars")

    # Mock edges
    if not (graph_dir / "edges.json").exists():
        persons = json.loads((ent_dir / "persons.json").read_text(encoding="utf-8"))
        edges = []
        pids = [p["person_id"] for p in persons]
        n_edges = min(200, len(pids) * 2)
        for i in range(n_edges):
            a = int(rng.choice(pids))
            b = int(rng.choice(pids))
            if a == b:
                continue
            a, b = min(a, b), max(a, b)
            etype = str(rng.choice(["friendship", "mentorship", "rivalry"]))
            edges.append({
                "src_id": a, "dst_id": b,
                "src_type": "person", "dst_type": "person",
                "edge_type": etype,
                "sign": "-" if etype == "rivalry" else "+",
                "weight": round(float(rng.uniform(0.2, 0.9)), 3),
                "valid_from": int(rng.randint(2010, 2020)),
                "valid_to": None,
            })
        with open(graph_dir / "edges.json", "w", encoding="utf-8") as f:
            json.dump(edges, f, indent=2)
        print(f"  Created {len(edges)} mock edges")

    # Mock keywords
    if not (ent_dir / "keywords.json").exists():
        keywords = [{"keyword_id": i, "keyword": f"keyword_{i}"}
                     for i in range(1, 201)]
        with open(ent_dir / "keywords.json", "w", encoding="utf-8") as f:
            json.dump(keywords, f, indent=2)
        print(f"  Created {len(keywords)} mock keywords")

    print("  Dry-run setup complete")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHECKPOINT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

CHECKPOINT_FILE = BASE_DIR / ".pipeline_checkpoint.json"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed_steps": [], "started_at": None}


def save_checkpoint(state: dict):
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def clear_checkpoint():
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# RUNNER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_step(step: dict, args: argparse.Namespace) -> bool:
    """Run a single pipeline step. Returns True on success."""
    script_path = BASE_DIR / step["script"]
    if not script_path.exists():
        print(f"  [SKIP] Missing script: {script_path.name}")
        return True
    script = str(script_path)
    step_args = step["args"](args)

    cmd = [sys.executable, script] + step_args
    cmd_str = " ".join(cmd)

    print(f"\n{'='*60}")
    print(f"  STEP {step['id']}: {step['name']}")
    print(f"  {step['description']}")
    print(f"  CMD: {cmd_str}")
    print(f"{'='*60}\n")

    if args.dry_run and step["requires_api"]:
        print(f"  SKIP (dry-run mode, requires API)")
        return True

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            timeout=args.timeout,
            check=False,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
            return False
        print(f"\n  OK ({elapsed:.1f}s)")
        return True
    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT after {args.timeout}s")
        return False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test 11 Pipeline Orchestrator -- run the full data generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. Normalize Entities      (normalize raw JSON)
  2. Generate Extras          (procedural expansion)
  3. Generate Characters      (character name bank)
  4. Generate Latent Vars     (LLM: latent variables)
  5. Generate Edge Graph      (build social graph)
  6. Entities to CSV          (JSON -> CSV conversion)
  7. Generate Movies          (core assembly + temporal evolution)
  8. Generate Plot Summaries  (LLM: plot text)
""",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock entities, skip LLM API steps")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Resume from this step number (default: 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this step number")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if outputs exist")
    parser.add_argument("--n-movies", type=int, default=None, dest="n_movies",
                        help="Number of movies to generate (default: 50 for dry-run, 7500 for full)")
    parser.add_argument("--model", default=None,
                        help="Override Gemini model for all LLM steps")
    parser.add_argument("--enable-llm-evolution", action="store_true",
                        help="Enable LLM temporal evolution during step 7 (OFF by default in v16)")
    parser.add_argument("--enable-llm-critic", action="store_true",
                        help="Enable LLM post-generation critic/repair pass in step 7")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout per step in seconds (default: no timeout)")
    parser.add_argument("--auto", action="store_true", default=True,
                        help="Skip interactive prompts (default: True)")
    parser.add_argument("--clear-checkpoint", action="store_true",
                        help="Clear checkpoint and start fresh")
    args = parser.parse_args()

    # R13-FIX: Use None sentinel so explicit --n-movies 50 is never silently scaled
    # to 7500. Old code would scale any value == 50 on a non-dry-run, which meant
    # passing --n-movies 50 explicitly behaved identically to passing nothing.
    # V19: non-dry-run default is None → generate_movies.py auto-detects from title bank.
    if args.n_movies is None:
        args.n_movies = 50 if args.dry_run else None


    print("=" * 60)
    print("  TEST 11 PIPELINE ORCHESTRATOR")
    print("=" * 60)
    print(f"  Mode:     {'DRY-RUN' if args.dry_run else 'FULL'}")
    print(f"  Movies:   {args.n_movies}")
    print(f"  Model:    {args.model or '(tier defaults)'}")
    print(f"  Base dir: {BASE_DIR}")
    print(f"  From:     step {args.from_step}")
    print("=" * 60)

    if args.clear_checkpoint:
        clear_checkpoint()
        print("  Checkpoint cleared")

    # Dry-run: create mock entities
    if args.dry_run:
        print("\n  Setting up dry-run mock data...")
        setup_dry_run(BASE_DIR)

    # Load checkpoint
    state = load_checkpoint()
    completed = set(state.get("completed_steps", []))
    if state.get("started_at") is None:
        state["started_at"] = datetime.now().isoformat()

    # Determine which steps to run
    if args.only is not None:
        steps_to_run = [s for s in PIPELINE_STEPS if s["id"] == args.only]
    else:
        steps_to_run = [s for s in PIPELINE_STEPS if s["id"] >= args.from_step]

    # Dry-run: skip expensive steps that aren't needed for mock data
    if args.dry_run:
        # Skip: normalize (1) -- mock data is already clean
        # Skip: generate_extras (2) -- would expand 100 persons to 7K+ (overkill for dry-run)
        # Skip: generate_edges_hybrid (5) -- O(nÂ²) pairwise similarity on 7K persons
        # We already provide mock edges in setup_dry_run()
        steps_to_run = [s for s in steps_to_run if s["id"] not in (1, 2, 5)]

    total = len(steps_to_run)
    succeeded = 0
    failed = 0
    skipped = 0
    t_start = time.time()

    for step in steps_to_run:
        sid = step["id"]

        # Check if already completed (checkpoint)
        if sid in completed and not args.force:
            print(f"\n  Step {sid} ({step['name']}): CHECKPOINT -- already completed")
            skipped += 1
            continue

        # Check if outputs already exist
        if not args.force and step["check"](BASE_DIR):
            print(f"\n  Step {sid} ({step['name']}): SKIP -- outputs already exist")
            completed.add(sid)
            state["completed_steps"] = sorted(completed)
            save_checkpoint(state)
            skipped += 1
            continue

        # Run the step
        ok = run_step(step, args)
        if ok:
            succeeded += 1
            completed.add(sid)
            state["completed_steps"] = sorted(completed)
            save_checkpoint(state)
        else:
            failed += 1
            if not args.force:
                print(f"\n  Pipeline halted at step {sid}. Use --from-step {sid} to resume.")
                break

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()







