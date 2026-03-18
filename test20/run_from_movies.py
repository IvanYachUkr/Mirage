"""
V14 Master Runner -- from movies generation to the very end.
=============================================================
Runs steps 7-9 in correct order:
  7. generate_movies.py        -- Assembles movies with temporal evolution + LLM
  8. generate_plot_summaries_api.py -- LLM plot summaries for movies
  9. generate_tv_summaries.py  -- LLM TV series overviews + episode descriptions

Usage:
  python run_from_movies.py                     # run all steps 7-9
  python run_from_movies.py --n-movies 5151     # specify movie count
  python run_from_movies.py --skip-to 8         # start from step 8
  python run_from_movies.py --clean             # wipe CSVs/checkpoints first
"""
import subprocess
import sys
import os
import time
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
PYTHON = sys.executable

STEPS = [
    {
        "id": 7,
        "name": "Generate Movies",
        "cmd": lambda args: [
            PYTHON, str(BASE_DIR / "generate_movies.py"),
            "--n_movies", str(args.get("n_movies", 5151)),
        ],
        "check": lambda: (BASE_DIR / "movie.csv").exists(),
    },
    {
        "id": 8,
        "name": "Generate Plot Summaries",
        "cmd": lambda args: [
            PYTHON, str(BASE_DIR / "generate_plot_summaries_api.py"),
            "--auto",
        ],
        "check": lambda: _has_column_filled("movie.csv", "plot_summary"),
    },
    {
        "id": 9,
        "name": "Generate TV Summaries",
        "cmd": lambda args: [
            PYTHON, str(BASE_DIR / "generate_tv_summaries.py"),
        ],
        "check": lambda: _has_column_filled("tv_series.csv", "plot_summary"),
    },
]


def _has_column_filled(csv_name: str, col: str) -> bool:
    """Check if a CSV exists and has at least one non-null value in a column."""
    p = BASE_DIR / csv_name
    if not p.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(p, nrows=5)
        return col in df.columns and df[col].notna().any()
    except Exception:
        return False


def clean_output():
    """Remove CSVs, checkpoints, snapshots before a fresh run."""
    print("Cleaning output...")
    for f in BASE_DIR.glob("*.csv"):
        if f.name not in ("persons.csv", "persons_enriched.csv", "companies.csv",
                          "title_bank.csv"):
            f.unlink()
            print(f"  Removed {f.name}")
    for d in ["checkpoints", "graph/temporal_patches"]:
        p = BASE_DIR / d
        if p.exists():
            shutil.rmtree(p)
            print(f"  Removed {d}/")
    cp = BASE_DIR / ".pipeline_checkpoint.json"
    if cp.exists():
        cp.unlink()
        print("  Removed .pipeline_checkpoint.json")
    print("Clean complete.\n")


def run_step(step: dict, args: dict) -> bool:
    """Run a single step. Returns True on success."""
    step_id = step["id"]
    name = step["name"]
    cmd = step["cmd"](args)

    print(f"\n{'='*60}")
    print(f"  STEP {step_id}: {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False
    else:
        print(f"\n  OK ({elapsed:.1f}s)")
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V14: Run steps 7-9 (movies -> summaries)")
    parser.add_argument("--n-movies", type=int, default=5151,
                        help="Number of movies to generate (default: 5151)")
    parser.add_argument("--skip-to", type=int, default=7,
                        help="Start from this step (default: 7)")
    parser.add_argument("--clean", action="store_true",
                        help="Clean output before running")
    parser.add_argument("--force", action="store_true",
                        help="Skip output checks, re-run even if output exists")
    cli_args = parser.parse_args()

    if cli_args.clean:
        clean_output()

    args = {"n_movies": cli_args.n_movies}

    print("="*60)
    print("  V14 PIPELINE -- STEPS 7-9")
    print("="*60)
    print(f"  Movies:    {args['n_movies']}")
    print(f"  Start at:  step {cli_args.skip_to}")
    print(f"  Force:     {cli_args.force}")
    print(f"  Base dir:  {BASE_DIR}")
    print("="*60)

    results = {}
    for step in STEPS:
        if step["id"] < cli_args.skip_to:
            continue

        # Check if already done
        if not cli_args.force and step["check"]():
            print(f"\n  Step {step['id']} ({step['name']}): output exists, skipping.")
            results[step["id"]] = "SKIPPED"
            continue

        ok = run_step(step, args)
        results[step["id"]] = "OK" if ok else "FAILED"

        if not ok:
            print(f"\n  Step {step['id']} FAILED. Stopping pipeline.")
            break

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print("="*60)
    for sid, status in results.items():
        name = next(s["name"] for s in STEPS if s["id"] == sid)
        print(f"  Step {sid} ({name}): {status}")
    print("="*60)

    # Exit with error if any step failed
    if any(v == "FAILED" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
