# -*- coding: utf-8 -*-
"""
V17 Overnight Runner
====================
Chains: backfill_avoid_genres -> generate_movies -> regen_secondary

Usage:
    python v17_overnight.py              # full run
    python v17_overnight.py --skip-backfill  # if backfill already done
"""
import subprocess, sys, time, os
from pathlib import Path

BASE = Path(__file__).parent

def run_step(name, cmd, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE), **kwargs)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        sys.exit(1)
    print(f"\n  OK ({elapsed:.0f}s)", flush=True)
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V17 overnight pipeline")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Skip avoid_genres backfill (if already done)")
    parser.add_argument("--n-movies", type=int, default=20000,
                        help="Number of movies (default: 20000)")
    parser.add_argument("--enable-llm-evolution", action="store_true", default=True,
                        help="Enable LLM temporal evolution (default: True)")
    parser.add_argument("--skip-secondary", action="store_true",
                        help="Skip secondary table regeneration")
    args = parser.parse_args()

    print("=" * 60)
    print("  V17 OVERNIGHT RUNNER")
    print("=" * 60)
    print(f"  Movies:        {args.n_movies}")
    print(f"  LLM evolution: {args.enable_llm_evolution}")
    print(f"  Skip backfill: {args.skip_backfill}")
    print(f"  Skip secondary: {args.skip_secondary}")
    print("=" * 60)

    t_total = time.time()

    # Step 1: Backfill avoid_genres
    if not args.skip_backfill:
        run_step(
            "Step 1/4: Backfill avoid_genres",
            [sys.executable, str(BASE / "backfill_avoid_genres.py"), "--auto"]
        )
    else:
        print("\n  [SKIP] Backfill avoid_genres")

    # Step 2: Graph stitching (rebuild edge_graph.csv with fixed mappings)
    # This is done inside generate_movies.py via WorldState.load()
    # which calls edge_graph.build_affinity_index() -- so no separate step needed.

    # Step 3: Generate movies
    movie_cmd = [
        sys.executable, str(BASE / "generate_movies.py"),
        "--n_movies", str(args.n_movies),
    ]
    if args.enable_llm_evolution:
        movie_cmd.append("--enable_llm_evolution")
    run_step("Step 2/4: Generate Movies", movie_cmd)

    # Step 4: Regenerate secondary tables
    if not args.skip_secondary:
        run_step(
            "Step 3/4: Regenerate Secondary Tables",
            [sys.executable, str(BASE / "regen_secondary.py")]
        )
    else:
        print("\n  [SKIP] Secondary tables")

    # Step 5: Generate plot summaries (optional, uses LLM)
    # Skipped by default in overnight -- can be run separately
    print("\n  [SKIP] Plot summaries (run separately: python generate_plot_summaries_api.py --auto)")

    elapsed_total = time.time() - t_total
    hours = elapsed_total / 3600
    print(f"\n{'='*60}")
    print(f"  V17 OVERNIGHT COMPLETE")
    print(f"  Total time: {elapsed_total:.0f}s ({hours:.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
