"""
overnight.py -- V15 Overnight Orchestration Script
===================================================
Kills any running latent job, then runs all remaining steps in sequence:

  Step 1: persons latent vars  (generate_latent_vars_api.py --persons-only)
  Step 2: agencies generation  (generate_agencies.py)     [expand + assign]
  Step 3: main pipeline        (run_pipeline.py --auto)

Designed to be left running overnight. All steps are resumable -- if anything
fails mid-way, just re-run overnight.py and it picks up where it left off.

Usage:
    python test15/overnight.py
    python test15/overnight.py --skip-latent   # if latent already done
    python test15/overnight.py --skip-pipeline # run only entity prep steps
"""
import os, sys, subprocess, time, json, argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent

def run(cmd, desc):
    """Run a command, stream output, raise on failure."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n", flush=True)
    # Inject -u after the python executable so child output is unbuffered too.
    # Without this, Python buffers child stdout in 8KB chunks when piped through tee.
    exe = cmd[0]
    rest = cmd[1:]
    full_cmd = [exe, "-u"] + list(rest)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(full_cmd, cwd=BASE_DIR, env=env)
    if result.returncode != 0:
        print(f"\n!! STEP FAILED (exit {result.returncode}): {desc}")
        print("   Overnight script stopping. Re-run overnight.py to resume.")
        sys.exit(result.returncode)
    print(f"\n[OK] {desc}\n", flush=True)



def check_persons_latent_done():
    latent_path = BASE_DIR / "entities" / "persons_latent.json"
    persons_path = BASE_DIR / "entities" / "persons.json"
    if not latent_path.exists():
        return False, 0, 0
    latent = json.loads(latent_path.read_text(encoding="utf-8"))
    persons = json.loads(persons_path.read_text(encoding="utf-8"))
    return len(latent) >= len(persons) * 0.98, len(latent), len(persons)


def main():
    parser = argparse.ArgumentParser(description="V15 overnight orchestration")
    parser.add_argument("--skip-latent",   action="store_true",
                        help="Skip persons latent step (if already done)")
    parser.add_argument("--skip-agencies", action="store_true",
                        help="Skip agencies step")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip pipeline (only run entity prep)")
    args = parser.parse_args()

    py = sys.executable
    print("\n" + "="*60)
    print("  V15 OVERNIGHT ORCHESTRATION")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ── Step 1: Persons latent vars ───────────────────────────────────────────
    if not args.skip_latent:
        done, have, need = check_persons_latent_done()
        if done:
            print(f"\n[SKIP] Persons latent already complete ({have}/{need})")
        else:
            print(f"\n  Persons latent: {have}/{need} done, running for the rest...")
            run(
                [py, "generate_latent_vars_api.py",
                 "--persons-only", "--auto", "--batch-size", "30"],
                "Step 1/3 -- Persons latent vars (batch=30, sleep=0.5s)",
            )
    else:
        print("\n[SKIP] Persons latent (--skip-latent)")

    # ── Step 2: Agencies ──────────────────────────────────────────────────────
    if not args.skip_agencies:
        agencies_path = BASE_DIR / "entities" / "agencies.json"
        agencies = json.loads(agencies_path.read_text(encoding="utf-8")) if agencies_path.exists() else []
        persons_path = BASE_DIR / "entities" / "persons.json"
        persons = json.loads(persons_path.read_text(encoding="utf-8"))
        with_agency = sum(1 for p in persons if p.get("agency"))
        if len(agencies) >= 50 and with_agency >= len(persons) * 0.95:
            print(f"\n[SKIP] Agencies already expanded ({len(agencies)}) and assigned ({with_agency}/{len(persons)})")
        else:
            run(
                [py, "generate_agencies.py", "--auto"],
                "Step 2/3 -- Agencies expansion + person assignment",
            )
    else:
        print("\n[SKIP] Agencies (--skip-agencies)")

    # ── Step 3: Main pipeline ─────────────────────────────────────────────────
    if not args.skip_pipeline:
        run(
            [py, "run_pipeline.py", "--auto"],
            "Step 3/3 -- Main pipeline (movies, edges, assembly)",
        )
    else:
        print("\n[SKIP] Pipeline (--skip-pipeline)")

    print("\n" + "="*60)
    print(f"  OVERNIGHT COMPLETE -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
