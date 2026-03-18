"""
refill_template_bios.py
------------------------
Re-runs LLM bio generation ONLY for persons whose bio was filled by the
fallback template ("...collaborative track record across modern productions.").

Usage:
    python test16/refill_template_bios.py
    python test16/refill_template_bios.py --model gemini-2.0-flash-lite-001
    python test16/refill_template_bios.py --batch-size 60 --timeout 70

Safe to run multiple times - only patches template-bio persons, leaves all
LLM-generated bios untouched.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
PERSONS_PATH = BASE_DIR / "entities" / "persons.json"

# ── marker used to detect template bios ───────────────────────────────────────
TEMPLATE_MARKER = "collaborative track record across modern productions"

# ── load vocabulary from contracts ────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from contracts import GENRES, STYLE_TAGS, MARKETS

# ── default model ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
PRICING = {"input": 0.075, "output": 0.30}   # $ per 1M tokens (lite-preview)


# ── API timeout helper (no-hang version) ──────────────────────────────────────
def call_with_timeout(client, model, contents, config, timeout):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(client.models.generate_content,
                         model=model, contents=contents, config=config)
    try:
        result = future.result(timeout=timeout)
        pool.shutdown(wait=False, cancel_futures=True)
        return result
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"API exceeded {timeout}s")


def call_with_retry(client, model, contents, config, timeout, max_retries=5):
    for attempt in range(max_retries):
        try:
            return call_with_timeout(client, model, contents, config, timeout)
        except TimeoutError:
            print(f"  TIMEOUT (retry {attempt+1}/{max_retries})")
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            is503 = "503" in str(e) or "unavailable" in str(e).lower()
            label = "503 API ERROR" if is503 else "API ERROR"
            print(f"  {label} (retry {attempt+1}/{max_retries}): {e}")
            time.sleep((60 if is503 else 5) * (attempt + 1))
    raise RuntimeError(f"All {max_retries} retries failed")


# ── loose JSON parsing ────────────────────────────────────────────────────────
def parse_json_loose(text: str):
    """Extract JSON from LLM response that may have markdown fences."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("[") or p.startswith("{"):
                text = p
                break
    try:
        return json.loads(text)
    except Exception:
        # Try to find the first [ ... ] block
        start = text.find("[")
        end   = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return []


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    load_dotenv(BASE_DIR.parent / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--timeout",    type=int, default=70)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Identify template bios but don't call API or save")
    args = parser.parse_args()

    # Update pricing if using flash (not lite)
    global PRICING
    if "flash-lite" not in args.model and "lite" not in args.model:
        PRICING = {"input": 0.50, "output": 3.00}

    # ── load persons ──────────────────────────────────────────────────────────
    print(f"Loading {PERSONS_PATH} ...")
    persons = json.loads(PERSONS_PATH.read_text(encoding="utf-8"))
    by_id   = {int(p.get("person_id", 0) or 0): p for p in persons}

    # ── find template-bio persons ─────────────────────────────────────────────
    todo = [p for p in persons if TEMPLATE_MARKER in str(p.get("bio", ""))]
    print(f"Total persons:         {len(persons)}")
    print(f"Template bio persons:  {len(todo)}")

    if not todo:
        print("Nothing to do — all bios already LLM-generated.")
        return

    if args.dry_run:
        print("Dry-run mode — not calling API.")
        for p in todo[:5]:
            print(f"  [{p.get('person_id')}] {p.get('name')}: {p.get('bio', '')[:80]}")
        return

    # ── API client ───────────────────────────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY / GOOGLE_API_KEY not set")
    client = genai.Client(api_key=api_key)

    # ── process in batches ───────────────────────────────────────────────────
    total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    tin = tout = cost = 0.0
    patched = skipped = 0

    print(f"\nModel:       {args.model}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Batches:     {total_batches}")
    print(f"Timeout:     {args.timeout}s\n")

    for b_idx, start in enumerate(range(0, len(todo), args.batch_size)):
        batch = todo[start : start + args.batch_size]
        print(f"  Batch {b_idx+1}/{total_batches}  ({len(batch)} persons)", flush=True)

        # Build same prompt as stage 05
        lines = []
        for x in batch:
            roles = x.get("roles", ["actor"])
            if isinstance(roles, str):
                roles = [z.strip() for z in roles.split(",") if z.strip()]
            lines.append(
                f"[{x.get('person_id')}] {x.get('name')} | "
                f"{x.get('nationality')} | {x.get('gender')} | "
                f"{','.join(roles)} | {x.get('career_stage', 'prime')}"
            )

        prompt = (
            f"Complete details for {len(batch)} movie-industry persons. "
            "Keep identity fields unchanged.\n"
            "Return JSON array of objects with: person_id, bio, style_tags, "
            "genre_affinity, market_fit.\n"
            f"style_tags from {STYLE_TAGS[:20]}\n"
            f"genre_affinity from {GENRES}\n"
            f"market_fit from {MARKETS}\n"
            "JSON only.\n\nPERSONS:\n" + "\n".join(lines)
        )

        config = {
            "temperature": 0.7,
            "max_output_tokens": len(batch) * 260,
            "thinking_config": {"thinking_budget": 0},
        }

        try:
            resp = call_with_retry(client, args.model, prompt, config,
                                   timeout=args.timeout)
        except RuntimeError as e:
            print(f"  FAILED batch {b_idx+1}: {e}  — keeping template bios for this batch")
            skipped += len(batch)
            continue

        # Track tokens / cost
        usage = resp.usage_metadata
        inp = getattr(usage, "prompt_token_count", 0) or 0
        out = getattr(usage, "candidates_token_count", 0) or 0
        tin  += inp
        tout += out
        batch_cost = (inp * PRICING["input"] + out * PRICING["output"]) / 1_000_000
        cost += batch_cost

        parsed = parse_json_loose(resp.text)
        if isinstance(parsed, dict):
            parsed = (parsed.get("persons") or parsed.get("data") or [])
        if not isinstance(parsed, list):
            parsed = []

        out_by_id = {}
        for z in parsed:
            if isinstance(z, dict):
                pid = int(z.get("person_id", 0) or 0)
                if pid > 0:
                    out_by_id[pid] = z

        for x in batch:
            pid = int(x.get("person_id", 0) or 0)
            z   = out_by_id.get(pid)
            if isinstance(z, dict):
                bio = str(z.get("bio", "")).strip()
                stx = z.get("style_tags", [])
                gx  = z.get("genre_affinity", [])
                mx  = z.get("market_fit", [])
                if isinstance(stx, str):
                    stx = [v.strip() for v in stx.split(",") if v.strip()]
                if isinstance(gx, str):
                    gx  = [v.strip() for v in gx.split(",")  if v.strip()]
                if isinstance(mx, str):
                    mx  = [v.strip() for v in mx.split(",")  if v.strip()]
                if len(bio) >= 40 and TEMPLATE_MARKER not in bio:
                    by_id[pid]["bio"]           = bio
                    by_id[pid]["style_tags"]    = [v for v in stx if v in STYLE_TAGS][:4] or x.get("style_tags", [])
                    by_id[pid]["genre_affinity"] = [v for v in gx  if v in GENRES][:3]     or x.get("genre_affinity", [])
                    by_id[pid]["market_fit"]    = [v for v in mx  if v in MARKETS][:2]     or x.get("market_fit", [])
                    patched += 1
                else:
                    skipped += 1  # LLM returned bad/template bio — keep for next run
            else:
                skipped += 1  # person not in response

        print(f"    {inp}in+{out}out=${batch_cost:.4f}  "
              f"patched={patched}  skipped={skipped}", flush=True)

        # Save after every batch (incremental)
        out_rows = sorted(by_id.values(), key=lambda z: int(z.get("person_id", 0) or 0))
        PERSONS_PATH.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TEMPLATE BIO REFILL COMPLETE")
    print("="*60)
    print(f"  Patched:  {patched}")
    print(f"  Skipped:  {skipped}  (will need another run if > 0)")
    print(f"  Tokens:   {int(tin)}in + {int(tout)}out")
    print(f"  Cost:     ${cost:.4f}")
    print(f"  Saved:    {PERSONS_PATH}")

    # Report remaining template bios
    persons_final = json.loads(PERSONS_PATH.read_text(encoding="utf-8"))
    remaining = sum(1 for p in persons_final if TEMPLATE_MARKER in str(p.get("bio", "")))
    print(f"  Remaining template bios: {remaining}")


if __name__ == "__main__":
    main()
