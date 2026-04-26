from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = max(0.0, min(1.0, float(pct))) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_ms(samples_sec: list[float]) -> dict[str, float | int | None]:
    if not samples_sec:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
            "first_ms": None,
        }
    samples_ms = [float(value) * 1000.0 for value in samples_sec]
    return {
        "count": int(len(samples_ms)),
        "mean_ms": round(statistics.fmean(samples_ms), 3),
        "p50_ms": round(percentile(samples_ms, 0.50) or 0.0, 3),
        "p95_ms": round(percentile(samples_ms, 0.95) or 0.0, 3),
        "max_ms": round(max(samples_ms), 3),
        "first_ms": round(samples_ms[0], 3),
    }


def cache_size(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "__len__"):
        try:
            return int(len(value))
        except Exception:
            return None
    return None


def snapshot_caches(world: Any) -> dict[str, int | None]:
    return {
        "cast_year_cache": cache_size(getattr(world, "_cast_year_cache", None)),
        "crew_year_pool_cache": cache_size(getattr(world, "_crew_year_pool_cache", None)),
        "company_by_tier_genre": cache_size(getattr(world, "_company_by_tier_genre", None)),
        "director_year_edge_cache": cache_size(getattr(world, "_director_year_edge_cache", None)),
        "director_pc_affinity_cache": cache_size(getattr(world, "_director_pc_affinity_cache", None)),
        "selection_year_state_cache": cache_size(getattr(world, "_selection_year_state_cache", None)),
        "title_bank_all_idx": cache_size(getattr(world, "_title_bank_all_idx", None)),
        "used_titles": cache_size(getattr(world, "used_titles", None)),
    }


def ensure_scratch_dirs(path: Path) -> None:
    for rel in ("graph/temporal_patches", "decision_logs", "graph"):
        (path / rel).mkdir(parents=True, exist_ok=True)


def plausible_financials(concept: dict[str, Any], rng: random.Random) -> tuple[float, float]:
    tier = str(concept.get("tier", "Mid"))
    base_budget = {
        "Epic": 180_000_000.0,
        "A": 70_000_000.0,
        "Mid": 22_000_000.0,
        "Indie": 4_500_000.0,
        "Micro": 900_000.0,
    }.get(tier, 22_000_000.0)
    budget = base_budget * rng.uniform(0.75, 1.35)
    box_office = budget * rng.uniform(0.60, 1.85)
    return float(budget), float(box_office)


def build_year_bucket(
    world: Any,
    *,
    from_year: int,
    count: int,
    id_offset: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    from assembly import pick_cast, pick_companies, pick_director, sample_movie_concept

    rows: list[dict[str, Any]] = []
    for idx in range(count):
        movie_id = int(id_offset + idx)
        concept = sample_movie_concept(world, movie_id=movie_id, forced_year=int(from_year))
        director_id = pick_director(world, concept)
        companies = pick_companies(world, concept, int(director_id) if director_id is not None else -1)
        cast_rows, competition_pairs = pick_cast(world, concept, int(director_id) if director_id is not None else -1, max_retries=1)
        budget_usd, box_office_usd = plausible_financials(concept, rng)
        rows.append(
            {
                "movie_id": int(movie_id),
                "director_id": None if director_id is None else int(director_id),
                "cast_ids": [int(row["person_id"]) for row in cast_rows],
                "company_ids": [int(row["company_id"]) for row in companies],
                "genre": str(concept.get("genre", "Drama")),
                "tier": str(concept.get("tier", "Mid")),
                "rating": round(float(rng.uniform(5.6, 8.4)), 3),
                "budget_usd": float(round(budget_usd, 2)),
                "box_office_usd": float(round(box_office_usd, 2)),
                "performance_ratio": float(round(box_office_usd / max(1.0, budget_usd), 4)),
                "market_regime_score": float(round(rng.uniform(-0.25, 0.30), 4)),
                "company_momentum": float(round(rng.uniform(-0.20, 0.35), 4)),
                "director_momentum": float(round(rng.uniform(-0.20, 0.35), 4)),
                "genre_heat": float(round(rng.uniform(-0.15, 0.25), 4)),
                "slate_pressure": float(round(rng.uniform(0.85, 1.20), 4)),
                "competition_pairs": [(int(a), int(b)) for a, b in competition_pairs[:12]],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a focused live runtime probe for step-100 hotspots.")
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset workspace, e.g. test32.")
    parser.add_argument("--out-json", required=True, help="Path for the JSON report.")
    parser.add_argument("--out-md", default=None, help="Optional path for the markdown summary.")
    parser.add_argument("--scratch-dir", default=None, help="Scratch workspace for year-transition side effects.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title-samples", type=int, default=128)
    parser.add_argument("--year-bucket-size", type=int, default=8)
    parser.add_argument("--director-samples", type=int, default=24)
    parser.add_argument("--from-year", type=int, default=1950)
    parser.add_argument("--to-year", type=int, default=1951)
    parser.add_argument("--title-profile-text", default=None, help="Optional path for one-call warmed pick_title cProfile text output.")
    parser.add_argument("--cast-profile-text", default=None, help="Optional path for one-call post-boundary pick_cast cProfile text output.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = Path(args.out_md).resolve() if args.out_md else None
    title_profile_text = Path(args.title_profile_text).resolve() if args.title_profile_text else None
    cast_profile_text = Path(args.cast_profile_text).resolve() if args.cast_profile_text else None
    scratch_dir = Path(args.scratch_dir).resolve() if args.scratch_dir else out_json.parent / "_probe_scratch"
    ensure_scratch_dirs(scratch_dir)
    try:
        sys.path.insert(0, str(dataset_dir))
        os.environ.setdefault("DATA_SYS_PIPELINE_MODE", "research")

        from assembly import pick_companies, pick_director, pick_title, sample_movie_concept
        from world_state import WorldState
        from year_planner import evolve_year

        rng = random.Random(int(args.seed))

        load_started = time.perf_counter()
        world = WorldState(str(dataset_dir), seed=int(args.seed))
        world.load()
        load_elapsed = time.perf_counter() - load_started

        # Keep all probe writes out of the baseline dataset workspace.
        world.base_dir = str(scratch_dir)
        world.decision_log_path = None
        world.decision_log_latest_path = None

        title_samples_sec: list[float] = []
        title_genre_counts: dict[str, int] = {}
        for idx in range(int(args.title_samples)):
            concept = sample_movie_concept(world, movie_id=500_000 + idx, forced_year=int(args.to_year))
            genre = str(concept.get("genre", ""))
            title_genre_counts[genre] = int(title_genre_counts.get(genre, 0) + 1)
            started = time.perf_counter()
            pick_title(world, concept)
            title_samples_sec.append(time.perf_counter() - started)

        title_profile_summary = None
        title_profile_text_content = None
        if title_profile_text is not None:
            target_concept = sample_movie_concept(world, movie_id=600_000, forced_year=int(args.to_year))
            profiler = cProfile.Profile()
            started = time.perf_counter()
            profiler.enable()
            prof_title, prof_tagline, prof_award = pick_title(world, target_concept)
            profiler.disable()
            prof_elapsed = time.perf_counter() - started
            stats_stream = io.StringIO()
            pstats.Stats(profiler, stream=stats_stream).sort_stats("cumulative").print_stats(40)
            title_profile_text_content = stats_stream.getvalue()
            title_profile_summary = {
                "elapsed_ms": round(float(prof_elapsed) * 1000.0, 3),
                "title": str(prof_title),
                "tagline": str(prof_tagline),
                "award": bool(prof_award),
            }

        prewarm_concept = sample_movie_concept(world, movie_id=700_000, forced_year=int(args.to_year))
        prewarm_director = pick_director(world, prewarm_concept)
        pick_companies(world, prewarm_concept, int(prewarm_director) if prewarm_director is not None else -1)
        from assembly import pick_cast  # defer to keep top-level imports clearer
        pick_cast(world, prewarm_concept, int(prewarm_director) if prewarm_director is not None else -1, max_retries=1)

        caches_before = snapshot_caches(world)
        year_bucket = build_year_bucket(
            world,
            from_year=int(args.from_year),
            count=int(args.year_bucket_size),
            id_offset=800_000,
            rng=rng,
        )

        year_started = time.perf_counter()
        rep = evolve_year(
            world,
            from_year=int(args.from_year),
            to_year=int(args.to_year),
            year_bucket=year_bucket,
            enable_llm=False,
            model=None,
            log_dir=str(scratch_dir / "decision_logs"),
            speed_audit=None,
        )
        year_elapsed = time.perf_counter() - year_started

        caches_after = snapshot_caches(world)

        cast_profile_summary = None
        cast_profile_text_content = None
        if cast_profile_text is not None:
            profile_concept = sample_movie_concept(world, movie_id=890_000, forced_year=int(args.to_year))
            profile_director_id = pick_director(world, profile_concept)
            cast_profiler = cProfile.Profile()
            cast_started = time.perf_counter()
            cast_profiler.enable()
            prof_cast_rows, prof_competition_pairs = pick_cast(
                world,
                profile_concept,
                int(profile_director_id) if profile_director_id is not None else -1,
                max_retries=1,
            )
            cast_profiler.disable()
            cast_elapsed = time.perf_counter() - cast_started
            cast_stats_stream = io.StringIO()
            pstats.Stats(cast_profiler, stream=cast_stats_stream).sort_stats("cumulative").print_stats(40)
            cast_profile_text_content = cast_stats_stream.getvalue()
            cast_profile_summary = {
                "elapsed_ms": round(float(cast_elapsed) * 1000.0, 3),
                "cast_size": int(len(prof_cast_rows)),
                "competition_pairs": int(len(prof_competition_pairs)),
            }

        director_samples_sec: list[float] = []
        cast_samples_sec: list[float] = []
        cast_sizes: list[int] = []
        director_movie_ids: list[int] = []
        for idx in range(int(args.director_samples)):
            movie_id = 900_000 + idx
            concept = sample_movie_concept(world, movie_id=movie_id, forced_year=int(args.to_year))
            started = time.perf_counter()
            director_id = pick_director(world, concept)
            director_samples_sec.append(time.perf_counter() - started)
            cast_started = time.perf_counter()
            cast_rows, _competition_pairs = pick_cast(
                world,
                concept,
                int(director_id) if director_id is not None else -1,
                max_retries=1,
            )
            cast_samples_sec.append(time.perf_counter() - cast_started)
            cast_sizes.append(int(len(cast_rows)))
            director_movie_ids.append(int(movie_id))

        payload = {
            "ok": True,
            "probe_version": "2026-04-19-title-profile-v1",
            "dataset_dir": str(dataset_dir),
            "scratch_dir": str(scratch_dir),
            "title_profile_text": None if title_profile_text is None else str(title_profile_text),
            "cast_profile_text": None if cast_profile_text is None else str(cast_profile_text),
            "seed": int(args.seed),
            "world_load_sec": round(float(load_elapsed), 4),
            "title_probe": {
                "samples": summarize_ms(title_samples_sec),
                "genre_counts": dict(sorted(title_genre_counts.items())),
                "profile": title_profile_summary,
            },
            "year_probe": {
                "from_year": int(args.from_year),
                "to_year": int(args.to_year),
                "year_bucket_size": int(len(year_bucket)),
                "year_elapsed_sec": round(float(year_elapsed), 4),
                "planner_source": str(getattr(rep, "planner_source", "")),
                "triggered_events": list(getattr(rep, "triggered_events", []) or []),
                "report": {
                    "applied": int(getattr(rep, "applied", 0)),
                    "skipped": int(getattr(rep, "skipped", 0)),
                    "errors": int(getattr(rep, "errors", 0)),
                    "invalidated_actor_cache": bool(getattr(rep, "invalidated_actor_cache", False)),
                    "invalidated_company_cache": bool(getattr(rep, "invalidated_company_cache", False)),
                    "edge_applied_by_family": {str(k): int(v) for k, v in getattr(rep, "edge_applied_by_family", {}).items()},
                    "edge_applied_by_mode": {str(k): int(v) for k, v in getattr(rep, "edge_applied_by_mode", {}).items()},
                },
                "caches_before": caches_before,
                "caches_after": caches_after,
                "director_post_boundary": summarize_ms(director_samples_sec),
                "cast_post_boundary": {
                    **summarize_ms(cast_samples_sec),
                    "mean_cast_size": round(statistics.fmean(cast_sizes), 2) if cast_sizes else None,
                    "max_cast_size": max(cast_sizes) if cast_sizes else None,
                },
                "cast_profile": cast_profile_summary,
                "director_movie_ids": director_movie_ids[:8],
            },
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "probe_version": "2026-04-19-title-profile-v1",
            "dataset_dir": str(dataset_dir),
            "scratch_dir": str(scratch_dir),
            "title_profile_text": None if title_profile_text is None else str(title_profile_text),
            "cast_profile_text": None if cast_profile_text is None else str(cast_profile_text),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if title_profile_text is not None and payload.get("ok") and title_profile_text_content is not None:
        title_profile_text.parent.mkdir(parents=True, exist_ok=True)
        title_profile_text.write_text(title_profile_text_content, encoding="utf-8")
    if cast_profile_text is not None and payload.get("ok") and cast_profile_text_content is not None:
        cast_profile_text.parent.mkdir(parents=True, exist_ok=True)
        cast_profile_text.write_text(cast_profile_text_content, encoding="utf-8")

    if out_md is not None and payload.get("ok"):
        lines = [
            "# Live Step-100 Probe",
            "",
            f"- dataset: `{dataset_dir}`",
            f"- scratch dir: `{scratch_dir}`",
            f"- world load: `{payload['world_load_sec']}` sec",
            "",
            "## Title Probe",
            "",
            f"- samples: `{payload['title_probe']['samples']['count']}`",
            f"- first: `{payload['title_probe']['samples']['first_ms']}` ms",
            f"- p50: `{payload['title_probe']['samples']['p50_ms']}` ms",
            f"- p95: `{payload['title_probe']['samples']['p95_ms']}` ms",
            f"- max: `{payload['title_probe']['samples']['max_ms']}` ms",
            f"- profiled steady-state call: `{(payload['title_probe'].get('profile') or {}).get('elapsed_ms')}` ms",
            "",
            "## Year Boundary Probe",
            "",
            f"- transition: `{args.from_year}->{args.to_year}`",
            f"- year program elapsed: `{payload['year_probe']['year_elapsed_sec']}` sec",
            f"- planner source: `{payload['year_probe']['planner_source']}`",
            f"- applied/skipped/errors: `{payload['year_probe']['report']['applied']}/{payload['year_probe']['report']['skipped']}/{payload['year_probe']['report']['errors']}`",
            f"- actor/company cache invalidated: `{payload['year_probe']['report']['invalidated_actor_cache']}/{payload['year_probe']['report']['invalidated_company_cache']}`",
            f"- director first post-boundary: `{payload['year_probe']['director_post_boundary']['first_ms']}` ms",
            f"- director p50 post-boundary: `{payload['year_probe']['director_post_boundary']['p50_ms']}` ms",
            f"- director p95 post-boundary: `{payload['year_probe']['director_post_boundary']['p95_ms']}` ms",
            f"- cast p50 post-boundary: `{payload['year_probe']['cast_post_boundary']['p50_ms']}` ms",
            f"- cast p95 post-boundary: `{payload['year_probe']['cast_post_boundary']['p95_ms']}` ms",
            f"- profiled post-boundary cast call: `{(payload['year_probe'].get('cast_profile') or {}).get('elapsed_ms')}` ms",
            "",
            "### Cache Snapshot",
            "",
            f"- before: `{json.dumps(payload['year_probe']['caches_before'], ensure_ascii=True, sort_keys=True)}`",
            f"- after: `{json.dumps(payload['year_probe']['caches_after'], ensure_ascii=True, sort_keys=True)}`",
        ]
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
