from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = max(0.0, min(1.0, float(pct))) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def stage_category(stage: str) -> str:
    text = str(stage or "")
    if text.startswith("secondary:") or text in {"release_dates", "box_office_daily"}:
        return "secondary_tables"
    if text.startswith("write_"):
        return "writes"
    if text.startswith("pick_"):
        return "selection"
    if "financial" in text:
        return "financials"
    if "graph" in text:
        return "graph"
    if "record_" in text:
        return "updates"
    return "other"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def find_progress_log(run_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None and explicit_path.exists():
        return explicit_path
    decision_dir = run_dir / "decision_logs"
    dated = sorted(decision_dir.glob("*_movie_generation_progress.jsonl"))
    if dated:
        return dated[-1]
    latest = decision_dir / "movie_generation_progress.jsonl"
    if latest.exists():
        return latest
    raise FileNotFoundError(f"No movie generation progress log found under {decision_dir}")


def profile_run(run_dir: Path, progress_log: Path) -> dict[str, Any]:
    stage_durations: dict[str, list[float]] = defaultdict(list)
    year_stage_totals: dict[int, Counter[str]] = defaultdict(Counter)
    year_totals: Counter[int] = Counter()
    year_movie_ids: dict[int, set[int]] = defaultdict(set)
    movie_last_elapsed: dict[int, float] = {}
    event_counts: Counter[str] = Counter()
    keyword_stage_counts: Counter[str] = Counter()
    stage_row_counts: dict[str, list[int]] = defaultdict(list)

    with progress_log.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            event = str(row.get("event", "") or "")
            event_counts[event] += 1

            if event == "step100_start":
                movie_last_elapsed.clear()
                continue

            if event == "movie_started":
                movie_id = int(row.get("movie_id", 0) or 0)
                if movie_id > 0:
                    movie_last_elapsed[movie_id] = 0.0
                    year = int(row.get("year", 0) or 0)
                    if year:
                        year_movie_ids[year].add(movie_id)
                continue

            if event == "keyword_stage":
                keyword_stage_counts[str(row.get("stage", "") or "")] += 1
                continue

            if event != "movie_stage":
                continue

            stage = str(row.get("stage", "") or "")
            movie_id = int(row.get("movie_id", 0) or 0)
            year = int(row.get("year", 0) or 0)
            elapsed = float(row.get("elapsed_sec", 0.0) or 0.0)
            prev = float(movie_last_elapsed.get(movie_id, 0.0))
            if elapsed < prev:
                prev = 0.0
            delta = max(0.0, elapsed - prev)
            movie_last_elapsed[movie_id] = elapsed

            stage_durations[stage].append(delta)
            year_stage_totals[year][stage] += delta
            year_totals[year] += delta
            if movie_id > 0:
                year_movie_ids[year].add(movie_id)
            if row.get("row_count") is not None:
                try:
                    stage_row_counts[stage].append(int(row.get("row_count", 0) or 0))
                except Exception:
                    pass

    stage_summary: list[dict[str, Any]] = []
    for stage, durations in stage_durations.items():
        row_counts = stage_row_counts.get(stage, [])
        stage_summary.append(
            {
                "stage": stage,
                "category": stage_category(stage),
                "count": len(durations),
                "total_sec": round(sum(durations), 4),
                "mean_sec": round(statistics.fmean(durations), 4) if durations else 0.0,
                "p50_sec": round(percentile(durations, 0.50) or 0.0, 4),
                "p95_sec": round(percentile(durations, 0.95) or 0.0, 4),
                "max_sec": round(max(durations) if durations else 0.0, 4),
                "avg_rows": round(statistics.fmean(row_counts), 2) if row_counts else None,
            }
        )
    stage_summary.sort(key=lambda row: row["total_sec"], reverse=True)

    category_totals: Counter[str] = Counter()
    for row in stage_summary:
        category_totals[row["category"]] += float(row["total_sec"])

    year_summary: list[dict[str, Any]] = []
    for year, total_sec in sorted(year_totals.items()):
        top_stage, top_stage_sec = ("", 0.0)
        if year_stage_totals[year]:
            top_stage, top_stage_sec = year_stage_totals[year].most_common(1)[0]
        year_summary.append(
            {
                "year": int(year),
                "movie_count": len(year_movie_ids.get(year, set())),
                "observed_total_sec": round(float(total_sec), 4),
                "top_stage": top_stage,
                "top_stage_sec": round(float(top_stage_sec), 4),
            }
        )
    year_summary.sort(key=lambda row: row["observed_total_sec"], reverse=True)

    graph_timing_path = run_dir / "graph" / "graph_build_timing.json"
    graph_summary = load_json(graph_timing_path) if graph_timing_path.exists() else None

    audit_path = run_dir / "decision_logs" / "research_mode_audit.json"
    audit_summary = load_json(audit_path) if audit_path.exists() else None

    pipeline_timing_path = run_dir / "pipeline_timing.jsonl"
    pipeline_step100 = None
    if pipeline_timing_path.exists():
        with pipeline_timing_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                row = json.loads(raw_line)
                if int(row.get("step_id", -1)) == 100:
                    pipeline_step100 = row
                    break

    return {
        "run_dir": str(run_dir),
        "progress_log": str(progress_log),
        "event_counts": dict(event_counts),
        "keyword_stage_counts": dict(keyword_stage_counts),
        "stage_summary": stage_summary,
        "category_totals": {
            key: round(float(value), 4) for key, value in sorted(category_totals.items(), key=lambda item: (-item[1], item[0]))
        },
        "year_summary": year_summary,
        "graph_timing": graph_summary,
        "audit_summary": audit_summary,
        "pipeline_step100": pipeline_step100,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, profile: dict[str, Any]) -> None:
    stage_summary = profile["stage_summary"]
    year_summary = profile["year_summary"]
    top_stages = stage_summary[:12]
    top_years = year_summary[:12]
    graph_timing = profile.get("graph_timing") or {}
    audit_summary = profile.get("audit_summary") or {}
    pipeline_step100 = profile.get("pipeline_step100") or {}

    lines = [
        "# Step 100 Runtime Profile",
        "",
        f"- run dir: `{profile['run_dir']}`",
        f"- progress log: `{profile['progress_log']}`",
        f"- observed movie stages: `{profile['event_counts'].get('movie_stage', 0)}`",
        f"- observed keyword stages: `{profile['event_counts'].get('keyword_stage', 0)}`",
        f"- observed completed movies: `{profile['event_counts'].get('movie_completed', 0)}`",
        f"- pipeline step100 elapsed_sec: `{pipeline_step100.get('elapsed_sec')}`",
        f"- pipeline step100 ok: `{pipeline_step100.get('ok')}`",
        f"- audit fallback_hit_count: `{audit_summary.get('fallback_hit_count')}`",
        "",
        "## Category Totals",
        "",
    ]
    for category, total_sec in profile["category_totals"].items():
        lines.append(f"- `{category}`: `{total_sec}` sec")

    lines.extend(
        [
            "",
            "## Top Stages",
            "",
        ]
    )
    for row in top_stages:
        lines.append(
            "- "
            + f"`{row['stage']}` total=`{row['total_sec']}` sec "
            + f"mean=`{row['mean_sec']}` sec p95=`{row['p95_sec']}` sec "
            + f"count=`{row['count']}`"
        )

    lines.extend(
        [
            "",
            "## Top Years",
            "",
        ]
    )
    for row in top_years:
        lines.append(
            "- "
            + f"`{row['year']}` movies=`{row['movie_count']}` observed_total=`{row['observed_total_sec']}` sec "
            + f"top_stage=`{row['top_stage']}` top_stage_sec=`{row['top_stage_sec']}`"
        )

    if isinstance(graph_timing, dict) and graph_timing.get("phases"):
        lines.extend(
            [
                "",
                "## Graph Build",
                "",
                f"- total graph elapsed_sec: `{graph_timing.get('total_elapsed_sec')}`",
            ]
        )
        for phase in graph_timing.get("phases", [])[:12]:
            lines.append(
                "- "
                + f"`{phase.get('phase')}` elapsed_sec=`{phase.get('elapsed_sec')}`"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile step-100 runtime from existing movie-generation logs.")
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. test32.")
    parser.add_argument("--out-dir", required=True, help="Where to write profiling artifacts.")
    parser.add_argument("--progress-log", default=None, help="Optional explicit progress log path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    progress_log = find_progress_log(run_dir, Path(args.progress_log).resolve() if args.progress_log else None)
    profile = profile_run(run_dir, progress_log)
    write_json(out_dir / "step100_profile.json", profile)
    write_markdown(out_dir / "step100_profile.md", profile)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "top_stage": profile["stage_summary"][0]["stage"] if profile["stage_summary"] else None,
                "top_stage_total_sec": profile["stage_summary"][0]["total_sec"] if profile["stage_summary"] else None,
                "top_year": profile["year_summary"][0]["year"] if profile["year_summary"] else None,
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
