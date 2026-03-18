"""Cost-aware preflight estimator for scaling v16 generation to 10k+ movies (no API calls)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_STAGE_TOKENS = {
    "temporal_evolution": 220,   # tokens per movie equivalent
    "plot_summaries": 340,       # tokens per movie equivalent
    "tv_summaries": 45000,       # fixed overhead for ~150 series
    "latent_topup": 600,         # tokens per newly added entity
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _split_csv_arg(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _metric_or_none(value) -> float | None:
    """Convert to float or return None for NaN/invalid."""
    try:
        x = float(value)
    except Exception:
        return None
    if x != x:
        return None
    return float(x)


def _series_cv(series: pd.Series) -> float | None:
    """Coefficient of variation for a numeric series."""
    if series is None:
        return None
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) < 2:
        return None
    mean = float(s.mean())
    if abs(mean) < 1e-9:
        return None
    return float(s.std(ddof=0) / abs(mean))


def _output_realism_audit(output_dir: Path) -> dict:
    """V17: Comprehensive post-generation quality metrics.

    Reads generated CSVs from output_dir and computes realism metrics
    including financial correlations, temporal variation, cast structure,
    contract linkage, collaboration signals, and a composite hardness score.
    """
    movie = _read_csv(output_dir / 'movie.csv')
    if movie.empty:
        return {
            'status': 'unavailable',
            'reason': 'movie.csv not found in output_dir',
        }

    cast_info = _read_csv(output_dir / 'cast_info.csv')
    movie_companies = _read_csv(output_dir / 'movie_companies.csv')
    world_events = _read_csv(output_dir / 'world_events.csv')
    awards = _read_csv(output_dir / 'awards.csv')
    person_contracts = _read_csv(output_dir / 'person_contracts.csv')
    production_timeline = _read_csv(output_dir / 'production_timeline.csv')
    person_collaborations = _read_csv(output_dir / 'person_collaborations.csv')
    media_links = _read_csv(output_dir / 'media_links.csv')

    graph_metrics = {}
    graph_metrics_path = output_dir / 'graph' / 'graph_metrics.json'
    if graph_metrics_path.exists():
        try:
            graph_metrics = json.loads(graph_metrics_path.read_text(encoding='utf-8'))
        except Exception:
            graph_metrics = {}

    metrics: dict[str, float | int | None] = {
        'movie_count': int(len(movie)),
        'country_count': int(movie['country'].nunique()) if 'country' in movie.columns else None,
        'genre_count': int(movie['genre'].nunique()) if 'genre' in movie.columns else None,
    }

    # Financial correlations
    if {'budget_usd', 'box_office_usd'}.issubset(movie.columns):
        budget = pd.to_numeric(movie['budget_usd'], errors='coerce').clip(lower=1)
        box_office = pd.to_numeric(movie['box_office_usd'], errors='coerce')
        perf_ratio = (box_office / budget).replace([float('inf'), float('-inf')], pd.NA).dropna()
        metrics['budget_boxoffice_corr'] = _metric_or_none(
            movie[['budget_usd', 'box_office_usd']].corr(numeric_only=True).iloc[0, 1])
        metrics['performance_ratio_cv'] = _series_cv(perf_ratio)
    else:
        metrics['budget_boxoffice_corr'] = None
        metrics['performance_ratio_cv'] = None

    # Temporal variation
    if {'year', 'rating'}.issubset(movie.columns):
        year_means = movie.groupby('year')['rating'].mean()
        metrics['rating_year_cv'] = _series_cv(year_means)
    else:
        metrics['rating_year_cv'] = None

    # Genre concentration
    if 'genre' in movie.columns and len(movie) > 0:
        shares = movie['genre'].value_counts(normalize=True)
        metrics['genre_hhi'] = float((shares * shares).sum())
    else:
        metrics['genre_hhi'] = None

    # Cast structure
    if not cast_info.empty and 'title_id' in cast_info.columns:
        cast_sizes = cast_info.groupby('title_id').size()
        metrics['avg_cast_size'] = round(float(cast_sizes.mean()), 3)
    else:
        metrics['avg_cast_size'] = None

    # Company structure
    if not movie_companies.empty and 'title_id' in movie_companies.columns:
        company_counts = movie_companies.groupby('title_id').size()
        metrics['multi_company_rate'] = round(float((company_counts > 1).mean()), 3)
    else:
        metrics['multi_company_rate'] = None

    # World events
    metrics['world_event_density'] = round(float(len(world_events) / max(1, len(movie))), 3)

    # Awards
    metrics['award_win_rate'] = None
    if not awards.empty and 'outcome' in awards.columns:
        metrics['award_win_rate'] = round(
            float((awards['outcome'].astype(str) == 'Won').sum() / max(1, len(movie))), 3)

    # Production timeline coverage
    metrics['timeline_phase_coverage'] = None
    if not production_timeline.empty and {'movie_id', 'phase'}.issubset(production_timeline.columns):
        phase_counts = production_timeline.groupby('movie_id')['phase'].nunique()
        if len(phase_counts) > 0:
            metrics['timeline_phase_coverage'] = round(float((phase_counts >= 4).mean()), 3)

    # Contract linkage
    metrics['contract_match_rate'] = None
    if (not cast_info.empty and not movie_companies.empty and not person_contracts.empty
            and {'title_id', 'person_id'}.issubset(cast_info.columns)
            and {'title_id', 'company_id'}.issubset(movie_companies.columns)
            and {'person_id', 'company_id', 'start_date', 'end_date'}.issubset(person_contracts.columns)
            and {'title_id', 'year'}.issubset(movie.columns)):
        cast_pairs = cast_info[['title_id', 'person_id']].drop_duplicates()
        if len(cast_pairs) > 200000:
            cast_pairs = cast_pairs.sample(200000, random_state=17)
        movie_year = movie[['title_id', 'year']].drop_duplicates()
        company_pairs = movie_companies[['title_id', 'company_id']].drop_duplicates()
        contracts = person_contracts[['person_id', 'company_id', 'start_date', 'end_date']].copy()
        contracts['start_year'] = pd.to_numeric(contracts['start_date'].astype(str).str[:4], errors='coerce').fillna(0)
        contracts['end_year'] = pd.to_numeric(contracts['end_date'].astype(str).str[:4], errors='coerce').fillna(9999)
        joined = (cast_pairs
                  .merge(movie_year, on='title_id', how='inner')
                  .merge(company_pairs, on='title_id', how='inner')
                  .merge(contracts[['person_id', 'company_id', 'start_year', 'end_year']],
                         on=['person_id', 'company_id'], how='left'))
        if not joined.empty:
            joined['has_match'] = (joined['start_year'] <= joined['year']) & (joined['end_year'] >= joined['year'])
            metrics['contract_match_rate'] = round(
                float(joined.groupby(['title_id', 'person_id'])['has_match'].max().mean()), 3)

    # Collaboration signal
    metrics['repeat_collaboration_rate'] = None
    if (not cast_info.empty and not person_collaborations.empty
            and {'title_id', 'person_id'}.issubset(cast_info.columns)
            and {'person_a_id', 'person_b_id'}.issubset(person_collaborations.columns)):
        repeated_pairs = {
            (min(int(row.person_a_id), int(row.person_b_id)), max(int(row.person_a_id), int(row.person_b_id)))
            for row in person_collaborations[['person_a_id', 'person_b_id']].drop_duplicates().itertuples(index=False)
        }
        movie_cast = cast_info[['title_id', 'person_id']].drop_duplicates().groupby('title_id')['person_id'].apply(list)
        if len(movie_cast) > 20000:
            movie_cast = movie_cast.sample(20000, random_state=17)
        hits = 0
        total = 0
        for pids in movie_cast.tolist():
            ordered = sorted({int(pid) for pid in pids if int(pid) > 0})
            if len(ordered) < 2:
                continue
            total += 1
            for i in range(len(ordered)):
                found = False
                for j in range(i + 1, len(ordered)):
                    if (ordered[i], ordered[j]) in repeated_pairs:
                        hits += 1
                        found = True
                        break
                if found:
                    break
        metrics['repeat_collaboration_rate'] = round(float(hits / max(1, total)), 3) if total > 0 else None

    # Media bridges
    metrics['media_bridge_rate'] = round(float(len(media_links) / max(1, len(movie))), 3) if not media_links.empty else 0.0

    # Graph structure
    metrics['graph_giant_component_ratio'] = _metric_or_none(graph_metrics.get('giant_component_ratio'))
    metrics['graph_bridge_ratio'] = _metric_or_none(
        graph_metrics.get('block_bridge_ratio', graph_metrics.get('bridge_ratio')))

    # Composite benchmark hardness score
    hardness_components = []
    for value, scale in (
        (metrics['contract_match_rate'], 0.35),
        (metrics['timeline_phase_coverage'], 0.80),
        (metrics['repeat_collaboration_rate'], 0.10),
        (metrics['media_bridge_rate'], 0.02),
        (metrics['graph_bridge_ratio'], 0.08),
    ):
        if value is not None:
            hardness_components.append(min(1.0, float(value) / scale))
    metrics['benchmark_hardness_score'] = (
        round(float(sum(hardness_components) / len(hardness_components)), 3)
        if hardness_components else None
    )

    # Pass/fail checks
    checks = {
        'temporal_variation': (
            (metrics['rating_year_cv'] is not None and metrics['rating_year_cv'] >= 0.03)
            or metrics['world_event_density'] > 0.0
        ),
        'financial_dependency': (
            metrics['budget_boxoffice_corr'] is not None and metrics['budget_boxoffice_corr'] >= 0.35
        ),
        'market_variation': (
            metrics['performance_ratio_cv'] is not None and metrics['performance_ratio_cv'] >= 0.20
        ),
        'ensemble_structure': (
            (metrics['avg_cast_size'] is not None and metrics['avg_cast_size'] >= 4.0)
            and (metrics['multi_company_rate'] is not None and metrics['multi_company_rate'] >= 0.08)
        ),
        'graph_structure': (
            (metrics['graph_giant_component_ratio'] is None and metrics['graph_bridge_ratio'] is None)
            or ((metrics['graph_giant_component_ratio'] is None or metrics['graph_giant_component_ratio'] >= 0.80)
                and (metrics['graph_bridge_ratio'] is None or 0.02 <= metrics['graph_bridge_ratio'] <= 0.18))
        ),
        'temporal_joinability': (
            metrics['timeline_phase_coverage'] is not None and metrics['timeline_phase_coverage'] >= 0.80
        ),
        'contract_linkage': (
            metrics['contract_match_rate'] is not None and metrics['contract_match_rate'] >= 0.25
        ),
        'repeat_collaboration_signal': (
            metrics['repeat_collaboration_rate'] is not None and metrics['repeat_collaboration_rate'] >= 0.10
        ),
        'media_bridge_signal': (
            metrics['media_bridge_rate'] is not None and metrics['media_bridge_rate'] >= 0.01
        ),
    }
    checks['benchmark_hardness'] = bool(
        checks['contract_linkage']
        and checks['temporal_joinability']
        and (checks['repeat_collaboration_signal'] or checks['media_bridge_signal'] or checks['graph_structure'])
    )
    scored = [bool(v) for v in checks.values() if v is not None]
    passed = sum(1 for v in scored if v)
    total_checks = len(scored)
    if total_checks == 0:
        verdict = 'unknown'
    elif passed / total_checks >= 0.8:
        verdict = 'strong'
    elif passed / total_checks >= 0.5:
        verdict = 'mixed'
    else:
        verdict = 'weak'

    return {
        'status': 'available',
        'verdict': verdict,
        'metrics': metrics,
        'checks': checks,
    }



def estimate(base_dir: Path, target_movies: int, price_per_million: float, stages: list[str]) -> dict:
    ent = base_dir / "entities"
    graph = base_dir / "graph"

    person = _read_csv(ent / "person.csv")
    roles = _read_csv(ent / "person_roles.csv")
    company = _read_csv(ent / "company.csv")
    keyword = _read_csv(ent / "keyword.csv")
    title_bank = _read_csv(ent / "title_bank.csv")
    char_bank = _read_csv(ent / "character_bank.csv")
    persons_latent = _read_csv(ent / "persons_latent.csv")
    companies_latent = _read_csv(ent / "companies_latent.csv")

    # JSON latent fallbacks
    if persons_latent.empty and (ent / "persons_latent.json").exists():
        persons_latent = pd.read_json(ent / "persons_latent.json")
    if companies_latent.empty and (ent / "companies_latent.json").exists():
        companies_latent = pd.read_json(ent / "companies_latent.json")

    edge_rows = 0
    edge_file = graph / "edge_graph.csv"
    if edge_file.exists():
        edge_rows = int(sum(1 for _ in open(edge_file, "r", encoding="utf-8", errors="replace")) - 1)

    actors_available = 0
    directors_available = 0
    if not roles.empty:
        if "role_type" in roles.columns and "person_id" in roles.columns:
            actors_available = int(roles[roles["role_type"] == "actor"]["person_id"].nunique())
            directors_available = int(roles[roles["role_type"] == "director"]["person_id"].nunique())
    if actors_available == 0 and not person.empty:
        actors_available = int(person["person_id"].nunique())

    # v16 dynamic cast profile expectation (weighted average).
    tier_weights = {
        "Epic": 0.05,
        "A": 0.15,
        "Mid": 0.40,
        "Indie": 0.30,
        "Micro": 0.10,
    }
    tier_avg_cast = {
        "Epic": 39.0,
        "A": 17.0,
        "Mid": 7.5,
        "Indie": 4.2,
        "Micro": 2.1,
    }
    projected_avg_cast = sum(tier_weights[t] * tier_avg_cast[t] for t in tier_weights)
    projected_cast_rows = int(round(target_movies * projected_avg_cast))

    desired_films_per_actor = 3.2
    required_actor_pool = int(round(projected_cast_rows / desired_films_per_actor))
    actor_gap = max(0, required_actor_pool - actors_available)

    title_count = int(len(title_bank)) if not title_bank.empty else 0
    title_gap = max(0, target_movies - title_count)

    # Token and cost forecast
    est_tokens = 0
    stage_details = {}
    for stage in stages:
        if stage == "latent_topup":
            units = actor_gap
            tokens = int(units * DEFAULT_STAGE_TOKENS[stage])
        elif stage in ("temporal_evolution", "plot_summaries"):
            units = target_movies
            tokens = int(units * DEFAULT_STAGE_TOKENS[stage])
        elif stage == "tv_summaries":
            units = 1
            tokens = int(DEFAULT_STAGE_TOKENS[stage])
        else:
            units = 0
            tokens = 0
        est_tokens += tokens
        stage_details[stage] = {"units": units, "tokens": tokens, "cost_usd": tokens / 1_000_000 * price_per_million}

    est_cost = est_tokens / 1_000_000 * price_per_million

    saturation_ratio = (required_actor_pool / max(1, actors_available)) if actors_available else float("inf")
    if saturation_ratio <= 0.95:
        risk = "low"
    elif saturation_ratio <= 1.15:
        risk = "medium"
    else:
        risk = "high"

    return {
        "target_movies": target_movies,
        "price_per_million": price_per_million,
        "reuse_inventory": {
            "persons": int(len(person)),
            "actors_available": actors_available,
            "directors_available": directors_available,
            "companies": int(len(company)),
            "keywords": int(len(keyword)),
            "title_bank": title_count,
            "character_bank": int(len(char_bank)),
            "persons_latent": int(len(persons_latent)),
            "companies_latent": int(len(companies_latent)),
            "edge_graph_rows": edge_rows,
        },
        "projection": {
            "projected_avg_cast_per_movie": round(projected_avg_cast, 2),
            "projected_cast_rows": projected_cast_rows,
            "required_actor_pool": required_actor_pool,
            "actor_gap": actor_gap,
            "actor_saturation_ratio": round(saturation_ratio, 3) if saturation_ratio != float("inf") else None,
            "actor_saturation_risk": risk,
            "title_gap": title_gap,
        },
        "llm_forecast": {
            "assumed_stages": stages,
            "stage_details": stage_details,
            "total_tokens": est_tokens,
            "total_cost_usd": round(est_cost, 4),
        },
        "next_actions": [
            f"Top up title bank deterministically by {title_gap} titles before generation." if title_gap > 0 else "Title bank already covers target movie count.",
            "Do not start pipeline automatically; this preflight performs no API calls.",
            "Reuse current entities/graph first, then generate only missing parts.",
        ],
    }


def print_report(rep: dict):
    inv = rep["reuse_inventory"]
    proj = rep["projection"]
    llm = rep["llm_forecast"]

    print("=" * 70)
    print("V16 COST-AWARE PREFLIGHT (NO API CALLS)")
    print("=" * 70)
    print(f"Target movies:           {rep['target_movies']:,}")
    print(f"Price per 1M tokens:     ${rep['price_per_million']:.4f}")
    print()
    print("Reusable inventory")
    print(f"  persons:               {inv['persons']:,}")
    print(f"  actors available:      {inv['actors_available']:,}")
    print(f"  directors available:   {inv['directors_available']:,}")
    print(f"  companies:             {inv['companies']:,}")
    print(f"  keywords:              {inv['keywords']:,}")
    print(f"  title bank:            {inv['title_bank']:,}")
    print(f"  character bank:        {inv['character_bank']:,}")
    print(f"  persons latent:        {inv['persons_latent']:,}")
    print(f"  companies latent:      {inv['companies_latent']:,}")
    print(f"  edge graph rows:       {inv['edge_graph_rows']:,}")
    print()
    print("10k projection")
    print(f"  projected avg cast:    {proj['projected_avg_cast_per_movie']}")
    print(f"  projected cast rows:   {proj['projected_cast_rows']:,}")
    print(f"  required actor pool:   {proj['required_actor_pool']:,}")
    print(f"  actor gap:             {proj['actor_gap']:,}")
    print(f"  saturation risk:       {proj['actor_saturation_risk']}")
    print(f"  title gap:             {proj['title_gap']:,}")
    print()
    print("LLM forecast (optional stages)")
    for stage, d in llm["stage_details"].items():
        print(f"  {stage:<20} units={d['units']:<8} tokens={d['tokens']:<10,} cost=${d['cost_usd']:.4f}")
    print(f"  {'TOTAL':<20} tokens={llm['total_tokens']:<10,} cost=${llm['total_cost_usd']:.4f}")
    print()
    print("Immediate next actions")
    for i, step in enumerate(rep["next_actions"], start=1):
        print(f"  {i}. {step}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="v16 preflight estimator (no pipeline/API execution)")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent),
                        help="v16 working directory (contains entities/, graph/)")
    parser.add_argument("--target-movies", type=int, default=10_000)
    parser.add_argument("--price-per-million", type=float, default=1.50)
    parser.add_argument("--assume-llm-stages", default="temporal_evolution,plot_summaries,tv_summaries,latent_topup",
                        help="Comma-separated list of optional LLM stages to forecast")
    parser.add_argument("--report", default=None, help="Optional path to write JSON report")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    stages = _split_csv_arg(args.assume_llm_stages)

    rep = estimate(
        base_dir=base_dir,
        target_movies=int(args.target_movies),
        price_per_million=float(args.price_per_million),
        stages=stages,
    )
    print_report(rep)

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
