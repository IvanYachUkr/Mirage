from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feather_sink import read_table
from policy_runtime import comparison_report_path, write_json


def _load_table(base_dir: Path, name: str, table_name: str | None = None) -> pd.DataFrame:
    return read_table(str(base_dir / name), table_name)


def _first_present(frame: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _movie_key(frame: pd.DataFrame) -> str | None:
    return _first_present(frame, "movie_id", "title_id")


def _director_key(frame: pd.DataFrame) -> str | None:
    return _first_present(frame, "director_id", "person_id")


def _cramers_v(frame: pd.DataFrame, left: str, right: str) -> float:
    if frame.empty or left not in frame.columns or right not in frame.columns:
        return 0.0
    table = pd.crosstab(frame[left], frame[right])
    if table.empty:
        return 0.0
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total <= 0:
        return 0.0
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    mask = expected > 0
    chi2 = float(((observed - expected) ** 2 / np.where(mask, expected, 1.0))[mask].sum())
    n = total
    r, k = observed.shape
    denom = max(1.0, min(k - 1, r - 1))
    return float(np.sqrt(max(0.0, chi2 / n / denom)))


def _gini(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[arr >= 0]
    if arr.size == 0 or float(arr.sum()) <= 0:
        return 0.0
    idx = np.arange(1, arr.size + 1, dtype=float)
    return float((2.0 * np.sum(idx * arr) - (arr.size + 1.0) * np.sum(arr)) / (arr.size * np.sum(arr)))


def _company_genre_concentration(base_dir: Path) -> float:
    movie = _load_table(base_dir, "movie", "movie")
    movie_companies = _load_table(base_dir, "movie_companies", "movie_companies")
    if movie.empty or movie_companies.empty:
        return 0.0
    movie_key = _movie_key(movie)
    company_key = _movie_key(movie_companies)
    if movie_key is None or company_key is None or "genre" not in movie.columns:
        return 0.0
    joined = movie_companies.merge(
        movie[[movie_key, "genre"]],
        left_on=company_key,
        right_on=movie_key,
        how="inner",
    )
    if joined.empty:
        return 0.0
    shares = []
    for _cid, group in joined.groupby("company_id"):
        counts = group["genre"].value_counts(normalize=True)
        if not counts.empty:
            shares.append(float(counts.iloc[0]))
    return float(np.mean(shares)) if shares else 0.0


def _repeat_collaboration_persistence(base_dir: Path) -> float:
    cast = _load_table(base_dir, "cast_info", "cast_info")
    directors = _load_table(base_dir, "movie_directors", "movie_directors")
    if cast.empty or directors.empty:
        return 0.0
    cast_key = _movie_key(cast)
    director_movie_key = _movie_key(directors)
    director_person_key = _director_key(directors)
    if cast_key is None or director_movie_key is None or director_person_key is None or "person_id" not in cast.columns:
        return 0.0
    joined = cast.merge(
        directors[[director_movie_key, director_person_key]].rename(columns={director_person_key: "director_id"}),
        left_on=cast_key,
        right_on=director_movie_key,
        how="inner",
    )
    if joined.empty:
        return 0.0
    pairs = joined.groupby(["director_id", "person_id"]).size()
    if pairs.empty:
        return 0.0
    return float((pairs > 1).mean())


def _cast_clique_recurrence(base_dir: Path) -> float:
    cast = _load_table(base_dir, "cast_info", "cast_info")
    movie_key = _movie_key(cast)
    if cast.empty or movie_key is None or "person_id" not in cast.columns:
        return 0.0
    pair_counts: dict[tuple[int, int], int] = {}
    for _movie_id, group in cast.groupby(movie_key):
        ids = sorted(int(value) for value in group["person_id"].dropna().astype(int).tolist())
        for idx, left in enumerate(ids):
            for right in ids[idx + 1 :]:
                key = (left, right)
                pair_counts[key] = pair_counts.get(key, 0) + 1
    if not pair_counts:
        return 0.0
    return float(sum(1 for count in pair_counts.values() if count > 1) / max(1, len(pair_counts)))


def _keyword_selectivity_skew(base_dir: Path) -> float:
    movie_keyword = _load_table(base_dir, "movie_keyword", "movie_keyword")
    if movie_keyword.empty:
        return 0.0
    counts = movie_keyword.groupby("keyword_id").size().astype(float).tolist()
    return _gini(counts)


def _average_keyword_count(base_dir: Path) -> float:
    movie_keyword = _load_table(base_dir, "movie_keyword", "movie_keyword")
    movie_key = _movie_key(movie_keyword)
    if movie_keyword.empty or movie_key is None:
        return 0.0
    counts = movie_keyword.groupby(movie_key).size()
    return float(counts.mean()) if len(counts) else 0.0


def _duplicate_keyword_set_rate(base_dir: Path) -> float:
    movie_keyword = _load_table(base_dir, "movie_keyword", "movie_keyword")
    movie_key = _movie_key(movie_keyword)
    if movie_keyword.empty or movie_key is None or "keyword_id" not in movie_keyword.columns:
        return 0.0
    sets = movie_keyword.groupby(movie_key)["keyword_id"].agg(lambda s: tuple(sorted(set(int(v) for v in s.dropna().astype(int).tolist()))))
    if sets.empty:
        return 0.0
    dup_counts = sets.value_counts()
    duplicate_movies = int(dup_counts[dup_counts > 1].sum()) if not dup_counts.empty else 0
    return float(duplicate_movies / max(1, len(sets)))


def _generic_keyword_fraction(base_dir: Path) -> float:
    movie_keyword = _load_table(base_dir, "movie_keyword", "movie_keyword")
    keyword_csv = base_dir / "entities" / "keyword.csv"
    if movie_keyword.empty or not keyword_csv.exists() or "keyword_id" not in movie_keyword.columns:
        return 0.0
    keyword_df = pd.read_csv(keyword_csv, low_memory=False)
    if keyword_df.empty or "keyword_id" not in keyword_df.columns:
        return 0.0
    joined = movie_keyword.merge(keyword_df[["keyword_id"] + [c for c in ["specificity_tier", "motif_family"] if c in keyword_df.columns]], on="keyword_id", how="left")
    if joined.empty:
        return 0.0
    specificity = pd.to_numeric(joined.get("specificity_tier", 2), errors="coerce").fillna(2).astype(float)
    motif_family = joined.get("motif_family", pd.Series([""] * len(joined))).fillna("").astype(str)
    generic_mask = (specificity <= 1.0) | motif_family.isin(["genre", "tone"])
    return float(generic_mask.mean())


def _franchise_keyword_recurrence_vs_drift(base_dir: Path) -> float:
    movie = _load_table(base_dir, "movie", "movie")
    movie_keyword = _load_table(base_dir, "movie_keyword", "movie_keyword")
    if movie.empty or movie_keyword.empty:
        return 0.0
    movie_key = _movie_key(movie)
    keyword_movie_key = _movie_key(movie_keyword)
    required = {"franchise_id", "installment_no"}
    if movie_key is None or keyword_movie_key is None or not required.issubset(movie.columns):
        return 0.0
    kw_sets = movie_keyword.groupby(keyword_movie_key)["keyword_id"].agg(lambda s: set(int(v) for v in s.dropna().astype(int).tolist())).to_dict()
    scores = []
    franchised = movie.dropna(subset=["franchise_id"]).copy()
    if franchised.empty:
        return 0.0
    franchised["installment_no"] = pd.to_numeric(franchised["installment_no"], errors="coerce").fillna(0).astype(int)
    for _fid, group in franchised.groupby("franchise_id"):
        ordered = group.sort_values("installment_no")
        prev_set = None
        for movie_id in ordered[movie_key].astype(int).tolist():
            cur_set = kw_sets.get(movie_id, set())
            if prev_set is not None and cur_set:
                union = prev_set | cur_set
                if union:
                    jaccard = len(prev_set & cur_set) / len(union)
                    identical = 1.0 if prev_set == cur_set else 0.0
                    scores.append(max(0.0, jaccard - 0.35 * identical))
            prev_set = cur_set
    return float(np.mean(scores)) if scores else 0.0


def _keyword_dimension_selectivity(base_dir: Path, dimension: str) -> float:
    movie = _load_table(base_dir, "movie", "movie")
    movie_keyword = _load_table(base_dir, "movie_keyword", "movie_keyword")
    if movie.empty or movie_keyword.empty:
        return 0.0
    movie_key = _movie_key(movie)
    keyword_movie_key = _movie_key(movie_keyword)
    if movie_key is None or keyword_movie_key is None:
        return 0.0
    if dimension == "company":
        movie_companies = _load_table(base_dir, "movie_companies", "movie_companies")
        company_movie_key = _movie_key(movie_companies)
        if movie_companies.empty or company_movie_key is None or "company_id" not in movie_companies.columns:
            return 0.0
        company_rows = movie_companies.sort_values([company_movie_key, "company_id"]).drop_duplicates(company_movie_key)
        joined = movie_keyword.merge(
            company_rows[[company_movie_key, "company_id"]],
            left_on=keyword_movie_key,
            right_on=company_movie_key,
            how="inner",
        )
        dim_col = "company_id"
    else:
        if dimension not in movie.columns:
            return 0.0
        joined = movie_keyword.merge(
            movie[[movie_key, dimension]],
            left_on=keyword_movie_key,
            right_on=movie_key,
            how="inner",
        )
        dim_col = dimension
    if joined.empty or "keyword_id" not in joined.columns:
        return 0.0
    scores = []
    for _keyword_id, group in joined.groupby("keyword_id"):
        counts = group[dim_col].astype(str).value_counts(normalize=True)
        if not counts.empty:
            scores.append(float(counts.iloc[0]))
    return float(np.mean(scores)) if scores else 0.0


def _build_metrics(base_dir: Path) -> dict[str, float]:
    movie = _load_table(base_dir, "movie", "movie")
    return {
        "genre_country_association_strength": _cramers_v(movie, "genre", "country"),
        "company_genre_concentration": _company_genre_concentration(base_dir),
        "repeat_collaboration_persistence": _repeat_collaboration_persistence(base_dir),
        "cast_clique_recurrence": _cast_clique_recurrence(base_dir),
        "keyword_selectivity_skew": _keyword_selectivity_skew(base_dir),
        "average_keyword_count_per_movie": _average_keyword_count(base_dir),
        "duplicate_keyword_set_rate": _duplicate_keyword_set_rate(base_dir),
        "generic_keyword_fraction": _generic_keyword_fraction(base_dir),
        "franchise_keyword_recurrence_vs_drift": _franchise_keyword_recurrence_vs_drift(base_dir),
        "keyword_genre_selectivity": _keyword_dimension_selectivity(base_dir, "genre"),
        "keyword_country_selectivity": _keyword_dimension_selectivity(base_dir, "country"),
        "keyword_company_selectivity": _keyword_dimension_selectivity(base_dir, "company"),
        "keyword_franchise_selectivity": _keyword_dimension_selectivity(base_dir, "franchise_id"),
    }


def _metric_delta(base: float, variant: float) -> float:
    return float(variant - base)


def compare_runs(base_a: Path, base_b: Path, out_path: Path | None = None) -> dict[str, Any]:
    metrics_a = _build_metrics(base_a)
    metrics_b = _build_metrics(base_b)
    report = {
        "baseline": str(base_a),
        "variant": str(base_b),
        "metrics": {
            key: {
                "baseline": round(float(metrics_a.get(key, 0.0)), 6),
                "variant": round(float(metrics_b.get(key, 0.0)), 6),
                "delta": round(_metric_delta(metrics_a.get(key, 0.0), metrics_b.get(key, 0.0)), 6),
            }
            for key in sorted(set(metrics_a) | set(metrics_b))
        },
    }
    target = out_path or comparison_report_path(base_b)
    write_json(target, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare correlation metrics between two DATA_SYS runs.")
    parser.add_argument("--baseline", default=str(Path(__file__).resolve().parent.parent / "test25"))
    parser.add_argument("--variant", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = compare_runs(
        Path(args.baseline).resolve(),
        Path(args.variant).resolve(),
        Path(args.out).resolve() if args.out else None,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
