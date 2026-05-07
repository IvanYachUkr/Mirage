"""Build paper-facing signed Q-error plot bundle.

The signed scale is log10(Q-error), with positive values for overestimates,
negative values for underestimates, and zero for exact predictions. This makes
the plot show both magnitude and direction of estimator bias.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_BENCH = SCRIPT_DIR.parent
WORKSPACE = FINAL_BENCH.parents[1]
FINAL_RESULTS = FINAL_BENCH / "final_paper_results"
OUT_DIR = FINAL_RESULTS / "signed_qerror_plots"

MSCN_CARDINALITY_RUN = (
    WORKSPACE
    / "benchmark"
    / "cardinality_methods"
    / "runs"
    / "test42_100k_paper_mscn_cardinality_100k_100ep_3seed_20260504_exact3600s_quotefix"
)
ZEROSHOT_COST_RUN = (
    WORKSPACE
    / "benchmark"
    / "cardinality_methods"
    / "cost_models"
    / "runs"
    / "pretrained_zeroshot_mirage_test42_100k"
)

WORKLOAD_ORDER = ["JOB-Light", "JOB", "JOB-Complex"]
WORKLOAD_LABELS = {
    "job_light": "JOB-Light",
    "job_exact": "JOB",
    "job": "JOB",
    "job_complex": "JOB-Complex",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def signed_log_qerror(q_error: float, prediction: float, actual: float) -> float:
    if q_error <= 1.0 + 1e-12:
        return 0.0
    sign = 1.0 if prediction > actual else -1.0
    return sign * math.log10(q_error)


def signed_log_from_bias(q_error: float, bias: str) -> float:
    if q_error <= 1.0 + 1e-12 or bias == "exact":
        return 0.0
    return math.log10(q_error) if bias == "over" else -math.log10(q_error)


def add_record(
    rows: list[dict],
    *,
    plot_group: str,
    estimator: str,
    workload: str,
    query_id: str,
    q_error: float,
    signed_log10_qerror: float,
    source: str,
    seed: str = "",
) -> None:
    rows.append(
        {
            "plot_group": plot_group,
            "estimator": estimator,
            "workload": workload,
            "query_id": query_id,
            "seed": seed,
            "q_error": q_error,
            "signed_log10_qerror": signed_log10_qerror,
            "source": source,
        }
    )


def collect_base_selection(rows: list[dict]) -> None:
    path = FINAL_RESULTS / "base_selection_table1" / "base_selection_qerrors.csv"
    for row in read_csv(path):
        workload = WORKLOAD_LABELS[row["workload"]]
        estimator = {"postgres": "PostgreSQL", "duckdb": "DuckDB"}[row["estimator"]]
        q_error = float(row["q_error"])
        add_record(
            rows,
            plot_group="base_selection_cardinality",
            estimator=estimator,
            workload=workload,
            query_id=f"{row['query_id']}:{row['alias']}",
            q_error=q_error,
            signed_log10_qerror=signed_log_from_bias(q_error, row["bias"]),
            source=str(path),
        )


def collect_mscn_cardinality(rows: list[dict]) -> None:
    file_by_workload = {
        "job_light": "job_light.csv",
        "job_exact": "job_exact.csv",
        "job_complex": "job_complex.csv",
    }
    for seed in ("1", "2", "3"):
        for workload_key, filename in file_by_workload.items():
            path = (
                MSCN_CARDINALITY_RUN
                / "model_seeds"
                / f"seed_{seed}"
                / "predictions"
                / filename
            )
            for row in read_csv(path):
                prediction = float(row["prediction"])
                actual = float(row["actual"])
                q_error = float(row["q_error"])
                add_record(
                    rows,
                    plot_group="mscn_full_query_cardinality",
                    estimator="MSCN",
                    workload=WORKLOAD_LABELS[workload_key],
                    query_id=row["query_id"],
                    seed=seed,
                    q_error=q_error,
                    signed_log10_qerror=signed_log_qerror(q_error, prediction, actual),
                    source=str(path),
                )


def collect_postgres_cost(rows: list[dict]) -> None:
    per_query = FINAL_RESULTS / "cost_postgres_selected_plan" / "postgres_cost_per_query.csv"
    summary = FINAL_RESULTS / "cost_postgres_selected_plan" / "postgres_cost_summary.csv"
    scale_by_workload = {
        row["workload"]: float(row["median_cost_to_ms_scale"]) for row in read_csv(summary)
    }
    for row in read_csv(per_query):
        workload_key = row["workload"]
        prediction_ms = float(row["total_cost"]) * scale_by_workload[workload_key]
        actual_ms = float(row["execution_ms"])
        q_error = max(prediction_ms / actual_ms, actual_ms / prediction_ms)
        add_record(
            rows,
            plot_group="selected_plan_cost_runtime",
            estimator="PostgreSQL",
            workload=WORKLOAD_LABELS[workload_key],
            query_id=row["query_id"],
            q_error=q_error,
            signed_log10_qerror=signed_log_qerror(q_error, prediction_ms, actual_ms),
            source=str(per_query),
        )


def collect_zeroshot_cost(rows: list[dict]) -> None:
    file_specs = [
        ("job_light", "mirage_job_light_{seed}_test_pred.csv"),
        ("job", "mirage_job_{seed}_test_pred.csv"),
        ("job_complex", "mirage_job_complex_{seed}_test_pred.csv"),
    ]
    for seed in ("0", "1", "2"):
        for workload_key, pattern in file_specs:
            path = ZEROSHOT_COST_RUN / f"seed_{seed}" / pattern.format(seed=seed)
            for row in read_csv(path):
                prediction = float(row["prediction"])
                actual = float(row["label"])
                q_error = float(row["qerror"])
                add_record(
                    rows,
                    plot_group="selected_plan_cost_runtime",
                    estimator="Pretrained ZeroShot",
                    workload=WORKLOAD_LABELS[workload_key],
                    query_id=row["query_index"],
                    seed=seed,
                    q_error=q_error,
                    signed_log10_qerror=signed_log_qerror(q_error, prediction, actual),
                    source=str(path),
                )


def collect_mscn_cost(rows: list[dict]) -> None:
    file_by_workload = {
        "job_light": "job_light.csv",
        "job": "job.csv",
        "job_complex": "job_complex.csv",
    }
    for seed in ("1", "2", "3"):
        for workload_key, filename in file_by_workload.items():
            path = (
                FINAL_RESULTS
                / "cost_mscn_selected_plan"
                / f"seed_{seed}"
                / "predictions"
                / filename
            )
            for row in read_csv(path):
                prediction = float(row["prediction_us"])
                actual = float(row["actual_us"])
                q_error = float(row["q_error"])
                add_record(
                    rows,
                    plot_group="selected_plan_cost_runtime",
                    estimator="MSCN",
                    workload=WORKLOAD_LABELS[workload_key],
                    query_id=row["query_id"],
                    seed=seed,
                    q_error=q_error,
                    signed_log10_qerror=signed_log_qerror(q_error, prediction, actual),
                    source=str(path),
                )


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_points(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["plot_group"], row["estimator"], row["workload"])].append(
            float(row["signed_log10_qerror"])
        )

    summary_rows = []
    for (plot_group, estimator, workload), values in sorted(grouped.items()):
        abs_values = [abs(value) for value in values]
        over = sum(1 for value in values if value > 1e-12)
        under = sum(1 for value in values if value < -1e-12)
        exact = len(values) - over - under
        summary_rows.append(
            {
                "plot_group": plot_group,
                "estimator": estimator,
                "workload": workload,
                "n": len(values),
                "over": over,
                "under": under,
                "exact": exact,
                "signed_log10_mean": sum(values) / len(values),
                "signed_log10_median": percentile(values, 0.5),
                "abs_log10_median": percentile(abs_values, 0.5),
                "abs_log10_p95": percentile(abs_values, 0.95),
                "abs_log10_max": max(abs_values),
            }
        )
    return summary_rows


def tick_positions(values: list[float]) -> list[int]:
    max_abs = max(1, math.ceil(max(abs(value) for value in values)))
    max_abs = min(max_abs, 8)
    return list(range(-max_abs, max_abs + 1))


def tick_label(value: int) -> str:
    if value == 0:
        return "0"
    prefix = "+" if value > 0 else "-"
    return f"{prefix}10^{abs(value)}"


def plot_signed_box(
    rows: list[dict],
    *,
    plot_group: str,
    estimators: list[str],
    title: str,
    output_stem: str,
) -> None:
    group_rows = [row for row in rows if row["plot_group"] == plot_group]
    labels = []
    data = []
    for workload in WORKLOAD_ORDER:
        for estimator in estimators:
            values = [
                float(row["signed_log10_qerror"])
                for row in group_rows
                if row["workload"] == workload and row["estimator"] == estimator
            ]
            if values:
                labels.append(f"{workload}\n{estimator}")
                data.append(values)

    if not data:
        return

    fig_width = max(8.5, 0.72 * len(data) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)
    positions = list(range(1, len(data) + 1))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#f28e2b",
            "markeredgecolor": "#7a3b00",
            "markersize": 4,
        },
        medianprops={"color": "#111111", "linewidth": 1.4},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#4e79a7",
            "markeredgecolor": "none",
            "alpha": 0.16,
            "markersize": 2.8,
        },
    )

    colors = ["#d7e8f5", "#f6d7cf", "#dcebd3", "#eadcf2"]
    for patch, color in zip(box["boxes"], colors * 16):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.86)

    # Deterministic light jitter, so individual points are visible without
    # making the figure depend on randomness.
    for pos, values in zip(positions, data):
        for idx, value in enumerate(values):
            jitter = ((idx * 37) % 17 - 8) / 95.0
            ax.scatter(
                pos + jitter,
                value,
                s=9,
                color="#333333",
                alpha=0.24,
                linewidths=0,
                zorder=2,
            )

    flat = [value for values in data for value in values]
    ticks = tick_positions(flat)
    ax.set_yticks(ticks)
    ax.set_yticklabels([tick_label(tick) for tick in ticks])
    ax.axhline(0, color="#111111", linewidth=1.0)
    ax.grid(axis="y", color="#d0d0d0", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Signed log10 Q-error\n(overestimate +, underestimate -)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.margins(x=0.02)

    for suffix in ("svg", "png"):
        fig.savefig(OUT_DIR / f"{output_stem}.{suffix}", dpi=220)
    plt.close(fig)


def write_readme() -> None:
    readme = """# Signed Q-Error Plots

This folder contains paper-facing signed Q-error plots for the final Test42
100K benchmark instance.

Signed Q-error is plotted as `sign * log10(Q-error)`, where positive values are
overestimates, negative values are underestimates, and zero is exact. Each plot
uses boxplots with overlaid individual query points and an orange diamond for
the mean.

Plots:

- `signed_base_selection_cardinality_qerror.svg/.png`: PostgreSQL and DuckDB
  base-selection cardinality Q-errors.
- `signed_mscn_full_query_cardinality_qerror.svg/.png`: MSCN full-query
  cardinality Q-errors, pooled across the three trained seeds.
- `signed_selected_plan_cost_runtime_qerror.svg/.png`: selected-plan
  cost/runtime Q-errors for PostgreSQL, pretrained ZeroShot, and MSCN.

Data files:

- `signed_qerror_plot_points.csv`: one row per plotted estimate.
- `signed_qerror_plot_summary.csv`: compact summary by plot/model/workload.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    collect_base_selection(rows)
    collect_mscn_cardinality(rows)
    collect_postgres_cost(rows)
    collect_zeroshot_cost(rows)
    collect_mscn_cost(rows)

    fields = [
        "plot_group",
        "estimator",
        "workload",
        "query_id",
        "seed",
        "q_error",
        "signed_log10_qerror",
        "source",
    ]
    write_csv(OUT_DIR / "signed_qerror_plot_points.csv", rows, fields)
    summary_rows = summarize_points(rows)
    write_csv(
        OUT_DIR / "signed_qerror_plot_summary.csv",
        summary_rows,
        [
            "plot_group",
            "estimator",
            "workload",
            "n",
            "over",
            "under",
            "exact",
            "signed_log10_mean",
            "signed_log10_median",
            "abs_log10_median",
            "abs_log10_p95",
            "abs_log10_max",
        ],
    )

    plot_signed_box(
        rows,
        plot_group="base_selection_cardinality",
        estimators=["PostgreSQL", "DuckDB"],
        title="Base-Selection Cardinality Signed Q-Error",
        output_stem="signed_base_selection_cardinality_qerror",
    )
    plot_signed_box(
        rows,
        plot_group="mscn_full_query_cardinality",
        estimators=["MSCN"],
        title="MSCN Full-Query Cardinality Signed Q-Error",
        output_stem="signed_mscn_full_query_cardinality_qerror",
    )
    plot_signed_box(
        rows,
        plot_group="selected_plan_cost_runtime",
        estimators=["PostgreSQL", "Pretrained ZeroShot", "MSCN"],
        title="Selected-Plan Cost/Runtime Signed Q-Error",
        output_stem="signed_selected_plan_cost_runtime_qerror",
    )
    write_readme()
    manifest = {
        "points": len(rows),
        "summary_rows": len(summary_rows),
        "plots": [
            "signed_base_selection_cardinality_qerror.svg",
            "signed_mscn_full_query_cardinality_qerror.svg",
            "signed_selected_plan_cost_runtime_qerror.svg",
        ],
        "signed_qerror_definition": (
            "sign * log10(Q-error), with positive values for overestimates "
            "and negative values for underestimates"
        ),
    }
    (OUT_DIR / "signed_qerror_plot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
