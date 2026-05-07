"""Build compact paper-style signed Q-error interval plots.

These figures are intentionally cleaner than the diagnostic point-cloud plots:
each row shows the median signed log10 Q-error, a thick IQR interval, and a thin
5th-95th percentile interval.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_BENCH = SCRIPT_DIR.parent
OUT_DIR = FINAL_BENCH / "final_paper_results" / "signed_qerror_plots"
POINTS_CSV = OUT_DIR / "signed_qerror_plot_points.csv"

WORKLOAD_ORDER = ["JOB-Light", "JOB", "JOB-Complex"]

PLOT_SPECS = [
    {
        "plot_group": "base_selection_cardinality",
        "estimators": ["PostgreSQL", "DuckDB"],
        "title": "Base-Selection Cardinality Error",
        "stem": "paper_clean_base_selection_cardinality_qerror",
    },
    {
        "plot_group": "mscn_full_query_cardinality",
        "estimators": ["MSCN"],
        "title": "MSCN Full-Query Cardinality Error",
        "stem": "paper_clean_mscn_full_query_cardinality_qerror",
    },
    {
        "plot_group": "selected_plan_cost_runtime",
        "estimators": ["PostgreSQL", "Pretrained ZeroShot", "MSCN"],
        "title": "Selected-Plan Cost/Runtime Error",
        "stem": "paper_clean_selected_plan_cost_runtime_qerror",
    },
]

COLORS = {
    "PostgreSQL": "#4E79A7",
    "DuckDB": "#F28E2B",
    "MSCN": "#59A14F",
    "Pretrained ZeroShot": "#B07AA1",
}


def read_points() -> list[dict[str, str]]:
    with POINTS_CSV.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def interval_rows(points: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in points:
        grouped[(row["plot_group"], row["workload"], row["estimator"])].append(
            float(row["signed_log10_qerror"])
        )

    rows: list[dict] = []
    for (plot_group, workload, estimator), values in sorted(grouped.items()):
        rows.append(
            {
                "plot_group": plot_group,
                "workload": workload,
                "estimator": estimator,
                "n": len(values),
                "p05": percentile(values, 0.05),
                "p25": percentile(values, 0.25),
                "median": percentile(values, 0.50),
                "p75": percentile(values, 0.75),
                "p95": percentile(values, 0.95),
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def tick_label(value: int) -> str:
    if value == 0:
        return "0"
    prefix = "+" if value > 0 else "-"
    return f"{prefix}10^{abs(value)}"


def symmetric_limit(rows: list[dict]) -> int:
    max_abs = max(
        1.0,
        max(abs(float(row["p05"])) for row in rows),
        max(abs(float(row["p95"])) for row in rows),
    )
    return min(6, max(2, math.ceil(max_abs)))


def ordered_rows(rows: list[dict], plot_group: str, estimators: list[str]) -> list[dict]:
    selected = []
    for workload in WORKLOAD_ORDER:
        for estimator in estimators:
            for row in rows:
                if (
                    row["plot_group"] == plot_group
                    and row["workload"] == workload
                    and row["estimator"] == estimator
                ):
                    selected.append(row)
    return selected


def plot_interval(rows: list[dict], spec: dict) -> None:
    selected = ordered_rows(rows, spec["plot_group"], spec["estimators"])
    if not selected:
        return

    height = max(3.2, 0.36 * len(selected) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, height), constrained_layout=True)
    y_positions = list(range(len(selected), 0, -1))
    xlim = symmetric_limit(selected)

    previous_workload = None
    for y, row in zip(y_positions, selected):
        workload = row["workload"]
        estimator = row["estimator"]
        color = COLORS.get(estimator, "#555555")

        if previous_workload is not None and previous_workload != workload:
            ax.axhline(y + 0.5, color="#e0e0e0", linewidth=0.8)
        previous_workload = workload

        ax.plot([row["p05"], row["p95"]], [y, y], color=color, alpha=0.35, linewidth=2.0)
        ax.plot([row["p25"], row["p75"]], [y, y], color=color, alpha=0.95, linewidth=6.0)
        ax.scatter(
            [row["median"]],
            [y],
            color="white",
            edgecolor=color,
            linewidth=1.6,
            s=34,
            zorder=3,
        )

    labels = [f"{row['workload']}  |  {row['estimator']}" for row in selected]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(-xlim, xlim)
    ticks = list(range(-xlim, xlim + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick_label(tick) for tick in ticks])
    ax.axvline(0, color="#222222", linewidth=1.0)
    ax.grid(axis="x", color="#d9d9d9", linestyle=":", linewidth=0.8)
    ax.set_title(spec["title"], fontsize=11.5, pad=8)
    ax.set_xlabel("Signed log10 Q-error (underestimates \u2190 0 \u2192 overestimates)", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend = [
        Line2D([0], [0], color="#777777", linewidth=2, alpha=0.35, label="5th-95th percentile"),
        Line2D([0], [0], color="#777777", linewidth=6, alpha=0.95, label="IQR"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markeredgecolor="#777777",
            linestyle="None",
            label="median",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    for suffix in ("svg", "png"):
        fig.savefig(OUT_DIR / f"{spec['stem']}.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)


def update_readme() -> None:
    readme = OUT_DIR / "README.md"
    text = readme.read_text(encoding="utf-8")
    addition = """

## Clean Paper-Style Figures

The `paper_clean_*` figures are the preferred paper-facing versions. They avoid
the diagnostic point cloud and instead show one horizontal interval per
model/workload:

- thin line: 5th-95th percentile signed log10 Q-error;
- thick line: interquartile range;
- hollow marker: median.

Preferred files:

- `paper_clean_base_selection_cardinality_qerror.svg/.png`
- `paper_clean_mscn_full_query_cardinality_qerror.svg/.png`
- `paper_clean_selected_plan_cost_runtime_qerror.svg/.png`
- `paper_clean_signed_qerror_interval_summary.csv`
"""
    if "## Clean Paper-Style Figures" not in text:
        readme.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def main() -> None:
    points = read_points()
    rows = interval_rows(points)
    write_csv(
        OUT_DIR / "paper_clean_signed_qerror_interval_summary.csv",
        rows,
        ["plot_group", "workload", "estimator", "n", "p05", "p25", "median", "p75", "p95", "min", "max"],
    )
    for spec in PLOT_SPECS:
        plot_interval(rows, spec)
    update_readme()
    print(f"wrote {len(PLOT_SPECS)} clean plot families to {OUT_DIR}")


if __name__ == "__main__":
    main()
