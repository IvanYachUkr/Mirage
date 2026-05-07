"""Build vertical JOB-style signed Q-error figures.

Inspired by the original JOB paper's signed boxplot layout: panels by estimator,
vertical signed log10 Q-error axis, and workload-colored boxes.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_BENCH = SCRIPT_DIR.parent
OUT_DIR = FINAL_BENCH / "final_paper_results" / "signed_qerror_plots"
POINTS_CSV = OUT_DIR / "signed_qerror_plot_points.csv"

WORKLOAD_ORDER = ["JOB-Light", "JOB", "JOB-Complex"]
WORKLOAD_COLORS = {
    "JOB-Light": "#4E79A7",
    "JOB": "#F28E2B",
    "JOB-Complex": "#59A14F",
}

PLOT_SPECS = [
    {
        "plot_group": "base_selection_cardinality",
        "estimators": ["PostgreSQL", "DuckDB"],
        "title": "Base-Selection Cardinality Error",
        "stem": "paper_jobstyle_base_selection_cardinality_qerror",
    },
    {
        "plot_group": "mscn_full_query_cardinality",
        "estimators": ["MSCN"],
        "title": "MSCN Full-Query Cardinality Error",
        "stem": "paper_jobstyle_mscn_full_query_cardinality_qerror",
    },
    {
        "plot_group": "selected_plan_cost_runtime",
        "estimators": ["PostgreSQL", "Pretrained ZeroShot", "MSCN"],
        "title": "Selected-Plan Cost/Runtime Error",
        "stem": "paper_jobstyle_selected_plan_cost_runtime_qerror",
    },
]


def read_points() -> list[dict[str, str]]:
    with POINTS_CSV.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


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


def tick_label(value: int) -> str:
    if value == 0:
        return "0"
    prefix = "+" if value > 0 else "-"
    return f"{prefix}10^{abs(value)}"


def values_by_group(points: list[dict[str, str]]) -> dict[tuple[str, str, str], list[float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in points:
        grouped[(row["plot_group"], row["estimator"], row["workload"])].append(
            float(row["signed_log10_qerror"])
        )
    return grouped


def y_limit(values: list[float]) -> int:
    if not values:
        return 2
    max_abs = max(abs(value) for value in values)
    return min(8, max(2, math.ceil(max_abs)))


def sampled_tail_points(values: list[float], lower: float, upper: float, max_points: int = 10) -> list[float]:
    tails = sorted([value for value in values if value < lower or value > upper])
    if len(tails) <= max_points:
        return tails
    if max_points == 1:
        return [tails[len(tails) // 2]]
    indexes = [
        round(idx * (len(tails) - 1) / (max_points - 1)) for idx in range(max_points)
    ]
    return [tails[index] for index in indexes]


def plot_spec(points: list[dict[str, str]], spec: dict) -> None:
    grouped = values_by_group(points)
    selected_values = [
        value
        for estimator in spec["estimators"]
        for workload in WORKLOAD_ORDER
        for value in grouped.get((spec["plot_group"], estimator, workload), [])
    ]
    ylim = y_limit(selected_values)

    panel_count = len(spec["estimators"])
    width = max(4.2, 2.15 * panel_count)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(width, 3.25),
        sharey=True,
        constrained_layout=True,
    )
    if panel_count == 1:
        axes = [axes]

    for ax, estimator in zip(axes, spec["estimators"]):
        data = [
            grouped.get((spec["plot_group"], estimator, workload), [])
            for workload in WORKLOAD_ORDER
        ]
        positions = list(range(1, len(WORKLOAD_ORDER) + 1))
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.58,
            whis=[5, 95],
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#111111", "linewidth": 1.0},
            whiskerprops={"color": "#333333", "linewidth": 0.9},
            capprops={"color": "#333333", "linewidth": 0.9},
            boxprops={"color": "#333333", "linewidth": 0.9},
        )

        for patch, workload in zip(box["boxes"], WORKLOAD_ORDER):
            patch.set_facecolor(WORKLOAD_COLORS[workload])
            patch.set_alpha(0.62)

        for pos, workload, values in zip(positions, WORKLOAD_ORDER, data):
            if not values:
                continue
            p05 = percentile(values, 0.05)
            p95 = percentile(values, 0.95)
            tails = sampled_tail_points(values, p05, p95)
            for idx, value in enumerate(tails):
                jitter = ((idx * 19) % 7 - 3) / 70.0
                ax.plot(
                    pos + jitter,
                    value,
                    marker=".",
                    markersize=2.2,
                    color="#222222",
                    alpha=0.55,
                    linestyle="None",
                )

        ax.axhline(0, color="#111111", linewidth=0.9)
        ax.grid(axis="y", color="#d9d9d9", linestyle="-", linewidth=0.45)
        ax.set_title(estimator, fontsize=9.5, pad=5)
        ax.set_xticks(positions)
        ax.set_xticklabels(WORKLOAD_ORDER, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(-ylim, ylim)
        ax.set_yticks(list(range(-ylim, ylim + 1)))
        ax.set_yticklabels([tick_label(tick) for tick in range(-ylim, ylim + 1)], fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("signed log10 Q-error\nunder <- 0 -> over", fontsize=8.5)

    for suffix in ("svg", "png"):
        fig.savefig(OUT_DIR / f"{spec['stem']}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_readme() -> None:
    readme = OUT_DIR / "README.md"
    text = readme.read_text(encoding="utf-8")
    addition = """

## Vertical JOB-Style Figures

The `paper_jobstyle_*` figures follow the visual style of the original JOB
paper more closely: vertical signed log-scale boxplots, panels by estimator,
and colors by workload. Boxes show the IQR, whiskers show the 5th-95th
percentile, and only a small deterministic sample of tail points is drawn.

Preferred vertical files:

- `paper_jobstyle_base_selection_cardinality_qerror.svg/.png`
- `paper_jobstyle_mscn_full_query_cardinality_qerror.svg/.png`
- `paper_jobstyle_selected_plan_cost_runtime_qerror.svg/.png`
"""
    if "## Vertical JOB-Style Figures" not in text:
        readme.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def main() -> None:
    points = read_points()
    for spec in PLOT_SPECS:
        plot_spec(points, spec)
    update_readme()
    print(f"wrote {len(PLOT_SPECS)} vertical JOB-style plot families to {OUT_DIR}")


if __name__ == "__main__":
    main()
