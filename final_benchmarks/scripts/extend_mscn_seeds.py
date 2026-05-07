"""Train additional MSCN seeds from existing prepared workloads.

This script deliberately reuses existing workload.jsonl/bitmaps.bin artifacts.
It does not regenerate SQL queries, rerun PostgreSQL labels, or touch the
paper-current copied result folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "benchmark" / "cardinality_methods" / "scripts"
MSCN_DIR = ROOT / "benchmark" / "cardinality_methods" / "mscn_paper_full"
COST_DIR = ROOT / "benchmark" / "cardinality_methods" / "cost_models"
for path in (SCRIPTS_DIR, MSCN_DIR, COST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_rich_mscn_experiment as rich  # noqa: E402
import paper_mscn_runner as card_runner  # noqa: E402
import mscn_selected_plan_cost_runner as cost_runner  # noqa: E402


DEFAULT_CARD_RUN = (
    ROOT
    / "benchmark"
    / "cardinality_methods"
    / "runs"
    / "test42_100k_paper_mscn_cardinality_100k_100ep_3seed_20260504_exact3600s_quotefix"
)
DEFAULT_COST_RUN = (
    ROOT
    / "benchmark"
    / "cardinality_methods"
    / "cost_models"
    / "runs"
    / "test42_100k_selected_plan_mscn_cost_10k_100ep_3seed_20260505"
)


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(chunk))
    return sorted(dict.fromkeys(seeds))


def load_prepared_data(
    *,
    run_dir: Path,
    num_samples: int,
    eval_names: list[str] | None = None,
) -> tuple[Any, Any, list[Any], Any, list[Any], dict[str, tuple[Any, list[Any]]]]:
    train_items, train_bitmaps = rich.load_workload(
        run_dir / "rich_train" / "workload.jsonl",
        run_dir / "rich_train" / "bitmaps.bin",
        num_samples,
    )
    split = int(len(train_items) * 0.9)
    train_core_items = train_items[:split]
    train_core_bitmaps = train_bitmaps[:split]
    val_items = train_items[split:]
    val_bitmaps = train_bitmaps[split:]

    encoder = rich.RichEncoder(train_items, train_bitmaps)
    loaded_eval_items: dict[str, list[Any]] = {}
    loaded_eval_bitmaps: dict[str, list[list[np.ndarray]]] = {}
    eval_dirs = sorted(run_dir.glob("rich_eval_*"))
    for eval_dir in eval_dirs:
        if not eval_dir.is_dir():
            continue
        name = eval_dir.name.removeprefix("rich_eval_")
        if eval_names is not None and name not in eval_names:
            continue
        workload_path = eval_dir / "workload.jsonl"
        bitmaps_path = eval_dir / "bitmaps.bin"
        if not workload_path.exists() or not bitmaps_path.exists():
            continue
        items, bitmaps = rich.load_workload(workload_path, bitmaps_path, num_samples)
        loaded_eval_items[name] = items
        loaded_eval_bitmaps[name] = bitmaps

    card_runner.extend_encoder_capacity(
        encoder, [item for items in loaded_eval_items.values() for item in items]
    )
    train_dataset, _ = encoder.encode_dataset(train_core_items, train_core_bitmaps)
    val_dataset, _ = encoder.encode_dataset(val_items, val_bitmaps)
    eval_datasets = {
        name: (encoder.encode_dataset(items, loaded_eval_bitmaps[name])[0], items)
        for name, items in loaded_eval_items.items()
    }
    return encoder, train_dataset, train_core_items, val_dataset, val_items, eval_datasets


def seed_done(kind: str, run_dir: Path, seed: int) -> bool:
    if kind == "cardinality":
        return (run_dir / "model_seeds" / f"seed_{seed}" / "qerror_summaries.json").exists()
    return (run_dir / f"seed_{seed}" / "qerror_summaries.json").exists()


def load_seed_summaries(kind: str, run_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if kind == "cardinality":
        seed_dirs = sorted((run_dir / "model_seeds").glob("seed_*"))
    else:
        seed_dirs = sorted(run_dir.glob("seed_*"))
    for seed_dir in seed_dirs:
        path = seed_dir / "qerror_summaries.json"
        if not path.exists():
            continue
        seed = seed_dir.name.removeprefix("seed_")
        summaries[seed] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def write_flat_seed_metrics(kind: str, run_dir: Path, summaries: dict[str, dict[str, Any]]) -> Path:
    out_path = run_dir / f"{kind}_all_seed_metrics_extra.csv"
    rows: list[dict[str, Any]] = []
    for seed, splits in sorted(summaries.items(), key=lambda kv: int(kv[0])):
        for split, summary in sorted(splits.items()):
            if isinstance(summary, dict) and "median" in summary:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "count": summary.get("count"),
                        "median": summary.get("median"),
                        "p90": summary.get("p90"),
                        "p95": summary.get("p95"),
                        "p99": summary.get("p99"),
                        "max": summary.get("max"),
                        "mean": summary.get("mean"),
                    }
                )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["seed", "split", "count", "median", "p90", "p95", "p99", "max", "mean"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["cardinality", "cost"], required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seeds", required=True, help="Comma/range list, e.g. 4-13 or 4,5,6")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--eval-names",
        default="",
        help="Optional comma list of rich_eval_* names to include. Empty means all prepared evals.",
    )
    args = parser.parse_args()

    run_dir = (args.run_dir or (DEFAULT_CARD_RUN if args.kind == "cardinality" else DEFAULT_COST_RUN)).resolve()
    seeds = parse_seeds(args.seeds)
    eval_names = [x.strip() for x in args.eval_names.split(",") if x.strip()] or None
    print(
        json.dumps(
            {
                "kind": args.kind,
                "run_dir": str(run_dir),
                "seeds": seeds,
                "epochs": args.epochs,
                "device": args.device,
                "eval_names": eval_names or "all",
            },
            indent=2,
        ),
        flush=True,
    )

    encoder, train_dataset, train_items, val_dataset, val_items, eval_datasets = load_prepared_data(
        run_dir=run_dir,
        num_samples=args.num_samples,
        eval_names=eval_names,
    )
    runner_args = argparse.Namespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hid=args.hid,
        device=args.device,
        num_samples=args.num_samples,
    )

    new_summaries: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        if args.skip_existing and seed_done(args.kind, run_dir, seed):
            print(json.dumps({"skip_existing_seed": seed}), flush=True)
            continue
        print(json.dumps({"start_seed": seed, "kind": args.kind, "time": time.strftime("%Y-%m-%d %H:%M:%S")}), flush=True)
        if args.kind == "cardinality":
            new_summaries[str(seed)] = card_runner.train_and_eval_seed(
                seed=seed,
                run_dir=run_dir,
                encoder=encoder,
                train_dataset=train_dataset,
                train_items=train_items,
                val_dataset=val_dataset,
                val_items=val_items,
                eval_datasets=eval_datasets,
                args=runner_args,
            )
        else:
            device = cost_runner.resolve_torch_device(args.device)
            new_summaries[str(seed)] = cost_runner.train_eval_seed(
                seed=seed,
                run_dir=run_dir,
                encoder=encoder,
                train_dataset=train_dataset,
                train_items=train_items,
                val_dataset=val_dataset,
                val_items=val_items,
                eval_datasets=eval_datasets,
                args=argparse.Namespace(**{**vars(runner_args), "device": device}),
            )
        print(json.dumps({"done_seed": seed, "kind": args.kind, "time": time.strftime("%Y-%m-%d %H:%M:%S")}), flush=True)

    all_summaries = load_seed_summaries(args.kind, run_dir)
    aggregate = (
        card_runner.aggregate_seed_summaries(all_summaries)
        if args.kind == "cardinality"
        else cost_runner.aggregate_seed_summaries(all_summaries)
    )
    metrics_path = write_flat_seed_metrics(args.kind, run_dir, all_summaries)
    report_path = run_dir / f"{args.kind}_extra_seed_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "kind": args.kind,
        "run_dir": str(run_dir),
        "requested_seeds": seeds,
        "new_seed_count": len(new_summaries),
        "all_seed_count": len(all_summaries),
        "epochs": args.epochs,
        "aggregate": aggregate,
        "metrics_csv": str(metrics_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "report": str(report_path), "metrics_csv": str(metrics_path), "aggregate": aggregate}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
