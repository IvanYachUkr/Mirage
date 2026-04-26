#!/usr/bin/env python3
"""Create a non-destructive cleanup audit for Mirage benchmark workspaces."""

from __future__ import annotations

import argparse
from pathlib import Path


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--current-run", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    current_run = Path(args.current_run).resolve() if args.current_run else None
    out_path = Path(args.out).resolve()
    safe_review: list[str] = []
    keep: list[str] = []

    runs_dir = root / "benchmark" / "cardinality_methods" / "runs"
    if runs_dir.exists():
        for path in sorted(runs_dir.iterdir()):
            if not path.is_dir():
                continue
            if current_run and path.resolve() == current_run:
                keep.append(f"`{rel(path, root)}`: current CE run from this task.")
            else:
                safe_review.append(
                    f"`{rel(path, root)}`: older CE artifact run; safe to review for deletion only after copying any reports you still need."
                )

    for base in [root / "benchmark", root / "test40"]:
        if not base.exists():
            continue
        for path in base.rglob("__pycache__"):
            safe_review.append(f"`{rel(path, root)}`: Python bytecode cache.")
        for path in base.rglob(".pytest_cache"):
            safe_review.append(f"`{rel(path, root)}`: pytest cache.")

    tmp_smoke = Path("/tmp/mirage_ce_smoke_label")
    if tmp_smoke.exists():
        safe_review.append(f"`{tmp_smoke}`: temporary smoke-test artifacts from CE label validation.")

    for path in sorted(root.glob("test*")):
        if path.is_dir():
            keep.append(f"`{rel(path, root)}`: generated/test dataset directory; do not delete without explicit dataset review.")

    for path in [
        root / "benchmark" / "job_exact_v1",
        root / "benchmark" / "job_complex_v1",
        root / "benchmark" / "cardinality_methods" / "sources",
        root / "imdb_job_dataset",
    ]:
        if path.exists():
            keep.append(f"`{rel(path, root)}`: benchmark/source workspace needed for reproducibility.")

    lines = [
        "# Non-Destructive Cleanup Audit",
        "",
        "This report only identifies candidates. Nothing was deleted.",
        "",
        "## Safe To Review Later",
        *(f"- {item}" for item in safe_review),
        "",
        "## Do Not Touch Yet",
        *(f"- {item}" for item in keep),
        "",
        "## Notes",
        "- Old generated datasets and Git-related files are intentionally excluded from automatic cleanup.",
        "- Review older benchmark runs by report value before deleting; some are useful baselines for the paper narrative.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
