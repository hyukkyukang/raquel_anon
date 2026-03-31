"""Build a consolidated dataset-quality report for an aligned-build directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.aligned_db.dataset_quality import build_dataset_quality_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a consolidated dataset-quality report for an aligned-build directory."
    )
    parser.add_argument(
        "--aligned-dir",
        required=True,
        help="Path to the aligned build output directory",
    )
    parser.add_argument(
        "--aligned-db",
        default=None,
        help="Optional aligned database name to include in the report",
    )
    parser.add_argument(
        "--null-db",
        default=None,
        help="Optional null database name to include in the report",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name to include in the report",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to results/performance_review/dataset_quality_<aligned-dir-name>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    aligned_dir = Path(args.aligned_dir)
    report = build_dataset_quality_report(
        aligned_dir=aligned_dir,
        aligned_db=args.aligned_db,
        null_db=args.null_db,
        model_name=args.model_name,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path("results")
            / "performance_review"
            / f"dataset_quality_{aligned_dir.name}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
