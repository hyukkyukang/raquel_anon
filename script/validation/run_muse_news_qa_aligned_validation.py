#!/usr/bin/env python3
"""Run a bounded native-QA MUSE-News aligned-db validation workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "local_validation"
    / f"muse_news_qa_aligned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
DEFAULT_PROFILE_CONFIG = (
    REPO_ROOT / "config" / "validation" / "muse_news_qa_aligned_bounded.yaml"
)


def _default_args_map() -> dict[str, Any]:
    return {
        "profile_config": str(DEFAULT_PROFILE_CONFIG),
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "dataset_override": "muse_news_qa_aligned",
        "replay_source_root": None,
        "sample_num": 100,
        "aligned_db_name": "muse_news_qa_aligned",
        "null_db_name": "muse_news_qa_aligned_null",
        "overwrite": True,
        "run_dataset_quality": True,
        "run_raw_text_audit": True,
        "run_relation_audit": True,
        "extraction_max_concurrency": None,
        "extraction_requests_per_second": None,
        "round_trip_max_concurrency": None,
        "round_trip_requests_per_second": None,
    }


def _load_profile_config(profile_path: str | None) -> dict[str, Any]:
    if not profile_path:
        return {}
    path = Path(profile_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Validation profile must be a mapping: {profile_path}")
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded aligned-db generation from native MUSE-News QA splits."
    )
    parser.add_argument("--profile-config", default=argparse.SUPPRESS)
    parser.add_argument("--output-root", default=argparse.SUPPRESS)
    parser.add_argument("--dataset-override", default=argparse.SUPPRESS)
    parser.add_argument("--replay-source-root", default=argparse.SUPPRESS)
    parser.add_argument("--sample-num", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--aligned-db-name", default=argparse.SUPPRESS)
    parser.add_argument("--null-db-name", default=argparse.SUPPRESS)
    parser.add_argument("--extraction-max-concurrency", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--extraction-requests-per-second",
        type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--round-trip-max-concurrency", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--round-trip-requests-per-second",
        type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--overwrite", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument(
        "--run-dataset-quality",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-raw-text-audit",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-relation-audit",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parsed = parser.parse_args(argv)
    parsed_dict = vars(parsed)
    merged = _default_args_map()
    merged.update(_load_profile_config(parsed_dict.get("profile_config")))
    merged.update(parsed_dict)
    if merged.get("output_root") in (None, ""):
        merged["output_root"] = str(DEFAULT_OUTPUT_ROOT)
    return argparse.Namespace(**merged)


def _run_command(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_construct_command(
    args: argparse.Namespace,
    *,
    output_root: Path,
    aligned_dir: Path,
) -> list[str]:
    if args.replay_source_root:
        command = [
            sys.executable,
            "script/validation/replay_aligned_build_from_frozen_discovery.py",
            f"model.dir_path={aligned_dir}",
            f"database.db_id={args.aligned_db_name}",
            f"database.null_db_id={args.null_db_name}",
            f"log_dir={output_root / 'logs_construct'}",
            f"+replay_source_dir={Path(args.replay_source_root).resolve()}",
        ]
    else:
        command = [
            sys.executable,
            "script/stages/construct_aligned_db.py",
            f"dataset={args.dataset_override}",
            f"model.dir_path={aligned_dir}",
            f"model.aligned_db.sample_num={args.sample_num}",
            f"database.db_id={args.aligned_db_name}",
            f"database.null_db_id={args.null_db_name}",
            f"log_dir={output_root / 'logs_construct'}",
        ]
    if args.extraction_max_concurrency is not None:
        command.append(
            f"model.aligned_db.extraction_max_concurrency={args.extraction_max_concurrency}"
        )
    if args.extraction_requests_per_second is not None:
        command.append(
            "model.aligned_db.extraction_requests_per_second="
            f"{args.extraction_requests_per_second}"
        )
    if args.round_trip_max_concurrency is not None:
        command.append(
            f"model.aligned_db.round_trip_max_concurrency={args.round_trip_max_concurrency}"
        )
    if args.round_trip_requests_per_second is not None:
        command.append(
            "model.aligned_db.round_trip_requests_per_second="
            f"{args.round_trip_requests_per_second}"
        )
    if args.overwrite:
        command.append("overwrite=true")
    return command


def _build_nullify_command(
    args: argparse.Namespace,
    *,
    output_root: Path,
    aligned_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "script/stages/update_null.py",
        f"dataset={args.dataset_override}",
        f"model.dir_path={aligned_dir}",
        f"database.db_id={args.aligned_db_name}",
        f"database.null_db_id={args.null_db_name}",
        f"log_dir={output_root / 'logs_nullify'}",
    ]
    if args.overwrite:
        command.append("overwrite=true")
    return command


def _build_dataset_quality_command(
    aligned_dir: Path,
    *,
    aligned_db_name: str,
    null_db_name: str,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "script/analysis/dataset_quality_report.py",
        "--aligned-dir",
        str(aligned_dir),
        "--aligned-db",
        aligned_db_name,
        "--null-db",
        null_db_name,
        "--output",
        str(output_path),
    ]


def _build_raw_text_audit_command(
    aligned_dir: Path,
    *,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "script/analysis/raw_text_plausibility_audit.py",
        "--aligned-dir",
        str(aligned_dir),
        "--output",
        str(output_path),
    ]


def _build_relation_audit_command(
    aligned_dir: Path,
    *,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "script/analysis/qa_extraction_relation_audit.py",
        "--input",
        str(aligned_dir / "qa_extractions.json"),
        "--schema_registry",
        str(aligned_dir / "schema_registry.json"),
        "--output",
        str(output_path),
    ]


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    aligned_dir = output_root / "aligned_build"
    reports_dir = output_root / "reports"
    dataset_quality_path = reports_dir / "dataset_quality.json"
    raw_text_audit_path = reports_dir / "raw_text_plausibility.json"
    relation_audit_path = reports_dir / "relation_audit.json"
    summary_path = output_root / "validation_summary.json"

    _run_command(
        _build_construct_command(args, output_root=output_root, aligned_dir=aligned_dir)
    )
    _run_command(
        _build_nullify_command(args, output_root=output_root, aligned_dir=aligned_dir)
    )

    if args.run_dataset_quality:
        _run_command(
            _build_dataset_quality_command(
                aligned_dir,
                aligned_db_name=args.aligned_db_name,
                null_db_name=args.null_db_name,
                output_path=dataset_quality_path,
            )
        )

    if args.run_raw_text_audit:
        _run_command(
            _build_raw_text_audit_command(
                aligned_dir,
                output_path=raw_text_audit_path,
            )
        )

    if args.run_relation_audit:
        _run_command(
            _build_relation_audit_command(
                aligned_dir,
                output_path=relation_audit_path,
            )
        )

    summary = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "output_root": str(output_root),
        "dataset_override": args.dataset_override,
        "replay_source_root": args.replay_source_root,
        "sample_num": args.sample_num,
        "aligned_db_name": args.aligned_db_name,
        "null_db_name": args.null_db_name,
        "artifacts": {
            "aligned_dir": str(aligned_dir),
            "qa_extractions": str(aligned_dir / "qa_extractions.json"),
            "schema_registry": str(aligned_dir / "schema_registry.json"),
            "verification_summary": str(aligned_dir / "verification_summary.json"),
            "nullification_summary": str(aligned_dir / "log" / "nullify" / "summary.json"),
            "dataset_quality": str(dataset_quality_path) if args.run_dataset_quality else None,
            "raw_text_plausibility": str(raw_text_audit_path)
            if args.run_raw_text_audit
            else None,
            "relation_audit": str(relation_audit_path) if args.run_relation_audit else None,
        },
    }
    verification_summary_path = aligned_dir / "verification_summary.json"
    nullification_summary_path = aligned_dir / "log" / "nullify" / "summary.json"
    if verification_summary_path.exists():
        summary["verification"] = _load_json(verification_summary_path)
    if nullification_summary_path.exists():
        summary["nullification"] = _load_json(nullification_summary_path)

    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
