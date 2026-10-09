"""Inspect one binary CSV/KEEL dataset; optionally save descriptive JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.config import RESULTS_DIR
from src.data_loader import DatasetError, LoadedDataset, load_dataset
from src.imbalance_stats import ImbalanceStats, calculate_statistics, format_summary


def build_parser() -> argparse.ArgumentParser:
    """Document only implemented Stage 2 arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to a UTF-8 .csv or KEEL .dat file.")
    parser.add_argument("--target", help="CSV target column (required); optional KEEL @outputs override.")
    parser.add_argument("--minority-label", help="Exact original text label; required for equal class counts.")
    parser.add_argument(
        "--save-json", nargs="?", const="", metavar="FILENAME",
        help="Save under results/stage2/; default <dataset-stem>_report.json. Never overwrite an existing file.",
    )
    return parser


def save_report(dataset: LoadedDataset, stats: ImbalanceStats, filename: str) -> Path:
    """Save metadata and statistics, without samples, exclusively under results/stage2."""
    if not filename or "/" in filename or "\\" in filename or Path(filename).name != filename or not filename.endswith(".json"):
        raise DatasetError("JSON report name must be a simple filename ending in .json (no directory path).")
    report = {
        "source_filename": dataset.source_filename,
        "source_format": dataset.source_format,
        "target_name": dataset.target_name,
        "label_mapping": dataset.label_mapping,
        "majority_label": dataset.majority_label,
        "minority_label": dataset.minority_label,
        "feature_names": list(dataset.feature_names),
        "numeric_feature_names": list(dataset.numeric_feature_names),
        "categorical_feature_names": list(dataset.categorical_feature_names),
        "statistics": stats.to_dict(),
        "duplicate_definition": "Additional occurrences beyond the first; complete rows include the target.",
    }
    destination = RESULTS_DIR / "stage2" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    """Run inspection; readable errors return 1, argument errors return 2."""
    args = build_parser().parse_args(argv)
    try:
        dataset = load_dataset(args.data, args.target, args.minority_label)
        stats = calculate_statistics(dataset)
        print(format_summary(dataset, stats))
        if args.save_json is not None:
            filename = args.save_json or f"{args.data.stem}_report.json"
            destination = save_report(dataset, stats, filename)
            print(f"JSON report saved: {destination}")
    except (DatasetError, OSError, ValueError) as exc:
        print(f"Stage 2 error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
