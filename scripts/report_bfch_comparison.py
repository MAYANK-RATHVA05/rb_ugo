"""Export a same-fold comparison table from one saved evaluation run."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.comparison_report import export_comparison_report, summarize_saved_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directory with manifest.json and job-*.json.")
    parser.add_argument("--output", type=Path, help="CSV path; defaults to RUN/comparison.csv.")
    args = parser.parse_args()
    report = summarize_saved_run(args.run)
    csv_path, json_path = export_comparison_report(report, args.output or args.run / "comparison.csv")
    print(f"Saved {csv_path.resolve()} and {json_path.resolve()}")
    print(f"Dataset={report['dataset_id']} status={report['status']} run_id={report['run_id']}")
    print("Shared folds by classifier / metric:")
    for classifier in report["classifiers"]:
        for metric in ("iam", "mcc", "f1", "gmean", "auc"):
            row = next(row for row in report["rows"]
                       if row["classifier"] == classifier and row["metric"] == metric)
            print(f"  {classifier} {metric}: {row['shared_fold_count']}/10")


if __name__ == "__main__":
    main()
