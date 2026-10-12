"""Run a paired, two-dataset EC-EBO comparison on verified local inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from scripts.run_bfch_small_study import DATASETS, prepare_dataset
from src.comparison_report import export_comparison_report, summarize_saved_run
from src.evaluation import EvaluationConfig, evaluate_ten_fold
from src.results import _atomic_json


CONFIG = EvaluationConfig(classifiers=("LogReg", "GNB"),
                          methods=("nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo"),
                          seed=42, scale=True)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def ec_ebo_diagnostics(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep every successful fold's original cluster size, quota and shortfall."""
    details = []
    summaries = []
    for classifier in CONFIG.classifiers:
        selected = [record for record in records if record["job"]["classifier"] == classifier
                    and record["job"]["method"] == "ec_ebo"]
        successes = [record for record in selected if record["status"] == "success"]
        requested = generated = intrusions = ties = cluster_total = duplicates = 0
        shortfall_reasons: Counter[str] = Counter()
        for record in successes:
            diagnostics = record["resampler_diagnostics"]
            allocation = diagnostics["allocation"]
            sizes = allocation["cluster_sizes"]
            entropies = allocation["cluster_mean_entropies"]
            quotas = diagnostics["cluster_allocations"]
            created = diagnostics["cluster_generated_counts"]
            reasons = diagnostics["cluster_shortfall_reasons"]
            if not (len(sizes) == len(entropies) == len(quotas) == len(created) == len(reasons)):
                raise ValueError("EC-EBO cluster diagnostic lengths disagree.")
            if sum(quotas) != diagnostics["requested_count"] or sum(created) != diagnostics["generated_count"]:
                raise ValueError("EC-EBO cluster diagnostic totals disagree.")
            requested += diagnostics["requested_count"]
            generated += diagnostics["generated_count"]
            cluster_total += len(sizes)
            proxy = diagnostics["nearest_original_majority_proxy"]
            intrusions += proxy["count"]
            ties += proxy["tie_count"]
            duplicates += diagnostics["duplicate_generated_count"]
            for cluster_id, (size, entropy, quota, count, reason) in enumerate(zip(
                    sizes, entropies, quotas, created, reasons)):
                shortfall_reasons.update(reason)
                details.append({"classifier": classifier, "fold": record["job"]["fold"],
                                "cluster_id": cluster_id, "original_minority_rows": size,
                                "mean_entropy": entropy, "quota": quota, "generated": count,
                                "shortfall": quota - count,
                                "shortfall_reasons": json.dumps(reason, sort_keys=True)})
        summaries.append({"classifier": classifier, "successful_folds": len(successes),
                          "failed_folds": len(selected) - len(successes),
                          "mean_cluster_count": cluster_total / len(successes) if successes else None,
                          "requested_total": requested, "generated_total": generated,
                          "shortfall_total": requested - generated,
                          "duplicate_generated_total": duplicates,
                          "shortfall_reason_counts": dict(sorted(shortfall_reasons.items())),
                          "intrusion_proxy_count": intrusions,
                          "intrusion_proxy_rate": intrusions / generated if generated else None,
                          "nearest_distance_tie_count": ties})
    return details, summaries


def fpr_rows(records: list[dict]) -> list[dict]:
    """Held-out FP/(TN+FP), paired on folds successful for every method."""
    rows = []
    for classifier in CONFIG.classifiers:
        indexed = {(record["job"]["fold"], record["job"]["method"]): record
                   for record in records if record["job"]["classifier"] == classifier}
        shared = [fold for fold in range(10) if all(
            (fold, method) in indexed and indexed[fold, method]["status"] == "success"
            and (indexed[fold, method]["metrics"]["counts"]["tn"]
                 + indexed[fold, method]["metrics"]["counts"]["fp"]) > 0
            for method in CONFIG.methods)]
        for method in CONFIG.methods:
            values = []
            for fold in shared:
                counts = indexed[fold, method]["metrics"]["counts"]
                values.append(counts["fp"] / (counts["tn"] + counts["fp"]))
            rows.append({"classifier": classifier, "method": method,
                         "shared_fold_ids": shared, "shared_fold_count": len(shared),
                         "fpr_mean": float(np.mean(values)) if values else None,
                         "fpr_sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/ec-ebo-small-study"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    overview = {"purpose": "two-dataset descriptive EC-EBO comparison",
                "not_author_paper_scores": True,
                "config": {"classifiers": list(CONFIG.classifiers), "methods": list(CONFIG.methods),
                           "seed": CONFIG.seed, "scale": CONFIG.scale}, "datasets": []}
    for name in args.datasets:
        dataset_id, X, y, provenance, folds = prepare_dataset(name)
        provenance["kind"] = "verified local KEEL binary variant; author rows unverified"
        provenance["comparison"] = "same inputs and folds as prior BFCH small study"
        run_dir = args.output / dataset_id
        result = evaluate_ten_fold(X, y, dataset_id=dataset_id, positive_label=1,
                                   provenance=provenance, output_dir=run_dir,
                                   config=CONFIG, supplied_folds=folds)
        report = summarize_saved_run(run_dir)
        export_comparison_report(report, run_dir / "comparison.csv")
        details, diagnostics = ec_ebo_diagnostics(result["records"])
        fpr = fpr_rows(result["records"])
        write_csv(run_dir / "ec_ebo_clusters.csv", details,
                  ("classifier", "fold", "cluster_id", "original_minority_rows",
                   "mean_entropy", "quota", "generated", "shortfall", "shortfall_reasons"))
        write_csv(run_dir / "ec_ebo_diagnostics.csv", diagnostics,
                  ("classifier", "successful_folds", "failed_folds", "mean_cluster_count",
                   "requested_total", "generated_total", "shortfall_total",
                   "duplicate_generated_total", "shortfall_reason_counts",
                   "intrusion_proxy_count", "intrusion_proxy_rate", "nearest_distance_tie_count"))
        write_csv(run_dir / "fpr.csv", fpr,
                  ("classifier", "method", "shared_fold_ids", "shared_fold_count",
                   "fpr_mean", "fpr_sample_sd"))
        overview["datasets"].append({"dataset_id": dataset_id, "source_id": name,
                                     "input_sha256": provenance["source_sha256"],
                                     "rows": len(y), "features": X.shape[1],
                                     "minority": int(np.sum(y == 1)),
                                     "majority": int(np.sum(y == 0)),
                                     "run_id": result["run_id"], "reused_jobs": result["reused_jobs"],
                                     "report_status": report["status"],
                                     "coverage": report["coverage"],
                                     "ec_ebo_diagnostics": diagnostics,
                                     "source_fidelity": provenance["source_fidelity"],
                                     "excluded_features": provenance["excluded_features"],
                                     "fold_source": provenance["fold_source"]})
        print(f"{dataset_id}: {report['status']}, reused {result['reused_jobs']} jobs; "
              f"report at {run_dir / 'comparison.csv'}", flush=True)
        _atomic_json(args.output / "overview.json", overview)


if __name__ == "__main__":
    main()
