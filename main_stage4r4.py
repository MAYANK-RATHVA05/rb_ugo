"""Inspect Stage 4R-4 inputs and capabilities without evaluation or results."""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.classifier_registry import CLASSIFIERS, smoke_classifier_on_fitting_data
from src.dataset_registry import check_all_candidates
from src.simulations import make_linear_simulation, make_nonlinear_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect provisional UGO experiment setup.")
    parser.add_argument("--smoke", action="store_true",
                        help="Fit supported classifiers on generated fitting rows and check predict_proba; no scores.")
    args = parser.parse_args()
    simulations = [make_linear_simulation(), make_nonlinear_simulation()]
    summaries = []
    for dataset in simulations:
        classes, counts = np.unique(dataset.y, return_counts=True)
        summaries.append({"name": dataset.name, "rows": len(dataset.X),
                          "features": dataset.X.shape[1],
                          "counts_by_label": dict(zip([str(x) for x in classes],
                                                      [int(x) for x in counts])),
                          "relabeled_majority_rows": len(dataset.relabeled_indices),
                          "seed": dataset.metadata["seed"],
                          "status": "declared_paper_reconstruction_not_author_rows"})
    report = {
        "stage": "4R-4", "simulations": summaries,
        "benchmark_candidates": [check.__dict__ for check in check_all_candidates()],
        "paper_74_list_status": "supplement_not_available",
        "classifiers": [{"name": spec.name, "status": spec.implementation_status}
                        for spec in CLASSIFIERS],
    }
    if args.smoke:
        fitting = simulations[0]
        report["training_only_probability_smoke"] = [
            smoke_classifier_on_fitting_data(spec.name, fitting.X, fitting.y).__dict__
            for spec in CLASSIFIERS if spec.implementation_status == "python_approximation"
        ]
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
