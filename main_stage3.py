"""Provisional UGO software sanity check; no benchmark or final-model scoring."""
from __future__ import annotations

import json

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE
from src.ugo import UGO


def main() -> None:
    """Run both variants on one deterministic artificial fitting dataset."""
    print("PROVISIONAL: full paper unavailable; this is not verified reproduction.")
    print("Software sanity check only; no validation/test data or performance metrics.")
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0,
        weights=[0.8, 0.2], flip_y=0, class_sep=0.5, random_state=RANDOM_STATE,
    )
    print(f"Original class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    for variant in ("ros", "smote"):
        sampler = UGO(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            variant=variant, random_state=RANDOM_STATE, allow_provisional=True,
        )
        _, augmented_y = sampler.fit_resample(X, y)
        diagnostics = sampler.diagnostics_
        print(f"\nUGO-{variant.upper()} (provisional)")
        print(f"Final class counts: {dict(zip(*np.unique(augmented_y, return_counts=True)))}")
        for field in ["number_of_synthetic_samples", "number_of_iterations",
                      "pre_detected_noisy_samples", "generated_noisy_samples_relocated",
                      "generation_cap", "stopping_reason", "reproduction_status"]:
            print(f"{field}: {diagnostics[field]}")
        print("Selected classes:", [record["selected_class"] for record in diagnostics["history"]])
        if diagnostics["history"]:
            print("First iteration:", json.dumps(diagnostics["history"][0], allow_nan=False))
        print("Effective parameters:", json.dumps(diagnostics["effective_parameters"]))
        print("Assumptions:", json.dumps(diagnostics["warnings"]))


if __name__ == "__main__":
    main()
