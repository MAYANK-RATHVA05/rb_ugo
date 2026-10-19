"""Run the frozen three-dataset Soft-NADOS feasibility screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.nados import NADOS
from src.soft_nados import SoftNADOS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "soft_nados_pilot_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "soft-nados-pilot-v1"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    design = config.get("scientific_design", {})
    if config.get("protocol_id") != "soft-nados-feasibility-v1":
        raise ValueError("Unexpected Soft-NADOS pilot protocol.")
    if design.get("methods") != ["NORES", "WEIGHTED", "NADOS", "SOFT_NADOS"]:
        raise ValueError("The frozen pilot methods changed.")
    if design.get("soft_nados", {}).get("temperature") != .05:
        raise ValueError("The frozen Soft-NADOS temperature changed.")
    return config


def design_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config["scientific_design"], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cancer = load_breast_cancer()
    wine = load_wine()
    digits = load_digits()

    cancer_y = (cancer.target == 0).astype(int)  # malignant is the minority.
    wine_y = (wine.target == 2).astype(int)
    digits_y = (digits.target == 1).astype(int)
    minority = np.flatnonzero(digits_y == 1)
    majority = np.flatnonzero(digits_y == 0)
    rng = np.random.default_rng(314159)
    kept_majority = np.sort(rng.choice(majority, size=3 * len(minority), replace=False))
    kept = np.sort(np.r_[minority, kept_majority])

    return {
        "breast_cancer": (np.asarray(cancer.data, dtype=float), cancer_y),
        "wine_class_2_vs_rest": (np.asarray(wine.data, dtype=float), wine_y),
        "digits_1_vs_rest_3_to_1": (np.asarray(digits.data[kept], dtype=float), digits_y[kept]),
    }


def inject_symmetric_noise(y: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Flip a fixed fraction inside each true class; return labels and flip mask."""
    y = np.asarray(y, dtype=int)
    noisy = y.copy()
    flipped = np.zeros(len(y), dtype=bool)
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        candidates = np.flatnonzero(y == label)
        count = int(np.floor(rate * len(candidates)))
        chosen = rng.choice(candidates, size=count, replace=False)
        noisy[chosen] = 1 - label
        flipped[chosen] = True
    return noisy, flipped


def _sampler(method: str, design: dict[str, Any], seed: int):
    settings = design["nados"]
    kwargs = {
        "k_neighbors": settings["k_neighbors"],
        "reliability_weight": settings["reliability_weight"],
        "difficulty_weight": settings["difficulty_weight"],
        "reliability_threshold": settings["reliability_threshold"],
        "target_ratio": settings["target_ratio"],
        "minority_label": 1,
        "random_state": seed,
    }
    if method == "NADOS":
        return NADOS(**kwargs)
    if method == "SOFT_NADOS":
        return SoftNADOS(**kwargs, temperature=design["soft_nados"]["temperature"])
    return None


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "gmean": float(np.sqrt(sensitivity * specificity)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def evaluate_dataset(name: str, X: np.ndarray, y: np.ndarray,
                     design: dict[str, Any]) -> list[dict[str, Any]]:
    folds = design["folds"]
    splitter = StratifiedKFold(n_splits=folds["n_splits"], shuffle=folds["shuffle"],
                              random_state=folds["random_state"])
    rows: list[dict[str, Any]] = []
    noise = design["training_label_noise"]
    classifier = design["classifier"]
    for fold, (train, test) in enumerate(splitter.split(X, y), start=1):
        fold_seed = int(folds["random_state"] + fold)
        y_noisy, corrupted = inject_symmetric_noise(
            y[train], noise["rate_per_true_class"], noise["fold_seed_offset"] + fold
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])
        for method in design["methods"]:
            sampler = _sampler(method, design, fold_seed)
            if sampler is None:
                fit_X, fit_y = X_train, y_noisy
                generated = 0
                exposure = np.nan
            else:
                fit_X, fit_y = sampler.fit_resample(X_train, y_noisy)
                generated = len(fit_X) - len(X_train)
                parent_corrupted = (corrupted[sampler.generated_seed_indices_]
                                    | corrupted[sampler.generated_partner_indices_])
                exposure = float(parent_corrupted.mean()) if generated else 0.0

            model = LogisticRegression(
                max_iter=classifier["max_iter"], solver=classifier["solver"],
                random_state=classifier["random_state"],
                class_weight="balanced" if method == "WEIGHTED" else None,
            )
            model.fit(fit_X, fit_y)
            probabilities = model.predict_proba(X_test)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
            rows.append({
                "dataset": name,
                "fold": fold,
                "method": method,
                "train_rows": len(train),
                "test_rows": len(test),
                "observed_train_minority": int(np.sum(y_noisy == 1)),
                "flipped_train_labels": int(corrupted.sum()),
                "generated": generated,
                "corrupted_parent_exposure": exposure,
                **_metrics(y[test], probabilities),
            })
    return rows


def summarize(scores: pd.DataFrame, design: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = (scores.groupby(["dataset", "method"], sort=False, as_index=False)
               .agg(f1=("f1", "mean"), gmean=("gmean", "mean"),
                    average_precision=("average_precision", "mean"),
                    corrupted_parent_exposure=("corrupted_parent_exposure", "mean")))
    pivot = summary.pivot(index="dataset", columns="method")
    f1_delta = pivot["f1"]["SOFT_NADOS"] - pivot["f1"]["NADOS"]
    ap_delta = (pivot["average_precision"]["SOFT_NADOS"]
                - pivot["average_precision"]["NADOS"])
    exposure_delta = (pivot["corrupted_parent_exposure"]["SOFT_NADOS"]
                      - pivot["corrupted_parent_exposure"]["NADOS"])
    rule = design["decision_rule"]
    checks = {
        "mean_f1": bool(f1_delta.mean() >= rule["mean_f1_delta_soft_minus_nados_at_least"]),
        "dataset_f1_wins": bool((f1_delta > 0).sum() >= rule["dataset_f1_wins_required"]),
        "mean_average_precision": bool(ap_delta.mean() >= rule["mean_average_precision_delta_at_least"]),
        "mean_exposure": bool(exposure_delta.mean() <= rule["mean_exposure_increase_at_most"]),
    }
    decision = {
        "status": "pilot_trigger_met" if all(checks.values()) else "pilot_trigger_not_met",
        "checks": checks,
        "soft_minus_nados": {
            "mean_dataset_f1": float(f1_delta.mean()),
            "dataset_f1_wins": int((f1_delta > 0).sum()),
            "mean_dataset_average_precision": float(ap_delta.mean()),
            "mean_corrupted_parent_exposure": float(exposure_delta.mean()),
        },
        "scope": "Three-dataset feasibility screen; not confirmatory research evidence.",
    }
    return summary, decision


def _report(summary: pd.DataFrame, decision: dict[str, Any], digest: str) -> str:
    display = summary.copy()
    for column in ("f1", "gmean", "average_precision", "corrupted_parent_exposure"):
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    delta = decision["soft_minus_nados"]
    return (
        "# Soft-NADOS feasibility pilot\n\n"
        f"Design SHA-256: `{digest}`. This is a small feasibility screen, not final evidence.\n\n"
        + display.to_markdown(index=False)
        + "\n\n## Frozen decision\n\n"
        f"Status: **{decision['status']}**. Soft-NADOS minus NADOS: mean dataset F1 "
        f"{delta['mean_dataset_f1']:+.4f}, F1 wins {delta['dataset_f1_wins']}/3, "
        f"mean AP {delta['mean_dataset_average_precision']:+.4f}, and mean corrupted-parent "
        f"exposure {delta['mean_corrupted_parent_exposure']:+.4f}.\n"
    )


def run(output: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    design = config["scientific_design"]
    rows: list[dict[str, Any]] = []
    for name, (X, y) in datasets().items():
        rows.extend(evaluate_dataset(name, X, y, design))
    scores = pd.DataFrame(rows)
    summary, decision = summarize(scores, design)
    digest = design_hash(config)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "fold_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    payload = {"protocol_id": config["protocol_id"], "design_sha256": digest, **decision}
    (output / "decision.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(_report(summary, decision, digest), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check_only:
        print(f"Design {design_hash(config)}: 3 datasets x 5 folds x 4 methods = 60 fits")
        return 0
    decision = run(args.output, args.config)
    print(f"Results: {args.output}")
    print(f"Decision: {decision['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
