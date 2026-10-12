"""Provisional prompt-based UGO-ROS/SMOTE, pending full-paper verification.

This is not RB-UGO. No validation/test data, performance scores, or risk budgets
are used. Read docs/ugo_reproduction.md before research use.
"""
from __future__ import annotations

from dataclasses import asdict
from numbers import Integral
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer
from src.noise_detection import (
    nearest_indices, noise_final_detect_and_relocate, noise_pre_detect,
)
from src.stopping import generation_cap, welch_stopping
from src.uncertainty import modified_margin_uncertainty


ASSUMPTIONS = (
    "Full paper unavailable; implementation is provisional, not verified reproduction.",
    "Cap is floor(phi * number_of_classes * original_largest_count - original_total).",
    "Uncertainty and seed pools are original pre-clean fitting samples only.",
    "Noisy original samples remain in estimator fitting data; they cannot be seeds.",
    "Selection count is ceil(lambda_select * eligible_selected_class_count), at least one.",
    "Batch size is ceil(lambda_step * current_selected_class_count), truncated at cap.",
    "Binary neighbor-vote ties retain assigned labels; distance and score ties use row order.",
    "SMOTE neighbors are original pre-clean same-class rows, excluding each seed by index.",
    "Final votes use all original fitting rows; relocation uses original pre-clean same-class rows.",
    "Undefined Welch tests stop via engineering safeguard, not failure to reject.",
)


def informative_seed_indices(
    scores: Any, labels: Any, selected_class: Any, noise_mask: Any,
    lambda_select: float = 0.5,
) -> NDArray[np.int64]:
    """Highest-MMA clean original seeds; ceil fraction, stable index ties."""
    uncertainty = np.asarray(scores)
    y = np.asarray(labels)
    noise = np.asarray(noise_mask)
    if uncertainty.ndim != 1 or y.shape != uncertainty.shape or noise.shape != y.shape:
        raise ValueError("Scores, labels, and noise mask must be aligned vectors.")
    if noise.dtype.kind != "b" or not np.isfinite(uncertainty).all():
        raise ValueError("Noise mask must be boolean and scores finite.")
    lambda_select = finite_real(lambda_select, "lambda_select")
    if not 0 < lambda_select <= 1:
        raise ValueError("lambda_select must lie in (0, 1].")
    eligible = np.flatnonzero((y == selected_class) & ~noise)
    if not len(eligible):
        return eligible
    count = max(1, int(np.ceil(lambda_select * len(eligible))))
    order = np.argsort(-uncertainty[eligible], kind="stable")
    return eligible[order[:count]]


class UGO(BaseEstimator):
    """Reusable binary oversampler with an explicitly provisional status.

    estimator must be sklearn-cloneable with fit/predict_proba. Each iteration
    clones it, fits a copied augmented fitting set, and measures MMA on clean
    original rows. Only fitting data are accepted by fit_resample(X, y).

    Use variant='ros' or 'smote'. Defaults match the supplied prompt; unresolved
    details are listed in ASSUMPTIONS. allow_provisional=True is required until
    the full paper is checked. max_iter=100 is an engineering bound, not a paper
    stopping rule. k_smote=5 is a provisional neighborhood choice. Random states
    unset on estimator/nested estimators are filled from random_state; explicit
    estimator random states are preserved and reported.
    """

    def __init__(
        self, estimator: Any, variant: str = "ros", lambda_select: float = 0.5,
        lambda_step: float = 0.1, phi: float = 2.5, k_pre: int = 5,
        k_after: int = 1, alpha: float = 0.05, random_state: int = 42,
        max_iter: int = 100, k_smote: int = 5, allow_provisional: bool = False,
    ) -> None:
        self.estimator = estimator
        self.variant = variant
        self.lambda_select = lambda_select
        self.lambda_step = lambda_step
        self.phi = phi
        self.k_pre = k_pre
        self.k_after = k_after
        self.alpha = alpha
        self.random_state = random_state
        self.max_iter = max_iter
        self.k_smote = k_smote
        self.allow_provisional = allow_provisional

    def _validate_parameters(self) -> Any:
        if not isinstance(self.allow_provisional, bool) or not self.allow_provisional:
            raise ValueError("Paper verification is unresolved; set allow_provisional=True only for a provisional run.")
        if self.variant not in {"ros", "smote"}:
            raise ValueError("variant must be 'ros' or 'smote'.")
        for name in ["lambda_select", "lambda_step"]:
            value = finite_real(getattr(self, name), name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and lie in (0, 1].")
        for name in ["k_pre", "k_after", "k_smote", "max_iter"]:
            positive_integer(getattr(self, name), name)
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral) or not 0 <= self.random_state < 2 ** 32:
            raise ValueError("random_state must be an integer in [0, 2**32).")
        if not 0 < finite_real(self.alpha, "alpha") < 1:
            raise ValueError("alpha must lie in (0, 1).")
        for method in ["fit", "predict_proba"]:
            if not callable(getattr(self.estimator, method, None)):
                raise ValueError(f"estimator must implement {method}().")
        try:
            return clone(self.estimator)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("estimator must be sklearn-cloneable.") from exc

    def _generate(
        self, X: NDArray, y: NDArray, seeds: NDArray, selected: Any,
        pre_noise: NDArray, batch_size: int, rng: np.random.Generator,
    ) -> tuple[NDArray | None, dict[str, Any]]:
        draws = rng.choice(seeds, size=batch_size, replace=True)
        info: dict[str, Any] = {"drawn_seed_indices": draws.tolist()}
        if self.variant == "ros":
            return X[draws].copy(), info
        eligible = np.flatnonzero((y == selected) & ~pre_noise)
        if len(eligible) < 2:
            return None, {"effective_k_smote": 0}
        k = min(self.k_smote, len(eligible) - 1)
        neighborhoods = nearest_indices(X[eligible], X[eligible], k, exclude_self=True)
        seed_to_local = {int(index): local for local, index in enumerate(eligible)}
        partners = np.array([
            eligible[rng.choice(neighborhoods[seed_to_local[int(seed)]])]
            for seed in draws
        ])
        steps = rng.random((batch_size, 1))
        # Convex form avoids overflow in X_neighbor-X_seed for large values.
        generated = (1 - steps) * X[draws] + steps * X[partners]
        if not np.isfinite(generated).all():
            raise ValueError("Synthetic interpolation produced nonfinite values.")
        info.update({"neighbor_indices": partners.tolist(), "interpolation_steps": steps[:, 0].tolist(),
                     "effective_k_smote": k})
        return generated, info

    def fit_resample(self, X: Any, y: Any) -> tuple[NDArray, NDArray]:
        """Return copied augmented fitting data and expose diagnostics_.

        No validation/test arguments exist. Already balanced counts do not bypass
        the uncertainty rule. Undefined Welch tests, no eligible seeds, or too few
        clean SMOTE neighbors produce diagnosed no-ops/termination, never a silent
        switch to ordinary SMOTE or ROS. Incomplete runs return added batches only.
        """
        # Remove fitted state from an earlier successful run before validating a
        # new input, so failures cannot expose stale diagnostics as current results.
        for attr in ["diagnostics_", "estimator_", "pre_noise_mask_", "classes_", "n_features_in_"]:
            if hasattr(self, attr):
                delattr(self, attr)
        template = self._validate_parameters()
        original_X = numeric_matrix(X)
        original_y, classes = binary_labels(y, len(original_X))
        counts = np.array([(original_y == label).sum() for label in classes])
        cap = generation_cap(counts, self.phi)
        warnings.warn("Provisional UGO: full-paper alignment and cap remain unverified.",
                      UserWarning, stacklevel=2)
        rng = np.random.default_rng(self.random_state)
        pre_noise = noise_pre_detect(original_X, original_y, self.k_pre)
        self.pre_noise_mask_ = pre_noise.copy()
        self.classes_ = classes.copy()
        self.n_features_in_ = original_X.shape[1]
        current_X, current_y = original_X.copy(), original_y.copy()
        parameters = {name: self._scalar(getattr(self, name)) for name in [
            "variant", "lambda_select", "lambda_step", "phi", "k_pre", "k_after",
            "alpha", "random_state", "max_iter", "k_smote", "allow_provisional",
        ]}
        estimator_seeds = {key: value for key, value in template.get_params(deep=True).items()
                           if key == "random_state" or key.endswith("__random_state")}
        filled_seeds = {key: self.random_state for key, value in estimator_seeds.items() if value is None}
        if filled_seeds:
            template.set_params(**filled_seeds)
        parameters["estimator_random_states"] = {**estimator_seeds, **filled_seeds}
        notes = list(ASSUMPTIONS)
        if self.k_pre > len(original_X) - 1:
            notes.append("k_pre clamped to original sample count minus one.")
        if self.k_after > len(original_X):
            notes.append("k_after clamped to original sample count.")
        self.diagnostics_ = {
            "reproduction_status": "provisional_prompt_based",
            "original_class_counts": self._counts(original_y, classes),
            "generation_cap": cap, "cap_interpretation": "provisional_total_augmented_budget",
            "pre_detected_noisy_samples": int(pre_noise.sum()),
            "effective_k_pre": min(self.k_pre, len(original_X) - 1),
            "effective_parameters": parameters, "warnings": notes,
            "history": [], "number_of_iterations": 0,
            "number_of_synthetic_samples": 0, "generated_noisy_samples_relocated": 0,
            "stopping_reason": None,
        }
        eligible = ~pre_noise
        reason = "max_iterations_safeguard"
        for iteration in range(1, self.max_iter + 1):
            remaining = cap - (len(current_X) - len(original_X))
            if remaining <= 0:
                reason = "provisional_generation_cap"
                break
            if not eligible.any():
                reason = "no_eligible_non_noisy_seeds"
                break
            if any(not np.any(eligible & (original_y == label)) for label in classes):
                reason = "class_without_eligible_non_noisy_samples"
                break
            model = clone(template)
            # A defensive copy prevents an estimator from mutating stored data.
            model.fit(current_X.copy(), current_y.copy())
            if not hasattr(model, "classes_"):
                raise ValueError("Fitted estimator must expose classes_ for probability ordering.")
            uncertainty = modified_margin_uncertainty(
                model.predict_proba(original_X[eligible].copy()), original_y[eligible], model.classes_,
            )
            if set(model.classes_) != set(classes):
                raise ValueError("Estimator class labels differ from the supplied fitting labels.")
            self.estimator_ = model  # Last scoring estimator, not a final evaluation model.
            scores = np.zeros(len(original_X))
            scores[eligible] = uncertainty
            groups = [scores[eligible & (original_y == label)] for label in classes]
            means = [float(group.mean()) for group in groups]
            test = welch_stopping(*groups, self.alpha)
            record = {"iteration": iteration,
                      "class_mean_uncertainty": [{"label": self._scalar(label), "mean": mean}
                                                  for label, mean in zip(classes, means)],
                      "welch": asdict(test), "selected_class": None,
                      "seed_indices": [], "generated_count": 0, "relocated_count": 0}
            self.diagnostics_["history"].append(record)
            self.diagnostics_["number_of_iterations"] += 1
            if not test.defined:
                reason = f"undefined_welch_safeguard:{test.reason}"
                break
            if test.stop:
                reason = "welch_fail_to_reject"
                break
            selected = classes[int(np.argmax(means))]
            record["selected_class"] = self._scalar(selected)
            seeds = informative_seed_indices(scores, original_y, selected, pre_noise, self.lambda_select)
            record["seed_indices"] = seeds.tolist()
            if not len(seeds):
                reason = "no_eligible_non_noisy_seeds"
                break
            batch_size = min(remaining, max(1, int(np.ceil(self.lambda_step * (current_y == selected).sum()))))
            generated, generation_info = self._generate(
                original_X, original_y, seeds, selected, pre_noise, batch_size, rng,
            )
            record["generation"] = generation_info
            if generated is None:
                reason = "insufficient_same_class_neighbors"
                break
            corrected, relocation = noise_final_detect_and_relocate(
                generated, selected, original_X, original_y, self.k_after,
                reference_noise_mask=pre_noise,
            )
            record["generated_count"] = len(corrected)
            record["relocated_count"] = relocation["relocated_count"]
            record["relocation_reference_indices"] = relocation["relocation_reference_indices"].tolist()
            record["effective_k_after"] = relocation["effective_k_after"]
            self.diagnostics_["generated_noisy_samples_relocated"] += relocation["relocated_count"]
            current_X = np.vstack((current_X, corrected))
            current_y = np.concatenate((current_y, np.full(len(corrected), selected, dtype=original_y.dtype)))
            if len(current_X) - len(original_X) == cap:
                reason = "provisional_generation_cap"
                break
        self.diagnostics_["stopping_reason"] = reason
        self.diagnostics_["final_class_counts"] = self._counts(current_y, classes)
        self.diagnostics_["number_of_synthetic_samples"] = len(current_X) - len(original_X)
        return current_X, current_y

    @staticmethod
    def _scalar(label: Any) -> Any:
        return label.item() if isinstance(label, np.generic) else label

    @classmethod
    def _counts(cls, y: NDArray, classes: NDArray) -> list[dict[str, Any]]:
        return [{"label": cls._scalar(label), "count": int((y == label).sum())} for label in classes]
