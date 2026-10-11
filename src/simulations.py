"""Declared reconstructions of UGO §4.1.1's two synthetic datasets.

The paper does not provide generation seeds or all distribution/noise choices.
These outputs are reproducible software inputs, not the author's exact rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from src._validation import finite_real, positive_integer


@dataclass(frozen=True)
class SimulationDataset:
    """Original features, pre-noise labels, observed labels and source choices."""

    name: str
    X: NDArray[np.float64]
    y: NDArray[np.int64]
    pre_noise_y: NDArray[np.int64]
    relabeled_indices: NDArray[np.int64]
    majority_label: int
    minority_label: int
    metadata: dict[str, Any]


def _validate_common(seed: int, noise_fraction: float) -> float:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    fraction = finite_real(noise_fraction, "noise_fraction")
    if not 0 <= fraction <= 1:
        raise ValueError("noise_fraction must lie in [0, 1].")
    return fraction


def _relabel_majority(
    y: NDArray[np.int64], majority_label: int, minority_label: int,
    noise_fraction: float, rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    minority_before = int(np.sum(y == minority_label))
    # Paper says relabel n_P * 0.3 majority examples. Round-half-up is an
    # explicit integer convention; n_P refers to the pre-noise minority count.
    amount = int(np.floor(noise_fraction * minority_before + 0.5))
    majority_indices = np.flatnonzero(y == majority_label)
    if amount > len(majority_indices):
        raise ValueError("Requested majority-to-minority relabel count exceeds the majority pool.")
    selected = np.sort(rng.choice(majority_indices, size=amount, replace=False))
    observed = y.copy()
    observed[selected] = minority_label
    return observed, selected


def make_linear_simulation(
    *, n_majority: int = 1000, n_minority: int = 50,
    noise_fraction: float = 0.3, normal_second_parameter: str = "sd",
    seed: int = 42,
) -> SimulationDataset:
    """2D independent normals; convert 30% of pre-noise minority count.

    The article writes N(0,3) and N(6,2) without defining whether its
    second parameter denotes variance or standard deviation. 'sd' is the
    declared default; 'variance' is also available for sensitivity checks.
    The default pre-noise counts 1000/50 are inferred from Figure 7's
    post-noise 985/65 under 15 relabels, not explicitly given by the text.
    """
    n_majority = positive_integer(n_majority, "n_majority")
    n_minority = positive_integer(n_minority, "n_minority")
    fraction = _validate_common(seed, noise_fraction)
    if normal_second_parameter not in {"sd", "variance"}:
        raise ValueError("normal_second_parameter must be 'sd' or 'variance'.")
    majority_scale = 3.0 if normal_second_parameter == "sd" else np.sqrt(3.0)
    minority_scale = 2.0 if normal_second_parameter == "sd" else np.sqrt(2.0)
    rng = np.random.default_rng(seed)
    X = np.vstack((rng.normal(0, majority_scale, size=(n_majority, 2)),
                   rng.normal(6, minority_scale, size=(n_minority, 2))))
    pre_noise_y = np.concatenate((np.zeros(n_majority, dtype=np.int64),
                                  np.ones(n_minority, dtype=np.int64)))
    y, relabeled = _relabel_majority(pre_noise_y, 0, 1, fraction, rng)
    if np.sum(y == 0) <= np.sum(y == 1):
        raise ValueError("Linear simulation no longer has the stated majority/minority ordering.")
    return SimulationDataset(
        "linear", X, y, pre_noise_y, relabeled, 0, 1,
        {"source": "UGO §4.1.1 p7 and Figure 7 p10", "seed": int(seed),
         "noise_fraction_of_pre_noise_minority": fraction,
         "noise_rounding": "floor_fraction_plus_one_half",
         "noise_direction": "majority_to_minority",
         "normal_second_parameter": normal_second_parameter,
         "coordinate_independence": "assumed_iid",
         "n_majority_before_noise": n_majority,
         "n_minority_before_noise": n_minority,
         "figure_counts_match_by_declared_inference": bool(n_majority == 1000 and n_minority == 50
                                                      and fraction == 0.3)})


def make_nonlinear_simulation(
    *, n_samples: int = 1000, noise_fraction: float = 0.3,
    seed: int = 42,
) -> SimulationDataset:
    """Apply the article's Z/logistic/rounding equation, then relabel noise."""
    n_samples = positive_integer(n_samples, "n_samples")
    fraction = _validate_common(seed, noise_fraction)
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, 2))
    x1, x2 = X[:, 0], X[:, 1]
    z = np.sin(x1) ** 2 + np.cos(x2) + np.sin(x1) * np.cos(x2)
    pre_noise_y = np.rint(expit(z)).astype(np.int64)
    if len(np.unique(pre_noise_y)) != 2:
        raise ValueError("The sampled nonlinear dataset has fewer than two pre-noise classes; use another seed or size.")
    if np.sum(pre_noise_y == 1) <= np.sum(pre_noise_y == 0):
        raise ValueError("Nonlinear sample does not have the paper's Y=1 majority ordering.")
    # The article explicitly identifies Y=1 as majority and Y=0 as minority.
    y, relabeled = _relabel_majority(pre_noise_y, 1, 0, fraction, rng)
    if np.sum(y == 1) <= np.sum(y == 0):
        raise ValueError("Nonlinear simulation no longer has the stated majority/minority ordering.")
    return SimulationDataset(
        "nonlinear", X, y, pre_noise_y, relabeled, 1, 0,
        {"source": "UGO §4.1.1 pp7–8 and Figure 7 p10", "seed": int(seed),
         "n_samples": n_samples, "z_equation": "sin(x1)^2 + cos(x2) + sin(x1)*cos(x2)",
         "label_equation": "round(1/(1+exp(-z)))",
         "noise_fraction_of_pre_noise_minority": fraction,
         "noise_rounding": "floor_fraction_plus_one_half",
         "noise_direction": "majority_to_minority",
         "coordinate_independence": "assumed_iid",
         "figure_counts_are_source_example_not_asserted_output": [915, 85]})
