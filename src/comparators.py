"""Fitting-only constructors for the UGO paper's standalone comparators."""
from __future__ import annotations

from typing import Any

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator

from src._validation import binary_labels, numeric_matrix
from src.generators import RandomWalkOversampler
from src.leao import LEAOOversampler
from src.mcsmote import MCSMOTEOversampler
from src.bfch_smote import BFCHSMOTE


class NoResampling(BaseEstimator):
    """Explicit original-fitting-data reference condition (NORES)."""

    def fit_resample(self, X: Any, y: Any):
        points = numeric_matrix(X)
        labels, _ = binary_labels(y, len(points))
        return points, labels


def make_comparator(name: str, *, random_state: int = 42, **kwargs: Any):
    """Construct a fresh standalone comparator; pass only fitting folds to it.

    RUS appears in UGO Section 4.2 but is omitted from its main tables.
    imbalanced-learn 0.12.4 supplies RUS, ROS and SMOTE here.
    """
    if name == "nores":
        return NoResampling()
    if name == "rus":
        return RandomUnderSampler(random_state=random_state, **kwargs)
    if name == "ros":
        return RandomOverSampler(random_state=random_state, **kwargs)
    if name == "smote":
        return SMOTE(random_state=random_state, **kwargs)
    if name == "rwo":
        return RandomWalkOversampler(random_state=random_state,
                                    allow_provisional=True, **kwargs)
    if name == "mcsmote":
        return MCSMOTEOversampler(random_state=random_state,
                                  allow_provisional=True, **kwargs)
    if name == "leao":
        return LEAOOversampler(random_state=random_state,
                               allow_provisional=True, **kwargs)
    if name == "bfch_smote":
        return BFCHSMOTE(random_state=random_state, **kwargs)
    raise ValueError("Unknown comparator. Choose nores, rus, ros, smote, rwo, mcsmote, leao or bfch_smote.")
