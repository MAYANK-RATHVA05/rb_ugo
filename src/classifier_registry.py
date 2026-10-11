"""Fourteen UGO classifier identities, source settings, and local availability.

Constructors marked python_approximation are usable smoke models, not claims
of R/Python author-package equivalence. Others fail explicitly by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src._validation import binary_labels, numeric_matrix


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    groups: tuple[str, ...]
    paper_settings: str
    implementation_status: str
    implementation_detail: str


_GENERAL = "general"
CLASSIFIERS: tuple[ClassifierSpec, ...] = (
    ClassifierSpec("BART", (_GENERAL, "nonlinear", "tree", "bayesian"), "ntree=20; Bayesian prediction averages 100 posterior samples", "unavailable", "No verified Bayesian additive regression tree package/prior/posterior interface."),
    ClassifierSpec("BNNET", (_GENERAL, "nonlinear", "neural", "bayesian"), "2 layers; 20 nodes; dropout=.25; warm-up=100; RMSprop; patience=20; 100 posterior samples", "unavailable", "No verified Bayesian neural-network/prior and posterior probability implementation."),
    ClassifierSpec("C5.0", (_GENERAL, "nonlinear", "tree"), "CF=.25; earlyStopping=TRUE", "unavailable", "No native C5.0 implementation; sklearn DecisionTree is a different method."),
    ClassifierSpec("CART", (_GENERAL, "nonlinear", "tree"), "cp=.01; minsplit=20", "python_approximation", "sklearn ccp_alpha=.01 and min_samples_split=20; pruning semantics differ from R cp."),
    ClassifierSpec("GKNB", (_GENERAL, "nonlinear", "density"), "unbiased cross-validation bandwidth", "unavailable", "Kernel NB estimator, kernel and bandwidth selection settings are unspecified."),
    ClassifierSpec("GNB", (_GENERAL, "nonlinear", "density"), "package defaults", "python_approximation", "sklearn GaussianNB defaults; author package/version unknown."),
    ClassifierSpec("KNN", (_GENERAL, "nonlinear", "neighborhood"), "k=5", "python_approximation", "sklearn Euclidean/uniform 5-NN; source scaling/tie conventions unknown."),
    ClassifierSpec("LDA", (_GENERAL, "linear", "density"), "package defaults", "python_approximation", "sklearn LDA SVD defaults; author package/version unknown."),
    ClassifierSpec("LogReg", (_GENERAL, "linear"), "maxit=1000", "python_approximation", "sklearn LogisticRegression max_iter=1000 with default L2 regularization; R defaults may differ."),
    ClassifierSpec("NNET", (_GENERAL, "nonlinear", "neural"), "2 layers; 20 nodes; dropout=.25; warm-up=100; RMSprop; patience=20", "unavailable", "sklearn MLP lacks the stated dropout and RMSprop configuration; native package unspecified."),
    ClassifierSpec("QDA", (_GENERAL, "density"), "package defaults", "python_approximation", "sklearn QDA defaults; covariance and regularization conventions may differ."),
    ClassifierSpec("RF", (_GENERAL, "nonlinear", "tree"), "500 trees; mtry=sqrt(p); unlimited depth; min.node.size=1; min.bucket=1", "python_approximation", "sklearn RF with 500 trees, max_features=sqrt, unlimited depth and leaf size 1; other package semantics may differ."),
    ClassifierSpec("SVMLinear", (_GENERAL, "linear"), "C=.95", "python_approximation", "sklearn SVC linear kernel, C=.95, probability=True; internal training-only calibration and package defaults differ."),
    ClassifierSpec("SVMRadial", (_GENERAL, "nonlinear"), "C=.95; sigma=.5", "python_approximation", "sklearn SVC rbf gamma=.5 is a declared sigma mapping; probability calibration/package defaults may differ."),
)


class UnavailableClassifierError(ValueError):
    """A named paper classifier needs a verified implementation or settings."""


def get_classifier_spec(name: str) -> ClassifierSpec:
    for spec in CLASSIFIERS:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown UGO classifier {name!r}.")


def make_classifier(name: str, *, seed: int = 42) -> Any:
    """Construct a local training-only smoke model for a supported identity."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    spec = get_classifier_spec(name)
    if spec.implementation_status == "unavailable":
        raise UnavailableClassifierError(f"{name} is unavailable: {spec.implementation_detail}")
    constructors = {
        "CART": lambda: DecisionTreeClassifier(ccp_alpha=0.01, min_samples_split=20, random_state=seed),
        "GNB": lambda: GaussianNB(),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "LogReg": lambda: LogisticRegression(max_iter=1000, random_state=seed),
        "QDA": lambda: QuadraticDiscriminantAnalysis(),
        "RF": lambda: RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                              max_depth=None, min_samples_leaf=1,
                                              random_state=seed, n_jobs=1),
        "SVMLinear": lambda: SVC(kernel="linear", C=0.95, probability=True, random_state=seed),
        "SVMRadial": lambda: SVC(kernel="rbf", C=0.95, gamma=0.5,
                                probability=True, random_state=seed),
    }
    return constructors[name]()


@dataclass(frozen=True)
class ProbabilitySmoke:
    """Fitting-data-only software check; no evaluation or performance metric."""

    name: str
    rows: int
    features: int
    classes: tuple[Any, Any]
    status: str


def smoke_classifier_on_fitting_data(name: str, X_fit: Any, y_fit: Any,
                                     *, seed: int = 42) -> ProbabilitySmoke:
    """Fit a fresh model and check its probability contract on fitting rows."""
    X = numeric_matrix(X_fit)
    y, classes = binary_labels(y_fit, len(X))
    model = clone(make_classifier(name, seed=seed))
    model.fit(X.copy(), y.copy())
    probabilities = np.asarray(model.predict_proba(X.copy()))
    if (probabilities.shape != (len(X), 2) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)
            or not np.array_equal(model.classes_, classes)):
        raise ValueError(f"{name} violated the fitting-data probability contract.")
    return ProbabilitySmoke(name, len(X), X.shape[1], tuple(classes),
                            get_classifier_spec(name).implementation_status)
