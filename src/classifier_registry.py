"""Fourteen UGO classifier identities, source settings, and local availability.

Constructors marked python_approximation are usable local models, not claims
of R/Python author-package equivalence. Optional backends fail explicitly.
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
from src.classifier_backends import BARTClassifier, C50Classifier
from src.classifier_errors import UnavailableClassifierError
from src.kernel_naive_bayes import GaussianKernelNB
from src.neural_classifiers import RMSpropNeuralClassifier


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    groups: tuple[str, ...]
    paper_settings: str
    implementation_status: str
    implementation_detail: str


_GENERAL = "general"
CLASSIFIERS: tuple[ClassifierSpec, ...] = (
    ClassifierSpec("BART", (_GENERAL, "nonlinear", "tree", "bayesian"), "ntree=20; Bayesian prediction averages 100 posterior samples", "python_approximation", "stochtree 0.2.1 probit BART with 20 trees, 100 burn-in and 100 retained draws; unstated priors use package defaults except fixed outcome variance."),
    ClassifierSpec("BNNET", (_GENERAL, "nonlinear", "neural", "bayesian"), "2 layers; 20 nodes; dropout=.25; warm-up=100; RMSprop; patience=20; 100 posterior samples", "python_approximation", "NumPy two-hidden-layer mean-field Gaussian variational network; ReLU, Gaussian prior, optimizer scalars, minibatches and validation split are declared choices."),
    ClassifierSpec("C5.0", (_GENERAL, "nonlinear", "tree"), "CF=.25; earlyStopping=TRUE", "python_approximation", "Pinned CRAN C50 0.2.0 native learner in tree mode, CF=.25, early stopping enabled and one trial; package version/trials were unstated."),
    ClassifierSpec("CART", (_GENERAL, "nonlinear", "tree"), "cp=.01; minsplit=20", "python_approximation", "sklearn ccp_alpha=.01 and min_samples_split=20; pruning semantics differ from R cp."),
    ClassifierSpec("GKNB", (_GENERAL, "nonlinear", "density"), "unbiased cross-validation bandwidth", "python_approximation", "Gaussian product-kernel NB with per-class/per-feature binned unbiased least-squares CV bandwidth; paper kernel and implementation are unstated."),
    ClassifierSpec("GNB", (_GENERAL, "nonlinear", "density"), "package defaults", "python_approximation", "sklearn GaussianNB defaults; author package/version unknown."),
    ClassifierSpec("KNN", (_GENERAL, "nonlinear", "neighborhood"), "k=5", "python_approximation", "sklearn Euclidean/uniform 5-NN; source scaling/tie conventions unknown."),
    ClassifierSpec("LDA", (_GENERAL, "linear", "density"), "package defaults", "python_approximation", "sklearn LDA SVD defaults; author package/version unknown."),
    ClassifierSpec("LogReg", (_GENERAL, "linear"), "maxit=1000", "python_approximation", "sklearn LogisticRegression max_iter=1000 with default L2 regularization; R defaults may differ."),
    ClassifierSpec("NNET", (_GENERAL, "nonlinear", "neural"), "2 layers; 20 nodes; dropout=.25; warm-up=100; RMSprop; patience=20", "python_approximation", "NumPy two-hidden-layer ReLU network with dropout and RMSprop; learning rate, minibatches, validation split and epoch cap are declared choices."),
    ClassifierSpec("QDA", (_GENERAL, "density"), "package defaults", "python_approximation", "sklearn QDA defaults; covariance and regularization conventions may differ."),
    ClassifierSpec("RF", (_GENERAL, "nonlinear", "tree"), "500 trees; mtry=sqrt(p); unlimited depth; min.node.size=1; min.bucket=1", "python_approximation", "sklearn RF with 500 trees, max_features=sqrt, unlimited depth and leaf size 1; other package semantics may differ."),
    ClassifierSpec("SVMLinear", (_GENERAL, "linear"), "C=.95", "python_approximation", "sklearn SVC linear kernel, C=.95, probability=True; internal training-only calibration and package defaults differ."),
    ClassifierSpec("SVMRadial", (_GENERAL, "nonlinear"), "C=.95; sigma=.5", "python_approximation", "sklearn SVC rbf gamma=.5 is a declared sigma mapping; probability calibration/package defaults may differ."),
)


def get_classifier_spec(name: str) -> ClassifierSpec:
    for spec in CLASSIFIERS:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown UGO classifier {name!r}.")


def make_classifier(name: str, *, seed: int = 42) -> Any:
    """Construct a fresh local approximation for a supported identity."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    spec = get_classifier_spec(name)
    constructors = {
        "BART": lambda: BARTClassifier(num_trees=20, posterior_samples=100,
                                        burnin=100, random_state=seed),
        "BNNET": lambda: RMSpropNeuralClassifier(
            hidden_units=20, dropout=.25, warmup_epochs=100, patience=20,
            bayesian=True, posterior_samples=100, random_state=seed),
        "C5.0": lambda: C50Classifier(confidence_factor=.25, early_stopping=True,
                                       random_state=seed),
        "CART": lambda: DecisionTreeClassifier(ccp_alpha=0.01, min_samples_split=20, random_state=seed),
        "GNB": lambda: GaussianNB(),
        "GKNB": lambda: GaussianKernelNB(),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "LogReg": lambda: LogisticRegression(max_iter=1000, random_state=seed),
        "NNET": lambda: RMSpropNeuralClassifier(
            hidden_units=20, dropout=.25, warmup_epochs=100, patience=20,
            bayesian=False, random_state=seed),
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
