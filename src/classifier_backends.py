"""Pinned BART and process-isolated native C5.0 classifier adapters."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer
from src.classifier_errors import UnavailableClassifierError


ROOT = Path(__file__).resolve().parents[1]
C50_RUNTIME = ROOT / ".runtime" / "c50"


def _stochtree_model():
    try:
        from stochtree import BARTModel
        return BARTModel
    except (ImportError, OSError) as exc:
        raise UnavailableClassifierError(
            "BART requires stochtree==0.2.1 from requirements-classifiers.txt."
        ) from exc


class BARTClassifier(ClassifierMixin, BaseEstimator):
    """Binary probit BART; probabilities average retained posterior draws."""

    def __init__(self, num_trees: int = 20, posterior_samples: int = 100,
                 burnin: int = 100, random_state: int = 42):
        self.num_trees = num_trees
        self.posterior_samples = posterior_samples
        self.burnin = burnin
        self.random_state = random_state

    def fit(self, X: Any, y: Any):
        for attr in ("model_", "classes_", "n_features_in_"):
            self.__dict__.pop(attr, None)
        Xv = numeric_matrix(X)
        yv, classes = binary_labels(y, len(Xv))
        trees = positive_integer(self.num_trees, "num_trees")
        samples = positive_integer(self.posterior_samples, "posterior_samples")
        burnin = positive_integer(self.burnin, "burnin")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral) \
                or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be in [0, 2**32).")
        seed = int(self.random_state)
        model = _stochtree_model()()
        model.sample(
            Xv, (yv == classes[1]).astype(float), num_gfr=0,
            num_burnin=burnin, num_mcmc=samples,
            general_params={"random_seed": seed, "probit_outcome_model": True,
                            "sample_sigma2_global": False},
            mean_forest_params={"num_trees": trees, "sample_sigma2_leaf": False},
        )
        self.model_, self.classes_, self.n_features_in_ = model, classes, Xv.shape[1]
        self.diagnostics_ = {"backend": "stochtree", "num_trees": trees,
                             "burnin": burnin, "posterior_samples": samples,
                             "probability_aggregation": "mean of probit posterior draws"}
        return self

    def posterior_probabilities(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        Xv = numeric_matrix(X)
        if Xv.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature width differs from fitting data.")
        draws = np.asarray(self.model_.predict(
            Xv, type="posterior", scale="probability", terms="y_hat"), dtype=float)
        if draws.ndim == 1 and self.posterior_samples == 1:
            draws = draws[:, None]
        if draws.shape == (self.posterior_samples, len(Xv)):
            draws = draws.T
        if draws.shape != (len(Xv), self.posterior_samples) or not np.isfinite(draws).all():
            raise ValueError("BART backend returned an invalid posterior probability array.")
        return draws

    def predict_proba(self, X: Any) -> np.ndarray:
        p1 = self.posterior_probabilities(X).mean(axis=1)
        return np.column_stack((1.0 - p1, p1))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def c50_backend_identity() -> dict[str, Any]:
    executable, manifest = C50_RUNTIME / "c50_bridge", C50_RUNTIME / "backend.json"
    if not executable.is_file() or not manifest.is_file():
        raise UnavailableClassifierError(
            "C5.0 native backend is absent; run `.venv/bin/python scripts/setup_c50.py`."
        )
    record = json.loads(manifest.read_text())
    actual = hashlib.sha256(executable.read_bytes()).hexdigest()
    if record.get("executable_sha256") != actual:
        raise UnavailableClassifierError("C5.0 native backend checksum differs from its build manifest.")
    return record


def _c50_names(width: int) -> str:
    return "c0,c1.\n" + "".join(f"x{i}: continuous.\n" for i in range(width))


def _c50_rows(X: np.ndarray, targets: np.ndarray | None = None) -> str:
    rows = []
    for i, row in enumerate(X):
        suffix = "?" if targets is None else f"c{int(targets[i])}"
        rows.append(",".join([*(format(float(v), ".17g") for v in row), suffix]))
    return "\n".join(rows) + "\n"


class C50Classifier(ClassifierMixin, BaseEstimator):
    """CRAN C50 0.2.0 tree mode through a short-lived native process."""

    def __init__(self, confidence_factor: float = 0.25, early_stopping: bool = True,
                 random_state: int = 42, timeout_seconds: int = 300):
        self.confidence_factor = confidence_factor
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.timeout_seconds = timeout_seconds

    def fit(self, X: Any, y: Any):
        for attr in ("tree_text_", "classes_", "n_features_in_", "backend_identity_"):
            self.__dict__.pop(attr, None)
        Xv = numeric_matrix(X)
        yv, classes = binary_labels(y, len(Xv))
        cf = finite_real(self.confidence_factor, "confidence_factor")
        if not 0 < cf < 1:
            raise ValueError("confidence_factor must lie strictly between zero and one.")
        if self.early_stopping is not True:
            raise ValueError("This paper adapter requires early_stopping=True.")
        timeout = positive_integer(self.timeout_seconds, "timeout_seconds")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)) \
                or not 0 <= int(self.random_state) < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32).")
        identity = c50_backend_identity()
        native_seed = int(self.random_state) % (2**30)
        with tempfile.TemporaryDirectory(prefix="ugo-c50-") as directory:
            tmp = Path(directory)
            names, data, tree = tmp / "model.names", tmp / "model.data", tmp / "model.tree"
            names.write_text(_c50_names(Xv.shape[1]))
            data.write_text(_c50_rows(Xv, (yv == classes[1]).astype(int)))
            result = subprocess.run(
                [str(C50_RUNTIME / "c50_bridge"), "fit", str(names), str(data),
                 str(tree), str(native_seed), format(cf, ".17g")],
                capture_output=True, text=True, timeout=timeout)
            if result.returncode or not tree.is_file():
                raise RuntimeError(f"Native C5.0 fit failed: {result.stderr.strip()[:1000]}")
            tree_text = tree.read_text()
        self.tree_text_, self.classes_, self.n_features_in_ = tree_text, classes, Xv.shape[1]
        self.backend_identity_ = identity
        self.diagnostics_ = {"backend": "CRAN C50 0.2.0", "confidence_factor": cf,
                             "early_stopping": True, "trials": 1,
                             "native_seed": native_seed, "process_isolated": True}
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "tree_text_")
        Xv = numeric_matrix(X)
        if Xv.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature width differs from fitting data.")
        if c50_backend_identity() != self.backend_identity_:
            raise UnavailableClassifierError("C5.0 backend identity changed after fitting.")
        with tempfile.TemporaryDirectory(prefix="ugo-c50-") as directory:
            tmp = Path(directory)
            names, cases, tree, output = (tmp / name for name in
                                           ("model.names", "model.cases", "model.tree", "prob.csv"))
            names.write_text(_c50_names(Xv.shape[1]))
            cases.write_text(_c50_rows(Xv))
            tree.write_text(self.tree_text_)
            result = subprocess.run(
                [str(C50_RUNTIME / "c50_bridge"), "predict", str(names), str(cases),
                 str(tree), str(output), str(len(Xv))],
                capture_output=True, text=True, timeout=positive_integer(self.timeout_seconds, "timeout_seconds"))
            if result.returncode or not output.is_file():
                raise RuntimeError(f"Native C5.0 prediction failed: {result.stderr.strip()[:1000]}")
            probabilities = np.loadtxt(output, delimiter=",", ndmin=2)
        if probabilities.shape != (len(Xv), 2) or not np.isfinite(probabilities).all():
            raise ValueError("Native C5.0 returned an invalid probability array.")
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
