import pickle

import numpy as np
import pytest
from sklearn.base import clone

import src.classifier_backends as backends
from src.classifier_backends import BARTClassifier, C50Classifier
from src.classifier_errors import UnavailableClassifierError
from src.kernel_naive_bayes import GaussianKernelNB, _ucv_objective
from src.neural_classifiers import RMSpropNeuralClassifier


@pytest.fixture
def separated_data():
    rng = np.random.default_rng(4)
    X = np.r_[rng.normal(-2, .35, (24, 2)), rng.normal(2, .35, (24, 2))]
    y = np.array(["major"] * 24 + ["minor"] * 24)
    return X, y


def assert_probability_contract(model, X, classes=("major", "minor")):
    probability = model.predict_proba(X)
    assert probability.shape == (len(X), 2)
    assert np.isfinite(probability).all() and (probability >= 0).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-10)
    np.testing.assert_array_equal(model.classes_, classes)


def test_bart_averages_retained_probability_draws_and_is_reproducible(separated_data):
    X, y = separated_data
    first = BARTClassifier(num_trees=4, posterior_samples=8, burnin=5, random_state=7).fit(X, y)
    second = clone(first).fit(X, y)
    draws = first.posterior_probabilities(X[:5])
    assert draws.shape == (5, 8) and np.any(np.std(draws, axis=1) > 0)
    np.testing.assert_allclose(first.predict_proba(X[:5])[:, 1], draws.mean(axis=1))
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))


def test_optional_bart_backend_absence_is_explicit(monkeypatch, separated_data):
    def absent():
        raise UnavailableClassifierError("BART backend absent")
    monkeypatch.setattr(backends, "_stochtree_model", absent)
    with pytest.raises(UnavailableClassifierError, match="absent"):
        BARTClassifier(num_trees=2, posterior_samples=2, burnin=2).fit(*separated_data)


def test_native_c50_probability_and_pickled_model(separated_data):
    X, y = separated_data
    model = C50Classifier(random_state=9).fit(X, y)
    assert_probability_contract(model, X)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    assert np.mean(model.predict(X) == y) > .9


def test_kernel_nb_ucv_formula_and_probability_contract(separated_data):
    centers = np.array([-1., 0., 2.]); counts = np.array([2., 1., 2.]); h = .7
    expanded = np.repeat(centers, counts.astype(int))
    differences = expanded[:, None] - expanded[None, :]
    direct = (np.exp(-differences**2/(4*h*h)).sum()/(25*h*np.sqrt(4*np.pi))
              - 2*(np.exp(-differences**2/(2*h*h)).sum()-5)/(20*h*np.sqrt(2*np.pi)))
    assert _ucv_objective(np.log(h), centers, counts, 5) == pytest.approx(direct)
    model = GaussianKernelNB(n_bins=16).fit(*separated_data)
    assert_probability_contract(model, separated_data[0])
    assert (model.bandwidths_ > 0).all()


@pytest.mark.parametrize("bayesian", [False, True])
def test_neural_models_learn_and_are_seed_reproducible(separated_data, bayesian):
    X, y = separated_data
    settings = dict(hidden_units=8, warmup_epochs=10, patience=5, max_epochs=80,
                    batch_size=16, posterior_samples=12, random_state=11,
                    bayesian=bayesian)
    first = RMSpropNeuralClassifier(**settings).fit(X, y)
    second = RMSpropNeuralClassifier(**settings).fit(X, y)
    assert_probability_contract(first, X)
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))
    assert np.mean(first.predict(X) == y) > .9
    if bayesian:
        draws = first.posterior_probabilities(X[:4])
        assert draws.shape == (4, 12) and np.any(np.std(draws, axis=1) > 0)
        np.testing.assert_allclose(first.predict_proba(X[:4])[:, 1], draws.mean(axis=1))


def test_failed_refit_removes_old_fitted_state(separated_data):
    model = RMSpropNeuralClassifier(warmup_epochs=2, patience=2, max_epochs=3).fit(*separated_data)
    with pytest.raises(ValueError, match="Exactly two"):
        model.fit(separated_data[0], np.zeros(len(separated_data[0])))
    with pytest.raises(Exception):
        model.predict_proba(separated_data[0])
