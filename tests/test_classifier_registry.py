"""Fourteen names are recorded; supported substitutes fit on training rows only."""
import numpy as np
import pytest
from sklearn.base import clone

from src.classifier_registry import (CLASSIFIERS, UnavailableClassifierError,
                                     get_classifier_spec, make_classifier,
                                     smoke_classifier_on_fitting_data)
from src.simulations import make_linear_simulation


def test_all_fourteen_paper_identities_and_eight_groups_are_recorded():
    assert len(CLASSIFIERS) == len({spec.name for spec in CLASSIFIERS}) == 14
    assert {group for spec in CLASSIFIERS for group in spec.groups} == {
        "general", "linear", "nonlinear", "neural", "tree", "density",
        "neighborhood", "bayesian",
    }
    assert {spec.name for spec in CLASSIFIERS if spec.implementation_status == "unavailable"} == {
        "BART", "BNNET", "C5.0", "GKNB", "NNET",
    }
    assert get_classifier_spec("RF").paper_settings.startswith("500 trees")


@pytest.mark.parametrize("name", [spec.name for spec in CLASSIFIERS
                                  if spec.implementation_status == "python_approximation"])
def test_available_classifier_probability_contract_on_fitting_only(name):
    simulation = make_linear_simulation(n_majority=50, n_minority=10)
    before = simulation.X.copy(), simulation.y.copy()
    result = smoke_classifier_on_fitting_data(name, simulation.X, simulation.y)
    assert result.rows == 60 and result.features == 2
    assert result.classes == (0, 1)
    assert result.status == "python_approximation"
    np.testing.assert_array_equal(simulation.X, before[0])
    np.testing.assert_array_equal(simulation.y, before[1])
    assert clone(make_classifier(name)) is not None


@pytest.mark.parametrize("name", ["BART", "BNNET", "C5.0", "GKNB", "NNET"])
def test_unavailable_methods_are_never_silently_replaced(name):
    with pytest.raises(UnavailableClassifierError, match=name):
        make_classifier(name)


def test_classifier_seed_and_unknown_name_fail():
    with pytest.raises(ValueError, match="seed"):
        make_classifier("RF", seed=None)
    with pytest.raises(ValueError, match="Unknown"):
        make_classifier("not-a-paper-method")
