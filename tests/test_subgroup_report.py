import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from scripts.report_subgroup_diagnostic import independent_ap


@pytest.mark.parametrize("scores", [[.1, .6, .4, .9], [.5, .5, .5, .5], [.4, .8, .4, .8]])
def test_independent_ap_handles_ties_at_one_threshold(scores):
    y = np.array([0, 0, 1, 1])
    assert independent_ap(y, scores) == pytest.approx(average_precision_score(y, scores))


def test_independent_ap_rejects_missing_positive_class():
    with pytest.raises(ValueError):
        independent_ap(np.array([0, 0]), np.array([.2, .7]))
