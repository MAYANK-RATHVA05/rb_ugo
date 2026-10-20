import numpy as np
import pytest

from scripts.analyze_original_support_hs_predictions import (
    fixed_metrics, operating_points, recall_at_precision_floor, validate_artifacts,
)


def test_hand_precision_recall_fpr_and_rare_recall():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([.8, .3, .7, .6, .1])
    groups = np.array([2, 1, 0, 0, 0])
    metrics = fixed_metrics(y, scores, groups)
    assert metrics['tp'] == 1 and metrics['fp'] == 2
    assert metrics['precision'] == pytest.approx(1 / 3)
    assert metrics['recall'] == .5
    assert metrics['f1'] == .4
    assert metrics['average_precision'] == .75
    assert metrics['false_positive_rate'] == pytest.approx(2 / 3)
    assert metrics['rare_recall'] == 1


def test_precision_sweep_keeps_ties_whole_and_infeasible_floor_explicit():
    y = np.array([1, 0, 1, 0])
    scores = np.array([.8, .8, .3, .1])
    points = operating_points(y, scores)
    assert len(points) == 3
    assert list(points.tp) == [1, 2, 2]
    assert list(points.fp) == [1, 1, 2]
    assert not recall_at_precision_floor(points, .75)['feasible']
    result = recall_at_precision_floor(points, .6)
    assert result['feasible'] and result['recall'] == 1
    assert result['precision'] == pytest.approx(2 / 3)
    assert result['threshold'] == .3


def test_sweep_uses_max_recall_not_nearest_precision_or_empty_prediction():
    y = np.array([1, 0, 1, 0, 1, 0])
    points = operating_points(y, np.array([.9, .8, .7, .6, .5, .4]))
    at_floor = recall_at_precision_floor(points, .6)
    assert at_floor['recall'] == 1
    assert at_floor['precision'] == .6
    assert at_floor['threshold'] == .5


def test_no_positive_predictions_has_explicit_undefined_precision():
    metrics = fixed_metrics(np.array([1, 0]), np.array([.1, .2]))
    assert metrics['precision'] == 0 and not metrics['precision_defined']
    assert metrics['recall'] == 0 and metrics['false_positive_rate'] == 0


def test_missing_saved_artifact_reported_before_analysis(tmp_path):
    import json
    published = tmp_path / 'published'; published.mkdir()
    (published / 'provenance.json').write_text(json.dumps({'raw_artifacts': {'case.npz': 'hash'}}))
    with pytest.raises(FileNotFoundError, match='case.npz'):
        validate_artifacts(tmp_path / 'raw', published)
