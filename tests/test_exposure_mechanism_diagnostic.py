"""Evaluator and replication checks; artificial examples are not study evidence."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.exposure_mechanism_diagnostic as diagnostic
from scripts.report_exposure_mechanism import decision


def test_oracle_mixture_uses_actual_pool_prior_and_stable_joint_components():
    x = np.array([[0., 0.], [1000., 1000.], [-1000., 0.]])
    p = diagnostic.oracle_components(x, separation=3., prior=.2)
    expected = .2 * np.exp(-1.125) / (.8 + .2 * np.exp(-1.125))
    assert p[0].sum() == pytest.approx(expected)
    assert p[0, 0] == pytest.approx(p[0, 1])
    assert np.isfinite(p).all() and np.all(p.sum(axis=1) <= 1 + 1e-12)
    low_prior = diagnostic.oracle_components(x[:1], 3., .02)
    assert low_prior.sum() < p[0].sum()
    np.testing.assert_array_equal(diagnostic.oracle_components(np.empty((0, 2)), 3., .2).shape, (0, 2))


def test_minority_mode_coverage_uses_fractional_mass_and_weights():
    components = np.array([[.2, .1], [.3, .4]])
    low = diagnostic.coverage(components, np.array([2., 1.]))
    np.testing.assert_allclose(low['expected_mass'], [.7, .6])
    assert low['coverage'] == 0.
    high = diagnostic.coverage(components, np.array([4., 1.]))
    np.testing.assert_allclose(high['expected_mass'], [1.1, .8])
    assert high['coverage'] == .5


def test_continuous_binding_is_distinct_from_flooring_loss():
    bank = dict(candidates=[np.zeros((8, 2))], probabilities=np.array([.8]), capacity=8, budget=.5)
    record = diagnostic.bank_activity(dict(banks=[bank]))[0]
    assert record['binding'] and record['restrictive'] and record['integer_count_reduced']
    assert record['continuous_exposure'] == pytest.approx(.5)
    assert record['rounding_loss'] == pytest.approx(.5)
    # A loose count-limited optimum can lose a row during floating-point flooring
    # without being an exposure-binding decision.
    bank['budget'] = 8.
    loose = diagnostic.bank_activity(dict(banks=[bank]))[0]
    assert not loose['binding'] and not loose['restrictive']


def case(seed, budget_value, control_value, active=True, failed=False):
    return dict(seed=seed, records=[
        dict(method='budget', status='success', active=active,
             metrics=dict(mcc=budget_value, oracle_endpoint_error=budget_value if active else None,
                          selected_endpoint_error=budget_value if active else None)),
        dict(method='count_matched', status='failed' if failed else 'success', active=active,
             metrics=dict(mcc=control_value, oracle_endpoint_error=control_value if active else None,
                          selected_endpoint_error=control_value if active else None))])


def test_pairing_retains_inactive_classification_and_excludes_missing_errors():
    cases = [case(42, .6, .5), case(43, .4, .4, active=False), case(44, .9, .1, failed=True)]
    score = diagnostic.paired_summary(cases, 'count_matched', 'mcc')
    assert score['seeds'] == [42, 43]
    assert score['difference']['mean'] == pytest.approx(.05)
    assert score['shared_successes'] == 2 and score['budget_wins'] == score['ties'] == 1
    risk = diagnostic.paired_summary(cases, 'count_matched', 'oracle_endpoint_error')
    assert risk['seeds'] == [42] and risk['difference']['n'] == 1
    assert risk['budget_losses'] == 1  # Lower error is better.
    assert diagnostic.paired_summary(cases, 'count_matched', 'selected_endpoint_error')['budget_losses'] == 1
    conditional = diagnostic.paired_summary(cases, 'count_matched', 'mcc', active_only=True)
    assert conditional['seeds'] == [42]


def test_summary_preserves_nulls_and_rejects_nonfinite_successes():
    assert diagnostic.describe([None, None])['mean'] is None
    result = diagnostic.describe([1., 3., None])
    assert result['n'] == 2 and result['mean'] == 2. and result['se'] == pytest.approx(1.)
    with pytest.raises(ValueError, match='Nonfinite'):
        diagnostic.describe([np.nan])


def test_independent_generation_and_training_interface_exclude_oracles(monkeypatch):
    config = json.loads((Path(__file__).parents[1] / 'configs/exposure_mechanism_v1.json').read_text())
    cell = dict(separation=3., labeled_minority_count=8, unlabeled_minority_prevalence=.02)
    # Test fixtures use seed7, not a trial/search of any registered dataset seed.
    first, evaluator = diagnostic.generate_case(config, 0, cell, 7)
    repeat, _ = diagnostic.generate_case(config, 0, cell, 7)
    np.testing.assert_array_equal(first.L, repeat.L)
    assert np.sum(first.labels == 1) == 8 and np.sum(first.labels == 0) == 200
    assert first.U.shape == (400, 2) and evaluator.test.shape == (4000, 2)
    assert not np.array_equal(first.U, first.L[:len(first.U)])
    assert list(inspect.signature(diagnostic.fit_case).parameters) == ['inputs', 'fixed', 'algorithm_seed']
    def forbidden(*args, **kwargs):
        raise AssertionError('Oracle consulted during fitting')
    monkeypatch.setattr(diagnostic, 'oracle_components', forbidden)
    frozen = diagnostic.fit_case(first, config['fixed'])
    assert all(v['status'] == 'success' for v in frozen['variants'].values())
    assert frozen['decision_sha256']
    # Hidden values are not members of the training input object.
    assert set(vars(first)) == {'L', 'labels', 'U'}


def test_no_added_rows_have_null_error_and_zero_mode_coverage():
    # Small fixture, not a preregistered trial. All decisions precede evaluation.
    L = np.array([[-3., 0.], [-3.1, .1], [-3.2, -.1], [-3.3, .2],
                  [3., -1.5], [3.1, -1.4], [3.2, 1.5]])
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    training = diagnostic.TrainingInputs(L, labels, np.array([[3., 0.], [-3., .2]]))
    frozen = diagnostic.fit_case(training, dict(threshold=.9, exposure_rate=.025, endpoint_cap=5))
    evaluation = diagnostic.EvaluationInputs(training.U, np.array([1, 0]), np.array([0, -1]),
                                             L, labels, np.array([-1] * 4 + [0, 0, 1]),
                                             np.array([-1] * 4 + [0, 0, 1]), 3., .2)
    result = diagnostic.evaluate_case(frozen, evaluation)
    nores = next(r for r in result['records'] if r['method'] == 'nores')
    assert nores['status'] == 'success'
    assert nores['metrics']['oracle_added_minority_error'] is None
    assert nores['metrics']['realized_lineage_error'] is None
    assert nores['metrics']['mode_coverage'] == 0.
    assert all(v['count'] > 0 for v in nores['metrics']['mode_recall'])
    changed = deepcopy(evaluation); changed.hidden_labels = 1 - evaluation.hidden_labels
    assert diagnostic.evaluate_case(frozen, changed)['decision_sha256'] == result['decision_sha256']


def test_existing_output_and_changed_prototype_are_refused(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    config = root / 'configs/exposure_mechanism_v1.json'
    out = tmp_path / 'existing'; out.mkdir(); (out / 'keep').write_text('preserve')
    with pytest.raises(ValueError, match='fresh output'):
        diagnostic.run(config, out)
    assert (out / 'keep').read_text() == 'preserve'
    monkeypatch.setattr(diagnostic, 'sha256', lambda p: 'wrong')
    with pytest.raises(ValueError, match='Protected prototype'):
        diagnostic.run(config, tmp_path / 'new')


def decision_fixture(active=8, primary=-.003, outcome=.001, paired=8):
    config = dict(decision_rule=dict(minimum_paired_active_seeds=5,
                                    mechanism_practical_error_difference=.01,
                                    downstream_practical_difference=.005))
    pairs = [dict(control=control, active_only=True, metric=metric,
                  difference=dict(n=paired, mean=(primary if metric == 'oracle_endpoint_error' else outcome)
                                  if paired else None))
             for control in ['threshold_matched', 'weighted_pl']
             for metric in ['oracle_endpoint_error', 'mcc', 'gmean', 'ap']]
    summary = dict(cells=[dict(cell_index=0, methods=dict(budget=dict(active=active)), paired=pairs)])
    return summary, config


def test_preregistered_decision_rejects_only_current_grid_advantage_when_controls_match():
    observed = decision(*decision_fixture())
    assert observed['outcome'] == 'reject_current_added_value_claim_on_this_grid'
    assert observed['adequately_active_cells'] == [0]
    stronger = decision(*decision_fixture(primary=-.02, outcome=.01))
    assert stronger['outcome'] == 'inconclusive_or_revise'


def test_inactive_or_failed_pairs_cannot_establish_matching_controls():
    assert decision(*decision_fixture(active=4))['adequately_active_cells'] == []
    missing = decision(*decision_fixture(paired=0))
    assert missing['outcome'] == 'inconclusive_or_revise'
    assert not missing['controls']['weighted_pl']['matches_every_adequately_active_cell']
