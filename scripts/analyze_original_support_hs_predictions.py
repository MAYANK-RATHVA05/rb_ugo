"""Exploratory diagnostics of saved HS pilot test scores; never fit a model.

Threshold sweeps describe held-out precision/recall curves retrospectively.
They are not validation-selected thresholds, deployment rules or new headline
performance. The original advancement decision is read and preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
B = 'B_SMOTE_STANDARD_HS'
C = 'C_SMOTE_ORIGINAL_HS'
# Diagnostic floors fixed before inspecting the new analysis results.
PRECISION_FLOORS = (.25, .50, .75, .90)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_metrics(y, scores, groups=None):
    y, scores = np.asarray(y), np.asarray(scores)
    pred = scores >= .5
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': tp / (tp + fp) if tp + fp else 0.,
        'precision_defined': bool(tp + fp),
        'recall': tp / (tp + fn),
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        'average_precision': float(average_precision_score(y, scores)),
        'false_positive_rate': fp / (fp + tn),
        'predicted_positive_fraction': float(pred.mean()),
        'rare_recall': float(np.mean(pred[groups == 2])) if groups is not None else np.nan,
    }


def operating_points(y, scores, groups=None):
    """All deterministic, nonempty predicted-positive sets; keep score ties whole."""
    y, scores = np.asarray(y), np.asarray(scores)
    order = np.argsort(-scores, kind='stable')
    sorted_y, sorted_scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y) - 1]
    tp = np.cumsum(sorted_y == 1)[ends]
    total = ends + 1
    fp = total - tp
    rare_recall = np.full(len(ends), np.nan)
    if groups is not None:
        rare = np.asarray(groups) == 2
        rare_recall = np.cumsum(rare[order])[ends] / np.sum(rare)
    return pd.DataFrame({
        'threshold': sorted_scores[ends], 'precision': tp / total,
        'recall': tp / np.sum(y == 1), 'false_positive_rate': fp / np.sum(y == 0),
        'rare_recall': rare_recall, 'tp': tp, 'fp': fp,
    })


def recall_at_precision_floor(points, floor):
    """Exploratory upper envelope at a shared minimum precision.

    No interpolation, tie splitting, randomization, empty-prediction fake
    precision or extrapolation. Unattainable floors remain explicit missing.
    Tie-break maximum recall by highest precision, then highest threshold.
    """
    eligible = points[points.precision >= floor]
    if eligible.empty:
        return {'feasible': False, 'threshold': np.nan, 'precision': np.nan,
                'recall': np.nan, 'false_positive_rate': np.nan, 'rare_recall': np.nan}
    best = eligible.sort_values(['recall', 'precision', 'threshold'], ascending=False).iloc[0]
    return {'feasible': True, **{k: float(best[k]) for k in
            ['threshold', 'precision', 'recall', 'false_positive_rate', 'rare_recall']}}


def validate_artifacts(directory, published):
    directory, published = Path(directory), Path(published)
    provenance = json.loads((published / 'provenance.json').read_text())
    missing = [name for name in provenance['raw_artifacts'] if not (directory / 'cases' / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing saved prediction/case artifacts: ' + ', '.join(missing))
    observed = {p.name for p in (directory / 'cases').iterdir() if p.suffix in ['.json', '.npz']}
    if observed != set(provenance['raw_artifacts']):
        raise ValueError('Unexpected artifacts in the frozen case directory.')
    for name, expected in provenance['raw_artifacts'].items():
        if digest(directory / 'cases' / name) != expected:
            raise ValueError('Saved artifact checksum mismatch: ' + name)
    for path in sorted((directory / 'cases').glob('*.json')):
        record = json.loads(path.read_text())
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as vectors:
            required = ['test_y', B, C]
            if record['dataset'] == 'synthetic_local_concentration':
                required.append('test_groups')
            absent = [key for key in required if key not in vectors]
            if absent:
                raise FileNotFoundError(f'{path.stem}: missing saved vectors {absent}')
            y = vectors['test_y']
            if set(np.unique(y)) != {0, 1}:
                raise ValueError('Both binary test classes are required.')
            for arm in [B, C]:
                p = vectors[arm]
                if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise ValueError('Invalid saved test probability vector.')
    decision_before = digest(published / 'decision.json')
    if digest(directory / 'decision.json') != decision_before:
        raise ValueError('Raw and published original advancement decisions differ.')
    return decision_before


def summarize_paired(frame, metrics):
    output = []
    for setting, block in frame.groupby('setting', sort=False):
        for metric in metrics:
            b, c = block['B_' + metric], block['C_' + metric]
            valid = b.notna() & c.notna()
            delta = c[valid] - b[valid]
            if not len(delta):
                continue
            n = len(delta)
            sd = float(delta.std(ddof=1)) if n > 1 else np.nan
            half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
            output.append({'setting': setting, 'metric': metric, 'n_pairs': n,
                           'B_mean': float(b[valid].mean()), 'C_mean': float(c[valid].mean()),
                           'mean_C_minus_B': float(delta.mean()), 'paired_sd': sd,
                           'descriptive_95_low': float(delta.mean() - half),
                           'descriptive_95_high': float(delta.mean() + half),
                           'wins': int((delta > 1e-12).sum()),
                           'ties': int((abs(delta) <= 1e-12).sum()),
                           'losses': int((delta < -1e-12).sum())})
    return pd.DataFrame(output)


def analyze(directory, published):
    directory, published = Path(directory), Path(published)
    decision_before = validate_artifacts(directory, published)
    pairs, matched, parameter_rows = [], [], []
    metrics = ['precision', 'recall', 'f1', 'average_precision', 'false_positive_rate',
               'rare_recall', 'predicted_positive_fraction']
    for path in sorted((directory / 'cases').glob('*.json')):
        record = json.loads(path.read_text())
        name = path.stem
        setting = record['dataset'] if record['concentration'] is None else f"synthetic_c{record['concentration']:g}"
        identity = {'case_id': name, 'setting': setting, 'dataset': record['dataset'],
                    'concentration': record['concentration'], 'seed': record['seed']}
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            y = z['test_y']
            groups = z['test_groups'] if 'test_groups' in z else None
            scores = {'B': z[B], 'C': z[C]}
            m = {arm: fixed_metrics(y, p, groups) for arm, p in scores.items()}
            row = dict(identity)
            for arm in ['B', 'C']:
                row.update({arm + '_' + key: value for key, value in m[arm].items()})
                saved_arm = B if arm == 'B' else C
                saved_row = next(r for r in record['rows'] if r['arm'] == saved_arm)
                if (abs(saved_row['average_precision'] - m[arm]['average_precision']) > 1e-12
                        or abs(saved_row['minority_f1'] - m[arm]['f1']) > 1e-12):
                    raise ValueError('Saved headline score failed recomputation.')
                parameters = saved_row['selected_parameters']
                row[arm + '_relative_lambda'] = parameters['relative_lambda']
                row[arm + '_lambda'] = parameters['lambda']
                parameter_rows.append({**identity, 'arm': arm, **parameters,
                                       'validation_AP': saved_row['validation_AP'],
                                       'validation_F1': saved_row['validation_F1'],
                                       **{k: m[arm][k] for k in metrics}})
            for metric in metrics:
                row['delta_' + metric] = m['C'][metric] - m['B'][metric]
            pb, pc = scores['B'] >= .5, scores['C'] >= .5
            row.update(
                gained_true_positives=int(np.sum(~pb & pc & (y == 1))),
                lost_true_positives=int(np.sum(pb & ~pc & (y == 1))),
                gained_false_positives=int(np.sum(~pb & pc & (y == 0))),
                lost_false_positives=int(np.sum(pb & ~pc & (y == 0))),
                negative_mean_probability_shift=float(np.mean(scores['C'][y == 0] - scores['B'][y == 0])),
                positive_mean_probability_shift=float(np.mean(scores['C'][y == 1] - scores['B'][y == 1])),
                identical_probabilities=bool(np.array_equal(scores['B'], scores['C'])),
                identical_decisions=bool(np.array_equal(pb, pc)),
                probability_rank_spearman=float(spearmanr(scores['B'], scores['C']).statistic)
                    if len(np.unique(scores['B'])) > 1 and len(np.unique(scores['C'])) > 1 else np.nan,
            )
            if groups is not None:
                row['gained_rare_true_positives'] = int(np.sum(~pb & pc & (groups == 2)))
                row['lost_rare_true_positives'] = int(np.sum(pb & ~pc & (groups == 2)))
            pairs.append(row)
            curves = {arm: operating_points(y, p, groups) for arm, p in scores.items()}
            targets = [('fixed_floor', float(floor), str(floor)) for floor in PRECISION_FLOORS]
            for arm in ['B', 'C']:
                if m[arm]['precision_defined']:
                    targets.append((arm + '_precision_at_0.5', m[arm]['precision'], arm))
            for kind, floor, label in targets:
                result = dict(identity, comparison=kind, floor_label=label, precision_floor=floor,
                              interpretation='exploratory_test_PR_envelope_not_deployment_threshold')
                for arm in ['B', 'C']:
                    at_floor = recall_at_precision_floor(curves[arm], floor)
                    result.update({arm + '_' + key: value for key, value in at_floor.items()})
                both = result['B_feasible'] and result['C_feasible']
                result['both_feasible'] = both
                for metric in ['recall', 'precision', 'rare_recall', 'false_positive_rate']:
                    result['delta_' + metric] = result['C_' + metric] - result['B_' + metric] if both else np.nan
                matched.append(result)
    pair_frame = pd.DataFrame(pairs)
    matched_frame = pd.DataFrame(matched)
    summary = summarize_paired(pair_frame, metrics)
    matched_summaries = []
    coverage = []
    for (kind, label), block in matched_frame.groupby(['comparison', 'floor_label']):
        for setting, setting_block in block.groupby('setting', sort=False):
            paired_block = setting_block[setting_block.both_feasible]
            coverage.append({'comparison': kind, 'floor_label': label, 'setting': setting,
                             'total_seeds': len(setting_block),
                             'B_feasible': int(setting_block.B_feasible.sum()),
                             'C_feasible': int(setting_block.C_feasible.sum()),
                             'paired_feasible': len(paired_block)})
            if len(paired_block):
                estimates = summarize_paired(paired_block, ['recall', 'precision', 'rare_recall', 'false_positive_rate'])
                estimates['comparison'] = kind
                estimates['floor_label'] = label
                estimates['B_feasible_pairs'] = int(setting_block.B_feasible.sum())
                estimates['C_feasible_pairs'] = int(setting_block.C_feasible.sum())
                matched_summaries.append(estimates)
    matched_summary = pd.concat(matched_summaries, ignore_index=True)
    verification = {
        'status': 'passed', 'cases': len(pair_frame), 'prediction_vectors': 2 * len(pair_frame),
        'classifier_refits': 0, 'preprocessing_refits': 0, 'new_synthetic_rows': 0,
        'required_test_artifacts_missing': [],
        'unavailable_artifacts': ['validation probability vectors; cannot calibrate thresholds on validation'],
        'original_decision_sha256': decision_before,
        'original_decision': json.loads((published / 'decision.json').read_text())['status'],
        'fixed_threshold': .5, 'exploratory_precision_floors': list(PRECISION_FLOORS),
        'matched_precision_policy': 'maximum recall at a common minimum precision; tied scores remain whole; no interpolation; report actual precision and infeasible cases',
        'thresholds_for_deployment_selected': False,
    }
    if digest(published / 'decision.json') != decision_before:
        raise ValueError('Original decision changed during analysis.')
    return {'per_seed': pair_frame, 'summary': summary,
            'matched_precision_per_seed': matched_frame, 'matched_precision_summary': matched_summary,
            'matched_precision_coverage': pd.DataFrame(coverage),
            'selected_parameters': pd.DataFrame(parameter_rows)}, verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/original-support-hs-v1')
    parser.add_argument('--published', type=Path, default=ROOT / 'docs/research/original_support_hs/results_v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/research/original_support_hs/exploratory_v1')
    args = parser.parse_args()
    tables, verification = analyze(args.input, args.published)
    if args.output.exists():
        raise ValueError('Use a fresh analysis output directory to preserve saved evidence.')
    args.output.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(args.output / (name + '.csv'), index=False)
    (args.output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
