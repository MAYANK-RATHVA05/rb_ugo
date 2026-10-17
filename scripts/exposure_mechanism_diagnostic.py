"""Preregistered Gaussian mechanism diagnostic around the unchanged prototype.

Training accepts only observed L, feature-only U and fixed method parameters.
Oracle posteriors, hidden labels, modes and test data are evaluator-only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

from src.exploratory_exposure import allocate, prepare, replay
from src.ugo import UGO


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('parent', 'budget', 'unconstrained', 'count_matched', 'threshold_matched', 'weighted_pl', 'nores')
ERROR_METRICS = ('realized_lineage_error', 'oracle_endpoint_error', 'oracle_u_child_error',
                 'oracle_added_minority_error')
METRICS = ('mcc', 'gmean', 'ap', *ERROR_METRICS, 'estimated_endpoint_error',
           'selected_endpoint_error', 'u_child_mode_coverage', 'mode_coverage', 'selected_endpoints',
           'effective_endpoints', 'u_mass', 'incremental_seconds')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class TrainingInputs:
    L: np.ndarray
    labels: np.ndarray
    U: np.ndarray


@dataclass
class EvaluationInputs:
    U: np.ndarray
    hidden_labels: np.ndarray
    pool_modes: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    test_modes: np.ndarray
    labeled_modes: np.ndarray
    separation: float
    pool_prior: float


def draw_population(rng, count, separation, prior):
    labels = (rng.random(count) < prior).astype(int)
    modes = np.where(labels == 1, rng.integers(0, 2, count), -1)
    means = np.column_stack((np.where(labels == 1, separation / 2, -separation / 2),
                             np.where(labels == 1, np.where(modes == 0, -1.5, 1.5), 0.)))
    return rng.normal(size=(count, 2)) + means, labels, modes


def generate_case(config, cell_index, cell, seed):
    """Simulation-design utility; its labels/probabilities never enter a guide."""
    distribution = config['distribution']; separation = cell['separation']
    rng = np.random.default_rng(np.random.SeedSequence([seed, cell_index, 0]))
    labels = np.r_[np.zeros(distribution['labeled_majority_count'], dtype=int),
                   np.ones(cell['labeled_minority_count'], dtype=int)]
    modes = np.where(labels == 1, rng.integers(0, 2, len(labels)), -1)
    means = np.column_stack((np.where(labels == 1, separation / 2, -separation / 2),
                             np.where(labels == 1, np.where(modes == 0, -1.5, 1.5), 0.)))
    features = rng.normal(size=(len(labels), 2)) + means
    order = rng.permutation(len(labels))
    U, hidden, pool_modes = draw_population(
        np.random.default_rng(np.random.SeedSequence([seed, cell_index, 1])),
        distribution['unlabeled_count'], separation, cell['unlabeled_minority_prevalence'])
    test, truth, test_modes = draw_population(
        np.random.default_rng(np.random.SeedSequence([seed, cell_index, 2])),
        distribution['test_count'], separation, distribution['test_minority_prevalence'])
    return (TrainingInputs(features[order], labels[order], U),
            EvaluationInputs(U.copy(), hidden, pool_modes, test, truth, test_modes,
                             modes[order], separation, cell['unlabeled_minority_prevalence']))


def oracle_components(features, separation, prior):
    """Evaluator-only joint minority-mode posterior under a specified prior.

    Returned columns sum to the minority posterior. The omitted shared Gaussian
    normalizing constant cancels, with logsumexp preserving tail stability.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not 0 < prior < 1:
        raise ValueError('Oracle requires two-dimensional features and a valid prior')
    centers = np.array([[-separation / 2, 0.], [separation / 2, -1.5], [separation / 2, 1.5]])
    log_density = -.5 * ((x[:, None, :] - centers) ** 2).sum(axis=2)
    log_joint = log_density + np.log([1 - prior, prior / 2, prior / 2])
    return np.exp(log_joint[:, 1:] - logsumexp(log_joint, axis=1)[:, None])


def bank_activity(prepared):
    """Non-oracle feasibility/activity diagnostics; does not change allocation."""
    records = []
    for bank in prepared['banks']:
        limits = np.array([len(c) for c in bank['candidates']], dtype=int)
        if not limits.sum():
            records.append(dict(usable=False, binding=None, restrictive=None,
                                integer_count_reduced=None, rounding_loss=None))
            continue
        errors = 1 - bank['probabilities']
        constrained = allocate(errors, limits, bank['capacity'], bank['budget'])
        loose = allocate(errors, limits, bank['capacity'], float(bank['capacity']))
        tolerance = 1e-7 * max(1., bank['budget'])
        records.append(dict(usable=True,
                            binding=bool(abs(errors @ constrained.continuous - bank['budget']) <= tolerance),
                            restrictive=bool(errors @ loose.continuous > bank['budget'] + tolerance),
                            integer_count_reduced=bool(constrained.counts.sum() < loose.counts.sum()),
                            rounding_loss=constrained.rounding_loss,
                            continuous_exposure=float(errors @ constrained.continuous),
                            loose_exposure=float(errors @ loose.continuous), budget=bank['budget']))
    return records


def fit_case(inputs: TrainingInputs, fixed, algorithm_seed=42):
    """No evaluator object, simulation prior or oracle is accepted here."""
    started = time.perf_counter()
    scaler = StandardScaler().fit(inputs.L)
    L, U = scaler.transform(inputs.L), scaler.transform(inputs.U)
    preprocess_seconds = time.perf_counter() - started
    classifier = LogisticRegression(max_iter=1000, random_state=algorithm_seed)
    parent = UGO(classifier, variant='smote', random_state=algorithm_seed, allow_provisional=True)
    start = time.perf_counter()
    prepared = None; preparation_error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            prepared = prepare(L, inputs.labels, U, parent, threshold=fixed['threshold'],
                               exposure_rate=fixed['exposure_rate'], endpoint_cap=fixed['endpoint_cap'])
        except Exception as error:
            preparation_error = f'{type(error).__name__}: {error}'
    shared_seconds = time.perf_counter() - start
    variants = {}; activity = []; activity_error = None
    if prepared is not None:
        start = time.perf_counter()
        try:
            activity = bank_activity(prepared)
        except Exception as error:
            activity_error = f'{type(error).__name__}: {error}'
        diagnostic_seconds = time.perf_counter() - start
    else:
        diagnostic_seconds = 0.
    for method in POLICIES:
        start = time.perf_counter()
        try:
            if method == 'nores':
                data = dict(X=L.copy(), y=inputs.labels.copy(), sample_weight=np.ones(len(L)),
                            ledger=[], rounds=[], u_informed_count=0, new_u_vectors=0)
            else:
                if prepared is None:
                    raise RuntimeError(preparation_error)
                data = replay(prepared, method)
            replay_seconds = time.perf_counter() - start
            start = time.perf_counter()
            model = LogisticRegression(max_iter=1000, random_state=algorithm_seed)
            model.fit(data['X'], data['y'], sample_weight=data['sample_weight'])
            variants[method] = dict(status='success', data=data, model=model,
                                    replay_seconds=replay_seconds, fit_seconds=time.perf_counter() - start)
        except Exception as error:
            variants[method] = dict(status='failed', error=f'{type(error).__name__}: {error}',
                                    elapsed_seconds=time.perf_counter() - start)
    # A decision digest is persisted before any evaluation. Times are excluded.
    digest = hashlib.sha256()
    metadata = {}
    for method, value in variants.items():
        metadata[method] = {k: value[k] for k in ['status', 'error'] if k in value}
        if value['status'] == 'success':
            data = value['data']
            for array in [data['X'], data['y'], data['sample_weight'],
                          value['model'].coef_, value['model'].intercept_]:
                digest.update(np.ascontiguousarray(array).tobytes())
            metadata[method].update(ledger=data['ledger'], rounds=data['rounds'])
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    return dict(variants=variants, prepared=prepared, scaler=scaler, activity=activity,
                activity_error=activity_error, decision_sha256=digest.hexdigest(),
                frozen_decisions=metadata, warnings=[str(w.message) for w in caught],
                preprocess_seconds=preprocess_seconds, shared_seconds=shared_seconds,
                diagnostic_seconds=diagnostic_seconds)


def classification(truth, probability, modes):
    predicted = (probability >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    recalls = []
    for mode in range(2):
        mask = (truth == 1) & (modes == mode)
        recalls.append(dict(count=int(mask.sum()), recall=float(predicted[mask].mean()) if mask.any() else None))
    return dict(mcc=float(matthews_corrcoef(truth, predicted)),
                gmean=float(np.sqrt(sensitivity * specificity)) if sensitivity is not None and specificity is not None else None,
                ap=float(average_precision_score(truth, probability)) if np.any(truth == 1) else None,
                mode_recall=recalls)


def coverage(components, weights):
    mass = (components * weights[:, None]).sum(axis=0)
    return dict(expected_mass=mass.tolist(), coverage=float(np.mean(mass >= 1.)))


def evaluate_case(frozen, evaluator: EvaluationInputs):
    """Hidden labels/posteriors are used only after all training is finished."""
    prepared = frozen['prepared']; banks = prepared['banks'] if prepared is not None else []
    eligible = sorted({int(j) for b in banks for j in b['endpoint_ids']})
    usable = sorted({int(j) for b in banks for j, c in zip(b['endpoint_ids'], b['candidates']) if len(c)})
    total_banks = len(banks); activity = frozen['activity']
    usable_banks = [r for r in activity if r['usable']]
    pool_components = oracle_components(evaluator.U, evaluator.separation, evaluator.pool_prior)
    shared = dict(eligible_endpoints=len(eligible), usable_endpoints=len(usable),
                  eligible_realized_error=float(np.mean(evaluator.hidden_labels[eligible] == 0)) if eligible else None,
                  eligible_oracle_error=float(np.mean(1 - pool_components[eligible].sum(axis=1))) if eligible else None,
                  minority_generation_rounds=total_banks, usable_rounds=len(usable_banks),
                  binding_rounds=sum(r['binding'] for r in usable_banks),
                  restrictive_rounds=sum(r['restrictive'] for r in usable_banks),
                  integer_count_reduced_rounds=sum(r['integer_count_reduced'] for r in usable_banks),
                  pool_minority_count=int(evaluator.hidden_labels.sum()),
                  labeled_mode_counts=[int((evaluator.labeled_modes == k).sum()) for k in range(2)],
                  pool_mode_counts=[int((evaluator.pool_modes == k).sum()) for k in range(2)],
                  activity_error=frozen['activity_error'],
                  preprocess_seconds=frozen['preprocess_seconds'], shared_seconds=frozen['shared_seconds'],
                  diagnostic_seconds=frozen['diagnostic_seconds'])
    records = []
    for method, value in frozen['variants'].items():
        if value['status'] != 'success':
            records.append(dict(method=method, **value)); continue
        try:
            data = value['data']; ledger = data['ledger']; model = value['model']
            start = time.perf_counter()
            test = frozen['scaler'].transform(evaluator.test)
            probability = model.predict_proba(test)[:, list(model.classes_).index(1)]
            predict_seconds = time.perf_counter() - start
            metrics = classification(evaluator.test_labels, probability, evaluator.test_modes)
            count = len(ledger)
            ids = np.array([r['endpoint'] for r in ledger], dtype=int)
            unique, multiplicities = np.unique(ids, return_counts=True)
            metrics.update(u_mass=count, selected_endpoints=len(unique),
                           effective_endpoints=float(count ** 2 / np.sum(multiplicities ** 2)) if count else None,
                           selected_usable_fraction=float(len(unique) / len(usable)) if usable else None,
                           realized_lineage_error=float(np.mean(evaluator.hidden_labels[ids] == 0)) if count else None,
                           selected_endpoint_error=float(np.mean(evaluator.hidden_labels[unique] == 0)) if count else None,
                           oracle_endpoint_error=float(np.mean(1 - pool_components[ids].sum(axis=1))) if count else None,
                           estimated_endpoint_error=float(np.mean([1 - r['guide_probability'] for r in ledger])) if count else None,
                           oracle_u_child_error=None, u_child_mode_coverage=None, u_child_mode_mass=None)
            if count and method != 'weighted_pl':
                children = data['X'][[r['output_index'] for r in ledger]]
                components = oracle_components(frozen['scaler'].inverse_transform(children),
                                               evaluator.separation, evaluator.pool_prior)
                child_coverage = coverage(components, np.ones(count))
                metrics.update(oracle_u_child_error=float(np.mean(1 - components.sum(axis=1))),
                               u_child_mode_coverage=child_coverage['coverage'],
                               u_child_mode_mass=child_coverage['expected_mass'])
            first_added = len(prepared['X']) if prepared is not None else len(data['y'])
            mask = np.arange(len(data['y'])) >= first_added
            mask &= data['y'] == 1
            added = (frozen['scaler'].inverse_transform(data['X'][mask]) if mask.any()
                     else np.empty((0, data['X'].shape[1])))
            weights = data['sample_weight'][mask]
            components = oracle_components(added, evaluator.separation, evaluator.pool_prior)
            mode_summary = coverage(components, weights)
            metrics.update(oracle_added_minority_error=float(np.average(1 - components.sum(axis=1), weights=weights)) if len(added) else None,
                           added_minority_mass=float(weights.sum()), mode_coverage=mode_summary['coverage'],
                           mode_expected_mass=mode_summary['expected_mass'],
                           incremental_seconds=value['replay_seconds'] + value['fit_seconds'] + predict_seconds)
            if method in ('parent', 'nores'):
                reason = 'supervised_control'
            elif count:
                reason = 'active'
            elif not banks:
                reason = 'no_minority_generation_round'
            elif not eligible:
                reason = 'no_eligible_endpoint'
            elif not usable:
                reason = 'no_surviving_candidate'
            else:
                reason = 'no_budgeted_integer_allocation'
            records.append(dict(method=method, status='success', inactive_reason=reason,
                                active=bool(count), metrics=metrics,
                                fitting_rows=len(data['y']), fitting_weight=float(data['sample_weight'].sum()),
                                replay_seconds=value['replay_seconds'], fit_seconds=value['fit_seconds'],
                                predict_seconds=predict_seconds,
                                predictions=(probability >= .5).astype(int).tolist(), probabilities=probability.tolist()))
        except Exception as error:
            records.append(dict(method=method, status='failed', error=f'{type(error).__name__}: {error}'))
    return dict(shared=shared, records=records, decision_sha256=frozen['decision_sha256'])


def describe(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    if len(a) and not np.isfinite(a).all():
        raise ValueError('Nonfinite observations cannot become aggregate successes')
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                sd=float(a.std(ddof=1)) if len(a) > 1 else None,
                se=float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None,
                minimum=float(a.min()) if len(a) else None, maximum=float(a.max()) if len(a) else None)


def paired_summary(cases, control, metric, active_only=False):
    differences = []; seeds = []; successes = 0
    for case in cases:
        methods = {r['method']: r for r in case['records']}
        budget, other = methods.get('budget'), methods.get(control)
        if not budget or not other or budget['status'] != 'success' or other['status'] != 'success':
            continue
        successes += 1
        if active_only and not budget['active']:
            continue
        a, b = budget['metrics'].get(metric), other['metrics'].get(metric)
        if a is None or b is None:
            continue
        differences.append(a - b); seeds.append(case['seed'])
    lower_is_better = metric.endswith('error') or metric == 'incremental_seconds'
    direction = -1 if lower_is_better else 1
    oriented = np.array(differences) * direction
    return dict(control=control, metric=metric, active_only=active_only, shared_successes=successes,
                seeds=seeds, difference=describe(differences),
                budget_wins=int(np.sum(oriented > 1e-10)), ties=int(np.sum(np.abs(oriented) <= 1e-10)),
                budget_losses=int(np.sum(oriented < -1e-10)))


def summarize(cases, config, wall_seconds):
    cells = []
    controls = [m for m in POLICIES if m != 'budget']
    for cell_index in sorted({r['cell_index'] for r in cases}):
        group = [r for r in cases if r['cell_index'] == cell_index]
        methods = {}
        for method in POLICIES:
            jobs = [r for c in group for r in c['records'] if r['method'] == method]
            good = [r for r in jobs if r['status'] == 'success']
            methods[method] = dict(successes=len(good), failures=len(jobs) - len(good),
                                   active=sum(r['active'] for r in good),
                                   inactive_reasons={reason: sum(r['inactive_reason'] == reason for r in good)
                                                     for reason in sorted({r['inactive_reason'] for r in good})},
                                   metrics={m: describe([r['metrics'].get(m) for r in good]) for m in METRICS},
                                   mode_expected_mass=[describe([r['metrics']['mode_expected_mass'][k] for r in good]) for k in range(2)],
                                   u_child_mode_expected_mass=[describe([r['metrics']['u_child_mode_mass'][k]
                                                                         if r['metrics']['u_child_mode_mass'] is not None else None
                                                                         for r in good]) for k in range(2)],
                                   mode_recall=[describe([r['metrics']['mode_recall'][k]['recall'] for r in good]) for k in range(2)],
                                   mode_test_counts=[describe([r['metrics']['mode_recall'][k]['count'] for r in good]) for k in range(2)])
        usable = sum(c['shared']['usable_rounds'] for c in group)
        cells.append(dict(cell_index=cell_index, cell=group[0]['cell'], cases=len(group), methods=methods,
                          eligible_endpoints=describe([c['shared']['eligible_endpoints'] for c in group]),
                          usable_endpoints=describe([c['shared']['usable_endpoints'] for c in group]),
                          eligible_realized_error=describe([c['shared']['eligible_realized_error'] for c in group]),
                          eligible_oracle_error=describe([c['shared']['eligible_oracle_error'] for c in group]),
                          binding_rounds=sum(c['shared']['binding_rounds'] for c in group),
                          restrictive_rounds=sum(c['shared']['restrictive_rounds'] for c in group),
                          usable_rounds=usable,
                          binding_frequency=sum(c['shared']['binding_rounds'] for c in group) / usable if usable else None,
                          restrictive_frequency=sum(c['shared']['restrictive_rounds'] for c in group) / usable if usable else None,
                          paired=[paired_summary(group, control, metric, active)
                                  for control in controls for metric in METRICS for active in [False, True]]))
    # Pooled summary has ten seed-replicates, each averaging eight cell effects.
    pooled = []
    for control in controls:
        for metric in METRICS:
            values = []; seeds = []
            for seed in config['dataset_seeds']:
                seed_cases = [c for c in cases if c['seed'] == seed]
                deltas = []
                for c in seed_cases:
                    pair = paired_summary([c], control, metric)
                    if pair['difference']['n']:
                        deltas.append(pair['difference']['mean'])
                if len(deltas) == len(cells):
                    values.append(float(np.mean(deltas))); seeds.append(seed)
            pooled.append(dict(control=control, metric=metric, seeds=seeds, difference=describe(values),
                               requirement='all eight cell differences present per seed'))
    failed = [dict(cell=c['cell_index'], seed=c['seed'], method=r['method'], error=r['error'])
              for c in cases for r in c['records'] if r['status'] != 'success']
    return dict(study_id=config['study_id'], cases=len(cases), method_jobs=sum(len(c['records']) for c in cases),
                failures=failed, cells=cells, pooled=pooled,
                runtime=dict(wall_seconds=wall_seconds,
                             shared_preparation=describe([c['shared']['shared_seconds'] for c in cases]),
                             shared_preparation_total_seconds=sum(c['shared']['shared_seconds'] for c in cases),
                             diagnostic_total_seconds=sum(c['shared']['diagnostic_seconds'] for c in cases)),
                interpretation='Exploratory initial grid; no adaptive extension or confirmed general advantage')


def run(config_path, output):
    config = json.loads(config_path.read_text())
    if tuple(config['methods']) != POLICIES:
        raise ValueError('Configuration methods differ from the preregistered comparison')
    for path, expected in config['preserved_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f'Protected prototype/evidence changed: {path}')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory; results are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((ROOT / 'src').glob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    manifest = dict(config=config, config_sha256=sha256(config_path),
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
                    python=platform.python_version(), platform=platform.platform(),
                    packages={n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'scikit-learn']})
    write_json(output / 'manifest.json', manifest)
    cells = [dict(zip(config['grid'], values)) for values in itertools.product(*config['grid'].values())]
    all_results = []; started = time.perf_counter()
    for cell_index, cell in enumerate(cells):
        for seed in config['dataset_seeds']:
            directory = output / f'cell-{cell_index}' / f'seed-{seed}'
            directory.mkdir(parents=True)
            training, evaluator = generate_case(config, cell_index, cell, seed)
            np.savez_compressed(directory / 'evaluator_inputs.npz', L=training.L, labels=training.labels,
                                U=training.U, hidden_labels=evaluator.hidden_labels, pool_modes=evaluator.pool_modes,
                                test=evaluator.test, test_labels=evaluator.test_labels, test_modes=evaluator.test_modes,
                                labeled_modes=evaluator.labeled_modes)
            frozen = fit_case(training, config['fixed'], config['algorithm_seed'])
            training_arrays = {}
            for method, value in frozen['variants'].items():
                if value['status'] == 'success':
                    for key in ['X', 'y', 'sample_weight']:
                        training_arrays[f'{method}_{key}'] = value['data'][key]
            np.savez_compressed(directory / 'frozen_training.npz', **training_arrays)
            write_json(directory / 'frozen_decisions.json', dict(
                decision_sha256=frozen['decision_sha256'], decisions=frozen['frozen_decisions'],
                activity=frozen['activity'], warnings=frozen['warnings'],
                parent_diagnostics=frozen['prepared']['parent'].diagnostics_ if frozen['prepared'] is not None else None,
                banks=[dict(endpoint_ids=b['endpoint_ids'].tolist(), probabilities=b['probabilities'].tolist(),
                            capacity=b['capacity'], budget=b['budget'], offset=b['offset'],
                            surviving_candidates=[len(c) for c in b['candidates']],
                            rejected_candidates=b['rejected_candidates'])
                       for b in frozen['prepared']['banks']] if frozen['prepared'] is not None else []))
            result = evaluate_case(frozen, evaluator)
            result.update(seed=seed, cell_index=cell_index, cell=cell)
            write_json(directory / 'results.json', result)
            all_results.append(result)
        completed = all_results[-len(config['dataset_seeds']):]
        active = sum(next(r for r in c['records'] if r['method'] == 'budget').get('active', False) for c in completed)
        failures = sum(r['status'] != 'success' for c in completed for r in c['records'])
        print(json.dumps(dict(cell_index=cell_index, cell=cell, completed=len(completed), budget_active=active, failures=failures)), flush=True)
    summary = summarize(all_results, config, time.perf_counter() - started)
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/exposure_mechanism_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config, args.output)
    print(json.dumps(dict(cases=summary['cases'], method_jobs=summary['method_jobs'],
                          failed_jobs=len(summary['failures']), runtime_seconds=summary['runtime']['wall_seconds'])))
    return int(bool(summary['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
