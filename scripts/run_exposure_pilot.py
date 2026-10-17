"""Opt-in exploratory software/mechanism pilot, not validated research evidence.

Runs independent Gaussian draws with known posterior and optionally the first
provider Ecoli2/Ecoli4 fold. No source downloads and no default benchmark changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset
from src.exploratory_exposure import prepare, replay
from src.sshr import SSHR
from src.ugo import UGO
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scripts.ugo_real_fold_smoke import verify as verify_provider_folds


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def draw(rng, n, separation, prior=.2):
    labels = (rng.random(n) < prior).astype(int)
    X = rng.normal(size=(n, 2)) + (2 * labels[:, None] - 1) * separation / 2
    return X, labels


def posterior(X, separation, prior=.2):
    return expit(np.log(prior / (1 - prior)) + separation * X.sum(axis=1))


def scores(y, probability):
    predicted = (probability >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(mcc=float(matthews_corrcoef(y, predicted)),
                gmean=float(np.sqrt(tp / (tp + fn) * tn / (tn + fp))),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                macro_f1=float(f1_score(y, predicted, average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                ap=float(average_precision_score(y, probability)),
                fpr=float(fp / (tn + fp)), confusion=[[int(tn), int(fp)], [int(fn), int(tp)]])


def run_case(name, Xtrain, ytrain, Xtest, ytest, output, config, separation=None):
    # Design utility alone uses training labels to freeze the common mask.
    split = StratifiedShuffleSplit(n_splits=1, train_size=.5, random_state=config['seed'])
    labeled, hidden = next(split.split(Xtrain, ytrain))
    # This runner supports unique coordinate inputs; reject ambiguous boundaries
    # rather than silently splitting duplicates or deleting provider records.
    all_rows = np.vstack((Xtrain, Xtest))
    if len(np.unique(all_rows, axis=0)) != len(all_rows):
        raise ValueError('Pilot requires unique coordinates; use a separately declared grouped protocol')
    path = output / name; path.mkdir()
    # Detailed local records remain ignored. Hidden labels live only in this
    # evaluator artifact, never the prepare/replay training interface.
    np.savez_compressed(path / 'evaluator_inputs.npz', Xtrain=Xtrain, ytrain=ytrain,
                        Xtest=Xtest, ytest=ytest)
    write_json(path / 'split.json', dict(labeled=labeled.tolist(), hidden=hidden.tolist(),
                                       train_rows=len(Xtrain), test_rows=len(Xtest), config=config))
    scaler = StandardScaler().fit(Xtrain[labeled])
    L = scaler.transform(Xtrain[labeled]); labels = ytrain[labeled].copy()
    U = scaler.transform(Xtrain[hidden])
    classifier = LogisticRegression(max_iter=1000, random_state=config['seed'])
    parent = UGO(clone(classifier), variant='smote', random_state=config['seed'], allow_provisional=True)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        prepared = prepare(L, labels, U, parent, threshold=config['threshold'],
                           exposure_rate=config['exposure_rate'], endpoint_cap=config['endpoint_cap'])
        variants = {p: replay(prepared, p) for p in
                    ['parent', 'budget', 'unconstrained', 'count_matched', 'threshold_matched', 'weighted_pl']}
        variants['nores'] = dict(X=L.copy(), y=labels.copy())
        variants['class_weighted'] = dict(X=L.copy(), y=labels.copy())
        for method, sampler in [('ros', RandomOverSampler(random_state=config['seed'])),
                                ('smote', SMOTE(random_state=config['seed'], k_neighbors=min(5, int(np.bincount(labels).min()) - 1))),
                                ('sshr', SSHR(minority_label=1, random_state=config['seed']))]:
            try:
                a, b = sampler.fit_resample(L, labels, U) if method == 'sshr' else sampler.fit_resample(L, labels)
                variants[method] = dict(X=a, y=b)
            except Exception as exc:
                variants[method] = dict(error=f'{type(exc).__name__}: {exc}')
        # PL-only and PL+UGO use exactly the same frozen admitted endpoint set.
        ids = sorted({int(j) for bank in prepared['banks'] for j in bank['endpoint_ids']})
        pl_X = np.vstack((L, U[ids])); pl_y = np.r_[labels, np.ones(len(ids), dtype=labels.dtype)]
        variants['pl_only'] = dict(X=pl_X, y=pl_y)
        try:
            a, b = clone(parent).fit_resample(pl_X, pl_y)
            variants['pl_then_ugo'] = dict(X=a, y=b)
        except Exception as exc:
            variants['pl_then_ugo'] = dict(error=f'{type(exc).__name__}: {exc}')
    # Freeze every generation decision before any outer prediction or hidden-y
    # diagnostic. No oracle values return to prepare/replay or method selection.
    write_json(path / 'generation.json', dict(
        guide_status=prepared['guide_status'], parent=prepared['parent'].diagnostics_,
        scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
        pool_probabilities=prepared['probabilities'].tolist(),
        banks=[dict(offset=b['offset'], capacity=b['capacity'], budget=b['budget'],
                    endpoint_ids=b['endpoint_ids'].tolist(), radius=b['radius'],
                    rejected_candidates=b['rejected_candidates']) for b in prepared['banks']],
        policies={k: {field: v[field] for field in ['ledger', 'rounds'] if field in v}
                  for k, v in variants.items()}, warnings=[str(w.message) for w in caught]))
    fitted = {}; records = []
    for method, data in variants.items():
        if 'error' in data:
            records.append(dict(method=method, status='failed', error=data['error'])); continue
        try:
            model = clone(classifier)
            if method == 'class_weighted': model.set_params(class_weight='balanced')
            model.fit(data['X'], data['y'], sample_weight=data.get('sample_weight'))
            fitted[method] = model
        except Exception as exc:
            records.append(dict(method=method, status='failed', error=f'{type(exc).__name__}: {exc}'))
    test = scaler.transform(Xtest)
    for method, model in fitted.items():
        data = variants[method]; probability = model.predict_proba(test)[:, list(model.classes_).index(1)]
        record = dict(method=method, status='success', metrics=scores(ytest, probability),
                      fitting_rows=len(data['X']), fitting_weight=float(data.get('sample_weight', np.ones(len(data['X']))).sum()),
                      u_children=data.get('new_u_vectors', None),
                      u_endpoint_weight=data['u_informed_count'] if method == 'weighted_pl' else 0,
                      predictions=(probability >= .5).astype(int).tolist(),
                      probabilities=probability.tolist())
        ledger = data.get('ledger', [])
        if ledger:
            endpoints = np.array([r['endpoint'] for r in ledger])
            unique = np.unique(endpoints)
            record['lineage'] = dict(count=len(ledger), unique_endpoints=len(unique),
                                     endpoint_error=float(np.mean(ytrain[hidden][unique] == 0)),
                                     descendant_weighted_endpoint_error=float(np.mean(ytrain[hidden][endpoints] == 0)),
                                     estimated_exposure=float(sum(1 - r['guide_probability'] for r in ledger)))
            if separation is not None and method != 'weighted_pl':
                children = data['X'][[r['output_index'] for r in ledger]]
                original = scaler.inverse_transform(children)
                record['oracle_child_expected_error'] = float(np.mean(1 - posterior(original, separation)))
                record['unique_children'] = len(np.unique(children, axis=0))
        else:
            record['lineage'] = None  # No descendants is not zero error.
        records.append(record)
    result = dict(case=name, purpose='exploratory diagnostic only; no validated benefit or novelty',
                  elapsed_seconds=time.perf_counter() - started, records=records,
                  truth=ytest.tolist(), failures=sum(r['status'] == 'failed' for r in records),
                  limitations=['Fixed supervised UGO trace; not adaptive SSL UGO',
                               '0.5 threshold fixed for all; threshold tuning not implemented',
                               'SEG-SSC reproduction unavailable; not substituted by interpolation',
                               'No pointwise guide error bound or synthetic correctness certificate',
                               'Single seed per case; descriptive only'])
    write_json(path / 'results.json', result)
    return dict(case=name, jobs=len(records), failures=result['failures'],
                budget_children=variants['budget']['u_informed_count'],
                unconstrained_children=variants['unconstrained']['u_informed_count'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-real-folds', action='store_true', help='Use verified provider first Ecoli2/Ecoli4 folds')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use a fresh output directory; existing results are never overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(seed=42, threshold=.9, exposure_rate=.025, endpoint_cap=5,
                  labeled_fraction=.5, candidate_kernel='uniform open segment; reject unsupported U candidates',
                  trace='frozen supervised UGO-SMOTE; original defaults',
                  claims='exploratory only; no novelty or validated effectiveness')
    root = Path(__file__).resolve().parents[1]
    sources = {str(f.relative_to(root)): digest(f) for directory in ['src', 'scripts']
               for f in sorted((root / directory).glob('*.py'))}
    import importlib.metadata
    manifest = dict(config=config, python=platform.python_version(), source_sha256=sources,
                    packages={n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'scikit-learn', 'imbalanced-learn']},
                    planned_cases=['separated', 'overlap'], seed=42)
    if args.include_real_folds:
        source_manifest = json.loads((root / 'data/manifests/ugo_real_fold_sources.json').read_text())
        report = verify_provider_folds(source_manifest, root)
        manifest['provider'] = {k: source_manifest[k] for k in ['repository', 'commit']}
        manifest['real_folds'] = {}
        for name in ['ecoli2', 'ecoli4']:
            fold = report[name]
            if fold['status'] != 'VERIFIED_MULTISET':
                raise ValueError('Provider folds must pass verification before this pilot')
            directory = root / 'data/raw/keel' / name
            pair = dict(k=5, fold=1, train_path=f'{name}-5-1tra.dat', test_path=f'{name}-5-1tst.dat')
            for role in ['train', 'test']:
                pair[role + '_sha256'] = digest(directory / pair[role + '_path'])
            manifest['real_folds'][name] = pair
    write_json(args.output / 'manifest.json', manifest)
    summary = []
    for index, (name, separation) in enumerate([('separated', 3.), ('overlap', .8)]):
        rng = np.random.default_rng(np.random.SeedSequence([42, index]))
        X, y = draw(rng, 600, separation); Xt, yt = draw(rng, 2000, separation)
        summary.append(run_case(name, X, y, Xt, yt, args.output, config, separation))
    if args.include_real_folds:
        for name, pair in manifest['real_folds'].items():
            directory = root / 'data/raw/keel' / name
            train = load_dataset(directory / pair['train_path'], minority_label='positive')
            test = load_dataset(directory / pair['test_path'], minority_label='positive')
            summary.append(run_case(name, train.X.to_numpy(), train.y.to_numpy(),
                                    test.X.to_numpy(), test.y.to_numpy(), args.output, config))
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary))
    return int(any(r['failures'] for r in summary))


if __name__ == '__main__':
    raise SystemExit(main())
