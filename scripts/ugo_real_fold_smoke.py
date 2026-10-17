"""Verify pinned provider files and run first-training-fold UGO smoke checks.

No downloads unless --download is explicit. No predictive test-set evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset
from src.ugo import UGO


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_counts(dataset):
    return Counter(tuple(row) + (label,) for row, label in
                   zip(dataset.X.to_numpy(), dataset.original_y.to_numpy()))


def verify(manifest, root, download=False):
    """Check all hashes, five train/test multisets, schemas and test coverage."""
    for item in manifest['files']:
        path = root / item['path']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Manifest path escapes root')
        if not path.exists() and download:
            with urllib.request.urlopen(item['url'], timeout=30) as response:
                contents = response.read(2_000_000)
            if hashlib.sha256(contents).hexdigest() != item['sha256']:
                raise ValueError(f'Provider checksum mismatch: {path.name}')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream: stream.write(contents)
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError(f'Missing or changed provider file: {path}')
    reports = {}
    for name in ['ecoli2', 'yeast3', 'ecoli4']:
        directory = root / 'data/raw/keel' / name
        full = load_dataset(directory / f'{name}.dat', minority_label='positive')
        expected = row_counts(full); tests = Counter(); overlaps = []
        for i in range(1, 6):
            train = load_dataset(directory / f'{name}-5-{i}tra.dat', minority_label='positive')
            test = load_dataset(directory / f'{name}-5-{i}tst.dat', minority_label='positive')
            for partition in [train, test]:
                if (partition.feature_names != full.feature_names or
                        partition.numeric_feature_names != full.numeric_feature_names or
                        partition.categorical_feature_names != full.categorical_feature_names):
                    raise ValueError('Provider schemas differ')
            tr, te = row_counts(train), row_counts(test)
            if tr + te != expected:
                raise ValueError(f'{name} fold {i}: full-file multiset mismatch')
            tests.update(te)
            overlaps.append(sum((tr & te).values()))
        if tests != expected:
            raise ValueError(f'{name}: five test folds do not cover full data once')
        reports[name] = dict(status='VERIFIED_MULTISET', folds=5, rows=len(full.X),
                             duplicate_complete_occurrences_across_fold_boundaries=overlaps,
                             limit='Provider repository folds; not verified UGO author folds or unique record IDs')
    return reports


def training_smoke(path, variant):
    """This function opens only one training file, never a held-out file."""
    data = load_dataset(path, minority_label='positive')
    if data.categorical_feature_names or data.X.isna().any().any():
        raise ValueError('Smoke requires finite numerical training inputs')
    X = StandardScaler().fit_transform(data.X); y = data.y.to_numpy()
    before, labels = X.copy(), y.copy()
    params = dict(estimator=LogisticRegression(max_iter=1000, random_state=42),
                  variant=variant, random_state=42, allow_provisional=True)
    first, second = UGO(**params), UGO(**params)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        a, b = first.fit_resample(X, y); c, d = second.fit_resample(X, y)
    checks = [np.array_equal(X, before), np.array_equal(y, labels),
              np.array_equal(a[:len(X)], X), np.array_equal(b[:len(y)], y),
              a.shape[1] == X.shape[1], len(a) == len(b), np.isfinite(a).all(),
              np.array_equal(a, c), np.array_equal(b, d), first.diagnostics_ == second.diagnostics_]
    if not all(checks): raise RuntimeError('UGO training-fold smoke invariant failed')
    return dict(status='passed', variant=variant, input_rows=len(X), output_rows=len(a),
                deterministic=True, seed=42, diagnostics=first.diagnostics_)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Output exists; preserve it and choose a fresh filename')
    manifest = json.loads((ROOT / 'data/manifests/ugo_real_fold_sources.json').read_text())
    reports = verify(manifest, ROOT, args.download)
    records = []
    for name in reports:
        path = ROOT / f'data/raw/keel/{name}/{name}-5-1tra.dat'
        for variant in ['ros', 'smote']:
            record = training_smoke(path, variant)
            record.update(dataset=name, training_sha256=sha256(path))
            records.append(record)
            print(name, variant, record['status'], record['diagnostics']['number_of_synthetic_samples'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(provider=manifest['repository'], commit=manifest['commit'],
                                          verification=reports, records=records), indent=2) + '\n')
    return 0


if __name__ == '__main__': raise SystemExit(main())
