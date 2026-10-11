"""Offline full-file statistics and multiset validation of supplied KEEL folds."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.import_resources import sha256
from src.data_loader import DatasetError, load_dataset
from src.imbalance_stats import calculate_statistics

FOLD = re.compile(r'^(.*)-(\d+)-(\d+)(tra|tst)\.dat$', re.I)


def rows(dataset, features_only=False) -> Counter:
    frame = dataset.X.astype(object).copy()
    if not features_only:
        frame['__target__'] = dataset.original_y.to_numpy()
    import pandas as pd
    return Counter(tuple(('__missing__',) if pd.isna(v) else v for v in row)
                   for row in frame.itertuples(index=False, name=None))


def load_with_mapping(path: Path, mapping: dict[str, int]):
    """Parse separately, then apply training-established roles even if counts reverse."""
    try:
        d = load_dataset(path)
    except DatasetError as exc:
        if 'Equal class counts' not in str(exc):
            raise
        d = load_dataset(path, minority_label=next(k for k, v in mapping.items() if v == 1))
    if set(d.original_y) != set(mapping):
        raise DatasetError('Training/test original label vocabularies differ')
    y = d.original_y.map(mapping).astype('int64')
    minority = next(k for k, v in mapping.items() if v == 1)
    majority = next(k for k, v in mapping.items() if v == 0)
    return replace(d, y=y, label_mapping=dict(mapping), minority_label=minority,
                   majority_label=majority, minority_count=int((y == 1).sum()),
                   majority_count=int((y == 0).sum()))


def audit_folds(directory: Path, full_path: Path | None) -> dict:
    matches = {}
    for p in sorted(directory.rglob('*.dat')):
        m = FOLD.match(p.name)
        if m:
            key = (m[1], int(m[2]), int(m[3]))
            if m[4].lower() in matches.setdefault(key, {}):
                raise DatasetError(f'Duplicate fold part: {p.name}')
            matches[key][m[4].lower()] = p
    if not matches:
        return {'status': 'MISSING_FOLDS', 'folds': [], 'coverage': 'UNVERIFIED',
                'reason': 'Supplied archive contains no filename-identifiable train/test pairs'}
    full = load_dataset(full_path) if full_path else None
    expected = rows(full) if full else None
    records = []; tests = Counter(); errors = []
    families = {(base, k) for base, k, _ in matches}
    if len(families) != 1:
        errors.append('Multiple fold families: audit separately; do not merge repeated CV')
    for (base, k, i), parts in sorted(matches.items()):
        record = {'base': base, 'k': k, 'fold': i,
                  'train': parts.get('tra').name if 'tra' in parts else None,
                  'test': parts.get('tst').name if 'tst' in parts else None,
                  'train_path': parts['tra'].relative_to(directory).as_posix() if 'tra' in parts else None,
                  'test_path': parts['tst'].relative_to(directory).as_posix() if 'tst' in parts else None}
        if set(parts) != {'tra', 'tst'}:
            errors.append(f'Incomplete fold {i}');record['status']='INCOMPLETE';records.append(record);continue
        try:
            train = load_dataset(parts['tra'])
            test = load_with_mapping(parts['tst'], train.label_mapping)
            if (train.feature_names, train.numeric_feature_names, train.categorical_feature_names) != (test.feature_names, test.numeric_feature_names, test.categorical_feature_names):
                raise DatasetError('Training/test schema mismatch')
            tr, te = rows(train), rows(test)
            tests.update(te)
            complete_overlap = sum((tr & te).values())
            feature_overlap = sum((rows(train, True) & rows(test, True)).values())
            record.update(status='LOADED', train_count=len(train.X), test_count=len(test.X),
                          mapping=train.label_mapping, train_sha256=sha256(parts['tra']),
                          test_sha256=sha256(parts['tst']), identical_complete_occurrences=complete_overlap,
                          identical_feature_occurrences=feature_overlap,
                          pair_matches_full_multiset=(tr + te == expected) if expected is not None else None)
            if expected is not None and tr + te != expected:
                errors.append(f'Fold {i} train+test differs from full-file multiset')
        except (DatasetError, ValueError) as exc:
            record.update(status='ERROR', error=str(exc));errors.append(f'Fold {i}: {exc}')
        records.append(record)
    if len(families)==1:
        base,k=next(iter(families))
        if {i for b,n,i in matches} != set(range(1,k+1)):
            errors.append('Fold identifiers do not cover 1..k')
    cover = tests == expected if expected is not None and len(families)==1 else None
    if cover is False: errors.append('Test partitions do not cover the full-file multiset exactly once')
    return {'status':'INVALID' if errors else 'VERIFIED_MULTISET', 'folds':records,
            'coverage':cover, 'errors':errors,
            'identity_limit':'Identical-valued occurrences are not distinguishable record IDs; provider provenance not authenticated'}


def verify(directory: Path, archive_dir: Path) -> tuple[list[dict], dict]:
    manifest=[];folds={}
    for group in sorted(p for p in directory.iterdir() if p.is_dir()):
        files=sorted(group.rglob('*.dat'))
        full_files=[p for p in files if not FOLD.match(p.name)]
        if len(full_files)!=1:
            raise DatasetError(f'{group.name}: expected one full .dat, found {len(full_files)}')
        p=full_files[0];d=load_dataset(p);s=calculate_statistics(d)
        relation=next(line.strip()[len('@relation'):].strip() for line in p.read_text(encoding='utf-8-sig').splitlines() if line.lower().strip().startswith('@relation '))
        archive=archive_dir/(group.name+'.zip')
        fold=audit_folds(group,p);folds[group.name]=fold
        manifest.append({'dataset':group.name,'relation':relation,'original_filename':p.name,
                         'file_sha256':sha256(p),'archive_filename':archive.name if archive.exists() else '',
                         'archive_sha256':sha256(archive) if archive.exists() else '',
                         'target_name':d.target_name,'target_labels':json.dumps(sorted(set(d.original_y))),
                         'majority_label':d.majority_label,'minority_label':d.minority_label,
                         'numeric_feature_names':json.dumps(d.numeric_feature_names),
                         'categorical_feature_names':json.dumps(d.categorical_feature_names),
                         **s.to_dict(),'available_fold_pairs':sum(r['status']=='LOADED' for r in fold['folds']),
                         'fold_status':fold['status'],'source_provenance':'User-supplied KEEL-format archive; provider URL/license not independently authenticated'})
    return manifest,folds


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--data-root',type=Path,default=root/'data/raw/keel')
    p.add_argument('--archive-dir',type=Path,default=root/'data/incoming')
    p.add_argument('--manifest',type=Path,default=root/'data/manifests/keel_dataset_manifest.csv')
    p.add_argument('--fold-report',type=Path,default=root/'results/verification/fold_report.json')
    a=p.parse_args()
    try:
        manifest,folds=verify(a.data_root,a.archive_dir)
        if not manifest:raise DatasetError('No dataset directories found')
        a.manifest.parent.mkdir(parents=True,exist_ok=True)
        with a.manifest.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
        a.fold_report.parent.mkdir(parents=True,exist_ok=True)
        a.fold_report.write_text(json.dumps(folds,indent=2)+'\n',encoding='utf-8')
        for d in manifest:print(f"{d['dataset']}: {d['total_samples']} rows; {d['fold_status']}")
        return 2 if any(f['status']=='INVALID' for f in folds.values()) else 0
    except (OSError,DatasetError) as e:p.exit(2,f'Verification failed: {e}\n')


if __name__=='__main__':raise SystemExit(main())
