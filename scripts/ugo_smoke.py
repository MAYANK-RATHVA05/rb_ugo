"""Training-fold-only UGO software smoke checks; never predictive benchmarks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import warnings

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_dataset
from scripts.import_resources import sha256
from src.ugo import UGO


def run_smoke(data_root: Path, fold_reports: dict, datasets: list[str], output: Path) -> bool:
    records=[];blocked=False
    for name in datasets:
        report=fold_reports.get(name,{})
        pairs=sorted((r for r in report.get('folds',[]) if r.get('status')=='LOADED' and r.get('k')==5),key=lambda r:r['fold'])
        if report.get('status')!='VERIFIED_MULTISET' or not pairs:
            blocked=True
            records.extend({'dataset':name,'variant':v,'status':'BLOCKED_MISSING_VERIFIED_FIVE_FOLD','reason':report.get('reason',report.get('status','Missing dataset/fold report'))} for v in ['ros','smote'])
            continue
        train_path=data_root/name/(pairs[0].get('train_path') or pairs[0]['train'])
        # Only the training file is loaded here; test paths are not opened.
        try:
            if not train_path.resolve().is_relative_to((data_root/name).resolve()):
                raise ValueError('Training path escapes dataset directory')
            if sha256(train_path)!=pairs[0]['train_sha256']:
                raise ValueError('Training checksum differs from verified fold report')
            d=load_dataset(train_path)
            if d.categorical_feature_names or d.X.isna().any().any():
                raise ValueError('This smoke runner requires finite numerical training features; no silent category encoding/imputation')
            X=StandardScaler().fit_transform(d.X);y=d.y.to_numpy();before=X.copy();labels=y.copy()
            for variant in ['ros','smote']:
                sampler=UGO(LogisticRegression(max_iter=1000,random_state=42),variant=variant,allow_provisional=True,random_state=42)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('always');xa,ya=sampler.fit_resample(X,y)
                    repeat=UGO(LogisticRegression(max_iter=1000,random_state=42),variant=variant,allow_provisional=True,random_state=42)
                    xb,yb=repeat.fit_resample(X,y)
                assert np.array_equal(X,before) and np.array_equal(y,labels)
                assert np.array_equal(xa[:len(X)],X) and np.array_equal(ya[:len(y)],y)
                assert len(xa)==len(ya) and xa.shape[1]==X.shape[1]
                assert set(ya)==set(y) and np.isfinite(xa).all()
                assert np.array_equal(xa,xb) and np.array_equal(ya,yb)
                assert sampler.diagnostics_==repeat.diagnostics_
                diag=sampler.diagnostics_
                records.append({'dataset':name,'variant':variant,'status':'PASSED','reason':diag['stopping_reason'],
                                'training_filename':train_path.name,'input_rows':len(X),'output_rows':len(xa),
                                'synthetic_rows':diag['number_of_synthetic_samples'],'noise_original':diag['pre_detected_noisy_samples'],
                                'relocated_generated':diag['generated_noisy_samples_relocated'],'history':json.dumps(diag['history']),
                                'deterministic':True,'seed':42})
        except (ValueError,OSError,AssertionError) as exc:
            blocked=True
            for v in ['ros','smote']:
                if not any(r['dataset']==name and r['variant']==v for r in records):
                    records.append({'dataset':name,'variant':v,'status':'FAILED','reason':str(exc)})
    output.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in records for k in r))
    with output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    for r in records:print(r['dataset'],r['variant'],r['status'])
    return not blocked


def main() -> int:
    root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=root/'data/raw/keel')
    p.add_argument('--fold-report',type=Path,default=root/'results/verification/fold_report.json')
    p.add_argument('--datasets',nargs='+',default=['ecoli2','yeast3','ecoli4'])
    p.add_argument('--output',type=Path,default=root/'results/verification/ugo_smoke_summary.csv')
    a=p.parse_args()
    try:return 0 if run_smoke(a.data_root,json.loads(a.fold_report.read_text()),a.datasets,a.output) else 2
    except (OSError,ValueError) as e:p.exit(2,f'Smoke verification failed: {e}\n')


if __name__=='__main__':raise SystemExit(main())
