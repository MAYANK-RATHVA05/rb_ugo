"""Run only the frozen 30-case original-support HS pilot; no plotting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd

from src.original_support_hs_study import (
    ROOT,CONFIG,evaluate_case,load_config,load_real_case,synthetic_case,
    paired_summary,decision,
)


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def revision():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def source_hashes():
    names=['src/original_support_hs.py','src/original_support_hs_study.py',
           'scripts/run_original_support_hs_pilot.py','configs/original_support_hs_v1.json']
    return {name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for name in names}


def cases(design):
    for entry in design['datasets']:
        for seed in design['seeds']:
            yield load_real_case(entry,seed)
    for concentration in design['synthetic']['concentration_levels']:
        for seed in design['seeds']:
            yield synthetic_case(design,seed,concentration)


def case_name(case):
    concentration='' if case.concentration is None else f'-c{case.concentration:g}'
    return f'{case.dataset}{concentration}-seed{case.seed}'


def run(output,config_path=CONFIG):
    config=load_config(config_path); design=config['scientific_design']
    output=Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory; existing experiment evidence is never overwritten.')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
    if dirty.strip():
        raise ValueError('Commit the frozen implementation before running the pilot.')
    output.mkdir(parents=True)
    (output/'cases').mkdir()
    manifest={'protocol_id':config['protocol_id'],'design_sha256':config['design_sha256'],
              'source_commit':revision(),'source_hashes':source_hashes(),
              'packages':{name:importlib.metadata.version(name) for name in
                          ['numpy','pandas','scipy','scikit-learn','imbalanced-learn']},
              'planned_cases':30,'planned_validation_candidates':1800,'planned_unique_tree_fits':1110}
    json_write(output/'manifest.json',manifest)
    scores=[]; candidates=[]; failures=[]; total_fits=0
    start=perf_counter()
    for case in cases(design):
        name=case_name(case)
        try:
            record,vectors=evaluate_case(case,design)
            artifact=output/'cases'/(name+'.npz')
            np.savez_compressed(artifact,**vectors)
            record['prediction_artifact_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
            json_write(output/'cases'/(name+'.json'),record)
            base={'case_id':name,'dataset':case.dataset,'seed':case.seed,'concentration':case.concentration}
            for row in record['rows']:
                row={**base,**row,'selected_parameters':json.dumps(row['selected_parameters'],sort_keys=True)}
                scores.append(row)
            candidates.extend({**base,**row,'parameters':json.dumps(row['parameters'],sort_keys=True)}
                              for row in record['candidates'])
            total_fits+=record['fits']
            print(f'{name}: success, {record["fits"]} unique fits',flush=True)
        except Exception as exc:
            failure={'case_id':name,'type':type(exc).__name__,'message':str(exc)}
            failures.append(failure)
            json_write(output/'cases'/(name+'-failure.json'),failure)
            print(f'{name}: FAILED {type(exc).__name__}: {exc}',flush=True)
    frame=pd.DataFrame(scores)
    frame.to_csv(output/'scores.csv',index=False)
    pd.DataFrame(candidates).to_csv(output/'selection.csv',index=False)
    paired=pd.DataFrame(paired_summary(frame)) if len(frame) else pd.DataFrame()
    paired.to_csv(output/'paired.csv',index=False)
    summary=(frame.groupby(['dataset','concentration','arm'],dropna=False)
             .agg(n=('seed','count'),AP_mean=('average_precision','mean'),AP_sd=('average_precision','std'),
                  F1_mean=('minority_f1','mean'),F1_sd=('minority_f1','std'),
                  rare_recall_mean=('rare_recall','mean'),
                  tuning_seconds_mean=('selection_seconds_including_shared_base','mean'))
             .reset_index()) if len(frame) else pd.DataFrame()
    summary.to_csv(output/'summary.csv',index=False)
    result=decision(frame,design,failures)
    result.update(source_commit=manifest['source_commit'],design_sha256=config['design_sha256'],
                  completed_cases=len(frame)//5,completed_scores=len(frame),
                  completed_validation_candidates=len(candidates),successful_tree_fits=total_fits,
                  runtime_seconds=perf_counter()-start,failures=failures)
    json_write(output/'decision.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/original-support-hs-v1')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        config=load_config(); total=0
        for case in cases(config['scientific_design']):
            if np.sum(case.fitting.y==1)<=3:
                raise ValueError(f'Insufficient fitting minority rows: {case_name(case)}')
            total+=1
        print(f'{total} cases ready; 1800 validation candidates; 1110 unique tree fits; no fitting performed.')
        print(f'Design: {config["design_sha256"]}')
        return
    run(args.output)


if __name__=='__main__':
    main()
