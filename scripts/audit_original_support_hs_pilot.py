"""Recompute frozen HS pilot evidence without fitting any classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from scripts.run_original_support_hs_pilot import cases,case_name,json_write,source_hashes
from src.original_support_hs_study import load_config,array_hash,score_probabilities


def route(X,left,right,feature,threshold):
    counts=np.zeros(len(left),dtype=int)
    leaves=np.empty(len(X),dtype=int)
    # sklearn tree comparisons use float32 input; cast before threshold walk.
    for i,row in enumerate(np.asarray(X,dtype=np.float32)):
        node=0
        while True:
            counts[node]+=1
            if left[node]==-1:
                leaves[i]=node
                break
            node=left[node] if row[feature[node]]<=threshold[node] else right[node]
    return counts,leaves


def independent_shrink(left,right,raw,counts,lam):
    """Independent leaf-to-root summation; not the sampler's forward recursion."""
    parents=np.full(len(left),-1,dtype=int)
    for p in range(len(left)):
        if left[p]!=-1:
            parents[left[p]]=p; parents[right[p]]=p
    result=[]
    for node in range(len(left)):
        increments=[]
        while parents[node]!=-1:
            parent=parents[node]
            factor=1. if lam==0 else counts[parent]/(counts[parent]+lam)
            increments.append((raw[node]-raw[parent])*factor)
            node=parent
        result.append(raw[0]+sum(reversed(increments),start=np.zeros(raw.shape[1])))
    return np.asarray(result)


def audit(directory):
    from src.original_support_hs_study import concentrated_smote
    directory=Path(directory)
    config=load_config(); design=config['scientific_design']
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['design_sha256']==config['design_sha256']
    assert manifest['source_hashes']==source_hashes(), 'Run source differs from current audit source.'
    max_error=0.; rows=0; validated_cases=0
    for case in cases(design):
        name=case_name(case)
        record=json.loads((directory/'cases'/(name+'.json')).read_text())
        artifact=directory/'cases'/(name+'.npz')
        assert hashlib.sha256(artifact.read_bytes()).hexdigest()==record['prediction_artifact_sha256']
        for partition in ['fitting','validation','test']:
            actual=getattr(case,partition)
            assert array_hash(actual.X)==record['partitions'][partition]['X']
            assert array_hash(actual.y)==record['partitions'][partition]['y']
        scaler=StandardScaler().fit(case.fitting.X)
        X=scaler.transform(case.fitting.X); T=scaler.transform(case.test.X)
        assert scaler.n_samples_seen_==record['scaler_rows']==len(X)
        np.testing.assert_array_equal(scaler.mean_,record['scaler_mean'])
        if case.concentration is None:
            aug_X,aug_y=SMOTE(k_neighbors=3,random_state=case.seed).fit_resample(X,case.fitting.y)
        else:
            aug_X,aug_y,_=concentrated_smote(X,case.fitting.y,case.concentration,case.seed)
        assert array_hash(aug_X)==record['augmentation_X_sha256']
        assert array_hash(aug_y)==record['augmentation_y_sha256']
        assert len(record['candidates'])==60
        for arm in design['arms']:
            candidate=[r for r in record['candidates'] if r['arm']==arm]
            assert len(candidate)==12
            chosen=max(candidate,key=lambda r:(r['average_precision'],r['minority_f1']))
            row=next(r for r in record['rows'] if r['arm']==arm)
            assert row['selected_candidate']==chosen['candidate_index']
            assert row['selected_parameters']==chosen['parameters']
        with np.load(artifact,allow_pickle=False) as vectors:
            np.testing.assert_array_equal(vectors['test_y'],case.test.y)
            for row in record['rows']:
                arm=row['arm']; scores=vectors[arm]
                recomputed=score_probabilities(case.test.y,scores)
                for metric,value in recomputed.items():
                    error=abs(row[metric]-value); max_error=max(max_error,error)
                    assert error<1e-12
                if case.test.groups is not None:
                    expected=float(np.mean(scores[case.test.groups==2]>=.5))
                    assert row['rare_recall']==expected
                rows+=1
            B='B_SMOTE_STANDARD_HS'; C='C_SMOTE_ORIGINAL_HS'
            for suffix in ['left','right','feature','threshold','raw','original_counts','augmented_counts']:
                np.testing.assert_array_equal(vectors[B+'__'+suffix],vectors[C+'__'+suffix])
            for arm in [B,C]:
                get=lambda field:vectors[arm+'__'+field]
                left,right,feature,threshold=(get(f) for f in ['left','right','feature','threshold'])
                original,_=route(X,left,right,feature,threshold)
                augmented,_=route(aug_X,left,right,feature,threshold)
                positive,_=route(aug_X[aug_y==1],left,right,feature,threshold)
                np.testing.assert_array_equal(original,get('original_counts'))
                np.testing.assert_array_equal(augmented,get('augmented_counts'))
                raw=np.column_stack((1-positive/augmented,positive/augmented))
                np.testing.assert_allclose(raw,get('raw'),rtol=0,atol=1e-14)
                _,leaves=route(T,left,right,feature,threshold)
                np.testing.assert_array_equal(leaves,get('test_leaves'))
                row=next(r for r in record['rows'] if r['arm']==arm)
                counts=augmented if arm==B else original
                shrunk=independent_shrink(left,right,raw,counts,row['selected_parameters']['lambda'])
                np.testing.assert_allclose(shrunk,get('shrunk'),rtol=0,atol=1e-12)
                np.testing.assert_allclose(shrunk[leaves,1],vectors[arm],rtol=0,atol=1e-12)
        validated_cases+=1
    assert validated_cases==30 and rows==150
    return {'status':'passed','cases':validated_cases,'test_score_rows':rows,
            'candidate_selections_checked':150,'classifier_refits':0,
            'standard_and_original_HS_shared_tree_checks':30,
            'original_and_augmented_node_counts_replayed':60,
            'max_metric_error':max_error,'source_commit':manifest['source_commit'],
            'limits':'Validation AP/F1 selection arithmetic checked from recorded candidate scores; validation probability vectors were not saved. Training/selection isolation is covered by source and tests.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.input)
    json_write(args.input/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
