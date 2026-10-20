from copy import deepcopy

import numpy as np
import pytest

from src.original_support_hs_study import (
    concentrated_smote,load_config,synthetic_case,tune_models,paired_summary,decision,
)


def test_frozen_design_rejects_post_outcome_changes(tmp_path):
    import json
    config=load_config()
    config['scientific_design']['seeds']=[1,2,3]
    path=tmp_path/'config.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='frozen'):
        load_config(path)


def test_concentration_changes_only_synthetic_rows_and_cannot_use_hidden_groups():
    design=load_config()['scientific_design']
    a=synthetic_case(design,42,0)
    b=synthetic_case(design,42,1)
    for name in ['fitting','validation','test']:
        np.testing.assert_array_equal(getattr(a,name).X,getattr(b,name).X)
        np.testing.assert_array_equal(getattr(a,name).y,getattr(b,name).y)
    X,y=a.fitting.X,a.fitting.y
    augmented=[]
    for c in [0,.5,1]:
        out_X,out_y,trace=concentrated_smote(X,y,c,42)
        assert len(out_X)==2000 and np.sum(out_y==1)==1000
        np.testing.assert_array_equal(out_X[:len(X)],X)
        np.testing.assert_array_equal(out_y[:len(y)],y)
        reconstructed=X[trace['seed_indices']]+trace['fractions'][:,None]*(
            X[trace['partner_indices']]-X[trace['seed_indices']])
        np.testing.assert_array_equal(out_X[len(X):],reconstructed)
        assert np.all(y[trace['seed_indices']]==1) and np.all(y[trace['partner_indices']]==1)
        augmented.append(out_X)
    assert not np.array_equal(augmented[0],augmented[-1])
    with pytest.raises(TypeError):
        concentrated_smote(X,y,0,42,groups=a.fitting.groups)


def test_selection_budget_shared_structure_and_training_only_scaler():
    design=load_config()['scientific_design']
    rng=np.random.default_rng(4)
    X=np.r_[rng.normal(size=(60,2)),rng.normal(1,size=(20,2))]
    y=np.r_[np.zeros(60,dtype=int),np.ones(20,dtype=int)]
    V=np.r_[rng.normal(100,size=(20,2)),rng.normal(101,size=(10,2))]
    vy=np.r_[np.zeros(20,dtype=int),np.ones(10,dtype=int)]
    fitted=tune_models(X,y,V,vy,design,42)
    np.testing.assert_allclose(fitted['scaler'].mean_,X.mean(axis=0))
    assert fitted['scaler'].n_samples_seen_==len(X)
    assert fitted['fit_count']==37
    assert len(fitted['candidates'])==60
    B=[r for r in fitted['candidates'] if r['arm']=='B_SMOTE_STANDARD_HS']
    C=[r for r in fitted['candidates'] if r['arm']=='C_SMOTE_ORIGINAL_HS']
    for b,c in zip(B,C):
        assert b['base_tree_sha256']==c['base_tree_sha256']
        assert b['parameters']['relative_lambda']==c['parameters']['relative_lambda']
    assert (fitted['selected']['B_SMOTE_STANDARD_HS']['candidate']['base_tree_sha256'] ==
            fitted['selected']['C_SMOTE_ORIGINAL_HS']['candidate']['base_tree_sha256'])
    # Test data / hidden metadata are not accepted at the selection boundary.
    with pytest.raises(TypeError):
        tune_models(X,y,V,vy,design,42,test_y=vy)
    with pytest.raises(TypeError):
        tune_models(X,y,V,vy,design,42,groups=np.zeros(len(X)))


def test_selection_independent_of_test_changes(monkeypatch):
    import src.original_support_hs_study as study
    design=load_config()['scientific_design']
    case=synthetic_case(design,42,0)
    # Smaller fitting sets for a meaningful end-to-end isolation fixture.
    for name in ['fitting','validation','test']:
        partition=getattr(case,name)
        ids=np.r_[np.flatnonzero(partition.y==0)[:50],np.flatnonzero(partition.y==1)[:20]]
        partition.X=partition.X[ids]; partition.y=partition.y[ids]; partition.groups=partition.groups[ids]
    first,_=study.evaluate_case(case,design)
    changed=deepcopy(case)
    changed.test.X+=17
    changed.test.y=1-changed.test.y
    changed.test.groups=np.full(len(changed.test.y),2)
    second,_=study.evaluate_case(changed,design)
    assert first['candidates']==second['candidates']
    assert first['augmentation_X_sha256']==second['augmentation_X_sha256']
    assert first['scaler_mean']==second['scaler_mean']


def test_missing_coverage_is_explicitly_inconclusive():
    import pandas as pd
    result=decision(pd.DataFrame(),load_config()['scientific_design'],[{'message':'fixture'}])
    assert result['status']=='incomplete_pilot_no_advancement'
