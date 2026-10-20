"""Correctness fixtures only; no pilot performance expectations."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.gmmsampling_overlap import (OverlapSampler, allocation_weights, component_quotas,
    density_accept, fit_class_gmm, safe_levels)
from src.gmmsampling_overlap_study import CONFIG, generate_partition, metrics, run_case

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(max_components=3,valid_size=.25,n_init=1,max_iter=100,tol=.001,reg_covar=1e-6)


@pytest.fixture
def sampler():
    data = generate_partition(42,100,[100,40,40],1.5,.5)
    with threadpool_limits(limits=1):
        return OverlapSampler(StandardScaler().fit_transform(data["X"]),data["y"],[1,2],42,SETTINGS)


def author_harness():
    path = ROOT / "docs/research/gmmsampling_overlap/pilot/pinned_author_gmm_sampler.py.txt"
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == json.loads(CONFIG.read_text())["author_file_sha256"]
    parsed = ast.parse(source)
    author = next(n for n in parsed.body if isinstance(n,ast.ClassDef))
    names = {"_check_sample_neighborhood","_get_probas_for_samples_in_component","_fit_single_class","_perform_step","_init_model"}
    author.bases, author.decorator_list = [], []
    author.body = [n for n in author.body if isinstance(n,ast.FunctionDef) and n.name in names]
    for node in author.body:
        node.decorator_list = []
    namespace = dict(np=np,deepcopy=deepcopy,GaussianMixture=GaussianMixture,train_test_split=train_test_split)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),author],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),"exec"),namespace)
    return namespace["GMMSampler"]()


def test_density_stability_zero_and_invalid():
    logs = np.array([[-10000,-10001],[-10001,-10000],[-np.inf,0],[0,-np.inf],[-np.inf,-np.inf],[np.nan,0],[0,np.inf]])
    accepted, invalid = density_accept(logs,0,.55)
    np.testing.assert_array_equal(accepted,[True,False,False,True,False,False,False])
    np.testing.assert_array_equal(invalid,[False,False,False,False,True,True,True])
    shifted, _ = density_accept(logs[:2]+9000,0,.55)
    np.testing.assert_array_equal(shifted,accepted[:2])
    np.testing.assert_array_equal(density_accept(np.array([[0.,0.]]),0,.5)[0],[True])
    assert density_accept(logs,0,0)[0].all()


def test_author_safe_and_allocation_exact(sampler):
    author = author_harness()
    author.class_sizes = Counter(sampler.y)
    author.k_neighbors = 7
    # Explicit fixed neighbor labels exercise unequal/equal cross-class weights.
    assert author._check_sample_neighborhood(1,Counter({0:2,1:2,2:3})) == (2*.4+5)/7
    safety = safe_levels(sampler.X,sampler.y,7)
    distance = ((sampler.X[:,None]-sampler.X[None,:])**2).sum(axis=2)
    np.fill_diagonal(distance,np.inf)
    ids = np.argsort(distance,axis=1,kind="stable")[:,:7]
    expected = [author._check_sample_neighborhood(c,Counter(sampler.y[n])) for c,n in zip(sampler.y,ids)]
    np.testing.assert_allclose(safety,expected,atol=1e-15)
    author.gaussian_mixtures = sampler.models
    author.neighborhood = dict(enumerate(safety))
    for c in [1,2]:
        np.testing.assert_array_equal(allocation_weights(sampler.responsibilities[c],safety[sampler.y==c]), author._get_probas_for_samples_in_component(sampler.X,sampler.y,c))
    np.testing.assert_array_equal(component_quotas([.5,.5],5),[3,2])
    np.testing.assert_array_equal(component_quotas([1,0],0),[0,0])
    assert np.allclose(allocation_weights(np.ones((3,2))*.5,np.ones(3)),[.5,.5])


def test_author_gmm_selection_and_parameters_exact(sampler):
    author = author_harness()
    for key,value in dict(valid_size=.25,random_state=42,min_components=1,max_components=3,n_init=1,max_iter=100,tol=.001,covariance_type="full",likelihood_threshold=0).items():
        setattr(author,key,value)
    author.gaussian_mixtures,author.likelihoods = {},{}
    with threadpool_limits(limits=1):
        author._fit_single_class(sampler.X,sampler.y,1)
        adapted,_ = fit_class_gmm(sampler.X[sampler.y==1],42,SETTINGS)
    model = author.gaussian_mixtures[1]
    assert adapted.n_components == model.n_components
    for key in ("means_","weights_","covariances_"):
        np.testing.assert_array_equal(getattr(adapted,key),getattr(model,key))


def test_quota_bypass_alignment_and_global_rng(sampler):
    np.random.seed(918)
    state = np.random.get_state()
    a = sampler.sample("A",60)
    for arm in ("B","C"):
        candidate = sampler.sample(arm,60,tau=0)
        np.testing.assert_array_equal(a.X,candidate.X)
        np.testing.assert_array_equal(a.y,candidate.y)
    np.testing.assert_array_equal(a.X[:len(sampler.X)],sampler.X)
    np.testing.assert_array_equal(a.y[:len(sampler.y)],sampler.y)
    assert len(a.X)==len(a.y)==len(a.parent)==len(a.kind)==len(a.component)
    assert Counter(a.y) == {0:100,1:60,2:60}
    assert np.array_equal(np.random.get_state()[1],state[1])


def test_rejection_bounded_fallback_and_provenance(sampler):
    # Identical competing models force posterior .5: every .7 proposal fails.
    sampler.models[2] = sampler.models[1]
    c = sampler.sample("C",60,tau=.7,retry_limit=2)
    b = sampler.sample("B",60,tau=.7)
    d = sampler.sample("D",60)
    for row in c.counts:
        assert row["accepted"]==0
        assert row["generated"]==row["requested"]*2
        assert row["fallback"]==row["requested"]==row["realized"]
    assert sum(r["realized"] for r in b.counts)==0
    np.testing.assert_array_equal(c.X,d.X)
    np.testing.assert_array_equal(c.parent,d.parent)
    mask = c.kind=="duplicate"
    np.testing.assert_array_equal(c.X[mask],sampler.X[c.parent[mask]])
    np.testing.assert_array_equal(c.y[mask],sampler.y[c.parent[mask]])
    for idx in np.flatnonzero(mask):
        assert c.parent[idx] in sampler.members[c.y[idx]][c.component[idx]]
    again = sampler.sample("C",60,tau=.7,retry_limit=2)
    np.testing.assert_array_equal(c.X,again.X)


def test_acceptance_matches_saved_density_and_prefix(sampler):
    b = sampler.sample("B",60,tau=.55)
    c = sampler.sample("C",60,tau=.55,retry_limit=2)
    for row in c.counts:
        label,j = row["label"],row["component"]
        key = f"{label}_{j}"
        draw = c.proposals[f"{key}_X"]
        expected,_ = density_accept(np.column_stack([sampler.models[d].score_samples(draw) for d in sampler.labels]),sampler.labels.index(label),.55)
        np.testing.assert_array_equal(expected,c.proposals[f"{key}_accepted"])
        np.testing.assert_array_equal(b.proposals[f"{key}_X"],draw[:row["requested"]])
        assert row["accepted"]+row["fallback"]==row["requested"]
        assert row["generated"]<=2*row["requested"]
        assert row["generated"]==row["accepted"]+row["rejected"]


def test_empty_component_explicit_failure(sampler):
    for j in sampler.members[1]:
        sampler.members[1][j] = np.array([],int)
    with pytest.raises(ValueError,match="no original hard-MAP"):
        sampler.sample("C",60,tau=.7)
    with pytest.raises(ValueError):
        sampler.sample("X",60)


def test_data_pairing_partition_separation_and_arbitrary_labels(sampler):
    a = generate_partition(42,100,[100,40,40],1.5,.9)
    b = generate_partition(42,100,[100,40,40],4.,.9)
    np.testing.assert_array_equal(a["id"],b["id"])
    np.testing.assert_allclose(a["X"][:,0],b["X"][:,0])
    assert not set(a["id"]) & set(generate_partition(42,200,[100,40,40],1.5,.9)["id"])
    with threadpool_limits(limits=1):
        labels = np.array(["major","small-a","small-b"])[sampler.y]
        s = OverlapSampler(sampler.X,labels,["small-a","small-b"],42,SETTINGS)
        result = s.sample("D",60)
        assert set(result.y) == {"major","small-a","small-b"}
        assert Counter(result.y) == {"major":100,"small-a":60,"small-b":60}


def test_pipeline_training_only_and_selection_order(tmp_path,monkeypatch):
    import src.gmmsampling_overlap_study as study
    cfg = json.loads(CONFIG.read_text())
    cfg.update(partition_counts={"fit":[60,16,16],"validation":[60,16,16],"test":[60,16,16]},target=24,classifier_C=[1.],gmm={**SETTINGS,"max_components":2})
    seen = []
    original = study.LogisticRegression
    class Spy:
        def __init__(self,**kw): self.model=original(**kw)
        def fit(self,X,y):
            seen.append(("fit",np.array(X)))
            self.model.fit(X,y)
            self.classes_=self.model.classes_
            self.coef_,self.intercept_=self.model.coef_,self.model.intercept_
            return self
        def predict_proba(self,X):
            seen.append(("predict",np.array(X)))
            return self.model.predict_proba(X)
    monkeypatch.setattr(study,"LogisticRegression",Spy)
    with threadpool_limits(limits=1):
        rows = run_case(cfg,42,1.5,.5,tmp_path)
    assert len(rows)==6
    raw = np.load(tmp_path/"original_partitions.npz")
    scale = np.load(tmp_path/"density_state.npz")
    np.testing.assert_allclose(scale["scaler_mean"],raw["fit_X"].mean(axis=0))
    fit_scaled = (raw["fit_X"]-scale["scaler_mean"])/scale["scaler_scale"]
    for op,X in seen:
        if op=="fit":
            np.testing.assert_array_equal(X[:len(fit_scaled)],fit_scaled)
    # All fit calls precede the final six predict operations (selected test only).
    assert all(op=="predict" for op,_ in seen[-6:])
    for arm in cfg["methods"]:
        saved = np.load(tmp_path/f"selected_{arm}_test.npz")
        np.testing.assert_array_equal(saved["sample_ids"],raw["test_id"])
        assert np.isfinite(saved["probabilities"]).all()
        np.testing.assert_allclose(saved["probabilities"].sum(axis=1),1)
