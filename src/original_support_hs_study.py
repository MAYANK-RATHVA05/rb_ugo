"""Bounded, prospectively configured support-aware HS feasibility study."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.stats import t
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_dataset
from src.original_support_hs import SupportAwareHS, original_node_counts


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/original_support_hs_v1.json'
FROZEN_DESIGN = '7330b7ab05d3a7dbe1699fb94c980c6e66ad09dea157b716390f77497cc8ff1f'


def design_hash(design):
    return hashlib.sha256(json.dumps(design,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def load_config(path=CONFIG):
    config=json.loads(Path(path).read_text())
    digest=design_hash(config['scientific_design'])
    if digest != FROZEN_DESIGN or config['design_sha256'] != digest:
        raise ValueError('Configuration differs from frozen HS pilot design.')
    return config


@dataclass
class Partition:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray | None = None


@dataclass
class Case:
    dataset: str
    seed: int
    concentration: float | None
    fitting: Partition
    validation: Partition
    test: Partition
    provenance: dict


def array_hash(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def load_real_case(entry, seed, root=ROOT):
    path=root/entry['relative_path']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f"Source fingerprint differs: {entry['dataset_id']}")
    loaded=load_dataset(path,target_column=entry['target_column'],minority_label=entry['positive_label'])
    X,y=loaded.X.to_numpy(float),loaded.y.to_numpy()
    if not np.isfinite(X).all() or X.shape != (entry['rows'],entry['input_features']):
        raise ValueError('Real input shape or numerical contract failed.')
    _, coordinate_groups=np.unique(X,axis=0,return_inverse=True)
    outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
    development,test=next(outer.split(X,y,coordinate_groups))
    inner=StratifiedGroupKFold(4,shuffle=True,random_state=seed+100)
    fit_local,val_local=next(inner.split(X[development],y[development],coordinate_groups[development]))
    fit,val=development[fit_local],development[val_local]
    for a,b in ((fit,val),(fit,test),(val,test)):
        if set(coordinate_groups[a]) & set(coordinate_groups[b]):
            raise ValueError('Feature-coordinate groups cross partition boundaries.')
    for indices in (fit,val,test):
        if len(np.unique(y[indices])) != 2:
            raise ValueError('Predeclared partition lacks a class; no redraw allowed.')
    return Case(entry['dataset_id'],seed,None,Partition(X[fit],y[fit]),
                Partition(X[val],y[val]),Partition(X[test],y[test]),
                {'source_sha256':digest,'source_fidelity':entry['fidelity'],
                 'fit_indices':fit.tolist(),'validation_indices':val.tolist(),
                 'test_indices':test.tolist(),
                 'coordinate_groups_sha256':array_hash(coordinate_groups)})


def synthetic_case(design, seed, concentration):
    spec=design['synthetic']
    parts=[]
    for partition_name,stream in zip(('fitting','validation','test'),spec['partition_streams']):
        rng=np.random.default_rng(np.random.SeedSequence([seed,stream]))
        n0,n1,n2=spec[partition_name+'_counts']
        X=np.vstack([rng.normal(spec[name+'_mean'],spec[name+'_sd'],size=(count,2))
                     for name,count in zip(('majority','common','rare'),(n0,n1,n2))])
        y=np.r_[np.zeros(n0,dtype=int),np.ones(n1+n2,dtype=int)]
        groups=np.r_[np.zeros(n0,dtype=int),np.ones(n1,dtype=int),np.full(n2,2)]
        order=rng.permutation(len(X))
        parts.append(Partition(X[order],y[order],groups[order]))
    return Case('synthetic_local_concentration',seed,float(concentration),*parts,
                {'source':'prospective independent Gaussian partition streams',
                 'concentration':concentration})


def concentrated_smote(X, y, concentration, seed, *, k_neighbors=3):
    """Controlled SMOTE interpolation; seed concentration uses features only.

    Same count, minority kNN pool and random uniforms at all concentration
    levels. Latent subgroup identities are never accepted by this API.
    """
    minority=np.flatnonzero(y==1)
    if len(minority)<=k_neighbors:
        raise ValueError('Insufficient fitting minority rows for controlled SMOTE.')
    points=X[minority]
    # Stable ties, excluding only the query row, not duplicate coordinates.
    neighbors=[]
    for i,row in enumerate(points):
        distances=np.linalg.norm(points-row,axis=1)
        distances[i]=np.inf
        neighbors.append(np.argsort(distances,kind='stable')[:k_neighbors])
    neighbors=np.asarray(neighbors)
    anchor=points[np.argmax(points[:,0])]
    log_weights=-.5*np.sum((points-anchor)**2,axis=1)/(.75**2)
    weights=np.exp(log_weights-log_weights.max()); weights/=weights.sum()
    q=(1-concentration)/len(points)+concentration*weights
    q/=q.sum()
    n=int(np.sum(y==0)-len(minority))
    rng=np.random.default_rng(seed)
    uniforms=rng.random((n,3))
    seed_pos=np.searchsorted(np.cumsum(q),uniforms[:,0],side='right')
    seed_pos=np.minimum(seed_pos,len(points)-1)
    partner_pos=neighbors[seed_pos,(uniforms[:,1]*k_neighbors).astype(int)]
    fraction=uniforms[:,2,None]
    synthetic=points[seed_pos]+fraction*(points[partner_pos]-points[seed_pos])
    trace={'seed_indices':minority[seed_pos],'partner_indices':minority[partner_pos],
           'fractions':uniforms[:,2],'seed_probabilities':q}
    return np.vstack((X,synthetic)),np.r_[y,np.ones(n,dtype=y.dtype)],trace


def score_probabilities(y,proba):
    pred=(proba>=.5).astype(int)
    return {'average_precision':float(average_precision_score(y,proba)),
            'minority_f1':float(f1_score(y,pred,zero_division=0))}


def tree_fingerprint(tree):
    state=tree.tree_.__getstate__()
    return hashlib.sha256(state['nodes'].tobytes()+state['values'].tobytes()).hexdigest()


def tune_models(X_fit,y_fit,X_val,y_val,design,seed,concentration=None):
    """Selection boundary: cannot receive any test rows, labels or subgroup IDs."""
    started=perf_counter()
    scaler=StandardScaler().fit(X_fit)
    X=scaler.transform(X_fit); V=scaler.transform(X_val)
    if concentration is None:
        aug_X,aug_y=SMOTE(k_neighbors=design['smote']['k_neighbors'],
                          random_state=seed).fit_resample(X,y_fit)
        trace=None
    else:
        aug_X,aug_y,trace=concentrated_smote(X,y_fit,concentration,seed)
    selected={}; candidates=[]; fit_count=0
    arm_seconds={arm:0. for arm in design['arms']}

    def consider(arm,model,parameters,candidate_index,base_fingerprint):
        scores=score_probabilities(y_val,model.predict_proba(V)[:,1])
        row={'arm':arm,'candidate_index':candidate_index,'parameters':parameters,
             'base_tree_sha256':base_fingerprint,**scores}
        candidates.append(row)
        # Remaining ties preserve grid order (first candidate).
        key=(scores['average_precision'],scores['minority_f1'])
        if arm not in selected or key>selected[arm]['selection_key']:
            selected[arm]={'model':model,'selection_key':key,'candidate':row}

    for arm,oversample,weighted in [('A_SMOTE_TREE',True,False),
                                   ('D_WEIGHTED_TREE',False,True),
                                   ('E_NORES_TREE',False,False)]:
        arm_start=perf_counter()
        for index,parameters in enumerate(design['tree_grid']):
            tree=DecisionTreeClassifier(random_state=seed,class_weight='balanced' if weighted else None,
                                        **parameters)
            tree.fit(aug_X if oversample else X,aug_y if oversample else y_fit)
            fit_count+=1
            consider(arm,tree,parameters,index,tree_fingerprint(tree))
        arm_seconds[arm]=perf_counter()-arm_start

    for structure_index,parameters in enumerate(design['hs_tree_grid']):
        start=perf_counter()
        tree=DecisionTreeClassifier(random_state=seed,**parameters).fit(aug_X,aug_y)
        fit_count+=1
        fingerprint=tree_fingerprint(tree)
        tree_seconds=perf_counter()-start
        counts=original_node_counts(tree,X)
        for arm,mode,root in [('B_SMOTE_STANDARD_HS','augmented',float(tree.tree_.weighted_n_node_samples[0])),
                              ('C_SMOTE_ORIGINAL_HS','original',float(counts[0]))]:
            arm_start=perf_counter()
            for lambda_index,relative in enumerate(design['relative_lambdas']):
                lam=relative*root
                model=SupportAwareHS(tree,reg_param=lam,support_mode=mode).fit(X if mode=='original' else None)
                consider(arm,model,{**parameters,'relative_lambda':relative,'lambda':lam},
                         structure_index*len(design['relative_lambdas'])+lambda_index,fingerprint)
            arm_seconds[arm]+=perf_counter()-arm_start+tree_seconds
        if tree_fingerprint(tree)!=fingerprint:
            raise ValueError('HS altered shared base-tree structure/raw predictions.')

    for arm in design['arms']:
        if sum(row['arm']==arm for row in candidates)!=12:
            raise ValueError('Candidate budget differs between arms.')
    return {'scaler':scaler,'selected':selected,'candidates':candidates,'fit_count':fit_count,
            'selection_seconds':perf_counter()-started,'arm_seconds':arm_seconds,
            'original_scaled':X,'augmented_scaled':aug_X,'augmentation_trace':trace,
            'augmented_y':aug_y}


def evaluate_case(case,design):
    started=perf_counter()
    # Once tune_models returns, every model choice is locked.
    fitted=tune_models(case.fitting.X,case.fitting.y,case.validation.X,case.validation.y,
                       design,case.seed,case.concentration)
    test_X=fitted['scaler'].transform(case.test.X)
    vectors={'test_y':case.test.y}
    if case.test.groups is not None:
        vectors['test_groups']=case.test.groups
    rows=[]
    for arm,selected in fitted['selected'].items():
        evaluation_start=perf_counter()
        model=selected['model']
        proba=model.predict_proba(test_X)[:,1]
        vectors[arm]=proba
        tree=model.estimator if isinstance(model,SupportAwareHS) else model
        counts=original_node_counts(tree,fitted['original_scaled'])
        if isinstance(model,SupportAwareHS):
            vectors[arm+'__original_counts']=counts
            vectors[arm+'__augmented_counts']=tree.tree_.weighted_n_node_samples.copy()
            vectors[arm+'__left']=tree.tree_.children_left.copy()
            vectors[arm+'__right']=tree.tree_.children_right.copy()
            vectors[arm+'__feature']=tree.tree_.feature.copy()
            vectors[arm+'__threshold']=tree.tree_.threshold.copy()
            vectors[arm+'__raw']=model.raw_probabilities_.copy()
            vectors[arm+'__shrunk']=model.node_probabilities_.copy()
            vectors[arm+'__test_leaves']=tree.apply(test_X)
        internal=tree.tree_.children_left>=0
        nonzero=internal&(counts>0)
        ratios=tree.tree_.n_node_samples[nonzero]/counts[nonzero]
        candidate=selected['candidate']
        row={'arm':arm,**score_probabilities(case.test.y,proba),
             'rare_recall':float(np.mean(proba[case.test.groups==2]>=.5))
                           if case.test.groups is not None and np.any(case.test.groups==2) else None,
             'selected_parameters':candidate['parameters'],
             'selected_candidate':candidate['candidate_index'],
             'validation_AP':candidate['average_precision'],
             'validation_F1':candidate['minority_f1'],
             'base_tree_sha256':candidate['base_tree_sha256'],
             'nodes':tree.tree_.node_count,'depth':tree.tree_.max_depth,
             'zero_original_nodes':int(np.sum(counts==0)),
             'zero_original_internal_nodes':int(np.sum(internal&(counts==0))),
             'internal_count_ratio_min':float(ratios.min()) if len(ratios) else None,
             'internal_count_ratio_max':float(ratios.max()) if len(ratios) else None,
             'selection_seconds_including_shared_base':fitted['arm_seconds'][arm],
             'test_seconds':perf_counter()-evaluation_start}
        rows.append(row)
    metadata={'dataset':case.dataset,'seed':case.seed,'concentration':case.concentration,
              'status':'success','fits':fitted['fit_count'],
              'seconds':perf_counter()-started,'provenance':case.provenance,
              'partitions':{name:{'X':array_hash(getattr(case,name).X),
                                  'y':array_hash(getattr(case,name).y),
                                  'rows':len(getattr(case,name).y)}
                            for name in ['fitting','validation','test']},
              'scaler_mean':fitted['scaler'].mean_.tolist(),
              'scaler_rows':int(fitted['scaler'].n_samples_seen_),
              'augmentation_X_sha256':array_hash(fitted['augmented_scaled']),
              'augmentation_y_sha256':array_hash(fitted['augmented_y']),
              'rows':rows,'candidates':fitted['candidates']}
    return metadata,vectors


def paired_summary(scores):
    """Case settings separate; paired seeds are the unit, never individual rows."""
    result=[]
    for (dataset,concentration),frame in scores.groupby(['dataset','concentration'],dropna=False):
        for control in ['B_SMOTE_STANDARD_HS','A_SMOTE_TREE','D_WEIGHTED_TREE','E_NORES_TREE']:
            ours=frame[frame.arm=='C_SMOTE_ORIGINAL_HS'].set_index('seed')
            other=frame[frame.arm==control].set_index('seed')
            common=ours.index.intersection(other.index)
            for metric in ['average_precision','minority_f1','rare_recall']:
                delta=(ours.loc[common,metric]-other.loc[common,metric]).dropna()
                if not len(delta):
                    continue
                n=len(delta); mean=float(delta.mean()); sd=float(delta.std(ddof=1)) if n>1 else None
                half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
                result.append({'dataset':dataset,'concentration':concentration,'control':control,
                               'metric':metric,'n_pairs':n,'mean_C_minus_control':mean,
                               'sample_sd':sd,'descriptive_95_low':mean-half if half is not None else None,
                               'descriptive_95_high':mean+half if half is not None else None,
                               'wins':int((delta>1e-12).sum()),'ties':int((abs(delta)<=1e-12).sum()),
                               'losses':int((delta< -1e-12).sum())})
    return result


def decision(scores,design,failures):
    if failures or len(scores)!=150:
        return {'status':'incomplete_pilot_no_advancement','failure_count':len(failures),
                'checks':{'complete_coverage':False},
                'scope':'Incomplete coverage; do not interpret success-only averages as full evidence.'}
    def deltas(real,metric):
        frame=scores[scores.dataset.ne('synthetic_local_concentration') if real else
                     scores.dataset.eq('synthetic_local_concentration')]
        pivot=frame.pivot(index=['dataset','concentration','seed'],columns='arm',values=metric)
        return pivot['C_SMOTE_ORIGINAL_HS']-pivot['B_SMOTE_STANDARD_HS']
    rule=design['advancement']
    real_ap=deltas(True,'average_precision'); real_f1=deltas(True,'minority_f1')
    dataset_ap=real_ap.groupby(level='dataset').mean()
    synthetic_ap=deltas(False,'average_precision')
    synthetic_rare=deltas(False,'rare_recall')
    checks={
        'complete_coverage':len(scores)==150 and not failures,
        'real_dataset_AP_wins':int((dataset_ap>=.01).sum())>=rule['minimum_real_datasets_with_mean_AP_gain_at_least_0_01'],
        'real_macro_AP':float(dataset_ap.mean())>=rule['real_macro_AP_gain_at_least'],
        'real_macro_F1':float(real_f1.mean())>=rule['real_macro_F1_delta_at_least'],
        'synthetic_macro_AP':float(synthetic_ap.mean())>=rule['synthetic_macro_AP_gain_at_least'],
        'synthetic_rare_recall':float(synthetic_rare.mean())>=rule['synthetic_macro_rare_recall_delta_at_least'],
        'no_large_real_AP_loss':bool((dataset_ap>=rule['no_real_dataset_mean_AP_loss_below']).all())}
    return {'status':'supports_broader_evaluation' if all(checks.values()) else 'advancement_trigger_not_met',
            'checks':checks,'real_mean_AP_deltas':dataset_ap.to_dict(),
            'real_macro_AP_delta':float(dataset_ap.mean()),'real_macro_F1_delta':float(real_f1.mean()),
            'synthetic_macro_AP_delta':float(synthetic_ap.mean()),
            'synthetic_macro_rare_recall_delta':float(synthetic_rare.mean()),
            'failure_count':len(failures),
            'scope':'Bounded feasibility decision; no significance, verified novelty or generality claim.'}
