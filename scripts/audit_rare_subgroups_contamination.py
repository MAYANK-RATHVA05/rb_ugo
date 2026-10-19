"""Read-only audit of frozen contamination artifacts; no classifier fitting."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from src.results import content_hash
from src.rare_subgroups_protocol import array_hash

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected, name='value'):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), (name, actual, expected)


def ap(labels, probabilities):
    ordered = np.argsort(-probabilities, kind='stable')
    scores, labels = probabilities[ordered], labels[ordered]
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores)-1]
    tp = np.cumsum(labels, dtype=np.int64)[ends]
    return float(np.sum(np.diff(np.r_[0., tp / labels.sum()]) * (tp / (ends+1))))


def regenerate(d, job):
    result = {}
    nrare = job['rare_count']
    counts = {'fitting': {'majority': 1000, 'common': 100-nrare, 'rare': nrare},
              'selection': {'majority': 2000, 'common': 180, 'rare': 20},
              'calibration': {'majority': 2000, 'common': 0, 'rare': 0},
              'test': {'majority': 20000, 'common': 1800, 'rare': 200}}
    for partition, pid in [('fitting',1),('selection',2),('calibration',3),('test',4)]:
        features, groups = [], []
        for component, name in enumerate(('majority','common','rare')):
            count = counts[partition][name]
            pool = (100 if component == 1 else 16) if partition == 'fitting' and component > 0 else count
            draw = np.random.default_rng(np.random.SeedSequence([job['seed'],pid,component])).standard_normal((pool,2))[:count]
            mean, scale = ((0.,0.),1.) if component == 0 else (((3.,0.),.35) if component == 1 else ((0.,3.),job['rare_sigma']))
            features.append(np.asarray(mean)+scale*draw)
            groups.append(np.full(count,component,dtype=np.int8))
        X, group = np.vstack(features), np.concatenate(groups)
        order = np.random.default_rng(np.random.SeedSequence([job['seed'],pid,99])).permutation(len(group))
        X, group = X[order], group[order]
        truth = (group > 0).astype(np.int8)
        part = {'X':X, 'y':truth.copy(), 'groups':group}
        if partition == 'fitting':
            chosen = np.random.default_rng(np.random.SeedSequence([job['seed'],5,100])).permutation(1000)[:job['flipped_count']]
            flags = np.isin(order,chosen)
            part['y'][flags] = 1
            part.update(true_y=truth,corrupted=flags,stable_ids=order)
        result[partition] = part
    return result


def audit_aw(candidate, X, observed, generated):
    fit, trace = candidate['fit'], candidate['trace']
    minority = np.flatnonzero(observed == 1)
    distances = np.linalg.norm(X[minority,None,:]-X[None,minority,:],axis=2)
    np.fill_diagonal(distances,np.inf)
    fifth = np.sort(distances,axis=1)[:,4]
    epsilon = fit['resampler_diagnostics']['epsilon']
    close(epsilon,np.percentile(fifth,candidate['eps_percentile'],method='linear'),'epsilon')
    labels = np.asarray(trace['cluster_labels'])
    # A reference-library replay checks cluster identity. Distance/weight/quota
    # arithmetic below is independently reconstructed with NumPy.
    replay = DBSCAN(eps=epsilon,min_samples=candidate['min_samples'],metric='euclidean',algorithm='brute',n_jobs=1).fit_predict(X[minority])
    np.testing.assert_array_equal(labels[minority],replay)
    assert np.all(labels[observed==0] == -2)
    eligible = np.asarray(trace['eligible_indices'],dtype=int)
    cleaned = np.sort(np.r_[np.flatnonzero(observed==0),eligible])
    np.testing.assert_array_equal(np.setdiff1d(minority,eligible),trace['excluded_indices'])
    near = np.linalg.norm(X[eligible,None,:]-X[None,cleaned,:],axis=2)
    near[eligible[:,None] == cleaned[None,:]] = np.inf
    neighbors = cleaned[np.argsort(near,kind='stable',axis=1)[:,:5]]
    np.testing.assert_array_equal(neighbors,trace['boundary_neighbors'])
    boundary = np.asarray(trace['boundary_counts'])
    np.testing.assert_array_equal(boundary[eligible],(observed[neighbors]==0).sum(axis=1))
    clusters = np.unique(labels[eligible])
    weights = np.array([boundary[labels==c].max() for c in clusters])
    np.testing.assert_array_equal(weights,trace['cluster_weights'])
    effective = weights if weights.any() else np.ones(len(weights))
    quotas = np.floor(effective / effective.sum()*generated+.5).astype(int)
    quotas[np.argmax(effective)] += generated-int(quotas.sum())
    np.testing.assert_array_equal(quotas,trace['cluster_quotas'])
    assert quotas.sum() == generated and np.all(quotas>=0)
    for cluster in clusters:
        ids = np.flatnonzero(labels==cluster)
        b = boundary[ids].astype(float)
        b = b/b.sum() if b.any() else np.full(len(ids),1/len(ids))
        local = np.linalg.norm(X[ids,None,:]-X[None,ids,:],axis=2)
        density = 1/(local.sum(axis=1)/(len(ids)-1)+1e-8)
        density /= density.sum()
        alpha = 1/(1+np.exp(-5*(b-.2)))
        probabilities = alpha*b+(1-alpha)*density
        probabilities /= probabilities.sum()
        np.testing.assert_allclose(probabilities,np.asarray(trace['sample_probabilities'])[ids],rtol=1e-12,atol=1e-14)
    seeds = np.asarray(trace['generated_seed_indices'],dtype=int)
    partners = np.asarray(trace['generated_partner_indices'],dtype=int)
    fractions = np.asarray(trace['interpolation_fractions'])
    assert len(seeds)==len(partners)==len(fractions)==generated
    assert np.all(seeds!=partners) and np.all((fractions>=0)&(fractions<1))
    assert np.all(observed[seeds]==1) and np.all(observed[partners]==1)
    np.testing.assert_array_equal(labels[seeds],labels[partners])
    np.testing.assert_array_equal(labels[seeds],trace['generated_cluster_labels'])
    for cluster, quota in zip(clusters,quotas):
        assert np.count_nonzero(labels[seeds]==cluster)==quota
    rng=np.random.default_rng(fit['classifier_parameters']['random_state'])
    draw_seeds=[];draw_partners=[];draw_fractions=[]
    for cluster,quota in zip(clusters,quotas):
        members=np.flatnonzero(labels==cluster)
        probabilities=np.asarray(trace['sample_probabilities'])[members]
        allocation=np.bincount(rng.choice(len(members),size=int(quota),p=probabilities),minlength=len(members))
        for local,count in enumerate(allocation):
            others=np.delete(members,local)
            for _ in range(int(count)):
                draw_seeds.append(members[local]);draw_partners.append(rng.choice(others));draw_fractions.append(rng.uniform(0.,1.))
    np.testing.assert_array_equal(seeds,draw_seeds)
    np.testing.assert_array_equal(partners,draw_partners)
    np.testing.assert_array_equal(fractions,draw_fractions)
    synthetic = (1-fractions[:,None])*X[seeds]+fractions[:,None]*X[partners]
    assert array_hash(np.vstack((X[cleaned],synthetic))) == fit['output_X_sha256']
    output_y = np.r_[observed[cleaned],np.ones(generated,dtype=observed.dtype)]
    assert array_hash(output_y) == fit['output_y_sha256']
    return generated


def audit_screen(records, d, saved):
    index = {(r['job']['rare_count'],r['job']['rare_sigma'],r['job']['flipped_count'],r['job']['seed'],row['method']):row
             for r in records.values() for row in r['rows']}
    qualifying = set()
    for assessed in saved['settings']:
        r,s,m = assessed['rare_count'],assessed['rare_sigma'],assessed['flipped_count']
        groups = {(noise,method):[index[r,s,noise,seed,method] for seed in range(42,47)]
                  for noise in (0,m) for method in ('AW_REMOVE','AW_SEED_FILTER_RETAIN')}
        D,K,D0,K0 = (groups[key] for key in ((m,'AW_REMOVE'),(m,'AW_SEED_FILTER_RETAIN'),(0,'AW_REMOVE'),(0,'AW_SEED_FILTER_RETAIN')))
        delta = [k['metrics']['rare_recall']-dd['metrics']['rare_recall'] for k,dd in zip(K,D)]
        dc = statistics.mean(k['metrics']['common_recall']-dd['metrics']['common_recall'] for k,dd in zip(K,D))
        df = statistics.mean(abs(k['metrics']['test_fpr']-dd['metrics']['test_fpr']) for k,dd in zip(K,D))
        interaction = [(k['metrics']['ap']-k0['metrics']['ap'])-(dd['metrics']['ap']-dd0['metrics']['ap'])
                       for k,k0,dd,dd0 in zip(K,K0,D,D0)]
        removed = {g:sum(row['group_retention'][g]['removed'] for row in D) for g in ('common','rare','noise')}
        rare_ok = removed['rare']>=1 and statistics.mean(delta)>=.05-1e-12 and sum(v>1e-12 for v in delta)>=4 and dc>=-.02-1e-12 and df<=.01+1e-12
        noise_ok = removed['noise']>=1 and statistics.mean(interaction)<=-.02+1e-12 and sum(v < -1e-12 for v in interaction)>=4
        assert assessed['rare_protection'] == rare_ok and assessed['noise_tradeoff'] == noise_ok
        assert assessed['qualifies'] == bool(rare_ok and noise_ok)
        assert assessed['removed_originals']==removed
        close(assessed['mean_rare_recall_difference'],statistics.mean(delta))
        close(assessed['mean_AP_interaction'],statistics.mean(interaction))
        close(assessed['mean_absolute_test_fpr_difference'],df)
        if rare_ok and noise_ok: qualifying.add((r,s))
    covered = set()
    for assessed in saved['ordinary_controls']:
        r,s,method = assessed['rare_count'],assessed['rare_sigma'],assessed['method']
        okay = []
        for m,check in zip((0,5,10),assessed['levels']):
            C=[index[r,s,m,seed,method] for seed in range(42,47)]
            K=[index[r,s,m,seed,'AW_SEED_FILTER_RETAIN'] for seed in range(42,47)]
            D=[index[r,s,m,seed,'AW_REMOVE'] for seed in range(42,47)]
            mean=lambda rows,key:statistics.mean(row['metrics'][key] for row in rows)
            condition=(mean(C,'rare_recall')>=mean(K,'rare_recall')-.02-1e-12
                       and mean(C,'common_recall')>=mean(K,'common_recall')-.02-1e-12
                       and mean(C,'ap')>=max(mean(K,'ap'),mean(D,'ap'))-.01-1e-12
                       and statistics.mean(abs(k['metrics']['test_fpr']-c['metrics']['test_fpr']) for k,c in zip(K,C))<=.01+1e-12)
            assert check['covers']==condition
            okay.append(condition)
        assert assessed['covers_all_noise_levels']==all(okay)
        if all(okay): covered.add((r,s))
    status='tradeoff_trigger_not_met' if len(qualifying)<2 else ('simple_controls_suffice' if qualifying<=covered else 'residual_tradeoff_requires_replication')
    assert saved['qualifying_base_settings']==len(qualifying) and saved['status']==status
    return status


def audit(run: Path, clean_reference: Path | None = None) -> dict:
    RUN=Path(run)
    envelope=json.loads((RUN/'manifest.json').read_text())
    manifest=envelope['manifest']; cfg=manifest['config']; d=cfg['scientific_design']
    assert envelope['run_id']==content_hash(manifest)
    assert cfg['execution_enabled'] is True and manifest['git']['tracked_changes'] is False
    assert hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==manifest['design_sha256']=='adfdb2961963ff83b9552e84b06963689f938d853859d9bafc224106a08849db'
    for name,digest in manifest['source_sha256'].items():
        original=subprocess.check_output(['git','show',manifest['git']['commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest()==digest,name
    original_config=subprocess.check_output(['git','show',manifest['git']['commit']+':configs/rare_subgroups_contamination_v1.json'],cwd=ROOT)
    assert hashlib.sha256(original_config).hexdigest()==manifest['config_file_sha256']
    records={}; metrics_checked=candidates_checked=interpolations_checked=smote_checked=ros_checked=0
    max_metric_error=max_selection_error=0.; fprs=[]; counts={}; paired_hashes={}; clean_reference_vectors=0
    for job in manifest['jobs']:
        payload=json.loads((RUN/f'job-{content_hash(job)}.json').read_text())
        record=payload['record']
        assert payload['sha256']==content_hash(record) and record['run_id']==envelope['run_id'] and record['job']==job
        assert record['status']=='success' and record['case_status']=='complete' and record['classifier_fit_attempts']==14
        records[tuple(job[k] for k in ('rare_count','rare_sigma','flipped_count','seed'))]=record
        artifact=RUN/record['predictions']['path']
        assert artifact.name==f'case-{content_hash(job)}.npz'
        assert hashlib.sha256(artifact.read_bytes()).hexdigest()==record['predictions']['sha256']
        data=regenerate(d,job)
        fitting=data['fitting']; observed=fitting['y']; truth=fitting['true_y']; flags=fitting['corrupted']; X0=fitting['X']
        generated=900-2*job['flipped_count']
        assert np.sum(observed!=truth)==job['flipped_count'] and np.all(truth[flags]==0)
        assert np.sum(observed==0)==1000-job['flipped_count'] and np.sum(observed==1)==100+job['flipped_count']
        for name,part in data.items():
            for field,values in part.items():
                assert array_hash(values)==record['partitions_sha256'][name][field],(job,name,field)
        base=(job['rare_count'],job['rare_sigma'],job['seed'])
        hashes={field:record['partitions_sha256']['fitting'][field] for field in ('X','true_y','groups','stable_ids')}
        assert paired_hashes.setdefault(base,hashes)==hashes
        np.testing.assert_allclose(record['scaler']['mean'],X0.mean(axis=0),rtol=1e-13,atol=1e-14)
        np.testing.assert_allclose(record['scaler']['scale'],X0.std(axis=0),rtol=1e-13,atol=1e-14)
        assert record['scaler']['fitted_rows']==1100
        X=(X0-np.asarray(record['scaler']['mean']))/np.asarray(record['scaler']['scale'])
        assert array_hash(X)==record['scaled_fitting_X_sha256']
        with np.load(artifact,allow_pickle=False) as vectors:
            assert set(vectors.files)==set(record['vector_sha256'])
            for name in vectors.files: assert array_hash(vectors[name])==record['vector_sha256'][name]
            for name,values in [('selection_y',data['selection']['y']),('calibration_y',data['calibration']['y']),('test_y',data['test']['y']),('test_groups',data['test']['groups'])]:
                np.testing.assert_array_equal(vectors[name],values)
            for field in ('y','true_y','groups','corrupted','stable_ids'):
                np.testing.assert_array_equal(vectors['fitting_'+field],fitting[field])
            for label,pid,n in [('calibration',3,2000),('test',4,22000)]:
                expected=np.random.default_rng(np.random.SeedSequence([job['seed'],pid,101])).random(n)
                np.testing.assert_array_equal(vectors[label+'_ties'],expected)
            for candidate in record['selection_candidates']:
                assert candidate['status']=='success'
                candidates_checked+=1
                error=abs(ap(vectors['selection_y'],vectors[candidate['selection_vector']])-candidate['validation_ap'])
                max_selection_error=max(max_selection_error,error)
                assert error<1e-12
                interpolations_checked+=audit_aw(candidate,X,observed,generated)
            best=min(record['selection_candidates'],key=lambda c:(-c['validation_ap'],0 if(c['min_samples'],c['eps_percentile'])==(2,65) else 1,c['min_samples'],c['eps_percentile']))
            rows={row['method']:row for row in record['rows']}
            assert len(rows)==7 and len(record['selection_candidates'])==9
            tuned=rows['AW_REMOVE_TUNED']
            assert tuned['selected_candidate_id']==best['candidate_id']
            assert tuned['selected_parameters']=={key:best[key] for key in ('min_samples','eps_percentile')}
            assert tuned['output_X_sha256']==best['fit']['output_X_sha256']
            assert rows['AW_REMOVE']['trace']==rows['AW_SEED_FILTER_RETAIN']['trace']
            for method,row in rows.items():
                assert row['status']=='success'
                m=row['metrics']; cal=vectors[row['vectors']['calibration']]; scores=vectors[row['vectors']['test']]
                ucal=vectors['calibration_ties']; utest=vectors['test_ties']
                assert cal.shape==(2000,) and scores.shape==(22000,) and np.isfinite(cal).all() and np.isfinite(scores).all()
                assert np.all((cal>=0)&(cal<=1)) and np.all((scores>=0)&(scores<=1))
                cutoff=sorted(zip(cal.tolist(),ucal.tolist()),reverse=True)[99]
                assert cutoff==(m['threshold'],m['tie_cutoff'])
                pcal=(cal>cutoff[0])|((cal==cutoff[0])&(ucal>=cutoff[1]))
                assert int(pcal.sum())==m['calibration_fp']==100 and m['calibration_fpr']==.05 and m['calibration_negatives']==2000
                predicted=(scores>cutoff[0])|((scores==cutoff[0])&(utest>=cutoff[1]))
                y,groups=vectors['test_y'],vectors['test_groups']
                fp=int(predicted[y==0].sum()); tp=int(predicted[y==1].sum())
                common=float(predicted[groups==1].mean()); rare=float(predicted[groups==2].mean())
                expected={'rare_recall':rare,'common_recall':common,'positive_recall':tp/2000,'test_fpr':fp/20000,
                          'f1':2*tp/(tp+fp+2000),'ap':ap(y,scores),'worst_group_recall':min(common,rare)}
                for key,value in expected.items():
                    error=abs(value-m[key]); max_metric_error=max(max_metric_error,error)
                    assert error<1e-12,(job,method,key,error)
                assert m['test_fp']==fp and m['test_tp']==tp and m['test_negatives']==20000 and m['test_positives']==2000
                assert m['common_count']==1800 and m['rare_count']==200
                assert m['common_tp']==int(predicted[groups==1].sum()) and m['rare_tp']==int(predicted[groups==2].sum())
                retained=np.asarray(row['retained_indices'],dtype=int)
                masks={'common':fitting['groups']==1,'rare':fitting['groups']==2,'noise':flags}
                for group,mask in masks.items():
                    n=int(mask.sum()); kept=int(mask[retained].sum())
                    expected_ret={'original':n,'retained':kept,'removed':n-kept,'fraction':kept/n if n else None}
                    assert row['group_retention'][group]==expected_ret
                trace=row['trace']; diag=row['parent_diagnostics']
                if method in ('NORES','WEIGHTED'):
                    assert row['generated_count']==0
                    assert array_hash(X)==row['output_X_sha256'] and array_hash(observed)==row['output_y_sha256']
                elif method=='ROS':
                    ids=np.asarray(trace['duplicated_indices'],dtype=int)
                    assert len(ids)==generated and np.all(observed[ids]==1)
                    expected_ids=np.random.RandomState(job['seed']).choice(np.flatnonzero(observed==1),size=generated,replace=True)
                    np.testing.assert_array_equal(ids,expected_ids)
                    assert array_hash(np.vstack((X,X[ids])))==row['output_X_sha256']
                    assert array_hash(np.r_[observed,observed[ids]])==row['output_y_sha256']
                    assert diag['corrupted_parents']==int(flags[ids].sum())
                    close(diag['corrupted_parent_fraction'],flags[ids].mean())
                    ros_checked+=len(ids)
                else:
                    seed_ids=np.asarray(trace['generated_seed_indices'],dtype=int); partner_ids=np.asarray(trace['generated_partner_indices'],dtype=int)
                    fractions=np.asarray(trace['interpolation_fractions'])
                    assert len(seed_ids)==len(partner_ids)==len(fractions)==generated
                    assert np.all(observed[seed_ids]==1) and np.all(observed[partner_ids]==1) and np.all((fractions>=0)&(fractions<1))
                    synthetic=(X[seed_ids]+fractions[:,None]*(X[partner_ids]-X[seed_ids]) if method=='SMOTE' else
                               (1-fractions[:,None])*X[seed_ids]+fractions[:,None]*X[partner_ids])
                    assert array_hash(np.vstack((X[retained],synthetic)))==row['output_X_sha256']
                    assert array_hash(np.r_[observed[retained],np.ones(generated,dtype=observed.dtype)])==row['output_y_sha256']
                    a,b=flags[seed_ids],flags[partner_ids]
                    assert diag['clean_clean']==int(np.sum(~a&~b)) and diag['clean_corrupt']==int(np.sum(a!=b)) and diag['corrupt_corrupt']==int(np.sum(a&b))
                    if method=='SMOTE':
                        positive=np.flatnonzero(observed==1)
                        distances=np.linalg.norm(X[positive,None,:]-X[None,positive,:],axis=2)
                        np.fill_diagonal(distances,np.inf)
                        nearest=np.argsort(distances,axis=1)[:,:5]
                        rng=np.random.RandomState(job['seed'])
                        choices=rng.randint(0,nearest.size,size=generated)
                        local_seed,local_neighbor=choices//5,choices%5
                        np.testing.assert_array_equal(seed_ids,positive[local_seed])
                        np.testing.assert_array_equal(partner_ids,positive[nearest[local_seed,local_neighbor]])
                        np.testing.assert_array_equal(fractions,rng.uniform(size=generated))
                        smote_checked+=len(seed_ids)
                expected_counts={'majority':int(np.sum(observed[retained]==0)),
                                 'minority':int(np.sum(observed[retained]==1))+row['generated_count']}
                assert row['output_counts']==expected_counts
                metrics_checked+=1; fprs.append(m['test_fpr'])
            scale=(record['scaler']['mean'],record['scaler']['scale'])
            assert counts.setdefault(base,scale)==scale
            if job['flipped_count']==0 and clean_reference is not None:
                old_job={key:job[key] for key in ('seed','rare_count','rare_sigma')}
                old_root=Path(clean_reference)
                old_envelope=json.loads((old_root/f'job-{content_hash(old_job)}.json').read_text())
                old_record=old_envelope['record']
                assert content_hash(old_record)==old_envelope['sha256']
                for partition in ('fitting','selection','calibration','test'):
                    for field in ('X','y','groups'):
                        assert record['partitions_sha256'][partition][field]==old_record['partitions_sha256'][partition][field]
                with np.load(old_root/old_record['predictions']['path'],allow_pickle=False) as old_vectors:
                    for method,row in rows.items():
                        np.testing.assert_array_equal(vectors[row['vectors']['test']],old_vectors[method])
                        clean_reference_vectors+=1
    assert len(records)==90 and metrics_checked==630 and candidates_checked==810
    screening=json.loads((RUN/'reports/screening.json').read_text())
    status=audit_screen(records,d,screening)
    score_rows=list(csv.DictReader((RUN/'reports/scores.csv').open()))
    assert len(score_rows)==630
    for entry in score_rows:
        key=(int(entry['fit_rare_count']),float(entry['rare_sigma']),int(entry['flipped_count']),int(entry['seed']))
        row=next(row for row in records[key]['rows'] if row['method']==entry['method'])
        assert entry['status']==row['status']=='success'
        for metric,value in row['metrics'].items():close(entry[metric],value,'score CSV '+metric)
        for group in ('common','rare','noise'):
            fraction=row['group_retention'][group]['fraction']
            if fraction is None:assert entry[group+'_retained_fraction']==''
            else:close(entry[group+'_retained_fraction'],fraction)
    paired_rows=list(csv.DictReader((RUN/'reports/paired_changes.csv').open()))
    assert len(paired_rows)==420
    for entry in paired_rows:
        r,s,m,seed=int(entry['fit_rare_count']),float(entry['rare_sigma']),int(entry['flipped_count']),int(entry['seed'])
        noisy=next(row for row in records[r,s,m,seed]['rows'] if row['method']==entry['method'])
        clean=next(row for row in records[r,s,0,seed]['rows'] if row['method']==entry['method'])
        assert entry['paired_complete']=='True'
        for metric in ('rare_recall','common_recall','positive_recall','test_fpr','f1','ap','worst_group_recall'):
            close(entry[metric+'_noisy_minus_clean'],noisy['metrics'][metric]-clean['metrics'][metric])
    coverage=list(csv.DictReader((RUN/'reports/coverage.csv').open()))
    assert len(coverage)==126
    for entry in coverage:
        assert int(entry['success'])==int(entry['shared_successes'])==int(entry['planned'])==5
        assert all(int(entry[key])==0 for key in ('missing','failed','unsupported'))
    summary=list(csv.DictReader((RUN/'reports/summary.csv').open()))
    for row in summary:
        source=[records[int(row['fit_rare_count']),float(row['rare_sigma']),int(row['flipped_count']),seed] for seed in range(42,47)]
        source=[next(r for r in record['rows'] if r['method']==row['method']) for record in source]
        if row['metric'].endswith('_retained_fraction'):
            group=row['metric'].split('_retained_fraction')[0]
            values=[r['group_retention'][group]['fraction'] for r in source]
            values=[v for v in values if v is not None]
        else: values=[r['metrics'][row['metric']] for r in source]
        assert int(row['n_values'])==len(values) and int(row['n_shared_seeds'])==5
        if values: close(row['mean'],statistics.mean(values));close(row['sample_sd'],statistics.stdev(values))
        else: assert row['mean']==row['sample_sd']==''
    result={'status':'passed','run_commit':manifest['git']['commit'],'run_id':envelope['run_id'],
            'design_sha256':manifest['design_sha256'],'case_records_checked':90,'prediction_metric_rows_checked':metrics_checked,
            'selection_AP_vectors_checked':candidates_checked,'calibration_vectors_and_cutoffs_checked':metrics_checked,
            'AW_candidate_traces_checked':candidates_checked,'AW_candidate_interpolations_checked':interpolations_checked,
            'SMOTE_interpolations_checked':smote_checked,'ROS_duplications_checked':ros_checked,
            'summary_rows_checked':len(summary),'screening_status':status,'clean_probability_vectors_equal_to_first_pilot':clean_reference_vectors,
            'max_absolute_metric_error':max_metric_error,'max_absolute_selection_AP_error':max_selection_error,
            'test_fpr_range':[min(fprs),max(fprs)],'test_fpr_above_point05':sum(v>.05 for v in fprs),
            'calibration_fpr_range':[.05,.05],'classifier_fits_in_original_run':1260,'additional_classifier_fits_for_audit':0,
            'checks':['Frozen source/config/run identity and case/archive/array checksums','Independently regenerated Gaussian partitions and nested label corruption',
                      'Original-fitting-only scaling and shared feature/true-group identities','All saved candidate AP values and selected fit reuse',
                      'All negative-only tie cutoffs and all seven held-out metrics','DBSCAN label replay and independent radius/neighbors/weights/quotas',
                      'AW and SMOTE interpolation/output reconstruction; ROS output reconstruction','Original-group retention and corrupted-parent counts',
                      'All summary means/SDs and all predeclared screen/control decisions'],
            'limitations':['Classifier fitting was not repeated; training isolation is covered by software tests and committed source. DBSCAN label verification replays the reference library, while radius/neighbor/weight arithmetic is independently reconstructed.']}
    if clean_reference is None:
        result['limitations'].append('The optional first-pilot probability comparison was not requested.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/rare-subgroups-contamination-v1')
    parser.add_argument('--clean-reference',type=Path,help='Optional raw first-pilot directory for exact zero-noise probability comparison.')
    parser.add_argument('--output',type=Path,help='Optional audit JSON path; default prints only.')
    args=parser.parse_args()
    result=audit(args.input,args.clean_reference)
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
