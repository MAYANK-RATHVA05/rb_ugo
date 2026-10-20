"""Read-only evidence audit. Recompute metrics/selection/lineage without refitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from src.gmmsampling_overlap import density_accept
from src.gmmsampling_overlap_study import advancement, dump, metrics


def audit(output):
    output = Path(output)
    cfg = json.loads((output/"resolved_configuration.json").read_text())
    versions = json.loads((output/"source_versions.json").read_text())
    expected = json.dumps(cfg,indent=2)  # Hash uses original configuration, copied separately.
    del expected
    assert versions["configuration_sha256"] == hashlib.sha256((output/"frozen_configuration_bytes.json").read_bytes()).hexdigest()
    failures = json.loads((output/"failures.json").read_text())
    assert failures == [], failures
    results = []
    checked_candidates, checked_proposals = 0,0
    for case in sorted(output.glob("delta-*_common-*_seed-*")):
        raw = np.load(case/"original_partitions.npz")
        density = np.load(case/"density_state.npz")
        np.testing.assert_allclose(density["scaler_mean"],raw["fit_X"].mean(axis=0),atol=1e-12)
        fit = (raw["fit_X"]-density["scaler_mean"])/density["scaler_scale"]
        sets = [set(raw[f"{p}_id"]) for p in cfg["partition_counts"]]
        assert all(not sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2)))
        candidates = json.loads((case/"candidates.json").read_text())
        assert len(candidates)==36
        for candidate in candidates:
            saved = np.load(case/f"{candidate['id']}_validation.npz")
            probabilities = saved["probabilities"]
            assert probabilities.shape==(len(raw["validation_y"]),3)
            assert np.isfinite(probabilities).all() and (probabilities>=0).all()
            np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-12)
            np.testing.assert_array_equal(saved["sample_ids"],raw["validation_id"])
            np.testing.assert_array_equal(saved["labels"],raw["validation_y"])
            np.testing.assert_array_equal(saved["evaluator_subgroup"],raw["validation_group"])
            m = metrics(saved["labels"],probabilities,saved["evaluator_subgroup"])
            for key in ("macro_ap","macro_f1"):
                assert abs(m[key]-candidate[f"validation_{key}"]) < 1e-12
            checked_candidates += 1
        selected = json.loads((case/"selected_results.json").read_text())
        for row in selected:
            choices = [r for r in candidates if r["method"]==row["method"]]
            best = max(choices,key=lambda r:(r["validation_macro_ap"],r["validation_macro_f1"],-r["grid_order"]))
            assert row["id"]==best["id"]
            saved = np.load(case/f"selected_{row['method']}_test.npz")
            prob = saved["probabilities"]
            np.testing.assert_allclose(prob.sum(axis=1),1,atol=1e-12)
            assert np.isfinite(prob).all() and (prob>=0).all()
            np.testing.assert_array_equal(saved["labels"],raw["test_y"])
            np.testing.assert_array_equal(saved["sample_ids"],raw["test_id"])
            np.testing.assert_array_equal(saved["evaluator_subgroup"],raw["test_group"])
            for key,value in metrics(saved["labels"],prob,saved["evaluator_subgroup"]).items():
                np.testing.assert_allclose(value,row[key],atol=1e-12)
            # Reconstruct probabilities from saved selected fitting-only coefficients.
            Xtest = (raw["test_X"]-density["scaler_mean"])/density["scaler_scale"]
            logits = Xtest@saved["coefficients"].T+saved["intercept"]
            np.testing.assert_allclose(np.exp(logits-logsumexp(logits,axis=1)[:,None]),prob,atol=1e-12)
        for path in sorted(case.glob("samples_*.npz")):
            key = path.stem[len("samples_"):]
            method,value = key.split("_")
            sample = np.load(path)
            np.testing.assert_array_equal(sample["X"][:len(fit)],fit)
            np.testing.assert_array_equal(sample["y"][:len(fit)],raw["fit_y"])
            diagnostic = json.loads((case/f"sampling_{key}.json").read_text())
            for count in diagnostic["counts"]:
                c,j,q = count["label"],count["component"],count["requested"]
                added = (np.arange(len(sample["y"]))>=len(fit))&(sample["y"]==c)&(sample["component"]==j)
                assert added.sum()==count["realized"]
                duplicate = added&(sample["kind"]=="duplicate")
                parent = sample["parent"][duplicate]
                assert np.all(parent>=0) and np.all(parent<len(fit))
                np.testing.assert_array_equal(sample["X"][duplicate],fit[parent])
                np.testing.assert_array_equal(sample["y"][duplicate],raw["fit_y"][parent])
                class_ids = density[f"{c}_original_ids"]
                assigned = density[f"{c}_responsibilities"].argmax(axis=1)
                assert set(parent).issubset(set(class_ids[assigned==j]))
                proposals = sample[f"{c}_{j}_X"]
                accept = sample[f"{c}_{j}_accepted"]
                assert len(proposals)==count["generated"]==count["accepted"]+count["rejected"]
                np.testing.assert_array_equal(sample["X"][added&(sample["kind"]=="gaussian")],proposals[accept])
                if method in ("B","C") and float(value)>0 and len(proposals):
                    logs=[]
                    for d in cfg["minority_labels"]:
                        components = [multivariate_normal.logpdf(proposals,mean,cov)+np.log(w) for w,mean,cov in zip(density[f"{d}_weights_"],density[f"{d}_means_"],density[f"{d}_covariances_"])]
                        logs.append(logsumexp(np.column_stack(components),axis=1))
                    expected,invalid = density_accept(np.column_stack(logs),cfg["minority_labels"].index(c),float(value))
                    np.testing.assert_array_equal(accept,expected)
                    np.testing.assert_array_equal(sample[f"{c}_{j}_numerical_failure"],invalid)
                if method=="B":
                    assert count["generated"]==q and count["fallback"]==0
                elif method=="C":
                    assert count["generated"]<=cfg["retry_limit"]*q
                    assert count["accepted"]+count["fallback"]==q
                elif method=="D":
                    assert count["fallback"]==q and count["generated"]==0
                if method!="B":
                    assert count["realized"]==q
                checked_proposals += len(proposals)
        a = np.load(case/"samples_A_7.npz")
        for method in ("B","C"):
            zero = np.load(case/f"samples_{method}_0.0.npz")
            np.testing.assert_array_equal(a["X"],zero["X"])
        results.extend(selected)
    assert len(results)==180 and checked_candidates==1080
    frame = pd.DataFrame(results)
    decision = advancement(frame,cfg,True)
    assert decision==json.loads((output/"decision.json").read_text())
    result={"cases":30,"selected_prediction_files":180,"candidate_validation_files":checked_candidates,"checked_proposals":checked_proposals,"all_checks_passed":True,"refits":0,"advancement_passed":decision["advancement_passed"]}
    dump(output/"evidence_audit.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/gmmsampling-overlap-pilot"))
    print(audit(parser.parse_args().output))
