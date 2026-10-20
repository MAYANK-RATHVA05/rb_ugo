"""Frozen synthetic screening pipeline; all true subgroup metadata evaluator-only."""
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, t
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.gmmsampling_overlap import OverlapSampler, keyed_rng

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/gmmsampling_overlap_pilot.json"


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generate_partition(seed, stream, counts, delta, common):
    """Return X/y plus evaluator metadata kept outside model-fitting functions."""
    X, y, groups, ids = [], [], [], []
    for c, n in enumerate(counts):
        ns = [n] if c == 0 else [round(n * common), n - round(n * common)]
        for g, size in enumerate(ns):
            z = keyed_rng(seed, stream, c, g).normal(size=(size, 2))
            if c == 0:
                rows = z * [.5, .5] + [-8, 0]
            else:
                rows = z * ([.7, .8] if g == 0 else [.3, .35]) + [0 if g == 0 else 3, (-1 if c == 1 else 1)*delta/2]
            X.extend(rows)
            y.extend([c]*size)
            groups.extend([-1 if c == 0 else (c-1)*2+g]*size)
            ids.extend([f"{stream}:{c}:{g}:{i}" for i in range(size)])
    order = keyed_rng(seed, stream, 999).permutation(len(y))
    return {"X": np.asarray(X)[order], "y": np.asarray(y)[order], "group": np.asarray(groups)[order], "id": np.asarray(ids)[order]}


def metrics(y, probabilities, groups):
    pred = probabilities.argmax(axis=1)
    recalls = recall_score(y, pred, labels=[0,1,2], average=None, zero_division=0)
    precisions = precision_score(y, pred, labels=[0,1,2], average=None, zero_division=0)
    result = {"macro_ap": float(average_precision_score(np.eye(3)[y], probabilities, average="macro")),
              "macro_f1": float(f1_score(y, pred, labels=[0,1,2], average="macro", zero_division=0)),
              "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
              "minority_macro_ap": float(average_precision_score(np.eye(3)[y][:,1:], probabilities[:,1:], average="macro")),
              "pair_confusion": float(((y == 1) & (pred == 2) | (y == 2) & (pred == 1)).sum() / (y != 0).sum()),
              "minority_to_majority": float(((y != 0) & (pred == 0)).sum() / (y != 0).sum()),
              "majority_to_minority": float(((y == 0) & (pred != 0)).sum() / (y == 0).sum())}
    for c in range(3):
        result[f"recall_{c}"] = float(recalls[c])
        result[f"precision_{c}"] = float(precisions[c])
    for g in range(4):
        mask = groups == g
        result[f"subgroup_recall_{g}"] = float((pred[mask] == y[mask]).mean())
    result["worst_subgroup_recall"] = min(result[f"subgroup_recall_{g}"] for g in range(4))
    result["confusion_counts"] = confusion_matrix(y, pred, labels=[0,1,2]).tolist()
    result["confusion_normalized"] = confusion_matrix(y, pred, labels=[0,1,2], normalize="true").tolist()
    return result


def known_density_diagnostics(X, labels, delta, common):
    """Evaluator only: analytical minority posteriors, not certified labels."""
    if not len(X):
        return {"n":0}
    logs, responsibilities = [], []
    for c in (1,2):
        parts = np.column_stack([
            multivariate_normal.logpdf(X, [0, (-1 if c == 1 else 1)*delta/2], np.diag([.7**2,.8**2])) + np.log(common),
            multivariate_normal.logpdf(X, [3, (-1 if c == 1 else 1)*delta/2], np.diag([.3**2,.35**2])) + np.log(1-common)])
        logs.append(logsumexp(parts, axis=1))
        responsibilities.append(np.exp(parts[:,1] - logs[-1]))
    logs = np.column_stack(logs)
    own = (labels == 2).astype(int)
    posterior = np.exp(logs[np.arange(len(X)), own] - logsumexp(logs, axis=1))
    rare = np.column_stack(responsibilities)[np.arange(len(X)), own]
    return {"n":len(X), "true_rival_dominant_fraction":float((posterior < .5-1e-12).mean()),
            "true_ambiguous_fraction":float(((posterior >= .4) & (posterior <= .6)).mean()),
            "true_rare_soft_fraction":float(rare.mean())}


def candidate_specs(cfg):
    for method in cfg["methods"]:
        values = cfg["safe_k_grid"] if method == "A" else cfg["gate_tau_grid"] if method in ("B","C") else [None]
        for value, C in itertools.product(values, cfg["classifier_C"]):
            yield method, value, C


def run_case(cfg, seed, delta, common, output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = {p:generate_partition(seed, cfg["stream_ids"][p], cfg["partition_counts"][p], delta, common) for p in cfg["partition_counts"]}
    np.savez_compressed(output / "original_partitions.npz", **{f"{p}_{key}":v for p,d in data.items() for key,v in d.items()})
    scaler = StandardScaler().fit(data["fit"]["X"])
    transformed = {p:scaler.transform(d["X"]) for p,d in data.items()}
    t0 = time.perf_counter()
    sampler = OverlapSampler(transformed["fit"], data["fit"]["y"], cfg["minority_labels"], seed, cfg["gmm"])
    density_seconds = time.perf_counter()-t0
    density_state = {"scaler_mean":scaler.mean_, "scaler_scale":scaler.scale_}
    for c, model in sampler.models.items():
        for key in ("weights_", "means_", "covariances_", "precisions_cholesky_"):
            density_state[f"{c}_{key}"] = getattr(model,key)
        density_state[f"{c}_original_ids"] = np.flatnonzero(sampler.y == c)
        density_state[f"{c}_responsibilities"] = sampler.responsibilities[c]
    np.savez_compressed(output / "density_state.npz", **density_state)
    dump(output / "density_fit.json", {"seconds":density_seconds,"selection_traces":sampler.traces,
         "em_fits":sum(len(trace)+1 for trace in sampler.traces.values()),
         "em_initializations":cfg["gmm"]["n_init"]*sum(len(trace)+1 for trace in sampler.traces.values())})
    cache, candidates, fitted, selection = {}, [], {}, {}
    for order, (method, value, C) in enumerate(candidate_specs(cfg)):
        key = f"{method}_{value}"
        if key not in cache:
            start = time.perf_counter()
            if method in ("NORES", "WEIGHTED"):
                cache[key] = (transformed["fit"], data["fit"]["y"], None, time.perf_counter()-start)
            else:
                sample = sampler.sample(method, cfg["target"], k=int(value) if method == "A" else cfg["gate_safe_k"], tau=float(value) if method in ("B","C") else 0., retry_limit=cfg["retry_limit"])
                duration = time.perf_counter()-start
                cache[key] = (sample.X, sample.y, sample, duration)
                np.savez_compressed(output / f"samples_{key}.npz", X=sample.X, y=sample.y, parent=sample.parent, component=sample.component, kind=sample.kind, **sample.proposals)
                diagnostics = {}
                for kind in ("gaussian", "duplicate"):
                    mask = sample.kind == kind
                    diagnostics[kind] = known_density_diagnostics(sample.X[mask]*scaler.scale_+scaler.mean_, sample.y[mask], delta, common)
                dump(output / f"sampling_{key}.json", {"counts":sample.counts,"seconds":duration,"oracle_evaluator_only":diagnostics})
        Xfit, yfit, sample, sampling_time = cache[key]
        begin = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = LogisticRegression(C=C, max_iter=cfg["classifier_max_iter"], solver="lbfgs", random_state=seed,
                                       class_weight="balanced" if method == "WEIGHTED" else None).fit(Xfit, yfit)
        assert np.array_equal(model.classes_, [0,1,2])
        prob = model.predict_proba(transformed["validation"])
        score = metrics(data["validation"]["y"], prob, data["validation"]["group"])
        ident = f"candidate_{order:02d}"
        np.savez_compressed(output / f"{ident}_validation.npz", probabilities=prob, labels=data["validation"]["y"], sample_ids=data["validation"]["id"], evaluator_subgroup=data["validation"]["group"])
        record = {"id":ident,"method":method,"sampler_value":value,"classifier_C":C,"validation_macro_ap":score["macro_ap"],"validation_macro_f1":score["macro_f1"],"fit_seconds":time.perf_counter()-begin,"sampling_seconds":sampling_time,"fitting_rows":len(Xfit),"grid_order":order}
        candidates.append(record)
        rank = (score["macro_ap"], score["macro_f1"], -order)
        if method not in selection or rank > selection[method][0]:
            selection[method] = (rank, record, key)
            fitted[method] = model
    # Validation selection is complete for ALL methods before any test prediction.
    dump(output / "candidates.json", candidates)
    baseline = sampler.sample("A",cfg["target"],k=7)
    for method in ("B","C"):
        bypass = cache[f"{method}_0.0"][2]
        if not np.array_equal(baseline.X,bypass.X) or not np.array_equal(baseline.y,bypass.y):
            raise AssertionError("Tau-zero baseline identity failed")
    rows = []
    for method, (_, selected, key) in selection.items():
        model = fitted[method]
        probs = model.predict_proba(transformed["test"])
        np.savez_compressed(output / f"selected_{method}_test.npz", probabilities=probs, labels=data["test"]["y"], sample_ids=data["test"]["id"], evaluator_subgroup=data["test"]["group"], coefficients=model.coef_, intercept=model.intercept_)
        score = metrics(data["test"]["y"], probs, data["test"]["group"])
        Xfit, yfit, sample, _ = cache[key]
        count = {name:sum(r[name] for r in sample.counts) if sample else 0 for name in ("requested","generated","accepted","rejected","numerical_rejections","fallback","realized","unfilled")}
        rows.append({"seed":seed,"delta":delta,"common":common,"method":method,**selected,**score,**count,"density_seconds":density_seconds})
    dump(output / "selected_results.json", rows)
    dump(output / "completion.json", {"case_seconds":time.perf_counter()-started,"candidate_count":len(candidates),"tau_zero_identity":True})
    return rows


def advancement(frame, cfg, complete):
    criteria = cfg["criteria"]
    pivot = {m: frame[frame.method == m].set_index(["delta","common","seed"]).sort_index() for m in cfg["methods"]}
    checks = {"complete_accounting":complete}
    effects = {}
    for ref in ("A","B","D","NORES","WEIGHTED"):
        effects[ref] = pivot["C"].select_dtypes("number") - pivot[ref].select_dtypes("number")
    partial = lambda table: table.xs(1.5,level="delta")
    for mix in cfg["common_proportions"]:
        d = effects["A"].xs((1.5,mix),level=("delta","common"))
        checks[f"pair_gain_{mix}"] = bool(d.pair_confusion.mean() <= -criteria["partial_pair_reduction"]+1e-12 and (d.pair_confusion < 0).sum() >= criteria["partial_seed_wins"])
        checks[f"subgroup_safeguard_{mix}"] = bool(min(d[f"subgroup_recall_{g}"].mean() for g in range(4)) >= -criteria["subgroup_loss"]-1e-12)
        checks[f"other_safeguards_{mix}"] = bool(d.minority_to_majority.mean() <= criteria["minority_majority_increase"]+1e-12 and d.macro_f1.mean() >= -criteria["macro_f1_loss"]-1e-12)
        weak = effects["A"].xs((4.,mix),level=("delta","common"))
        checks[f"weak_control_{mix}"] = bool(weak.macro_ap.mean() >= -criteria["weak_ap_f1_loss"]-1e-12 and weak.macro_f1.mean() >= -criteria["weak_ap_f1_loss"]-1e-12)
        selected = pivot["C"].xs((1.5,mix),level=("delta","common"))
        checks[f"fresh_fraction_{mix}"] = bool(selected.accepted.sum()/selected.realized.sum() >= criteria["fresh_fraction"])
    checks["primary_ap_vs_A"] = bool(partial(effects["A"]).macro_ap.mean() >= criteria["macro_ap_gain"])
    checks["primary_ap_vs_D"] = bool(partial(effects["D"]).macro_ap.mean() >= criteria["macro_ap_gain"])
    b = effects["B"].xs((1.5,.9),level=("delta","common"))
    checks["coverage_vs_B"] = bool(b.worst_subgroup_recall.mean() >= criteria["C_vs_B_worst_recall_gain"] and b.pair_confusion.mean() <= criteria["C_vs_B_pair_increase"]+1e-12)
    for method in ("NORES","WEIGHTED"):
        covered = []
        for mix in cfg["common_proportions"]:
            d = effects[method].xs((1.5,mix),level=("delta","common"))
            covered.append(d.pair_confusion.mean() >= -criteria["simple_pair_tolerance"]-1e-12 and d.worst_subgroup_recall.mean() <= criteria["simple_subgroup_tolerance"]+1e-12 and d.macro_ap.mean() <= criteria["simple_ap_tolerance"]+1e-12)
        checks[f"distinct_from_{method}"] = not all(covered)
    return {"advancement_passed":all(checks.values()), "checks":checks}


def summarize(frame, output, cfg, complete):
    scalar = ["macro_ap","macro_f1","balanced_accuracy","pair_confusion","worst_subgroup_recall","minority_to_majority","majority_to_minority","recall_0","recall_1","recall_2",*[f"subgroup_recall_{g}" for g in range(4)],"requested","realized","accepted","rejected","fallback","fit_seconds"]
    frame.to_csv(output / "per_seed_results.csv", index=False)
    frame.groupby(["delta","common","method"])[scalar].agg(["mean","std"]).to_csv(output / "condition_summary.csv")
    differences = []
    for ref in ("A","B","D","NORES","WEIGHTED"):
        c = frame[frame.method == "C"].set_index(["delta","common","seed"])
        r = frame[frame.method == ref].set_index(["delta","common","seed"])
        diff = (c[scalar]-r[scalar]).reset_index()
        diff["comparison"] = f"C-minus-{ref}"
        differences.append(diff)
    paired = pd.concat(differences,ignore_index=True)
    paired.to_csv(output / "paired_per_seed.csv",index=False)
    summaries = []
    for keys, group in paired.groupby(["delta","common","comparison"]):
        for metric in scalar:
            mean, sd = group[metric].mean(), group[metric].std(ddof=1)
            half = t.ppf(.975,len(group)-1)*sd/np.sqrt(len(group))
            summaries.append(dict(delta=keys[0],common=keys[1],comparison=keys[2],metric=metric,n=len(group),mean=mean,sd=sd,descriptive_lower=mean-half,descriptive_upper=mean+half))
    pd.DataFrame(summaries).to_csv(output / "paired_summary.csv",index=False)
    decision = advancement(frame,cfg,complete)
    dump(output / "decision.json",decision)
    return decision


def run(output, config_path=CONFIG):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; previous results are never overwritten")
    output.mkdir(parents=True,exist_ok=True)
    cfg_bytes = Path(config_path).read_bytes()
    cfg = json.loads(cfg_bytes)
    (output / "frozen_configuration_bytes.json").write_bytes(cfg_bytes)
    source = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dump(output / "resolved_configuration.json",cfg)
    dump(output / "source_versions.json",{"source_commit":source,"configuration_sha256":hashlib.sha256(cfg_bytes).hexdigest(),"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"author_commit":cfg["author_commit"],"author_file_sha256":cfg["author_file_sha256"]})
    all_rows, failures = [], []
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for delta, common, seed in itertools.product(cfg["deltas"],cfg["common_proportions"],cfg["seeds"]):
            case = f"delta-{delta}_common-{common}_seed-{seed}"
            print(f"Running {case}",flush=True)
            try:
                rows = run_case(cfg,seed,delta,common,output / case)
                all_rows.extend(rows)
                print(f"Completed {case}: {len(rows)} selected methods",flush=True)
            except Exception as exc:
                failure = {"case":case,"error":repr(exc),"traceback":traceback.format_exc()}
                failures.append(failure)
                dump(output / case / "failure.json",failure)
                print(f"FAILED {case}: {exc}",flush=True)
    dump(output / "failures.json",failures)
    complete = not failures and len(all_rows) == 180
    if all_rows:
        decision = summarize(pd.DataFrame(all_rows),output,cfg,complete)
    else:
        decision = {"advancement_passed":False,"checks":{"complete_accounting":False}}
        dump(output / "decision.json",decision)
    dump(output / "run_completion.json",{"seconds":time.perf_counter()-started,"successful_cases":len(all_rows)//6,"failed_cases":len(failures),"selected_rows":len(all_rows),"advancement_passed":decision["advancement_passed"]})
    return decision
