"""Opt-in supervised diagnostic harness; no new research sampler is registered.

Ground-truth modes enter evaluator functions only. Original UGO is untouched.
The two ablations are explicitly artificial and preserve its generation helper.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Any
from unittest.mock import patch
import warnings

import numpy as np
from imblearn.over_sampling import KMeansSMOTE, RandomOverSampler, SMOTE
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score

from src.classifier_registry import make_classifier
from src.ec_ebo import ECEBO, entropy_balance_score
from src.evaluation import _software_identity
from src.metrics import binary_metrics
from src.preprocessing import FoldPreprocessor
from src.results import ResultStore, _atomic_json, content_hash, json_value
from src.ugo import UGO
from src.uncertainty import modified_margin_uncertainty


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/manifests/question_b_diagnostic_protocol.json"
BASIC_ARMS = ("nores", "weighted", "ros", "smote", "kmeans_smote", "ec_ebo")


@dataclass(frozen=True)
class Partition:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray  # Evaluator metadata, never a resampler/model argument.


def load_protocol(path: Path = PROTOCOL) -> dict:
    p = json.loads(path.read_text())
    if p.get("schema_version") != 1 or not p.get("seeds"):
        raise ValueError("A schema-1 protocol with seeds is required.")
    if len(set(p["seeds"])) != len(p["seeds"]) or any(
            isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32
            for s in p["seeds"]):
        raise ValueError("Seeds must be unique uint32 integers.")
    if any(n < 200 or n % 200 for n in p["sizes"].values()):
        raise ValueError("Sizes must support exact 70/25/5 minority strata.")
    if p["target_positive_prior"] != .1 or p["minority_weights"] != [.7, .25, .05]:
        raise ValueError("This diagnostic fixes the target law to the declared prior/weights.")
    if any(not 0 <= v <= 1 for v in p["regimes"].values()):
        raise ValueError("Overlap fractions must lie in [0,1].")
    if not 0 < p["threshold_fpr"] < 1:
        raise ValueError("FPR target must lie in (0,1).")
    return p


def make_partition(p: dict, seed: int, split: str, regime: str) -> Partition:
    """Exact label/mode strata, independently drawn features and row permutation."""
    purpose = {"fit": 1, "validation": 2, "test": 3}[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, purpose]))
    n = p["sizes"][split]
    n1 = n // 10
    counts = np.rint(n1 * np.array(p["minority_weights"])).astype(int)
    centers = np.asarray(p["centers"])

    def disks(g):
        angle = rng.uniform(0, 2*np.pi, len(g))
        radius = p["disk_radius"] * np.sqrt(rng.random(len(g)))
        return centers[g] + radius[:, None] * np.c_[np.cos(angle), np.sin(angle)]

    groups = np.repeat(np.arange(3), counts)
    positive = disks(groups)
    negative_groups = rng.integers(0, 3, n-n1)
    lo, hi = p["annulus_radii"]
    radius = np.sqrt(lo**2 + rng.random(n-n1)*(hi**2-lo**2))
    angle = rng.uniform(0, 2*np.pi, n-n1)
    negative = centers[negative_groups] + radius[:, None]*np.c_[np.cos(angle), np.sin(angle)]
    overlap = rng.random(n-n1) < p["regimes"][regime]
    negative[overlap] = disks(rng.choice(3, int(overlap.sum()), p=p["minority_weights"]))
    X = np.vstack([negative, positive])
    y = np.r_[np.zeros(n-n1, dtype=int), np.ones(n1, dtype=int)]
    g = np.r_[np.full(n-n1, -1), groups]
    order = rng.permutation(n)
    return Partition(X[order], y[order], g[order])


def array_hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def make_model(name: str, seed: int, *, weighted=False, capacity=None):
    model = make_classifier("RF" if name == "RF100" else name, seed=seed)
    if name == "RF100":
        model.set_params(n_estimators=100, n_jobs=1, min_samples_leaf=capacity or 1)
    elif capacity is not None:
        model.set_params(C=capacity)
    if weighted:
        model.set_params(class_weight="balanced")
    return model


class _TracedUGO(UGO):
    """Harness-only tracing/ablation. The baseline implementation is inherited."""

    def _generate(self, X, y, seeds, selected, pre_noise, batch_size, rng):
        scores = modified_margin_uncertainty(
            self.estimator_.predict_proba(X.copy()), y, self.estimator_.classes_)
        all_class = np.flatnonzero(y == selected)
        ranked = all_class[np.argsort(-scores[all_class], kind="stable")[:len(seeds)]]
        actual = ranked if getattr(self, "_unfilter_generation", False) else seeds
        mask = np.zeros(len(y), dtype=bool) if getattr(self, "_unfilter_generation", False) else pre_noise
        result, info = super()._generate(X, y, actual, selected, mask, batch_size, rng)
        info["counterfactual_unfiltered_same_size_seed_pool"] = ranked.tolist()
        info["actual_seed_pool"] = actual.tolist()
        return result, info


def ugo_arms(p: dict) -> tuple[str, ...]:
    return ("ugo_natural", *(f"ugo_k{k}" for k in p["neighborhood_grid_in_tie_order"]),
            "ugo_prefilter_disabled", "ugo_generation_filter_disabled")


def all_arms(p: dict) -> tuple[str, ...]:
    return (*BASIC_ARMS, *ugo_arms(p), *(f"ros_match_{a}" for a in ugo_arms(p)),
            "weights_match_generation")


def _capture_smote_pools(sampler, X):
    lookup = {tuple(row): i for i, row in enumerate(X)}
    if len(lookup) != len(X):
        raise ValueError("Exact-coordinate duplicates need a separate group-aware design.")
    eligible = set()
    original = sampler._make_samples

    def capture(self, pool, *args, **kwargs):
        eligible.update(lookup[tuple(row)] for row in pool)
        return original(pool, *args, **kwargs)

    sampler._make_samples = MethodType(capture, sampler)
    return eligible


def _ec_eligible(sampler, X):
    eligible = set()
    a = sampler.allocation_
    c = a.clustering_
    if sampler.step_ <= 0:
        return eligible
    for members, center, entropy, quota in zip(c.clusters_, c.centroids_,
                                               a.cluster_entropies_, a.allocated_counts_):
        if not quota:
            continue
        majority = c.majority_indices_
        near = np.argsort(np.linalg.norm(X[majority]-center, axis=1), kind="stable")[:sampler.k_majority]
        for ref in majority[near]:
            for i in members:
                if entropy_balance_score(X[i], center, X[ref], entropy,
                                         a.sample_entropies_[ref], sampler.alpha) > 0:
                    eligible.add(int(i))
    return eligible


def resample_arm(X, y, model, arm, p, seed):
    """Fitting-only API: no groups, validation or test arguments exist."""
    if arm not in (*BASIC_ARMS, *ugo_arms(p)):
        raise ValueError(f"Unknown diagnostic arm {arm!r}.")
    sampler = None
    eligible = None
    if arm in {"nores", "weighted"}:
        return X.copy(), y.copy(), {"eligibility_stage": "not_applicable"}, None
    if arm.startswith("ugo_"):
        params = deepcopy(p["ugo"])
        if arm == "ugo_natural":
            params["phi"] = p["natural_phi"]
        if arm.startswith("ugo_k"):
            params["k_pre"] = int(arm[5:])
        sampler = _TracedUGO(model, variant="smote", random_state=seed,
                             allow_provisional=True, **params)
        sampler._unfilter_generation = arm == "ugo_generation_filter_disabled"
        if arm == "ugo_prefilter_disabled":
            # Sequential scoped ablation; the source module is restored on failure.
            with patch("src.ugo.noise_pre_detect", lambda X, y, k: np.zeros(len(y), dtype=bool)):
                augmented, labels = sampler.fit_resample(X, y)
        else:
            augmented, labels = sampler.fit_resample(X, y)
        eligible = np.flatnonzero(~sampler.pre_noise_mask_ & (y == 1))
        generation_eligible = np.flatnonzero(y == 1) if sampler._unfilter_generation else eligible
        meta = {"eligibility_stage": "UGO original minority generation pool",
                "eligible_indices": generation_eligible, "scoring_eligible_indices": eligible,
                "pre_noise_mask": sampler.pre_noise_mask_, "diagnostics": sampler.diagnostics_}
        return augmented, labels, meta, sampler
    if arm == "ros":
        sampler = RandomOverSampler(random_state=seed)
    elif arm == "smote":
        sampler = SMOTE(random_state=seed, k_neighbors=p["ugo"]["k_smote"])
    elif arm == "kmeans_smote":
        kw = p["kmeans_smote"]
        sampler = KMeansSMOTE(random_state=seed, k_neighbors=kw["k_neighbors"],
            kmeans_estimator=KMeans(n_clusters=kw["n_clusters"], n_init=kw["n_init"], random_state=seed),
            cluster_balance_threshold=kw["cluster_balance_threshold"], density_exponent=kw["density_exponent"])
    elif arm == "ec_ebo":
        sampler = ECEBO(random_state=seed, minority_label=1)
    else:
        raise ValueError(f"Unknown diagnostic arm {arm!r}.")
    if arm in {"smote", "kmeans_smote"}:
        eligible = _capture_smote_pools(sampler, X)
    augmented, labels = sampler.fit_resample(X, y)
    if arm == "ros":
        eligible = np.flatnonzero(y == 1)
    if arm == "ec_ebo":
        eligible = _ec_eligible(sampler, X)
    meta = {"eligible_indices": sorted(eligible),
            "eligibility_stage": "actual synthesis pools" if arm != "ec_ebo" else "union positive-psi pools with quota and positive step",
            "diagnostics": getattr(sampler, "diagnostics_", None)}
    return augmented, labels, meta, sampler


def _by_group(indices, groups):
    return [int(np.sum(groups[np.asarray(indices, dtype=int)] == g)) for g in range(3)]


def trace_groups(meta, groups):
    """Evaluator join after resampling. Group identities never enter a method."""
    out = {"eligible_by_group": (_by_group(meta["eligible_indices"], groups)
                                 if "eligible_indices" in meta else None),
           "eligibility_stage": meta["eligibility_stage"]}
    if "pre_noise_mask" not in meta:
        return out
    noise = np.asarray(meta["pre_noise_mask"])
    out.update(scoring_eligible_by_group=_by_group(meta["scoring_eligible_indices"], groups),
               pre_excluded_by_group=[int(np.sum((groups == g) & noise)) for g in range(3)],
               generation_rounds=0, minority_generation_rounds=0,
               rare_filter_opportunity_rounds=0, ranked_rare_filter_opportunity_rounds=0,
               drawn_seed_counts_by_group=[0, 0, 0], ranked_seed_events_by_group=[0, 0, 0],
               relocated_count=meta["diagnostics"]["generated_noisy_samples_relocated"],
               stopping_reason=meta["diagnostics"]["stopping_reason"])
    for h in meta["diagnostics"]["history"]:
        if h["generated_count"] <= 0:
            continue
        out["generation_rounds"] += 1
        if h["selected_class"] != 1:
            continue
        out["minority_generation_rounds"] += 1
        g = h["generation"]
        cf = np.asarray(g["counterfactual_unfiltered_same_size_seed_pool"], dtype=int)
        excluded = (groups == 2) & noise
        out["rare_filter_opportunity_rounds"] += int(excluded.any())
        out["ranked_rare_filter_opportunity_rounds"] += int(excluded[cf].any())
        for key, source in (("drawn_seed_counts_by_group", "drawn_seed_indices"),
                            ("ranked_seed_events_by_group", "actual_seed_pool")):
            out[key] = (np.array(out[key]) + _by_group(g[source], groups)).tolist()
    return out


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_arrays(store, job, X, y, weights=None):
    name = f"arrays-{content_hash(job)}.npz"
    path = store.directory / name
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(temporary, X=X, y=y,
                        weights=np.array([]) if weights is None else weights)
    temporary.replace(path)
    return {"file": name, "sha256": _file_hash(path)}


def _read_arrays(store, record):
    path = store.directory / record["arrays"]["file"]
    if _file_hash(path) != record["arrays"]["sha256"]:
        raise ValueError("Resampled-array checksum mismatch.")
    with np.load(path, allow_pickle=False) as d:
        return d["X"], d["y"], d["weights"] if d["weights"].size else None


def counts(y):
    return [int(np.sum(y == c)) for c in (0, 1)]


def _describe_arrays(pre, X, y, original_X, original_y, p):
    n = len(original_y)
    if not np.array_equal(X[:n], original_X) or not np.array_equal(y[:n], original_y):
        raise ValueError("An arm removed or changed original classifier training rows.")
    generated = pre.scaler_.inverse_transform(X[n:]) if len(X) > n else np.empty((0, 2))
    minority = generated[y[n:] == 1]
    distances = np.linalg.norm(minority[:, None, :] - np.asarray(p["centers"])[None, :, :], axis=2)
    inside = distances <= p["disk_radius"] + 1e-12
    return {"original_rows_retained": n, "original_rows_removed": 0,
            "final_class_counts": counts(y), "generated_by_class": counts(y[n:]),
            "minority_generated_inside_support_by_group": inside.sum(axis=0).tolist(),
            "minority_generated_outside_support": int(np.sum(~inside.any(axis=1)))}


def threshold_at_fpr(y, scores, alpha=.05):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if (y.shape != scores.shape or not np.isfinite(scores).all()
            or not 0 < alpha < 1 or not np.any(y == 0) or not np.any(y == 1)):
        raise ValueError("Finite aligned validation scores, two classes and valid FPR required.")
    negative = np.sort(scores[y == 0])
    threshold = float(negative[len(negative)-int(np.floor(alpha*len(negative)))-1])
    prediction = scores > threshold
    fpr = float(prediction[y == 0].mean())
    tpr = float(prediction[y == 1].mean())
    return {"threshold": threshold, "fpr": fpr, "tpr": tpr,
            "balanced_accuracy": .5*(tpr+1-fpr)}


def select_validation(records, candidates):
    valid = [c for c in candidates if c in records and records[c].get("status") == "success"]
    if not valid:
        return None
    return max(valid, key=lambda c: records[c]["validation"]["balanced_accuracy"])


def test_metrics(y, scores, groups, threshold):
    prediction = scores > threshold
    result = binary_metrics(y, prediction, negative_label=0, positive_label=1,
                            positive_probabilities=scores)
    recalls, denominators = [], []
    for g in range(3):
        mask = (y == 1) & (groups == g)
        denominators.append(int(mask.sum()))
        recalls.append(float(prediction[mask].mean()) if mask.any() else None)
    c = result["counts"]
    fpr = c["fp"] / (c["fp"]+c["tn"])
    tpr = c["tp"] / (c["tp"]+c["fn"])
    result.update(average_precision=float(average_precision_score(y, scores)),
                  fpr=fpr, balanced_accuracy=.5*(tpr+1-fpr),
                  subgroup_recall=recalls, subgroup_denominators=denominators,
                  macro_recall=float(np.mean(recalls)) if None not in recalls else None,
                  worst_recall=min(recalls) if None not in recalls else None,
                  population_error=float(np.mean(prediction != y)))
    return result


def _scores(model, X):
    p = model.predict_proba(X)
    if (not np.array_equal(model.classes_, [0, 1]) or p.shape != (len(X), 2)
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("Invalid classifier probability contract.")
    return p[:, 1]


def open_store(output, p):
    fits = {}
    for regime in p["regimes"]:
        for seed in p["seeds"]:
            fitting = make_partition(p, seed, "fit", regime)
            fits[f"{regime}/{seed}"] = array_hash(fitting.X, fitting.y, fitting.groups)
    manifest = {"protocol": p, "protocol_sha256": content_hash(p), "software": _software_identity(),
                "runner_sha256": _file_hash(ROOT / "scripts/run_subgroup_diagnostic.py"),
                "fitting_hashes": fits}
    return ResultStore(output, manifest)


def _job(role, regime, seed, classifier, arm):
    return dict(role=role, regime=regime, seed=seed, classifier=classifier, arm=arm)


def _prepared(partition):
    pre = FoldPreprocessor()
    X = pre.fit_transform(partition.X, partition.y)
    if X.shape[1] != 2 or not pre.keep_mask_.all():
        raise ValueError("Diagnostic requires its original two numerical coordinates.")
    return pre, X


def run_mechanism(store, p, progress=print):
    all_records = []
    for regime in p["regimes"]:
        for seed in p["seeds"]:
            F = make_partition(p, seed, "fit", regime)  # No V or T is constructed.
            pre, X = _prepared(F)
            for classifier in p["classifiers"]:
                cell = {}
                for arm in all_arms(p):
                    job = _job("mechanism", regime, seed, classifier, arm)
                    existing = store.read(job)
                    if existing:
                        if existing["status"] == "success":
                            _read_arrays(store, existing)
                        cell[arm] = existing
                        all_records.append(existing)
                        continue
                    start = perf_counter()
                    record = {"job": job, "status": "failed", "error": None}
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            weights = None
                            if arm.startswith("ros_match_") or arm == "weights_match_generation":
                                target = ("ugo_generation_filter_disabled" if arm == "weights_match_generation"
                                          else arm[len("ros_match_"):])
                                if cell[target]["status"] != "success":
                                    raise ValueError("Matched UGO target did not complete.")
                                target_counts = cell[target]["training"]["final_class_counts"]
                                ros = RandomOverSampler(sampling_strategy=dict(enumerate(target_counts)), random_state=seed)
                                augmented, labels = ros.fit_resample(X, F.y)
                                meta = {"eligible_indices": np.flatnonzero(F.y == 1),
                                        "eligibility_stage": "ROS original minority rows"}
                                if arm == "weights_match_generation":
                                    weights = np.bincount(ros.sample_indices_, minlength=len(F.y)).astype(float)
                                    augmented, labels = X.copy(), F.y.copy()
                                    meta = {"eligibility_stage": "not_applicable"}
                            else:
                                augmented, labels, meta, _ = resample_arm(
                                    X, F.y, make_model(classifier, seed), arm, p, seed)
                            training = _describe_arrays(pre, augmented, labels, X, F.y, p)
                            training.update(trace_groups(meta, F.groups))
                            if weights is not None:
                                training["total_sample_weight"] = float(weights.sum())
                                training["weighted_class_counts"] = [float(weights[F.y == c].sum()) for c in (0, 1)]
                            record.update(status="success", training=training, meta=json_value(meta),
                                          arrays=_save_arrays(store, job, augmented, labels, weights))
                        except Exception as exc:
                            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                        record["warnings"] = [str(w.message) for w in caught]
                    record["resampling_seconds"] = perf_counter()-start
                    cell[arm] = store.write(record)
                    all_records.append(cell[arm])
                progress(f"mechanism {regime} seed={seed} {classifier}: "
                         f"{Counter(r['status'] for r in cell.values())}")
    envelope = {"run_id": store.run_id, "jobs": len(all_records),
                "status_counts": dict(Counter(r["status"] for r in all_records)),
                "original_ugo": [{"job": r["job"], "status": r["status"],
                                  "training": r.get("training"), "error": r.get("error")}
                                 for r in all_records if r["job"]["arm"] in {"ugo_natural", "ugo_k5"}]}
    _atomic_json(store.directory / "mechanism.json", envelope)
    return all_records


def run_pilot(store, p, progress=print):
    gate = store.directory / "mechanism.json"
    if not gate.exists() or json.loads(gate.read_text()).get("run_id") != store.run_id:
        raise ValueError("Complete the matching fitting-only mechanism phase first.")
    # Check the entire gate before any validation or test feature construction.
    for regime in p["regimes"]:
        for seed in p["seeds"]:
            for c in p["classifiers"]:
                for arm in all_arms(p):
                    if store.read(_job("mechanism", regime, seed, c, arm)) is None:
                        raise ValueError("Incomplete mechanism phase.")
    records = []
    for regime in p["regimes"]:
        for seed in p["seeds"]:
            F = make_partition(p, seed, "fit", regime)
            V = make_partition(p, seed, "validation", regime)
            pre, X = _prepared(F)
            val = pre.transform(V.X)
            for classifier in p["classifiers"]:
                cell, pending = {}, {}
                configurations = [(a, a, None) for a in all_arms(p)]
                configurations += [(f"nores_capacity_{i}", "nores", value)
                                   for i, value in enumerate(p["nores_capacity_grid"][classifier]) if i]
                for arm, training_arm, capacity in configurations:
                    job = _job("pilot", regime, seed, classifier, arm)
                    saved = store.read(job)
                    if saved:
                        cell[arm] = saved
                        continue
                    record = {"job": job, "status": "failed", "error": None}
                    mechanism = store.read(_job("mechanism", regime, seed, classifier, training_arm))
                    if mechanism["status"] != "success":
                        record["error"] = {"stage": "resampling", **mechanism["error"]}
                        cell[arm] = store.write(record)
                        continue
                    start = perf_counter()
                    try:
                        augmented, labels, weights = _read_arrays(store, mechanism)
                        _describe_arrays(pre, augmented, labels, X, F.y, p)
                        model = make_model(classifier, seed, weighted=arm == "weighted", capacity=capacity)
                        kwargs = {} if weights is None else {"sample_weight": weights}
                        model.fit(augmented, labels, **kwargs)
                        validation = threshold_at_fpr(V.y, _scores(model, val), p["threshold_fpr"])
                        record.update(status="success", validation=validation,
                                      training=mechanism["training"], model_parameters=model.get_params(),
                                      resampling_seconds=mechanism["resampling_seconds"])
                        pending[arm] = model
                    except Exception as exc:
                        record["error"] = {"stage": "fit_or_validation", "type": type(exc).__name__, "message": str(exc)}
                    record["fit_validation_seconds"] = perf_counter()-start
                    cell[arm] = record
                # The selector gets validation fields only, before T exists.
                selector_input = {a: {"status": r["status"], "validation": r.get("validation")}
                                  for a, r in cell.items()}
                selected = {
                    "ugo_tuned_k": select_validation(selector_input, [f"ugo_k{k}" for k in p["neighborhood_grid_in_tie_order"]]),
                    "nores_tuned_capacity": select_validation(selector_input, ["nores", "nores_capacity_1", "nores_capacity_2", "nores_capacity_3"])}
                T = make_partition(p, seed, "test", regime)
                testing = pre.transform(T.X)
                for arm, r in cell.items():
                    if arm in pending:
                        try:
                            scores = _scores(pending[arm], testing)
                            r["metrics"] = test_metrics(T.y, scores, T.groups, r["validation"]["threshold"])
                            r["default_threshold_metrics"] = test_metrics(T.y, scores, T.groups, .5)
                            r["test_data_hash"] = array_hash(T.X, T.y, T.groups)
                        except Exception as exc:
                            r.update(status="failed", error={"stage": "test_prediction", "type": type(exc).__name__, "message": str(exc)})
                    if "run_id" not in r:
                        cell[arm] = store.write(r)
                for name, chosen in selected.items():
                    job = _job("pilot", regime, seed, classifier, name)
                    derived = ({"job": job, "status": "failed", "error": {"stage": "selection", "message": "No successful candidate"}}
                               if chosen is None else {**cell[chosen], "job": job, "selected_from": chosen})
                    saved = store.read(job)
                    if saved and saved.get("selected_from") != chosen:
                        raise ValueError("Saved validation choice does not reproduce.")
                    cell[name] = saved or store.write(derived)
                # Exact-count comparator corresponding to the selected k, no additional fit.
                name = "ros_match_ugo_tuned_k"
                chosen = selected["ugo_tuned_k"]
                source = f"ros_match_{chosen}" if chosen else None
                job = _job("pilot", regime, seed, classifier, name)
                r = ({**cell[source], "job": job, "selected_from": source} if source else
                     {"job": job, "status": "failed", "error": {"message": "No successful tuned target"}})
                cell[name] = store.read(job) or store.write(r)
                records.extend(cell.values())
                progress(f"pilot {regime} seed={seed} {classifier}: "
                         f"k={selected['ugo_tuned_k']}, {Counter(r['status'] for r in cell.values())}")
    return records


def _flat_metrics(r):
    m = r["metrics"]
    return {**{k: m[k] for k in ("f1", "average_precision", "balanced_accuracy", "fpr", "macro_recall", "worst_recall", "population_error")},
            **{f"recall_g{i}": v for i, v in enumerate(m["subgroup_recall"])}}


def paired_interval(differences, p):
    d = np.asarray(differences, dtype=float)
    if not len(d):
        return {"n": 0, "mean_difference": None, "low": None, "high": None}
    rng = np.random.default_rng(p["bootstrap"]["seed"])
    sample = d[rng.integers(0, len(d), (p["bootstrap"]["resamples"], len(d)))].mean(axis=1)
    lo, hi = np.quantile(sample, [.025, .975])
    return {"n": len(d), "mean_difference": float(d.mean()), "low": float(lo), "high": float(hi)}


def write_summary(store, records, p):
    rows = []
    for r in records:
        row = {**r["job"], "status": r["status"], "selected_from": r.get("selected_from")}
        if r["status"] == "success":
            row.update(_flat_metrics(r))
            t = r["training"]
            row.update(generated_majority=t["generated_by_class"][0], generated_minority=t["generated_by_class"][1],
                       outside_support=t["minority_generated_outside_support"],
                       eligible_rare=None if t["eligible_by_group"] is None else t["eligible_by_group"][2],
                       minority_rounds=t.get("minority_generation_rounds"),
                       ranked_rare_opportunity_rounds=t.get("ranked_rare_filter_opportunity_rounds"),
                       default_worst_recall=r["default_threshold_metrics"]["worst_recall"],
                       default_fpr=r["default_threshold_metrics"]["fpr"])
        rows.append(row)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (store.directory / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    pairs, aggregates = [], []
    arms = sorted({r["job"]["arm"] for r in records})
    for regime in p["regimes"]:
        for classifier in p["classifiers"]:
            cell = [r for r in records if r["job"]["regime"] == regime and r["job"]["classifier"] == classifier]
            index = {(r["job"]["arm"], r["job"]["seed"]): r for r in cell}
            for arm in arms:
                subset = [r for r in cell if r["job"]["arm"] == arm]
                successful = [r for r in subset if r["status"] == "success"]
                means = {k: float(np.mean([_flat_metrics(r)[k] for r in successful]))
                         for k in _flat_metrics(successful[0])} if successful else {}
                aggregates.append({"regime": regime, "classifier": classifier, "arm": arm,
                                   "status_counts": dict(Counter(r["status"] for r in subset)), "means": means})
                references = ["ugo_k5", "nores"]
                if arm in ugo_arms(p) or arm == "ugo_tuned_k":
                    references.append(f"ros_match_{arm}")
                if arm == "ugo_generation_filter_disabled":
                    references.append("weights_match_generation")
                for reference in references:
                    ids = [s for s in p["seeds"] if index.get((arm, s), {}).get("status") == "success"
                           and index.get((reference, s), {}).get("status") == "success"]
                    for metric in ("worst_recall", "recall_g2", "average_precision", "fpr", "population_error"):
                        delta = [_flat_metrics(index[arm, s])[metric] - _flat_metrics(index[reference, s])[metric] for s in ids]
                        pairs.append({"regime": regime, "classifier": classifier, "arm": arm,
                                      "reference": reference, "metric": metric, "paired_seeds": ids,
                                      **paired_interval(delta, p)})
    summary = {"run_id": store.run_id, "protocol_sha256": content_hash(p),
               "status_counts": dict(Counter(r["status"] for r in records)),
               "aggregates": aggregates, "paired": pairs,
               "interval_scope": "Exploratory, whole independent replicate resampling; secondary intervals unadjusted"}
    _atomic_json(store.directory / "summary.json", summary)
    return summary
