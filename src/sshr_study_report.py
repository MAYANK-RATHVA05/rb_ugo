"""Integrity-checked descriptive reports for saved controlled SSHR runs.

Repeated fold scores are dependent. Sample SD describes their spread; this
module computes neither p-values nor claims about superiority. Pseudo-label
and generation totals count fold events, not unique observations across CV.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from src.metrics import binary_metrics
from src.results import ResultStore, canonical_json, content_hash


STUDY_METRICS = ("f1", "gmean", "mcc", "probability_auc", "fpr")
SUMMARY_POLICY = (
    "Canonical comparison uses the intersection of successful finite-score "
    "repeat/fold events across every selected method separately per classifier. "
    "All-method-success summaries are supplementary. Repeated fold scores are "
    "dependent; sample SD is descriptive, not an inferential standard error. "
    "Counts describe fold events, not unique individuals; classifiers are not pooled."
)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value.")
    return float(value)


def _same_number(actual: Any, expected: float | None, name: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{name} disagrees with saved observations.")
    elif not np.isclose(_number(actual, name), expected, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} disagrees with saved observations.")


def _indices(value: Any, name: str, rows: int, empty: bool = False) -> list[int]:
    if (not isinstance(value, list) or not value and not empty
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= rows for i in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"Invalid {name} row indices.")
    return value


def _validate_manifest(manifest: dict[str, Any]) -> tuple[list[tuple[int, int]], dict, dict]:
    if manifest.get("stage") != "SSHR-3" or manifest.get("schema_version") != 1:
        raise ValueError("Expected a schema-1 SSHR-3 controlled-run manifest.")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("Missing controlled-run configuration.")
    methods, classifiers = config.get("methods"), config.get("classifiers")
    for name, values in (("methods", methods), ("classifiers", classifiers)):
        if (not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Invalid configured {name}.")
    if "nores" not in methods:
        raise ValueError("The controlled run must include nores.")
    splits, repeats = _integer(config.get("n_splits"), "n_splits"), _integer(config.get("n_repeats"), "n_repeats")
    if splits < 2 or not repeats:
        raise ValueError("At least two folds and one repeat are required.")
    rows = _integer(manifest.get("rows"), "rows")
    if not rows:
        raise ValueError("Input rows must be positive.")
    expected = [(repeat, fold) for repeat in range(repeats) for fold in range(splits)]
    maps = []
    for name, digest in (("folds", "folds_sha256"), ("masks", "masks_sha256")):
        values = manifest.get(name)
        if not isinstance(values, list) or content_hash(values) != manifest.get(digest):
            raise ValueError(f"Invalid {name} hash.")
        mapped = {}
        for item in values:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid {name} record.")
            key = (_integer(item.get("repeat"), "repeat"), _integer(item.get("fold"), "fold"))
            if key in mapped or key not in expected:
                raise ValueError(f"Duplicate or unexpected {name} identifier.")
            mapped[key] = item
        if set(mapped) != set(expected):
            raise ValueError(f"Missing configured {name} identifiers.")
        maps.append(mapped)
    folds, masks = maps
    for repeat in range(repeats):
        tested = []
        for fold in range(splits):
            item, mask = folds[(repeat, fold)], masks[(repeat, fold)]
            train = _indices(item.get("train_indices"), "training", rows)
            test = _indices(item.get("test_indices"), "test", rows)
            labeled = _indices(mask.get("labeled_indices"), "labeled", rows)
            hidden = _indices(mask.get("unlabeled_indices"), "unlabeled", rows, empty=True)
            if set(train) & set(test) or set(train) | set(test) != set(range(rows)):
                raise ValueError("Outer partitions must be disjoint exact complements.")
            if set(labeled) & set(hidden) or set(labeled) | set(hidden) != set(train):
                raise ValueError("Label masks must partition the outer training fold.")
            tested.extend(test)
        if sorted(tested) != list(range(rows)):
            raise ValueError("Every row must be tested exactly once per repeat.")
    return expected, folds, masks


def _validate_record_identity(record: dict, manifest: dict, fold: dict, mask: dict) -> None:
    for field in ("train_indices", "test_indices"):
        if record.get(field) != fold[field]:
            raise ValueError(f"Saved job {field} disagrees with its manifest.")
    for field in ("labeled_indices", "unlabeled_indices"):
        if record.get(field) != mask[field]:
            raise ValueError(f"Saved job {field} disagrees with its manifest.")
    expected_u = (mask["unlabeled_indices"] if record["job"]["method"] == "sshr"
                  and manifest["config"].get("use_unlabeled") else [])
    if record.get("algorithm_unlabeled_indices") != expected_u:
        raise ValueError("Algorithm unlabelled pool disagrees with the training-only manifest.")


def _validate_success(record: dict, manifest: dict, fold: dict, mask: dict) -> None:
    positive, labels = manifest.get("positive_label"), manifest.get("labels")
    if (not isinstance(labels, list) or len(labels) != 2 or labels[0] == labels[1]
            or positive not in labels or record.get("positive_label") != positive
            or record.get("probability_labels") != labels):
        raise ValueError("Saved probability labels or positive-class identity are invalid.")
    truth, prediction = record.get("true_labels"), record.get("predictions")
    if not isinstance(truth, list) or len(truth) != len(fold["test_indices"]):
        raise ValueError("Saved held-out truth is not aligned to the outer test fold.")
    probabilities = np.asarray(record.get("probabilities"))
    if (probabilities.shape != (len(truth), 2) or probabilities.dtype.kind not in "fiu"
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("Invalid saved held-out probabilities.")
    if set(truth) != set(labels):
        raise ValueError("Each controlled outer test fold must contain both classes.")
    negative = labels[1] if positive == labels[0] else labels[0]
    expected = binary_metrics(truth, prediction, negative_label=negative, positive_label=positive,
                              positive_probabilities=probabilities[:, labels.index(positive)])
    expected["probability_auc"] = expected["auc"]
    counts = expected["counts"]
    expected["fpr"] = counts["fp"] / (counts["tn"] + counts["fp"])
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Successful job is missing metrics.")
    for metric in STUDY_METRICS:
        _same_number(metrics.get(metric), expected[metric], f"saved {metric}")
    if metrics.get("counts") != expected["counts"]:
        raise ValueError("Saved confusion counts disagree with held-out observations.")
    _validate_counts(record)


def _validate_counts(record: dict) -> None:
    counts = record.get("sample_counts")
    if not isinstance(counts, dict):
        raise ValueError("Successful job is missing sample counts.")
    names = ("labeled_training", "resampled_training", "retained_labeled", "selected_pseudo_labeled",
             "generated_synthetic", "generated_minority", "generated_majority", "removed_majority", "removed_minority")
    n = {name: _integer(counts.get(name), name) for name in names}
    if n["labeled_training"] != len(record["labeled_indices"]):
        raise ValueError("Labeled sample count disagrees with the saved training mask.")
    if (n["retained_labeled"] + n["removed_majority"] + n["removed_minority"] != n["labeled_training"]
            or n["retained_labeled"] + n["selected_pseudo_labeled"] + n["generated_synthetic"] != n["resampled_training"]
            or n["generated_minority"] + n["generated_majority"] != n["generated_synthetic"]):
        raise ValueError("Saved sample counts do not conserve output rows.")
    proxy = record.get("training_overlap_proxy")
    if not isinstance(proxy, dict):
        raise ValueError("Missing training-only overlap proxy.")
    expected = ({"pseudo_labeled": n["selected_pseudo_labeled"], "smote": n["generated_synthetic"]}
                if record["job"]["method"] == "sshr" else
                {"synthetic_all": n["generated_synthetic"], "synthetic_minority": n["generated_minority"],
                 "synthetic_majority": n["generated_majority"]})
    for name, size in expected.items():
        item = proxy.get(name)
        if not isinstance(item, dict) or item.get("rows") != size:
            raise ValueError("Overlap proxy denominator disagrees with generated/selected rows.")
        majority = _integer(item.get("nearest_majority_count"), "nearest_majority_count")
        ties = _integer(item.get("distance_tie_count"), "distance_tie_count")
        if majority + ties > size:
            raise ValueError("Overlap proxy counts exceed their denominator.")
        _same_number(item.get("nearest_majority_rate"), majority / size if size else None, "overlap proxy rate")
    if record["job"]["method"] != "sshr":
        if n["selected_pseudo_labeled"] or n["removed_majority"] or n["removed_minority"]:
            raise ValueError("Supervised comparator has unsupported removals or pseudo-labels.")
        return
    if n["generated_majority"] or counts.get("generated_smote") != n["generated_synthetic"]:
        raise ValueError("SSHR synthetic rows must be minority SMOTE rows.")
    audit = record.get("pseudo_label_audit")
    if not isinstance(audit, dict):
        raise ValueError("SSHR is missing its retrospective pseudo-label audit.")
    selected = _integer(audit.get("selected_count"), "selected_count")
    correct = _integer(audit.get("correct_count"), "correct_count")
    incorrect = _integer(audit.get("incorrect_count"), "incorrect_count")
    if selected != n["selected_pseudo_labeled"] or selected != correct + incorrect:
        raise ValueError("Pseudo-label audit does not conserve selected rows.")
    source_rows = audit.get("selected_global_indices")
    if (not isinstance(source_rows, list) or len(source_rows) != selected
            or len(set(source_rows)) != selected
            or not set(source_rows) <= set(record["algorithm_unlabeled_indices"])):
        raise ValueError("Selected pseudo-label rows are not unique members of the training-only U pool.")
    _same_number(audit.get("accuracy"), correct / selected if selected else None, "pseudo-label accuracy")
    lineage = record.get("row_lineage")
    if not isinstance(lineage, list) or len(lineage) != n["resampled_training"]:
        raise ValueError("SSHR lineage does not conserve output rows.")
    kinds = Counter(item.get("kind") for item in lineage)
    if kinds != Counter({"labeled": n["retained_labeled"], "pseudo_labeled": selected,
                         "smote": n["generated_synthetic"]}):
        raise ValueError("SSHR lineage counts disagree with sample counts.")
    diagnostics = record.get("resampler_diagnostics")
    quotas = diagnostics.get("quotas") if isinstance(diagnostics, dict) else None
    if not isinstance(quotas, dict):
        raise ValueError("Successful SSHR record is missing quotas.")
    q = {name: _integer(quotas.get(name), name) for name in
         ("weighted_removal_requested", "weighted_removal_actual", "weighted_removal_unmet",
          "additions_requested", "pseudo_selected", "smote_generated", "additions_unmet")}
    if (q["weighted_removal_requested"] != q["weighted_removal_actual"] + q["weighted_removal_unmet"]
            or q["additions_requested"] != q["pseudo_selected"] + q["smote_generated"] + q["additions_unmet"]
            or q["pseudo_selected"] != selected or q["smote_generated"] != n["generated_synthetic"]):
        raise ValueError("SSHR quotas do not conserve requested/selected/generated rows.")


def _load(directory: str | Path) -> tuple[dict, str, dict, list]:
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError("Saved controlled-run manifest does not exist.")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    manifest = envelope.get("manifest")
    if not isinstance(manifest, dict) or envelope.get("run_id") != content_hash(manifest):
        raise ValueError("Invalid/corrupt controlled-run manifest hash.")
    keys, folds, masks = _validate_manifest(manifest)
    store = ResultStore(directory, manifest)
    config = manifest["config"]
    jobs = [{"repeat": repeat, "fold": fold, "classifier": classifier, "method": method}
            for repeat, fold in keys for classifier in config["classifiers"] for method in config["methods"]]
    filenames = {f"job-{content_hash(job)}.json" for job in jobs}
    if any(path.name not in filenames for path in directory.glob("job-*.json")):
        raise ValueError("Unexpected saved job outside the configured repeat/fold/classifier/method grid.")
    records = {}
    for job in jobs:
        record = store.read(job)
        key = (job["repeat"], job["fold"])
        if record is not None:
            _validate_record_identity(record, manifest, folds[key], masks[key])
            if record["status"] == "success":
                _validate_success(record, manifest, folds[key], masks[key])
        records[(job["classifier"], job["method"], *key)] = record
    return manifest, store.run_id, records, keys


def _statistics(values: list[float]) -> tuple[float | None, float | None]:
    return (float(np.mean(values)) if values else None,
            float(np.std(values, ddof=1)) if len(values) > 1 else None)


def _identity(manifest: dict, run_id: str) -> dict:
    return {"dataset_id": manifest["dataset_id"], "run_id": run_id,
            "source_sha256": manifest["provenance"].get("source_sha256"),
            "input_sha256": manifest["input_sha256"], "folds_sha256": manifest["folds_sha256"],
            "masks_sha256": manifest["masks_sha256"], "software": manifest.get("software"),
            "config": manifest["config"], "provenance": manifest["provenance"],
            "fold_protocol": manifest.get("fold_protocol"), "mask_protocol": manifest.get("mask_protocol"),
            "metric_policy": manifest.get("metric_policy"), "preprocessing_policy": manifest.get("preprocessing_policy")}


def _shortfalls(diagnostics: Any) -> dict[str, int]:
    if not isinstance(diagnostics, dict):
        return {}
    result = Counter()
    for item in diagnostics.get("shortfalls", []):
        if not isinstance(item, dict):
            raise ValueError("Invalid saved shortfall record.")
        result[f"{item.get('stage', 'unspecified')}:{item.get('cause', 'unspecified')}"] += _integer(item.get("count"), "shortfall count")
    for reasons in diagnostics.get("cluster_shortfall_reasons", []):
        for cause, count in reasons.items():
            result[f"ec_ebo:{cause}"] += _integer(count, "shortfall count")
    return dict(sorted(result.items()))


def _aggregate_diagnostics(manifest: dict, run_id: str, classifier: str, method: str,
                           records: list[dict]) -> dict:
    successful = [record for record in records if record["status"] == "success"]
    totals = Counter()
    proxy_totals: dict[str, Counter] = {}
    statuses, stopping, causes = Counter(), Counter(), Counter()
    duplicate_totals, duplicate_coverage = Counter(), Counter()
    failed_removals, failed_removal_coverage = Counter(), Counter()
    pseudo_accuracies = []
    for record in records:
        diagnostics = record.get("resampler_diagnostics") or {}
        if diagnostics.get("status") is not None:
            statuses[str(diagnostics["status"])] += 1
        if diagnostics.get("stopping_reason") is not None:
            stopping[str(diagnostics["stopping_reason"])] += 1
        causes.update(_shortfalls(diagnostics))
        if record["status"] != "success":
            failure_counts = diagnostics.get("counts") or {}
            for name in ("removed_majority", "removed_minority", "retained_labeled"):
                if name in failure_counts and failure_counts[name] is not None:
                    failed_removals[name] += _integer(failure_counts[name], f"failed {name}")
                    failed_removal_coverage[name] += 1
            continue
        counts = record["sample_counts"]
        for name in ("labeled_training", "resampled_training", "selected_pseudo_labeled", "generated_synthetic",
                     "generated_minority", "generated_majority", "removed_majority", "removed_minority", "retained_labeled"):
            totals[name] += counts[name]
        totals["hidden_training"] += len(record["unlabeled_indices"])
        totals["algorithm_unlabeled"] += len(record["algorithm_unlabeled_indices"])
        if method == "sshr":
            for name, value in (diagnostics.get("coordinate_duplicates") or {}).items():
                if name != "comparison_policy" and value is not None:
                    duplicate_totals[name] += _integer(value, f"coordinate duplicate {name}")
                    duplicate_coverage[name] += 1
            audit = record["pseudo_label_audit"]
            totals["pseudo_correct"] += audit["correct_count"]
            if audit["accuracy"] is not None:
                pseudo_accuracies.append(audit["accuracy"])
            for name, value in diagnostics["quotas"].items():
                if name in ("weighted_removal_requested", "weighted_removal_actual", "weighted_removal_unmet",
                            "additions_requested", "additions_unmet", "smote_fallback_requested"):
                    totals[name] += _integer(value, name)
        if "shortfall_count" in diagnostics:
            totals["sampler_shortfall_count"] += _integer(diagnostics["shortfall_count"], "shortfall_count")
        for name, item in record["training_overlap_proxy"].items():
            if not isinstance(item, dict):
                continue
            aggregate = proxy_totals.setdefault(name, Counter())
            for field in ("rows", "nearest_majority_count", "distance_tie_count"):
                aggregate[field] += item[field]
    proxies = {name: {**dict(counts), "nearest_majority_rate": counts["nearest_majority_count"] / counts["rows"] if counts["rows"] else None}
               for name, counts in proxy_totals.items()}
    selected = totals.get("selected_pseudo_labeled", 0)
    return {"dataset_id": manifest["dataset_id"], "run_id": run_id, "classifier": classifier, "method": method,
            "n_saved_fold_events": len(records), "n_successful_fold_events": len(successful),
            "counts_scope": "successful fold events; repeated rows and separate classifiers are not unique observations",
            "totals": dict(totals), "pseudo_label_event_weighted_accuracy": totals.get("pseudo_correct", 0) / selected if selected else None,
            "pseudo_label_mean_nonempty_fold_accuracy": float(np.mean(pseudo_accuracies)) if pseudo_accuracies else None,
            "pseudo_label_nonempty_fold_events": len(pseudo_accuracies), "sampler_status_counts": dict(statuses),
            "sampler_stopping_reason_counts": dict(stopping), "shortfall_causes_all_saved_jobs": dict(causes),
            "coordinate_duplicate_fold_event_counts": dict(duplicate_totals),
            "coordinate_duplicate_reporting_coverage": {
                name: {"reported_successful_fold_events": count, "missing_successful_fold_events": len(successful) - count}
                for name, count in duplicate_coverage.items()},
            "coordinate_duplicate_scope": "successful fold events per classifier; missing diagnostic fields are not assumed zero",
            "failed_job_removal_totals": dict(failed_removals),
            "failed_job_removal_reporting_fold_counts": dict(failed_removal_coverage),
            "failed_job_removal_scope": "saved unsuccessful jobs with available counts; absent counts are unknown",
            "training_overlap_proxy": proxies,
            "overlap_policy": "strict nearest observed majority vs minority, original reduced L only; ties separate; not true overlap"}


def summarize_controlled_run(directory: str | Path) -> dict[str, Any]:
    """Read a fixed run without training; reject corrupt or inconsistent evidence."""
    manifest, run_id, records, keys = _load(directory)
    config, scores, coverage, diagnostics, clusters = manifest["config"], [], [], [], []
    for classifier in config["classifiers"]:
        successes = {method: {key for key in keys if (record := records[(classifier, method, *key)]) is not None
                              and record["status"] == "success"} for method in config["methods"]}
        shared = set.intersection(*successes.values())
        for method in config["methods"]:
            saved, statuses, issues = [], Counter(), []
            for key in keys:
                record = records[(classifier, method, *key)]
                status = record["status"] if record is not None else "missing"
                statuses[status] += 1
                if record is not None:
                    saved.append(record)
                if status != "success":
                    issues.append({"repeat": key[0], "fold": key[1], "status": status,
                                   "error": record.get("error") if record else None,
                                   "shortfalls": _shortfalls(record.get("resampler_diagnostics")) if record else {}})
            coverage.append({"dataset_id": manifest["dataset_id"], "run_id": run_id, "classifier": classifier,
                             "method": method, "expected": len(keys),
                             **{name: statuses[name] for name in ("success", "failed", "unavailable", "missing")},
                             "all_methods_shared_success": len(shared), "shared_fold_ids": [list(key) for key in sorted(shared)],
                             "issues": issues})
            for scope, chosen in (("all_methods_shared_success", shared), ("all_method_success", successes[method])):
                chosen = sorted(chosen)
                for metric in STUDY_METRICS:
                    values = [records[(classifier, method, *key)]["metrics"][metric] for key in chosen]
                    mean, sd = _statistics(values)
                    paired = [key for key in chosen if key in successes["nores"]]
                    baseline = [records[(classifier, "nores", *key)]["metrics"][metric] for key in paired]
                    differences = [records[(classifier, method, *key)]["metrics"][metric] - base
                                   for key, base in zip(paired, baseline)]
                    scores.append({"dataset_id": manifest["dataset_id"], "run_id": run_id, "classifier": classifier,
                                   "method": method, "scope": scope, "metric": metric, "mean": mean,
                                   "sample_sd": sd, "n_folds": len(chosen), "fold_ids": [list(key) for key in chosen],
                                   "n_paired_with_nores": len(paired), "nores_mean": _statistics(baseline)[0],
                                   "paired_mean_difference": _statistics(differences)[0]})
            diagnostics.append(_aggregate_diagnostics(manifest, run_id, classifier, method, saved))
            if method == "sshr":
                for record in saved:
                    diag = record.get("resampler_diagnostics") or {}
                    allocations = {item["cluster_id"]: item for item in diag.get("smote_allocations", [])}
                    for item in diag.get("clusters", []):
                        allocation = allocations.get(item["id"], {})
                        clusters.append({"dataset_id": manifest["dataset_id"], "run_id": run_id,
                                         **record["job"], "job_status": record["status"], "cluster_id": item["id"],
                                         **{field: item.get(field) for field in ("label", "size", "labeled_size", "unlabeled_size",
                                              "observed_minority_count", "observed_majority_count", "unanchored", "impure")},
                                         "minority_anchored": bool(item.get("observed_minority_count", 0)),
                                         "smote_pool_size": allocation.get("pool_size"),
                                         "smote_requested": allocation.get("requested"), "smote_generated": allocation.get("generated"),
                                         "smote_shortage_reason": allocation.get("shortage_reason")})
    return {"schema_version": 1, "identity": _identity(manifest, run_id), "policy": SUMMARY_POLICY,
            "scores": scores, "coverage": coverage, "diagnostics": diagnostics, "clusters": clusters}


def summarize_unlabeled_ablation(with_directory: str | Path, without_directory: str | Path) -> dict[str, Any]:
    """Compare SSHR with/without U on identical reduced L and untouched test folds."""
    full, full_id, with_records, keys = _load(with_directory)
    empty, empty_id, without_records, empty_keys = _load(without_directory)
    first, second = dict(full), dict(empty)
    full_config, empty_config = first.pop("config"), second.pop("config")
    if first != second or keys != empty_keys:
        raise ValueError("Ablation dataset, source, folds, masks, preprocessing or software identities differ.")
    if full_config.get("use_unlabeled") is not True or empty_config.get("use_unlabeled") is not False:
        raise ValueError("Ablation must compare use_unlabeled=True against False.")
    for field in set(full_config) | set(empty_config):
        if field not in {"use_unlabeled", "methods", "method_parameters"} and full_config.get(field) != empty_config.get(field):
            raise ValueError(f"Ablation configuration differs in {field}.")
    if not {"nores", "sshr"} <= set(full_config["methods"]) or set(empty_config["methods"]) != {"nores", "sshr"}:
        raise ValueError("Without-U ablation must contain exactly nores and sshr.")
    for method in ("nores", "sshr"):
        if full_config.get("method_parameters", {}).get(method, {}) != empty_config.get("method_parameters", {}).get(method, {}):
            raise ValueError("Ablation changes common-method parameters.")
    rows, coverage = [], []
    for classifier in full_config["classifiers"]:
        pairs = []
        issues = []
        identical_nores_pairs = []
        for key in keys:
            nores_a, nores_b = with_records[(classifier, "nores", *key)], without_records[(classifier, "nores", *key)]
            if nores_a is not None and nores_b is not None:
                for field in ("status", "metrics", "predictions", "probabilities", "probability_labels", "true_labels",
                              "preprocessing", "train_indices", "test_indices", "labeled_indices", "seed"):
                    if nores_a.get(field) != nores_b.get(field):
                        raise ValueError(f"Duplicated nores records differ in {field}; ablation is not controlled.")
                identical_nores_pairs.append(key)
            for name, record in (("with_u", nores_a), ("without_u", nores_b)):
                if record is None or record["status"] != "success":
                    issues.append({"repeat": key[0], "fold": key[1], "run": name, "method": "nores",
                                   "status": record["status"] if record else "missing", "error": record.get("error") if record else None})
            a, b = with_records[(classifier, "sshr", *key)], without_records[(classifier, "sshr", *key)]
            for name, record in (("with_u", a), ("without_u", b)):
                if record is None or record["status"] != "success":
                    issues.append({"repeat": key[0], "fold": key[1], "run": name,
                                   "method": "sshr",
                                   "status": record["status"] if record else "missing", "error": record.get("error") if record else None})
            if a is not None and b is not None and a["status"] == b["status"] == "success":
                if a.get("preprocessing") != b.get("preprocessing") or a.get("resampler_parameters") != b.get("resampler_parameters"):
                    raise ValueError("Ablation preprocessing or effective SSHR parameters differ.")
                if b["sample_counts"]["selected_pseudo_labeled"] or b.get("algorithm_unlabeled_indices") != []:
                    raise ValueError("Without-U SSHR record selected pseudo-labels or received unlabelled rows.")
                pairs.append(key)
        coverage.append({"dataset_id": full["dataset_id"], "classifier": classifier, "expected": len(keys),
                         "paired_success": len(pairs), "paired_fold_ids": [list(key) for key in pairs],
                         "identical_saved_nores_pairs": len(identical_nores_pairs),
                         "identical_saved_nores_fold_ids": [list(key) for key in identical_nores_pairs], "issues": issues})
        for metric in STUDY_METRICS:
            a_values = [with_records[(classifier, "sshr", *key)]["metrics"][metric] for key in pairs]
            b_values = [without_records[(classifier, "sshr", *key)]["metrics"][metric] for key in pairs]
            mean_difference, sd_difference = _statistics([a - b for a, b in zip(a_values, b_values)])
            totals = {}
            for name, records in (("with_u", with_records), ("without_u", without_records)):
                totals[name] = {field: sum(records[(classifier, "sshr", *key)]["sample_counts"][field] for key in pairs)
                                for field in ("selected_pseudo_labeled", "generated_synthetic", "removed_majority", "removed_minority")}
            rows.append({"dataset_id": full["dataset_id"], "with_u_run_id": full_id, "without_u_run_id": empty_id,
                         "classifier": classifier, "metric": metric, "n_paired_folds": len(pairs),
                         "fold_ids": [list(key) for key in pairs], "with_u_mean": _statistics(a_values)[0],
                         "without_u_mean": _statistics(b_values)[0], "paired_mean_difference": mean_difference,
                         "paired_difference_sample_sd": sd_difference, "paired_fold_event_counts": totals})
    return {"schema_version": 1, "identity": {"with_u": _identity(full, full_id), "without_u": _identity(empty, empty_id)},
            "policy": SUMMARY_POLICY, "rows": rows, "coverage": coverage}


def _csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical_json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def export_study_reports(report: dict, ablation: dict, output_dir: str | Path) -> dict[str, str]:
    """Export aggregate tables; raw features and individual predictions stay local."""
    if report.get("identity", {}).get("run_id") != ablation.get("identity", {}).get("with_u", {}).get("run_id"):
        raise ValueError("Report and ablation refer to different with-U runs.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in ("scores", "coverage", "diagnostics", "clusters", "ablation"):
        path = directory / f"{name}.csv"
        _csv(path, ablation["rows"] if name == "ablation" else report[name])
        result[name] = str(path)
    path = directory / "summary.json"
    path.write_text(canonical_json({"report": report, "ablation": ablation}) + "\n", encoding="utf-8")
    result["summary"] = str(path)
    return result


__all__ = ["STUDY_METRICS", "summarize_controlled_run", "summarize_unlabeled_ablation", "export_study_reports"]
