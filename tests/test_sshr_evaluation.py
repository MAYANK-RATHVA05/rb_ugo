"""Synthetic software checks for the training-only SSHR controlled runner."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

import src.evaluation as legacy_evaluation
import src.sshr_evaluation as evaluation
from src.classifier_registry import UnavailableClassifierError
from src.preprocessing import FoldPreprocessor
from src.sshr_evaluation import (
    ControlledEvaluationConfig,
    controlled_fold_assignments,
    controlled_label_masks,
    evaluate_controlled,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(121)
    y = np.array([0] * 48 + [1] * 24)
    X = pd.DataFrame(rng.normal(size=(72, 3)), columns=["a", "b", "c"])
    X.loc[y == 1, "a"] += 4.0
    return X, y


def config(**changes):
    return ControlledEvaluationConfig(n_splits=2, n_repeats=1, **changes)


def provenance(X, **changes):
    return {"kind": "artificial software fixture",
            "feature_types": {column: "continuous" for column in X.columns}, **changes}


def run(tmp_path, data, **kwargs):
    X, y = data
    return evaluate_controlled(
        X, y, dataset_id="controlled_software_fixture", positive_label=kwargs.pop("positive_label", 1),
        provenance=kwargs.pop("provenance", provenance(X)), output_dir=tmp_path,
        config=kwargs.pop("config", config()), **kwargs)


def fixed_assignments(data):
    X, y = data
    folds, _ = controlled_fold_assignments(y, n_splits=2, n_repeats=1)
    masks, _ = controlled_label_masks(y, folds)
    return folds, masks, provenance(X, fold_source="fixed software folds", mask_source="fixed software masks")


def test_real_paired_folds_masking_metrics_and_resume(tmp_path, data, monkeypatch):
    result = run(tmp_path, data)
    assert len(result["records"]) == 4
    assert all(record["status"] == "success" for record in result["records"]), result["records"]
    X, y = data
    for fold in range(2):
        baseline, sshr = result["records"][2 * fold:2 * fold + 2]
        for key in ("train_indices", "test_indices", "labeled_indices", "unlabeled_indices"):
            assert baseline[key] == sshr[key]
        labeled, unlabeled, testing = map(set, (sshr["labeled_indices"], sshr["unlabeled_indices"], sshr["test_indices"]))
        assert not labeled & unlabeled and not (labeled | unlabeled) & testing
        assert labeled | unlabeled == set(sshr["train_indices"])
        assert len(labeled) == 18 and len(unlabeled) == 18
        assert baseline["preprocessing"]["training_rows"] == 18
        assert sshr["preprocessing"]["training_rows"] == 18
        assert len(baseline["predictions"]) == len(testing)
        for record in (baseline, sshr):
            metrics = record["metrics"]
            assert {"f1", "gmean", "mcc", "probability_auc", "fpr"} <= set(metrics)
            from sklearn.metrics import roc_auc_score
            probs = np.asarray(record["probabilities"])
            assert metrics["probability_auc"] == pytest.approx(
                roc_auc_score(y[record["test_indices"]] == 1, probs[:, 1]))
            tn, fp = metrics["counts"]["tn"], metrics["counts"]["fp"]
            assert metrics["fpr"] == pytest.approx(fp / (tn + fp))
        audit = sshr["pseudo_label_audit"]
        assert audit["selected_count"] == audit["correct_count"] + audit["incorrect_count"]
        assert len(sshr["row_lineage"]) == sshr["sample_counts"]["resampled_training"]
        for lineage in sshr["row_lineage"]:
            if lineage["kind"] == "labeled":
                assert lineage["global_index"] == sshr["labeled_indices"][lineage["original_index"]]
            elif lineage["kind"] == "pseudo_labeled":
                assert lineage["global_index"] == sshr["algorithm_unlabeled_indices"][lineage["original_index"]]
                assert lineage["global_index"] in audit["selected_global_indices"]
            else:
                assert lineage["kind"] == "smote"
                for point in (lineage["anchor"], lineage["neighbor"]):
                    source = sshr["labeled_indices"] if point["kind"] == "labeled" else sshr["algorithm_unlabeled_indices"]
                    assert point["global_index"] == source[point["index"]]
        if audit["selected_count"]:
            assert audit["accuracy"] == pytest.approx(audit["correct_count"] / audit["selected_count"])
        else:
            assert audit["accuracy"] is None

    def do_not_fit(*args, **kwargs):
        raise AssertionError("A completed job was fitted during resume")
    monkeypatch.setattr(evaluation, "make_classifier", do_not_fit)
    resumed = run(tmp_path, data)
    assert resumed["records"] == result["records"]
    assert resumed["reused_jobs"] == 4


class SpyClassifier(BaseEstimator):
    fits = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fits.append((self, X.copy(), y.copy()))
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.75, 0.25], (len(X), 1))


class SpySampler(BaseEstimator):
    inputs = []

    def __init__(self, estimator=None, method="nores"):
        self.estimator = estimator
        self.method = method

    def fit_resample(self, X, y, X_unlabeled=None):
        self.inputs.append((self.method, X.copy(), y.copy(),
                            None if X_unlabeled is None else X_unlabeled.copy()))
        self.diagnostics_ = {"software_spy": True,
                             "counts": {"removed_majority": 0, "removed_minority": 0}}
        if self.method == "ugo_smote":
            self.estimator_ = self.estimator.fit(X.copy(), y.copy())
        if self.method == "sshr":
            self.selected_unlabeled_indices_ = np.array([0], dtype=int) if len(X_unlabeled) else np.array([], dtype=int)
            self.synthetic_samples_ = np.empty((0, X.shape[1]))
            self.retained_labeled_indices_ = np.arange(len(X))
            self.row_lineage_ = [{"kind": "labeled", "original_index": int(i)}
                                 for i in range(len(X))]
            if len(X_unlabeled):
                self.row_lineage_.append({"kind": "pseudo_labeled", "original_index": 0})
                return np.vstack([X, X_unlabeled[0]]), np.r_[y, 1]
        return X.copy(), y.copy()


def install_spies(monkeypatch):
    SpyClassifier.fits = []
    SpySampler.inputs = []
    monkeypatch.setattr(evaluation, "make_classifier", lambda *a, **kw: SpyClassifier())
    monkeypatch.setattr(evaluation, "make_controlled_resampler",
                        lambda method, estimator, **kw: SpySampler(estimator=estimator, method=method))


def test_all_six_methods_share_only_observed_training_labels_and_fresh_ugo_fit(tmp_path, data, monkeypatch):
    install_spies(monkeypatch)
    preprocessing_fits = []
    real_fit = FoldPreprocessor.fit

    def preprocessing_fit(self, X, y):
        preprocessing_fits.append((X.index.tolist(), np.asarray(y).copy()))
        return real_fit(self, X, y)

    monkeypatch.setattr(FoldPreprocessor, "fit", preprocessing_fit)
    methods = ("nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo", "sshr")
    result = run(tmp_path, data, config=config(methods=methods))
    assert all(record["status"] == "success" for record in result["records"]), result["records"]
    X, y = data
    assert len(SpySampler.inputs) == len(preprocessing_fits) == 12
    for position, record in enumerate(result["records"]):
        observed, hidden, testing = record["labeled_indices"], record["unlabeled_indices"], record["test_indices"]
        assert preprocessing_fits[position][0] == observed
        np.testing.assert_array_equal(preprocessing_fits[position][1], y[observed])
        method, sampled_X, sampled_y, unlabeled_X = SpySampler.inputs[position]
        expected_preprocessing = FoldPreprocessor().fit(X.iloc[observed], y[observed])
        np.testing.assert_allclose(sampled_X, expected_preprocessing.transform(X.iloc[observed]))
        np.testing.assert_array_equal(sampled_y, y[observed])
        if method == "sshr":
            np.testing.assert_allclose(unlabeled_X, expected_preprocessing.transform(X.iloc[hidden]))
        else:
            assert unlabeled_X is None
        assert not set(observed + hidden) & set(testing)
    # Each fold fits six final models plus one UGO-internal model. The final
    # UGO model must be an independent clone even if sampling fitted its template.
    assert len(SpyClassifier.fits) == 14
    for fold in range(2):
        internal, final = SpyClassifier.fits[7 * fold + 2:7 * fold + 4]
        assert internal[0] is not final[0]
    assert legacy_evaluation.EvaluationConfig().methods == legacy_evaluation.METHODS
    assert "sshr" not in legacy_evaluation.AVAILABLE_METHODS


def test_unlabeled_ablation_keeps_labeled_rows_and_outer_folds_identical(tmp_path, data, monkeypatch):
    install_spies(monkeypatch)
    with_pool = run(tmp_path / "with", data)
    with_inputs = [(a, b.copy(), c.copy(), None if d is None else d.copy()) for a, b, c, d in SpySampler.inputs]
    SpySampler.inputs = []
    without_pool = run(tmp_path / "without", data, config=config(use_unlabeled=False))
    assert all(r["status"] == "success" for r in without_pool["records"]), without_pool["records"]
    for original, ablated in zip(with_pool["records"], without_pool["records"]):
        for key in ("train_indices", "test_indices", "labeled_indices", "unlabeled_indices"):
            assert original[key] == ablated[key]
    for with_input, without_input in zip(with_inputs, SpySampler.inputs):
        np.testing.assert_array_equal(with_input[1], without_input[1])
        np.testing.assert_array_equal(with_input[2], without_input[2])
        if with_input[0] == "sshr":
            assert len(with_input[3]) == 18 and len(without_input[3]) == 0
    assert all(r["pseudo_label_audit"]["selected_count"] == 0 for r in without_pool["records"] if r["job"]["method"] == "sshr")


def test_frozen_masks_keep_hidden_labels_out_of_generation(tmp_path, data, monkeypatch):
    install_spies(monkeypatch)
    folds, masks, metadata = fixed_assignments(data)
    first = run(tmp_path / "original", data, supplied_folds=folds, supplied_masks=masks, provenance=metadata)
    first_input = copy.deepcopy(SpySampler.inputs[1])
    X, y = data
    changed_y = y.copy()
    changed_y[masks[0]["unlabeled_indices"][0]] = 1 - changed_y[masks[0]["unlabeled_indices"][0]]
    SpySampler.inputs = []
    second = run(tmp_path / "changed", (X, changed_y), supplied_folds=folds, supplied_masks=masks, provenance=metadata)
    changed_input = SpySampler.inputs[1]
    for left, right in zip(first_input[1:], changed_input[1:]):
        np.testing.assert_array_equal(left, right)
    a, b = first["records"][1], second["records"][1]
    assert a["resampler_diagnostics"] == b["resampler_diagnostics"]
    assert a["predictions"] == b["predictions"]
    assert a["pseudo_label_audit"]["accuracy"] != b["pseudo_label_audit"]["accuracy"]


def test_heldout_features_and_labels_cannot_change_fold_training(tmp_path, data, monkeypatch):
    install_spies(monkeypatch)
    folds, masks, metadata = fixed_assignments(data)
    run(tmp_path / "original", data, supplied_folds=folds, supplied_masks=masks, provenance=metadata)
    originals = copy.deepcopy(SpySampler.inputs[:2])
    X, y = data
    changed_X, changed_y = X.copy(), y.copy()
    test = folds[0]["test_indices"]
    changed_X.iloc[test] += 1000.0
    changed_y[test] = 1 - changed_y[test]
    SpySampler.inputs = []
    result = run(tmp_path / "changed", (changed_X, changed_y), supplied_folds=folds, supplied_masks=masks, provenance=metadata)
    assert all(r["status"] == "success" for r in result["records"][:2]), result["records"][:2]
    for left, right in zip(originals, SpySampler.inputs[:2]):
        for a, b in zip(left[1:], right[1:]):
            if a is None:
                assert b is None
            else:
                np.testing.assert_array_equal(a, b)


def test_interrupted_run_is_atomic_and_resumes_remaining_jobs(tmp_path, data):
    seen = []
    def stop(record, cached):
        seen.append(record)
        if len(seen) == 2:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, data, on_record=stop)
    assert len(list(tmp_path.glob("job-*.json"))) == 2
    resumed = run(tmp_path, data)
    assert resumed["reused_jobs"] == 2
    assert resumed["records"][:2] == seen


def test_failed_and_unavailable_jobs_have_no_scores_and_retry_is_explicit(tmp_path, data, monkeypatch):
    real_constructor = evaluation.make_classifier
    def classifier(name, **kwargs):
        if name == "NNET":
            raise UnavailableClassifierError("software fixture unavailable backend")
        return real_constructor(name, **kwargs)
    monkeypatch.setattr(evaluation, "make_classifier", classifier)
    selected = config(classifiers=("LogReg", "NNET"), methods=("nores", "smote"),
                      method_parameters={"smote": {"k_neighbors": 100}})
    result = run(tmp_path, data, config=selected)
    assert [r["status"] for r in result["records"]].count("success") == 2
    assert [r["status"] for r in result["records"]].count("failed") == 2
    assert [r["status"] for r in result["records"]].count("unavailable") == 4
    for record in result["records"]:
        if record["status"] != "success":
            assert record["metrics"] is record["predictions"] is record["probabilities"] is None
            assert record["error"]
    assert run(tmp_path, data, config=selected)["reused_jobs"] == 8
    assert run(tmp_path, data, config=selected, retry_failures=True)["reused_jobs"] == 6


def test_all_six_real_constructor_paths_run_on_synthetic_numeric_data(tmp_path, data):
    selected = config(
        methods=("nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo", "sshr"),
        method_parameters={"smote": {"k_neighbors": 1},
                           "ugo_smote": {"max_iter": 2, "k_smote": 1},
                           "bfch_smote": {"k_neighbors": 1},
                           "ec_ebo": {"n_iterations": 2},
                           "sshr": {"k_neighbors": 1}})
    result = run(tmp_path, data, config=selected)
    assert len(result["records"]) == 12
    assert all(record["status"] == "success" for record in result["records"]), [
        (record["job"], record["error"]) for record in result["records"] if record["status"] != "success"]
    assert all(len(item["paired_successful_folds_with_nores"]) == 2 for item in result["coverage"])
    for record in result["records"]:
        if record["job"]["method"] == "sshr":
            assert record["training_overlap_proxy"]["pseudo_labeled"]["rows"] == record["sample_counts"]["selected_pseudo_labeled"]
            assert record["training_overlap_proxy"]["smote"]["rows"] == record["sample_counts"]["generated_smote"]


def test_diagnosed_sshr_failure_preserves_diagnostics_without_scores(tmp_path, data, monkeypatch):
    from src.sshr_resampling import SSHRResamplingError
    real_factory = evaluation.make_controlled_resampler
    evidence = {"stage": "preliminary_cleaning", "counts": {"removed_minority": 3}}

    class FailingSampler(BaseEstimator):
        def fit_resample(self, X, y, X_unlabeled):
            self.diagnostics_ = copy.deepcopy(evidence)
            raise SSHRResamplingError("software fixture removed all observed minority anchors", evidence)

    def sampler(method, estimator, **kwargs):
        return FailingSampler() if method == "sshr" else real_factory(method, estimator, **kwargs)

    monkeypatch.setattr(evaluation, "make_controlled_resampler", sampler)
    result = run(tmp_path, data)
    for record in result["records"]:
        if record["job"]["method"] == "sshr":
            assert record["status"] == "failed"
            assert record["metrics"] is record["predictions"] is record["probabilities"] is None
            assert record["resampler_diagnostics"] == evidence
            assert record["error"]["stage"] == "resampling"
            assert record["pseudo_label_audit"] is None


def test_nonfinite_prediction_probabilities_are_recorded_as_failure(tmp_path, data, monkeypatch):
    class InvalidClassifier(SpyClassifier):
        def predict_proba(self, X):
            return np.full((len(X), 2), np.nan)

    monkeypatch.setattr(evaluation, "make_classifier", lambda *a, **kw: InvalidClassifier())
    result = run(tmp_path, data, config=config(methods=("nores",)))
    assert all(record["status"] == "failed" and record["metrics"] is None for record in result["records"])
    assert all(record["error"]["stage"] == "prediction" for record in result["records"])


@pytest.mark.parametrize("change", ["data", "seed", "positive", "provenance", "folds", "masks", "software"])
def test_resume_identity_covers_data_masks_folds_source_and_configuration(tmp_path, data, change, monkeypatch):
    folds, masks, metadata = fixed_assignments(data)
    run(tmp_path, data, supplied_folds=folds, supplied_masks=masks, provenance=metadata)
    X, y = data
    kwargs = {"supplied_folds": folds, "supplied_masks": masks, "provenance": metadata}
    if change == "data":
        X = X.copy()
        X.iloc[0, 0] += 0.125
    elif change == "seed":
        kwargs["config"] = config(seed=43)
    elif change == "positive":
        kwargs["positive_label"] = 0
    elif change == "provenance":
        kwargs["provenance"] = {**metadata, "source_note": "changed"}
    elif change == "folds":
        kwargs["supplied_folds"] = copy.deepcopy(folds)
        kwargs["supplied_folds"][0]["train_indices"].reverse()
    elif change == "masks":
        kwargs["supplied_masks"] = copy.deepcopy(masks)
        kwargs["supplied_masks"][0]["labeled_indices"].reverse()
    else:
        identity = evaluation._software_identity()
        identity["source_sha256"] = "changed source"
        monkeypatch.setattr(evaluation, "_software_identity", lambda: identity)
    with pytest.raises(ValueError, match="different or corrupt"):
        run(tmp_path, (X, y), **kwargs)


@pytest.mark.parametrize("kind", ["missing", "categorical", "onehot", "extra", "object", "bool", "complex", "duplicate"])
def test_source_domain_rejections_happen_before_results_are_written(tmp_path, data, kind):
    X, y = data
    X = X.copy()
    metadata = provenance(X)
    if kind == "missing":
        del metadata["feature_types"]
    elif kind in {"categorical", "onehot"}:
        metadata["feature_types"]["a"] = "categorical" if kind == "categorical" else "one_hot"
    elif kind == "extra":
        metadata["feature_types"]["not_a_source_column"] = "numeric"
    elif kind == "object":
        X["a"] = X["a"].astype(str)
    elif kind == "bool":
        X["a"] = X["a"] > 0
    elif kind == "complex":
        X["a"] = X["a"].astype(complex)
    else:
        X.iloc[1] = X.iloc[0]
    with pytest.raises(ValueError):
        run(tmp_path, (X, y), provenance=metadata)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("selected", [
    config(methods=("sshr",)), config(methods=("nores", "unknown")),
    config(seed=True), config(labeled_fraction=0), config(labeled_fraction=1.1),
    config(min_labeled_per_class=1), config(use_unlabeled=1),
    config(method_parameters={"sshr": {"random_state": 123}}),
    config(method_parameters={"sshr": {"minority_label": 0}}),
    config(method_parameters={"nores": {"arbitrary": 1}}),
    config(method_parameters={"smote": {"k_neighbors": 1}}),
    config(methods=None), 0,
])
def test_invalid_configuration_cannot_begin_a_run(tmp_path, data, selected):
    with pytest.raises(ValueError):
        run(tmp_path, data, config=selected)
    assert not list(tmp_path.iterdir())


def test_object_labels_cannot_hide_infinite_class_identity(tmp_path, data):
    X, y = data
    invalid = np.where(y == 1, np.inf, 0.0).astype(object)
    with pytest.raises(ValueError):
        run(tmp_path, (X, invalid), positive_label=np.inf, config=config(methods=("nores",)))
    assert not list(tmp_path.iterdir())


def test_supplied_masks_do_not_claim_a_generated_random_seed(data):
    X, y = data
    folds, masks, metadata = fixed_assignments(data)
    supplied, _ = controlled_label_masks(y, folds, seed=123, supplied_masks=masks)
    assert all(mask["mask_seed"] is None and mask["per_class_design"] is None for mask in supplied)
    for original, external in zip(masks, supplied):
        assert original["labeled_indices"] == external["labeled_indices"]
        assert original["unlabeled_indices"] == external["unlabeled_indices"]


def test_callback_mutation_cannot_change_frozen_run_parameters(tmp_path, data):
    parameters = {"smote": {"k_neighbors": 1}}
    selected = config(methods=("nores", "smote"), method_parameters=parameters)

    def mutate_caller_config(record, cached):
        parameters["smote"]["k_neighbors"] = 100

    result = run(tmp_path, data, config=selected, on_record=mutate_caller_config)
    assert all(record["status"] == "success" for record in result["records"]), result["records"]
    assert parameters["smote"]["k_neighbors"] == 100
    assert all(record["resampler_parameters"]["k_neighbors"] == 1
               for record in result["records"] if record["job"]["method"] == "smote")
    manifest = json.loads((tmp_path / "manifest.json").read_text())["manifest"]
    assert manifest["config"]["method_parameters"]["smote"]["k_neighbors"] == 1


@pytest.mark.parametrize("kind", ["test_overlap", "test_missing", "mask_overlap", "mask_missing", "mask_test", "missing_authority"])
def test_bad_supplied_partitions_are_rejected_before_writing(tmp_path, data, kind):
    folds, masks, metadata = fixed_assignments(data)
    if kind == "test_overlap":
        folds[0]["test_indices"][0] = folds[0]["train_indices"][0]
    elif kind == "test_missing":
        folds[0]["test_indices"].pop()
    elif kind == "mask_overlap":
        masks[0]["unlabeled_indices"][0] = masks[0]["labeled_indices"][0]
    elif kind == "mask_missing":
        masks[0]["unlabeled_indices"].pop()
    elif kind == "mask_test":
        masks[0]["labeled_indices"][0] = folds[0]["test_indices"][0]
    else:
        del metadata["mask_source"]
    with pytest.raises(ValueError):
        run(tmp_path, data, supplied_folds=folds, supplied_masks=masks, provenance=metadata)
    assert not list(tmp_path.iterdir())


def test_stratified_masks_are_deterministic_record_clamping_and_cover_only_train(data):
    X, y = data
    folds, _ = controlled_fold_assignments(y, n_splits=2, n_repeats=2)
    assert folds == controlled_fold_assignments(y, n_splits=2, n_repeats=2)[0]
    assert folds != controlled_fold_assignments(y, seed=43, n_splits=2, n_repeats=2)[0]
    for repeat in range(2):
        tested = [index for fold in folds if fold["repeat"] == repeat for index in fold["test_indices"]]
        assert sorted(tested) == list(range(len(X)))
    masks, protocol = controlled_label_masks(y, folds, labeled_fraction=0.01)
    assert masks == controlled_label_masks(y, folds, labeled_fraction=0.01)[0]
    for fold, mask in zip(folds, masks):
        assert set(mask["labeled_indices"]) | set(mask["unlabeled_indices"]) == set(fold["train_indices"])
        assert np.count_nonzero(y[mask["labeled_indices"]] == 0) == 2
        assert np.count_nonzero(y[mask["labeled_indices"]] == 1) == 2
        assert not set(mask["labeled_indices"]) & set(fold["test_indices"])
        assert all(item["minimum_clamp_applied"] for item in mask["per_class_design"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_provenance_cannot_collapse_to_null_run_identity(tmp_path, data, value):
    X, _ = data
    with pytest.raises(ValueError, match="Provenance.*finite JSON"):
        run(tmp_path, data, provenance=provenance(X, source_note=value))
    assert not list(tmp_path.iterdir())


def test_explicit_null_provenance_is_preserved(tmp_path, data):
    X, _ = data
    result = run(tmp_path, data, provenance=provenance(X, source_note=None),
                 config=config(methods=("nores",)))
    assert all(record["status"] == "success" for record in result["records"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())["manifest"]
    assert "source_note" in manifest["provenance"]
    assert manifest["provenance"]["source_note"] is None


@pytest.mark.parametrize("lineage", [
    [{"kind": "outer_test", "original_index": 0}],
    [{"kind": "labeled", "original_index": -1}],
    [{"kind": "pseudo_labeled", "original_index": 1}],
    [{"kind": "smote", "anchor": {"kind": "outer_test", "index": 0},
      "neighbor": {"kind": "labeled", "index": 0}}],
])
def test_global_lineage_cannot_map_unknown_or_out_of_pool_origins(lineage):
    with pytest.raises(ValueError):
        evaluation._global_lineage(lineage, np.array([4, 8]), np.array([12]))
