import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

import src.evaluation as evaluation
from src.comparators import NoResampling
from src.evaluation import EvaluationConfig, evaluate_ten_fold, ten_fold_assignments
from src.preprocessing import FoldPreprocessor


@pytest.fixture
def data():
    rng = np.random.default_rng(17)
    y = np.array([0] * 60 + [1] * 20)
    X = pd.DataFrame(rng.normal(size=(80, 2)), columns=["a", "b"])
    return X, y


def run(tmp_path, data, **kwargs):
    X, y = data
    return evaluate_ten_fold(X, y, dataset_id="software_fixture", positive_label=1,
                             provenance={"kind": "artificial software fixture"}, output_dir=tmp_path,
                             config=kwargs.pop("config", EvaluationConfig(methods=("nores", "ros"))), **kwargs)


def test_real_paired_folds_predictions_and_resume(tmp_path, data, monkeypatch):
    result = run(tmp_path, data)
    assert len(result["records"]) == 20
    assert all(r["status"] == "success" for r in result["records"])
    for fold in range(10):
        baseline, ros = result["records"][2 * fold:2 * fold + 2]
        assert baseline["test_indices"] == ros["test_indices"]
        assert baseline["train_indices"] == ros["train_indices"]
        assert baseline["sample_counts"]["added_rows"] == 0
        assert ros["sample_counts"]["added_rows"] == 36
        assert len(ros["predictions"]) == len(ros["test_indices"])
        assert ros["positive_label"] == 1
        assert ros["preprocessing"]["training_rows"] == 72
    manifest = json.loads((tmp_path / "manifest.json").read_text())["manifest"]
    assert manifest["classifier_specs"][0]["implementation_status"] == "python_approximation"
    assert "not author folds" in manifest["fold_protocol"]
    assert len(manifest["folds"]) == 10
    def should_not_fit(*args, **kwargs):
        raise AssertionError("Resume attempted to refit a completed job")
    monkeypatch.setattr(evaluation, "_evaluate_job", should_not_fit)
    resumed = run(tmp_path, data)
    assert resumed["records"] == result["records"]
    assert resumed["reused_jobs"] == 20


def test_all_eleven_methods_operate_through_real_runner(tmp_path, data):
    # This small artificial fixture checks integration, not paper performance.
    config = EvaluationConfig(classifiers=("GNB",), method_parameters={
        "leao": {"iterations": 2}, "ugo_ros": {"max_iter": 2},
        "ugo_smote": {"max_iter": 2}, "ugo_rwo": {"max_iter": 2},
        "ugo_mcsmote": {"max_iter": 2}})
    result = run(tmp_path, data, config=config)
    assert len(result["records"]) == 110
    assert all(r["status"] == "success" for r in result["records"]), [
        (r["job"], r["error"]) for r in result["records"] if r["status"] != "success"]
    assert all(len(c["paired_successful_folds_with_nores"]) == 10 for c in result["coverage"])
    assert any(r["resampler_diagnostics"] for r in result["records"] if r["job"]["method"].startswith("ugo_"))


def test_training_only_inputs_and_fresh_final_fit(tmp_path, data, monkeypatch):
    fits, sampled, preprocess_fits = [], [], []

    class SpyClassifier(BaseEstimator):
        def fit(self, X, y):
            fits.append((id(self), X.copy(), y.copy()))
            self.classes_ = np.unique(y)
            self.rows_ = len(X)
            return self
        def predict(self, X):
            return np.zeros(len(X), dtype=int)
        def predict_proba(self, X):
            return np.tile([.75, .25], (len(X), 1))

    class SpySampler(BaseEstimator):
        def __init__(self, estimator):
            self.estimator = estimator
        def fit_resample(self, X, y):
            self.estimator_ = self.estimator.fit(X, y)
            sampled.append((X.copy(), y.copy()))
            self.diagnostics_ = {"fake_internal_training_rows": len(X)}
            return np.vstack([X, [88., 88.]]), np.r_[y, 1]

    real_fit = FoldPreprocessor.fit
    def fit_spy(self, X, y):
        preprocess_fits.append((X.index.tolist(), np.array(y)))
        return real_fit(self, X, y)
    monkeypatch.setattr(FoldPreprocessor, "fit", fit_spy)
    monkeypatch.setattr(evaluation, "make_classifier", lambda *a, **kw: SpyClassifier())
    monkeypatch.setattr(evaluation, "make_evaluation_resampler",
                        lambda method, estimator, **kw: NoResampling() if method == "nores" else SpySampler(estimator))
    result = run(tmp_path, data, config=EvaluationConfig(methods=("nores", "ugo_ros")))
    assert all(r["status"] == "success" for r in result["records"])
    X, y = data
    assert len(preprocess_fits) == 20
    assert len(fits) == 30
    assert len(sampled) == 10
    for fold in range(10):
        record = result["records"][2 * fold + 1]
        train, test = record["train_indices"], record["test_indices"]
        assert not set(train) & set(test)
        assert preprocess_fits[2 * fold][0] == train
        assert preprocess_fits[2 * fold + 1][0] == train
        np.testing.assert_array_equal(preprocess_fits[2 * fold + 1][1], y[train])
        expected = FoldPreprocessor().fit_transform(X.iloc[train], y[train])
        np.testing.assert_allclose(sampled[fold][0], expected)
        np.testing.assert_array_equal(sampled[fold][1], y[train])
        baseline, internal, final = fits[3 * fold:3 * fold + 3]
        assert len(baseline[1]) == len(internal[1]) == 72
        assert len(final[1]) == 73
        assert final[0] != internal[0]
        np.testing.assert_array_equal(final[1][-1], [88., 88.])


def test_failure_and_unavailable_are_preserved_with_no_scores(tmp_path, data, monkeypatch):
    real_constructor = evaluation.make_classifier
    def constructor(name, **kwargs):
        if name == "NNET":
            raise evaluation.UnavailableClassifierError("simulated optional backend absence")
        return real_constructor(name, **kwargs)
    monkeypatch.setattr(evaluation, "make_classifier", constructor)
    config = EvaluationConfig(classifiers=("LogReg", "NNET"), methods=("nores", "smote"),
                              method_parameters={"smote": {"k_neighbors": 100}})
    result = run(tmp_path, data, config=config)
    assert len(result["records"]) == 40
    assert sum(r["status"] == "success" for r in result["records"]) == 10
    assert sum(r["status"] == "failed" for r in result["records"]) == 10
    assert sum(r["status"] == "unavailable" for r in result["records"]) == 20
    for record in result["records"]:
        if record["status"] != "success":
            assert record["metrics"] is record["predictions"] is record["probabilities"] is None
            assert record["error"]
    assert result["coverage"][1]["paired_successful_folds_with_nores"] == []
    assert run(tmp_path, data, config=config)["reused_jobs"] == 40
    assert run(tmp_path, data, config=config, retry_failures=True)["reused_jobs"] == 30


def test_interruption_resumes_only_remaining_jobs(tmp_path, data):
    seen = []
    def interrupt(record, cached):
        seen.append(record)
        if len(seen) == 3:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, data, on_record=interrupt)
    assert len(list(tmp_path.glob("job-*.json"))) == 3
    result = run(tmp_path, data)
    assert result["reused_jobs"] == 3
    assert result["records"][:3] == seen


@pytest.mark.parametrize("change", ["data", "seed", "positive", "provenance", "folds", "software"])
def test_changed_run_identity_cannot_mix_results(tmp_path, data, change, monkeypatch):
    run(tmp_path, data)
    X, y = data
    kwargs = {}
    if change == "data":
        X = X.copy()
        X.loc[0, "a"] += 1.
    elif change == "seed":
        kwargs["config"] = EvaluationConfig(methods=("nores", "ros"), seed=43)
    elif change == "positive":
        kwargs["positive_label"] = 0
    elif change == "provenance":
        kwargs["provenance"] = {"kind": "other source"}
    elif change == "folds":
        folds, _ = ten_fold_assignments(y, seed=99)
        kwargs["supplied_folds"] = [(f["train_indices"], f["test_indices"]) for f in folds]
        kwargs["provenance"] = {"fold_source": "changed software folds"}
    else:
        identity = evaluation._software_identity()
        identity["source_sha256"] = "different implementation"
        monkeypatch.setattr(evaluation, "_software_identity", lambda: identity)
    with pytest.raises(ValueError, match="different or corrupt"):
        evaluate_ten_fold(X, y, dataset_id="software_fixture", positive_label=kwargs.pop("positive_label", 1),
                          provenance=kwargs.pop("provenance", {"kind": "artificial software fixture"}),
                          output_dir=tmp_path, config=kwargs.pop("config", EvaluationConfig(methods=("nores", "ros"))), **kwargs)


def test_ten_fold_validation_reproducibility_and_supplied_provenance(tmp_path, data):
    X, y = data
    folds, protocol = ten_fold_assignments(y)
    assert folds == ten_fold_assignments(y)[0]
    assert folds != ten_fold_assignments(y, seed=43)[0]
    test_indices = [i for fold in folds for i in fold["test_indices"]]
    assert sorted(test_indices) == list(range(80))
    supplied = [(f["train_indices"], f["test_indices"]) for f in folds]
    with pytest.raises(ValueError, match="fold_source"):
        run(tmp_path, data, supplied_folds=supplied)
    result = evaluate_ten_fold(X, y, dataset_id="software_fixture", positive_label=0,
                               provenance={"fold_source": "independently generated software folds"},
                               output_dir=tmp_path, config=EvaluationConfig(methods=("nores",)),
                               supplied_folds=supplied)
    assert all(r["metrics"]["labels"] == [1, 0] for r in result["records"])
    supplied[0] = (supplied[0][0], supplied[1][1])
    with pytest.raises(ValueError, match="complement"):
        ten_fold_assignments(y, supplied_folds=supplied)
    with pytest.raises(ValueError, match="at least ten"):
        ten_fold_assignments([0] * 100 + [1] * 9)


@pytest.mark.parametrize("config", [EvaluationConfig(methods=("ros",)),
                                     EvaluationConfig(seed=True),
                                     EvaluationConfig(method_parameters={"ros": {"random_state": 1}}),
                                     EvaluationConfig(methods=("nores", "unknown"))])
def test_invalid_configuration_fails_before_writing(tmp_path, data, config):
    with pytest.raises(ValueError):
        run(tmp_path, data, config=config)
    assert not list(tmp_path.iterdir())


def test_group_constraints_cannot_use_ordinary_stratification(tmp_path, data):
    with pytest.raises(ValueError, match="group/time constraints"):
        evaluate_ten_fold(*data, dataset_id="grouped", positive_label=1,
                          provenance={"requires_group_or_time_split": True}, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_nonfinite_final_probabilities_are_a_recorded_failure(tmp_path, data, monkeypatch):
    class InvalidClassifier(BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self
        def predict(self, X):
            return np.zeros(len(X), dtype=int)
        def predict_proba(self, X):
            return np.full((len(X), 2), np.nan)
    monkeypatch.setattr(evaluation, "make_classifier", lambda *a, **kw: InvalidClassifier())
    result = run(tmp_path, data, config=EvaluationConfig(methods=("nores",)))
    assert all(r["status"] == "failed" and r["metrics"] is None for r in result["records"])
    assert all(r["error"]["stage"] == "prediction" for r in result["records"])
