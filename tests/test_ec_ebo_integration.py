"""Fold-local, opt-in EC-EBO evaluation checks on artificial data only."""

import json

import numpy as np
import pandas as pd
import pytest

from src.comparators import make_comparator
from src.comparison_report import summarize_saved_run
from src.ec_ebo import ECEBO
from src.evaluation import AVAILABLE_METHODS, METHODS, EvaluationConfig, evaluate_ten_fold
from src.preprocessing import FoldPreprocessor


def test_ec_ebo_is_opt_in_and_comparator_receives_runner_seed():
    assert "ec_ebo" in AVAILABLE_METHODS
    assert "ec_ebo" not in METHODS
    sampler = make_comparator("ec_ebo", random_state=53, n_iterations=7)
    assert isinstance(sampler, ECEBO)
    assert sampler.random_state == 53
    assert sampler.n_iterations == 7


def test_paired_runner_keeps_ec_ebo_training_only_and_saves_shortfalls(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(80, 2)), columns=["a", "b"])
    y = np.array([0] * 60 + [1] * 20)
    sampled_inputs = []
    original_fit_resample = ECEBO.fit_resample

    def capture_training_input(self, fitting, labels):
        sampled_inputs.append((fitting.copy(), labels.copy()))
        return original_fit_resample(self, fitting, labels)

    monkeypatch.setattr(ECEBO, "fit_resample", capture_training_input)
    config = EvaluationConfig(classifiers=("GNB",), methods=("nores", "ec_ebo"),
                              method_parameters={"ec_ebo": {"k_majority": 2,
                                                              "n_iterations": 4}})
    result = evaluate_ten_fold(
        X, y, dataset_id="ec_ebo_artificial_software_fixture", positive_label=1,
        provenance={"kind": "artificial software fixture", "paper_dataset": False},
        output_dir=tmp_path, config=config)

    assert len(sampled_inputs) == 10
    assert len(result["records"]) == 20
    assert all(record["status"] == "success" for record in result["records"])
    for fold, (baseline, ec_ebo) in enumerate(zip(result["records"][::2],
                                                  result["records"][1::2])):
        assert baseline["train_indices"] == ec_ebo["train_indices"]
        assert baseline["test_indices"] == ec_ebo["test_indices"]
        train = ec_ebo["train_indices"]
        test = ec_ebo["test_indices"]
        assert not set(train) & set(test)
        expected = FoldPreprocessor().fit_transform(X.iloc[train], y[train])
        np.testing.assert_allclose(sampled_inputs[fold][0], expected)
        np.testing.assert_array_equal(sampled_inputs[fold][1], y[train])
        diagnostics = ec_ebo["resampler_diagnostics"]
        assert diagnostics["interpretation"] == "audited_mean_entropy_minority_side"
        assert diagnostics["random_state"] == 42 + fold
        assert ec_ebo["resampler_parameters"]["k_majority"] == 2
        assert ec_ebo["resampler_parameters"]["n_iterations"] == 4
        assert diagnostics["requested_count"] == 36
        assert diagnostics["generated_count"] == ec_ebo["sample_counts"]["added_rows"]
        assert diagnostics["shortfall_count"] == (diagnostics["requested_count"]
                                                   - diagnostics["generated_count"])
        assert sum(diagnostics["cluster_allocations"]) == 36
        assert sum(diagnostics["cluster_generated_counts"]) == diagnostics["generated_count"]
        assert ec_ebo["resampled_counts"][1]["count"] == 18 + diagnostics["generated_count"]

    manifest = json.loads((tmp_path / "manifest.json").read_text())["manifest"]
    assert manifest["config"]["methods"] == ["nores", "ec_ebo"]
    assert len(list(tmp_path.glob("job-*.json"))) == 20
    report = summarize_saved_run(tmp_path)
    assert report["methods"] == ["nores", "ec_ebo"]
    assert report["status"] == "complete"
    assert all(row["shared_fold_count"] == 10 for row in report["rows"])
    assert sum(report["coverage"][1]["sampler_status_counts"].values()) == 10
    assert evaluate_ten_fold(
        X, y, dataset_id="ec_ebo_artificial_software_fixture", positive_label=1,
        provenance={"kind": "artificial software fixture", "paper_dataset": False},
        output_dir=tmp_path, config=config)["reused_jobs"] == 20
    assert len(sampled_inputs) == 10


@pytest.mark.parametrize("change, message", [
    ("categorical", "original numerical features"),
    ("boolean", "original numerical features"),
    ("no_scale", "fold-local scaling"),
])
def test_ec_ebo_rejects_unsupported_source_features_before_writing(tmp_path, change, message):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    y = np.array([0] * 30 + [1] * 10)
    if change == "categorical":
        X["kind"] = ["a", "b"] * 20
    elif change == "boolean":
        X["flag"] = [True, False] * 20
    config = EvaluationConfig(methods=("nores", "ec_ebo"), scale=change != "no_scale")
    with pytest.raises(ValueError, match=message):
        evaluate_ten_fold(
            X, y, dataset_id="unsupported_ec_ebo_input", positive_label=1,
            provenance={"kind": "artificial software fixture"},
            output_dir=tmp_path, config=config)
    assert not list(tmp_path.iterdir())
