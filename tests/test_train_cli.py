"""Portable CLI and saved-model checks using synthetic software fixtures."""
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.data_loader import load_dataset


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_csv(tmp_path):
    X, y = make_classification(n_samples=80, n_features=4, n_informative=3,
                               n_redundant=0, weights=[0.75, 0.25], random_state=42)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["class"] = np.where(y == 1, "rare", "normal")
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    return path


def invoke(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "main_train.py"), *map(str, arguments)],
                          cwd=ROOT, capture_output=True, text=True)


def test_cli_exports_reloadable_models_and_audit_records(synthetic_csv, tmp_path):
    output = tmp_path / "first_run"
    completed = invoke("--data", synthetic_csv, "--target", "class", "--max-iter", 3,
                       "--output-dir", output)
    assert completed.returncode == 0, completed.stderr
    assert "untouched test rows" in completed.stdout
    report = json.loads((output / "run.json").read_text())
    split = json.loads((output / "split.json").read_text())
    predictions = pd.read_csv(output / "predictions.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    assert set(metrics["method"]) == {"NORES", "UGO-ROS", "UGO-SMOTE"}
    assert report["run_metadata"]["source_filename"] == synthetic_csv.name
    assert len(report["run_metadata"]["source_sha256"]) == 64
    assert set(split["train_indices"]).isdisjoint(split["test_indices"])
    dataset = load_dataset(synthetic_csv, "class")
    X_test = dataset.X.iloc[split["test_indices"]]
    for name, filename in report["run_metadata"]["model_files"].items():
        model = joblib.load(output / filename)
        expected = predictions.loc[predictions["method"] == name]
        np.testing.assert_array_equal(model.predict(X_test), expected["prediction"])
        np.testing.assert_allclose(model.predict_proba(X_test)[:, 1], expected["minority_probability"])
    for name in ["ugo_ros", "ugo_smote"]:
        diagnostic = json.loads((output / f"{name}_diagnostics.json").read_text())
        assert diagnostic["effective_parameters"]["cap_policy"] == "section_3_3"


def test_cli_refuses_existing_output_without_overwriting(synthetic_csv, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("previous run")
    completed = invoke("--data", synthetic_csv, "--target", "class", "--output-dir", output)
    assert completed.returncode == 1
    assert "already exists" in completed.stderr
    assert sentinel.read_text() == "previous run"
    assert list(output.iterdir()) == [sentinel]


def test_cli_rejects_invalid_split_without_creating_outputs(synthetic_csv, tmp_path):
    output = tmp_path / "invalid_run"
    completed = invoke("--data", synthetic_csv, "--target", "class", "--test-size", 1,
                       "--output-dir", output)
    assert completed.returncode == 1
    assert "Training error" in completed.stderr
    assert not output.exists()


def test_cli_missing_file_is_readable_error(tmp_path):
    completed = invoke("--data", tmp_path / "missing.dat")
    assert completed.returncode == 1
    assert "Dataset file does not exist" in completed.stderr
