import hashlib
import json

import pandas as pd
import pytest

import src.benchmark as benchmark
from src.evaluation import EvaluationConfig


def fixture_manifest(tmp_path):
    frame = pd.DataFrame({"x": list(range(10)) + list(range(10)),
                          "Class": ["negative"] * 10 + ["positive"] * 10})
    source = tmp_path / "fixture.csv"
    frame.to_csv(source, index=False)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    entries = []
    for index in range(74):
        entries.append({
            "dataset_id": f"dataset_{index:02d}", "relative_path": "fixture.csv",
            "sha256": digest, "target_column": "Class", "positive_label": "positive",
            "minority_label": "positive", "rows": 20, "input_features": 1,
            "majority_count": 10, "minority_count": 10, "fidelity": "software_fixture",
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "dataset_count": 74,
                                "scope": "artificial benchmark software fixture",
                                "datasets": entries}))
    return path


def test_manifest_validation_and_selected_resume(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path)
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    output = tmp_path / "results"
    config = EvaluationConfig(classifiers=("GNB",), methods=("nores",))
    first = benchmark.run_benchmark(output, config, manifest_path=manifest,
                                    selected_datasets=("dataset_00",))
    assert first["status"]["completed_datasets"] == 1
    assert first["status"]["success"] == 10
    assert first["identity"]["expected_jobs"] == 740
    assert first["completed"][0]["reused_jobs"] == 0
    resumed = benchmark.run_benchmark(output, config, manifest_path=manifest,
                                      selected_datasets=("dataset_00",))
    assert resumed["completed"][0]["reused_jobs"] == 10


def test_manifest_checksum_and_output_identity_fail_closed(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path)
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    document = json.loads(manifest.read_text())
    document["datasets"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="checksum"):
        benchmark.load_benchmark_manifest(manifest)

    manifest = fixture_manifest(tmp_path)
    output = tmp_path / "locked"
    benchmark.run_benchmark(output, EvaluationConfig(classifiers=("GNB",), methods=("nores",)),
                            manifest_path=manifest, selected_datasets=("dataset_00",))
    with pytest.raises(ValueError, match="different configuration"):
        benchmark.run_benchmark(output, EvaluationConfig(classifiers=("LogReg",), methods=("nores",)),
                                manifest_path=manifest, selected_datasets=("dataset_00",))
