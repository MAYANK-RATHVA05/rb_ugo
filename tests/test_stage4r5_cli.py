import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run([sys.executable, str(ROOT / "main_stage4r5.py"), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def test_software_command_and_resume(tmp_path):
    first = command("--software-check", "--output", tmp_path)
    assert first.returncode == 0, first.stderr
    assert '"reused_jobs": 0' in first.stdout
    second = command("--software-check", "--output", tmp_path)
    assert second.returncode == 0, second.stderr
    assert '"reused_jobs": 20' in second.stdout


def test_csv_requires_explicit_original_positive_label_and_respects_it(tmp_path):
    path = tmp_path / "software.csv"
    rng = np.random.default_rng(10)
    frame = pd.DataFrame({"value": rng.normal(size=80), "category": ["a", "b"] * 40,
                          "outcome": ["control"] * 60 + ["case"] * 20})
    frame.loc[4, "value"] = np.nan
    frame.to_csv(path, index=False)
    output = tmp_path / "output"
    missing = command("--dataset", path, "--target", "outcome", "--output", output)
    assert missing.returncode == 2
    assert "requires --positive-label" in missing.stderr
    assert not output.exists()
    completed = command("--dataset", path, "--target", "outcome", "--positive-label", "control",
                        "--output", output, "--methods", "nores")
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())["manifest"]
    assert manifest["provenance"]["source_positive_label"] == "control"
    assert manifest["positive_label"] == 0  # Domain-positive class can be the majority.
    assert len(list(output.glob("job-*.json"))) == 10
    for record_path in output.glob("job-*.json"):
        record = json.loads(record_path.read_text())["record"]
        assert record["metrics"]["labels"] == [1, 0]


def test_unavailable_classifier_produces_nonzero_exit_and_records(tmp_path):
    result = command("--software-check", "--output", tmp_path,
                     "--classifiers", "NNET", "--methods", "nores")
    assert result.returncode == 1
    assert '"unavailable": 10' in result.stdout
    records = [json.loads(p.read_text())["record"] for p in tmp_path.glob("job-*.json")]
    assert len(records) == 10 and all(r["metrics"] is None for r in records)
