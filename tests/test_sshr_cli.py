"""SSHR CLI software checks; temporary input files are artificial fixtures."""
import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_sshr_study as cli


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_software_check_runs_six_methods_and_resumes(tmp_path, capsys):
    output = tmp_path / "results"
    assert cli.main(["--software-check", "--output", str(output)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["purpose"] == "software validation only"
    assert len(first["coverage"]) == 6
    assert all(item["success"] == 2 for item in first["coverage"])
    manifest = json.loads((output / "manifest.json").read_text())["manifest"]
    assert manifest["dataset_id"] == "sshr_artificial_software_check"
    assert manifest["rows"] == 120
    assert "not experimental evidence" in manifest["provenance"]["kind"]
    assert len(manifest["provenance"]["runner_script_sha256"]) == 64
    assert cli.main(["--software-check", "--output", str(output)]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["run_id"] == first["run_id"]
    assert resumed["reused_jobs"] == 12


@pytest.mark.parametrize("expected", [None, "bad", "0" * 64])
def test_bad_or_missing_hash_rejects_before_loading_or_training(tmp_path, monkeypatch, expected):
    source = tmp_path / "artificial.csv"
    source.write_text("x,class\n1,normal\n2,normal\n3,rare\n")
    output = tmp_path / "output"

    def forbidden(*args, **kwargs):
        raise AssertionError("The hash guard allowed data loading or evaluation")

    monkeypatch.setattr(cli, "load_dataset", forbidden)
    monkeypatch.setattr(cli, "evaluate_controlled", forbidden)
    args = ["--data", str(source), "--target", "class", "--numeric-source", "--output", str(output)]
    if expected is not None:
        args += ["--sha256", expected]
    with pytest.raises(SystemExit) as failure:
        cli.main(args)
    assert failure.value.code == 2
    assert not output.exists()


def test_csv_requires_original_numeric_source_confirmation(tmp_path, monkeypatch):
    source = tmp_path / "artificial.csv"
    source.write_text("x,class\n1,normal\n2,normal\n3,rare\n")
    output = tmp_path / "output"
    monkeypatch.setattr(cli, "load_dataset", lambda *a, **kw: pytest.fail("Unconfirmed CSV was loaded"))
    with pytest.raises(SystemExit) as failure:
        cli.main(["--data", str(source), "--sha256", sha(source), "--target", "class", "--output", str(output)])
    assert failure.value.code == 2
    assert not output.exists()


def test_source_mutation_during_loading_is_rejected_before_training(tmp_path, monkeypatch):
    source = tmp_path / "artificial.csv"
    source.write_text("x,class\n1,normal\n2,normal\n3,rare\n")
    output = tmp_path / "output"

    def mutate(*args, **kwargs):
        source.write_text("changed during loading")
        return None

    monkeypatch.setattr(cli, "load_dataset", mutate)
    monkeypatch.setattr(cli, "evaluate_controlled", lambda *a, **kw: pytest.fail("Changed source reached training"))
    with pytest.raises(SystemExit) as failure:
        cli.main(["--data", str(source), "--sha256", sha(source), "--target", "class", "--numeric-source", "--output", str(output)])
    assert failure.value.code == 2
    assert not output.exists()


def test_hash_verified_loader_passes_original_label_mapping_and_metadata(tmp_path, monkeypatch, capsys):
    source = tmp_path / "artificial.csv"
    source.write_text("x,z,class\n1,2,normal\n2,3,normal\n3,4,rare\n")
    expected_hash = sha(source)
    captured = {}

    def evaluate(X, y, **kwargs):
        captured.update(X=X, y=y, **kwargs)
        return {"records": [{"status": "success"}], "run_id": "software-spy",
                "output_dir": str(tmp_path / "output"), "reused_jobs": 0, "coverage": []}

    monkeypatch.setattr(cli, "evaluate_controlled", evaluate)
    assert cli.main(["--data", str(source), "--sha256", expected_hash,
                     "--target", "class", "--numeric-source", "--output", str(tmp_path / "output")]) == 0
    metadata = captured["provenance"]
    assert metadata["source_sha256"] == expected_hash
    assert metadata["original_label_mapping"] == {"normal": 0, "rare": 1}
    assert metadata["source_positive_label"] == "rare"
    assert metadata["feature_types"] == {"x": "numeric", "z": "numeric"}
    assert captured["positive_label"] == 1
    assert captured["config"].n_splits == captured["config"].n_repeats == 5
    assert "caller" in metadata["feature_metadata_authority"]
    capsys.readouterr()


def test_declared_keel_categorical_feature_cannot_enter_runner(tmp_path):
    source = Path(__file__).parent / "fixtures" / "toy.dat"
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="categorical"):
        cli.main(["--data", str(source), "--sha256", sha(source), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("argument", [["--sha256", "0" * 64], ["--target", "class"], ["--numeric-source"]])
def test_local_file_options_do_not_apply_to_artificial_generation(tmp_path, argument):
    with pytest.raises(SystemExit) as failure:
        cli.main(["--software-check", "--output", str(tmp_path / "output"), *argument])
    assert failure.value.code == 2
    assert not (tmp_path / "output").exists()
