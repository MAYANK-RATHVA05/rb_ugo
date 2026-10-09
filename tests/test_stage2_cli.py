"""CLI integration checks and isolated JSON report checks."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

import main_stage2

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(ROOT / "main_stage2.py"), *args],
                          cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("format_name", ["csv", "dat"])
def test_cli_loads_fixture(format_name: str) -> None:
    args = ["--data", str(FIXTURES / f"toy.{format_name}")]
    if format_name == "csv":
        args.extend(["--target", "class"])
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    assert "Samples: 6; features: 2" in result.stdout
    assert "Imbalance ratio (majority/minority): 5.0000" in result.stdout
    assert result.stderr == ""


def test_cli_help() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    for argument in ["--data", "--target", "--minority-label", "--save-json"]:
        assert argument in result.stdout


@pytest.mark.parametrize("args,message", [
    (["--data", "absent.csv", "--target", "class"], "does not exist"),
    (["--data", str(FIXTURES / "toy.csv")], "explicit target"),
    (["--data", str(FIXTURES / "toy.dat"), "--minority-label", "absent"], "not a target class"),
])
def test_cli_readable_errors(args: list[str], message: str) -> None:
    result = run_cli(*args)
    assert result.returncode == 1
    assert "Stage 2 error:" in result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_missing_required_argument() -> None:
    result = run_cli()
    assert result.returncode == 2


def test_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate output while retaining the production results/stage2 contract.
    monkeypatch.setattr(main_stage2, "RESULTS_DIR", tmp_path / "results")
    args = ["--data", str(FIXTURES / "toy.csv"), "--target", "class", "--save-json"]
    assert main_stage2.main(args) == 0
    report = json.loads((tmp_path / "results/stage2/toy_report.json").read_text(encoding="utf-8"))
    assert report["label_mapping"] == {"normal": 0, "rare": 1}
    assert report["statistics"]["imbalance_ratio"] == 5.0
    assert report["statistics"]["duplicated_complete_rows"] == 1
    assert report["feature_names"] == ["amount", "color"]
    assert report["source_format"] == "csv"
    assert report["source_filename"] == str((FIXTURES / "toy.csv").resolve())
    assert "X" not in report and "y" not in report
    assert main_stage2.main(args) == 1  # Preserve existing output.


@pytest.mark.parametrize("filename", ["../outside.json", "outside.txt", "dir/report.json", "dir\\report.json"])
def test_report_path_restrictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str) -> None:
    monkeypatch.setattr(main_stage2, "RESULTS_DIR", tmp_path / "results")
    assert main_stage2.main(["--data", str(FIXTURES / "toy.dat"), "--save-json", filename]) == 1
    assert not (tmp_path / "results").exists()


def test_named_json_and_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_stage2, "RESULTS_DIR", tmp_path / "results")
    args = ["--data", str(FIXTURES / "toy.dat"), "--save-json", "keel.json"]
    assert main_stage2.main(args) == 0
    assert (tmp_path / "results/stage2/keel.json").is_file()
    bad_root = tmp_path / "file"
    bad_root.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(main_stage2, "RESULTS_DIR", bad_root)
    assert main_stage2.main(args) == 1


def test_cli_tie_override(tmp_path: Path) -> None:
    path = tmp_path / "equal.csv"
    path.write_text("x,class\n1,a\n2,b\n", encoding="utf-8")
    result = run_cli("--data", str(path), "--target", "class", "--minority-label", "a")
    assert result.returncode == 0
    assert "Minority 'a' -> 1: 1 (50.00%)" in result.stdout
