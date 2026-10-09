"""Tests for configuration and project initialization only."""
from pathlib import Path

import pytest

import main_stage1
from src import config


def test_config_paths_and_seed() -> None:
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert config.RAW_DATA_DIR == config.PROJECT_ROOT / "data" / "raw"
    assert config.PROCESSED_DATA_DIR == config.PROJECT_ROOT / "data" / "processed"
    assert config.RESULTS_DIR == config.PROJECT_ROOT / "results"
    assert config.RANDOM_STATE == 42


def test_required_directories_exist() -> None:
    main_stage1.verify_project_directories()


def test_initialization_output(capsys: pytest.CaptureFixture[str]) -> None:
    main_stage1.main()
    output = capsys.readouterr().out
    assert "Python version:" in output
    for path in (config.PROJECT_ROOT, config.RAW_DATA_DIR,
                 config.PROCESSED_DATA_DIR, config.RESULTS_DIR):
        assert str(path) in output
    assert "Stage 1 project initialization successful." in output


def test_missing_directory_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setattr(main_stage1, "required_directories", lambda: (missing,))
    with pytest.raises(FileNotFoundError, match="Missing required project directories") as exc:
        main_stage1.verify_project_directories()
    assert str(missing) in str(exc.value)


def test_file_cannot_replace_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = tmp_path / "not_a_directory"
    file.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(main_stage1, "required_directories", lambda: (file,))
    with pytest.raises(FileNotFoundError):
        main_stage1.verify_project_directories()
