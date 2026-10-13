"""Failure behavior and audited header repair for local input preparation."""
import pytest

from scripts import prepare_all_methods_30_inputs as prep


def test_keel_header_repair_preserves_data_rows():
    original = (b"@relation sample\n@attribute a real\n@attribute class {negative,positive}\n"
                b"@input a,\n@output class\n@data\n1,negative\n2,positive\n")
    fixed = prep.normalize_keel_header(original)
    assert b"@inputs a\n@outputs class\n@data\n" in fixed
    assert fixed.split(b"@data\n", 1)[1] == original.split(b"@data\n", 1)[1]


def test_prepare_refuses_to_replace_existing_mismatched_input(tmp_path, monkeypatch):
    entry = {"dataset_id": "test", "relative_path": "data/raw/ugo74_approx/test.dat",
             "sha256": prep.digest(b"correct"), "provenance": {}}
    monkeypatch.setattr(prep, "registered_inputs", lambda: [entry])
    path = tmp_path / entry["relative_path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"user content")
    with pytest.raises(ValueError, match="left unchanged"):
        prep.prepare(tmp_path, tmp_path, cache=tmp_path / "cache")
    assert path.read_bytes() == b"user content"


def test_locate_archives_reports_missing_public_sources(tmp_path):
    with pytest.raises(FileNotFoundError, match="imb_IRlowerThan9.zip"):
        prep.locate_archives(tmp_path, {"imb_IRlowerThan9.zip"})


def test_fixed_selection_has_registered_source_hashes():
    entries = prep.registered_inputs()
    assert len(entries) == 30
    assert len({item["dataset_id"] for item in entries}) == 30
    assert all(len(item["sha256"]) == 64 for item in entries)
