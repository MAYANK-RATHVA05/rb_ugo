"""A historical file match is still not evidence of UGO paper membership."""
import hashlib
import json
from pathlib import Path

import pytest

from src.data_loader import load_dataset
from src.dataset_registry import check_all_candidates, load_candidate_registry


def test_eight_historical_candidates_are_unverified_and_absent_here(tmp_path):
    registry = load_candidate_registry()
    assert len(registry["candidates"]) == 8
    assert {entry["ugo_paper_membership"] for entry in registry["candidates"]} == {"unverified"}
    assert all(entry["official_fold_pairs"] == 0 for entry in registry["candidates"])
    # Absence is a property of this empty fixture, not of a user's checkout.
    # Real raw data may legitimately be present during development.
    assert {check.status for check in check_all_candidates(raw_dir=tmp_path)} == {"raw_file_absent"}


def _fixture_registry(tmp_path: Path) -> tuple[Path, Path]:
    source = Path(__file__).parent / "fixtures" / "toy.dat"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    file = raw_dir / "toy.dat"
    file.write_bytes(source.read_bytes())
    loaded = load_dataset(file)
    document = {"schema_version": 1, "candidates": [{
        "id": "toy", "filename": "toy.dat",
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "ugo_paper_membership": "unverified", "rows": len(loaded.X),
        "features": len(loaded.feature_names), "majority_count": loaded.majority_count,
        "minority_count": loaded.minority_count, "official_fold_pairs": 0,
        "target": loaded.target_name, "majority_label": loaded.majority_label,
        "minority_label": loaded.minority_label,
        "categorical_features": list(loaded.categorical_feature_names),
    }]}
    manifest = tmp_path / "registry.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return raw_dir, manifest


def test_hash_then_parse_then_metadata_check_never_verifies_membership(tmp_path):
    raw_dir, manifest = _fixture_registry(tmp_path)
    check = check_all_candidates(raw_dir, manifest)[0]
    assert check.status == "historical_bytes_verified"
    assert check.ugo_paper_membership == "unverified"
    (raw_dir / "toy.dat").write_text("changed", encoding="utf-8")
    assert check_all_candidates(raw_dir, manifest)[0].status == "sha256_mismatch"


@pytest.mark.parametrize("mutation", [
    {"ugo_paper_membership": "confirmed"},
    {"filename": "../outside.dat"},
    {"sha256": "bad"},
    {"rows": 999},
])
def test_registry_rejects_fabricated_identity_or_invalid_metadata(tmp_path, mutation):
    _, manifest = _fixture_registry(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["candidates"][0].update(mutation)
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_candidate_registry(manifest)
