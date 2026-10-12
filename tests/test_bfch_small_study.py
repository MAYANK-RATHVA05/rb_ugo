import hashlib
import json

import numpy as np
import pytest

import scripts.run_bfch_small_study as study


def source_fixture(tmp_path, monkeypatch, name, rows, features):
    path = tmp_path / f"{name}.dat"
    header = [f"@relation {name}"]
    header += [f"@attribute {feature} real" for feature in features]
    header += ["@attribute Class {positive, negative}",
               "@inputs " + ", ".join(features), "@outputs Class", "@data"]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    labels = [line.rsplit(",", 1)[-1].strip() for line in rows]
    entry = {"dataset_id": name, "relative_path": path.name,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
             "input_features": len(features), "minority_count": labels.count("positive"),
             "majority_count": labels.count("negative"), "fidelity": "test fixture"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"datasets": [entry]}), encoding="utf-8")
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "SOURCE_MANIFEST", manifest)
    return path


def test_wine_input_checksum_and_labels_are_checked(tmp_path, monkeypatch):
    rows = ["1,2,positive", "2,3,negative", "3,4,positive", "4,5,negative"]
    path = source_fixture(tmp_path, monkeypatch, "winequality_red_8_vs_6", rows, ["a", "b"])
    dataset_id, X, y, provenance, folds = study.prepare_dataset("winequality_red_8_vs_6")
    assert dataset_id == "winequality_red_8_vs_6" and folds is None
    assert X.shape == (4, 2)
    np.testing.assert_array_equal(y, [1, 0, 1, 0])
    assert provenance["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        study.prepare_dataset("winequality_red_8_vs_6")


def test_vowel_drops_identifiers_and_keeps_speakers_separate(tmp_path, monkeypatch):
    features = ["TT", "SpeakerNumber", "Sex", *study.ACOUSTIC_FEATURES]
    rows = []
    for speaker in range(10):
        for item in range(4):
            values = [item % 2, speaker, speaker % 2, *([float(speaker + item)] * 10)]
            rows.append(",".join(map(str, values)) + (",positive" if item < 2 else ",negative"))
    source_fixture(tmp_path, monkeypatch, "vowel0", rows, features)
    dataset_id, X, y, provenance, folds = study.prepare_dataset("vowel0")
    assert dataset_id == "vowel0_acoustic10_grouped"
    assert X.columns.tolist() == list(study.ACOUSTIC_FEATURES)
    assert provenance["excluded_features"] == ["TT", "SpeakerNumber", "Sex"]
    assert len(folds) == 10 and len(y) == 40
    groups = np.repeat(np.arange(10), 4)
    for training, testing in folds:
        assert not np.intersect1d(groups[training], groups[testing]).size
        assert set(y[training]) == set(y[testing]) == {0, 1}
