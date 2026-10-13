"""Software checks for the bounded, group-aware SSHR study driver.

All observations here are artificial. Raw study files are neither read nor
trained by these tests; evaluation is replaced with a recording callback.
"""

import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import pytest

import scripts.run_sshr_small_study as study


def grouped_artificial_rows():
    major_sizes = [1, 2, 3] * 10
    minor_sizes = [1, 2] * 15
    rows, labels = [], []
    for group, size in enumerate(major_sizes + minor_sizes):
        rows.extend([[float(group), float(group * group)]] * size)
        labels.extend([0 if group < len(major_sizes) else 1] * size)
    # Mixing the source order ensures global row positions, rather than sorted
    # feature values or deduplicated positions, remain the provenance indices.
    permutation = np.random.default_rng(19).permutation(len(rows))
    return pd.DataFrame(np.asarray(rows)[permutation], columns=["x", "z"]), np.asarray(labels)[permutation]


def assert_group_design(X, y, groups, folds, masks, *, repeats, splits, fraction=.5, minimum=2):
    assert len(groups) == len(X) == len(y)
    assert len(folds) == len(masks) == repeats * splits
    masks_by_key = {(record["repeat"], record["fold"]): record for record in masks}
    assert len(masks_by_key) == len(masks)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        assert np.unique(y[rows]).size == 1
        assert np.unique(X.iloc[rows].to_numpy(), axis=0).shape[0] == 1
    for repeat in range(repeats):
        testing = []
        repeat_folds = [record for record in folds if record["repeat"] == repeat]
        assert len(repeat_folds) == splits
        for record in repeat_folds:
            train = np.asarray(record["train_indices"], dtype=int)
            test = np.asarray(record["test_indices"], dtype=int)
            np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(X)))
            assert np.intersect1d(train, test).size == 0
            assert np.intersect1d(groups[train], groups[test]).size == 0
            assert len(np.unique(y[train])) == len(np.unique(y[test])) == 2
            assert train.tolist() == sorted(train.tolist())
            assert test.tolist() == sorted(test.tolist())
            testing.extend(test.tolist())
            mask = masks_by_key[(repeat, record["fold"])]
            labeled = np.asarray(mask["labeled_indices"], dtype=int)
            unlabeled = np.asarray(mask["unlabeled_indices"], dtype=int)
            np.testing.assert_array_equal(np.sort(np.r_[labeled, unlabeled]), train)
            assert np.intersect1d(labeled, unlabeled).size == 0
            assert np.intersect1d(groups[labeled], groups[unlabeled]).size == 0
            assert np.intersect1d(test, unlabeled).size == 0
            assert labeled.tolist() == sorted(labeled.tolist())
            assert unlabeled.tolist() == sorted(unlabeled.tolist())
            assert mask["per_class_design"] is not None
            class_design = {item["label"]: item for item in mask["per_class_design"]}
            for label in (0, 1):
                candidates = train[y[train] == label]
                selected = labeled[y[labeled] == label]
                target = min(len(candidates), max(minimum, int(np.floor(fraction * len(candidates)))))
                sizes = [np.count_nonzero(groups[candidates] == group)
                         for group in np.unique(groups[candidates])]
                assert len(selected) >= target
                assert len(selected) - target <= max(sizes) - 1
                detail = class_design[label]
                assert detail["training_count"] == len(candidates)
                assert detail["fraction_requested_count"] == int(np.floor(fraction * len(candidates)))
                assert detail["target_labeled_count"] == target
                assert detail["labeled_count"] == len(selected)
                assert detail["whole_group_overshoot"] == len(selected) - target
                assert detail["minimum_clamp_applied"] == (target != detail["fraction_requested_count"])
                assert set(detail["selected_coordinate_groups"]) == set(groups[selected])
        np.testing.assert_array_equal(np.sort(testing), np.arange(len(X)))


def test_group_aware_repeated_folds_preserve_every_original_row_and_training_mask():
    X, y = grouped_artificial_rows()
    before_X, before_y = X.copy(deep=True), y.copy()
    groups, folds, masks = study.group_folds_and_masks(X, y, n_splits=5, n_repeats=2)
    assert_group_design(X, y, groups, folds, masks, repeats=2, splits=5)
    assert len(np.unique(groups)) == 60
    assert len(X) > len(np.unique(groups))
    pd.testing.assert_frame_equal(X, before_X)
    np.testing.assert_array_equal(y, before_y)
    json.dumps(folds, allow_nan=False)
    json.dumps(masks, allow_nan=False)


def test_group_folds_and_masks_reproduce_by_seed_and_have_explicit_repeat_keys():
    X, y = grouped_artificial_rows()
    first = study.group_folds_and_masks(X, y, seed=42, n_splits=5, n_repeats=2)
    again = study.group_folds_and_masks(X, y, seed=42, n_splits=5, n_repeats=2)
    changed = study.group_folds_and_masks(X, y, seed=43, n_splits=5, n_repeats=2)
    np.testing.assert_array_equal(first[0], again[0])
    assert first[1:] == again[1:]
    assert first[1:] != changed[1:]
    assert {(record["repeat"], record["fold"]) for record in first[1]} == {
        (repeat, fold) for repeat in range(2) for fold in range(5)}


def test_group_masks_document_minimum_clamping_and_bounded_whole_group_overshoot():
    X, y = grouped_artificial_rows()
    groups, folds, masks = study.group_folds_and_masks(
        X, y, n_splits=5, n_repeats=1, labeled_fraction=.01, min_labeled_per_class=2)
    assert_group_design(X, y, groups, folds, masks, repeats=1, splits=5, fraction=.01)
    assert all(record["per_class_design"] for record in masks)


def test_full_label_fraction_keeps_all_training_groups_labeled():
    X, y = grouped_artificial_rows()
    groups, folds, masks = study.group_folds_and_masks(
        X, y, n_splits=5, n_repeats=1, labeled_fraction=1)
    assert_group_design(X, y, groups, folds, masks, repeats=1, splits=5, fraction=1)
    assert all(record["unlabeled_indices"] == [] for record in masks)


def test_conflicting_labels_on_identical_coordinates_are_rejected():
    X, y = grouped_artificial_rows()
    duplicates = np.flatnonzero(X.duplicated(keep=False))
    bad_y = y.copy()
    bad_y[duplicates[0]] = 1 - bad_y[duplicates[0]]
    with pytest.raises(ValueError, match="conflict|label|homogeneous"):
        study.group_folds_and_masks(X, bad_y)


def test_insufficient_minority_groups_raise_without_silently_reducing_folds():
    X = pd.DataFrame({"x": list(range(10)) + [20.] * 6 + [21.] * 6})
    y = np.array([0] * 10 + [1] * 12)
    with pytest.raises(ValueError, match="group|class|fold"):
        study.group_folds_and_masks(X, y, n_splits=5)


def test_training_class_too_small_for_required_label_budget_is_not_redrawn():
    X = pd.DataFrame({"x": list(range(12))})
    y = np.array([0] * 9 + [1] * 3)
    with pytest.raises(ValueError, match="class|labeled|minimum|fold"):
        study.group_folds_and_masks(X, y, n_splits=3, min_labeled_per_class=3)


def registered_artificial_source(tmp_path, *, categorical=False):
    """Register a temporary KEEL fixture under an allowed pilot ID."""
    raw = tmp_path / "data/raw/ecoli4.dat"
    raw.parent.mkdir(parents=True)
    header = (
        "@relation artificial_software_fixture\n"
        "@attribute x real [-1000000,1000000]\n"
        + ("@attribute z {a,b}\n" if categorical else "@attribute z real [-1000000,1000000]\n")
        + "@attribute Class {negative,positive}\n"
        "@inputs x,z\n@outputs Class\n@data\n")
    rows = []
    for index in range(60):
        feature = ("a" if index % 2 else "b") if categorical else str(2 * index + .5)
        rows.append(f"{index},{feature},{'negative' if index < 40 else 'positive'}\n")
    raw.write_text(header + "".join(rows), encoding="utf-8")
    entry = {
        "dataset_id": "ecoli4", "relative_path": "data/raw/ecoli4.dat",
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "bytes": raw.stat().st_size,
        "rows": 60, "input_features": 2, "feature_names": ["x", "z"],
        "target_name": "Class", "label_mapping": {"negative": 0, "positive": 1},
        "majority_count": 40, "minority_count": 20,
        "duplicate_coordinate_groups": 0, "duplicate_coordinate_pairs": 0,
        "duplicate_excess_rows": 0,
        "archive": {"relative_path": "data/incoming/ecoli4.zip", "sha256": "0" * 64,
                    "member": "ecoli4.dat", "required_for_execution": False},
    }
    document = {"schema_version": 1, "source_fidelity": "Artificial software fixture only",
                "duplicate_policy": "Preserve all rows and whole coordinate groups",
                "selection_policy": "Software checks only, no experimental evidence",
                "datasets": [entry]}
    manifest = tmp_path / study.SOURCE_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return raw, manifest, document


def rewrite_manifest(manifest, document):
    manifest.write_text(json.dumps(document), encoding="utf-8")


def test_prepare_verifies_artificial_source_without_deduplicating_or_reordering_rows(tmp_path):
    raw, manifest, document = registered_artificial_source(tmp_path)
    dataset_id, X, y, provenance, folds, masks = study.prepare_dataset("ecoli4", root=tmp_path)
    assert dataset_id == "ecoli4"
    assert X.shape == (60, 2)
    np.testing.assert_array_equal(X["x"].to_numpy(), np.arange(60))
    np.testing.assert_array_equal(X["z"].to_numpy(), 2 * np.arange(60) + .5)
    np.testing.assert_array_equal(y, [0] * 40 + [1] * 20)
    assert provenance["source_sha256"] == document["datasets"][0]["sha256"]
    assert provenance["source_bytes"] == raw.stat().st_size
    assert provenance["original_label_mapping"] == {"negative": 0, "positive": 1}
    assert provenance["feature_types"] == {"x": "numeric", "z": "numeric"}
    assert provenance["author_folds_unverified"] is True
    assert provenance["author_unlabeled_pool_protocol_unverified"] is True
    assert "UNVERIFIED" in provenance["archive"]["verification"]
    assert provenance["archive"]["required_for_execution"] is False
    assert len(provenance["source_registration_sha256"]) == 64
    assert len(provenance["runner_script_sha256"]) == 64
    groups = np.asarray(provenance["coordinate_groups"])
    assert_group_design(X, y, groups, folds, masks, repeats=5, splits=5)
    assert provenance["frozen_design"]["masks"] == masks
    assert provenance["frozen_design"]["folds_sha256"] == study.content_hash(folds)
    assert provenance["frozen_design"]["masks_sha256"] == study.content_hash(masks)
    assert provenance["duplicate_audit"]["duplicate_excess_rows"] == 0


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", 1)])
def test_source_hash_or_size_mismatch_is_rejected_before_loading(tmp_path, monkeypatch, field, value):
    _, manifest, document = registered_artificial_source(tmp_path)
    document["datasets"][0][field] = value
    rewrite_manifest(manifest, document)
    monkeypatch.setattr(study, "load_dataset", lambda *a, **kw: pytest.fail("Mismatched source was loaded"))
    with pytest.raises(ValueError, match="SHA-256|byte"):
        study.prepare_dataset("ecoli4", root=tmp_path)


def test_source_changed_during_loading_is_rejected_before_design(tmp_path, monkeypatch):
    raw, _, _ = registered_artificial_source(tmp_path)
    real_loader = study.load_dataset

    def mutating_loader(*args, **kwargs):
        loaded = real_loader(*args, **kwargs)
        raw.write_bytes(raw.read_bytes() + b"\n")
        return loaded

    monkeypatch.setattr(study, "load_dataset", mutating_loader)
    monkeypatch.setattr(study, "group_folds_and_masks", lambda *a, **kw: pytest.fail("Changed source reached design"))
    with pytest.raises(ValueError, match="changed during loading"):
        study.prepare_dataset("ecoli4", root=tmp_path)


@pytest.mark.parametrize("field,value", [
    ("rows", 59), ("input_features", 3), ("target_name", "wrong"),
    ("label_mapping", {"negative": 1, "positive": 0}),
    ("majority_count", 39), ("minority_count", 21),
])
def test_registered_dimensions_or_labels_mismatch_is_rejected_before_design(tmp_path, monkeypatch, field, value):
    _, manifest, document = registered_artificial_source(tmp_path)
    document["datasets"][0][field] = value
    rewrite_manifest(manifest, document)
    monkeypatch.setattr(study, "group_folds_and_masks", lambda *a, **kw: pytest.fail("Inconsistent source reached design"))
    with pytest.raises(ValueError, match="dimensions|labels|registration"):
        study.prepare_dataset("ecoli4", root=tmp_path)


def test_original_categorical_attribute_is_rejected_before_group_design(tmp_path, monkeypatch):
    registered_artificial_source(tmp_path, categorical=True)
    monkeypatch.setattr(study, "group_folds_and_masks", lambda *a, **kw: pytest.fail("Categorical source reached design"))
    with pytest.raises(ValueError, match="numeric|categorical"):
        study.prepare_dataset("ecoli4", root=tmp_path)


def test_registered_duplicate_audit_mismatch_is_rejected(tmp_path):
    _, manifest, document = registered_artificial_source(tmp_path)
    document["datasets"][0]["duplicate_excess_rows"] = 1
    rewrite_manifest(manifest, document)
    with pytest.raises(ValueError, match="Duplicate-coordinate"):
        study.prepare_dataset("ecoli4", root=tmp_path)


def test_missing_registered_file_is_reported_without_download(tmp_path):
    raw, _, _ = registered_artificial_source(tmp_path)
    raw.unlink()
    with pytest.raises(FileNotFoundError, match="never downloads"):
        study.prepare_dataset("ecoli4", root=tmp_path)


def test_matching_optional_archive_and_member_are_verified(tmp_path):
    raw, manifest, document = registered_artificial_source(tmp_path)
    archive = tmp_path / document["datasets"][0]["archive"]["relative_path"]
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ecoli4.dat", raw.read_bytes())
    document["datasets"][0]["archive"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    rewrite_manifest(manifest, document)
    _, _, _, provenance, _, _ = study.prepare_dataset("ecoli4", root=tmp_path)
    assert "verified" in provenance["archive"]["verification"]
    assert "UNVERIFIED" not in provenance["archive"]["verification"]


def test_optional_archive_with_matching_zip_hash_but_wrong_member_bytes_is_rejected(tmp_path):
    _, manifest, document = registered_artificial_source(tmp_path)
    archive = tmp_path / document["datasets"][0]["archive"]["relative_path"]
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ecoli4.dat", "Different source bytes")
    document["datasets"][0]["archive"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    rewrite_manifest(manifest, document)
    with pytest.raises(ValueError, match="member|extracted|SHA-256|source"):
        study.prepare_dataset("ecoli4", root=tmp_path)


def test_present_optional_archive_with_wrong_registered_hash_is_rejected(tmp_path):
    _, _, document = registered_artificial_source(tmp_path)
    archive = tmp_path / document["datasets"][0]["archive"]["relative_path"]
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ecoli4.dat", "Different source bytes")
    with pytest.raises(ValueError, match="archive|SHA-256"):
        study.prepare_dataset("ecoli4", root=tmp_path)


def test_committed_source_manifest_registers_exactly_two_known_hashes():
    # Registration is public metadata; this deliberately does not open raw data.
    document = json.loads((study.ROOT / study.SOURCE_MANIFEST).read_text(encoding="utf-8"))
    entries = {item["dataset_id"]: item for item in document["datasets"]}
    assert set(entries) == {"ecoli4", "yeast4"}
    assert entries["ecoli4"]["sha256"] == "1890c32ba2550cfe106d0524ebf6a42c2b50730f6521bd4f22eb58e7bda86bcb"
    assert entries["yeast4"]["sha256"] == "2e50d1b43c97875d2071859428d718487086e695a1a6ec559336ccadb40c1ff0"
    assert (entries["ecoli4"]["rows"], entries["ecoli4"]["input_features"],
            entries["ecoli4"]["majority_count"], entries["ecoli4"]["minority_count"]) == (336, 7, 316, 20)
    assert (entries["yeast4"]["rows"], entries["yeast4"]["input_features"],
            entries["yeast4"]["majority_count"], entries["yeast4"]["minority_count"]) == (1484, 8, 1433, 51)
    assert entries["yeast4"]["duplicate_coordinate_groups"] == 31
    assert entries["yeast4"]["duplicate_coordinate_pairs"] == 31
    assert entries["yeast4"]["duplicate_excess_rows"] == 31
    assert document["raw_datasets_committed"] is False
    for entry in entries.values():
        assert re.fullmatch("[0-9a-f]{64}", entry["sha256"])
        assert re.fullmatch("[0-9a-f]{64}", entry["archive"]["sha256"])
        assert entry["archive"]["required_for_execution"] is False


@pytest.mark.parametrize("name", ["all", "44", "74", "yeast3"])
def test_prepare_rejects_unregistered_benchmark_selection(tmp_path, name):
    registered_artificial_source(tmp_path)
    with pytest.raises(ValueError, match="Unknown|ambiguous"):
        study.prepare_dataset(name, root=tmp_path)


@pytest.mark.parametrize("selection", [["all"], ["ecoli4", "ecoli4"]])
def test_cli_rejects_expanded_or_duplicate_dataset_selection_before_preparation(tmp_path, monkeypatch, selection):
    monkeypatch.setattr(study, "prepare_dataset", lambda *a, **kw: pytest.fail("Unsupported selection reached preparation"))
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as caught:
        study.main(["--output", str(output), "--datasets", *selection])
    assert caught.value.code == 2
    assert not output.exists()


def install_software_main_spies(tmp_path, monkeypatch, *, fail=False):
    """Replace data preparation and all learning/report reads with spies."""
    import src.sshr_study_report as reports

    X, y = grouped_artificial_rows()
    groups, folds, masks = study.group_folds_and_masks(X, y, n_splits=5, n_repeats=5)
    prepared, events, calls, exports = {}, [], [], []
    for name in study.DATASETS:
        source = tmp_path / f"artificial-{name}.txt"
        source.write_text("Artificial software fixture; no classifier training\n", encoding="utf-8")
        provenance = {
            "source_relative_path": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_fidelity": "Artificial software fixture only",
            "fold_source": "Frozen artificial group folds",
            "mask_source": "Frozen artificial training-only masks",
            "duplicate_audit": {"duplicate_excess_rows": len(y) - len(np.unique(groups))},
            "coordinate_groups": groups.tolist(),
            "frozen_design": {"folds_sha256": study.content_hash(folds),
                              "masks_sha256": study.content_hash(masks), "masks": masks},
        }
        prepared[name] = (name, X.copy(), y.copy(), provenance, folds, masks)

    def prepare(name):
        events.append(("prepare", name))
        return prepared[name]

    def evaluate(X_seen, y_seen, **kwargs):
        name, config = kwargs["dataset_id"], kwargs["config"]
        events.append(("evaluate", name))
        calls.append((X_seen.copy(), y_seen.copy(), kwargs))
        records = []
        for fold in kwargs["supplied_folds"]:
            for classifier in config.classifiers:
                for method in config.methods:
                    record = {"status": "failure" if fail and not records else "success",
                              "repeat": fold["repeat"], "fold": fold["fold"],
                              "classifier": classifier, "method": method}
                    records.append(record)
                    kwargs["on_record"](record, True)
        return {"run_id": f"artificial-{name}-{config.use_unlabeled}",
                "output_dir": str(kwargs["output_dir"]), "records": records,
                "reused_jobs": len(records), "coverage": []}

    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "prepare_dataset", prepare)
    monkeypatch.setattr(study, "evaluate_controlled", evaluate)
    monkeypatch.setattr(reports, "summarize_controlled_run", lambda path: {"artificial": str(path)})
    monkeypatch.setattr(reports, "summarize_unlabeled_ablation", lambda first, second: {"artificial": [str(first), str(second)]})
    monkeypatch.setattr(reports, "export_study_reports", lambda report, ablation, path: exports.append((report, ablation, path)))
    return prepared, events, calls, exports


def test_main_freezes_sources_then_runs_paired_defaults_and_identical_no_u_ablation(tmp_path, monkeypatch, capsys):
    prepared, events, calls, exports = install_software_main_spies(tmp_path, monkeypatch)
    output, report_dir = tmp_path / "output", tmp_path / "reports"
    assert study.main(["--output", str(output), "--report-dir", str(report_dir)]) == 0
    assert events[:2] == [("prepare", "ecoli4"), ("prepare", "yeast4")]
    assert len(calls) == 4
    assert len(exports) == 2
    for name in study.DATASETS:
        paired = [item for item in calls if item[2]["dataset_id"] == name]
        assert len(paired) == 2
        full, ablation = paired
        for X_seen, y_seen, kwargs in paired:
            pd.testing.assert_frame_equal(X_seen, prepared[name][1])
            np.testing.assert_array_equal(y_seen, prepared[name][2])
            assert kwargs["positive_label"] == 1
            assert kwargs["supplied_folds"] == prepared[name][4]
            assert kwargs["supplied_masks"] == prepared[name][5]
            config = kwargs["config"]
            assert config.classifiers == ("LogReg", "GNB")
            assert config.n_splits == config.n_repeats == 5
            assert config.seed == 42
            assert config.scale is True
            assert config.labeled_fraction == .5
            assert config.min_labeled_per_class == 2
            assert config.method_parameters == {}
        assert full[2]["config"].methods == ("nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo", "sshr")
        assert full[2]["config"].use_unlabeled is True
        assert ablation[2]["config"].methods == ("nores", "sshr")
        assert ablation[2]["config"].use_unlabeled is False
        assert full[2]["supplied_folds"] == ablation[2]["supplied_folds"]
        assert full[2]["supplied_masks"] == ablation[2]["supplied_masks"]
    overview = json.loads((output / "overview.json").read_text(encoding="utf-8"))
    assert overview["not_author_paper_scores"] is True
    assert overview["no_superiority_claim"] is True
    assert len(overview["datasets"]) == 2
    assert sum(entry["runs"]["with-u"]["reused_jobs"] for entry in overview["datasets"]) == 600
    assert sum(entry["runs"]["without-u"]["reused_jobs"] for entry in overview["datasets"]) == 200
    for entry in overview["datasets"]:
        assert entry["folds"] == prepared[entry["dataset_id"]][4]
        assert entry["frozen_design"]["masks"] == prepared[entry["dataset_id"]][5]
    assert "cached=True" in capsys.readouterr().out


def test_main_reports_failure_without_dropping_methods_or_retrying_folds(tmp_path, monkeypatch, capsys):
    _, _, calls, _ = install_software_main_spies(tmp_path, monkeypatch, fail=True)
    assert study.main(["--output", str(tmp_path / "output")]) == 1
    assert len(calls) == 4
    assert all(kwargs["config"].n_splits == kwargs["config"].n_repeats == 5 for _, _, kwargs in calls)
    capsys.readouterr()


def test_main_source_changed_after_preparation_is_rejected_before_any_evaluation(tmp_path, monkeypatch):
    prepared, _, calls, _ = install_software_main_spies(tmp_path, monkeypatch)
    source = tmp_path / prepared["ecoli4"][3]["source_relative_path"]
    source.write_text("Changed after source preparation\n", encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="changed after preparation"):
        study.main(["--output", str(output)])
    assert calls == []
    assert not output.exists()


def test_main_prepares_all_selected_sources_before_creating_results(tmp_path, monkeypatch):
    prepared, _, calls, _ = install_software_main_spies(tmp_path, monkeypatch)

    def prepare(name):
        if name == "yeast4":
            raise ValueError("Second artificial source failed audit")
        return prepared[name]

    monkeypatch.setattr(study, "prepare_dataset", prepare)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Second artificial source"):
        study.main(["--output", str(output)])
    assert calls == []
    assert not output.exists()
