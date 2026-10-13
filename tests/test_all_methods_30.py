"""Artificial input and export checks for the fixed 30-variant driver."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_all_methods_30 as study


def test_fixed_selection_and_shared_method_contract():
    names, entries, selection_hash, source_hash = study.read_registry()
    assert len(names) == len(set(names)) == 30
    assert {"ecoli4", "yeast4", "ecoli_0_1_4_6_vs_5"} <= set(names)
    assert "vowel0" not in names
    assert all(name in entries for name in names)
    assert len(selection_hash) == len(source_hash) == 64
    assert study.METHODS == (
        "nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo", "sshr")
    assert study.ABLATION_CONFIG.use_unlabeled is False
    assert study.CONFIG.use_unlabeled is True


def _artificial_source(root: Path) -> dict:
    raw = root / "data" / "raw" / "artificial.csv"
    raw.parent.mkdir(parents=True)
    frame = pd.DataFrame({"x": np.arange(80, dtype=float),
                          "z": np.arange(80, dtype=float) ** 2,
                          "Class": ["negative"] * 60 + ["positive"] * 20})
    frame.to_csv(raw, index=False)
    return {"dataset_id": "artificial", "relative_path": "data/raw/artificial.csv",
            "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "target_column": "Class", "positive_label": "positive",
            "categorical_features": [], "missing_feature_cells": 0,
            "feature_names": ["x", "z"], "rows": 80, "input_features": 2,
            "majority_count": 60, "minority_count": 20,
            "fidelity": "artificial_test_fixture", "paper_dataset_name": "artificial"}


def test_preflight_freezes_fitting_only_folds_and_masks(tmp_path):
    entry = _artificial_source(tmp_path)
    before = (tmp_path / entry["relative_path"]).read_bytes()
    X, y, provenance, folds, masks, summary = study.prepare_dataset(
        "artificial", entry, root=tmp_path,
        selection_hash="selection", source_manifest_hash="source")
    assert len(X) == len(y) == 80
    assert summary["folds"] == 25
    assert summary["min_test_minority"] > 0
    assert len(folds) == len(masks) == 25
    assert provenance["feature_types"] == {"x": "numeric", "z": "numeric"}
    assert provenance["frozen_design"]["folds_sha256"] == study.content_hash(folds)
    for fold, mask in zip(folds, masks):
        assert set(fold["train_indices"]).isdisjoint(fold["test_indices"])
        assert set(mask["labeled_indices"]) | set(mask["unlabeled_indices"]) == set(fold["train_indices"])
        assert set(mask["unlabeled_indices"]).isdisjoint(fold["test_indices"])
    assert (tmp_path / entry["relative_path"]).read_bytes() == before


def test_preflight_rejects_missing_and_modified_sources(tmp_path):
    entry = _artificial_source(tmp_path)
    raw = tmp_path / entry["relative_path"]
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        study.prepare_dataset("artificial", entry, root=tmp_path,
                              selection_hash="selection", source_manifest_hash="source")
    raw.unlink()
    with pytest.raises(FileNotFoundError, match="Missing artificial"):
        study.prepare_dataset("artificial", entry, root=tmp_path,
                              selection_hash="selection", source_manifest_hash="source")


def test_combined_scores_keeps_only_shared_fold_scope(tmp_path):
    report = tmp_path / "ecoli4" / "reports"
    report.mkdir(parents=True)
    with (report / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "dataset_id", "run_id", "classifier", "method", "scope", "metric",
            "mean", "sample_sd", "n_folds", "fold_ids", "n_paired_with_nores",
            "nores_mean", "paired_mean_difference"))
        writer.writeheader()
        for scope in ("all_methods_shared_success", "all_method_success"):
            writer.writerow({"dataset_id": "ecoli4", "run_id": "test", "classifier": "LogReg",
                             "method": "nores", "scope": scope, "metric": "f1", "mean": "0.5",
                             "sample_sd": "0.1", "n_folds": "5", "fold_ids": "[]",
                             "n_paired_with_nores": "5", "nores_mean": "0.5",
                             "paired_mean_difference": "0"})
    path = study.export_combined_scores(tmp_path, ["ecoli4", "yeast4"])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["scope"] == "all_methods_shared_success"
    assert rows[0]["dataset_id"] == "ecoli4"
