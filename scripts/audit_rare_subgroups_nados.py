"""Read-only audit and compact publication of the completed NADOS diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src.results import content_hash
from src.rare_subgroups_protocol import array_hash


ROOT = Path(__file__).resolve().parents[1]
DESIGN_SHA256 = "c86301b697708983c3884256a0e055bde57f60ee3f7d8f5df5e42b6da9693b9f"
PARENT_DESIGN_SHA256 = "adfdb2961963ff83b9552e84b06963689f938d853859d9bafc224106a08849db"
METRICS = ("rare_recall", "common_recall", "positive_recall", "test_fpr", "f1", "ap",
           "worst_group_recall")
CONTROLS = ("NORES", "WEIGHTED", "ROS", "SMOTE", "AW_REMOVE",
            "AW_SEED_FILTER_RETAIN", "AW_REMOVE_TUNED")


def _close(actual, expected, name="value") -> float:
    error = abs(float(actual) - float(expected))
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError((name, actual, expected, error))
    return error


def _ap(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    true_positives = np.cumsum(labels, dtype=np.int64)[ends]
    recall = true_positives / labels.sum()
    precision = true_positives / (ends + 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _regenerate(job: dict) -> dict:
    result = {}
    rare = job["rare_count"]
    counts = {
        "fitting": {"majority": 1000, "common": 100 - rare, "rare": rare},
        "selection": {"majority": 2000, "common": 180, "rare": 20},
        "calibration": {"majority": 2000, "common": 0, "rare": 0},
        "test": {"majority": 20000, "common": 1800, "rare": 200},
    }
    for partition, partition_id in (("fitting", 1), ("selection", 2),
                                    ("calibration", 3), ("test", 4)):
        features, groups = [], []
        for component, name in enumerate(("majority", "common", "rare")):
            count = counts[partition][name]
            pool = ((100 if component == 1 else 16)
                    if partition == "fitting" and component > 0 else count)
            draw = np.random.default_rng(
                np.random.SeedSequence([job["seed"], partition_id, component])
            ).standard_normal((pool, 2))[:count]
            if component == 0:
                mean, scale = (0.0, 0.0), 1.0
            elif component == 1:
                mean, scale = (3.0, 0.0), .35
            else:
                mean, scale = (0.0, 3.0), job["rare_sigma"]
            features.append(np.asarray(mean) + scale * draw)
            groups.append(np.full(count, component, dtype=np.int8))
        X, group = np.vstack(features), np.concatenate(groups)
        order = np.random.default_rng(
            np.random.SeedSequence([job["seed"], partition_id, 99])
        ).permutation(len(group))
        X, group = X[order], group[order]
        truth = (group > 0).astype(np.int8)
        part = {"X": X, "y": truth.copy(), "groups": group}
        if partition == "fitting":
            chosen = np.random.default_rng(
                np.random.SeedSequence([job["seed"], 5, 100])
            ).permutation(1000)[:job["flipped_count"]]
            flags = np.isin(order, chosen)
            part["y"][flags] = 1
            # The measured Windows run serialized this permutation as int32.
            # Preserve that recorded dtype for byte-level array fingerprints;
            # the values are identical to NumPy's platform-native intp order.
            part.update(true_y=truth, corrupted=flags, stable_ids=order.astype(np.int32))
        result[partition] = part
    return result


def _source_hash_matches(commit: str, name: str, expected: str) -> bool:
    blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
    normalized = blob.replace(b"\r\n", b"\n")
    variants = (normalized, normalized.replace(b"\n", b"\r\n"))
    return expected in {hashlib.sha256(value).hexdigest() for value in variants}


def _audit_nados(trace: dict[str, np.ndarray], X: np.ndarray, observed: np.ndarray,
                 row: dict, seed: int) -> int:
    minority = np.flatnonzero(observed == 1)
    np.testing.assert_array_equal(trace["minority_indices"], minority)
    neighbors = trace["neighbor_indices"].astype(int)
    distances = trace["neighbor_distances"]
    assert neighbors.shape == distances.shape == (len(X), 5)
    for index in minority:
        expected_distances = np.linalg.norm(X - X[index], axis=1)
        expected_distances[index] = np.inf
        expected_neighbors = np.argsort(expected_distances, kind="stable")[:5]
        np.testing.assert_array_equal(neighbors[index], expected_neighbors)
        np.testing.assert_array_equal(distances[index], expected_distances[expected_neighbors])

    local = neighbors[minority]
    local_distances = distances[minority]
    local_minority = observed[local] == 1
    counts = local_minority.sum(axis=1)
    support = counts / 5
    mean_minority = np.full(len(minority), np.nan)
    mean_majority = np.full(len(minority), np.nan)
    evidence = np.empty(len(minority))
    partners = []
    for position in range(len(minority)):
        minority_distances = local_distances[position, local_minority[position]]
        majority_distances = local_distances[position, ~local_minority[position]]
        partners.append(local[position, local_minority[position]])
        if not len(minority_distances):
            evidence[position] = 0.0
        elif not len(majority_distances):
            mean_minority[position] = minority_distances.mean()
            evidence[position] = 1.0
        else:
            mean_minority[position] = minority_distances.mean()
            mean_majority[position] = majority_distances.mean()
            evidence[position] = mean_majority[position] / (
                mean_minority[position] + mean_majority[position]
            )
    reliability = .5 * support + .5 * evidence
    boundary = 1 - np.abs(2 * support - 1)
    difficulty = .5 * boundary + .5 * (1 - support)
    eligible = (reliability >= .35) & (counts > 0)
    priority = np.where(eligible, reliability * difficulty, 0.0)
    probabilities = priority / priority.sum()
    expected = {
        "minority_neighbor_counts": counts,
        "minority_support": support,
        "mean_minority_distance": mean_minority,
        "mean_majority_distance": mean_majority,
        "relative_distance_evidence": evidence,
        "reliability_scores": reliability,
        "boundary_scores": boundary,
        "difficulty_scores": difficulty,
        "eligible_mask": eligible,
        "eligible_indices": minority[eligible],
        "excluded_indices": minority[~eligible],
        "synthesis_priorities": priority,
        "seed_probabilities": probabilities,
        "retained_indices": np.arange(len(X)),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(trace[name], values, rtol=1e-13, atol=1e-14,
                                   equal_nan=True, err_msg=name)

    generated = int(np.sum(observed == 0) - np.sum(observed == 1))
    rng = np.random.default_rng(seed)
    seeds, partners_drawn, fractions = [], [], []
    positions = np.arange(len(minority))
    for _ in range(generated):
        position = int(rng.choice(positions, p=probabilities))
        seeds.append(minority[position])
        partners_drawn.append(rng.choice(partners[position]))
        fractions.append(rng.random())
    seeds = np.asarray(seeds)
    partners_drawn = np.asarray(partners_drawn)
    fractions = np.asarray(fractions)
    np.testing.assert_array_equal(trace["generated_seed_indices"], seeds)
    np.testing.assert_array_equal(trace["generated_partner_indices"], partners_drawn)
    np.testing.assert_array_equal(trace["interpolation_fractions"], fractions)
    synthetic = X[seeds] + fractions[:, None] * (X[partners_drawn] - X[seeds])
    assert array_hash(np.vstack((X, synthetic))) == row["output_X_sha256"]
    output_y = np.r_[observed, np.ones(generated, dtype=observed.dtype)]
    assert array_hash(output_y) == row["output_y_sha256"]
    assert row["generated_count"] == generated
    return generated


def _audit_diagnostics(row: dict, fitting: dict, trace: dict) -> None:
    minority = trace["minority_indices"].astype(int)
    eligible = trace["eligible_mask"].astype(bool)
    masks = {
        "common": (fitting["groups"][minority] == 1) & ~fitting["corrupted"][minority],
        "rare": (fitting["groups"][minority] == 2) & ~fitting["corrupted"][minority],
        "noise": fitting["corrupted"][minority],
    }
    for name, mask in masks.items():
        total, accepted = int(mask.sum()), int(np.sum(mask & eligible))
        expected = {"original": total, "eligible": accepted, "excluded": total - accepted,
                    "fraction": accepted / total if total else None,
                    "mean_reliability": (float(np.mean(trace["reliability_scores"][mask]))
                                         if total else None),
                    "mean_difficulty": (float(np.mean(trace["difficulty_scores"][mask]))
                                        if total else None)}
        assert row["eligibility"][name] == expected
    seed_flags = fitting["corrupted"][trace["generated_seed_indices"].astype(int)]
    partner_flags = fitting["corrupted"][trace["generated_partner_indices"].astype(int)]
    any_corrupted = seed_flags | partner_flags
    expected_parent = {"kind": "interpolation", "generated": len(seed_flags),
                       "clean_clean": int(np.sum(~seed_flags & ~partner_flags)),
                       "clean_corrupt": int(np.sum(seed_flags != partner_flags)),
                       "corrupt_corrupt": int(np.sum(seed_flags & partner_flags)),
                       "any_corrupted_parent": int(any_corrupted.sum()),
                       "corrupted_parent_fraction": float(any_corrupted.mean())}
    assert row["parent_diagnostics"] == expected_parent


def _audit_metrics(row: dict, vectors) -> float:
    calibration_scores = vectors["calibration__NADOS"]
    calibration_ties = vectors["calibration_ties"]
    order = np.lexsort((calibration_ties, calibration_scores))
    index = order[-100]
    threshold, cutoff = float(calibration_scores[index]), float(calibration_ties[index])
    calibration_predictions = ((calibration_scores > threshold)
                               | ((calibration_scores == threshold) & (calibration_ties >= cutoff)))
    assert int(calibration_predictions.sum()) == 100
    scores, ties = vectors["test__NADOS"], vectors["test_ties"]
    predicted = (scores > threshold) | ((scores == threshold) & (ties >= cutoff))
    labels, groups = vectors["test_y"], vectors["test_groups"]
    fp, tp = int(predicted[labels == 0].sum()), int(predicted[labels == 1].sum())
    common, rare = float(predicted[groups == 1].mean()), float(predicted[groups == 2].mean())
    expected = {"threshold": threshold, "tie_cutoff": cutoff, "calibration_fp": 100,
                "calibration_negatives": 2000, "calibration_fpr": .05,
                "rare_recall": rare, "common_recall": common, "positive_recall": tp / 2000,
                "test_fpr": fp / 20000, "f1": 2 * tp / (tp + fp + 2000),
                "ap": _ap(labels, scores), "worst_group_recall": min(common, rare),
                "test_fp": fp, "test_negatives": 20000, "test_tp": tp,
                "test_positives": 2000, "rare_tp": int(predicted[groups == 2].sum()),
                "rare_count": 200, "common_tp": int(predicted[groups == 1].sum()),
                "common_count": 1800}
    errors = [_close(row["metrics"][name], value, name) for name, value in expected.items()]
    return max(errors)


def audit(run: Path) -> tuple[dict, list[dict]]:
    run = Path(run)
    envelope = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest = envelope["manifest"]
    assert envelope["run_id"] == content_hash(manifest)
    assert manifest["study"] == "rare_subgroups_nados_v1"
    assert manifest["design_sha256"] == DESIGN_SHA256
    assert manifest["parent_design_sha256"] == PARENT_DESIGN_SHA256
    assert manifest["config"]["execution_enabled"] is True
    assert manifest["git"]["tracked_changes"] is False
    commit = manifest["git"]["commit"]
    for name, digest in manifest["source_sha256"].items():
        assert _source_hash_matches(commit, name, digest), name
    assert _source_hash_matches(commit, "configs/rare_subgroups_nados_v1.json",
                                manifest["config_file_sha256"])
    assert len(manifest["jobs"]) == 90

    expected_jobs = {f"job-{content_hash(job)}.json" for job in manifest["jobs"]}
    assert {path.name for path in run.glob("job-*.json")} == expected_jobs
    records, max_metric_error, generated_total = [], 0.0, 0
    for job in manifest["jobs"]:
        payload = json.loads((run / f"job-{content_hash(job)}.json").read_text(encoding="utf-8"))
        record = payload["record"]
        assert payload["sha256"] == content_hash(record)
        assert record["run_id"] == envelope["run_id"] and record["job"] == job
        assert record["status"] == "success" and record["case_status"] == "complete"
        assert record["classifier_fit_attempts"] == 1 and len(record["rows"]) == 1
        row = record["rows"][0]
        assert row["status"] == "success" and row["method"] == "NADOS"
        artifact = run / record["predictions"]["path"]
        assert artifact.name == f"case-{content_hash(job)}.npz"
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record["predictions"]["sha256"]
        data = _regenerate(job)
        for partition, values in data.items():
            for field, array in values.items():
                assert array_hash(array) == record["partitions_sha256"][partition][field]
        fitting = data["fitting"]
        mean, scale = fitting["X"].mean(axis=0), fitting["X"].std(axis=0)
        np.testing.assert_allclose(record["scaler"]["mean"], mean, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(record["scaler"]["scale"], scale, rtol=1e-13, atol=1e-14)
        assert record["scaler"]["fitted_rows"] == 1100
        X = (fitting["X"] - mean) / scale
        assert array_hash(X) == record["scaled_fitting_X_sha256"]
        with np.load(artifact, allow_pickle=False) as vectors:
            assert set(vectors.files) == set(record["vector_sha256"])
            for name in vectors.files:
                assert array_hash(vectors[name]) == record["vector_sha256"][name]
            for field in ("y", "true_y", "groups", "corrupted", "stable_ids"):
                np.testing.assert_array_equal(vectors["fitting_" + field], fitting[field])
            trace = {name: vectors[key] for name, key in row["trace_vectors"].items()}
            generated_total += _audit_nados(trace, X, fitting["y"], row, job["seed"])
            _audit_diagnostics(row, fitting, trace)
            max_metric_error = max(max_metric_error, _audit_metrics(row, vectors))
        records.append(record)
    assert len(records) == 90
    audit_result = {
        "status": "passed",
        "run_commit": commit,
        "run_id": envelope["run_id"],
        "design_sha256": DESIGN_SHA256,
        "parent_design_sha256": PARENT_DESIGN_SHA256,
        "cases_verified": 90,
        "successful_rows_verified": 90,
        "classifier_fits_recorded": 90,
        "generated_rows_reconstructed": generated_total,
        "calibration_false_positives_per_case": 100,
        "maximum_metric_recomputation_error": max_metric_error,
        "source_line_endings": "manifest Windows CRLF hashes matched committed text after explicit normalization",
        "classifier_refits_during_audit": 0,
    }
    return audit_result, records


def _sample_sd(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def publish(records: list[dict], audit_result: dict, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    score_rows, eligibility_rows, exposure_rows = [], [], []
    for record in records:
        job, row = record["job"], record["rows"][0]
        base = {"case_id": record["case_id"], "fit_rare_count": job["rare_count"],
                "rare_sigma": job["rare_sigma"], "flipped_count": job["flipped_count"],
                "seed": job["seed"]}
        score_rows.append({**base, "method": "NADOS", "status": row["status"],
                           "generated_count": row["generated_count"], **row["metrics"]})
        for group, values in row["eligibility"].items():
            eligibility_rows.append({**base, "group": group, **values})
        exposure_rows.append({**base, **row["parent_diagnostics"]})
    scores, eligibility, exposure = map(pd.DataFrame, (score_rows, eligibility_rows, exposure_rows))
    scores.to_csv(output / "scores.csv", index=False)
    eligibility.to_csv(output / "eligibility.csv", index=False)
    exposure.to_csv(output / "parent_exposure.csv", index=False)

    group_keys = ["fit_rare_count", "rare_sigma", "flipped_count"]
    comparison = scores.groupby(group_keys)[list(METRICS)].agg(["mean", _sample_sd])
    comparison.columns = [f"{metric}_{'sample_sd' if stat == '_sample_sd' else stat}"
                          for metric, stat in comparison.columns]
    comparison.reset_index().to_csv(output / "comparison.csv", index=False)

    parent_scores = pd.read_csv(ROOT / "docs/research/rare_subgroups/contamination_v1/scores.csv")
    paired = []
    keys = ["case_id", *group_keys, "seed"]
    for method in CONTROLS:
        merged = scores.merge(parent_scores[parent_scores.method == method], on=keys,
                              suffixes=("_nados", "_control"), validate="one_to_one")
        item = {"control": method, "n_cases": len(merged)}
        for metric in ("rare_recall", "common_recall", "test_fpr", "f1", "ap"):
            delta = merged[f"{metric}_nados"] - merged[f"{metric}_control"]
            item[f"mean_NADOS_minus_control_{metric}"] = delta.mean()
        item.update(rare_recall_NADOS_wins=int((merged.rare_recall_nados > merged.rare_recall_control + 1e-12).sum()),
                    rare_recall_ties=int((abs(merged.rare_recall_nados - merged.rare_recall_control) <= 1e-12).sum()),
                    rare_recall_NADOS_losses=int((merged.rare_recall_nados < merged.rare_recall_control - 1e-12).sum()))
        paired.append(item)
    pd.DataFrame(paired).to_csv(output / "paired_control_comparison.csv", index=False)

    pivot = eligibility.pivot(index=[*group_keys, "seed"], columns="group", values="fraction").reset_index()
    pivot["eligibility_seed_pass"] = ((pivot.rare <= .5)
                                       & ((pivot.common - pivot.rare) >= .2 - 1e-12))
    eligibility_cells = pivot.groupby(group_keys).agg(
        eligibility_seed_passes=("eligibility_seed_pass", "sum"),
        rare_eligibility_mean=("rare", "mean"),
        common_eligibility_mean=("common", "mean"),
    ).reset_index()
    parent_exposure = pd.read_csv(
        ROOT / "docs/research/rare_subgroups/contamination_v1/parent_exposure.csv"
    )
    parent_exposure = parent_exposure[parent_exposure.method == "AW_SEED_FILTER_RETAIN"].copy()
    parent_exposure.rename(columns={"rare_count": "fit_rare_count"}, inplace=True)
    nados_exposure = exposure.groupby(group_keys).corrupted_parent_fraction.mean()
    aw_exposure = parent_exposure.groupby(group_keys).fraction_with_corrupted_parent.mean()

    decisions = []
    for cell, nados_group in scores[scores.flipped_count > 0].groupby(group_keys):
        fit_rare, sigma, noise = cell
        parent_cell = parent_scores[(parent_scores.fit_rare_count == fit_rare)
                                    & (parent_scores.rare_sigma == sigma)
                                    & (parent_scores.flipped_count == noise)]
        candidates = []
        for method in CONTROLS:
            control = parent_cell[parent_cell.method == method]
            joined = nados_group.merge(control, on=keys, suffixes=("_nados", "_control"),
                                       validate="one_to_one")
            candidates.append((method,
                               float((joined.rare_recall_control - joined.rare_recall_nados).mean()),
                               float(abs(joined.test_fpr_control - joined.test_fpr_nados).mean())))
        comparable = [item for item in candidates if item[2] <= .01 + 1e-12]
        best = max(comparable, key=lambda item: item[1])
        eligibility_row = eligibility_cells[
            (eligibility_cells.fit_rare_count == fit_rare)
            & (eligibility_cells.rare_sigma == sigma)
            & (eligibility_cells.flipped_count == noise)
        ].iloc[0]
        nados_fraction = float(nados_exposure.loc[cell])
        aw_fraction = float(aw_exposure.loc[cell])
        eligibility_gate = int(eligibility_row.eligibility_seed_passes) >= 4
        performance_gate = best[1] >= .05 - 1e-12
        exposure_gate = nados_fraction <= .5 * aw_fraction + 1e-12
        decisions.append({"fit_rare_count": fit_rare, "rare_sigma": sigma,
                          "flipped_count": noise,
                          "eligibility_seed_passes": int(eligibility_row.eligibility_seed_passes),
                          "rare_eligibility_mean": eligibility_row.rare_eligibility_mean,
                          "common_eligibility_mean": eligibility_row.common_eligibility_mean,
                          "best_comparable_control": best[0],
                          "control_minus_NADOS_rare_recall": best[1],
                          "mean_absolute_paired_test_fpr_difference": best[2],
                          "NADOS_corrupted_parent_fraction": nados_fraction,
                          "AW_retention_corrupted_parent_fraction": aw_fraction,
                          "eligibility_gate": eligibility_gate,
                          "performance_gate": performance_gate,
                          "exposure_gate": exposure_gate,
                          "qualifies": eligibility_gate and performance_gate and exposure_gate})
    decisions = pd.DataFrame(decisions)
    decisions.to_csv(output / "screen_decisions.csv", index=False)
    qualifying_bases = decisions[decisions.qualifies][["fit_rare_count", "rare_sigma"]].drop_duplicates()
    status = "diagnostic_trigger_not_met" if len(qualifying_bases) < 2 else "diagnostic_trigger_met"
    contaminated = exposure[exposure.flipped_count > 0]
    parent_contaminated = parent_exposure[parent_exposure.flipped_count > 0]
    nados_corrupted = int(contaminated.any_corrupted_parent.sum())
    nados_generated = int(contaminated.generated.sum())
    aw_corrupted = int((parent_contaminated.clean_corrupt.fillna(0)
                        + parent_contaminated.corrupt_corrupt.fillna(0)).sum())
    aw_generated = int(parent_contaminated.generated_count.sum())
    status_record = {
        "status": status,
        "cases": len(scores),
        "successful_rows": int((scores.status == "success").sum()),
        "qualifying_cells": int(decisions.qualifies.sum()),
        "qualifying_base_settings": len(qualifying_bases),
        "required_qualifying_base_settings": 2,
        "NADOS_contaminated_parent_exposure": nados_corrupted / nados_generated,
        "AW_retention_contaminated_parent_exposure": aw_corrupted / aw_generated,
        "relative_parent_exposure_reduction": 1 - ((nados_corrupted / nados_generated)
                                                   / (aw_corrupted / aw_generated)),
        "interpretation": "descriptive frozen screen; not statistical significance or novelty evidence",
    }
    (output / "status.json").write_text(json.dumps(status_record, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    provenance = {"run_commit": audit_result["run_commit"], "run_id": audit_result["run_id"],
                  "design_sha256": DESIGN_SHA256, "parent_design_sha256": PARENT_DESIGN_SHA256,
                  "source": "user-supplied complete raw result archive",
                  "raw_artifacts_committed": False,
                  "published_files_derived_without_classifier_refits": True}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    return status_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Complete rare-subgroups-nados-v1 result directory.")
    parser.add_argument("--audit-output", type=Path,
                        help="Optional path for the audit JSON.")
    parser.add_argument("--publish-dir", type=Path,
                        help="Optional directory for compact audited CSV/JSON outputs.")
    args = parser.parse_args(argv)
    audit_result, records = audit(args.input)
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    status = publish(records, audit_result, args.publish_dir) if args.publish_dir else None
    print(json.dumps({"audit": audit_result, "publication": status}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
