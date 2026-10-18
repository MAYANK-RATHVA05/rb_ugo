"""Run an explicitly bounded, opt-in SPSDO-component geometry diagnostic.

Subgroup labels and analytic support predicates are evaluation information.
Neither prototype fitting, generation, tuning nor the learner receives them.
Raw individual records stay in an ignored output directory; export_aggregates
creates descriptive aggregate evidence separately.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.ppeo_reference import allocate_anchor_quotas, fit_phhc, generate_large_smote


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("spsdo", "tuned_sphere", "ellipsoid", "screened_ellipsoid",
           "symmetric_clip", "paired_rejection", "ros", "nores")
METRICS = ("rare_macro_recall", "rare_1_recall", "rare_2_recall", "fpr", "ap",
           "overall_recall", "generation_seconds", "fit_seconds")
LEARNER_DEFAULTS = {"max_iter": 60, "max_leaf_nodes": 7, "min_samples_leaf": 8,
                    "learning_rate": .1, "l2_regularization": 1., "random_state": 42}


def _native(value):
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value):
    path.write_text(json.dumps(_native(value), indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + repr(array.shape).encode()
                          + array.tobytes()).hexdigest()


@dataclass(frozen=True)
class Ellipse:
    center: np.ndarray
    transform: np.ndarray
    label: int
    subgroup: int
    name: str


def make_components(elongation: float, angle_deg: float, offset: float) -> list[Ellipse]:
    """Compact true supports, with pairwise overlap controlled in disk coordinates."""
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    transform = rotation @ np.diag([.8 * np.sqrt(elongation), .8 / np.sqrt(elongation)])
    components = [Ellipse(np.array([-6., 0.]), np.eye(2) * .8, 1, 0, "minority_main"),
                  Ellipse(np.array([-2., 0.]), np.eye(2) * 1.2, 0, -1, "majority_main")]
    for group, y in enumerate((-5., 5.), start=1):
        center = np.array([0., y])
        components.append(Ellipse(center, transform, 1, group, f"minority_rare_{group}"))
        for sign in (-1, 1):
            components.append(Ellipse(center + transform @ np.array([sign * offset, 0.]),
                                      transform, 0, -1, f"majority_rare_{group}_{sign:+d}"))
    return components


def disk_overlap_fraction(offset: float) -> float:
    """Area of intersection of two unit disks, divided by pi."""
    if offset >= 2:
        return 0.0
    if offset <= 0:
        return 1.0
    return float((2 * np.arccos(offset / 2)
                  - .5 * offset * np.sqrt(4 - offset ** 2)) / np.pi)


def support_membership(X: np.ndarray, components: list[Ellipse], label: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    membership = np.zeros(len(X), dtype=bool)
    for component in components:
        if component.label == label:
            normalized = np.linalg.solve(component.transform, (X - component.center).T).T
            membership |= np.sum(normalized ** 2, axis=1) <= 1 + 1e-12
    return membership


def support_intersections(components: list[Ellipse]) -> list[dict]:
    """Numerically check separation by maximizing the convex support gap.

    Positive maximum means a separating axis exists. Nonpositive means overlap
    or tangency for these convex ellipses. This numerical check is evaluator-only;
    it is not an exact-arithmetic proof or a general ellipse-intersection solver.
    """
    records = []
    grid = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    directions = np.column_stack([np.cos(grid), np.sin(grid)])
    for a, b in itertools.combinations(components, 2):
        delta = b.center - a.center
        values = (directions @ delta - np.linalg.norm(directions @ a.transform, axis=1)
                  - np.linalg.norm(directions @ b.transform, axis=1))
        index = int(np.argmax(values))
        def negative_gap(theta):
            direction = np.array([np.cos(theta), np.sin(theta)])
            return -(direction @ delta - np.linalg.norm(direction @ a.transform)
                     - np.linalg.norm(direction @ b.transform))
        step = 2 * np.pi / len(grid)
        result = minimize_scalar(negative_gap, bounds=(grid[index] - step, grid[index] + step),
                                 method="bounded", options={"xatol": 1e-12})
        gap = float(-result.fun)
        if gap <= 1e-8:
            records.append({"first": a.name, "second": b.name,
                            "maximum_separating_gap": gap,
                            "different_labels": a.label != b.label})
    return records


def make_partition(components: list[Ellipse], multiplier: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent uniform-area draws; group/component proportions remain fixed."""
    points, labels, groups = [], [], []
    for component in components:
        if component.name == "minority_main":
            base_count = 80
        elif component.name == "majority_main":
            base_count = 240
        elif component.label == 1:
            base_count = 8
        else:
            base_count = 60
        count = base_count * multiplier
        angles = rng.uniform(0, 2 * np.pi, count)
        radii = np.sqrt(rng.random(count))
        disk = radii[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
        points.append(disk @ component.transform.T + component.center)
        labels.extend([component.label] * count)
        groups.extend([component.subgroup] * count)
    return np.vstack(points), np.asarray(labels, dtype=int), np.asarray(groups, dtype=int)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Independent grouped-tie, ranked PR step area (not trapezoidal PR AUC)."""
    y, scores = _score_inputs(y, scores)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(y)]
    cumulative_tp = np.cumsum(sorted_y)[ends - 1]
    increments = np.diff(np.r_[0, cumulative_tp])
    return float(np.sum(increments * cumulative_tp / ends) / positives)


def _score_inputs(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if (y.ndim != 1 or scores.shape != y.shape or not len(y)
            or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all()):
        raise ValueError("Scores require finite, nonempty aligned binary observations.")
    return y, scores


def choose_threshold(y: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    """Lowest score threshold meeting validation FPR; positives use score >= t."""
    y, scores = _score_inputs(y, scores)
    negatives = np.sum(y == 0)
    if negatives == 0 or not 0 <= fpr_target < 1:
        raise ValueError("Threshold selection needs negatives and 0 <= FPR target < 1.")
    candidates = np.r_[np.unique(scores), np.nextafter(np.max(scores), np.inf)]
    # Grouped cumulative counts avoid a quadratic threshold-by-row broadcast.
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    suffix_fp = np.r_[np.cumsum((y[order] == 0)[::-1])[::-1], 0]
    starts = np.searchsorted(sorted_scores, candidates, side="left")
    acceptable = suffix_fp[starts] / negatives <= fpr_target + 1e-15
    return float(candidates[np.flatnonzero(acceptable)[0]])


def evaluate_scores(y, scores, groups, threshold: float) -> dict:
    y, scores = _score_inputs(y, scores)
    groups = np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError("Evaluator-only subgroup IDs must align with rows.")
    predictions = scores >= threshold
    positive, negative = y == 1, y == 0
    if not positive.any() or not negative.any():
        raise ValueError("Held-out metrics require both classes.")
    rare = []
    for group in (1, 2):
        subset = positive & (groups == group)
        if not subset.any():
            raise ValueError("Each true rare subgroup must occur in evaluation data.")
        rare.append(float(np.mean(predictions[subset])))
    return {"rare_1_recall": rare[0], "rare_2_recall": rare[1],
            "rare_macro_recall": float(np.mean(rare)),
            "fpr": float(np.mean(predictions[negative])),
            "ap": average_precision(y, scores),
            "overall_recall": float(np.mean(predictions[positive])),
            "tp": int(np.sum(predictions & positive)),
            "fp": int(np.sum(predictions & negative)),
            "tn": int(np.sum(~predictions & negative)),
            "fn": int(np.sum(~predictions & positive))}


def support_metrics(X: np.ndarray, components: list[Ellipse]) -> dict:
    minority = support_membership(X, components, 1)
    majority = support_membership(X, components, 0)
    n = len(X)
    return {"rows": n, "outside_minority_count": int(np.sum(~minority)),
            "majority_support_count": int(np.sum(majority)),
            "majority_only_count": int(np.sum(majority & ~minority)),
            "outside_minority_fraction": float(np.mean(~minority)) if n else None,
            "majority_support_fraction": float(np.mean(majority)) if n else None,
            "majority_only_fraction": float(np.mean(majority & ~minority)) if n else None}


def fit_budget(block_count: int, method_count: int, sphere_scales: list) -> dict:
    """Selected sphere candidate is reused, so its final result adds no refit."""
    final = block_count * method_count
    actual = final + block_count * (len(sphere_scales) - 1)
    if final > 100:
        raise ValueError("Diagnostic exceeds its 100 final-classifier-fit limit.")
    return {"final_models": final, "total_fits_including_selection": actual}


def validate_generation(samples, anchor_indices, X_minority, X_majority, method, eta,
                        sphere_scale=1.) -> dict:
    """Check realized generation using originals only, before any classifier fit."""
    info = {"checked_rows": len(samples), "paired_mean_checked": False,
            "symmetric_constraints_checked": False, "conflicting_anchor_rows": 0,
            "maximum_anchor_mean_error": None, "minimum_constraint_slack": None,
            "strict_original_majority_certificate_rows": 0}
    if not len(samples):
        return info
    for anchor_index in np.unique(anchor_indices):
        anchor = X_minority[anchor_index]
        points = samples[anchor_indices == anchor_index]
        displacement = points - anchor
        differences = X_majority - anchor
        squared = np.sum(differences ** 2, axis=1)
        conflict = bool(np.any(squared == 0))
        if conflict:
            info["conflicting_anchor_rows"] += len(points)
            np.testing.assert_array_equal(points, np.broadcast_to(anchor, points.shape))
        if method in {"symmetric_clip", "paired_rejection"}:
            np.testing.assert_allclose(points.mean(axis=0), anchor, atol=1e-12, rtol=1e-12)
            error = float(np.max(np.abs(points.mean(axis=0) - anchor)))
            info["maximum_anchor_mean_error"] = max(info["maximum_anchor_mean_error"] or 0., error)
            slack = eta * squared[None, :] - 2 * np.abs(displacement @ differences.T)
            minimum_slack = float(np.min(slack))
            previous = info["minimum_constraint_slack"]
            info["minimum_constraint_slack"] = minimum_slack if previous is None else min(previous, minimum_slack)
            if np.min(slack) < -1e-10:
                raise AssertionError("Generated pair violates original-majority slab constraints.")
            if not conflict:
                # Squared distance to each majority minus distance to anchor is positive.
                gap = squared[None, :] - 2 * displacement @ differences.T
                if np.min(gap) <= 0:
                    raise AssertionError("Strict observed-majority anchor certificate failed.")
                info["strict_original_majority_certificate_rows"] += len(points)
            info["paired_mean_checked"] = True
            info["symmetric_constraints_checked"] = True
        if method in {"spsdo", "tuned_sphere"}:
            radius = float(np.sqrt(np.min(squared)))
            scale = sphere_scale if method == "tuned_sphere" else 1.
            if np.any(np.linalg.norm(displacement, axis=1) > radius * scale + 1e-12):
                raise AssertionError("Paper-specified radial bound was exceeded.")
    return info


def runtime_identity() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "platform": platform.platform()}


def verify_identity(config_path: Path, identity_path: Path) -> dict:
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    config_hash = identity.get("protocol_sha256", identity.get("config_sha256"))
    if config_hash != _sha(config_path):
        raise ValueError("Frozen protocol hash differs; no classifier fits were started.")
    sources = identity.get("source_sha256", identity.get("sources", {}))
    if not sources:
        raise ValueError("Frozen source hashes are required before classifier fitting.")
    for name, expected in sources.items():
        if isinstance(expected, dict):
            expected = expected["sha256"]
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"Frozen source identity differs: {name}")
    for required in ("src/ppeo_reference.py", "src/ppeo_geometry.py",
                     "scripts/diagnose_ppeo_geometry.py"):
        if required not in sources and str(ROOT / required) not in sources:
            raise ValueError(f"Missing frozen implementation source: {required}")
    expected_runtime = identity.get("runtime", {})
    for key, value in expected_runtime.items():
        if key in runtime_identity() and runtime_identity()[key] != value:
            raise ValueError(f"Frozen runtime differs: {key}")
    return identity


def _blocks(config: dict) -> list[dict]:
    if "blocks" in config:
        return config["blocks"]
    return [dict(elongation=e, angle_deg=a, offset=s) for e, a, s in itertools.product(
        config.get("elongations", [1, 6]), config.get("angles_deg", [0, 60]),
        config.get("offsets", [2.4, 1.2]))]


def _validate_config(config):
    methods = config.get("methods", list(METHODS))
    if methods != list(METHODS):
        raise ValueError("All eight prespecified methods in fixed order are required.")
    scales = config.get("sphere_scales", [.25, .5, .75])
    if scales != [.25, .5, .75]:
        raise ValueError("Only the frozen three-candidate smaller-sphere search is supported.")
    budget = fit_budget(len(_blocks(config)), len(methods), scales)
    if budget["total_fits_including_selection"] > 100:
        raise ValueError("This bounded implementation permits at most 100 total fits.")
    expected_counts = {"fit": {"main_minority": 80, "rare_minority_each": 8,
                               "main_majority": 240, "rare_majority_each_side": 60}}
    for partition, multiplier in (("validation", 5), ("test", 50)):
        expected_counts[partition] = {key: count * multiplier
                                      for key, count in expected_counts["fit"].items()}
    if config.get("sample_counts") != expected_counts:
        raise ValueError("Configured counts differ from the declared, proportion-matched synthetic law.")
    return methods, scales, budget


def check_allowed_intersections(components: list[Ellipse], offset: float) -> list[dict]:
    intersections = support_intersections(components)
    for pair in intersections:
        if not pair["different_labels"]:
            continue
        first, second = pair["first"], pair["second"]
        allowed = (offset < 2 and first.startswith("minority_rare_")
                   and second.startswith("majority_rare_")
                   and first.split("_")[2] == second.split("_")[2])
        if not allowed:
            raise ValueError(f"Unplanned cross-class support intersection: {first}, {second}")
    return intersections


def _ensure_ignored_output(output: Path):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Individual records must be in this repository's ignored results directory.")
    relative = output.relative_to(ROOT)
    if relative.parts[0] != "results":
        raise ValueError("Individual records must be saved below ignored results/.")
    check = subprocess.run(["git", "check-ignore", "--quiet", str(relative / "record.json")],
                           cwd=ROOT, capture_output=True)
    if check.returncode != 0:
        raise ValueError("Output directory is not Git-ignored.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite existing individual diagnostic records.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run(config_path: Path, identity_path: Path, output: Path,
        *, learner_factory=HistGradientBoostingClassifier) -> dict:
    """Execute once only after matching frozen protocol/source/runtime identities."""
    identity = verify_identity(config_path, identity_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    methods, scales, budget = _validate_config(config)
    output = _ensure_ignored_output(output)
    from src.ppeo_geometry import generate
    records, block_records, candidate_records = [], [], []
    _write_json(output / "identity.json", identity)
    _write_json(output / "protocol.json", config)
    base_seed = config.get("base_seed", 42)
    eta, max_attempts = config.get("eta", .9), config.get("max_attempts", 64)
    fpr_target = config.get("fpr_target", .05)
    learner_config = config.get("learner", LEARNER_DEFAULTS)
    if "parameters" in learner_config:
        learner_config = learner_config["parameters"]
    learner_config = {k: v for k, v in learner_config.items() if k in LEARNER_DEFAULTS}
    learner_config = dict(LEARNER_DEFAULTS, **learner_config)
    actual_fits = 0
    seed_children = np.random.SeedSequence(base_seed).spawn(len(_blocks(config)))
    with threadpool_limits(limits=1):
        for block_index, (block, block_seed) in enumerate(zip(_blocks(config), seed_children)):
            print(f"Starting condition {block_index + 1}/{len(seed_children)}: {block}", flush=True)
            streams = block_seed.spawn(7)
            components = make_components(block["elongation"], block["angle_deg"],
                                         block.get("offset", block.get("majority_offset")))
            intersections = check_allowed_intersections(
                components, block.get("offset", block.get("majority_offset")))
            fitting_X, fitting_y, fitting_groups = make_partition(
                components, 1, np.random.default_rng(streams[0]))
            validation_X, validation_y, validation_groups = make_partition(
                components, 5, np.random.default_rng(streams[1]))
            test_X, test_y, test_groups = make_partition(
                components, 50, np.random.default_rng(streams[2]))
            scaler = StandardScaler().fit(fitting_X)
            X_fit, X_validation, X_test = map(scaler.transform,
                                              (fitting_X, validation_X, test_X))
            X_minority, X_majority = X_fit[fitting_y == 1], X_fit[fitting_y == 0]
            prototypes = fit_phhc(X_fit, fitting_y)
            allocation_seed, common_seed, geometry_seed = [
                int(streams[k].generate_state(1)[0]) for k in (3, 4, 5)]
            gap = len(X_majority) - len(X_minority)
            quotas = allocate_anchor_quotas(prototypes.prototype_ids, gap, seed=allocation_seed)
            small_anchor_mask = prototypes.small_flags[prototypes.prototype_ids]
            small_quotas = np.where(small_anchor_mask, quotas, 0)
            common_started = perf_counter()
            common_samples, common_anchors, common_diagnostics = generate_large_smote(
                X_minority, prototypes.prototype_ids, quotas, prototypes.small_flags,
                seed=common_seed)
            common_seconds = perf_counter() - common_started
            small_groups = fitting_groups[fitting_y == 1][small_anchor_mask]
            block_info = {"block_index": block_index, "condition": block,
                          "prototype_diagnostics": prototypes.diagnostics,
                          "quotas_sha256": array_sha(quotas),
                          "prototype_ids_sha256": array_sha(prototypes.prototype_ids),
                          "data_sha256": {name: array_sha(value) for name, value in (
                              ("fitting_X", fitting_X), ("fitting_y", fitting_y),
                              ("validation_X", validation_X), ("validation_y", validation_y),
                              ("test_X", test_X), ("test_y", test_y))},
                          "small_prototype_count": int(np.sum(prototypes.small_flags)),
                          "small_anchor_count": int(np.sum(small_anchor_mask)),
                          "small_generated_quota": int(np.sum(small_quotas)),
                          "rare_small_anchor_coverage": {
                              str(group): int(np.sum(small_groups == group)) for group in (1, 2)},
                          "rare_small_generated_quota": {
                              str(group): int(np.sum(small_quotas[
                                  fitting_groups[fitting_y == 1] == group])) for group in (1, 2)},
                          "original_majority_count": len(X_majority),
                          "original_minority_count": len(X_minority),
                          "seeds": {"allocation": allocation_seed, "common_large": common_seed,
                                    "geometry": geometry_seed},
                          "support_intersections": intersections,
                          "support_intersection_check": "numerical support-function separation check",
                          "intended_disk_overlap_fraction": disk_overlap_fraction(
                              block.get("offset", block.get("majority_offset")))}
            block_records.append(block_info)
            _write_json(output / f"block-{block_index:02d}.json", block_info)

            def build(method, scale=1.):
                start = perf_counter()
                if method == "nores":
                    small = np.empty((0, X_fit.shape[1]))
                    samples, anchor_indices = small, np.empty(0, dtype=int)
                    diagnostics = {}
                elif method == "ros":
                    result = generate(X_minority, X_majority, prototypes.prototype_ids,
                                      quotas, method="ros", seed=geometry_seed, eta=eta,
                                      max_attempts=max_attempts)
                    samples, anchor_indices = result.samples, result.anchor_indices
                    small = samples[small_anchor_mask[anchor_indices]]
                    diagnostics = result.diagnostics
                else:
                    result = generate(X_minority, X_majority, prototypes.prototype_ids,
                                      small_quotas, method=method, seed=geometry_seed,
                                      eta=eta, sphere_scale=scale, max_attempts=max_attempts)
                    small = result.samples
                    samples = np.vstack([common_samples, small])
                    anchor_indices = np.r_[common_anchors, result.anchor_indices]
                    diagnostics = result.diagnostics
                generated_seconds = perf_counter() - start
                if method not in {"ros", "nores"}:
                    generated_seconds += common_seconds
                if not np.isfinite(samples).all():
                    raise ValueError("Nonfinite generated observations.")
                if method != "nores":
                    if len(samples) != gap or not np.array_equal(
                            np.bincount(anchor_indices, minlength=len(quotas)), quotas):
                        raise RuntimeError("Matched original anchor quotas were not preserved.")
                if method not in {"ros", "nores"}:
                    diagnostics["realized_invariants"] = validate_generation(
                        result.samples, result.anchor_indices, X_minority, X_majority,
                        method, eta, scale)
                return samples, small, diagnostics, generated_seconds

            def fit_one(method, scale=1.):
                nonlocal actual_fits
                samples, small, diagnostics, generation_seconds = build(method, scale)
                augmented_X = np.vstack([X_fit, samples])
                augmented_y = np.r_[fitting_y, np.ones(len(samples), dtype=int)]
                model = learner_factory(**learner_config)
                if actual_fits >= budget["total_fits_including_selection"]:
                    raise RuntimeError("Frozen classifier fit budget would be exceeded.")
                start = perf_counter()
                model.fit(augmented_X, augmented_y)
                fit_seconds = perf_counter() - start
                actual_fits += 1
                if actual_fits > budget["total_fits_including_selection"]:
                    raise RuntimeError("Frozen classifier fit budget exceeded.")
                validation_scores = model.predict_proba(X_validation)[:, 1]
                threshold = choose_threshold(validation_y, validation_scores, fpr_target)
                # Selection uses only validation overall recall/AP; groups are NOT accessed.
                validation_predictions = validation_scores >= threshold
                validation_selection = {
                    "overall_recall": float(np.mean(validation_predictions[validation_y == 1])),
                    "ap": average_precision(validation_y, validation_scores),
                    "fpr": float(np.mean(validation_predictions[validation_y == 0]))}
                selection = {"method": method, "scale": scale, "threshold": threshold,
                             "validation": validation_selection, "fit_seconds": fit_seconds,
                             "generation_seconds": generation_seconds}
                return (model, samples, small, diagnostics, selection)

            for method in methods:
                try:
                    start = perf_counter()
                    if method == "tuned_sphere":
                        candidates = [fit_one(method, scale) for scale in scales]
                        candidate_records.extend(dict(block_index=block_index, **row[4])
                                                 for row in candidates)
                        selected = max(candidates, key=lambda row: (
                            row[4]["validation"]["overall_recall"], row[4]["validation"]["ap"],
                            row[4]["scale"]))
                        selection_total_fit_seconds = sum(row[4]["fit_seconds"] for row in candidates)
                        selection_total_generation_seconds = sum(row[4]["generation_seconds"]
                                                                 for row in candidates)
                    else:
                        selected = fit_one(method)
                        selection_total_fit_seconds = selected[4]["fit_seconds"]
                        selection_total_generation_seconds = selected[4]["generation_seconds"]
                    model, samples, small, diagnostics, selection = selected
                    scores = model.predict_proba(X_test)[:, 1]
                    metrics = evaluate_scores(test_y, scores, test_groups, selection["threshold"])
                    metrics.update(generation_seconds=selection["generation_seconds"],
                                   fit_seconds=selection["fit_seconds"])
                    record = {"block_index": block_index, "condition": block, "method": method,
                              "status": "success", "metrics": metrics,
                              "threshold": selection["threshold"],
                              "selected_sphere_scale": selection["scale"] if method == "tuned_sphere" else None,
                              "validation_selection_metrics": selection["validation"],
                              "generated_count": len(samples), "small_generated_count": len(small),
                              "generation_diagnostics": diagnostics,
                              "common_large_diagnostics": common_diagnostics if method not in {"ros", "nores"} else {},
                              "support_all": support_metrics(scaler.inverse_transform(samples), components)
                                  if len(samples) else support_metrics(np.empty((0, 2)), components),
                              "support_small": support_metrics(scaler.inverse_transform(small), components)
                                  if len(small) else support_metrics(np.empty((0, 2)), components),
                              "quotas_sha256": block_info["quotas_sha256"],
                              "prototype_ids_sha256": block_info["prototype_ids_sha256"],
                              "selection_total_fit_seconds": selection_total_fit_seconds,
                              "selection_total_generation_seconds": selection_total_generation_seconds,
                              "total_method_seconds": perf_counter() - start}
                    records.append(record)
                    _write_json(output / f"record-{block_index:02d}-{method}.json", record)
                    print(f"  {method}: complete ({actual_fits} actual fits)", flush=True)
                except Exception as exc:
                    failed = {"block_index": block_index, "condition": block, "method": method,
                              "status": "failed", "error_type": type(exc).__name__,
                              "error": str(exc), "actual_fits_so_far": actual_fits,
                              "quotas_sha256": block_info["quotas_sha256"]}
                    _write_json(output / f"record-{block_index:02d}-{method}.json", failed)
                    raise
    if actual_fits != budget["total_fits_including_selection"]:
        raise RuntimeError("Classifier-fit accounting disagrees with the frozen budget.")
    summary = {"purpose": "paper-specified SPSDO component diagnostic; not full PPEO reproduction",
               "runtime": runtime_identity(), "fit_budget": budget, "actual_fits": actual_fits,
               "final_records": len(records), "records": records,
               "blocks": block_records, "sphere_candidates": candidate_records,
               "identity_sha256": _sha(identity_path), "protocol_sha256": _sha(config_path)}
    _write_json(output / "individual-results.json", summary)
    return summary


def aggregate_results(summary: dict) -> dict:
    """Descriptive aggregates and same-condition deltas; no fold-level records."""
    records = summary["records"]
    indexed = {(r["block_index"], r["method"]): r for r in records}
    blocks = sorted({r["block_index"] for r in records})
    result = {"purpose": summary["purpose"], "fit_budget": summary["fit_budget"],
              "actual_fits": summary["actual_fits"], "condition_blocks": len(blocks),
              "protocol_sha256": summary["protocol_sha256"],
              "identity_sha256": summary["identity_sha256"], "methods": {}, "paired": {},
              "conditions": [], "condition_method_aggregates": []}
    for block in summary["blocks"]:
        result["conditions"].append({key: block[key] for key in (
            "block_index", "condition", "small_prototype_count", "small_anchor_count",
            "small_generated_quota", "rare_small_anchor_coverage", "rare_small_generated_quota",
            "support_intersections", "support_intersection_check", "intended_disk_overlap_fraction")})
        result["conditions"][-1].update({key: block["prototype_diagnostics"][key] for key in (
            "prototype_sizes", "prototype_small_flags", "f1_history", "stop_reason")})
    for record in records:
        row = {key: record[key] for key in (
            "block_index", "condition", "method", "metrics", "threshold",
            "selected_sphere_scale", "validation_selection_metrics", "generated_count",
            "small_generated_count", "support_all", "support_small",
            "selection_total_fit_seconds", "selection_total_generation_seconds")}
        diagnostics = record["generation_diagnostics"]
        row["generation"] = {key: diagnostics.get(key, 0) for key in (
            "fallback_generated_rows", "duplication_rows", "odd_copy_rows", "attempts", "clip_count")}
        row["generation"]["fallback_counts"] = diagnostics.get("fallback_counts", {})
        scales = diagnostics.get("clip_scales", [])
        row["generation"]["clip_scales"] = {"count": len(scales),
                                               "mean": float(np.mean(scales)) if scales else None,
                                               "minimum": float(np.min(scales)) if scales else None,
                                               "maximum": float(np.max(scales)) if scales else None}
        row["generation"]["realized_invariants"] = diagnostics.get("realized_invariants", {})
        result["condition_method_aggregates"].append(row)
    for method in METHODS:
        subset = [indexed[(block, method)] for block in blocks]
        metrics = {metric: {"mean": float(np.mean([r["metrics"][metric] for r in subset])),
                            "minimum": float(np.min([r["metrics"][metric] for r in subset])),
                            "maximum": float(np.max([r["metrics"][metric] for r in subset]))}
                   for metric in METRICS}
        for scope in ("support_all", "support_small"):
            metrics[scope] = {}
            for metric in ("outside_minority_fraction", "majority_support_fraction", "majority_only_fraction"):
                values = [r[scope][metric] for r in subset if r[scope][metric] is not None]
                metrics[scope][metric] = float(np.mean(values)) if values else None
        metrics["selection_total_fit_seconds"] = float(np.sum(
            [r["selection_total_fit_seconds"] for r in subset]))
        result["methods"][method] = metrics
    for comparator in METHODS:
        if comparator == "symmetric_clip":
            continue
        contrasts = {}
        for metric in METRICS:
            differences = np.array([indexed[(block, "symmetric_clip")]["metrics"][metric]
                                    - indexed[(block, comparator)]["metrics"][metric]
                                    for block in blocks])
            contrasts[metric] = {"mean_difference": float(differences.mean()),
                                 "sample_sd": float(differences.std(ddof=1)) if len(blocks) > 1 else None,
                                 "positive_blocks": int(np.sum(differences > 1e-12)),
                                 "negative_blocks": int(np.sum(differences < -1e-12)),
                                 "tied_blocks": int(np.sum(np.abs(differences) <= 1e-12))}
        result["paired"][f"symmetric_clip_minus_{comparator}"] = contrasts
    return result


def export_aggregates(summary: dict, destination: Path):
    if destination.exists():
        raise FileExistsError("Refusing to overwrite aggregate evidence.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, aggregate_results(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-aggregate", type=Path)
    args = parser.parse_args()
    summary = run(args.config, args.identity, args.output)
    if args.export_aggregate:
        export_aggregates(summary, args.export_aggregate)


if __name__ == "__main__":
    main()
