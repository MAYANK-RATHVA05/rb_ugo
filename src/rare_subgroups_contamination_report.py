"""Complete coverage, paired noise effects and the frozen descriptive screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.rare_subgroups_contamination import case_id, planned_jobs
from src.rare_subgroups_protocol import METRICS
from src.rare_subgroups_report import _atomic_csv
from src.results import _atomic_json


def _successful(rows) -> bool:
    return all(row is not None and row["status"] == "success" for row in rows)


def _mean(rows, metric):
    return float(np.mean([row["metrics"][metric] for row in rows]))


def _screen(lookup: dict, design: dict, complete: bool) -> dict:
    rules = design["screening_rules"]
    tolerance, strict = rules["inclusive_boundary_tolerance"], rules["strict_positive_cutoff"]
    protection, tradeoff = rules["rare_protection"], rules["noise_tradeoff"]
    settings, qualifying_bases = [], set()
    seeds = design["seeds"]
    distribution = design["distribution"]
    for count in distribution["fit_counts"]["rare_grid"]:
        for sigma in distribution["rare_minority"]["isotropic_sigma_grid"]:
            for noise in rules["contaminated_levels"]:
                rows = { (m, method): [lookup[count, sigma, m, seed, method] for seed in seeds]
                         for m in (0, noise) for method in ("AW_REMOVE", "AW_SEED_FILTER_RETAIN") }
                paired = all(_successful(group) for group in rows.values())
                assessment = {"rare_count": count, "rare_sigma": sigma, "flipped_count": noise,
                              "paired_complete": paired, "rare_protection": None,
                              "noise_tradeoff": None, "qualifies": None}
                if paired:
                    remove, retain = rows[noise, "AW_REMOVE"], rows[noise, "AW_SEED_FILTER_RETAIN"]
                    clean_remove, clean_retain = rows[0, "AW_REMOVE"], rows[0, "AW_SEED_FILTER_RETAIN"]
                    rare = np.array([k["metrics"]["rare_recall"] - d["metrics"]["rare_recall"] for d, k in zip(remove, retain)])
                    common = _mean(retain, "common_recall") - _mean(remove, "common_recall")
                    fpr = float(np.mean([abs(k["metrics"]["test_fpr"] - d["metrics"]["test_fpr"]) for d, k in zip(remove, retain)]))
                    interaction = np.array([(k["metrics"]["ap"] - k0["metrics"]["ap"])
                                            - (d["metrics"]["ap"] - d0["metrics"]["ap"])
                                            for k, k0, d, d0 in zip(retain, clean_retain, remove, clean_remove)])
                    removed = {name: sum(row["group_retention"][name]["removed"] for row in remove)
                               for name in ("common", "rare", "noise")}
                    rare_ok = (removed["rare"] >= protection["minimum_total_removed_genuine_rare_originals"]
                               and rare.mean() + tolerance >= protection["minimum_mean_retention_minus_remove_rare_recall"]
                               and np.sum(rare > strict) >= protection["minimum_strict_positive_seed_differences"]
                               and common + tolerance >= protection["minimum_mean_retention_minus_remove_common_recall"]
                               and fpr <= protection["maximum_mean_absolute_paired_test_fpr_difference"] + tolerance)
                    noise_ok = (removed["noise"] >= tradeoff["minimum_total_removed_mislabeled_originals"]
                                and interaction.mean() <= tradeoff["maximum_mean_AP_interaction"] + tolerance
                                and np.sum(interaction < -strict) >= tradeoff["minimum_strict_negative_seed_interactions"])
                    qualifies = bool(rare_ok and noise_ok)
                    assessment.update(rare_protection=bool(rare_ok), noise_tradeoff=bool(noise_ok), qualifies=qualifies,
                                      mean_rare_recall_difference=float(rare.mean()), positive_seed_differences=int(np.sum(rare > strict)),
                                      mean_common_recall_difference=float(common), mean_absolute_test_fpr_difference=fpr,
                                      mean_AP_interaction=float(interaction.mean()), negative_seed_interactions=int(np.sum(interaction < -strict)),
                                      AP_interactions=interaction.tolist(), seeds=seeds, removed_originals=removed)
                    if qualifies:
                        qualifying_bases.add((count, sigma))
                settings.append(assessment)
    controls = []
    bounds = rules["simple_control_coverage"]
    rare_tolerance = float(bounds["minimum_mean_rare_recall"].split(" - ")[1])
    common_tolerance = float(bounds["minimum_mean_common_recall"].split(" - ")[1])
    ap_tolerance = float(bounds["minimum_mean_AP"].split(" - ")[1])
    # Show every control in every base setting, including nonqualifying settings.
    for count in distribution["fit_counts"]["rare_grid"]:
        for sigma in distribution["rare_minority"]["isotropic_sigma_grid"]:
            for method in rules["simple_controls"]:
                checks = []
                for noise in design["label_corruption"]["flipped_majority_count_grid"]:
                    control = [lookup[count, sigma, noise, seed, method] for seed in seeds]
                    retain = [lookup[count, sigma, noise, seed, "AW_SEED_FILTER_RETAIN"] for seed in seeds]
                    remove = [lookup[count, sigma, noise, seed, "AW_REMOVE"] for seed in seeds]
                    check = {"flipped_count": noise, "covers": None}
                    if _successful(control + retain + remove):
                        rare_gap = _mean(retain, "rare_recall") - _mean(control, "rare_recall")
                        common_gap = _mean(retain, "common_recall") - _mean(control, "common_recall")
                        ap_gap = max(_mean(retain, "ap"), _mean(remove, "ap")) - _mean(control, "ap")
                        fpr = float(np.mean([abs(k["metrics"]["test_fpr"] - c["metrics"]["test_fpr"])
                                             for c, k in zip(control, retain)]))
                        check.update(covers=bool(rare_gap <= rare_tolerance + tolerance
                                                and common_gap <= common_tolerance + tolerance
                                                and ap_gap <= ap_tolerance + tolerance
                                                and fpr <= bounds["maximum_mean_absolute_paired_test_fpr_difference_from_retention"] + tolerance),
                                     rare_gap=rare_gap, common_gap=common_gap, AP_gap=ap_gap,
                                     mean_absolute_test_fpr_difference=fpr)
                    checks.append(check)
                covers = None if any(c["covers"] is None for c in checks) else all(c["covers"] for c in checks)
                controls.append({"rare_count": count, "rare_sigma": sigma, "method": method,
                                 "covers_all_noise_levels": covers, "levels": checks})
    if not complete:
        status = "incomplete_coverage"
    elif len(qualifying_bases) < rules["minimum_distinct_qualifying_base_settings"]:
        status = "tradeoff_trigger_not_met"
    elif all(any(c["covers_all_noise_levels"] is True and (c["rare_count"], c["rare_sigma"]) == base
                 for c in controls) for base in qualifying_bases):
        status = "simple_controls_suffice"
    else:
        status = "residual_tradeoff_requires_replication"
    return {"status": status, "qualifying_base_settings": len(qualifying_bases), "settings": settings,
            "ordinary_controls": controls, "interpretation": "Descriptive prospective screen; no significance or novelty claim."}


def summarize_cases(records: dict[str, dict], design: dict) -> dict:
    jobs, methods = planned_jobs(design), design["methods"]
    if set(records) - {case_id(job) for job in jobs}:
        raise ValueError("Unexpected contamination case records.")
    scores, lookup = [], {}
    for job in jobs:
        record = records.get(case_id(job))
        rows = record["rows"] if record is not None else []
        if record is not None and (record.get("job") != job or len(rows) != len(methods)
                                   or {r["method"] for r in rows} != set(methods)):
            raise ValueError("Saved case has mismatched jobs, methods, or duplicate rows.")
        indexed = {row["method"]: row for row in rows}
        for method in methods:
            row = indexed.get(method)
            status = row["status"] if row is not None else "missing"
            if status not in {"success", "unsupported", "failed", "missing"}:
                raise ValueError("Unknown method status.")
            metrics = (row or {}).get("metrics") if status == "success" else None
            if status == "success" and (not isinstance(metrics, dict) or any(
                    metrics.get(m) is None or not np.isfinite(metrics[m]) for m in METRICS)):
                raise ValueError("A successful row lacks finite evaluation metrics.")
            lookup[job["rare_count"], job["rare_sigma"], job["flipped_count"], job["seed"], method] = row
            retention = (row or {}).get("group_retention") or {}
            counts = (row or {}).get("output_counts", {})
            error = (row or {}).get("error", {})
            scores.append({"case_id": case_id(job), "fit_rare_count": job["rare_count"], "rare_sigma": job["rare_sigma"],
                           "flipped_count": job["flipped_count"], "seed": job["seed"], "method": method, "status": status,
                           "error_stage": error.get("stage"), "reason": error.get("reason"),
                           "generated_count": (row or {}).get("generated_count"),
                           "output_majority": counts.get("majority"), "output_minority": counts.get("minority"),
                           **{name + "_retained_fraction": retention.get(name, {}).get("fraction") for name in ("common", "rare", "noise")},
                           "selected_parameters": json.dumps((row or {}).get("selected_parameters")),
                           **{m: (metrics or {}).get(m) for m in (*METRICS, "threshold", "tie_cutoff", "calibration_fp",
                               "calibration_negatives", "calibration_fpr", "test_fp", "test_negatives", "test_tp", "test_positives",
                               "rare_tp", "rare_count", "common_tp", "common_count")}})
    coverage, summary, changes = [], [], []
    for count, sigma, noise in dict.fromkeys((j["rare_count"], j["rare_sigma"], j["flipped_count"]) for j in jobs):
        indexed = {(seed, method): lookup[count, sigma, noise, seed, method] for seed in design["seeds"] for method in methods}
        shared = [seed for seed in design["seeds"] if _successful([indexed[seed, method] for method in methods])]
        for method in methods:
            statuses = [indexed[seed, method]["status"] if indexed[seed, method] is not None else "missing" for seed in design["seeds"]]
            coverage.append({"fit_rare_count": count, "rare_sigma": sigma, "flipped_count": noise, "method": method,
                             "planned": len(design["seeds"]), "shared_successes": len(shared),
                             **{s: statuses.count(s) for s in ("success", "unsupported", "failed", "missing")}})
            retention_metrics = {name + "_retained_fraction": name for name in ("common", "rare", "noise")}
            for metric in (*METRICS, *retention_metrics):
                if metric in retention_metrics:
                    values = [indexed[seed, method]["group_retention"][retention_metrics[metric]]["fraction"] for seed in shared]
                    values = [value for value in values if value is not None]
                else:
                    values = [indexed[seed, method]["metrics"][metric] for seed in shared]
                summary.append({"fit_rare_count": count, "rare_sigma": sigma, "flipped_count": noise, "method": method,
                                "metric": metric, "n_shared_seeds": len(shared), "n_values": len(values), "shared_seeds": json.dumps(shared),
                                "mean": float(np.mean(values)) if values else None,
                                "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
            if noise:
                for seed in design["seeds"]:
                    clean, noisy = lookup[count, sigma, 0, seed, method], indexed[seed, method]
                    pair_ok = _successful([clean, noisy])
                    changes.append({"fit_rare_count": count, "rare_sigma": sigma, "flipped_count": noise,
                                    "seed": seed, "method": method, "paired_complete": pair_ok,
                                    **{m + "_noisy_minus_clean": noisy["metrics"][m] - clean["metrics"][m] if pair_ok else None for m in METRICS}})
    complete = all(row["status"] == "success" for row in scores)
    return {"status": "complete" if complete else "partial", "scores": scores, "coverage": coverage,
            "summary": summary, "paired_changes": changes, "screening": _screen(lookup, design, complete)}


def export_reports(directory: Path, records: dict[str, dict], design: dict) -> dict:
    report = summarize_cases(records, design)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("scores", "coverage", "summary", "paired_changes"):
        if report[name]:
            _atomic_csv(directory / (name + ".csv"), report[name])
    _atomic_json(directory / "screening.json", report["screening"])
    _atomic_json(directory / "status.json", {
        "status": report["status"], "planned_cases": len(planned_jobs(design)), "saved_cases": len(records),
        "planned_rows": len(report["scores"]), "success_rows": sum(row["status"] == "success" for row in report["scores"]),
        "classifier_fit_attempts": sum(r.get("classifier_fit_attempts", 0) for r in records.values()),
        "screening_status": report["screening"]["status"], "not_novelty_or_paper_reproduction_evidence": True,
        "summary_scope": "Identical successful seeds across seven methods within each setting; sample SD across seeds.",
    })
    return report
