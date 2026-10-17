"""Audit the frozen diagnostic and export reviewed aggregate evidence only.

Added after execution; never changes the frozen sampler/runner identity. Optional
replay refits prespecified primary models and independently recomputes test AP.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from src.results import ResultStore, content_hash
from src.subgroup_diagnostic import _prepared, make_partition


def independent_ap(y, scores):
    """Non-interpolated AP, grouping equal-score observations at one threshold."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.shape != scores.shape or not np.isfinite(scores).all() or not np.any(y == 1):
        raise ValueError("Aligned finite scores and positive observations required.")
    order = np.argsort(-scores, kind="stable")
    truth, sorted_scores = y[order] == 1, scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(truth)-1]
    tp = np.cumsum(truth)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) / truth.sum() * tp/(ends+1)))


def _csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def export(input_dir, output_dir, *, replay=False):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    envelope = json.loads((input_dir / "manifest.json").read_text())
    store = ResultStore(input_dir, envelope["manifest"])
    if envelope["run_id"] != store.run_id:
        raise ValueError("Manifest run identity mismatch.")
    protocol = envelope["manifest"]["protocol"]
    records = []
    for path in sorted(input_dir.glob("job-*.json")):
        job = json.loads(path.read_text())["record"]["job"]
        records.append(store.read(job))  # Verifies each terminal-record checksum.
    mechanism = [r for r in records if r["job"]["role"] == "mechanism"]
    pilot = [r for r in records if r["job"]["role"] == "pilot"]
    cells = len(protocol["seeds"])*len(protocol["classifiers"])*len(protocol["regimes"])
    if len(mechanism) != 21*cells or len(pilot) != 27*cells:
        raise ValueError("Incomplete frozen diagnostic matrix.")
    for r in mechanism:
        if r["status"] == "success":
            path = input_dir / r["arrays"]["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != r["arrays"]["sha256"]:
                raise ValueError("Array checksum mismatch.")
            if r["training"]["original_rows_retained"] != protocol["sizes"]["fit"]:
                raise ValueError("An original fitting row was lost.")
    summary = json.loads((input_dir / "summary.json").read_text())
    if summary["run_id"] != store.run_id:
        raise ValueError("Summary belongs to a different run.")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = []
    for entry in summary["aggregates"]:
        regime, classifier, arm = (entry[k] for k in ("regime", "classifier", "arm"))
        chosen = [r for r in pilot if r["job"]["regime"] == regime and
                  r["job"]["classifier"] == classifier and r["job"]["arm"] == arm]
        success = [r for r in chosen if r["status"] == "success"]
        out = {"regime": regime, "classifier": classifier, "arm": arm,
               "success": len(success), "failed": len(chosen)-len(success),
               **{f"mean_{k}": v for k, v in entry["means"].items()}}
        if success:
            out.update(min_test_fpr=min(r["metrics"]["fpr"] for r in success),
                       max_test_fpr=max(r["metrics"]["fpr"] for r in success),
                       mean_default_worst_recall=float(np.mean([r["default_threshold_metrics"]["worst_recall"] for r in success])),
                       mean_default_fpr=float(np.mean([r["default_threshold_metrics"]["fpr"] for r in success])))
            for g in range(3):
                out[f"mean_generated_inside_g{g}"] = float(np.mean([r["training"]["minority_generated_inside_support_by_group"][g] for r in success]))
                eligible = [r["training"]["eligible_by_group"][g] for r in success
                            if r["training"]["eligible_by_group"] is not None]
                out[f"mean_eligible_g{g}"] = float(np.mean(eligible)) if eligible else None
            for c in (0, 1):
                added = [r["training"]["generated_by_class"][c] for r in success]
                out[f"mean_generated_class{c}"] = float(np.mean(added))
                out[f"min_generated_class{c}"], out[f"max_generated_class{c}"] = min(added), max(added)
            out["mean_outside_minority_support"] = float(np.mean([r["training"]["minority_generated_outside_support"] for r in success]))
        aggregates.append(out)
    _csv(output_dir / "aggregate_metrics.csv", aggregates)
    _csv(output_dir / "paired_contrasts.csv", [{**r, "paired_seeds": " ".join(map(str, r["paired_seeds"]))}
                                             for r in summary["paired"]])
    mechanism_rows = []
    for regime in protocol["regimes"]:
        for classifier in protocol["classifiers"]:
            arms = sorted({r["job"]["arm"] for r in mechanism})
            for arm in arms:
                rows = [r for r in mechanism if r["job"]["regime"] == regime and r["job"]["classifier"] == classifier and r["job"]["arm"] == arm]
                good = [r["training"] for r in rows if r["status"] == "success"]
                entry = {"regime": regime, "classifier": classifier, "arm": arm, "success": len(good),
                         "original_rows_retained_min": min((r["original_rows_retained"] for r in good), default=None)}
                if good and "generation_rounds" in good[0]:
                    for key in ("generation_rounds", "minority_generation_rounds", "rare_filter_opportunity_rounds", "ranked_rare_filter_opportunity_rounds"):
                        entry[f"total_{key}"] = sum(r[key] for r in good)
                        entry[f"seeds_with_{key}"] = sum(r[key] > 0 for r in good)
                    for g in range(3):
                        entry[f"mean_drawn_seed_g{g}"] = float(np.mean([r["drawn_seed_counts_by_group"][g] for r in good]))
                mechanism_rows.append(entry)
    _csv(output_dir / "mechanism_summary.csv", mechanism_rows)
    selections = Counter((r["job"]["regime"], r["job"]["classifier"], r["job"]["arm"], r.get("selected_from"))
                         for r in pilot if r["job"]["arm"] in {"ugo_tuned_k", "nores_tuned_capacity"})
    _csv(output_dir / "validation_selections.csv", [dict(regime=k[0], classifier=k[1], arm=k[2], selected=k[3], seeds=n)
                                                   for k, n in sorted(selections.items())])
    verified = []
    if replay:
        for r in pilot:
            job = r["job"]
            if (job["regime"] != "clean" or job["classifier"] != "RF100" or
                    job["arm"] not in {"ugo_k5", "ugo_generation_filter_disabled", "nores", "ros_match_ugo_generation_filter_disabled"}):
                continue
            F = make_partition(protocol, job["seed"], "fit", job["regime"])
            V = make_partition(protocol, job["seed"], "validation", job["regime"])
            T = make_partition(protocol, job["seed"], "test", job["regime"])
            pre, _ = _prepared(F)
            mjob = {**job, "role": "mechanism"}
            m = store.read(mjob)
            with np.load(input_dir / m["arrays"]["file"], allow_pickle=False) as d:
                model = RandomForestClassifier(**r["model_parameters"])
                model.fit(d["X"], d["y"])
            val_scores = model.predict_proba(pre.transform(V.X))[:, 1]
            negatives = np.sort(val_scores[V.y == 0])
            threshold = negatives[len(negatives)-int(protocol["threshold_fpr"]*len(negatives))-1]
            scores = model.predict_proba(pre.transform(T.X))[:, 1]
            prediction = scores > threshold
            measured = independent_ap(T.y, scores)
            if not np.isclose(measured, r["metrics"]["average_precision"], atol=1e-12, rtol=0):
                raise ValueError("Independent AP replay does not match.")
            if threshold != r["validation"]["threshold"] or prediction[T.y == 0].mean() != r["metrics"]["fpr"]:
                raise ValueError("Threshold/FPR replay does not match.")
            for g in range(3):
                if prediction[(T.y == 1) & (T.groups == g)].mean() != r["metrics"]["subgroup_recall"][g]:
                    raise ValueError("Group recall replay does not match.")
            verified.append({"seed": job["seed"], "arm": job["arm"]})
    metadata = {"run_id": store.run_id, "protocol_sha256": content_hash(protocol),
                "software": envelope["manifest"]["software"],
                "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "mechanism_status": dict(Counter(r["status"] for r in mechanism)),
                "pilot_status": dict(Counter(r["status"] for r in pilot)),
                "independently_replayed_primary_models": verified,
                "artifacts_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in output_dir.glob("*.csv")}}
    (output_dir / "validation.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-primary", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = export(args.input, args.output, replay=args.replay_primary)
    print(json.dumps({"run_id": result["run_id"], "replayed": len(result["independently_replayed_primary_models"])}))


if __name__ == "__main__":
    main()
