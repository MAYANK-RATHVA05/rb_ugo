"""Read-only numerical audit; all output is restricted to docs/audit/.

Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/audit/recompute_numbers.py
No fitting, result-store writes, source changes or Git mutations occur.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sklearn.metrics import roc_auc_score
from src.data_loader import load_dataset

OUT = ROOT / "docs/audit"
METRICS = ("f1", "gmean", "mcc", "probability_auc", "fpr")
METHODS = ("nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo", "sshr")


def measure(record):
    positive = record["positive_label"]
    y = record["true_labels"]
    predicted = record["predictions"]
    tp = sum(a == positive and b == positive for a, b in zip(y, predicted))
    tn = sum(a != positive and b != positive for a, b in zip(y, predicted))
    fp = sum(a != positive and b == positive for a, b in zip(y, predicted))
    fn = sum(a == positive and b != positive for a, b in zip(y, predicted))
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    column = record["probability_labels"].index(positive)
    values = {"f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
              "gmean": math.sqrt(tp / (tp + fn) * tn / (tn + fp)),
              "mcc": (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.,
              "probability_auc": float(roc_auc_score([a == positive for a in y],
                                       [p[column] for p in record["probabilities"]])),
              "fpr": fp / (tn + fp)}
    assert all(math.isclose(values[m], record["metrics"][m], abs_tol=1e-12, rel_tol=0) for m in METRICS)
    return values


def main():
    OUT.mkdir(exist_ok=True)
    score_rows, ablation_rows, pseudo_rows, comparisons = [], [], [], []
    counts = {"jobs": 0, "recomputed_score_cells": 0, "recomputed_ablation_cells": 0,
              "identical_nores_pairs": 0}
    artifact_hashes = {}
    for dataset in ("ecoli4", "yeast4"):
        pools = {}
        gold = load_dataset(ROOT / f"data/raw/{dataset}.dat", minority_label="positive").y.to_numpy()
        for pool in ("with-u", "without-u"):
            directory = ROOT / "results/sshr-small-study" / dataset / pool
            records = {}
            for p in sorted(directory.glob("job-*.json")):
                record = json.loads(p.read_text())["record"]
                assert record["status"] == "success"
                job = record["job"]
                key = (job["classifier"], job["method"], job["repeat"], job["fold"])
                assert key not in records
                assert record["true_labels"] == gold[record["test_indices"]].tolist()
                records[key] = (record, measure(record))
                counts["jobs"] += 1
            assert len(records) == (300 if pool == "with-u" else 100)
            pools[pool] = records
        directory = ROOT / "docs/sshr_pilot" / dataset
        summary = json.loads((directory / "summary.json").read_text())
        assert set(summary["report"]) == {"clusters", "coverage", "diagnostics", "identity", "policy", "schema_version", "scores"}
        with (directory / "scores.csv").open(newline="") as stream:
            scores = { (r["classifier"], r["method"], r["metric"]): (row, r)
                       for row, r in enumerate(csv.DictReader(stream), 2)
                       if r["scope"] == "all_methods_shared_success" }
        with (directory / "ablation.csv").open(newline="") as stream:
            ablations = {(r["classifier"], r["metric"]): (row, r)
                         for row, r in enumerate(csv.DictReader(stream), 2)}
        for classifier in ("LogReg", "GNB"):
            for method in METHODS:
                subset = [(record, value) for key, (record, value) in pools["with-u"].items()
                          if key[0:2] == (classifier, method)]
                assert len(subset) == 25
                for metric in METRICS:
                    values = [value[metric] for _, value in subset]
                    mean, sd = statistics.mean(values), statistics.stdev(values)
                    row, committed = scores[classifier, method, metric]
                    assert math.isclose(mean, float(committed["mean"]), abs_tol=1e-12, rel_tol=0)
                    assert math.isclose(sd, float(committed["sample_sd"]), abs_tol=1e-12, rel_tol=0)
                    score_rows.append({"dataset": dataset, "classifier": classifier, "method": method,
                                       "metric": metric, "mean": mean, "sample_sd": sd, "n": 25,
                                       "evidence": f"docs/sshr_pilot/{dataset}/scores.csv row {row}"})
                    counts["recomputed_score_cells"] += 1
            main = pools["with-u"]
            empty = pools["without-u"]
            paired_keys = sorted(k for k in main if k[0:2] == (classifier, "sshr"))
            assert len(paired_keys) == 25
            for metric in METRICS:
                deltas = [main[k][1][metric] - empty[k][1][metric] for k in paired_keys]
                row, committed = ablations[classifier, metric]
                delta, sd = statistics.mean(deltas), statistics.stdev(deltas)
                assert math.isclose(delta, float(committed["paired_mean_difference"]), abs_tol=1e-12, rel_tol=0)
                assert math.isclose(sd, float(committed["paired_difference_sample_sd"]), abs_tol=1e-12, rel_tol=0)
                ablation_rows.append({"dataset": dataset, "classifier": classifier, "metric": metric,
                                      "delta": delta, "sample_sd_delta": sd, "n": 25,
                                      "evidence": f"docs/sshr_pilot/{dataset}/ablation.csv row {row}"})
                counts["recomputed_ablation_cells"] += 1
            selected = correct = 0
            for key in paired_keys:
                record = main[key][0]
                audit = record["pseudo_label_audit"]
                observed = sum(gold[i] == record["positive_label"] for i in audit["selected_global_indices"])
                assert int(observed) == audit["correct_count"]
                selected += audit["selected_count"]
                correct += int(observed)
                baseline = (classifier, "nores", key[2], key[3])
                for field in ("metrics", "predictions", "probabilities", "preprocessing", "seed", "labeled_indices", "test_indices"):
                    assert main[baseline][0][field] == empty[baseline][0][field]
                counts["identical_nores_pairs"] += 1
            pseudo_rows.append({"dataset": dataset, "classifier": classifier, "correct": correct,
                                "selected": selected, "precision": correct / selected})
        for p in sorted(directory.glob("*")):
            if p.is_file():artifact_hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    specified = [("ecoli4", "LogReg", "ugo_smote", "f1", .8890),
                 ("ecoli4", "LogReg", "nores", "f1", .8720),
                 ("ecoli4", "LogReg", "sshr", "f1", .5838),
                 ("yeast4", "LogReg", "ec_ebo", "f1", .3189),
                 ("yeast4", "LogReg", "nores", "f1", .2018),
                 ("yeast4", "LogReg", "sshr", "f1", .2962),
                 ("yeast4", "LogReg", "nores", "fpr", .0046),
                 ("yeast4", "LogReg", "ec_ebo", "fpr", .0923)]
    for dataset, classifier, method, metric, expected in specified:
        row = next(r for r in score_rows if (r["dataset"], r["classifier"], r["method"], r["metric"]) == (dataset, classifier, method, metric))
        comparisons.append({"claim": f"{dataset} {classifier} {method} {metric}", "reported": expected,
                            "recomputed": row["mean"], "matches_4dp": round(row["mean"], 4) == expected,
                            "evidence": row["evidence"]})
    for dataset, classifier, expected in [("ecoli4", "LogReg", -.0777),("ecoli4", "GNB", -.0922),
                                           ("yeast4", "LogReg", -.0096),("yeast4", "GNB", -.0054)]:
        row = next(r for r in ablation_rows if (r["dataset"],r["classifier"],r["metric"]) == (dataset,classifier,"f1"))
        comparisons.append({"claim": f"{dataset} {classifier} SSHR delta f1", "reported": expected,
                            "recomputed": row["delta"], "matches_4dp": round(row["delta"],4)==expected,
                            "evidence": row["evidence"]})
    assert all(r["matches_4dp"] for r in comparisons)
    for filename, rows in [("recomputed_scores.csv", score_rows),("recomputed_ablation.csv",ablation_rows),("pseudo_label_audit.csv",pseudo_rows)]:
        with (OUT / filename).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(rows)
    report={"scope":"No refits; aggregate CSV/JSON compared with independently recomputed original saved job scores and original hidden gold labels.",
            "aggregate_folder_raw_fold_values_available":False,"counts":counts,"comparisons":comparisons,
            "scores":score_rows,"ablation":ablation_rows,"pseudo":pseudo_rows,
            "preexisting_artifact_sha256":artifact_hashes,"mismatches":[]}
    (OUT / "recomputed_numbers.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({**counts,"report_values_checked":len(comparisons),"mismatches":0,
                      "pseudo":pseudo_rows},sort_keys=True))


if __name__ == "__main__":
    main()
