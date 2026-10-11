"""Descriptive paper/pilot differences with explicit protocol and F1 limits."""
import argparse
import csv
import json
from pathlib import Path


def compare_pilot(summary, reference):
    methods = reference["methods_in_value_order"]
    rows = []
    for dataset in summary["datasets"]:
        name = dataset["dataset"]
        source = reference["source_tables"][name]
        for condition in dataset["summaries"]:
            classifier, method = condition["classifier"], condition["method"]
            if classifier not in reference["datasets"][name] or method not in methods:
                raise ValueError("Requested condition has no audited paper reference.")
            for metric, values in condition["metrics"].items():
                paper = reference["datasets"][name][classifier][metric][methods.index(method)]
                measured = values["method_mean"]
                rows.append({
                    "dataset": name, "table": source["table"], "page": source["page"],
                    "classifier": classifier, "method": method, "metric": metric,
                    "paper_value": paper, "pilot_mean": measured,
                    "pilot_minus_paper": measured - paper if measured is not None else None,
                    "paired_count": values["paired_count"], "classifier_fidelity": condition["classifier_fidelity"],
                    "comparison_status": ("F1 averaging unresolved; descriptive only" if metric == "f1"
                                          else "Different reconstruction/protocol; descriptive only"),
                })
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=root / "data/manifests/ugo_simulation_reference.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = compare_pilot(json.loads(args.summary.read_text(encoding="utf-8")),
                         json.loads(args.reference.read_text(encoding="utf-8")))
    if not rows:
        raise ValueError("No comparison rows.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} descriptive comparisons; no exact-match or significance claim.")


if __name__ == "__main__":
    main()
