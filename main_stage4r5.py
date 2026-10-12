"""Stage 4R-5 evaluation entry point; real inputs require explicit positive labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.classifier_registry import CLASSIFIERS
from src.data_loader import load_dataset
from src.evaluation import AVAILABLE_METHODS, EvaluationConfig, evaluate_ten_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="Local CSV or KEEL file; no automatic downloading.")
    source.add_argument("--software-check", action="store_true", help="Small artificial integration fixture.")
    parser.add_argument("--target", help="CSV target column (required by the loader).")
    parser.add_argument("--minority-label", help="Optional original source label for loader mapping.")
    parser.add_argument("--positive-label", help="Explicit original source/domain label for F1.")
    parser.add_argument("--output", type=Path, required=True, help="Dedicated run directory; identities must match to resume.")
    parser.add_argument("--classifiers", nargs="+", choices=[c.name for c in CLASSIFIERS], default=["LogReg"])
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, default=["nores", "ros"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--method-parameters", type=Path, help="JSON mapping method names to fixed parameters.")
    parser.add_argument("--folds", type=Path, help="JSON list of ten objects with train_indices/test_indices.")
    parser.add_argument("--fold-source", help="Authority/provenance for supplied folds; required with --folds.")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if args.dataset:
        if args.positive_label is None:
            parser.error("--dataset requires --positive-label with the original source/domain label.")
        loaded = load_dataset(args.dataset, target_column=args.target, minority_label=args.minority_label)
        if args.positive_label not in loaded.label_mapping:
            parser.error("--positive-label is not an original label in the input.")
        X, y = loaded.X, loaded.y
        positive = loaded.label_mapping[args.positive_label]
        dataset_id = args.dataset.stem
        provenance = {"kind": "caller-supplied local dataset", "filename": args.dataset.name,
                      "file_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                      "paper_74_membership": "unverified", "label_mapping": loaded.label_mapping,
                      "source_positive_label": args.positive_label}
    else:
        if args.target or args.minority_label or args.positive_label:
            parser.error("Input-label options apply to --dataset only.")
        rng = np.random.default_rng(args.seed)
        X = pd.DataFrame(rng.normal(size=(80, 2)), columns=["a", "b"])
        y = np.array([0] * 60 + [1] * 20)
        positive, dataset_id = 1, "stage4r5_artificial_software_fixture"
        provenance = {"kind": "artificial software fixture", "paper_dataset": False,
                      "purpose": "integration verification; not paper performance evidence"}
    folds = None
    if args.folds:
        if not args.fold_source:
            parser.error("--folds requires --fold-source.")
        definitions = json.loads(args.folds.read_text(encoding="utf-8"))
        folds = [(f["train_indices"], f["test_indices"]) for f in definitions]
        provenance["fold_source"] = args.fold_source
        provenance["fold_file_sha256"] = hashlib.sha256(args.folds.read_bytes()).hexdigest()
    elif args.fold_source:
        parser.error("--fold-source requires --folds.")
    parameters = json.loads(args.method_parameters.read_text(encoding="utf-8")) if args.method_parameters else {}
    config = EvaluationConfig(classifiers=tuple(args.classifiers), methods=tuple(args.methods),
                              seed=args.seed, scale=not args.no_scale, method_parameters=parameters)
    def progress(record, cached):
        job = record["job"]
        print(f"fold={job['fold']} {job['classifier']} {job['method']}: {record['status']}"
              f"{' (reused)' if cached else ''}", flush=True)
    result = evaluate_ten_fold(X, y, dataset_id=dataset_id, positive_label=positive,
                               provenance=provenance, output_dir=args.output, config=config,
                               supplied_folds=folds, retry_failures=args.retry_failures, on_record=progress)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if any(r["status"] != "success" for r in result["records"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
