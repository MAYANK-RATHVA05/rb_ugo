"""Run opt-in paired SSHR comparisons on artificial or hash-verified local data.

Stage 3 uses --software-check only. Real-data execution is reserved for an
explicitly authorized small study; this script never downloads datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.sshr_evaluation import CONTROLLED_METHODS, ControlledEvaluationConfig, evaluate_controlled


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--software-check", action="store_true", help="Unique artificial numeric rows; no real-data study.")
    source.add_argument("--data", type=Path, help="One existing local numeric CSV or KEEL .dat file.")
    parser.add_argument("--sha256", help="Required expected source-file SHA-256 for --data.")
    parser.add_argument("--target", help="CSV target column; optional KEEL override.")
    parser.add_argument("--minority-label", help="Explicit original source minority label for local input.")
    parser.add_argument("--numeric-source", action="store_true",
                        help="Confirm original CSV features are numerical, with no categorical/boolean or one-hot encoded sources.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=CONTROLLED_METHODS, default=list(CONTROLLED_METHODS))
    parser.add_argument("--classifiers", nargs="+", default=["LogReg"])
    parser.add_argument("--folds", type=int, help="Default: 2 for software checks, 5 for local studies.")
    parser.add_argument("--repeats", type=int, help="Default: 1 for software checks, 5 for local studies.")
    parser.add_argument("--labeled-fraction", type=float, default=0.5)
    parser.add_argument("--min-labeled-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-unlabeled", action="store_true", help="Keep identical L/masks; give SSHR an empty U pool.")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--method-parameters", help="Explicit JSON object keyed by selected method names.")
    args = parser.parse_args(argv)
    script_hash = _file_hash(Path(__file__))
    if args.software_check:
        if args.sha256 or args.target or args.minority_label or args.numeric_source:
            parser.error("--sha256, --target, --minority-label and --numeric-source apply only to --data.")
        if isinstance(args.seed, bool) or not 0 <= args.seed < 2**32:
            parser.error("seed must be in [0, 2**32).")
        rng = np.random.default_rng(args.seed)
        X = pd.DataFrame(np.r_[rng.normal(-2, 0.3, (80, 2)), rng.normal(2, 0.3, (40, 2))],
                         columns=["x0", "x1"])
        y = np.r_[np.zeros(80, dtype=int), np.ones(40, dtype=int)]
        dataset_id = "sshr_artificial_software_check"
        provenance = {"kind": "artificial software fixture; not experimental evidence",
                      "generation": "80 majority and 40 minority Gaussian rows; means -2/+2; SD .3; 2 coordinates",
                      "generation_seed": args.seed, "feature_types": {name: "continuous" for name in X},
                      "runner_script_sha256": script_hash}
        default_parameters = {method: parameters for method, parameters in {
            "smote": {"k_neighbors": 1},
            "ugo_smote": {"max_iter": 2, "k_pre": 1, "k_after": 1, "k_smote": 1},
        }.items() if method in args.methods}
    else:
        if not args.sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
            parser.error("--data requires an explicit 64-character --sha256.")
        if not args.data.is_file():
            parser.error("--data must be an existing regular local file.")
        source_hash = _file_hash(args.data)
        if source_hash != args.sha256.lower():
            parser.error("Source SHA-256 mismatch; no evaluation or output was created.")
        if args.data.suffix.lower() == ".csv" and not args.numeric_source:
            parser.error("CSV requires --numeric-source confirmation; numeric storage cannot verify source semantics.")
        loaded = load_dataset(args.data, target_column=args.target, minority_label=args.minority_label)
        if _file_hash(args.data) != source_hash:
            parser.error("Source file changed during loading; no evaluation or output was created.")
        X, y = loaded.X, loaded.y.to_numpy()
        dataset_id = args.data.stem
        provenance = {"kind": "hash-verified local numerical input; author dataset/folds unverified",
                      "source_path": str(args.data.resolve()), "source_sha256": source_hash,
                      "source_format": loaded.source_format, "target_name": loaded.target_name,
                      "original_label_mapping": loaded.label_mapping,
                      "source_positive_label": loaded.minority_label,
                      "feature_types": {name: ("numeric" if name in loaded.numeric_feature_names else "categorical")
                                        for name in loaded.feature_names},
                      "feature_metadata_authority": ("KEEL original attribute declarations" if loaded.source_format == "dat"
                                                     else "caller --numeric-source confirmation; CSV numeric parse alone cannot establish semantic types"),
                      "runner_script_sha256": script_hash}
        default_parameters = {}
    if args.method_parameters is not None:
        try:
            parameters = json.loads(args.method_parameters)
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid --method-parameters JSON: {exc}")
    else:
        parameters = default_parameters
    config = ControlledEvaluationConfig(
        classifiers=tuple(args.classifiers), methods=tuple(args.methods), seed=args.seed,
        scale=not args.no_scale, n_splits=args.folds if args.folds is not None else (2 if args.software_check else 5),
        n_repeats=args.repeats if args.repeats is not None else (1 if args.software_check else 5),
        labeled_fraction=args.labeled_fraction, min_labeled_per_class=args.min_labeled_per_class,
        use_unlabeled=not args.no_unlabeled, method_parameters=parameters)
    result = evaluate_controlled(X, y, dataset_id=dataset_id, positive_label=1,
                                 provenance=provenance, output_dir=args.output, config=config,
                                 retry_failures=args.retry_failures)
    print(json.dumps({"purpose": "software validation only" if args.software_check else "descriptive local controlled study",
                      "run_id": result["run_id"], "output_dir": result["output_dir"],
                      "reused_jobs": result["reused_jobs"], "coverage": result["coverage"]}, sort_keys=True))
    return 0 if all(record["status"] == "success" for record in result["records"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
