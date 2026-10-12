"""Run or resume the manifest-validated 74-dataset UGO benchmark matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark import (DEFAULT_MANIFEST, benchmark_status,
                           load_benchmark_manifest, run_benchmark)
from src.classifier_registry import CLASSIFIERS
from src.evaluation import AVAILABLE_METHODS, EvaluationConfig, METHODS


def main() -> None:
    document, _ = load_benchmark_manifest(DEFAULT_MANIFEST)
    names = [entry["dataset_id"] for entry in document["datasets"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/stage4r7-full"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selected-datasets", nargs="+", choices=names,
                        help="Execute only these shards while retaining the fixed full identity.")
    parser.add_argument("--classifiers", nargs="+", choices=[item.name for item in CLASSIFIERS],
                        default=[item.name for item in CLASSIFIERS])
    parser.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, default=list(METHODS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--method-parameters", type=Path)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--status-only", action="store_true",
                        help="Inspect completed datasets and saved job files without training.")
    args = parser.parse_args()
    if args.status_only:
        identity_path = args.output / "benchmark-config.json"
        if not identity_path.is_file():
            parser.error("--status-only requires an initialized benchmark output directory.")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        status = benchmark_status(args.output, identity)
        status["saved_job_files_including_partial_datasets"] = sum(
            1 for _ in (args.output / "datasets").glob("*/job-*.json"))
        print(json.dumps(status, indent=2))
        return
    parameters = (json.loads(args.method_parameters.read_text(encoding="utf-8"))
                  if args.method_parameters else {})
    config = EvaluationConfig(classifiers=tuple(args.classifiers), methods=tuple(args.methods),
                              seed=args.seed, scale=not args.no_scale,
                              method_parameters=parameters)
    def progress(report):
        print(f"dataset={report['dataset']} jobs={report['expected_jobs']} "
              f"success={report['success']} failed={report['failed']} "
              f"unavailable={report['unavailable']} reused={report['reused_jobs']} "
              f"seconds={report['invocation_wall_seconds']:.2f}", flush=True)
    result = run_benchmark(args.output, config, manifest_path=args.manifest,
                           selected_datasets=(tuple(args.selected_datasets)
                                              if args.selected_datasets else None),
                           workers=args.workers, retry_failures=args.retry_failures,
                           on_dataset=progress)
    print(json.dumps({"invocation": result["invocation"], "status": result["status"]}, indent=2))
    if result["status"]["failed"] or result["status"]["unavailable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
