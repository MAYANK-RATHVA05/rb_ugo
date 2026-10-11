"""Execute the frozen Stage 4R-6 simulation pilot and export descriptive results."""
import argparse
import json
from pathlib import Path

from src.pilot import export_pilot_report, run_simulation_pilot


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/stage4r6_pilot.json")
    parser.add_argument("--output", type=Path, default=root / "results/stage4r6-pilot")
    parser.add_argument("--report-dir", type=Path, help="Optional compact JSON/CSV report export.")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    def progress(dataset, record, cached):
        job = record["job"]
        print(f"{dataset} fold={job['fold']} {job['classifier']} {job['method']}: {record['status']}"
              f"{' (reused)' if cached else ''}", flush=True)
    report = run_simulation_pilot(spec, args.output, on_record=progress)
    if args.report_dir:
        export_pilot_report(report, args.report_dir)
    print(json.dumps(report["current_invocation"], indent=2))
    if any(s["failed"] or s["unavailable"] for d in report["datasets"] for s in d["summaries"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
