"""Run the frozen, opt-in supervised Question B diagnostic in two phases."""
import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.subgroup_diagnostic import load_protocol, open_store, run_mechanism, run_pilot, write_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("mechanism", "pilot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    with threadpool_limits(limits=1):
        store = open_store(args.output, protocol)
        if args.phase == "mechanism":
            records = run_mechanism(store, protocol)
            print(json.dumps({"run_id": store.run_id, "phase": args.phase, "jobs": len(records)}))
        else:
            records = run_pilot(store, protocol)
            summary = write_summary(store, records, protocol)
            print(json.dumps({"run_id": store.run_id, "phase": args.phase, "status_counts": summary["status_counts"]}))


if __name__ == "__main__":
    main()
