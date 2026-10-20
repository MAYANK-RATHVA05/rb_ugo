"""Run only the frozen 30-case synthetic pilot; no real-data benchmarking."""
import argparse
from pathlib import Path
from src.gmmsampling_overlap_study import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/gmmsampling-overlap-pilot"))
    args = parser.parse_args()
    decision = run(args.output)
    print(f"Saved: {args.output}; advancement_passed={decision['advancement_passed']}")


if __name__ == "__main__":
    main()
