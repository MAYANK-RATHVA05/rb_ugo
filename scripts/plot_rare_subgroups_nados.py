"""Regenerate publication figures from compact audited NADOS tables only."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs/research/rare_subgroups/nados_v1"
DEFAULT_OUTPUT = DEFAULT_INPUT / "figures"
COLORS = {"NADOS": "#0072B2", "WEIGHTED": "#D55E00", "AW_SEED_FILTER_RETAIN": "#009E73"}


def _save(fig, output: Path, name: str, pdf: PdfPages) -> None:
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot(input_dir: Path, output: Path) -> list[Path]:
    input_dir, output = Path(input_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(input_dir / "scores.csv")
    eligibility = pd.read_csv(input_dir / "eligibility.csv")
    exposure = pd.read_csv(input_dir / "parent_exposure.csv")
    parent_scores = pd.read_csv(
        ROOT / "docs/research/rare_subgroups/contamination_v1/scores.csv"
    )
    parent_exposure = pd.read_csv(
        ROOT / "docs/research/rare_subgroups/contamination_v1/parent_exposure.csv"
    )
    pdf_path = output / "NADOS_Diagnostic_Figures.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
        for axis, (rare_count, sigma) in zip(axes.flat, sorted(
                scores[["fit_rare_count", "rare_sigma"]].drop_duplicates().itertuples(index=False, name=None))):
            subset = scores[(scores.fit_rare_count == rare_count) & (scores.rare_sigma == sigma)]
            controls = parent_scores[(parent_scores.fit_rare_count == rare_count)
                                     & (parent_scores.rare_sigma == sigma)
                                     & parent_scores.method.isin(("WEIGHTED", "AW_SEED_FILTER_RETAIN"))]
            for method, data in (("NADOS", subset), *controls.groupby("method")):
                means = data.groupby("flipped_count").rare_recall.mean()
                axis.plot(means.index, means.values, marker="o", linewidth=2,
                          color=COLORS[method], label=method.replace("AW_SEED_FILTER_RETAIN", "AW retain"))
            axis.set_title(f"rare fit count={rare_count}, spread={sigma}")
            axis.set_xticks((0, 5, 10))
            axis.grid(alpha=.25)
        axes[0, 0].set_ylabel("Rare-group test recall")
        axes[1, 0].set_ylabel("Rare-group test recall")
        for axis in axes[1]:
            axis.set_xlabel("Wrong fitting labels")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle("NADOS rare-group recall versus two existing controls\nFive-seed descriptive means", fontsize=15)
        fig.tight_layout(rect=(0, .08, 1, .92))
        _save(fig, output, "rare_recall_comparison", pdf)

        contaminated = eligibility[eligibility.flipped_count > 0]
        grouped = contaminated.groupby(["fit_rare_count", "rare_sigma", "group"]).fraction.mean().unstack()
        labels = [f"r={r}\nσ={s}" for r, s in grouped.index]
        positions = np.arange(len(labels)); width = .36
        fig, axis = plt.subplots(figsize=(10, 5.8))
        axis.bar(positions - width / 2, grouped.common, width, label="Genuine common", color="#56B4E9")
        axis.bar(positions + width / 2, grouped.rare, width, label="Genuine rare", color="#CC79A7")
        axis.axhline(.5, color="black", linestyle="--", linewidth=1, label="Rare threshold 0.50")
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Mean eligible fraction")
        axis.set_title("NADOS seed eligibility exposes a rare/common gap\nAveraged over five seeds and both contaminated levels")
        axis.grid(axis="y", alpha=.25)
        axis.legend(frameon=False, ncol=3, loc="lower center")
        fig.tight_layout()
        _save(fig, output, "seed_eligibility", pdf)

        nados = exposure[exposure.flipped_count > 0].groupby("flipped_count").apply(
            lambda frame: frame.any_corrupted_parent.sum() / frame.generated.sum(),
            include_groups=False,
        )
        aw = parent_exposure[(parent_exposure.method == "AW_SEED_FILTER_RETAIN")
                             & (parent_exposure.flipped_count > 0)].groupby("flipped_count").apply(
            lambda frame: ((frame.clean_corrupt.fillna(0) + frame.corrupt_corrupt.fillna(0)).sum()
                           / frame.generated_count.sum()),
            include_groups=False,
        )
        positions = np.arange(2)
        fig, axis = plt.subplots(figsize=(8, 5.8))
        axis.bar(positions - width / 2, aw.values * 100, width, label="AW retention",
                 color=COLORS["AW_SEED_FILTER_RETAIN"])
        axis.bar(positions + width / 2, nados.values * 100, width, label="NADOS", color=COLORS["NADOS"])
        axis.set_xticks(positions, ["5 wrong labels", "10 wrong labels"])
        axis.set_ylabel("Generated rows with ≥1 corrupted parent (%)")
        axis.set_title("Reliability gating sharply reduces corrupted-parent exposure\nAggregated over 30 related cases per noise level")
        axis.grid(axis="y", alpha=.25)
        axis.legend(frameon=False)
        for bars in axis.containers:
            axis.bar_label(bars, fmt="%.2f%%", padding=3)
        fig.tight_layout()
        _save(fig, output, "corrupted_parent_exposure", pdf)
    return [pdf_path, *(output / name for name in
                        ("rare_recall_comparison.png", "seed_eligibility.png",
                         "corrupted_parent_exposure.png"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for path in plot(args.input, args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
