"""Plot saved complete pilot scores without generating data or fitting models."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "docs/research/rare_subgroups/pilot_v1"
STYLES = {
    "NORES": ("No resampling", "#666666", "o", "--"),
    "WEIGHTED": ("Class weighting", "#111111", "s", ":"),
    "ROS": ("ROS", "#0072B2", "^", "-"),
    "SMOTE": ("SMOTE", "#009E73", "D", "-"),
    "AW_REMOVE": ("AW removal", "#D55E00", "X", "-"),
    "AW_SEED_FILTER_RETAIN": ("AW retention control", "#E69F00", "v", "-"),
    "AW_REMOVE_TUNED": ("AW tuned", "#CC79A7", "P", "-"),
}


def plot_scores(scores_path: Path, output: Path) -> list[Path]:
    frame = pd.read_csv(scores_path)
    expected = {(count, sigma, seed, method) for count in (4, 8, 16) for sigma in (.35, .9)
                for seed in range(42, 47) for method in STYLES}
    actual = set(frame[["fit_rare_count", "rare_sigma", "seed", "method"]].itertuples(index=False, name=None))
    if len(frame) != len(expected) or actual != expected or not frame.status.eq("success").all():
        raise ValueError("This plot requires all 210 successful rows from the frozen pilot.")
    columns = ["rare_recall", "test_fpr", "common_retained_fraction", "rare_retained_fraction"]
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError("Plot metrics must be finite.")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9), sharex="col", sharey="row")
    handles = []
    for column, sigma in enumerate((.35, .9)):
        subset = frame[frame.rare_sigma == sigma]
        for method, (label, color, marker, line) in STYLES.items():
            rows = subset[subset.method == method]
            means = rows.groupby("fit_rare_count")[["rare_recall", "test_fpr"]].mean().sort_index()
            for row, metric in enumerate(("rare_recall", "test_fpr")):
                ax = axes[row, column]
                for count, group in rows.groupby("fit_rare_count"):
                    group = group.sort_values("seed")
                    ax.scatter(count + np.linspace(-.12, .12, 5), group[metric], color=color,
                               alpha=.25, s=13, edgecolors="none", zorder=2)
                drawn, = ax.plot(means.index, means[metric], color=color, marker=marker,
                                 linestyle=line, markersize=6, linewidth=1.8, label=label, zorder=3)
                if column == 0 and row == 0:
                    handles.append(drawn)
        axes[0, column].set_title(f"Rare-component spread = {sigma:.2f}", fontsize=13, weight="bold")
        axes[0, column].set_ylim(0, 1.04)
        axes[1, column].set_ylim(0, max(.065, frame.test_fpr.max() + .006))
        axes[1, column].axhline(.05, color="#888888", linestyle="--", linewidth=1)
        axes[1, column].set_xlabel("Rare minority fitting examples (out of 100)")
        for row in (0, 1):
            axes[row, column].set_xticks([4, 8, 16])
            axes[row, column].grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Rare-group recall")
    axes[1, 0].set_ylabel("Actual test false-positive rate")
    figure.suptitle("Rare-group recall and false alarms", fontsize=19, weight="bold", y=.98)
    figure.text(.5, .935, "Lines: five-seed means. Faint points: individual seeds. Six synthetic settings.", ha="center")
    figure.legend(handles=handles, labels=[h.get_label() for h in handles], loc="lower center",
                  bbox_to_anchor=(.5, .035), ncol=4, frameon=False)
    figure.text(.5, .012, "RF with 100 trees; calibration FPR target ≤ 0.05 (dashed). Achieved test FPR varies. Descriptive results.",
                ha="center", fontsize=9, color="#555555")
    figure.subplots_adjust(top=.87, bottom=.19, hspace=.21, wspace=.15, left=.075, right=.985)
    first = output / "rare_recall_and_fpr.png"
    figure.savefig(first, dpi=170)

    retention, retention_axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    default = frame[frame.method == "AW_REMOVE"]
    x = np.arange(3)
    for ax, sigma in zip(retention_axes, (.35, .9)):
        subset = default[default.rare_sigma == sigma]
        for offset, metric, label, color in ((-.18, "common_retained_fraction", "Common minority", "#777777"),
                                             (.18, "rare_retained_fraction", "Rare minority", "#D55E00")):
            means = subset.groupby("fit_rare_count")[metric].mean().sort_index()
            bars = ax.bar(x + offset, means, width=.32, color=color, label=label, alpha=.85)
            ax.bar_label(bars, labels=[f"{value:.1%}" for value in means], padding=4, fontsize=10)
            for index, count in enumerate((4, 8, 16)):
                values = subset[subset.fit_rare_count == count].sort_values("seed")[metric]
                ax.scatter(index + offset + np.linspace(-.045, .045, 5), values, s=18, color="black", alpha=.5)
        ax.set_title(f"Rare-component spread = {sigma:.2f}", fontsize=12, weight="bold")
        ax.set_xticks(x, [4, 8, 16])
        ax.set_xlabel("Rare minority fitting examples (out of 100)")
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    retention_axes[0].set_ylabel("Fraction of original group retained")
    retention.suptitle("Which correctly labelled observations did AW removal retain?", fontsize=16, weight="bold", y=.97)
    retention.legend(*retention_axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
                     bbox_to_anchor=(.5, .045), frameon=False)
    retention.text(.5, .015, "Bars: five-seed means; dots: individual seeds. Original fitting rows only; all generating labels are correct.",
                   ha="center", fontsize=9, color="#555555")
    retention.subplots_adjust(top=.81, bottom=.24, left=.07, right=.985, wspace=.12)
    second = output / "original_group_retention.png"
    retention.savefig(second, dpi=170)
    pdf = output / "Rare_Subgroup_Pilot_Figures.pdf"
    with PdfPages(pdf, metadata={"CreationDate": None, "ModDate": None}) as pages:
        pages.savefig(figure)
        pages.savefig(retention)
    plt.close(figure)
    plt.close(retention)
    return [first, second, pdf]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=PUBLISHED / "scores.csv")
    parser.add_argument("--output", type=Path, default=PUBLISHED / "figures")
    args = parser.parse_args()
    for path in plot_scores(args.scores, args.output):
        print(path)


if __name__ == "__main__":
    main()
