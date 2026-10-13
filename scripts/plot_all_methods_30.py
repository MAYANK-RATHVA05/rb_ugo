"""Make presentation figures from the audited 30-variant aggregate CSV files.

The figures are descriptive. Dataset variants share source families, and the
repeated folds are not independent observations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "docs/presentation/Imbalanced_Learning_30_Dataset_Comparison.csv"
ABLATION = ROOT / "docs/presentation/Imbalanced_Learning_30_Dataset_SSHR_Ablation.csv"
OUTPUT = ROOT / "docs/presentation/figures"
CLASSIFIERS = ("LogReg", "GNB")
METHODS = (
    ("nores", "No resampling", "#333333", "o"),
    ("smote", "SMOTE", "#0072B2", "s"),
    ("ugo_smote", "UGO-SMOTE", "#E69F00", "D"),
    ("bfch_smote", "BFCH-SMOTE", "#009E73", "^"),
    ("ec_ebo", "EC-EBO", "#CC79A7", "v"),
    ("sshr", "SSHR", "#D55E00", "P"),
)
SHORT_NAMES = {
    "smote": "SMOTE",
    "ugo_smote": "UGO-SMOTE",
    "bfch_smote": "BFCH-SMOTE",
    "ec_ebo": "EC-EBO",
    "sshr": "SSHR",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_data(table_path: Path, ablation_path: Path):
    table = {}
    for row in read_csv(table_path):
        key = (row["dataset_id"], row["classifier"], row["metric"])
        if key in table:
            raise ValueError(f"Duplicate comparison row: {key}")
        table[key] = {name: float(row[name]) for name, *_ in METHODS}
    ablation = {}
    for row in read_csv(ablation_path):
        key = (row["dataset_id"], row["classifier"], row["metric"])
        if key in ablation:
            raise ValueError(f"Duplicate ablation row: {key}")
        ablation[key] = float(row["paired_mean_difference"])
    names = {key[0] for key in table}
    if len(names) != 30 or len(table) != 30 * 2 * 5 or len(ablation) != 30 * 2 * 5:
        raise ValueError("Expected 30 variants, two classifiers and five metrics")
    if {key[0] for key in ablation} != names:
        raise ValueError("Comparison and ablation datasets differ")
    return table, ablation, sorted(names)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.7)
    ax.set_axisbelow(True)


def tradeoff_figure(table, names):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.1), sharex=True, sharey=True)
    for ax, classifier in zip(axes, CLASSIFIERS):
        for method, label, color, marker in METHODS:
            f1 = mean(table[name, classifier, "f1"][method] for name in names)
            fpr = mean(table[name, classifier, "fpr"][method] for name in names)
            ax.scatter(fpr, f1, s=100, c=color, marker=marker, edgecolor="white",
                       linewidth=0.8, label=label, zorder=3)
        ax.set_title(classifier, weight="bold")
        ax.set_xlim(0, 0.185)
        ax.set_ylim(0.39, 0.53)
        ax.set_xlabel("False-positive rate (lower is better)  ←")
        style(ax)
    axes[0].set_ylabel("Minority F1 (higher is better)  ↑")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.04))
    fig.suptitle("F1 versus false alarms across 30 dataset variants", fontsize=15, weight="bold")
    fig.text(0.5, 0.89, "Each point averages 30 dataset-level means; the upper-left direction is preferable.",
             ha="center", fontsize=9.5)
    fig.text(0.5, 0.025, "Descriptive averages; variants share source families. Other metrics remain in the report.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.09, right=0.98, wspace=0.12)
    return fig


def consistency_figure(table, names):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.1), sharex=True, sharey=True)
    methods = [method for method, *_ in METHODS if method != "nores"]
    for ax, classifier in zip(axes, CLASSIFIERS):
        for y, method in enumerate(methods):
            deltas = [table[name, classifier, "f1"][method]
                      - table[name, classifier, "f1"]["nores"] for name in names]
            improved = sum(value > 1e-8 for value in deltas)
            tied = sum(abs(value) <= 1e-8 for value in deltas)
            lower = sum(value < -1e-8 for value in deltas)
            left = 0
            for count, color in ((improved, "#009E73"), (tied, "#b9b9b9"), (lower, "#D55E00")):
                if count:
                    ax.barh(y, count, left=left, color=color, height=0.65, edgecolor="white")
                    if count >= 3:
                        ax.text(left + count / 2, y, str(count), va="center", ha="center",
                                color="white" if color != "#b9b9b9" else "black", weight="bold")
                left += count
        ax.set_yticks(range(len(methods)), [SHORT_NAMES[method] for method in methods])
        ax.set_xlim(0, 30)
        ax.set_xticks((0, 5, 10, 15, 20, 25, 30))
        ax.set_xlabel("Number of dataset variants (out of 30)")
        ax.set_title(classifier, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e3e3e3", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    from matplotlib.patches import Patch
    legend = [Patch(color=color, label=label) for color, label in
              (("#009E73", "F1 higher than NORES"),
               ("#b9b9b9", "Same F1"),
               ("#D55E00", "F1 lower than NORES"))]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("How consistently did each method improve F1?", fontsize=15, weight="bold")
    fig.text(0.5, 0.89, "Same-fold dataset means are compared with no resampling, separately for each classifier.",
             ha="center", fontsize=9.5)
    fig.text(0.5, 0.025, "Counts describe related dataset variants, not independent source datasets or statistical significance.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(top=0.81, bottom=0.22, left=0.13, right=0.98, wspace=0.16)
    return fig


def ablation_figure(ablation, names):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.1))
    for ax, metric, title, explanation in (
        (axes[0], "f1", "Change in F1", "Below zero: unlabelled data lowered F1"),
        (axes[1], "fpr", "Change in false-positive rate", "Above zero: unlabelled data raised false alarms"),
    ):
        means = {}
        for x, classifier, color in ((0, "LogReg", "#0072B2"), (1, "GNB", "#E69F00")):
            values = [ablation[name, classifier, metric] for name in names]
            means[classifier] = mean(values)
            # Fixed, evenly spaced offsets show all variants without random jitter.
            offsets = [((index * 17) % 30 - 14.5) / 120 for index in range(30)]
            ax.scatter([x + offset for offset in offsets], values, s=24,
                       color=color, alpha=0.63, edgecolor="none", zorder=2)
            ax.scatter([x], [mean(values)], s=85, color="black", marker="D", zorder=4)
        ax.axhline(0, color="#444444", linewidth=1.2)
        ax.set_xlim(-0.38, 1.38)
        ax.set_xticks((0, 1), CLASSIFIERS)
        ax.set_title(f"{title}\nMeans: LogReg {means['LogReg']:+.3f}; GNB {means['GNB']:+.3f}",
                     weight="bold")
        ax.set_ylabel("SSHR with U − SSHR without U")
        ax.text(0.5, -0.19, explanation, transform=ax.transAxes, ha="center",
                fontsize=8.5, color="#555555")
        style(ax)
    fig.suptitle("What did the unlabelled pool change for SSHR?", fontsize=15, weight="bold")
    fig.text(0.5, 0.89, "Each dot is one paired dataset-variant effect; black diamonds are 30-variant means.",
             ha="center", fontsize=9.5)
    fig.text(0.5, 0.025, "Same labelled pool and test folds; only access to training-only U changes. Descriptive effects.",
             ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(top=0.77, bottom=0.25, left=0.10, right=0.98, wspace=0.22)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--ablation", type=Path, default=ABLATION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    table, ablation, names = load_data(args.table, args.ablation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        ("01_f1_fpr_tradeoff", tradeoff_figure(table, names)),
        ("02_f1_consistency", consistency_figure(table, names)),
        ("03_sshr_unlabelled_effect", ablation_figure(ablation, names)),
    )
    with PdfPages(args.output_dir / "Imbalanced_Learning_30_Dataset_Figures.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(args.output_dir / f"{name}.png", dpi=180, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    print(f"Saved three PNGs and a three-page PDF to {args.output_dir}")


if __name__ == "__main__":
    main()
