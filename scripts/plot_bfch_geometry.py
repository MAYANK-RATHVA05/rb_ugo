"""Save a two-dimensional BFCH-SMOTE software illustration (not a result)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from src.bfch_smote import BFCHSMOTE


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    minority = np.array([[0., 0.], [0., 2.]])
    majority = np.array([[4., 0.], [4., 2.], [5., -1.],
                         [5., 1.], [5., 3.], [6., 1.]])
    return np.vstack((minority, majority)), np.r_[np.ones(2, dtype=int),
                                                   np.zeros(6, dtype=int)]


def _panel(ax, X: np.ndarray, y: np.ndarray, fusion: float) -> dict:
    sampler = BFCHSMOTE(k_neighbors=2, theta_min=0, theta_maj=0,
                        fusion=fusion, n_samples=25, random_state=42)
    augmented, _ = sampler.fit_resample(X, y)
    minority, majority = X[y == 1], X[y == 0]
    ax.scatter(minority[:, 0], minority[:, 1], s=90, color="#2166ac",
               edgecolor="white", linewidth=1, zorder=5, label="Original minority")
    ax.scatter(majority[:, 0], majority[:, 1], s=90, marker="x", color="#b2182b",
               linewidth=2, zorder=5, label="Original majority")
    boundary = sampler.boundary_indices_
    for kind, color in (("minority", "#2166ac"), ("majority", "#b2182b")):
        selected = X[boundary[kind]]
        ax.scatter(selected[:, 0], selected[:, 1], s=165, facecolors="none",
                   edgecolors=color, linewidth=1.5, zorder=4,
                   label=f"{kind.capitalize()} boundary")
    for candidate in sampler.candidate_hulls_:
        accepted = candidate["status"] == "accepted"
        color = "#1a9850" if accepted else "#d73027"
        ax.add_patch(Polygon(candidate["vertices"], closed=True,
                             facecolor=color if accepted else "none", alpha=0.25 if accepted else 1,
                             edgecolor=color, linewidth=2, linestyle="-" if accepted else "--",
                             label="Accepted triangle" if accepted else "Rejected triangle"))
        midpoint, fused = candidate["midpoint"], candidate["fusion_point"]
        ax.scatter(midpoint[0], midpoint[1], marker="*", s=130, color="#555555",
                   zorder=6, label="Minority midpoint")
        ax.scatter(fused[0], fused[1], marker="D", s=55, color="#762a83",
                   zorder=6, label="Fusion point")
    synthetic = augmented[len(X):]
    if len(synthetic):
        ax.scatter(synthetic[:, 0], synthetic[:, 1], s=22, color="#1a9850",
                   alpha=0.85, zorder=3, label="Generated minority")
    ax.set_title(f"Fusion L={fusion:g}: {sampler.diagnostics_['status'].replace('_', ' ')}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_xlim(-0.8, 6.8)
    ax.set_ylim(-1.8, 3.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    return sampler.diagnostics_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/bfch-geometry-check.png"))
    args = parser.parse_args()
    X, y = _fixture()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True,
                                constrained_layout=True)
    accepted = _panel(axes[0], X, y, 0.5)
    rejected = _panel(axes[1], X, y, 1.0)
    if (accepted["status"] != "generated" or accepted["generated_count"] != 25
            or rejected["status"] != "no_valid_hulls"
            or rejected["rejected_occupied"] != 1):
        raise RuntimeError("The artificial geometry no longer demonstrates the expected cases.")
    handles, labels = axes[0].get_legend_handles_labels()
    other_handles, other_labels = axes[1].get_legend_handles_labels()
    for handle, label in zip(other_handles, other_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle("BFCH-SMOTE artificial geometry check (not performance evidence)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Saved {args.output.resolve()}")
    print("L=0.5: 1 accepted triangle, 25 generated minority points")
    print("L=1: 1 rejected occupied triangle, 0 generated points")


if __name__ == "__main__":
    main()
