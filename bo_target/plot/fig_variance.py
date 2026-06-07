"""Normalized variance bar chart across acquisition methods and tolerance ratios."""

import pickle
from pathlib import Path

import numpy as np
import ultraplot as pplt

from bo_target.plot.color_wheel import color_wheel
from bo_target.plot.format_ax import format_ax

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "bo_fig"
DATA_PATH = REPO / "bo_target" / "data" / "analysis" / "stats_results.pkl"

TOLS = [0.2, 0.4, 0.6]
N_SEEDS = 10


def load_data():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def plot_variance_panel(ax, results, tol, colors):
    dataset_names = list(results["stats"].keys())
    acquisition_names = list(results["stats"][dataset_names[0]][tol].keys())
    n_datasets = len(dataset_names)

    var_per_acq = np.array([
        [results["stats"][ds][tol][acq]["avg_var_targets"] for ds in dataset_names]
        for acq in acquisition_names
    ])

    col_max = np.max(var_per_acq, axis=0, keepdims=True)
    safe_max = np.where(col_max == 0, 1.0, col_max)
    var_norm = var_per_acq / safe_max
    mean_vals = var_norm.mean(axis=1)
    sem_vals = var_norm.std(axis=1, ddof=0) / np.sqrt(n_datasets)

    x = np.arange(len(mean_vals))
    ax.bar(x, mean_vals, yerr=sem_vals, width=0.6, color=colors[0],
           edgecolor="k", capsize=4, alpha=0.95)
    ax.format(
        xticks=x,
        xticklabels=[temp.upper() for temp in acquisition_names],
        ylabel="Avg. Normalized Variance",
        ylim=(0, 1.0),
        xlim=(-0.5, len(acquisition_names) - 0.5),
    )
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    format_ax(ax)
    ax.text(0.97, 0.97, rf"$\mathit{{r}} = {tol}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)


def main():
    results = load_data()
    colors = color_wheel()

    fig, axs = pplt.subplots(ncols=len(TOLS), journal="nat2", refaspect=1.5)

    for idx, tol in enumerate(TOLS):
        plot_variance_panel(axs[idx], results, tol, colors)

    for idx in range(1, len(TOLS)):
        axs[idx].format(ylabel="")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_variance.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_variance.svg", dpi=300, bbox_inches="tight")
    pplt.show()


if __name__ == "__main__":
    main()
