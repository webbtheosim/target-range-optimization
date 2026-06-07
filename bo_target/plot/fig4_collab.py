"""Figure 4: cross-target hit ratio across datasets and tolerance ratios."""

import pickle
from pathlib import Path

import numpy as np
import ultraplot as pplt

from bo_target.plot.color_wheel import color_wheel
from bo_target.plot.format_ax import format_ax

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "bo_fig"

DATA_PATH = REPO / "bo_target" / "data" / "analysis" / "cross_target_hit_ratio.pkl"

DATASETS = ["branin", "qm9", "nanoparticle", "lipo"]
TOLS = [0.2, 0.4, 0.6]
N_SEEDS = 10


def load_data():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def plot_collab_panel(ax, runs, colors):
    """Plot cross-target hit ratio vs budget for one dataset/tolerance combination."""
    for k, d in enumerate(runs):
        x = np.concatenate([[0], d["budget"]])
        y = np.concatenate([[0], d["mean"]])
        sem = d["std"] / np.sqrt(N_SEEDS)
        y_low = np.concatenate([[0], d["mean"] - sem])
        y_high = np.concatenate([[0], d["mean"] + sem])

        ax.plot(x, y, label=d["short"], lw=1.0, color=colors[k], zorder=10 - k)
        ax.fill_between(x, y_low, y_high, color=colors[k], alpha=0.12)


def main():
    all_results = load_data()
    colors = color_wheel()

    fig, axs = pplt.subplots(
        nrows=len(DATASETS), ncols=len(TOLS),
        journal="nat1", sharex=True, sharey=True,
    )

    for i, ds in enumerate(DATASETS):
        for j, tol in enumerate(TOLS):
            ax = axs[i, j]
            plot_collab_panel(ax, all_results[ds][tol], colors)
            ax.format(
                xticks=[0, 125, 250],
                xticklabels=["0", "25", "50"],
                xlim=(0, 250),
                ylim=(0, 1.0),
                yticks=(0, 0.5, 1.0),
                xlabel="Iteration" if i == len(DATASETS) - 1 else "",
                ylabel="Cross-Target Hit Ratio" if j == 0 else "",
            )
            format_ax(ax)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="t", ncols=6,
            frame=True, fontsize=8, handlelength=1.2, handletextpad=0.4,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig4_collab.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_collab.svg", bbox_inches="tight")
    pplt.show()


if __name__ == "__main__":
    main()