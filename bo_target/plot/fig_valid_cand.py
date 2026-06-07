"""Valid-candidate curves per target over iterations for KMC and TDDFT datasets."""

import pickle
from pathlib import Path

import numpy as np
import ultraplot as pplt

from bo_target.plot.color_wheel import color_wheel
from bo_target.plot.format_ax import format_ax

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "bo_fig"
DATA_PATH = REPO / "bo_target" / "data" / "analysis" / "case_vs_budget.pkl"

SHORT_NAMES = ["TB", "HV", "EI", "LCB", "BAX", "RS"]
N_METHODS = len(SHORT_NAMES)
N_TARGETS = 5
BUDGET_DIV = 5


def load_data():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def _plot_panel(ax, results, dataset, tol, target_idx, colors, xlim, xticks):
    runs = [results[dataset][tol][j] for j in range(N_METHODS)]
    x = runs[0]["budget"] / BUDGET_DIV

    for j, d in enumerate(runs):
        ax.plot(x, d["mean"][:, target_idx], color=colors[j], lw=1.5,
                label=SHORT_NAMES[j] if target_idx == 0 else None)

    y_all = np.stack([d["mean"][:, target_idx] for d in runs])
    leader = np.argmax(y_all, axis=0)
    change_idx = np.where(np.diff(leader) != 0)[0] + 1
    starts = np.concatenate([[0], change_idx])
    ends = np.concatenate([change_idx, [len(leader)]])
    for s, e in zip(starts, ends):
        if e > s:
            ax.axvspan(x[s], x[e - 1], color=colors[leader[s]], alpha=0.13, zorder=-1)

    ax.format(
        ylabel="Number of Valid\nCandidates" if target_idx == 0 else "",
        xlabel="Iteration",
        xlim=xlim,
        xticks=xticks,
    )
    format_ax(ax)
    # target label inside the top-left corner
    ax.text(
        0.05, 0.95, f"$T_{target_idx + 1}$",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, zorder=5,
    )


def plot_row(results, dataset, tol, xlim, xticks, colors):
    fig, axs = pplt.subplots(ncols=N_TARGETS, nrows=1, journal="nat2",
                              hspace=1.8, wspace=1.8)
    for i in range(N_TARGETS):
        _plot_panel(axs[i], results, dataset, tol, i, colors, xlim, xticks)

    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncols=len(handles), loc="t", frame=True, fontsize=8)
    return fig


def main():
    results = load_data()
    colors = color_wheel()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_kw = dict(dpi=300, bbox_inches="tight")

    fig_kmc = plot_row(results, "kmc", 0.4, xlim=[0, 200], xticks=[0, 100, 200], colors=colors)
    fig_kmc.savefig(OUT_DIR / "fig_valid_cand_kmc.png", **save_kw)
    fig_kmc.savefig(OUT_DIR / "fig_valid_cand_kmc.svg", **save_kw)

    fig_tddft = plot_row(results, "tddft", 0.6, xlim=[0, 50], xticks=[0, 25, 50], colors=colors)
    fig_tddft.savefig(OUT_DIR / "fig_valid_cand_tddft.png", **save_kw)
    fig_tddft.savefig(OUT_DIR / "fig_valid_cand_tddft.svg", **save_kw)

    pplt.show()


if __name__ == "__main__":
    main()
