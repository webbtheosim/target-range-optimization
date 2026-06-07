"""Figure 5: UMAP projection of KMC design space with valid-candidate hits.

Plot data (precomputed UMAP coordinates) is built by
``bo_target/utils/run_fig5_kmc.py`` into ``data/analysis/fig5_kmc_umap.pkl``;
this script only renders it, so no ``umap`` dependency is needed here.
"""

import pickle
from pathlib import Path

import ultraplot as pplt
from matplotlib.lines import Line2D

from bo_target.plot.format_ax import format_ax

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "bo_fig"
DATA_PATH = REPO / "bo_target" / "data" / "analysis" / "fig5_kmc_umap.pkl"

N_TARGETS = 5
BG_COLOR = "#C7CDD4"
HIT_COLOR = "#E4256D"


def _load():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def plot_umap_panels(axs, valid_scaled, hits_scaled):
    """UMAP scatter: full design space in gray, valid-candidate hits in accent color."""
    for i in range(N_TARGETS):
        axs[i].scatter(
            valid_scaled[:, 0], valid_scaled[:, 1],
            s=4, alpha=0.45, linewidth=0, zorder=1,
            c=BG_COLOR, rasterized=True,
            label="Full design space" if i == 0 else None,
        )
        hits = hits_scaled[i]
        axs[i].scatter(
            hits[:, 0], hits[:, 1],
            s=13, zorder=3, c=HIT_COLOR,
            edgecolor="w", linewidth=0.4,
            label="Valid candidates" if i == 0 else None,
        )
        axs[i].format(
            xlabel=r"$Z_1$",
            ylabel=r"$Z_2$" if i == 0 else "",
        )
        format_ax(axs[i])
        axs[i].text(
            0.05, 0.95, f"$T_{i + 1}$",
            transform=axs[i].transAxes, ha="left", va="top",
            fontsize=9, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )


def main():
    data = _load()

    fig, axs = pplt.subplots(
        ncols=N_TARGETS, nrows=1, journal="nat2", hspace=1.8, wspace=1.8,
    )

    plot_umap_panels(axs, data["valid_scaled"], data["hits_scaled"])

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=BG_COLOR, markeredgecolor="none",
               label="Full design space"),
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=HIT_COLOR, markeredgecolor="w", markeredgewidth=0.5,
               label="Valid candidates"),
    ]
    fig.legend(
        legend_handles, [h.get_label() for h in legend_handles],
        loc="t", ncols=2, frame=True, fontsize=8,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig5_kmc_umap.png", dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()
    pplt.show()
