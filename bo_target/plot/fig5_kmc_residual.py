"""Figure 5: KMC target spectra and residuals for top-10 discovered molecules.

Plot data is precomputed by ``bo_target/utils/run_fig5_kmc.py`` into
``data/analysis/fig5_kmc_residual.pkl``; this script only renders it.
"""

import pickle
from pathlib import Path

import numpy as np
import ultraplot as pplt

from bo_target.plot.color_wheel import color_wheel
from bo_target.plot.format_ax import format_ax

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "bo_fig"
DATA_PATH = REPO / "bo_target" / "data" / "analysis" / "fig5_kmc_residual.pkl"

N_TARGETS = 5
N_EXAMPLES = 10


def _load():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def _corner_label(ax, i):
    ax.text(
        0.05, 0.95, f"$T_{i + 1}$",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )


def plot_kmc_rows(axs, XX, targets, discovered_curves, colors):
    """Top row: target + discovered spectra. Bottom row: residuals."""
    cc = pplt.Colormap("Reds")(np.linspace(0, 1, N_EXAMPLES))
    
    XX = XX + np.log10(104.15) # g/mol

    for i in range(N_TARGETS):
        axs[i].plot(XX, targets[i], c=colors[0], lw=2, zorder=0,
                    label="Target" if i == 0 else None)
        axs[i].format(
            xlim=[2, 7], ylim=[-0.01, 0.2],
            xticks=[2, 3, 4, 5, 6, 7],
            xlabel=r"$\log_{10}$ Molecular Weight $\mathit{M}$ (g mol$^{-1}$)",
            ylabel="Weight Fraction",
        )
        format_ax(axs[i])
        _corner_label(axs[i], i)

        residuals = (discovered_curves[i] - targets[i]).T
        n_curves = residuals.shape[1]
        for j in range(n_curves):
            axs[i + N_TARGETS].plot(
                XX, residuals[:, j], c=cc[j], alpha=1, zorder=1, lw=1,
                label="Residuals" if (i == 0 and j == n_curves - 1) else None,
            )
        axs[i + N_TARGETS].format(
            xlim=[2, 7], ylim=[-0.1, 0.1],
            xticks=[2, 3, 4, 5, 6, 7],
            xlabel=r"$\log_{10}$ Molecular Weight $\mathit{M}$ (g mol$^{-1}$)",
            ylabel="Residual",
        )
        format_ax(axs[i + N_TARGETS])
        _corner_label(axs[i + N_TARGETS], i)

    axs[0].legend(loc="upper right", fontsize=8)
    axs[N_TARGETS].legend(loc="upper right", fontsize=8)


def main():
    data = _load()
    colors = color_wheel()

    fig, axs = pplt.subplots(
        ncols=N_TARGETS, nrows=2, journal="nat2",
        refaspect=1.25, hspace=1.8, wspace=1.8,
    )

    plot_kmc_rows(axs, data["XX"], data["targets"], data["discovered_curves"], colors)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig5_kmc_residual.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_kmc_residual.svg", dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()
    pplt.show()
