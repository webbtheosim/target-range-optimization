"""Figure 3: Branin and QM9 acquisition trajectories with target bands.

Plot data is precomputed by ``bo_target/utils/run_fig3.py`` into
``data/analysis/fig3_branin_qm9.pkl``; this script only renders it.
"""

import pickle
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import ultraplot as pplt

from bo_target.plot.color_wheel import color_wheel
from bo_target.plot.format_ax import format_ax

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "bo_fig"
DATA_PATH = REPO / "bo_target" / "data" / "analysis" / "fig3_branin_qm9.pkl"

ACQ_NAMES = ("TB", "HV", "LCB", "BAX")


def _load():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def plot_branin_row(axs, X1, X2, Z, targets, tolerance, X_list, colors):
    """Branin landscape with acquisition points overlaid (top row)."""
    nT = len(np.atleast_1d(targets))

    for j, (name, X_dis) in enumerate(zip(ACQ_NAMES, X_list)):
        a = axs[0, j]
        pc = a.pcolor(X1, X2, Z, cmap="cividis", shading="auto", alpha=0.85)

        for t in np.atleast_1d(targets):
            a.contour(X1, X2, Z, levels=[t], colors="w", linestyles="--", linewidths=1)
            a.contour(X1, X2, Z, levels=[t - tolerance], colors="w", linewidths=0.6)
            a.contour(X1, X2, Z, levels=[t + tolerance], colors="w", linewidths=0.6)

        for i in range(nT):
            a.scatter(
                X_dis[i::nT, 0], X_dis[i::nT, 1],
                s=3, zorder=10 - i, c=colors[i % len(colors)],
            )

        a.format(
            xlabel=r"$x_1$", ylabel=r"$x_2$", title=name,
            xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        format_ax(a)

    return pc


def plot_qm9_row(axs, Y_all_orig, targets_orig, epsilon_orig, Y_list_orig, colors):
    """QM9 HOMO-LUMO space with acquisition points overlaid (bottom row)."""
    nT = len(targets_orig)

    for j, (name, Y_dis) in enumerate(zip(ACQ_NAMES, Y_list_orig)):
        a = axs[1, j]
        _, _, _, im = a.hist2d(
            Y_all_orig[:, 0], Y_all_orig[:, 1],
            bins=60, cmap="Blues", density=True, alpha=0.7, zorder=0,
        )

        for t in targets_orig:
            a.add_patch(mpatches.Ellipse(
                (t[0], t[1]),
                width=2 * epsilon_orig[0], height=2 * epsilon_orig[1],
                fill=False, lw=1, color="k", zorder=11,
            ))

        for i in range(nT):
            a.scatter(
                Y_dis[i::nT, 0], Y_dis[i::nT, 1],
                s=3, zorder=i + 1, c=colors[i % len(colors)],
            )

        a.format(
            xlabel="HOMO (eV)", ylabel="LUMO (eV)",
            xlim=[-0.3, -0.15], ylim=[-0.15, 0.15],
            xrotation=45,
            title=name,
        )
        format_ax(a)

    return im


def main():
    colors = color_wheel()
    data = _load()

    b = data["branin"]
    X1, X2, Z = b["X1"], b["X2"], b["Z"]
    targets_b, tolerance_b, X_list = b["targets"], b["tolerance"], b["X_list"]

    q = data["qm9"]
    Y_all_orig, targets_q = q["Y_all_orig"], q["targets_orig"]
    epsilon_q, Y_list_orig = q["epsilon_orig"], q["Y_list_orig"]

    fig, axs = pplt.subplots(
        nrows=2, ncols=4, journal="nat2",
        sharex=False, sharey=False, spanx=True,
        hspace=3.0, wspace=2.0, refaspect=1,
    )

    pc = plot_branin_row(axs, X1, X2, Z, targets_b, tolerance_b, X_list, colors)
    im = plot_qm9_row(axs, Y_all_orig, targets_q, epsilon_q, Y_list_orig, colors)

    for col in range(1, 4):
        axs[0, col].format(ylabel="", yticklabels=False)
        axs[1, col].format(ylabel="", yticklabels=False)

    cb_kw = dict(width="0.8em", labelsize=9, ticklabelsize=8)
    if pc is not None:
        fig.colorbar(pc, ax=axs[0, -1], loc="r", label=r"$f(\mathbf{x})$", **cb_kw)
    if im is not None:
        fig.colorbar(im, ax=axs[1, -1], loc="r", label="Density", **cb_kw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig3_branin_qm9.png", dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    main()
    pplt.show()
