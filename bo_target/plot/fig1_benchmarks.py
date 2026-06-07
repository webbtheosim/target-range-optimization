"""Figure 1: synthetic benchmark landscapes with target contours."""

import json
from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
import ultraplot as pplt

from bo_target.data._paths import base_path
from bo_target.plot.format_ax import format_ax

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "bo_fig"

CONFIG_DIR = base_path / "config"


def _load_config(dataset_name):
    """Load target values and epsilon from dataset config JSON."""
    path = CONFIG_DIR / f"{dataset_name}_config.json"
    with open(path) as f:
        config = json.load(f)
    return np.array(config["target"]), config["epsilon"]


def plot_benchmark_slice(ax, dataset_name="branin", axes=(0, 1), fixed_val=0.5):
    """Plot a 2D slice of a synthetic function with target contours."""
    from bo_target.utils.dataset_config import dataset_config

    targets, epsilon = _load_config(dataset_name)

    f, space, _ = dataset_config(dataset_name, verbose=False, testing=True)[:3]

    bounds = space.get_bounds()
    ndim = len(bounds)

    if len(axes) != 2 or axes[0] >= ndim or axes[1] >= ndim or axes[0] == axes[1]:
        raise ValueError(f"Invalid axes {axes} for {ndim}-dimensional space.")

    i, j = axes
    x1 = np.linspace(0, 1, 200)
    x2 = np.linspace(0, 1, 200)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.full((X1.size, ndim), fixed_val)
    X_grid[:, i] = X1.ravel()
    X_grid[:, j] = X2.ravel()

    Z = f(X_grid).ravel().reshape(X1.shape)

    ax.pcolor(X1, X2, Z, cmap="cividis", shading="auto", alpha=0.8)
    cs = ax.contour(
        X1, X2, Z,
        levels=np.atleast_1d(targets).squeeze(),
        colors="black", linewidths=1.0,
    )
    cs.set(path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    labels = ax.clabel(cs, inline=True, fontsize=7, fmt="%0.2f")
    for t in labels:
        t.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    format_ax(ax)
    ax.format(
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )


def main():
    fig, axs = pplt.subplots(
        ncols=4, nrows=3, journal="nat2",
        refaspect=1.0, hspace=2.0, wspace=2.0,
    )

    plot_benchmark_slice(axs[0], "branin", axes=(0, 1))
    plot_benchmark_slice(axs[1], "hartmann", axes=(0, 1))
    plot_benchmark_slice(axs[2], "hartmann", axes=(1, 2))
    plot_benchmark_slice(axs[3], "hartmann", axes=(0, 2))
    plot_benchmark_slice(axs[4], "ackley", axes=(0, 1), fixed_val=0.2)
    plot_benchmark_slice(axs[5], "ackley", axes=(0, 1), fixed_val=0.3)
    plot_benchmark_slice(axs[6], "ackley", axes=(0, 1), fixed_val=0.4)
    plot_benchmark_slice(axs[7], "ackley", axes=(0, 1), fixed_val=0.5)
    plot_benchmark_slice(axs[8], "layeb06", axes=(0, 1), fixed_val=0.1)
    plot_benchmark_slice(axs[9], "layeb06", axes=(1, 2), fixed_val=0.2)
    plot_benchmark_slice(axs[10], "layeb06", axes=(2, 3), fixed_val=0.3)
    plot_benchmark_slice(axs[11], "layeb06", axes=(3, 4), fixed_val=0.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig1_benchmarks.png", dpi=300, bbox_inches="tight")
    pplt.show()


if __name__ == "__main__":
    main()