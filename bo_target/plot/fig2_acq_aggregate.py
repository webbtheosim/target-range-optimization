"""Figure 2: acquisition performance aggregate -- average rank, best/worst fractions."""

import pickle
from colorsys import hls_to_rgb, rgb_to_hls
from pathlib import Path

import numpy as np
import ultraplot as pplt

from bo_target.plot.color_wheel import color_wheel
from bo_target.plot.format_ax import format_ax

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "bo_fig"

DATA_PATH = REPO / "bo_target" / "data" / "analysis" / "ranking_results.pkl"


def _generate_color_gradient(base_hex, n_colors):
    hexc = base_hex.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, _, s = rgb_to_hls(r, g, b)
    return [
        "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in hls_to_rgb(h, lum, s)))
        for lum in np.linspace(0.85, 0.15, n_colors)
    ]


def load_data():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    return (
        data["ranks_array"],
        data["short_names"],
        data["dataset_names"],
        data["tolerance_ratios"],
    )


def plot_avg_rank(ax, ranks_array, short_names, dataset_names, tolerance_ratios, tol_colors):
    """Average rank across all datasets, grouped by tolerance ratio."""
    n_methods = len(short_names)
    n_datasets = len(dataset_names)
    n_tols = len(tolerance_ratios)

    ranks_reshaped = ranks_array.reshape(n_datasets, n_tols, n_methods)
    avg_by_tol = ranks_reshaped.mean(axis=0)
    sem_by_tol = ranks_reshaped.std(axis=0, ddof=0) / np.sqrt(n_datasets)

    order = np.argsort(avg_by_tol.mean(axis=0))
    x = np.arange(n_methods)
    width = 0.18

    for i in range(n_tols):
        offset = (i - (n_tols - 1) / 2) * width
        ax.bar(
            x + offset,
            avg_by_tol[i, order],
            width,
            yerr=sem_by_tol[i, order],
            color=tol_colors[i],
            edgecolor="k",
            alpha=0.95,
            capsize=3,
            label=f"{tolerance_ratios[i]:.1f}",
        )

    ax.format(
        xticks=x,
        xticklabels=[short_names[j] for j in order],
        ylabel="Average Rank",
        ylim=[0, 5.1],
    )
    ax.legend(ncols=3, loc="upper left", title=r"$\bf{Tolerance~Ratio~}\it{r}$")
    format_ax(ax)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    # Print values
    print("\n=== Average Rank ===")
    print(f"{'Method':>6} " + " ".join(f"r={t:.1f}  " for t in tolerance_ratios) + "  Mean")
    for j in order:
        vals = " ".join(f"{avg_by_tol[i, j]:.3f}" for i in range(n_tols))
        mean_val = avg_by_tol[:, j].mean()
        print(f"{short_names[j]:>6} {vals}  {mean_val:.3f}")


def plot_best_ratio(ax, ranks_array, short_names, dataset_names, tolerance_ratios, tol_colors):
    """Fraction of datasets where method achieved best rank, grouped by tolerance."""
    n_methods = len(short_names)
    n_datasets = len(dataset_names)
    n_tols = len(tolerance_ratios)

    best_reshaped = (ranks_array == 0).reshape(n_datasets, n_tols, n_methods)
    best_by_tol = best_reshaped.mean(axis=0)
    sem_best = np.sqrt(best_by_tol * (1 - best_by_tol) / n_datasets)

    order = np.argsort(best_by_tol.mean(axis=0))[::-1]
    x = np.arange(n_methods)[:4]
    width = 0.27

    for i in range(n_tols):
        offset = (i - (n_tols - 1) / 2) * width
        ax.bar(
            x + offset,
            best_by_tol[i, order][:4],
            width,
            yerr=sem_best[i, order][:4],
            color=tol_colors[i],
            edgecolor="k",
            alpha=0.95,
            capsize=3,
            label=f"{tolerance_ratios[i]:.1f}",
        )

    ax.format(
        xticks=x,
        xticklabels=[short_names[j] for j in order],
        ylabel="Best Performance Fraction",
        ylim=[0, 1.05],
    )
    format_ax(ax)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    # Print values
    print("\n=== Best Performance Fraction ===")
    print(f"{'Method':>6} " + " ".join(f"r={t:.1f}  " for t in tolerance_ratios) + "  Mean")
    for j in order:
        vals = " ".join(f"{best_by_tol[i, j]:.3f}" for i in range(n_tols))
        mean_val = best_by_tol[:, j].mean()
        print(f"{short_names[j]:>6} {vals}  {mean_val:.3f}")


def plot_worst_ratio(ax, ranks_array, short_names, dataset_names, tolerance_ratios, tol_colors):
    """Fraction of datasets where method achieved worst rank, grouped by tolerance."""
    n_methods = len(short_names)
    n_datasets = len(dataset_names)
    n_tols = len(tolerance_ratios)

    worst_reshaped = (ranks_array == 5).reshape(n_datasets, n_tols, n_methods)
    worst_by_tol = worst_reshaped.mean(axis=0)
    sem_worst = np.sqrt(worst_by_tol * (1 - worst_by_tol) / n_datasets)

    order = np.argsort(worst_by_tol.mean(axis=0))[::-1]
    x = np.arange(n_methods)[:3]
    width = 0.2

    for i in range(n_tols):
        offset = (i - (n_tols - 1) / 2) * width
        ax.bar(
            x + offset,
            worst_by_tol[i, order][:3],
            width,
            yerr=sem_worst[i, order][:3],
            color=tol_colors[i],
            edgecolor="k",
            alpha=0.95,
            capsize=3,
            label=f"{tolerance_ratios[i]:.1f}",
        )

    ax.format(
        xticks=x,
        xticklabels=[short_names[j] for j in order],
        ylabel="Worst Performance Fraction",
        ylim=[0, 1.05],
    )
    format_ax(ax)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    # Print values
    print("\n=== Worst Performance Fraction ===")
    print(f"{'Method':>6} " + " ".join(f"r={t:.1f}  " for t in tolerance_ratios) + "  Mean")
    for j in order:
        vals = " ".join(f"{worst_by_tol[i, j]:.3f}" for i in range(n_tols))
        mean_val = worst_by_tol[:, j].mean()
        print(f"{short_names[j]:>6} {vals}  {mean_val:.3f}")


def main():
    ranks_array, short_names, dataset_names, tolerance_ratios = load_data()
    colors = color_wheel()
    tol_colors = _generate_color_gradient(colors[0], len(tolerance_ratios))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_kw = dict(dpi=300, bbox_inches="tight")

    fig1, ax1 = pplt.subplots(journal="nat2", refheight=2)
    plot_avg_rank(ax1, ranks_array, short_names, dataset_names, tolerance_ratios, tol_colors)
    fig1.savefig(OUT_DIR / "fig2_avg_rank.png", **save_kw)
    fig1.savefig(OUT_DIR / "fig2_avg_rank.svg", **save_kw)

    fig2, ax2 = pplt.subplots(journal="nat1", refheight=2)
    plot_best_ratio(ax2, ranks_array, short_names, dataset_names, tolerance_ratios, tol_colors)
    fig2.savefig(OUT_DIR / "fig2_best_ratio.png", **save_kw)
    fig2.savefig(OUT_DIR / "fig2_best_ratio.svg", **save_kw)

    fig3, ax3 = pplt.subplots(journal="nat1", refheight=2)
    plot_worst_ratio(ax3, ranks_array, short_names, dataset_names, tolerance_ratios, tol_colors)
    fig3.savefig(OUT_DIR / "fig2_worst_ratio.png", **save_kw)
    fig3.savefig(OUT_DIR / "fig2_worst_ratio.svg", **save_kw)

    pplt.show()


if __name__ == "__main__":
    main()