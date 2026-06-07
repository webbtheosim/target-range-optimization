"""Per-target AUC radar plots across all datasets, tolerance ratios, and acquisitions."""

import pickle
from pathlib import Path

import numpy as np
import ultraplot as pplt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = REPO / "bo_fig"
DATA_PATH = REPO / "bo_target" / "data" / "analysis" / "radar_aucs.pkl"

_AUCS = None


def _aucs(dataset_name, tolerance_ratio, acquisition_name):
    """Per-target AUC array (n_targets x n_runs), precomputed by run_radar_aucs.py."""
    global _AUCS
    if _AUCS is None:
        with open(DATA_PATH, "rb") as f:
            _AUCS = pickle.load(f)["aucs"]
    arr = _AUCS.get(dataset_name, {}).get(tolerance_ratio, {}).get(acquisition_name)
    return arr if arr is not None else np.empty((0, 0))


def plot_radar_aucs_all(
    dataset_name="toporg",
    tolerance_ratios=[0.2, 0.4, 0.6],
    acquisition_names=None,
    short_names=None,
    colors=None,
    stop_iter=250,
    testing=True,
    defined_ymax=None,
):

    if acquisition_names is None or short_names is None or colors is None:
        raise ValueError(
            "acquisition_names, short_names, and colors must be provided"
        )

    # Load all AUC arrays up front
    all_data = []
    for tolerance_ratio in tolerance_ratios:
        row_aucs = [
            _aucs(dataset_name, tolerance_ratio, name)
            for name in acquisition_names
        ]
        all_data.append(row_aucs)

        print(f"\n=== SHAPES for tolerance_ratio = {tolerance_ratio} ===")
        for name, arr in zip(acquisition_names, row_aucs):
            shape = arr.shape if arr.size > 0 else "empty (0)"
            print(f" {name:25s} -> {shape}")

    valid_all = [arr for row in all_data for arr in row if arr.size > 0]
    if not valid_all:
        print("All arrays empty -> skipping")
        return

    n_rows = len(tolerance_ratios)
    n_cols = len(acquisition_names)

    fig, axs = pplt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        journal="nat2",
        proj="polar",
        wspace=1.3,
        hspace=2.0,
    )

    row_labels = ["a", "b", "c"]

    for row_idx, (tolerance_ratio, row_aucs) in enumerate(
        zip(tolerance_ratios, all_data)
    ):
        valid_row = [arr for arr in row_aucs if arr.size > 0]
        if not valid_row:
            continue
        row_max = max(
            (arr.mean(axis=1) + arr.std(axis=1) / np.sqrt(arr.shape[1])).max()
            for arr in valid_row
        )
        rmax = (row_max if row_max > 0 else 1.0) * 1.05
        if defined_ymax is not None:
            rmax = defined_ymax

        for col_idx, (name, arr) in enumerate(
            zip(acquisition_names, row_aucs)
        ):
            ax = axs[row_idx, col_idx]

            if arr.size == 0:
                ax.format(
                    title=f"{short_names[col_idx]}\n(empty)",
                    titleweight="bold",
                )
                continue

            n_targets, n_runs = arr.shape
            means = arr.mean(axis=1)
            sems = arr.std(axis=1) / np.sqrt(n_runs)

            geo_per_run = np.prod(arr, axis=0) ** (1.0 / n_targets)
            geo_mean = geo_per_run.mean()
            geo_sem = geo_per_run.std() / np.sqrt(n_runs)

            arith_per_run = arr.mean(axis=0)
            arith_mean = arith_per_run.mean()
            arith_sem = arith_per_run.std() / np.sqrt(n_runs)

            theta_deg = np.linspace(0, 360, n_targets, endpoint=False)
            angles_rad = np.deg2rad(theta_deg)
            angles_closed = np.deg2rad(np.append(theta_deg, theta_deg[0]))

            target_labels = [rf"$T_{j + 1}$" for j in range(n_targets)]
            col = colors[0]

            values_closed = np.append(means, means[0])
            upper_closed = np.append(means + sems, (means + sems)[0])
            lower_closed = np.maximum(
                np.append(means - sems, (means - sems)[0]),
                0.0,
            )

            ax.fill_between(
                angles_closed,
                lower_closed,
                upper_closed,
                color=col,
                alpha=0.15,
            )
            ax.fill(
                angles_closed,
                values_closed,
                color=col,
                alpha=0.4,
            )
            ax.plot(
                angles_closed,
                values_closed,
                color=col,
                linewidth=1.26,
            )
            ax.errorbar(
                angles_rad,
                means,
                yerr=sems,
                fmt="none",
                ecolor="k",
                elinewidth=1,
                capsize=2,
                capthick=1,
            )

            ax.format(
                title=short_names[col_idx],
                titleweight="bold",
                titlesize=8,
                titlepad=0.8,
                ticklabelsize=7,
                rlim=(0, rmax),
                rgrid=True,
                thetagrid=True,
                thetalines=theta_deg,
                thetalabels=(target_labels, theta_deg),
                linewidth=1.0,
            )
            ax.set_rlabel_position(30)
            if not col_idx == 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="x", pad=-2)
            ax.text(
                np.deg2rad(55),
                rmax * 1.05,
                f"{geo_mean:.2f}",
                ha="left",
                va="bottom",
                fontsize=9,
                color="k",
                clip_on=False,
                fontweight="bold",
            )

            print(
                f"{short_names[col_idx]}_{tolerance_ratio}: "
                f"arith-mean AUC = {arith_mean:.4f} +/- {arith_sem:.4f} | "
                f"geo-mean AUC = {geo_mean:.4f} +/- {geo_sem:.4f} "
                f"({n_targets} targets, {n_runs} runs)"
            )

        # Bold row label (a, b, c) at top-left of the first column
        if row_idx < len(row_labels):
            ax0 = axs[row_idx, 0]
            ax0.text(
                -0.15,
                1.15,
                row_labels[row_idx],
                transform=ax0.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                ha="left",
                clip_on=False,
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / f"fig_radar_aucs_{dataset_name}_all.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / f"fig_radar_aucs_{dataset_name}_all.svg",
        bbox_inches="tight",
    )

    pplt.close()


if __name__ == "__main__":
    from bo_target.plot.color_wheel import color_wheel

    colors = color_wheel()

    short_names = ["TB", "HV", "EI", "LCB", "BAX", "RS"]
    acquisition_names = ["tb", "hv", "ei", "lcb", "bax", "rs"]

    dataset_tols = {
        "nanoparticle": [0.2, 0.4, 0.6],
        "qm9":          [0.2, 0.4, 0.6],
        "propensity":   [0.2, 0.4, 0.6],
        "toporg":       [0.2, 0.4, 0.6],
        "esol":         [0.2, 0.4, 0.6],
        "freesolv":     [0.2, 0.4, 0.6],
        "lipo":         [0.2, 0.4, 0.6],
        "bace":         [0.2, 0.4, 0.6],
        "branin":       [0.2, 0.4, 0.6],
        "ackley":       [0.2, 0.4, 0.6],
        "hartmann":     [0.2, 0.4, 0.6],
        "layeb06":      [0.2, 0.4, 0.6],
        "kmc":          [0.4],
        "tddft":        [0.6],
    }

    for dataset_name, tolerance_ratios in dataset_tols.items():
        print(f"\n{'=' * 60}")
        print(f"Generating combined radar plot for: {dataset_name.upper()}")
        print(f"{'=' * 60}")

        plot_radar_aucs_all(
            dataset_name=dataset_name,
            tolerance_ratios=tolerance_ratios,
            acquisition_names=acquisition_names,
            short_names=short_names,
            colors=colors,
        )
