"""Compute per-target AUC statistics and variance decompositions across all datasets."""

import csv
import pickle

import numpy as np

from bo_target.data._paths import base_path as data_base
from bo_target.utils.analysis import per_target_aucs
from bo_target.utils.dataset_config import RAW_RESULTS_DIR

OUT_DIR = data_base / "analysis"

ACQUISITIONS = {
    "tb": "TB",
    "hv": "HV",
    "ei": "EI",
    "lcb": "LCB",
    "bax": "BAX",
    "rs": "RS",
}
DATASETS = [
    "ackley",
    "bace",
    "branin",
    "esol",
    "freesolv",
    "hartmann",
    "kmc",
    "layeb06",
    "lipo",
    "nanoparticle",
    "propensity",
    "qm9",
    "toporg",
]
TOLS = [0.2, 0.4, 0.6]
STOP_ITER = 250
DDOF = 0


def summarize_auc_array(arr: np.ndarray):
    """Compute mean, variance-across-runs, and variance-across-targets for an AUC array."""
    if arr.size == 0:
        return {
            "n_targets": 0,
            "n_runs": 0,
            "mean_score": float("nan"),
            "var_across_runs_of_mean": float("nan"),
            "avg_var_targets": float("nan"),
            "avg_var_runs": float("nan"),
        }

    arr = np.atleast_2d(arr)
    n_targets, n_runs = arr.shape

    mean_per_run = arr.mean(axis=0)
    return {
        "n_targets": int(n_targets),
        "n_runs": int(n_runs),
        "mean_score": float(mean_per_run.mean()),
        "var_across_runs_of_mean": float(mean_per_run.var(ddof=DDOF)),
        "avg_var_targets": float(arr.var(axis=0, ddof=DDOF).mean()),
        "avg_var_runs": float(arr.var(axis=1, ddof=DDOF).mean()),
    }


def compute_stats_results():
    """Compute per-target AUC summary statistics for every dataset, tolerance, and acquisition."""
    acquisition_names = list(ACQUISITIONS.keys())
    stats: dict = {}
    rows: list = []

    for dataset in DATASETS:
        stats[dataset] = {}
        for tol in TOLS:
            stats[dataset][tol] = {}
            for acq in acquisition_names:
                arr = per_target_aucs(
                    acquisition_name=acq,
                    dataset_name=dataset,
                    target_index="all",
                    tolerance_ratio=tol,
                    stop_iter=STOP_ITER,
                    testing=True,
                    results_dir=RAW_RESULTS_DIR,
                )
                m = summarize_auc_array(arr)
                stats[dataset][tol][acq] = m
                rows.append({
                    "dataset": dataset,
                    "tolerance_ratio": float(tol),
                    "acquisition": acq,
                    **m,
                })

    for dataset in DATASETS:
        for tol in TOLS:
            means = np.array(
                [
                    stats[dataset][tol][acq]["mean_score"]
                    for acq in acquisition_names
                ],
                dtype=float,
            )
            if np.all(np.isnan(means)):
                ranks = np.full_like(means, np.nan)
            else:
                ranks = np.argsort(
                    np.argsort(-np.nan_to_num(means, nan=-np.inf))
                )
            for acq, r in zip(acquisition_names, ranks):
                stats[dataset][tol][acq]["rank_by_mean_score"] = (
                    float(r) if np.isfinite(r) else float("nan")
                )

    return {
        "stats": stats,
        "rows": rows,
        "acquisition_names": acquisition_names,
        "short_names": list(ACQUISITIONS.values()),
        "dataset_names": DATASETS,
        "tolerance_ratios": TOLS,
    }


def main():
    """Run full AUC variance analysis and save results as PKL and CSV."""
    results = compute_stats_results()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = OUT_DIR / "stats_results.pkl"
    csv_path = OUT_DIR / "stats_results.csv"

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    rows = results["rows"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {pkl_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
