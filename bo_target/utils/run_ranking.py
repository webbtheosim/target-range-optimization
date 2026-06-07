"""Compute acquisition method rankings (avg rank, best/worst fractions) across all datasets."""

import pickle

import numpy as np

from bo_target.data._paths import base_path as data_base
from bo_target.utils.analysis import per_target_aucs
from bo_target.utils.dataset_config import RAW_RESULTS_DIR

OUT_DIR = data_base / "analysis"


def compute_ranking_results():
    """Compute per-method ranks, average rank, and best/worst fractions across datasets."""
    short_names = ["TB", "HV", "EI", "LCB", "BAX", "RS"]
    acquisition_names = ["tb", "hv", "ei", "lcb", "bax", "rs"]

    dataset_names = [
        "ackley",
        "bace",
        "branin",
        "esol",
        "freesolv",
        "hartmann",
        "layeb06",
        "lipo",
        "nanoparticle",
        "propensity",
        "qm9",
        "toporg",
    ]

    tolerance_ratios = [0.2, 0.4, 0.6]

    ranks_list = []

    for dataset_name in dataset_names:
        for tolerance_ratio in tolerance_ratios:
            all_aucs_raw = [
                per_target_aucs(
                    acquisition_name=name,
                    dataset_name=dataset_name,
                    target_index="all",
                    tolerance_ratio=tolerance_ratio,
                    stop_iter=250,
                    testing=True,
                    results_dir=RAW_RESULTS_DIR,
                )
                for name in acquisition_names
            ]

            auc_stats = []
            for arr in all_aucs_raw:
                if arr.size == 0:
                    auc_stats.append(0.0)
                    continue

                n_targets, n_runs = arr.shape
                geo_per_run = np.prod(arr, axis=0) ** (1.0 / n_targets)
                auc_stats.append(geo_per_run.mean())

            auc_stats = np.array(auc_stats)

            ranks = np.argsort(np.argsort(-auc_stats))
            ranks_list.append(ranks)

    ranks_array = np.array(ranks_list)

    n_valid = len(ranks_array)
    n_methods = len(acquisition_names)

    avg_ranks = ranks_array.mean(axis=0)
    sem_ranks = ranks_array.std(axis=0, ddof=0) / np.sqrt(n_valid)
    best_freq = (ranks_array == 0).sum(axis=0)
    worst_freq = (ranks_array == n_methods - 1).sum(axis=0)

    return {
        "ranks_array": ranks_array,
        "avg_ranks": avg_ranks,
        "sem_ranks": sem_ranks,
        "best_freq": best_freq,
        "worst_freq": worst_freq,
        "acquisition_names": acquisition_names,
        "short_names": short_names,
        "dataset_names": dataset_names,
        "tolerance_ratios": tolerance_ratios,
    }

if __name__ == "__main__":
    results = compute_ranking_results()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "ranking_results.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved: {output_path}")