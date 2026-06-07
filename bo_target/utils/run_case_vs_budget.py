"""Compute valid-candidate count vs BO budget curves for KMC and TDDFT."""

import pickle

from bo_target.data._paths import base_path as data_base
from bo_target.utils.analysis import auc_vs_budget
from bo_target.utils.dataset_config import RAW_RESULTS_DIR

OUT_DIR = data_base / "analysis"


def compute_auc_results():
    """Compute per-method AUC-vs-budget curves for KMC and TDDFT datasets."""
    short_names = ["TB", "HV", "EI", "LCB", "BAX", "RS"]
    acquisition_names = ["tb", "hv", "ei", "lcb", "bax", "rs"]

    dataset_tol = {
        "kmc": [0.4],
        "tddft": [0.6],
    }

    all_results = {dataset: {} for dataset in dataset_tol}

    for dataset_name, tolerance_ratios in dataset_tol.items():
        for tolerance_ratio in tolerance_ratios:
            results_per_tol = []

            for acq_name, short_name in zip(acquisition_names, short_names):
                (
                    budget,
                    mean,
                    std,
                    _,
                    vmin,
                    vmax,
                ) = auc_vs_budget(
                    acquisition_name=acq_name,
                    dataset_name=dataset_name,
                    target_index="all",
                    tolerance_ratio=tolerance_ratio,
                    verbose=True,
                    testing=True,
                    stop_iter=1000 if dataset_name == "kmc" else 250,
                    results_dir=RAW_RESULTS_DIR,
                )

                results_per_tol.append({
                    "method": short_name,
                    "budget": budget,
                    "mean": mean,
                    "std": std,
                    "vmin": vmin,
                    "vmax": vmax,
                })

            all_results[dataset_name][tolerance_ratio] = results_per_tol

    return all_results

if __name__ == "__main__":
    all_results = compute_auc_results()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "case_vs_budget.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)
