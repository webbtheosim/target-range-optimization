"""Compute cross-target contamination curves: off-target hit ratio vs budget."""

import pickle

from bo_target.data._paths import base_path as data_base
from bo_target.utils.analysis import match_ratios_curve
from bo_target.utils.dataset_config import RAW_RESULTS_DIR

OUT_DIR = data_base / "analysis"


def compute_cross_target_contamination_curve():
    """Compute off-target hit ratio curves for every dataset, tolerance, and acquisition."""
    short_names = ["TB", "HV", "EI", "LCB", "BAX", "RS"]

    acquisition_names = ["tb", "hv", "ei", "lcb", "bax", "rs"]

    dataset_names = [
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

    tolerance_ratios = [0.2, 0.4, 0.6]

    all_results = {
        dataset: {tol: [] for tol in tolerance_ratios}
        for dataset in dataset_names
    }

    for dataset_name in dataset_names:
        for tolerance_ratio in tolerance_ratios:
            print(f"Processing {dataset_name} -- tol = {tolerance_ratio}")

            results_per_tol = []

            for acq_name, short_name in zip(acquisition_names, short_names):
                budgets, curves = match_ratios_curve(
                    acquisition_name=acq_name,
                    dataset_name=dataset_name,
                    target_index="all",
                    tolerance_ratio=tolerance_ratio,
                    testing=True,
                    results_dir=RAW_RESULTS_DIR,
                )

                if len(budgets) == 0:
                    continue

                mean_over_targets = curves.mean(axis=2)  # (n_runs, n_batches)
                mean = mean_over_targets.mean(axis=0)
                std = mean_over_targets.std(axis=0, ddof=1)

                results_per_tol.append(
                    {
                        "short": short_name,
                        "budget": budgets,
                        "mean": mean,
                        "std": std,
                    }
                )

            all_results[dataset_name][tolerance_ratio] = results_per_tol

    return all_results

if __name__ == "__main__":
    all_results = compute_cross_target_contamination_curve()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "cross_target_hit_ratio.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)

    print("Done. Data saved.")