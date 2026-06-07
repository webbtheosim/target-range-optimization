"""Build radar AUC data -> data/analysis/radar_aucs.pkl.

For every dataset, tolerance, and acquisition, store the per-target AUC array
(n_targets x n_runs) that fig_radar_aucs_all.py renders. This is the only figure
that needs the full BO sweep, so its arrays are precomputed here once.
"""

import argparse
import pickle

from bo_target.data._paths import base_path as data_base
from bo_target.utils.analysis import per_target_aucs
from bo_target.utils.dataset_config import RAW_RESULTS_DIR

ANALYSIS_DIR = data_base / "analysis"
OUT_PATH = ANALYSIS_DIR / "radar_aucs.pkl"

ACQUISITIONS = ["tb", "hv", "ei", "lcb", "bax", "rs"]
SHORT_NAMES = ["TB", "HV", "EI", "LCB", "BAX", "RS"]
# Per-dataset tolerance ratios: 12 non-spectral use all three,
# spectral datasets use their primary tolerance only.
DATASET_TOLS = {
    "ackley":       [0.2, 0.4, 0.6],
    "bace":         [0.2, 0.4, 0.6],
    "branin":       [0.2, 0.4, 0.6],
    "esol":         [0.2, 0.4, 0.6],
    "freesolv":     [0.2, 0.4, 0.6],
    "hartmann":     [0.2, 0.4, 0.6],
    "layeb06":      [0.2, 0.4, 0.6],
    "lipo":         [0.2, 0.4, 0.6],
    "nanoparticle": [0.2, 0.4, 0.6],
    "propensity":   [0.2, 0.4, 0.6],
    "qm9":          [0.2, 0.4, 0.6],
    "toporg":       [0.2, 0.4, 0.6],
    "kmc":          [0.4],
    "tddft":        [0.6],
}
# KMC sweep uses 1000 iterations; all others use 250.
DATASET_STOP_ITER = {ds: 1000 if ds == "kmc" else 250 for ds in DATASET_TOLS}


def main(results_dir=RAW_RESULTS_DIR):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    aucs = {}
    all_ds = list(DATASET_TOLS.keys())
    all_tols = sorted(set(t for tols in DATASET_TOLS.values() for t in tols))

    for ds in all_ds:
        aucs[ds] = {}
        stop_iter = DATASET_STOP_ITER[ds]
        for tol in DATASET_TOLS[ds]:
            aucs[ds][tol] = {}
            for acq in ACQUISITIONS:
                aucs[ds][tol][acq] = per_target_aucs(
                    acquisition_name=acq,
                    dataset_name=ds,
                    target_index="all",
                    tolerance_ratio=tol,
                    stop_iter=stop_iter,
                    testing=False,
                    results_dir=results_dir,
                )
            shapes = {a: aucs[ds][tol][a].shape for a in ACQUISITIONS}
            print(f"{ds:13s} tol={tol}: {shapes}")

    payload = {
        "aucs": aucs,
        "acquisition_names": ACQUISITIONS,
        "short_names": SHORT_NAMES,
        "dataset_names": all_ds,
        "dataset_tols": DATASET_TOLS,
        "dataset_stop_iter": DATASET_STOP_ITER,
        "tolerance_ratios": all_tols,
    }

    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RAW_RESULTS_DIR))
    args = parser.parse_args()
    main(results_dir=args.results_dir)
