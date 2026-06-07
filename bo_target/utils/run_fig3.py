"""Build Fig 3 plot data (Branin + QM9 trajectories) -> data/analysis/fig3_branin_qm9.pkl.

Reads the raw BO result JSONs once and bakes everything the figure plots into a
single pickle, so fig3_branin_qm9.py renders with no dataset loading and no
sibling bo_result/ directory.
"""

import argparse
import json
import pickle

import joblib
import numpy as np

from bo_target.data._paths import base_path as data_base
from bo_target.utils.dataset_config import dataset_config, RAW_RESULTS_DIR

ANALYSIS_DIR = data_base / "analysis"
OUT_PATH = ANALYSIS_DIR / "fig3_branin_qm9.pkl"

TOLERANCE_RATIO = 0.4
ACQ_KEYS = ("tb", "hv", "lcb", "bax")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def build_branin(results_dir):
    """Branin landscape grid, targets, tolerance, and per-acquisition discovered points."""
    func, _, _, targets, epsilon, seed_size, _, save_dir, _ = dataset_config(
        dataset_name="branin", results_dir=results_dir
    )

    data = {
        k: _load_json(save_dir / f"branin_{k}_{TOLERANCE_RATIO}_1.json")
        for k in ACQ_KEYS
    }
    n = seed_size
    X_list = [np.array(data[k]["X"])[n:n + 250] for k in ACQ_KEYS]

    x = np.linspace(0, 1, 200)
    X1, X2 = np.meshgrid(x, x)
    Z = func(np.column_stack((X1.ravel(), X2.ravel()))).reshape(X1.shape)

    targets = np.atleast_1d(targets).astype(float).squeeze()
    tolerance = epsilon * TOLERANCE_RATIO

    return {
        "X1": X1,
        "X2": X2,
        "Z": Z,
        "targets": targets,
        "tolerance": tolerance,
        "X_list": X_list,
    }


def build_qm9(results_dir):
    """QM9 HOMO-LUMO density, targets, tolerance ellipses, and discovered points (eV)."""
    func, _, valid_configs, targets, epsilon, _, _, save_dir, _ = dataset_config(
        dataset_name="qm9", results_dir=results_dir
    )

    seeds = {"tb": 9, "hv": 9, "lcb": 0, "bax": 0}
    data = {
        k: _load_json(save_dir / f"qm9_{k}_{TOLERANCE_RATIO}_{seeds[k]}.json")
        for k in ACQ_KEYS
    }
    Y_list = [np.array(data[k]["Y"])[50:300] for k in ACQ_KEYS]
    Y_all = func(valid_configs)

    scaler = joblib.load(data_base / "clean" / "qm9_output_scaler.pkl")

    targets_arr = np.asarray(targets)
    if targets_arr.ndim == 1:
        targets_arr = targets_arr.reshape(1, -1)

    Y_all_orig = scaler.inverse_transform(np.asarray(Y_all))
    targets_orig = scaler.inverse_transform(targets_arr)
    Y_list_orig = [scaler.inverse_transform(np.asarray(Y)) for Y in Y_list]
    epsilon_orig = epsilon * TOLERANCE_RATIO * scaler.scale_

    return {
        "Y_all_orig": Y_all_orig,
        "targets_orig": targets_orig,
        "epsilon_orig": epsilon_orig,
        "Y_list_orig": Y_list_orig,
    }


def main(results_dir=RAW_RESULTS_DIR):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "tolerance_ratio": TOLERANCE_RATIO,
        "acq_keys": list(ACQ_KEYS),
        "branin": build_branin(results_dir),
        "qm9": build_qm9(results_dir),
    }

    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RAW_RESULTS_DIR))
    args = parser.parse_args()
    main(results_dir=args.results_dir)
