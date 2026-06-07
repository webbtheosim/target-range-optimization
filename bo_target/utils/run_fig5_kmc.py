"""Build Fig 5 (KMC) plot data -> data/analysis/fig5_kmc_residual.pkl + fig5_kmc_umap.pkl.

Bakes the discovered-spectra residuals and the UMAP 2-D coordinates so the two
KMC figures render with no raw JSONs, no dataset loading, and no umap dependency.
"""

import argparse
import json
import pickle

import joblib
import numpy as np

from bo_target.data._paths import base_path as data_base
from bo_target.utils.dataset_config import dataset_config, RAW_RESULTS_DIR

ANALYSIS_DIR = data_base / "analysis"
CACHE_DIR = data_base.parents[1] / ".cache"  # REPO/.cache

DATASET = "kmc"
TOLERANCE_RATIO = 0.4
N_EXAMPLES = 10
N_TARGETS = 5
SEED = 0
XX = np.linspace(0, 5, 100)
UMAP_SEED = 26
UMAP_SCALE = 1.95


def _scale(arr, vmin, vmax):
    return UMAP_SCALE * (arr - vmin) / (vmax - vmin) - UMAP_SCALE / 2


def _load_run(results_dir):
    func, _, valid, targets, epsilon, _, _, save_dir, _ = dataset_config(
        dataset_name=DATASET, results_dir=results_dir
    )
    targets = np.atleast_1d(targets).squeeze()
    tolerance = epsilon * TOLERANCE_RATIO

    with open(save_dir / f"{DATASET}_tb_{TOLERANCE_RATIO}_{SEED}.json") as f:
        data = json.load(f)
    indices = np.array(data["indices"])

    return func, valid, targets, tolerance, indices


def build_residual(func, valid, targets, tolerance, indices):
    """Target spectra + up to N_EXAMPLES discovered spectra per target."""
    Y_pred = func(valid[indices])

    discovered_curves = []
    for i in range(N_TARGETS):
        sel = np.where(
            np.linalg.norm(Y_pred - targets[i], axis=1) < tolerance
        )[0][::5][:N_EXAMPLES]
        discovered_curves.append(Y_pred[sel])

    return {"XX": XX, "targets": targets, "discovered_curves": discovered_curves}


def _get_reducer(valid):
    import umap

    cache_path = CACHE_DIR / "kmc_umap_reducer.pkl"
    if cache_path.exists():
        return joblib.load(cache_path)

    reducer = umap.UMAP(n_components=2, random_state=UMAP_SEED)
    reducer.fit(valid)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, cache_path)
    return reducer


def build_umap(func, valid, targets, tolerance, indices):
    """Scaled UMAP coordinates for the full design space and per-target hits."""
    Y_pred = func(valid[indices])
    hit_indices = [
        indices[
            np.where(np.linalg.norm(Y_pred - targets[i], axis=1) < tolerance)[0]
        ]
        for i in range(N_TARGETS)
    ]

    reducer = _get_reducer(valid)
    x0, x1 = valid[:, 0].min(), valid[:, 0].max()
    y0, y1 = valid[:, 1].min(), valid[:, 1].max()

    valid_2d = reducer.embedding_
    valid_scaled = np.column_stack((
        _scale(valid_2d[:, 0], x0, x1),
        _scale(valid_2d[:, 1], y0, y1),
    ))

    hits_scaled = []
    for i in range(N_TARGETS):
        hits_2d = reducer.transform(valid[hit_indices[i]])
        hits_scaled.append(np.column_stack((
            _scale(hits_2d[:, 0], x0, x1),
            _scale(hits_2d[:, 1], y0, y1),
        )))

    return {"valid_scaled": valid_scaled, "hits_scaled": hits_scaled}


def main(results_dir=RAW_RESULTS_DIR):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    func, valid, targets, tolerance, indices = _load_run(results_dir)

    with open(ANALYSIS_DIR / "fig5_kmc_residual.pkl", "wb") as f:
        pickle.dump(build_residual(func, valid, targets, tolerance, indices), f)
    print(f"Saved -> {ANALYSIS_DIR / 'fig5_kmc_residual.pkl'}")

    with open(ANALYSIS_DIR / "fig5_kmc_umap.pkl", "wb") as f:
        pickle.dump(build_umap(func, valid, targets, tolerance, indices), f)
    print(f"Saved -> {ANALYSIS_DIR / 'fig5_kmc_umap.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RAW_RESULTS_DIR))
    args = parser.parse_args()
    main(results_dir=args.results_dir)
