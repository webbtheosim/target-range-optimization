"""Build Fig 6 TDDFT pair data -> data/analysis/fig6_tddft_pair.pkl.

For each target, select n structurally diverse molecules with near-identical
mu1 (eV) spectra from the discovered hits, and bake their dataset indices so
fig6_tddft_pair.py renders from shipped spectra/SMILES with no raw JSONs.
"""

import argparse
import json
import pickle

import numpy as np
from scipy.spatial.distance import pdist

from bo_target.data._paths import base_path as data_base
from bo_target.utils.dataset_config import dataset_config, RAW_RESULTS_DIR
from bo_target.utils.diverse_hits import _farthest_point_select, _aligned_smiles

ANALYSIS_DIR = data_base / "analysis"
OUT_PATH = ANALYSIS_DIR / "fig6_tddft_pair.pkl"

ACQUISITION = "tb"
TOLERANCE_RATIO = 0.6
TRIAL = 0
N_CURVES = 3
N_TARGETS = 5
POOL_MULT = 2.0


def _select_set(target_idx, results_dir, smiles, n=N_CURVES, pool_mult=POOL_MULT):
    """Replicates fig6_tddft_pair._select_set, reading from the canonical sweep."""
    from bo_target.data.tddft import (
        EPSILON_SCALE, N_GAUSSIANS, GAUSSIAN_EV_LABEL_FILE,
        GAUSSIAN_EV_SCALER_FILE, feature_file, config_dir, GroupMinMaxScaler,
    )

    t = target_idx - 1
    with open(config_dir / "tddft_config.json") as f:
        config = json.load(f)
    targets_scaled = np.array(config["target"])
    epsilon = config["epsilon"] * EPSILON_SCALE

    labels_scaled = np.load(GAUSSIAN_EV_LABEL_FILE)
    labels_bo = labels_scaled[:, 1:]
    features = np.load(feature_file)
    scaler = GroupMinMaxScaler.load(GAUSSIAN_EV_SCALER_FILE, n_gaussians=N_GAUSSIANS)
    phys = scaler.inverse_transform(labels_scaled)

    save_bo_dir = dataset_config("tddft", results_dir=results_dir)[7]
    result_path = save_bo_dir / f"tddft_{ACQUISITION}_{TOLERANCE_RATIO}_{TRIAL}.json"
    with open(result_path) as f:
        results = json.load(f)
    discovered = np.unique(np.array(results["indices"], dtype=int))

    out_dist = np.linalg.norm(labels_bo[discovered] - targets_scaled[t], axis=1)
    hit_idx = discovered[out_dist <= epsilon]
    if len(hit_idx) < n:
        raise ValueError(
            f"T{target_idx}: only {len(hit_idx)} discovered hit(s) within "
            f"epsilon={epsilon:.4f}; need at least {n}."
        )

    # tightest mu1 window: slide a length-M window over mu1-sorted hits
    mu1_hit = phys[hit_idx, 1]
    order = np.argsort(mu1_hit)
    hits_sorted = hit_idx[order]
    mu1_sorted = mu1_hit[order]
    M = min(len(hit_idx), max(n, int(round(pool_mult * n))))
    spans = mu1_sorted[M - 1:] - mu1_sorted[:len(mu1_sorted) - M + 1]
    lo = int(np.argmin(spans))
    pool = hits_sorted[lo:lo + M]

    # closest to target in BO (output) space, then farthest in input
    # among the pool — balances spectral similarity with structural diversity
    bo_dists = np.linalg.norm(labels_bo[pool] - targets_scaled[t], axis=1)
    if M == n:
        chosen = pool
    else:
        # greedy: pick closest, then farthest in input from already chosen
        order = np.argsort(bo_dists)
        chosen_idx = [order[0]]  # closest to target
        for _ in range(1, n):
            # among remaining, pick the one farthest in input from chosen
            remaining = [o for o in order if o not in chosen_idx]
            max_d = -1
            best = remaining[0]
            for r in remaining:
                d = np.min(np.linalg.norm(
                    features[pool[r]] - features[pool[chosen_idx]], axis=1))
                if d > max_d:
                    max_d = d
                    best = r
            chosen_idx.append(best)
        chosen = pool[np.array(chosen_idx)]

    mu1_vals = [float(phys[i, 1]) for i in chosen]
    return {
        "chosen": [int(i) for i in chosen],
        "smiles": [smiles[i] for i in chosen],
        "epsilon": float(epsilon),
        "mu1_spread": float(max(mu1_vals) - min(mu1_vals)),
        "in_spread": float(pdist(features[chosen]).max()),
        "mu1_vals": mu1_vals,
    }


def main(results_dir=RAW_RESULTS_DIR):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    smiles = _aligned_smiles()

    payload = {
        "acquisition": ACQUISITION,
        "tolerance_ratio": TOLERANCE_RATIO,
        "trial": TRIAL,
        "n_curves": N_CURVES,
        "targets": {
            ti: _select_set(ti, results_dir, smiles) for ti in range(1, N_TARGETS + 1)
        },
    }

    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RAW_RESULTS_DIR))
    args = parser.parse_args()
    main(results_dir=args.results_dir)
