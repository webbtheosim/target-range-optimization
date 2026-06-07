"""Post-hoc BO analysis: per-target AUC, AUC vs budget, and match-ratio curves."""

import glob
import json
import pickle

import numpy as np

from bo_target.data._paths import base_path as data_base
from bo_target.utils.dataset_config import dataset_config

config_dir = data_base / "config"
config_dir.mkdir(parents=True, exist_ok=True)

_SYNTHETIC = {"hartmann", "branin", "ackley", "layeb06"}
_SPECTRAL = {"kmc", "tddft"}

_SEED_MAX_DIST = {
    "branin": (20, 2.0 ** 0.5 * 0.1),
    "hartmann": (30, 3.0 ** 0.5 * 0.1),
    "ackley": (50, 5.0 ** 0.5 * 0.1),
    "layeb06": (60, 6.0 ** 0.5 * 0.1),
}

def analyze_discoveries(X, Y, target, d1_threshold=0.01, min_pairwise_dist=0.1):
    """Count diverse hits within a D1-threshold ball around a target."""
    D1 = np.sum((Y - target) ** 2, axis=1)

    good_mask = D1 < d1_threshold**2
    n_hits = int(good_mask.sum())
    X_good = X[good_mask]

    if n_hits == 0:
        return 0, 0, min_pairwise_dist, []

    if n_hits == 1:
        return 1, 1, min_pairwise_dist, np.where(good_mask)[0].tolist()

    accepted = np.zeros(n_hits, dtype=bool)
    accepted[0] = True
    count = 1

    for i in range(1, n_hits):
        dists_to_accepted = np.sqrt(
            np.sum(
                (X_good[i] - X_good[accepted]) ** 2,
                axis=1,
            )
        )
        if np.min(dists_to_accepted) > min_pairwise_dist:
            accepted[i] = True
            count += 1

    good_indices = np.where(good_mask)[0].tolist()

    return n_hits, count, min_pairwise_dist, good_indices

def per_target_aucs(
    acquisition_name="hv",
    dataset_name="branin",
    target_index="all",
    tolerance_ratio=0.1,
    trial_idx="all",
    stop_iter=250,
    verbose=False,
    testing=True,
    results_dir=None,
):
    """Compute per-target AUC for a given dataset, acquisition, and tolerance."""
    _, _, _, targets, epsilon, seed_size, _, save_dir, _ = dataset_config(
        dataset_name,
        testing=testing,
        results_dir=results_dir,
    )

    epsilon *= tolerance_ratio

    if target_index != "all":
        targets = (
            targets[target_index]
            if isinstance(target_index, int)
            else targets[target_index]
        )
    targets = np.array(targets)

    pattern = (
        f"{save_dir}/"
        f"{dataset_name}_{acquisition_name}_{tolerance_ratio}_[0-9].json"
    )
    files = sorted(glob.glob(pattern))

    if trial_idx != "all":
        files = [files[trial_idx]]

    if dataset_name in _SEED_MAX_DIST:
        seed_size, max_dist = _SEED_MAX_DIST[dataset_name]
    else:
        seed_size = 50
        max_dist = 1.0

    n_iter = float(stop_iter) / 5.0

    aucs = []
    kept_files = 0

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        if "X" in data:
            full_X = np.array(data["X"])
            full_Y = np.array(data["Y"])
            X = full_X[seed_size : seed_size + stop_iter]
        else:
            full_X = np.array(data["indices"]).reshape(-1, 1)
            full_Y = np.array(data["Y"])
            X = full_X[:stop_iter]

        Y = full_Y[seed_size : seed_size + stop_iter]

        run_aucs = []

        if dataset_name not in _SPECTRAL:
            data = pickle.load(
                open(
                    config_dir / f"{dataset_name}_epsilon_analysis.pkl",
                    "rb",
                )
            )

            counts_per_target = data[tolerance_ratio]["counts_per_target"]

        for idx, target in enumerate(targets):
            curve = np.array([
                analyze_discoveries(
                    X=X,
                    Y=Y,
                    target=target,
                    d1_threshold=epsilon,
                    min_pairwise_dist=d,
                )[1]
                for d in np.arange(0.0, max_dist, 0.1)
            ])

            if dataset_name in _SYNTHETIC:
                auc = np.trapz(curve, np.arange(0.0, max_dist, 0.1))
                auc /= n_iter * max_dist
            elif dataset_name in _SPECTRAL:
                auc = curve[0] / n_iter
            else:
                auc = curve[0] / min(n_iter, counts_per_target[idx])

            run_aucs.append(auc)

        aucs.append(run_aucs)
        kept_files += 1

    aucs = np.array(aucs).T

    if verbose:
        print(f"Kept {kept_files} complete runs for {acquisition_name}")

    return aucs

def auc_vs_budget(
    acquisition_name="hv",
    dataset_name="branin",
    target_index="all",
    tolerance_ratio=0.8,
    stop_iter=250,
    verbose=False,
    testing=True,
    results_dir=None,
):
    """Compute AUC as a function of BO iteration budget."""
    _, _, _, targets, epsilon, seed_size, _, save_dir, _ = dataset_config(
        dataset_name,
        testing=testing,
        results_dir=results_dir,
    )

    epsilon *= tolerance_ratio

    if target_index != "all":
        targets = (
            targets[target_index]
            if isinstance(target_index, int)
            else targets[target_index]
        )
    targets = np.array(targets)

    pattern = (
        f"{save_dir}/"
        f"{dataset_name}_{acquisition_name}_{tolerance_ratio}_[0-9].json"
    )
    files = sorted(glob.glob(pattern))
    n_runs = len(files)

    if n_runs == 0:
        budget = np.arange(0, stop_iter, 10)
        z = np.zeros_like(budget, dtype=float)
        return budget, z, z, np.empty((0, len(budget))), 0.0, 0.0

    if dataset_name in _SEED_MAX_DIST:
        seed_size, max_dist = _SEED_MAX_DIST[dataset_name]
    else:
        seed_size = 50
        max_dist = 1.0

    dist_grid = np.arange(0.0, max_dist, 0.1)
    budget = np.arange(5, stop_iter + 1, 5)

    aucs_all = []

    if dataset_name not in _SPECTRAL:
        data = pickle.load(
            open(
                config_dir / f"{dataset_name}_epsilon_analysis.pkl",
                "rb",
            )
        )


    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        if "X" in data:
            full_X = np.array(data["X"])
            full_Y = np.array(data["Y"])
        else:
            full_X = np.array(data["indices"]).reshape(-1, 1)
            full_Y = np.array(data["Y"])

        aucs_run = []

        for b in budget:
            if "X" in data:
                X = full_X[seed_size : seed_size + b]
            else:
                X = full_X[:b]

            Y = full_Y[seed_size : seed_size + b]

            run_targets = []

            for idx, target in enumerate(targets):
                curve = np.array([
                    analyze_discoveries(
                        X=X,
                        Y=Y,
                        target=target,
                        d1_threshold=epsilon,
                        min_pairwise_dist=d,
                    )[1]
                    for d in dist_grid
                ])

                if dataset_name in _SYNTHETIC:
                    auc = np.trapz(curve, dist_grid) / (b / 5 * max_dist)
                else:
                    auc = curve[0]

                run_targets.append(auc)

            aucs_run.append(run_targets)

        aucs_all.append(aucs_run)

    aucs_all = np.array(aucs_all)
    
    auc_mean = aucs_all.mean(axis=0)
    auc_sem = aucs_all.std(axis=0) / np.sqrt(n_runs)

    return (
        budget,
        auc_mean,
        auc_sem,
        aucs_all,
        auc_mean.min(),
        auc_mean.max(),
    )

def match_ratios_curve(
    acquisition_name="hv",
    dataset_name="branin",
    target_index="all",
    tolerance_ratio=0.4,
    testing=False,
    results_dir=None,
):
    """Compute off-target hit ratio as a function of budget for one BO run."""
    _, _, _, targets, epsilon, seed_size, _, save_dir, _ = dataset_config(
        dataset_name,
        testing=testing,
        results_dir=results_dir,
    )

    epsilon *= tolerance_ratio

    if target_index != "all":
        targets = (
            targets[target_index]
            if isinstance(target_index, int)
            else targets[target_index]
        )
    targets = np.array(targets)

    pattern = (
        f"{save_dir}/"
        f"{dataset_name}_{acquisition_name}_{tolerance_ratio}_[0-9].json"
    )
    files = sorted(glob.glob(pattern))

    if not files:
        return np.array([]), np.array([])

    with open(files[0]) as f:
        first_data = json.load(f)

    total_points = len(first_data["Y"]) - seed_size
    batch_sizes = list(range(10, total_points + 1, 10))
    if not batch_sizes:
        # Run too short (e.g. a truncated trial); treat as no usable data.
        return np.array([]), np.array([])
    if batch_sizes[-1] != total_points:
        batch_sizes.append(total_points)

    all_curves = []

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        if "X" in data:
            Y = np.array(data["Y"])[seed_size:]
        else:
            Y = np.array(data["Y"])[seed_size:]

        curve = np.zeros((len(batch_sizes), len(targets)))

        for bidx, bsize in enumerate(batch_sizes):
            Yb = Y[:bsize]

            for tidx, target in enumerate(targets):
                dists = np.sum((Yb - target) ** 2, axis=1)
                good_idx = np.where(dists < epsilon**2)[0]

                if len(good_idx) == 0:
                    curve[bidx, tidx] = 0.0
                    continue

                correct_mask = (good_idx % 5) == tidx
                n_correct = np.sum(correct_mask)

                curve[bidx, tidx] = (len(good_idx) - n_correct) / len(good_idx)

        all_curves.append(curve)

    budgets = np.array(batch_sizes)
    curves = np.array(all_curves)

    return budgets, curves
