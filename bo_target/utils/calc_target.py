"""Target selection via k-medoids / k-means clustering with density filtering."""

import numpy as np

from emukit.core.initial_designs.latin_design import LatinDesign
from scipy.spatial.distance import pdist, cdist
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

def _density_filtered_targets(y, k=5, seed=26, nn=20, min_frac=0.05):
    """Select K targets via k-medoids after filtering sparse outliers."""
    n = len(y)

    dists = cdist(y, y, metric="euclidean")
    nn_dists = np.sort(dists, axis=1)[:, 1 : nn + 1]
    density = 1.0 / (nn_dists.mean(axis=1) + 1e-12)

    # Iteratively increase density cutoff until all targets have min occupancy
    for pct_cut in [0, 10, 20, 30, 40, 50, 60]:
        cutoff = np.percentile(density, pct_cut)
        keep = density >= cutoff
        y_filtered = y[keep]
        if len(y_filtered) < k * 10:
            continue

        km = KMedoids(n_clusters=k, random_state=seed, method="pam")
        km.fit(y_filtered)
        targets = km.cluster_centers_

        eps = pdist(targets).min() / 2
        all_dists = cdist(y, targets, metric="euclidean")
        per_target_counts = (all_dists <= eps).sum(axis=0)
        min_occ = per_target_counts.min() / n

        if min_occ >= min_frac:
            targets = targets[np.argsort(targets[:, 0])]
            print(
                f"  Density cutoff: {pct_cut}th percentile -> "
                f"min occupancy = {min_occ:.1%}"
            )
            return targets

    print(
        f"  Warning: min occupancy {min_occ:.1%} < {min_frac:.1%} (best effort)"
    )
    targets = targets[np.argsort(targets[:, 0])]
    return targets

def get_targets_latin(
    data_func,
    param_space,
    valid_params=None,
    k=5,
    n_samples=10000,
    seed=26,
    method="kmeans",
):
    """Select K diverse targets from the output space via Latin design + clustering."""
    np.random.seed(seed)

    if valid_params is None:
        design = LatinDesign(param_space)
        x_samples = design.get_samples(n_samples)
    else:
        x_samples = valid_params

    y_samples = data_func(x_samples)
    y_samples = (
        np.asarray(y_samples).reshape(-1, 1)
        if y_samples.ndim == 1
        else np.asarray(y_samples)
    )

    if method == "kmeans":
        cluster = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10,
        )
        cluster.fit(y_samples)
        targets = cluster.cluster_centers_
        targets = targets[np.argsort(targets[:, 0])]
    elif method == "density_filtered":
        targets = _density_filtered_targets(y_samples, k=k, seed=seed)
    else:
        cluster = KMedoids(
            n_clusters=k,
            random_state=seed,
            method="pam",
        )
        cluster.fit(y_samples)
        targets = cluster.cluster_centers_
        targets = targets[np.argsort(targets[:, 0])]

    max_epsilon_no_overlap = pdist(targets).min() / 2

    return targets, max_epsilon_no_overlap
