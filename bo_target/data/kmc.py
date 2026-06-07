import json
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from bo_target.data._paths import base_path
from bo_target.data._pipeline import make_bandit_function, remove_duplicate_rows

clean_dir = base_path / "clean"
config_dir = base_path / "config"
desc_dir = base_path / "desc"
clean_dir.mkdir(parents=True, exist_ok=True)
config_dir.mkdir(parents=True, exist_ok=True)
desc_dir.mkdir(parents=True, exist_ok=True)

input_pickle = desc_dir / "kmc_input.pickle"
output_pickle = desc_dir / "kmc_output.pickle"
feature_file = clean_dir / "kmc_features_scaled.npy"
label_file = clean_dir / "kmc_labels_raw.npy"
config_file = config_dir / "kmc_config.json"

RERUN = "None"
force_rerun = RERUN in {"features"}

RANDOM_STATE = 26
EPSILON_SCALE = 0.6
EVAL_STRIDE = 100
SUBSAMPLE_MIN = 100
SUBSAMPLE_MAX = 5000
SUBSAMPLE_DIV = 100


def load_or_build_kmc_data(force_rerun=False):
    if not force_rerun and feature_file.exists() and label_file.exists():
        x = np.load(feature_file)
        y = np.load(label_file)
        print("Loading precomputed scaled features and raw labels...")
        print(f"Loaded scaled features with dimension: {x.shape[1]}")
        print(f"Loaded raw labels with shape: {y.shape}")
        return x, y

    print("Processing KMC dataset...")

    with open(input_pickle, "rb") as f:
        raw_x = pickle.load(f)
    with open(output_pickle, "rb") as f:
        raw_y = pickle.load(f)

    raw_x = np.asarray(raw_x, dtype=np.float32)[:, [0, 1, 2, 5, 6]]
    raw_y = np.asarray(raw_y, dtype=np.float32)

    print(f"Loaded raw features shape: {raw_x.shape}")
    print(f"Loaded raw labels shape: {raw_y.shape}")

    non_const_cols = raw_x.std(axis=0) > 0
    n_const_removed = np.sum(~non_const_cols)
    if n_const_removed > 0:
        print(f"Removing {n_const_removed} constant column(s).")
        raw_x = raw_x[:, non_const_cols]
        print(
            f"New feature shape after dropping constant columns: {raw_x.shape}"
        )
    else:
        print("No constant columns found.")

    original_n = raw_x.shape[0]
    raw_x, raw_y = remove_duplicate_rows(raw_x, raw_y)
    n_removed = original_n - len(raw_x)
    print(f"Removed {n_removed} duplicate rows -> New shape: {raw_x.shape}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_x = scaler.fit_transform(raw_x)

    np.save(feature_file, scaled_x)
    np.save(label_file, raw_y)

    print(f"Saved scaled features -> {feature_file} (shape: {scaled_x.shape})")
    print(f"Saved raw labels -> {label_file} (shape: {raw_y.shape})")

    return scaled_x, raw_y


SCALED_KMC_DATA, RAW_KMC_LABELS = load_or_build_kmc_data(
    force_rerun=force_rerun
)


def kmc_function():
    """KMC styrene NMP-to-FRP polymerization. 5 reaction parameters, 100-bin log-CLD targets."""
    return make_bandit_function(SCALED_KMC_DATA, RAW_KMC_LABELS)


def choose_subsample_indices(
    n_full, seed=26, min_size=100, max_size=5000, frac_div=100
):
    n_subsample = max(min_size, min(max_size, n_full // frac_div))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_full, size=n_subsample, replace=False)
    idx.sort()
    return idx


def save_config(targets, epsilon, random_state=26, name="kmc", max_iter=200):
    config = {
        "global": {"random_state": random_state, "max_iter": max_iter},
        "name": name,
        "format": "bandit",
        "target": targets.tolist(),
        "epsilon": round(float(epsilon), 4),
    }
    with open(config_file, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"Configuration file saved -> {config_file}")
    return config


def load_config():
    with open(config_file, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_curve_arrays(y, targets):
    if y.ndim != 2:
        raise ValueError(
            f"Expected y to have shape (n_curves, n_grid), got {y.shape}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"Expected targets to have shape (n_targets, n_grid), got {targets.shape}"
        )
    if y.shape[1] != targets.shape[1]:
        raise ValueError(
            f"Curve length mismatch: y has {y.shape[1]} columns, target_values has {targets.shape[1]}"
        )


def compute_target_statistics(y, targets, epsilon):
    validate_curve_arrays(y, targets)

    dist_l2 = cdist(y, targets, metric="euclidean")
    inside_mask = dist_l2 <= epsilon
    inside_any = inside_mask.any(axis=1)

    assigned_inside = np.full(y.shape[0], -1, dtype=int)
    for i in range(y.shape[0]):
        valid = np.where(inside_mask[i])[0]
        if valid.size > 0:
            assigned_inside[i] = valid[np.argmin(dist_l2[i, valid])]

    nearest_target = dist_l2.argmin(axis=1)
    nearest_dist_l2 = dist_l2[np.arange(y.shape[0]), nearest_target]

    per_target = []
    for t in range(targets.shape[0]):
        idx_inside = np.where(assigned_inside == t)[0]

        if idx_inside.size == 0:
            per_target.append({
                "target": t,
                "count_inside": 0,
                "frac_inside": 0.0,
                "l2_min": np.nan,
                "l2_q25": np.nan,
                "l2_median": np.nan,
                "l2_q75": np.nan,
                "l2_max": np.nan,
                "linf_min": np.nan,
                "linf_q25": np.nan,
                "linf_median": np.nan,
                "linf_q75": np.nan,
                "linf_max": np.nan,
                "empirical_linf_band": np.zeros(y.shape[1], dtype=float),
                "inside_indices": idx_inside,
            })
            continue

        y_inside = y[idx_inside]
        abs_dev = np.abs(y_inside - targets[t])
        l2_inside = np.linalg.norm(y_inside - targets[t], axis=1)
        linf_inside = np.max(abs_dev, axis=1)
        empirical_linf_band = abs_dev.max(axis=0)

        per_target.append({
            "target": t,
            "count_inside": int(idx_inside.size),
            "frac_inside": float(idx_inside.size / y.shape[0]),
            "l2_min": float(np.min(l2_inside)),
            "l2_q25": float(np.quantile(l2_inside, 0.25)),
            "l2_median": float(np.median(l2_inside)),
            "l2_q75": float(np.quantile(l2_inside, 0.75)),
            "l2_max": float(np.max(l2_inside)),
            "linf_min": float(np.min(linf_inside)),
            "linf_q25": float(np.quantile(linf_inside, 0.25)),
            "linf_median": float(np.median(linf_inside)),
            "linf_q75": float(np.quantile(linf_inside, 0.75)),
            "linf_max": float(np.max(linf_inside)),
            "empirical_linf_band": empirical_linf_band,
            "inside_indices": idx_inside,
        })

    summary = {
        "dist_l2": dist_l2,
        "inside_mask": inside_mask,
        "inside_any": inside_any,
        "assigned_inside": assigned_inside,
        "nearest_target": nearest_target,
        "nearest_dist_l2": nearest_dist_l2,
        "per_target": per_target,
        "n_total": int(y.shape[0]),
        "n_inside_any": int(inside_any.sum()),
        "n_outside_all": int((~inside_any).sum()),
    }
    return summary


def print_target_statistics(summary, epsilon):
    print(f"\n=== Global summary (epsilon = {epsilon:.4f}) ===")
    print(f"Total curves: {summary['n_total']}")
    print(f"Inside at least one L2 ball: {summary['n_inside_any']}")
    print(f"Outside all L2 balls: {summary['n_outside_all']}")

    nearest_dist_l2 = summary["nearest_dist_l2"]
    print("\n=== Nearest-target L2 distance summary ===")
    print(
        f"min={nearest_dist_l2.min():.4f}, "
        f"q25={np.quantile(nearest_dist_l2, 0.25):.4f}, "
        f"median={np.median(nearest_dist_l2):.4f}, "
        f"q75={np.quantile(nearest_dist_l2, 0.75):.4f}, "
        f"max={nearest_dist_l2.max():.4f}"
    )

    print("\n=== Per-target summary ===")
    for s in summary["per_target"]:
        print(
            f"Target {s['target']}: "
            f"count={s['count_inside']}, "
            f"frac={s['frac_inside']:.4f}, "
            f"L2[min/q25/med/q75/max]=[{s['l2_min']:.4f}, {s['l2_q25']:.4f}, {s['l2_median']:.4f}, {s['l2_q75']:.4f}, {s['l2_max']:.4f}], "
            f"Linf[min/q25/med/q75/max]=[{s['linf_min']:.4f}, {s['linf_q25']:.4f}, {s['linf_median']:.4f}, {s['linf_q75']:.4f}, {s['linf_max']:.4f}]"
        )


if __name__ == "__main__":
    from bo_target.utils.calc_target import get_targets_latin

    f, space, valid_configs = kmc_function()

    n_full = len(valid_configs)
    subsample_idx = choose_subsample_indices(
        n_full,
        seed=RANDOM_STATE,
        min_size=SUBSAMPLE_MIN,
        max_size=SUBSAMPLE_MAX,
        frac_div=SUBSAMPLE_DIV,
    )

    valid_configs_small = valid_configs[subsample_idx]
    y_small = RAW_KMC_LABELS[subsample_idx]

    print(f"Full dataset size: {n_full}")
    print(
        f"Using random subsample of size {len(subsample_idx)} for target/epsilon selection"
    )

    targets, max_epsilon_no_overlap = get_targets_latin(
        f,
        space,
        valid_configs_small,
        method="kmedoids",
    )

    n_iter_max = len(subsample_idx) // 5
    save_config(
        targets,
        max_epsilon_no_overlap,
        random_state=RANDOM_STATE,
        name="kmc",
        max_iter=min(200, n_iter_max),
    )

    loaded_config = load_config()
    epsilon = loaded_config["epsilon"] * EPSILON_SCALE
    target_values = np.asarray(loaded_config["target"], dtype=np.float32)

    eval_idx = np.arange(0, len(valid_configs), EVAL_STRIDE)
    y_eval = RAW_KMC_LABELS[eval_idx]

    summary = compute_target_statistics(y_eval, target_values, epsilon)
    print_target_statistics(summary, epsilon)
