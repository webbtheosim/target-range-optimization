
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from bo_target.data._paths import base_path
from bo_target.data._pipeline import (
    generate_config,
    make_bandit_function,
    remove_duplicate_rows,
)

clean_dir = base_path / "clean"
config_dir = base_path / "config"

clean_dir.mkdir(parents=True, exist_ok=True)
config_dir.mkdir(parents=True, exist_ok=True)

feature_data_file = base_path / "desc" / "propensity_features.csv"
label_data_file = base_path / "desc" / "propensity_labels.csv"

feature_file = clean_dir / "propensity_features_scaled.npy"
config_file = config_dir / "propensity_config.json"
label_file = clean_dir / "propensity_labels.npy"
output_scaler_file = clean_dir / "propensity_output_scaler.pkl"

RERUN = "None"

force_rerun = RERUN in ["features"]

needs_recompute = (
    force_rerun
    or not feature_file.exists()
    or not label_file.exists()
    or not output_scaler_file.exists()
)

if not needs_recompute:
    print(
        "Loading precomputed scaled features, standardized labels and scaler..."
    )
    SCALED_PROPENSITY_DATA = np.load(feature_file)
    SCALED_PROPENSITY_LABELS = np.load(label_file)
    output_scaler = joblib.load(output_scaler_file)

    col_means = SCALED_PROPENSITY_LABELS.mean(axis=0)
    col_stds = SCALED_PROPENSITY_LABELS.std(axis=0)
    is_standardized = np.allclose(col_means, 0.0, atol=1e-3) and np.allclose(
        col_stds, 1.0, atol=1e-3
    )
    if not is_standardized:
        print(
            "Cached labels are NOT standardized "
            f"(mean={col_means}, std={col_stds}); recomputing."
        )
        needs_recompute = True
    else:
        print(
            f"Loaded scaled features with dimension: {SCALED_PROPENSITY_DATA.shape[1]}"
        )
        print(
            f"Loaded standardized labels with shape: {SCALED_PROPENSITY_LABELS.shape} (3 objectives)"
        )
        print("Output labels are standardized (mean=0, std=1 per property)")

if needs_recompute:
    print("Processing propensity dataset...")

    RAW_FEATURES = np.loadtxt(
        feature_data_file, delimiter=",", skiprows=1
    ).astype(np.float32)

    labels_full = np.genfromtxt(
        label_data_file, delimiter=",", skip_header=1, filling_values=0
    )
    labels = labels_full[:, [4, 8, 10]].astype(np.float32)

    print(f"Loaded raw features shape: {RAW_FEATURES.shape}")
    print(f"Loaded labels shape: {labels.shape} (3 objectives)")

    print("Removing duplicate feature vectors...")
    original_n = RAW_FEATURES.shape[0]
    print(f"Original descriptor shape: {RAW_FEATURES.shape}")

    RAW_FEATURES, labels = remove_duplicate_rows(RAW_FEATURES, labels)
    n_removed = original_n - len(RAW_FEATURES)
    print(
        f"Removed {n_removed} duplicate rows -> New shape: {RAW_FEATURES.shape}"
    )

    print("Applying MinMaxScaler [0,1] to raw features...")
    mm = MinMaxScaler(feature_range=(0, 1))
    SCALED_PROPENSITY_DATA = mm.fit_transform(RAW_FEATURES)
    print(
        f"Features scaled to unit range -> final shape: {SCALED_PROPENSITY_DATA.shape}"
    )

    print("Standardizing output space (3 objectives)...")
    output_scaler = StandardScaler()
    SCALED_PROPENSITY_LABELS = output_scaler.fit_transform(labels)

    np.save(feature_file, SCALED_PROPENSITY_DATA)
    np.save(label_file, SCALED_PROPENSITY_LABELS)
    joblib.dump(output_scaler, output_scaler_file)

    print(
        f"Saved scaled features -> {feature_file} (shape: {SCALED_PROPENSITY_DATA.shape})"
    )
    print(
        f"Saved standardized labels -> {label_file} (shape: {SCALED_PROPENSITY_LABELS.shape})"
    )
    print(f"Saved output scaler -> {output_scaler_file}")


def propensity_function():
    """Intrinsically disordered proteins (Oliver, Jacobs & Webb, J. Phys. Chem. B 2025). 3 propensity scores as targets."""
    return make_bandit_function(SCALED_PROPENSITY_DATA, SCALED_PROPENSITY_LABELS)


if __name__ == "__main__":
    f, space, valid_configs = propensity_function()

    generate_config(
        f, space, valid_configs, config_file, "propensity",
        method="kmedoids", format_type="bandit", random_state=26,
    )
