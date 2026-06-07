
import joblib
import numpy as np
import pandas as pd
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

data_file = base_path / "desc" / "toporg.csv"
feature_file = clean_dir / "toporg_features_scaled.npy"
config_file = config_dir / "toporg_config.json"
label_file = clean_dir / "toporg_labels.npy"
output_scaler_file = clean_dir / "toporg_output_scaler.pkl"

RERUN = "None"

force_rerun = RERUN in ["features"]

if (
    not force_rerun
    and feature_file.exists()
    and label_file.exists()
    and output_scaler_file.exists()
):
    print(
        "Loading precomputed scaled features, standardized labels and scaler..."
    )
    SCALED_TOPORG_DATA = np.load(feature_file)
    SCALED_TOPORG_LABELS = np.load(label_file)
    output_scaler = joblib.load(output_scaler_file)
    print(
        f"Loaded scaled features with dimension: {SCALED_TOPORG_DATA.shape[1]}"
    )

else:
    print("Processing toporg dataset...")

    df_raw = pd.read_csv(data_file)

    feature_cols = df_raw.columns[1:9]
    label_cols = df_raw.columns[-1:]

    RAW_FEATURES = df_raw[feature_cols].values.astype(np.float32)
    labels = df_raw[label_cols].values.astype(np.float32)

    print(f"Loaded raw features shape: {RAW_FEATURES.shape} (8 descriptors)")
    print(f"Loaded labels shape: {labels.shape} (1 objective)")

    print("\nRemoving duplicate feature vectors...")
    print(f"Original descriptor shape: {RAW_FEATURES.shape}")

    RAW_FEATURES, labels = remove_duplicate_rows(RAW_FEATURES, labels)
    n_removed = len(df_raw) - len(RAW_FEATURES)
    print(
        f"Removed {n_removed} duplicate rows, new shape: {RAW_FEATURES.shape}"
    )

    print("\nApplying MinMaxScaler [0,1] to raw descriptors...")
    mm = MinMaxScaler(feature_range=(0, 1))
    SCALED_TOPORG_DATA = mm.fit_transform(RAW_FEATURES)
    print(
        f"Features scaled to unit range, final shape: {SCALED_TOPORG_DATA.shape}"
    )

    print("Standardizing output space...")
    output_scaler = StandardScaler()
    SCALED_TOPORG_LABELS = output_scaler.fit_transform(labels)

    np.save(feature_file, SCALED_TOPORG_DATA)
    np.save(label_file, SCALED_TOPORG_LABELS)
    joblib.dump(output_scaler, output_scaler_file)

    print(
        f"Saved scaled features -> {feature_file} (shape: {SCALED_TOPORG_DATA.shape})"
    )
    print(
        f"Saved standardized labels -> {label_file} (shape: {SCALED_TOPORG_LABELS.shape})"
    )
    print(f"Saved output scaler -> {output_scaler_file}")


def toporg_function():
    """Polymer topology (Jiang, Dieng & Webb, npj Comput. Mater. 2024). 8 VAE latent descriptors, target: radius of gyration."""
    return make_bandit_function(SCALED_TOPORG_DATA, SCALED_TOPORG_LABELS)


if __name__ == "__main__":
    f, space, valid_configs = toporg_function()

    generate_config(
        f, space, valid_configs, config_file, "toporg",
        method="kmedoids", format_type="bandit", random_state=26,
    )
