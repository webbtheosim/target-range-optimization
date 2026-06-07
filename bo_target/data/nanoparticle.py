import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from bo_target.data._paths import base_path
from bo_target.data._pipeline import (
    generate_config,
    make_bandit_function,
    remove_duplicate_rows,
    run_epsilon_analysis,
)

clean_dir = base_path / "clean"
config_dir = base_path / "config"

clean_dir.mkdir(parents=True, exist_ok=True)
config_dir.mkdir(parents=True, exist_ok=True)

data_file = base_path / "desc" / "np_synthesis.csv"
feature_file = clean_dir / "nanoparticle_features_scaled.npy"
config_file = config_dir / "nanoparticle_config.json"
label_file = clean_dir / "nanoparticle_labels.npy"
output_scaler_file = clean_dir / "nanoparticle_output_scaler.pkl"

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
    SCALED_NANOPARTICLE_DATA = np.load(feature_file)
    SCALED_NANOPARTICLE_LABELS = np.load(label_file)
    output_scaler = joblib.load(output_scaler_file)
    print(
        f"Loaded scaled features with dimension: {SCALED_NANOPARTICLE_DATA.shape[1]}"
    )

else:
    print("Processing nanoparticle dataset...")

    df_raw = pd.read_csv(data_file)

    feature_cols = df_raw.columns[1:5]
    label_cols = df_raw.columns[5:7]

    RAW_FEATURES = df_raw[feature_cols].values.astype(np.float32)
    labels = df_raw[label_cols].values.astype(np.float32)

    print(f"Loaded raw descriptors shape: {RAW_FEATURES.shape}")
    print(f"Loaded labels shape: {labels.shape} (2 objectives)")

    print(f"Original descriptor shape: {RAW_FEATURES.shape}")

    RAW_FEATURES, labels = remove_duplicate_rows(RAW_FEATURES, labels)
    n_removed = len(df_raw) - len(RAW_FEATURES)
    print(
        f"Removed {n_removed} duplicate rows -> New shape: {RAW_FEATURES.shape}"
    )

    mm = MinMaxScaler(feature_range=(0, 1))
    SCALED_NANOPARTICLE_DATA = mm.fit_transform(RAW_FEATURES)
    print(
        f"Features scaled to unit range -> final shape: {SCALED_NANOPARTICLE_DATA.shape}"
    )

    output_scaler = StandardScaler()
    SCALED_NANOPARTICLE_LABELS = output_scaler.fit_transform(labels)

    np.save(feature_file, SCALED_NANOPARTICLE_DATA)
    np.save(label_file, SCALED_NANOPARTICLE_LABELS)
    joblib.dump(output_scaler, output_scaler_file)

    print(f"Saved -> {feature_file} (shape: {SCALED_NANOPARTICLE_DATA.shape})")


def nanoparticle_function():
    """Nanoparticle synthesis (Pellegrino et al., Sci. Rep. 2020). 4 descriptors, 2 targets (radius, PDI)."""
    return make_bandit_function(SCALED_NANOPARTICLE_DATA, SCALED_NANOPARTICLE_LABELS)


if __name__ == "__main__":
    f, space, valid_configs = nanoparticle_function()

    config = generate_config(
        f, space, valid_configs, config_file, "nanoparticle",
        method="kmedoids", format_type="bandit", random_state=26,
    )

    with open(config_file) as fh:
        loaded_config = json.load(fh)

    y = f(valid_configs)
    target_values = np.asarray(loaded_config["target"])

    run_epsilon_analysis(
        y, target_values, loaded_config["epsilon"], config_dir,
        "nanoparticle", random_state=26, is_2d=True,
        analysis_type="BALL",
    )
