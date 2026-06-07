import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bo_target.data._paths import base_path
from bo_target.data._pipeline import (
    apply_pca_pipeline,
    make_bandit_function,
    remove_duplicate_rows,
    smiles_to_mordred,
)

mordred_dir = base_path / "mordred"
clean_dir = base_path / "clean"
config_dir = base_path / "config"

mordred_dir.mkdir(parents=True, exist_ok=True)
clean_dir.mkdir(parents=True, exist_ok=True)
config_dir.mkdir(parents=True, exist_ok=True)

DOWNSAMPLE_RATIO = 0.05
RERUN = "None"

property_files = [
    "qm9_homo.csv",
    "qm9_lumo.csv",
]

smiles_file = base_path / "smiles" / f"{property_files[0]}"
raw_mordred_file = mordred_dir / "qm9_mordred_raw.npy"
pca_feature_file = clean_dir / "qm9_features_pca.npy"
config_file = config_dir / "qm9_config.json"
label_file = clean_dir / "qm9_labels.npy"
output_scaler_file = clean_dir / "qm9_output_scaler.pkl"


force_mordred = RERUN == "mordred"
force_pca = RERUN in ["mordred", "pca"]

if (
    not force_pca
    and pca_feature_file.exists()
    and label_file.exists()
    and output_scaler_file.exists()
):
    print("Loading precomputed PCA features, standardized labels and scaler...")
    SCALED_QM9_DATA = np.load(pca_feature_file)
    SCALED_QM9_LABELS = np.load(label_file)
    output_scaler = joblib.load(output_scaler_file)
    QM9_DATA = None
    print(f"Loaded PCA features with dimension: {SCALED_QM9_DATA.shape[1]}")
    print(f"Loaded multi-objective labels shape: {SCALED_QM9_LABELS.shape}")

else:
    print("Processing QM9 multi-property dataset...")

    df_smiles = pd.read_csv(smiles_file)
    smiles_col = next(
        (
            col
            for col in df_smiles.columns
            if str(col).lower().startswith("smiles")
        ),
        df_smiles.columns[0],
    )
    smiles_list = df_smiles[smiles_col].astype(str).tolist()

    all_labels = []
    for f in property_files:
        df = pd.read_csv(base_path / "smiles" / f"{f}")
        label_col = df.columns[-1]
        all_labels.append(df[label_col].values.astype(np.float32))

    labels = np.column_stack(all_labels)

    print(
        f"Loaded {labels.shape[1]} properties, full labels shape: {labels.shape}"
    )

    print(f"\nRandom down-sampling to {DOWNSAMPLE_RATIO:.0%} before Mordred...")
    n_original = len(smiles_list)
    n_target = int(n_original * DOWNSAMPLE_RATIO)

    np.random.seed(26)
    idx = np.random.choice(n_original, size=n_target, replace=False)
    idx = np.sort(idx)

    smiles_list = [smiles_list[i] for i in idx]
    labels = labels[idx]

    print(f"Down-sampled from {n_original} -> {len(smiles_list)} molecules")

    if not force_mordred and raw_mordred_file.exists():
        print(f"Loading existing raw Mordred from: {raw_mordred_file}")
        QM9_DATA = np.load(raw_mordred_file)
    else:
        print("Computing new Mordred descriptors on down-sampled set...")
        QM9_DATA = smiles_to_mordred(smiles_list, use_stereo=True)
        np.save(raw_mordred_file, QM9_DATA)
        print(
            f"Saved raw Mordred -> {raw_mordred_file} (shape: {QM9_DATA.shape})"
        )

    print("\nRemoving duplicate feature vectors...")
    QM9_DATA, labels = remove_duplicate_rows(QM9_DATA, labels)
    n_removed = len(smiles_list) - len(QM9_DATA)
    print(f"Removed {n_removed} duplicate rows, final shape: {QM9_DATA.shape}")

    print("\nApplying StandardScaler -> PCA (95% variance) -> MinMax[0,1]...")
    SCALED_QM9_DATA = apply_pca_pipeline(QM9_DATA)

    print("Standardizing output space (multi-objective)...")
    output_scaler = StandardScaler()
    SCALED_QM9_LABELS = output_scaler.fit_transform(labels)

    np.save(pca_feature_file, SCALED_QM9_DATA)
    np.save(label_file, SCALED_QM9_LABELS)
    joblib.dump(output_scaler, output_scaler_file)

    print(
        f"Saved PCA features -> {pca_feature_file} (shape: {SCALED_QM9_DATA.shape})"
    )
    print(
        f"Saved standardized labels -> {label_file} (shape: {SCALED_QM9_LABELS.shape})"
    )
    print(f"Saved output scaler -> {output_scaler_file}")


def qm9_function():
    """QM9 HOMO-LUMO (MoleculeNet; Wu et al., Chem. Sci. 2018). 2 target properties."""
    return make_bandit_function(SCALED_QM9_DATA, SCALED_QM9_LABELS)


if __name__ == "__main__":
    from bo_target.utils.calc_target import get_targets_latin

    f, space, valid_configs = qm9_function()

    if valid_configs is not None:
        n_data = len(valid_configs)
        n_iter_max = int(n_data / 5)
    else:
        n_iter_max = 200

    targets, max_epsilon_no_overlap = get_targets_latin(
        f, space, valid_configs, method="kmedoids"
    )

    config = {
        "global": {"random_state": 26, "max_iter": min(200, n_iter_max)},
        "name": "qm9",
        "format": "bandit" if valid_configs is not None else "surface",
        "target": targets.tolist(),
        "epsilon": round(float(max_epsilon_no_overlap), 4),
    }

    with open(config_file, "w") as config_fh:
        json.dump(config, config_fh, indent=2)

    print(f"Configuration file saved -> {config_file}")
