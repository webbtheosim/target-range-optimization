import json

import numpy as np

from bo_target.data._paths import base_path
from bo_target.data._pipeline import (
    generate_config,
    make_bandit_function,
    process_smiles_dataset,
    run_epsilon_analysis,
)

mordred_dir = base_path / "mordred"
clean_dir = base_path / "clean"
config_dir = base_path / "config"

mordred_dir.mkdir(parents=True, exist_ok=True)
clean_dir.mkdir(parents=True, exist_ok=True)
config_dir.mkdir(parents=True, exist_ok=True)

smiles_file = base_path / "smiles" / "lipo.csv"
raw_mordred_file = mordred_dir / "lipo_mordred_raw.npy"
pca_feature_file = clean_dir / "lipo_features_pca.npy"
config_file = config_dir / "lipo_config.json"
label_file = clean_dir / "lipo_labels.npy"
output_scaler_file = clean_dir / "lipo_output_scaler.pkl"

RERUN = "None"


force_mordred = RERUN == "mordred"
force_pca = RERUN in ["mordred", "pca"]

SCALED_LIPO_DATA, SCALED_LIPO_LABELS, _output_scaler = process_smiles_dataset(
    smiles_file,
    raw_mordred_file,
    pca_feature_file,
    label_file,
    output_scaler_file,
    "LIPO",
    force_mordred,
    force_pca,
)


def lipo_function():
    """Lipophilicity (MoleculeNet; Wu et al., Chem. Sci. 2018)."""
    return make_bandit_function(SCALED_LIPO_DATA, SCALED_LIPO_LABELS)


if __name__ == "__main__":
    f, space, valid_configs = lipo_function()

    config = generate_config(
        f, space, valid_configs, config_file, "lipo",
        method="kmedoids", format_type="bandit", random_state=26,
    )

    with open(config_file) as fh:
        loaded_config = json.load(fh)

    y = f(valid_configs).flatten()
    target_values = np.asarray(loaded_config["target"]).flatten()

    run_epsilon_analysis(
        y, target_values, loaded_config["epsilon"], config_dir,
        "lipo", random_state=26, is_2d=False,
    )
