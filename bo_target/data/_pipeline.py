"""Shared utilities for data pipeline files."""

import json
import pickle

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from emukit.core import BanditParameter, ParameterSpace


def smiles_to_mordred(smiles_list, use_stereo=True):
    """Compute Mordred descriptors (Moriwaki et al., J. Cheminform. 2018)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from mordred import Calculator, descriptors
    from tqdm import tqdm

    calc = Calculator(descriptors, ignore_3D=False)
    mols = []

    print("Preparing molecules...")
    for smi in tqdm(smiles_list, desc="Preparing molecules", unit="mol"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mols.append(None)
            continue
        if use_stereo:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(
                mol, randomSeed=26, useExpTorsionAnglePrefs=True
            )
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        mols.append(mol)

    print("Calculating Mordred descriptors...")
    df = calc.pandas(mols, nproc=1, nmols=len(mols), quiet=False)

    df = df.dropna(axis=1)
    df = df.loc[:, df.nunique() != 1]
    df = df.select_dtypes(include=["number"])

    return df.values.astype(np.float32)


def apply_pca_pipeline(data, variance=0.95, random_state=26):
    """StandardScaler -> PCA (variance) -> MinMaxScaler[0,1]."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(data)

    pca = PCA(n_components=variance, random_state=random_state)
    X_pca = pca.fit_transform(X_std)
    n_dim = X_pca.shape[1]
    print(f"PCA reduced to {n_dim} dimensions ({variance:.0%} variance)")

    mm = MinMaxScaler(feature_range=(0, 1))
    return mm.fit_transform(X_pca)


def remove_duplicate_rows(data, labels):
    """Remove duplicate feature vectors, keeping first occurrence."""
    _, unique_idx = np.unique(data, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return data[unique_idx], labels[unique_idx]


def make_bandit_function(scaled_data, scaled_labels):
    """Create a bandit (f, parameter_space, valid_configs) triple."""
    n_features = scaled_data.shape[1]
    sub_parameter_names = [f"x{i + 1}" for i in range(n_features)]

    bandit_param = BanditParameter(
        "config", scaled_data, sub_parameter_names=sub_parameter_names
    )
    parameter_space = ParameterSpace([bandit_param])

    def _lookup(x):
        dist = cdist(x, scaled_data, metric="euclidean")
        closest_idx = dist.argmin(axis=1)
        return scaled_labels[closest_idx]

    return _lookup, parameter_space, scaled_data


def generate_config(
    f,
    space,
    valid_configs,
    config_file,
    dataset_name,
    method="kmedoids",
    format_type="bandit",
    random_state=26,
    max_iter_override=None,
    **extra_config,
):
    """Select target points via Latin hypercube + k-medoids, compute no-overlap epsilon, save BO config JSON."""
    from bo_target.utils.calc_target import get_targets_latin

    np.random.seed(random_state)
    targets, max_epsilon_no_overlap = get_targets_latin(
        f, space, valid_configs, method=method
    )

    if max_iter_override is not None:
        n_iter_max = max_iter_override
    elif valid_configs is not None:
        n_iter_max = int(len(valid_configs) / 5)
    else:
        n_iter_max = 200

    config = {
        "global": {
            "random_state": random_state,
            "max_iter": min(200, n_iter_max),
        },
        "name": dataset_name,
        "format": format_type if valid_configs is not None else "surface",
        "target": targets.tolist(),
        "epsilon": round(float(max_epsilon_no_overlap), 4),
        **extra_config,
    }

    with open(config_file, "w") as fh:
        json.dump(config, fh, indent=2)

    print(f"Config saved -> {config_file}")
    return config


def run_epsilon_analysis(
    y,
    target_values,
    base_epsilon,
    config_dir,
    dataset_name,
    random_state=26,
    is_2d=False,
    label="points",
    analysis_type="INTERVAL",
):
    """Multi-epsilon coverage analysis: what fraction of data falls within epsilon-balls of each target."""
    multipliers = [0.2, 0.4, 0.6]
    analysis_results = {}
    output_pickle = config_dir / f"{dataset_name}_epsilon_analysis.pkl"

    print(f"\n{'=' * 60}")
    print(f"MULTI-EPSILON {analysis_type} ANALYSIS -- {dataset_name.upper()}")
    print("=" * 60)

    for mult in multipliers:
        epsilon = base_epsilon * mult
        print(f"\n-> Multiplier = {mult} | epsilon = {epsilon:.4f}")

        if is_2d:
            dist_to_targets = cdist(y, target_values, metric="euclidean")
        else:
            dist_to_targets = np.abs(y[:, np.newaxis] - target_values)

        inside_any = (dist_to_targets <= epsilon).any(axis=1)
        counts_per_target = (dist_to_targets <= epsilon).sum(axis=0)

        n_inside = (dist_to_targets <= epsilon).astype(int).sum(axis=1)
        exactly_one = (n_inside == 1).sum()
        multiple = (n_inside > 1).sum()
        none = (n_inside == 0).sum()

        print(f"   Total {label}: {len(y)}")
        print(
            f"   Inside any: {inside_any.sum():5d} "
            f"({inside_any.sum() / len(y) * 100:5.1f}%)"
        )
        print(
            f"   Exactly one: {exactly_one:5d} "
            f"({exactly_one / len(y) * 100:5.1f}%)"
        )
        print(f"   Multiple: {multiple:5d} ({multiple / len(y) * 100:5.1f}%)")
        print(f"   Outside all: {none:5d} ({none / len(y) * 100:5.1f}%)")

        print("   Points per target:")
        for i, cnt in enumerate(counts_per_target):
            print(f"     Target {i:2d}: {cnt:5d} ({cnt / len(y) * 100:5.1f}%)")

        analysis_results[mult] = {
            "epsilon": float(epsilon),
            "multiplier": mult,
            "base_epsilon": float(base_epsilon),
            "n_total": len(y),
            "counts_per_target": counts_per_target.tolist(),
            "inside_any": int(inside_any.sum()),
            "exactly_one": int(exactly_one),
            "multiple": int(multiple),
            "none": int(none),
            "coverage_percent": float(inside_any.sum() / len(y) * 100),
            "random_state": random_state,
        }

    with open(output_pickle, "wb") as f:
        pickle.dump(analysis_results, f)

    print(f"\nAnalysis saved -> {output_pickle}")


def process_smiles_dataset(
    smiles_file,
    raw_mordred_file,
    pca_feature_file,
    label_file,
    output_scaler_file,
    dataset_name,
    force_mordred=False,
    force_pca=False,
):
    """Load cached or build: CSV -> Mordred -> dedupe -> PCA -> standardize -> save."""
    import joblib

    import pandas as pd

    if (
        not force_pca
        and pca_feature_file.exists()
        and label_file.exists()
        and output_scaler_file.exists()
    ):
        print(
            "Loading precomputed PCA features, standardized labels and scaler..."
        )
        scaled_data = np.load(pca_feature_file)
        scaled_labels = np.load(label_file).reshape(-1, 1)
        output_scaler = joblib.load(output_scaler_file)
        print(f"Loaded PCA features with dimension: {scaled_data.shape[1]}")
        return scaled_data, scaled_labels, output_scaler

    print(f"Processing {dataset_name} dataset...")

    df_raw = pd.read_csv(smiles_file)

    smiles_col = next(
        (
            col
            for col in df_raw.columns
            if str(col).lower().startswith("smiles")
        ),
        df_raw.columns[0],
    )
    label_col = df_raw.columns[-1]

    if not force_mordred and raw_mordred_file.exists():
        print(f"Loading raw Mordred from: {raw_mordred_file}")
        data = np.load(raw_mordred_file)
        labels = df_raw[label_col].values
    else:
        print("Computing Mordred descriptors...")
        smiles_list = df_raw[smiles_col].astype(str).tolist()
        labels = df_raw[label_col].values
        data = smiles_to_mordred(smiles_list, use_stereo=True)
        np.save(raw_mordred_file, data)
        print(f"Saved raw Mordred -> {raw_mordred_file} (shape: {data.shape})")

    print(f"Original Mordred shape: {data.shape}")

    data, labels = remove_duplicate_rows(data, labels)
    n_removed = len(df_raw) - len(data)
    print(f"Removed {n_removed} duplicate rows -> New shape: {data.shape}")

    scaled_data = apply_pca_pipeline(data)

    output_scaler = StandardScaler()
    scaled_labels = output_scaler.fit_transform(
        labels.astype(np.float32).reshape(-1, 1)
    )

    np.save(pca_feature_file, scaled_data)
    np.save(label_file, scaled_labels.ravel())
    joblib.dump(output_scaler, output_scaler_file)

    print(f"Saved -> {pca_feature_file} (shape: {scaled_data.shape})")

    return scaled_data, scaled_labels, output_scaler
