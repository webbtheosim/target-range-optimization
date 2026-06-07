"""Discover TDDFT molecules with near-identical spectra but diverse structures."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _aligned_smiles():
    """Return the SMILES list row-aligned to the cached feature/label arrays."""
    from bo_target.data.tddft import xlsx_file

    df = pd.read_excel(xlsx_file)
    smiles_col = next(
        (c for c in df.columns if str(c).strip().upper() == "SMILES"), None
    )
    if smiles_col is None:
        raise ValueError("Could not find SMILES column in xlsx.")
    smiles = df[smiles_col].astype(str).tolist()
    _, first_occurrence = np.unique(smiles, return_index=True)
    first_occurrence = np.sort(first_occurrence)
    return [smiles[i] for i in first_occurrence]


def _farthest_point_select(feats, k):
    """Greedy farthest-point sampling returning indices of k spread-out rows."""
    m = len(feats)
    if m <= k:
        return list(range(m))
    dist = cdist(feats, feats)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [int(i), int(j)]
    while len(selected) < k:
        min_to_sel = dist[:, selected].min(axis=1)
        min_to_sel[selected] = -np.inf
        selected.append(int(np.argmax(min_to_sel)))
    return selected


def discovered_diverse_smiles(
    acquisition="tb",
    tolerance_ratio=0.6,
    trial=0,
    top_n=5,
    save_csv=None,
    verbose=True,
):
    """Diverse discovered molecules per target: similar output, different input."""
    from bo_target.data.tddft import (
        N_GAUSSIANS,
        EPSILON_SCALE,
        GAUSSIAN_EV_LABEL_FILE,
        GAUSSIAN_EV_SCALER_FILE,
        feature_file,
        config_dir,
        GroupMinMaxScaler,
    )
    from bo_target.utils.dataset_config import dataset_config

    # targets and epsilon
    with open(config_dir / "tddft_config.json") as f:
        config = json.load(f)
    targets_scaled = np.array(config["target"])
    epsilon = config["epsilon"] * EPSILON_SCALE
    n_targets = targets_scaled.shape[0]

    # dataset arrays (row-aligned)
    labels_scaled = np.load(GAUSSIAN_EV_LABEL_FILE)
    labels_bo = labels_scaled[:, 1:]
    features = np.load(feature_file)
    smiles = _aligned_smiles()
    assert len(smiles) == labels_bo.shape[0] == features.shape[0], (
        "SMILES / labels / features are not row-aligned"
    )

    scaler = GroupMinMaxScaler.load(
        GAUSSIAN_EV_SCALER_FILE, n_gaussians=N_GAUSSIANS
    )
    phys = scaler.inverse_transform(labels_scaled)
    param_cols = ["mu1_eV", "A2", "mu2_eV", "A3", "mu3_eV"]

    # discovered indices for this run
    cfg = dataset_config("tddft", verbose=False, testing=True)
    save_bo_dir = cfg[7]
    result_path = (
        save_bo_dir / f"tddft_{acquisition}_{tolerance_ratio}_{trial}.json"
    )
    if not result_path.exists():
        raise FileNotFoundError(f"BO result not found: {result_path}")
    with open(result_path) as f:
        results = json.load(f)
    discovered = np.unique(np.array(results["indices"], dtype=int))

    # per-target selection
    out = {}
    frames = []
    for t in range(n_targets):
        # output distance (scaled) from each discovered molecule to target_t
        out_dist = np.linalg.norm(
            labels_bo[discovered] - targets_scaled[t], axis=1
        )
        hit_mask = out_dist <= epsilon
        hit_idx = discovered[hit_mask]

        if len(hit_idx) == 0:
            if verbose:
                print(
                    f"T{t + 1}: no discovered hits within epsilon={epsilon:.4f}"
                )
            out[t + 1] = pd.DataFrame()
            continue

        # most different inputs among the (output-similar) hits, in PCA space
        sel_local = _farthest_point_select(features[hit_idx], top_n)
        chosen = hit_idx[sel_local]

        # diversity score = min PCA distance to the rest of the chosen set
        chosen_feats = features[chosen]
        dmat = cdist(chosen_feats, chosen_feats)
        np.fill_diagonal(dmat, np.inf)
        diversity = dmat.min(axis=1)

        rows = []
        for rank, (idx, div) in enumerate(zip(chosen, diversity), start=1):
            rows.append({
                "target": t + 1,
                "rank": rank,
                "dataset_index": int(idx),
                "smiles": smiles[idx],
                "output_dist_to_target": float(
                    np.linalg.norm(labels_bo[idx] - targets_scaled[t])
                ),
                "input_diversity_pca": float(div),
                **{
                    c: float(phys[idx, j + 1]) for j, c in enumerate(param_cols)
                },
            })
        df_t = pd.DataFrame(rows)
        out[t + 1] = df_t
        frames.append(df_t)

        if verbose:
            print(
                f"\n=== T{t + 1}: {len(hit_idx)} discovered hit(s), "
                f"showing {len(df_t)} most input-diverse ==="
            )
            with pd.option_context(
                "display.max_colwidth", 60, "display.width", 160
            ):
                print(
                    df_t[
                        [
                            "dataset_index",
                            "smiles",
                            "output_dist_to_target",
                            "input_diversity_pca",
                        ]
                    ].to_string(index=False)
                )

    combined = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    out["all"] = combined

    if save_csv is not None and not combined.empty:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_csv, index=False)
        if verbose:
            print(f"\nSaved -> {save_csv}")

    return out


def main(argv=None):
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Find discovered molecules that hit the same TDDFT target "
            "(similar output) but differ most in molecular input."
        )
    )
    parser.add_argument(
        "-a",
        "--acquisition",
        default="tb",
        help="acquisition method / result-file prefix (default: tb)",
    )
    parser.add_argument(
        "-t",
        "--tolerance-ratio",
        type=float,
        default=0.6,
        help="tolerance ratio in the result filename (default: 0.6)",
    )
    parser.add_argument(
        "-n",
        "--trial",
        type=int,
        default=0,
        help="trial index (default: 0)",
    )
    parser.add_argument(
        "-k",
        "--top-n",
        type=int,
        default=5,
        help="number of diverse molecules per target (default: 5)",
    )
    parser.add_argument(
        "-o",
        "--save-csv",
        default=None,
        help="optional path to write the combined table as CSV",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the per-target summary printout",
    )
    args = parser.parse_args(argv)

    discovered_diverse_smiles(
        acquisition=args.acquisition,
        tolerance_ratio=args.tolerance_ratio,
        trial=args.trial,
        top_n=args.top_n,
        save_csv=args.save_csv,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
