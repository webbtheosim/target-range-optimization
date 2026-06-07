import json

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from tqdm import tqdm

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
smiles_dir = base_path / "smiles"

mordred_dir.mkdir(parents=True, exist_ok=True)
clean_dir.mkdir(parents=True, exist_ok=True)
config_dir.mkdir(parents=True, exist_ok=True)

xlsx_file = smiles_dir / "tddft_spectra.xlsx"
raw_mordred_file = mordred_dir / "tddft_mordred_raw.npy"
feature_file = clean_dir / "tddft_features_pca.npy"
label_file = clean_dir / "tddft_labels_raw.npy"
config_file = config_dir / "tddft_config.json"

RERUN = "None"
force_mordred = RERUN == "mordred"
force_pca = RERUN in ("mordred", "pca")
force_gaussian = RERUN in ("mordred", "pca", "gaussian")

RANDOM_STATE = 26
WAVELENGTH_START = 240
WAVELENGTH_END = 500
EPSILON_SCALE = 0.6
N_GAUSSIANS = 3
FIXED_SIGMA_EV = 0.12
N_TARGETS = 5
HC_EV_NM = 1239.841984

GAUSSIAN_EV_LABEL_FILE = (
    clean_dir / f"tddft_labels_gaussian_ev_k{N_GAUSSIANS}.npy"
)
GAUSSIAN_EV_SCALER_FILE = (
    clean_dir / f"tddft_gaussian_ev_scaler_k{N_GAUSSIANS}.json"
)


class GroupMinMaxScaler:
    """Scale Gaussian parameters group-wise to [0, 1]."""

    def __init__(self, n_gaussians=5, params_per_peak=2):
        self.n_gaussians = n_gaussians
        self.params_per_peak = params_per_peak
        if params_per_peak == 3:
            self.groups = {
                "A": [3 * k for k in range(n_gaussians)],
                "mu": [3 * k + 1 for k in range(n_gaussians)],
                "sigma": [3 * k + 2 for k in range(n_gaussians)],
            }
        else:
            self.groups = {
                "A": [2 * k for k in range(n_gaussians)],
                "mu": [2 * k + 1 for k in range(n_gaussians)],
            }
        self.mins = {}
        self.maxs = {}

    def fit(self, X):
        X = np.asarray(X)
        for name, cols in self.groups.items():
            self.mins[name] = float(X[:, cols].min())
            self.maxs[name] = float(X[:, cols].max())
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for name, cols in self.groups.items():
            lo, hi = self.mins[name], self.maxs[name]
            if hi > lo:
                X[:, cols] = (X[:, cols] - lo) / (hi - lo)
            else:
                X[:, cols] = 0.0
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for name, cols in self.groups.items():
            lo, hi = self.mins[name], self.maxs[name]
            X[:, cols] = X[:, cols] * (hi - lo) + lo
        return X

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(
                {
                    "mins": self.mins,
                    "maxs": self.maxs,
                    "n_gaussians": self.n_gaussians,
                    "params_per_peak": self.params_per_peak,
                },
                fh,
            )

    @classmethod
    def load(cls, path, n_gaussians=None):
        with open(path, "r") as fh:
            data = json.load(fh)
        n_g = n_gaussians or data.get("n_gaussians", 3)
        ppp = data.get("params_per_peak", 3)
        obj = cls(n_gaussians=n_g, params_per_peak=ppp)
        obj.mins = data["mins"]
        obj.maxs = data["maxs"]
        return obj


def wl_to_ev(wavelengths_nm):
    """Convert wavelength (nm) to energy (eV)."""
    return HC_EV_NM / np.asarray(wavelengths_nm, dtype=float)


def gaussian_params_to_spectrum_ev(params, wavelengths):
    """Reconstruct spectrum from Gaussian params [A1, mu1, A2, mu2, ...] in eV."""
    energies = wl_to_ev(wavelengths)
    n_gaussians = params.shape[1] // 2
    spectrum = np.zeros((params.shape[0], len(wavelengths)))
    for k in range(n_gaussians):
        A_k = params[:, 2 * k]
        mu_k = params[:, 2 * k + 1]
        for i in range(params.shape[0]):
            spectrum[i] += A_k[i] * np.exp(-0.5 * ((energies - mu_k[i]) / FIXED_SIGMA_EV) ** 2)
    return spectrum


def spectrum_to_gaussian_params_ev(wavelengths, spectrum, n_gaussians=5):
    """Fit K Gaussians in eV space with fixed sigma, return [A1, E1, A2, E2, ...]."""
    energies = wl_to_ev(wavelengths)
    e_sort = np.argsort(energies)
    energies_asc = energies[e_sort]
    spectrum_asc = np.asarray(spectrum)[e_sort]
    e_min, e_max = energies_asc[0], energies_asc[-1]

    peaks, props = find_peaks(
        spectrum_asc, height=0.005, distance=3, prominence=0.002
    )

    if len(peaks) == 0:
        params = np.zeros((n_gaussians, 2))
        params[:, 1] = np.linspace(e_min + 0.15, e_max - 0.15, n_gaussians)
        return params.flatten()

    peak_order = np.argsort(spectrum_asc[peaks])[::-1]
    peaks = peaks[peak_order][:n_gaussians]

    p0 = []
    bounds_low = []
    bounds_high = []

    for idx in peaks:
        amp = float(spectrum_asc[idx])
        mu_e = float(energies_asc[idx])
        p0.extend([amp, mu_e])
        bounds_low.extend([0.0, e_min])
        bounds_high.extend([2.0, e_max])

    while len(p0) < n_gaussians * 2:
        p0.extend([0.001, float(np.median(energies_asc))])
        bounds_low.extend([0.0, e_min])
        bounds_high.extend([0.01, e_max])

    def _model(x, *par):
        y = np.zeros_like(x)
        for k in range(n_gaussians):
            a = par[2 * k]
            mu_k = par[2 * k + 1]
            y += a * np.exp(-0.5 * ((x - mu_k) / FIXED_SIGMA_EV) ** 2)
        return y

    popt, _ = curve_fit(
        _model,
        energies_asc,
        spectrum_asc,
        p0=p0,
        bounds=(bounds_low, bounds_high),
        maxfev=15000,
        xtol=1e-8,
        ftol=1e-8,
    )

    params = np.array(popt).reshape(n_gaussians, 2)
    params = params[np.argsort(params[:, 0])[::-1]]

    amp_thresh = 0.02
    canon_e = float(np.median(energies_asc))
    unused = params[:, 0] < amp_thresh
    params[unused, 0] = 0.0
    params[unused, 1] = canon_e

    return params.flatten()


def spectra_to_gaussians_ev(
    spectra, wavelength_start=240, wavelength_end=500, n_gaussians=5
):
    """Batch-convert spectra to eV-space Gaussian params (A, mu only)."""
    wavelengths = np.linspace(
        wavelength_start, wavelength_end, spectra.shape[1]
    )
    n_spectra = spectra.shape[0]
    gaussian_params = np.empty((n_spectra, n_gaussians * 2), dtype=np.float64)

    for i in tqdm(
        range(n_spectra),
        desc="Fitting Gaussians (eV, fixed sigma)",
        unit="spec",
    ):
        gaussian_params[i] = spectrum_to_gaussian_params_ev(
            wavelengths, spectra[i], n_gaussians=n_gaussians
        )

    return gaussian_params


def _build_and_cache_gaussian_labels_ev(raw_spectra):
    """Fit K Gaussians in eV space (sigma fixed), scale, cache."""
    print(
        f"\nFitting {N_GAUSSIANS} Gaussians in eV space (sigma={FIXED_SIGMA_EV} eV fixed)..."
    )
    gaussian_raw = spectra_to_gaussians_ev(
        raw_spectra,
        wavelength_start=WAVELENGTH_START,
        wavelength_end=WAVELENGTH_END,
        n_gaussians=N_GAUSSIANS,
    )
    scaler = GroupMinMaxScaler(n_gaussians=N_GAUSSIANS, params_per_peak=2).fit(
        gaussian_raw
    )
    gaussian = scaler.transform(gaussian_raw)
    for name, lo, hi in zip(
        ["A", "E(eV)"], scaler.mins.values(), scaler.maxs.values()
    ):
        print(f"  {name}: [{lo:.3f}, {hi:.3f}] -> [0, 1]")
    np.save(GAUSSIAN_EV_LABEL_FILE, gaussian)
    scaler.save(GAUSSIAN_EV_SCALER_FILE)
    print(
        f"Saved eV Gaussian labels -> {GAUSSIAN_EV_LABEL_FILE} (shape: {gaussian.shape})"
    )
    print(f"Saved eV Gaussian scaler -> {GAUSSIAN_EV_SCALER_FILE}")
    return gaussian


def load_or_build_tddft_data(
    force_mordred=False, force_pca=False, force_gaussian=False
):
    """Load cached data or build from scratch (Mordred -> PCA -> eV Gaussian labels)."""
    rebuild = force_pca or force_mordred or force_gaussian

    if (
        not rebuild
        and feature_file.exists()
        and GAUSSIAN_EV_LABEL_FILE.exists()
    ):
        x = np.load(feature_file)
        y = np.load(GAUSSIAN_EV_LABEL_FILE)
        print("Loading precomputed PCA features and eV Gaussian labels...")
        print(f"Loaded PCA features with dimension: {x.shape[1]}")
        print(f"Loaded eV Gaussian labels with shape: {y.shape}")
        return x, y

    if not rebuild and feature_file.exists() and label_file.exists():
        print(
            "Features cached; rebuilding eV Gaussian labels from cached spectra..."
        )
        x = np.load(feature_file)
        raw_spectra = np.load(label_file)
        return x, _build_and_cache_gaussian_labels_ev(raw_spectra)

    print("Processing TDDFT dataset...")
    df_raw = pd.read_excel(xlsx_file)
    print(f"Raw xlsx shape: {df_raw.shape}")

    smiles_col = next(
        (col for col in df_raw.columns if str(col).strip().upper() == "SMILES"),
        None,
    )
    if smiles_col is None:
        raise ValueError("Could not find SMILES column in xlsx.")
    smiles_list = df_raw[smiles_col].astype(str).tolist()
    print(f"SMILES column: '{smiles_col}', {len(smiles_list)} entries")

    spectrum_cols = []
    for col in df_raw.columns:
        if isinstance(col, (int, float)) and 240 <= col <= 500:
            spectrum_cols.append(col)
    spectrum_cols = sorted(spectrum_cols)
    if not spectrum_cols:
        spectrum_cols = [
            c
            for c in df_raw.select_dtypes(include="number").columns
            if str(c).strip().upper() != "SMILES"
        ]
    n_bins = len(spectrum_cols)
    print(
        f"Spectrum bins: {n_bins} columns, range {spectrum_cols[0]}-{spectrum_cols[-1]}"
    )

    raw_spectra = df_raw[spectrum_cols].values.astype(np.float64)
    row_max = raw_spectra.max(axis=1, keepdims=True)
    zero_mask = (row_max == 0).ravel()
    if zero_mask.any():
        print(
            f"WARNING: {zero_mask.sum()} rows have zero spectra; leaving as all-zeros."
        )
        row_max[zero_mask] = 1.0
    raw_spectra = raw_spectra / row_max
    peak_heights = raw_spectra.max(axis=1)
    print(
        f"Spectra normalized to max peak=1. Peak heights: min={peak_heights.min():.6f}, max={peak_heights.max():.6f}"
    )

    _, first_occurrence = np.unique(smiles_list, return_index=True)
    first_occurrence = np.sort(first_occurrence)
    n_smi_removed = len(smiles_list) - len(first_occurrence)
    if n_smi_removed > 0:
        print(f"Removed {n_smi_removed} duplicate SMILES rows.")
        smiles_list = [smiles_list[i] for i in first_occurrence]
        raw_spectra = raw_spectra[first_occurrence]
        df_raw = df_raw.iloc[first_occurrence]

    if not force_mordred and raw_mordred_file.exists():
        print(f"Loading existing raw Mordred from: {raw_mordred_file}")
        mordred_data = np.load(raw_mordred_file)
    else:
        print("Computing new Mordred descriptors...")
        mordred_data = smiles_to_mordred(smiles_list, use_stereo=True)
        np.save(raw_mordred_file, mordred_data)
        print(
            f"Saved raw Mordred -> {raw_mordred_file} (shape: {mordred_data.shape})"
        )

    print(f"Original Mordred shape: {mordred_data.shape}")
    mordred_data, raw_spectra = remove_duplicate_rows(mordred_data, raw_spectra)
    n_removed = len(df_raw) - len(mordred_data)
    print(
        f"Removed {n_removed} duplicate rows -> New shape: {mordred_data.shape}"
    )

    print("Applying StandardScaler -> PCA (95% var) -> MinMaxScaler[0,1]...")
    scaled_x = apply_pca_pipeline(mordred_data, random_state=RANDOM_STATE)
    np.save(feature_file, scaled_x)
    np.save(label_file, raw_spectra)
    print(f"Saved scaled features -> {feature_file} (shape: {scaled_x.shape})")
    print(
        f"Saved raw spectra (backup) -> {label_file} (shape: {raw_spectra.shape})"
    )

    gaussian = _build_and_cache_gaussian_labels_ev(raw_spectra)
    return scaled_x, gaussian


SCALED_TDDFT_DATA, RAW_TDDFT_LABELS_FULL = load_or_build_tddft_data(
    force_mordred=force_mordred,
    force_pca=force_pca,
    force_gaussian=force_gaussian,
)
RAW_TDDFT_LABELS = RAW_TDDFT_LABELS_FULL[:, 1:]


def tddft_function():
    """Conjugated molecule TD-DFT spectra. Gaussian-fit eV-space parameters as targets."""
    return make_bandit_function(SCALED_TDDFT_DATA, RAW_TDDFT_LABELS)


def save_config(targets, epsilon, random_state=26, name="tddft", max_iter=200):
    config = {
        "global": {"random_state": random_state, "max_iter": max_iter},
        "name": name,
        "format": "bandit",
        "target": targets.tolist(),
        "epsilon": round(float(epsilon), 4),
    }
    with open(config_file, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"Config saved -> {config_file}")
    return config


def load_config():
    with open(config_file, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _band_counts(labels, targets, epsilon):
    return (cdist(labels, targets) <= epsilon).sum(axis=0)


if __name__ == "__main__":
    from bo_target.utils.calc_target import get_targets_latin

    f, space, valid_configs = tddft_function()

    n_full = len(valid_configs)
    print(f"\nFull dataset size: {n_full}")
    print(
        f"Target: eV-space Gaussian params ({RAW_TDDFT_LABELS.shape[1]}D, K={N_GAUSSIANS}, A1 dropped)"
    )

    candidates = {}
    for method in ("kmedoids", "density_filtered"):
        t, eps_no = get_targets_latin(
            f, space, valid_configs, method=method, k=N_TARGETS
        )
        counts = _band_counts(RAW_TDDFT_LABELS, t, EPSILON_SCALE * eps_no)
        candidates[method] = (t, eps_no, counts, int(counts.min()))
        print(
            f"  {method:16s} band counts@op-eps={counts.tolist()} min={int(counts.min())} eps_base={eps_no:.4f}"
        )

    best_method = max(candidates, key=lambda m: candidates[m][3])
    targets, max_epsilon_no_overlap, counts, _ = candidates[best_method]
    print(f"Selected '{best_method}' (most balanced min band count).")

    save_config(
        targets,
        max_epsilon_no_overlap,
        random_state=RANDOM_STATE,
        name="tddft",
        max_iter=min(200, n_full // 5),
    )

    loaded_config = load_config()
    epsilon = loaded_config["epsilon"] * EPSILON_SCALE
    print(f"\nMax epsilon (no overlap): {max_epsilon_no_overlap:.6f}")
    print(f"Diagnostic epsilon (x{EPSILON_SCALE}): {epsilon:.6f}")
    print(f"Gaussian-parameter output dim: {RAW_TDDFT_LABELS.shape[1]}")
    print(f"Config saved -> {config_file}")
