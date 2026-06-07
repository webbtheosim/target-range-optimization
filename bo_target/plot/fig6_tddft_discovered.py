"""Figure 6: TDDFT discovered spectra per target."""

import json
from pathlib import Path

import numpy as np
import ultraplot as pplt

from bo_target.plot.format_ax import format_ax


def plot_tddft_discovered(save_dir=None):
    """Plot, per target, the target spectrum and the closest dataset spectrum."""
    from bo_target.data.tddft import (
        WAVELENGTH_START, WAVELENGTH_END,
        N_GAUSSIANS, FIXED_SIGMA_EV, EPSILON_SCALE,
        GAUSSIAN_EV_SCALER_FILE, config_dir, label_file,
        wl_to_ev, GroupMinMaxScaler,
        gaussian_params_to_spectrum_ev,
    )

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[3] / "bo_fig"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load config and targets
    with open(config_dir / "tddft_config.json") as f:
        config = json.load(f)
    targets_scaled = np.array(config["target"])
    epsilon_base = config["epsilon"]
    epsilon = epsilon_base * EPSILON_SCALE

    scaler = GroupMinMaxScaler.load(GAUSSIAN_EV_SCALER_FILE,
                                     n_gaussians=N_GAUSSIANS)
    # Reconstruct full target params
    targets_full = np.zeros((targets_scaled.shape[0], N_GAUSSIANS * 2))
    targets_full[:, 0] = 1.0
    targets_full[:, 1:] = targets_scaled
    targets_phys = scaler.inverse_transform(targets_full)

    # spectra
    raw_spectra = np.load(label_file)
    wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, raw_spectra.shape[1])
    energies = wl_to_ev(wavelengths)
    e_sort = np.argsort(energies)
    energies_asc = energies[e_sort]

    # Reconstruct target spectra
    target_spec = np.zeros((targets_phys.shape[0], len(energies_asc)))
    for t in range(targets_phys.shape[0]):
        recon = gaussian_params_to_spectrum_ev(
            targets_phys[t].reshape(1, -1), wavelengths
        )
        target_spec[t] = recon[e_sort] if recon.ndim == 1 else recon[0, e_sort]

    # for each target, find best-matching discovered molecule
    colors = ["#E4256D", "#2495C1", "#F5A623", "#7B4B94", "#3CA474"]
    n_targets = targets_scaled.shape[0]

    # Load cached labels and find molecules matching each target
    from bo_target.data.tddft import GAUSSIAN_EV_LABEL_FILE
    all_labels_scaled = np.load(GAUSSIAN_EV_LABEL_FILE)  # (N, 6) full params
    all_labels_bo = all_labels_scaled[:, 1:]              # (N, 5) BO target

    fig, axes = pplt.subplots(
        nrows=n_targets, ncols=1,
        journal="nat1",
        refheight=1.3,
        sharex=True,
    )

    for t, ax in enumerate(axes):
        ax.plot(energies_asc, target_spec[t], color=colors[t], lw=1.5,
                label=f"Target T$_{t + 1}$")
        ax.fill_between(
            energies_asc,
            np.maximum(0, target_spec[t] - epsilon),
            target_spec[t] + epsilon,
            color=colors[t], alpha=0.08,
        )

        # Find molecule closest to this target in BO space
        dists = np.linalg.norm(all_labels_bo - targets_scaled[t], axis=1)
        best_idx = int(np.argmin(dists))
        best_spec = raw_spectra[best_idx][e_sort]
        ax.plot(energies_asc, best_spec, color="0.2", lw=1.0,
                label=f"Closest")
        ax.annotate(
            f"#{best_idx}",
            xy=(energies_asc[-1], best_spec[-1]),
            fontsize=6, color="0.3", ha="left", va="bottom",
        )

        ax.set_ylabel(f"T$_{t + 1}$", fontsize=8, color=colors[t])
        ax.set_ylim(-0.02, target_spec[t].max() * 1.4)
        ax.legend(loc="ur", fontsize=6, ncols=1, framealpha=0.8)
        format_ax(ax)

    axes[-1].set_xlabel("Energy (eV)")
    axes[0].set_xlim(energies_asc[0], energies_asc[-1])
    axes[0].invert_xaxis()

    for fmt in ("png", "svg"):
        out_path = save_dir / f"fig6_tddft_discovered.{fmt}"
        fig.savefig(out_path, dpi=300 if fmt == "png" else None,
                    bbox_inches="tight")
        print(f"Saved -> {out_path}")

    return fig, axes


if __name__ == "__main__":
    plot_tddft_discovered()
    pplt.show()
