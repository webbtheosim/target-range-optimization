"""Figure 6: TDDFT target absorption profiles in eV space."""

import json
from pathlib import Path

import numpy as np
import ultraplot as pplt

from bo_target.plot.format_ax import format_ax


def plot_tddft_targets(save_dir=None):
    """Plot the 5 TDDFT target spectra on the energy (eV) axis."""
    from bo_target.data.tddft import (
        config_dir, WAVELENGTH_START, WAVELENGTH_END,
        N_GAUSSIANS, FIXED_SIGMA_EV, EPSILON_SCALE,
        GAUSSIAN_EV_SCALER_FILE,
        wl_to_ev, GroupMinMaxScaler,
        gaussian_params_to_spectrum_ev,
    )

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[3] / "bo_fig"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load config and scaler
    config_path = config_dir / "tddft_config.json"
    with open(config_path) as f:
        config = json.load(f)
    targets_scaled = np.array(config["target"])         # (5, 5D) scaled
    epsilon_base = config["epsilon"]
    epsilon = epsilon_base * EPSILON_SCALE

    scaler = GroupMinMaxScaler.load(GAUSSIAN_EV_SCALER_FILE,
                                     n_gaussians=N_GAUSSIANS)

    # Reconstruct full 6D params (insert A1 = 1 in physical units, then scale)
    a1_scaled = (1.0 - scaler.mins["A"]) / (scaler.maxs["A"] - scaler.mins["A"])
    targets_full = np.zeros((targets_scaled.shape[0], N_GAUSSIANS * 2))
    targets_full[:, 0] = a1_scaled                     # A1 = 1 (physical)
    targets_full[:, 1:] = targets_scaled               # mu1, A2, mu2, A3, mu3
    targets_phys = scaler.inverse_transform(targets_full)

    wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, 500)
    energies = wl_to_ev(wavelengths)
    e_sort = np.argsort(energies)
    energies_asc = energies[e_sort]

    # Reconstruct each target spectrum + per-component Gaussians
    target_spectra = np.zeros((targets_phys.shape[0], len(energies_asc)))
    target_components = np.zeros((targets_phys.shape[0], N_GAUSSIANS, len(energies_asc)))
    for t in range(targets_phys.shape[0]):
        params_t = targets_phys[t].reshape(N_GAUSSIANS, 2)
        recon = np.zeros(len(energies_asc))
        for k in range(N_GAUSSIANS):
            A_k, mu_k = params_t[k]
            if A_k < 0.005:
                continue
            gauss_k = A_k * np.exp(-0.5 * ((energies_asc - mu_k) / FIXED_SIGMA_EV) ** 2)
            target_components[t, k] = gauss_k
            recon += gauss_k
        target_spectra[t] = recon

    # plot
    target_colors = ["#E4256D", "#2495C1", "#F5A623", "#7B4B94", "#3CA474"]
    gauss_colors = ["#E4256D", "#2495C1", "#F5A623"]  # G1, G2, G3

    fig, axes = pplt.subplots(
        nrows=5, ncols=1,
        journal="nat1",
        refheight=1.3,
        sharex=True,
        sharey=False,
    )

    # energy-axis bounds (inverted: high eV on the left) for label placement
    e_lo, e_hi = energies_asc[0], energies_asc[-1]
    e_center = 0.5 * (e_lo + e_hi)
    e_pad = 0.05 * (e_hi - e_lo)

    for t, ax in enumerate(axes):
        params_t = targets_phys[t].reshape(N_GAUSSIANS, 2)
        spectrum = target_spectra[t]

        # Shaded Gaussian components (no outline)
        for k in range(N_GAUSSIANS):
            A_k, mu_k = params_t[k]
            if A_k < 0.005:
                continue
            gauss_k = target_components[t, k]
            ax.fill_between(energies_asc, 0, gauss_k,
                            color=gauss_colors[k], alpha=0.15)

        # Gray spectrum on top
        ax.plot(energies_asc, spectrum, color="0.3", lw=1.4)

        # Vertical dotted lines at mu_k + parameter annotations
        for k in range(N_GAUSSIANS):
            A_k, mu_k = params_t[k]
            if A_k < 0.005:
                continue
            ax.plot([mu_k, mu_k], [0, A_k], color=gauss_colors[k], lw=1.2,
                    ls=":")
            # annotate with values; offset toward the plot interior and clamp
            # inside the axis so labels never spill past the (inverted) spines
            label = f"$\\mu_{k+1}$={mu_k:.2f}"
            if k > 0:
                label += f"\n$A_{k+1}$={A_k:.3f}"
            if mu_k >= e_center:          # high-eV peak (left side): grow right
                ha, tx = "left", mu_k - 0.03
            else:                          # low-eV peak (right side): grow left
                ha, tx = "right", mu_k + 0.03
            tx = min(max(tx, e_lo + e_pad), e_hi - e_pad)
            ax.annotate(
                label,
                xy=(mu_k, A_k),
                xytext=(tx, min(A_k + 0.06, 1.12)),
                fontsize=6, color=gauss_colors[k], ha=ha, va="bottom",
            )

        ax.set_ylabel(f"T$_{t + 1}$", fontsize=9)
        ax.set_ylim(-0.02, 1.22)
        format_ax(ax)

    axes[-1].set_xlabel("Energy (eV)")
    axes[0].set_xlim(energies_asc[0], energies_asc[-1])
    axes[0].invert_xaxis()

    # top wavelength (nm) axis on the top panel, mirroring the decomposition fig
    wl_top = axes[0].twiny()
    wl_ticks_nm = np.array([550, 500, 450, 400, 350, 300, 250])
    wl_ticks_ev = wl_to_ev(wl_ticks_nm)
    in_range = (wl_ticks_ev >= energies_asc[0]) & (wl_ticks_ev <= energies_asc[-1])
    wl_ticks_nm, wl_ticks_ev = wl_ticks_nm[in_range], wl_ticks_ev[in_range]
    wl_top.set_xlim(axes[0].get_xlim())
    wl_top.set_xticks(wl_ticks_ev)
    wl_top.set_xticklabels([str(int(w)) for w in wl_ticks_nm])
    wl_top.set_xlabel("Wavelength (nm)")
    # the top spine belongs to the wavelength twin; suppress the eV top ticks
    axes[0].tick_params(top=False, labeltop=False, which="both")

    # save
    for fmt in ("png", "svg"):
        out_path = save_dir / f"fig6_tddft_targets.{fmt}"
        fig.savefig(out_path, dpi=300 if fmt == "png" else None,
                    bbox_inches="tight")
        print(f"Saved -> {out_path}")

    return fig, axes


if __name__ == "__main__":
    plot_tddft_targets()
    pplt.show()
