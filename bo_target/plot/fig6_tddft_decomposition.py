"""Figure 6: eV-space Gaussian decomposition of a TDDFT spectrum."""

from pathlib import Path

import numpy as np
import ultraplot as pplt

from bo_target.plot.format_ax import format_ax


def plot_tddft_decomposition(
    molecule_idx=None,
    save_dir=None,
):
    """Plot the eV-space Gaussian decomposition for one molecule."""
    from bo_target.data.tddft import (
        WAVELENGTH_START, WAVELENGTH_END,
        N_GAUSSIANS, FIXED_SIGMA_EV,
        GAUSSIAN_EV_LABEL_FILE, GAUSSIAN_EV_SCALER_FILE, label_file,
        wl_to_ev, GroupMinMaxScaler, gaussian_params_to_spectrum_ev,
    )

    if save_dir is None:
        save_dir = Path(__file__).resolve().parents[3] / "bo_fig"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load data
    raw_spectra = np.load(label_file) if label_file.exists() else None
    gaussian_scaled = (
        np.load(GAUSSIAN_EV_LABEL_FILE)
        if GAUSSIAN_EV_LABEL_FILE.exists() else None
    )
    scaler = (
        GroupMinMaxScaler.load(GAUSSIAN_EV_SCALER_FILE, n_gaussians=N_GAUSSIANS)
        if GAUSSIAN_EV_SCALER_FILE.exists() else None
    )
    if raw_spectra is None or gaussian_scaled is None or scaler is None:
        print("Cached data not found -- run tddft.py first.")
        return

    phys_full = scaler.inverse_transform(gaussian_scaled)

    wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, raw_spectra.shape[1])
    energies = wl_to_ev(wavelengths)
    e_sort = np.argsort(energies)
    energies_asc = energies[e_sort]

    # auto-select best-fit molecule
    if molecule_idx is None:
        valid = np.where(
            (phys_full[:, 0] > 0.05)
            & (phys_full[:, 2] > 0.05)
            & (phys_full[:, 4] > 0.05)
        )[0]
        if len(valid) == 0:
            valid = np.arange(phys_full.shape[0])
        errors = np.zeros(len(valid))
        for i, idx in enumerate(valid):
            recon = gaussian_params_to_spectrum_ev(phys_full[idx].reshape(1, -1), wavelengths)
            raw_ev_i = raw_spectra[idx][e_sort]
            recon_ev_i = recon[e_sort] if recon.ndim == 1 else recon[0, e_sort]
            errors[i] = np.mean((raw_ev_i - recon_ev_i) ** 2)
        best = np.argmin(errors)
        molecule_idx = int(valid[best])

    params = phys_full[molecule_idx].reshape(N_GAUSSIANS, 2)
    raw_ev = raw_spectra[molecule_idx][e_sort]
    colors = ["#E4256D", "#2495C1", "#F5A623"]

    # build figure
    fig, (ax, ax_res) = pplt.subplots(
        nrows=2, ncols=1, journal="nat1", refaspect=1.8,
        sharex=True, sharey=False, spany=False, hratios=[4.5, 1.],
    )

    ax.plot(energies_asc, raw_ev, color="0.4", lw=0.9, label="Spectrum", zorder=10)

    recon_total = np.zeros_like(energies_asc)
    for k in range(N_GAUSSIANS):
        A_k, mu_k = params[k]
        if A_k < 0.005:
            continue
        gauss_k = A_k * np.exp(-0.5 * ((energies_asc - mu_k) / FIXED_SIGMA_EV) ** 2)
        recon_total += gauss_k
        ax.plot(energies_asc, gauss_k, color=colors[k], lw=1.0, ls="--",
                label=f"G$_{k + 1}$")
        ax.fill_between(energies_asc, 0, gauss_k, color=colors[k], alpha=0.10)
        ax.plot([mu_k, mu_k], [0, A_k], color=colors[k], lw=1.2, ls=":")

        # annotate the fitted amplitude / centre in the empty space near the peak
        label = f"$A_{k + 1}$={A_k:.2f}\n$\\mu_{k + 1}$={mu_k:.2f} eV"
        if A_k > 0.6:
            # tall peak: drop the label into the empty area on its high-eV side
            ax.text(mu_k + 0.55, 0.88, label, color=colors[k], fontsize=7,
                    ha="center", va="top")
        else:
            # anchor left-aligned so labels near the high-eV spine grow inward
            ax.text(mu_k - 0.05, A_k + 0.03, label, color=colors[k], fontsize=7,
                    ha="left", va="bottom")

    # residual panel
    residual = raw_ev - recon_total
    ax_res.plot(energies_asc, residual, color="0.3", lw=0.6)
    ax_res.axhline(0, color="0.5", lw=0.4, ls="--")
    ax_res.fill_between(energies_asc, residual, 0, color="0.3", alpha=0.08)
    ax_res.set_ylim(-0.1, 0.1)

    # formatting
    ax.set_xlim(energies_asc[0], energies_asc[-1])
    ax.invert_xaxis()
    ax.set_ylabel("Normalized Intensity")

    # top axis: wavelength (nm)
    wl_top = ax.twiny()
    wl_ticks_nm = np.array([550, 500, 450, 400, 350, 300, 250])
    wl_ticks_ev = wl_to_ev(wl_ticks_nm)
    # keep only ticks whose eV value lies within the plotted energy range,
    # otherwise out-of-range ticks (e.g. 550 nm) pile up against the spine
    ev_lo, ev_hi = energies_asc[0], energies_asc[-1]
    in_range = (wl_ticks_ev >= ev_lo) & (wl_ticks_ev <= ev_hi)
    wl_ticks_nm, wl_ticks_ev = wl_ticks_nm[in_range], wl_ticks_ev[in_range]
    wl_top.set_xlim(ax.get_xlim())
    wl_top.set_xticks(wl_ticks_ev)
    wl_top.set_xticklabels([str(int(w)) for w in wl_ticks_nm])
    wl_top.set_xlabel("Wavelength (nm)")

    ax.legend(loc="ur", fontsize=7, ncols=1, framealpha=0.9, edgecolor="0.7")
    ax.set_xlabel("")
    format_ax(ax)
    # the top spine is reserved for the wavelength twin axis; suppress the
    # main eV axis ticks there so the two tick sets don't overlap
    ax.tick_params(top=False, labeltop=False, which="both")
    format_ax(ax_res)
    ax_res.format(ylabel="Residual", xlabel="Energy (eV)")

    for fmt in ("png", "svg"):
        out_path = save_dir / f"fig6_tddft_decomposition.{fmt}"
        fig.savefig(out_path, dpi=300 if fmt == "png" else None, bbox_inches="tight")
        print(f"Saved -> {out_path}")

    return fig, (ax, ax_res)


if __name__ == "__main__":
    plot_tddft_decomposition()
    pplt.show()
