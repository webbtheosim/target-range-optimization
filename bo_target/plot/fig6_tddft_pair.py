"""Figure 6: structurally diverse molecules with near-identical spectra.

The diverse-molecule selection is precomputed by
``bo_target/utils/run_fig6_tddft.py`` into ``data/analysis/fig6_tddft_pair.pkl``
(chosen dataset indices + SMILES per target); this script only renders the
corresponding spectra, which are read from the shipped raw-spectra array.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import ultraplot as pplt

from bo_target.plot.format_ax import format_ax

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_PATH = REPO / "bo_target" / "data" / "analysis" / "fig6_tddft_pair.pkl"

N_TARGETS = 5


def _load():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def plot_tddft_pair(target_idx=1, save_dir=None, data=None):
    """Plot the n similar-output, different-input molecules selected for a target."""
    from bo_target.data.tddft import (
        WAVELENGTH_START, WAVELENGTH_END, label_file, wl_to_ev,
    )

    if save_dir is None:
        save_dir = REPO / "bo_fig"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if data is None:
        data = _load()
    sel = data["targets"][target_idx]
    chosen = sel["chosen"]
    smiles = sel["smiles"]
    mu1_vals = sel["mu1_vals"]
    mu1_spread = sel["mu1_spread"]
    in_spread = sel["in_spread"]
    epsilon = sel["epsilon"]

    raw_spectra = np.load(label_file)
    wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, raw_spectra.shape[1])
    energies = wl_to_ev(wavelengths)
    e_sort = np.argsort(energies)
    energies_asc = energies[e_sort]

    palette = ["#E4256D", "#2495C1", "#F5A623", "#7B4B94", "#3CA474"]
    names = [chr(ord("A") + i) for i in range(len(chosen))]
    colors = [palette[i % len(palette)] for i in range(len(chosen))]

    fig, ax = pplt.subplots(journal="nat1", refaspect=1.4)

    # report the chosen set + SMILES (for ChemDraw etc.)
    mu1_str = "/".join(f"{m:.2f}" for m in mu1_vals)
    print(f"\nTarget T{target_idx}: "
          + ", ".join(f"{names[c]} #{idx}" for c, idx in enumerate(chosen))
          + f"  [mu1={mu1_str} eV, mu1 spread={mu1_spread:.3f} eV, "
          f"input spread={in_spread:.2f}]")
    for c, idx in enumerate(chosen):
        print(f"  {names[c]} (#{idx}): {smiles[c]}")

    for c, idx in enumerate(chosen):
        spec = raw_spectra[idx][e_sort]
        ax.plot(energies_asc, spec, color=colors[c], lw=1.3,
                label=f"{names[c]}", zorder=5 - c)
        ax.fill_between(energies_asc, 0, spec, color=colors[c], alpha=0.08)

    ax.set_xlim(energies_asc[0], energies_asc[-1])
    ax.invert_xaxis()
    ax.set_ylim(-0.02, 1.15)
    ax.set_ylabel("Normalized Intensity")

    wl_top = ax.twiny()
    wl_ticks_nm = np.array([550, 500, 450, 400, 350, 300, 250])
    wl_ticks_ev = wl_to_ev(wl_ticks_nm)
    in_range = (wl_ticks_ev >= energies_asc[0]) & (wl_ticks_ev <= energies_asc[-1])
    wl_ticks_nm, wl_ticks_ev = wl_ticks_nm[in_range], wl_ticks_ev[in_range]
    wl_top.set_xlim(ax.get_xlim())
    wl_top.set_xticks(wl_ticks_ev)
    wl_top.set_xticklabels([str(int(w)) for w in wl_ticks_nm])
    wl_top.set_xlabel("Wavelength (nm)")

    ax.legend(loc="ul", fontsize=8, ncols=1, framealpha=0.9, edgecolor="0.7")
    format_ax(ax)
    ax.tick_params(top=False, labeltop=False, which="both")
    ax.set_xlabel("Energy (eV)")
    ax.format(title=f"Target T$_{target_idx}$")

    for fmt in ("png", "svg"):
        out_path = save_dir / f"fig6_tddft_pair_T{target_idx}.{fmt}"
        fig.savefig(out_path, dpi=300 if fmt == "png" else None,
                    bbox_inches="tight")
        print(f"Saved -> {out_path}  (mu1 spread={mu1_spread:.3f} eV, "
              f"eps={epsilon:.3f}, input spread={in_spread:.2f})")

    return fig, ax


def plot_all_pairs(save_dir=None, n_targets=N_TARGETS):
    """Produce one figure for every target T1..Tn."""
    data = _load()
    return [
        plot_tddft_pair(target_idx=ti, save_dir=save_dir, data=data)
        for ti in range(1, n_targets + 1)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--target-idx", type=int, default=None,
                        help="target number 1..5; omit to plot all five")
    args = parser.parse_args(argv)

    if args.target_idx is None:
        plot_all_pairs()
    else:
        plot_tddft_pair(target_idx=args.target_idx)


if __name__ == "__main__":
    main()
