"""One-click figure reproduction: render every manuscript figure from shipped data.

Every figure reads a precomputed pickle in ``bo_target/data/analysis/`` (or a
shipped config), so this works on a fresh ``git clone`` with no external
``bo_result/`` directory. Each figure runs in its own headless (Agg) subprocess
so one failure never aborts the rest.

    python -m bo_target.reproduce
    python -m bo_target.reproduce --only fig3_branin_qm9,fig5_kmc_umap
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = REPO / "bo_fig"

FIGURES = [
    "fig1_benchmarks",
    "fig2_acq_aggregate",
    "fig3_branin_qm9",
    "fig4_collab",
    "fig5_kmc_residual",
    "fig5_kmc_umap",
    "fig6_tddft_decomposition",
    "fig6_tddft_discovered",
    "fig6_tddft_pair",
    "fig6_tddft_targets",
    "fig_radar_aucs_all",
    "fig_valid_cand",
    "fig_variance",
]


def run_one(mod):
    """Render a single figure module in a headless subprocess."""
    env = dict(os.environ, MPLBACKEND="Agg")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", f"bo_target.plot.{mod}"],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0, time.time() - t0, proc.stdout


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render all manuscript figures."
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated subset of figure module names",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print each figure's captured output",
    )
    args = parser.parse_args(argv)

    figures = FIGURES
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        unknown = wanted - set(FIGURES)
        if unknown:
            print(f"Unknown figure(s) ignored: {sorted(unknown)}")
        figures = [f for f in FIGURES if f in wanted]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(figures)} figure(s) -> {OUT_DIR}\n")

    results = []
    for mod in figures:
        ok, dt, out = run_one(mod)
        print(f"[{'ok  ' if ok else 'FAIL'}] {mod:28s} ({dt:5.1f}s)")
        if not ok or args.verbose:
            tail = out.strip().splitlines()[-20:]
            print("\n".join("        " + ln for ln in tail))
        results.append((mod, ok))

    n_ok = sum(ok for _, ok in results)
    print(f"\n{n_ok}/{len(results)} figures rendered into {OUT_DIR}")
    failed = [m for m, ok in results if not ok]
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
