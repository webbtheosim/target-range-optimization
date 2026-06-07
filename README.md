# Target-Range Bayesian Optimization for Diverse Materials and Product Design


## Quick start

```bash
git clone https://github.com/webbtheosim/target-range-optimization.git
cd target-range-optimization
conda env create -f environment.yml
conda activate bo_target
```

`conda env create` also installs this package (editable), so the `bo_target`
imports below work immediately.

## Reproduce every figure

All figures read precomputed summaries shipped inside the repo
(`bo_target/data/analysis/*.pkl`), so the full figure set reproduces from a clean
`git clone` with no external data download.

```bash
python -m bo_target.reproduce
```

This renders all figures into `bo_fig/` and prints a per-figure pass/skip
report. Render a subset with `--only`:

```bash
python -m bo_target.reproduce --only fig3_branin_qm9,fig5_kmc_umap
```

Individual figure scripts also run standalone, for example
`python bo_target/plot/fig2_acq_aggregate.py`.

## Run Bayesian optimization on any dataset

`main_bo.py` exposes a command-line interface. No file editing is required.

```bash
# Default single trial
python -m bo_target.main_bo --dataset hartmann --acquisition tb --tolerance 0.4

# A pool-based (bandit) dataset
python -m bo_target.main_bo --dataset qm9 --acquisition hv --tolerance 0.4 --max-iter 50
```

Specify your own targets and tolerance range without touching any config file.
A target is given as a comma-separated vector (repeat `--target` for multiple
targets); `--epsilon` sets the absolute tolerance band. Use `=` so negative
numbers are not read as flags.

```bash
# One scalar target at -2.0 with a +/- 0.3 band
python -m bo_target.main_bo --dataset hartmann --target=-2.0 --epsilon=0.3

# Two 2-D targets (e.g. HOMO, LUMO), shared band 0.5
python -m bo_target.main_bo --dataset qm9 --target=-1.0,-1.0 --target=0.1,0.3 --epsilon=0.5
```

A property *range* `[lo, hi]` maps to `--target=(lo+hi)/2 --epsilon=(hi-lo)/2`.
You can also point at a custom config file: `--config my_config.json` (same
schema as `bo_target/data/config/*.json`).

| Flag | Meaning | Default |
|---|---|---|
| `--dataset` | dataset name (see list below) | `hartmann` |
| `--acquisition` | `tb`, `hv`, `ei`, `lcb`, `bax`, `rs` | `tb` |
| `--tolerance` | tolerance ratio scaling epsilon | `0.4` |
| `--trial` | trial index / random seed | `0` |
| `--max-iter` | number of BO iterations | `50` |
| `--target` | custom target vector (repeatable) | from config |
| `--epsilon` | absolute tolerance band (skips ratio scaling) | from config |
| `--config` | path to a custom dataset config JSON | built-in |
| `--out-dir` | where result JSONs are written | `bo_output/` |

Results are written to `bo_output/` (kept out of version control). Batch
execution on a cluster: `sbatch submit/submit_bo.submit`.

**Datasets**: `ackley`, `branin`, `hartmann`, `layeb06` (synthetic),
`bace`, `esol`, `freesolv`, `lipo`, `qm9` (molecular), `nanoparticle`,
`propensity`, `toporg` (materials pools), `kmc` (polymer molecular weight distribution),
`tddft` / oligomer (UV-vis spectra).

## Repository structure

```
target-range-optimization/
  bo_target/
    main_bo.py              Entry point: CLI single-trial and SLURM batch
    reproduce.py            One-command figure reproduction
    acquisitions/           TB, HV, EI, LCB, BAX, RS acquisition functions
    data/
      analysis/             Shipped figure-data summaries (.pkl); figures read these
      config/               Dataset target and epsilon configurations
      clean/, mordred/, smiles/   Precomputed dataset arrays and inputs
      ...                   Dataset loaders: synthetic, ML-pool, KMC, TD-DFT
    optimization/           Bandit acquisition optimizer for pool-based tasks
    plot/                   Figure-generation scripts (read data/analysis/*.pkl)
    utils/                  Analysis, target selection, and figure-data builders
  submit/                   SLURM submission scripts
  environment.yml           Conda environment specification
```

## Figures

Every figure renders from the shipped `.pkl` summaries in
`bo_target/data/analysis/`. Output goes to `bo_fig/`.

```bash
python bo_target/plot/fig1_benchmarks.py           # Benchmark landscapes
python bo_target/plot/fig2_acq_aggregate.py        # Avg rank, best/worst ratios
python bo_target/plot/fig3_branin_qm9.py           # BO acquisition trajectories
python bo_target/plot/fig4_collab.py               # Cross-target hit ratio
python bo_target/plot/fig5_kmc_residual.py         # KMC target spectra and residuals
python bo_target/plot/fig5_kmc_umap.py             # KMC UMAP projection
python bo_target/plot/fig6_tddft_decomposition.py  # TD-DFT Gaussian decomposition
python bo_target/plot/fig6_tddft_discovered.py     # TD-DFT discovered spectra
python bo_target/plot/fig6_tddft_pair.py           # TD-DFT structure diversity
python bo_target/plot/fig6_tddft_targets.py        # TD-DFT / oligomer target spectra
python bo_target/plot/fig_radar_aucs_all.py        # Per-target AUC radar (14 datasets, incl. KMC + TD-DFT/oligomer)
python bo_target/plot/fig_valid_cand.py            # Valid-candidate curves (KMC + TD-DFT)
python bo_target/plot/fig_variance.py              # Normalized variance bars
```

## Full reproduction from scratch (optional)

The shipped `.pkl` summaries are enough for every figure. To regenerate them
from the raw BO sweep, place the raw results as a sibling directory and run the
builders. The raw results are archived separately (see Data availability).

```
your_directory/
  target-range-optimization/   <-- this repository
  bo_result/
    results/
      results/                 <- raw BO result JSONs (10 trials per combo)
```

```bash
# Figure-specific summaries
python -m bo_target.utils.run_fig3          # --> fig3_branin_qm9.pkl
python -m bo_target.utils.run_fig5_kmc      # --> fig5_kmc_residual.pkl, fig5_kmc_umap.pkl
python -m bo_target.utils.run_fig6_tddft    # --> fig6_tddft_pair.pkl
python -m bo_target.utils.run_radar_aucs    # --> radar_aucs.pkl (all 14 datasets incl. KMC + TD-DFT)

# Aggregate summaries
python -m bo_target.utils.run_ranking          # --> ranking_results.pkl
python -m bo_target.utils.run_auc_variance     # --> stats_results.pkl
python -m bo_target.utils.run_cross_target     # --> cross_target_hit_ratio.pkl
python -m bo_target.utils.run_case_vs_budget   # --> case_vs_budget.pkl
python -m bo_target.utils.run_result_vs_budget # --> result_vs_budget.pkl
```

Each builder writes into `bo_target/data/analysis/` and accepts
`--results-dir` to point at a different raw-results location. To regenerate the
raw sweep itself: `python -m bo_target.main_bo ...` per trial, or
`sbatch submit/submit_bo.submit` for the full job array.

## Raw results download

To regenerate summaries from scratch, download the archived BO sweep
(~200 MB zip, 14 datasets x 6 acquisitions x 3 tolerance ratios x 10 seeds).

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=12IJlxkeArHOnhzFCW1PFnLH2Yd3jr5N3"
unzip bo_result.zip -d /path/to/your_directory/
```

After unpacking, your layout should match:

```
your_directory/
  target-range-optimization/   <-- this repository
  bo_result/
    search/                    <-- raw BO result JSONs (2520 files)
```

## Path conventions

All paths are resolved relative to the repository root via `pathlib.Path`.
Figure summaries are shipped inside `bo_target/data/analysis/`. User BO runs
write to `bo_output/` (gitignored). Raw results, when present, are read from
a sibling `bo_result/` directory. Cross-platform: macOS, Linux, Windows.

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11 | |
| numpy | 1.26 | Numerical arrays |
| scipy | 1.11 | Optimization, distance matrices |
| GPy | 1.13 | Gaussian process models |
| emukit | pinned fork | Bayesian optimization framework |
| autograd | 1.8 | Automatic differentiation |
| matplotlib | 3.10 | Plotting backend |
| ultraplot | 1.66 | Publication-quality figure engine |
| pandas | 2.3 | Tabular data I/O |
| scikit-learn | 1.8 | PCA, scaling, k-means |
| joblib | 1.5 | Model and transformer caching |
| tqdm | 4.67 | Progress bars |
| openpyxl | 3.1 | Excel file I/O |

Regeneration-only extras (not needed to reproduce figures or run `main_bo`):
`scikit-learn-extra` (k-medoids target selection), `rdkit` and `mordred`
(molecular descriptors), `umap-learn` (KMC projection).

## Data availability

Raw BO results: [Google Drive](https://drive.google.com/file/d/12IJlxkeArHOnhzFCW1PFnLH2Yd3jr5N3/view?usp=sharing). Precomputed figure summaries are shipped with the repository
under `bo_target/data/analysis/`.