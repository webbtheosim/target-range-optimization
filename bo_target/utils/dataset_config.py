"""Dataset registry and configuration loader: function dispatch, targets, epsilon."""

import json
from pathlib import Path

import numpy as np

from bo_target.data._paths import base_path as data_base
from bo_target.data.ackley import ackley_function
from bo_target.data.bace import bace_function
from bo_target.data.branin import branin_function
from bo_target.data.esol import esol_function
from bo_target.data.freesolv import freesolv_function
from bo_target.data.hartmann import hartmann_function
from bo_target.data.kmc import kmc_function
from bo_target.data.layeb06 import layeb06_function
from bo_target.data.lipo import lipo_function
from bo_target.data.nanoparticle import nanoparticle_function
from bo_target.data.propensity import propensity_function
from bo_target.data.qm9 import qm9_function
from bo_target.data.tddft import tddft_function
from bo_target.data.toporg import toporg_function

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS_ROOT = REPO.parent / "bo_result"

# Canonical raw BO-result sweep, read by the analysis builders during full
# reproduction (reviewers never need this — figures load shipped pkls instead).
RAW_RESULTS_DIR = RESULTS_ROOT / "search"

# Default writable output dir for reviewer runs of main_bo (in-repo, gitignored).
DEFAULT_OUTPUT_DIR = REPO / "bo_output"


def dataset_config(
    dataset_name,
    max_iter=None,
    verbose=False,
    testing=False,
    config_path=None,
    target_override=None,
    epsilon_override=None,
    results_dir=None,
):
    """Load dataset function, parameter space, targets, and epsilon from config.

    Overrides let callers run a dataset with custom targets/tolerance without
    editing any file: ``config_path`` points at a custom JSON, ``target_override``
    / ``epsilon_override`` replace the loaded values, and ``results_dir`` redirects
    where results are written/read.
    """
    if results_dir is not None:
        save_dir = Path(results_dir)
    elif testing:
        save_dir = RESULTS_ROOT / "search_test"
    else:
        save_dir = RESULTS_ROOT / "search"

    config_dir = data_base / "config"
    if config_path is None:
        config_path = config_dir / f"{dataset_name}_config.json"
    else:
        config_path = Path(config_path)

    # Note: callers that *write* results (main_bo) create save_dir themselves;
    # we avoid creating it here so that read-only callers (figures, builders)
    # never spawn a stray bo_result/ tree on a fresh clone.
    config_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Directories ready -> config: {config_dir} | results: {save_dir}")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found for dataset '{dataset_name}': {config_path}"
        )

    with open(config_path) as f:
        config = json.load(f)

    dataset_registry = {
        "nanoparticle": (nanoparticle_function, 50),
        "toporg": (toporg_function, 50),
        "propensity": (propensity_function, 50),
        "hartmann": (hartmann_function, 30),
        "branin": (branin_function, 20),
        "ackley": (ackley_function, 50),
        "layeb06": (layeb06_function, 100),
        "lipo": (lipo_function, 50),
        "qm9": (qm9_function, 50),
        "freesolv": (freesolv_function, 50),
        "esol": (esol_function, 50),
        "bace": (bace_function, 50),
        "kmc": (kmc_function, 50),
        "tddft": (tddft_function, 50),
    }

    if dataset_name not in dataset_registry:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    func, seed_size = dataset_registry[dataset_name]
    f, space, valid_configs = func()

    targets = (
        np.array(target_override)
        if target_override is not None
        else np.array(config["target"])
    )
    epsilon = epsilon_override if epsilon_override is not None else config["epsilon"]

    if max_iter is None:
        max_iter = config.get("global", {}).get("max_iter", 200)

    return (
        f,
        space,
        valid_configs,
        targets,
        epsilon,
        seed_size,
        config,
        save_dir,
        max_iter,
    )
