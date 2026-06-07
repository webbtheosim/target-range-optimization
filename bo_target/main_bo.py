"""Bayesian optimisation entry point: single-trial and SLURM batch execution."""

import itertools
import json
import logging
import os
import pickle
import random
import time

import GPy
import numpy as np
from GPy.models import GPRegression

from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.optimization.gradient_acquisition_optimizer import (
    GradientAcquisitionOptimizer,
)
from emukit.core.optimization.random_search_acquisition_optimizer import (
    RandomSearchAcquisitionOptimizer,
)
from emukit.model_wrappers import GPyModelWrapper

from bo_target.optimization import BanditAcquisitionOptimizer
from bo_target.utils.acq_config import acq_config
from bo_target.utils.dataset_config import (
    dataset_config,
    DEFAULT_OUTPUT_DIR,
    RAW_RESULTS_DIR,
)


def create_distance_mask(y_samples, target, epsilon):
    """Boolean mask of samples within epsilon L2 distance of target."""
    target = np.asarray(target).reshape(1, 1, -1)
    squared_diffs = (y_samples - target) ** 2
    l2_distances = np.sqrt(np.sum(squared_diffs, axis=-1))
    return l2_distances <= epsilon


def make_exe_algorithm(target, epsilon):
    """Closure: True for samples within epsilon of target."""

    def exe_algorithm(f_samples):
        return create_distance_mask(f_samples, target, epsilon)

    return exe_algorithm


def main(
    dataset_name: str = "hartmann",
    acquisition_name: str = "tb",
    trial_index: int = 0,
    max_iter: int = None,
    tolerance_ratio: float = 0.1,
    target_override=None,
    epsilon_override=None,
    config_path=None,
    out_dir=None,
):
    """Run Bayesian Optimization for a given dataset and acquisition.

    ``target_override`` / ``epsilon_override`` / ``config_path`` let a user run
    with custom targets and tolerance without editing any file. When
    ``epsilon_override`` is given it is used as the absolute tolerance band
    (the ``tolerance_ratio`` scaling is skipped).
    """

    if out_dir is None:
        out_dir = DEFAULT_OUTPUT_DIR

    print(
        f"{dataset_name} | {acquisition_name} | trial={trial_index} | tol_ratio={tolerance_ratio}  -> started"
    )

    np.random.seed(trial_index)
    random.seed(trial_index)

    (
        f,
        space,
        valid_configs,
        targets,
        epsilon,
        seed_size,
        config,
        save_dir,
        max_iter_defined,
    ) = dataset_config(
        dataset_name,
        max_iter,
        verbose=False,
        config_path=config_path,
        target_override=target_override,
        epsilon_override=epsilon_override,
        results_dir=out_dir,
    )

    if max_iter is None:
        max_iter = max_iter_defined

    if epsilon_override is None:
        epsilon = tolerance_ratio * epsilon

    AcqClass = acq_config(acquisition_name)

    design = LatinDesign(space)
    X_init = design.get_samples(seed_size)
    Y_init = f(X_init)

    # Re-seed after Latin design so model RNG is deterministic per trial.
    np.random.seed(trial_index)
    random.seed(trial_index)

    kernel = GPy.kern.Matern52(input_dim=X_init.shape[1], ARD=True)
    gpy_model = GPRegression(X_init, Y_init, kernel=kernel)
    model = GPyModelWrapper(gpy_model, n_restarts=10)

    if "bax" in acquisition_name:
        acq_kwargs = {"model": model, "space": space, "n_algorithm_samples": 15}
        acquisition_optimizer = RandomSearchAcquisitionOptimizer(
            space=space, num_eval_points=1000
        )
    else:
        acquisition_optimizer = GradientAcquisitionOptimizer(
            space=space, num_samples=1000, num_anchor=15
        )

    # Bandit optimizer for datasets with a finite valid-config pool;
    # continuous synthetic targets stay on the gradient/random-search optimizer.
    if dataset_name not in ["ackley", "branin", "hartmann", "layeb06"]:
        acquisition_optimizer = BanditAcquisitionOptimizer(
            space=space, valid_config=valid_configs
        )

    acquisitions = []
    if "bax" in acquisition_name:
        for target in targets:
            exe_algorithm = make_exe_algorithm(target, epsilon)
            acq = AcqClass(exe_algorithm=exe_algorithm, **acq_kwargs)
            acquisitions.append(acq)
    elif "rs" in acquisition_name:
        for _ in targets:
            acq = AcqClass(space=space)
            acquisitions.append(acq)
    else:
        for t in targets:
            target_2d = np.atleast_2d(t)
            acq = AcqClass(model=model, target=target_2d, epsilon=epsilon)
            acquisitions.append(acq)

    opt_loop = BayesianOptimizationLoop(
        model=model,
        space=space,
        acquisition=acquisitions,
        batch_size=len(acquisitions),
        acquisition_optimizer=acquisition_optimizer,
    )

    start_time = time.time()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(
        "Search summary:",
        f"dataset={dataset_name}",
        f"acq={acquisition_name}",
        f"batch_size={len(acquisitions)}",
    )
    print(
        "Valid configs:",
        len(valid_configs) if valid_configs is not None else None,
    )
    print("Acquisitions:", [type(a).__name__ for a in acquisitions])

    # Wrap f() so per-batch progress is visible without touching Emukit internals.
    def f_progress(x):
        y = f(x)
        try:
            total_observed = len(model.X) + len(x)
        except Exception:
            total_observed = "unknown"
        print(
            f"Evaluated {len(x)} point(s); total observed after eval: {total_observed}"
        )
        return y

    opt_loop.run_loop(f_progress, max_iter)
    total_time = time.time() - start_time

    # Save indices for bandit mode (compact); full X vectors otherwise.
    results = {
        "Y": model.Y.tolist(),
        "total_wall_time_seconds": total_time,
    }

    if isinstance(acquisition_optimizer, BanditAcquisitionOptimizer):
        indices = acquisition_optimizer.get_selected_indices()
        results["indices"] = [int(i) for i in indices]
        results["data_mode"] = "indices"
        results["num_valid_configs"] = len(
            acquisition_optimizer.get_valid_config()
        )
    else:
        results["X"] = model.X.tolist()
        results["data_mode"] = "vectors"

    save_dir.mkdir(parents=True, exist_ok=True)
    path = (
        save_dir
        / f"{dataset_name}_{acquisition_name}_{tolerance_ratio}_{trial_index}.json"
    )

    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)

    model_path = (
        save_dir / f"{dataset_name}_{acquisition_name}_{trial_index}_model.pkl"
    )
    with open(model_path, "wb") as fh:
        pickle.dump(gpy_model, fh)

    print(
        f"{dataset_name} | {acquisition_name} | trial={trial_index} | tol_ratio={tolerance_ratio}  -> completed"
    )


def gen_combinations():
    """Cartesian product of trials x datasets x acquisitions x tolerances, shuffled."""
    trials = list(range(10))

    datasets = [
        "ackley",
        "bace",
        "branin",
        "esol",
        "freesolv",
        "hartmann",
        "layeb06",
        "lipo",
        "nanoparticle",
        "propensity",
        "qm9",
        "toporg",
        "kmc",
        "tddft",
    ]

    acq_functions = ["tb", "hv", "ei", "lcb", "bax", "rs"]

    tolerance_ratios = [0.2, 0.4, 0.6]

    combinations = list(
        itertools.product(trials, datasets, acq_functions, tolerance_ratios)
    )

    np.random.RandomState(26).shuffle(combinations)

    return combinations


def get_job_array_slice(job_index: int, total_jobs: int, combinations):
    """Even split of combinations across SLURM array tasks."""
    n = len(combinations)
    base = n // total_jobs
    rem = n % total_jobs

    if job_index < rem:
        start = job_index * (base + 1)
        end = start + base + 1
    else:
        start = (rem * (base + 1)) + (job_index - rem) * base
        end = start + base

    return combinations[start:end]


def _parse_targets(target_args):
    """Parse repeated --target values (each a comma-separated vector) into a 2D list."""
    if not target_args:
        return None
    targets = []
    for spec in target_args:
        vals = [float(v) for v in spec.replace(" ", "").split(",") if v != ""]
        targets.append(vals)
    return targets


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run a single band-aware BO trial on any dataset, optionally with "
            "custom targets and tolerance."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="hartmann",
        help="dataset name (hartmann, branin, qm9, kmc, tddft, ...)",
    )
    parser.add_argument(
        "--acquisition",
        default="tb",
        help="acquisition function: tb, hv, ei, lcb, bax, rs",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.4,
        help="tolerance ratio scaling epsilon (default: 0.4)",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="trial index / random seed (default: 0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="number of BO iterations (default: 50)",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="custom target vector as a comma list, repeatable for "
        "multiple targets. Use '=' for negatives, e.g. "
        "--target=-1.0,0.5 --target=0.2,-0.3",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="absolute tolerance band (overrides config epsilon and "
        "skips tolerance-ratio scaling). A property range [lo,hi] "
        "maps to --target=(lo+hi)/2 --epsilon=(hi-lo)/2",
    )
    parser.add_argument(
        "--config", default=None, help="path to a custom {dataset}_config.json"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for result JSONs (default: bo_output/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if (
        "SLURM_ARRAY_TASK_ID" in os.environ
        and "SLURM_ARRAY_TASK_MAX" in os.environ
    ):
        job_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        total_jobs = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1

        combinations = gen_combinations()
        job_slice = get_job_array_slice(job_index, total_jobs, combinations)

        for job in job_slice:
            trial, dataset, acq_function, tolerance_ratio = job
            main(
                dataset_name=dataset,
                acquisition_name=acq_function,
                trial_index=trial,
                max_iter=50,
                tolerance_ratio=tolerance_ratio,
                out_dir=RAW_RESULTS_DIR,
            )
    else:
        args = _cli()
        main(
            dataset_name=args.dataset,
            acquisition_name=args.acquisition,
            trial_index=args.trial,
            max_iter=args.max_iter,
            tolerance_ratio=args.tolerance,
            target_override=_parse_targets(args.target),
            epsilon_override=args.epsilon,
            config_path=args.config,
            out_dir=args.out_dir,
        )
