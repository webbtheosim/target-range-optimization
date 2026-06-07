import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace

from bo_target.data._paths import base_path


RANDOM_STATE = 26
np.random.seed(RANDOM_STATE)

config_dir = base_path / "config"
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / "ackley_config.json"


def _ackley_raw(x):
    """Ackley function on [0,1]^5 (analytic benchmark, see SI S1)."""
    x = np.asarray(x, dtype=float)
    x_scaled = x * 10.0 - 5.0

    d = x_scaled.shape[1]
    sum_sq = np.sum(x_scaled**2, axis=1)
    sum_cos = np.sum(np.cos(2 * np.pi * x_scaled), axis=1)

    a = 20.0
    b = 0.2

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    y_clean = term1 + term2 + a + np.exp(1)

    return y_clean.reshape(-1, 1)


def ackley_function():
    """Ackley function, raw output (no standardization)."""
    parameter_space = ParameterSpace([
        ContinuousParameter("x1", 0, 1),
        ContinuousParameter("x2", 0, 1),
        ContinuousParameter("x3", 0, 1),
        ContinuousParameter("x4", 0, 1),
        ContinuousParameter("x5", 0, 1),
    ])

    def f(x):
        return _ackley_raw(x)

    return f, parameter_space, None


if __name__ == "__main__":
    from bo_target.data._pipeline import generate_config, run_epsilon_analysis

    np.random.seed(RANDOM_STATE)

    f, space, valid_configs = ackley_function()

    config = generate_config(
        f,
        space,
        valid_configs,
        config_file,
        "ackley",
        method="kmeans",
        format_type="surface",
        random_state=RANDOM_STATE,
        max_iter_override=200,
        output_standardized=False,
    )

    np.random.seed(RANDOM_STATE)
    X_samples = space.sample_uniform(20000)
    y = f(X_samples)[:, 0]
    target_values = np.asarray(config["target"]).flatten()

    run_epsilon_analysis(
        y,
        target_values,
        config["epsilon"],
        config_dir,
        "ackley",
        random_state=RANDOM_STATE,
        is_2d=False,
        label="samples",
        analysis_type="BAND",
    )
