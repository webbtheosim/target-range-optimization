import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace

from bo_target.data._paths import base_path


RANDOM_STATE = 26
np.random.seed(RANDOM_STATE)

config_dir = base_path / "config"
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / "hartmann_config.json"


def _hartmann_raw(x):
    """3D Hartmann function on [0,1]^3 (analytic benchmark, see SI S1)."""
    x = np.asarray(x, dtype=float)

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828],
    ])

    y = np.zeros(x.shape[0])
    for i in range(4):
        inner = np.sum(A[i, :] * (x - P[i, :]) ** 2, axis=1)
        y += alpha[i] * np.exp(-inner)

    return (-y).reshape(-1, 1)


def hartmann_function():
    """3D Hartmann function, raw output (no standardization)."""
    parameter_space = ParameterSpace([
        ContinuousParameter("x1", 0, 1),
        ContinuousParameter("x2", 0, 1),
        ContinuousParameter("x3", 0, 1),
    ])

    def f(x):
        return _hartmann_raw(x)

    return f, parameter_space, None


if __name__ == "__main__":
    from bo_target.data._pipeline import generate_config, run_epsilon_analysis

    np.random.seed(RANDOM_STATE)

    f, space, valid_configs = hartmann_function()

    config = generate_config(
        f, space, valid_configs, config_file, "hartmann",
        method="kmeans", format_type="surface",
        random_state=RANDOM_STATE, max_iter_override=200,
        output_standardized=False,
    )

    np.random.seed(RANDOM_STATE)
    X_samples = space.sample_uniform(20000)
    y = f(X_samples)[:, 0]
    target_values = np.asarray(config["target"]).flatten()

    run_epsilon_analysis(
        y, target_values, config["epsilon"], config_dir,
        "hartmann", random_state=RANDOM_STATE, is_2d=False,
        label="samples", analysis_type="BAND",
    )
