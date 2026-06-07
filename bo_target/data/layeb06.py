import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace

from bo_target.data._paths import base_path


RANDOM_STATE = 26
np.random.seed(RANDOM_STATE)

config_dir = base_path / "config"
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / "layeb06_config.json"


def _layeb06_raw(x):
    """Layeb-6 function on [0,1]^10 (analytic benchmark, see SI S1)."""
    x = np.asarray(x, dtype=float)
    x_scaled = x * 20.0 - 10.0

    n_points, d = x_scaled.shape
    y_clean = np.zeros(n_points)

    for i in range(d - 1):
        xi = x_scaled[:, i]
        xi1 = x_scaled[:, i + 1]
        r = np.sqrt(xi**2 + xi1**2)
        term = np.cos(r) * np.sin(xi1) + np.cos(xi1) + 1
        term = np.abs(term)
        y_clean += term**0.1

    return y_clean.reshape(-1, 1)


def layeb06_function():
    """10D Layeb06 function, raw output (no standardization)."""
    parameter_space = ParameterSpace([
        ContinuousParameter("x1", 0, 1),
        ContinuousParameter("x2", 0, 1),
        ContinuousParameter("x3", 0, 1),
        ContinuousParameter("x4", 0, 1),
        ContinuousParameter("x5", 0, 1),
        ContinuousParameter("x6", 0, 1),
        ContinuousParameter("x7", 0, 1),
        ContinuousParameter("x8", 0, 1),
        ContinuousParameter("x9", 0, 1),
        ContinuousParameter("x10", 0, 1),
    ])

    def f(x):
        return _layeb06_raw(x)

    return f, parameter_space, None


if __name__ == "__main__":
    from bo_target.data._pipeline import generate_config, run_epsilon_analysis

    np.random.seed(RANDOM_STATE)

    f, space, valid_configs = layeb06_function()

    config = generate_config(
        f, space, valid_configs, config_file, "layeb06",
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
        "layeb06", random_state=RANDOM_STATE, is_2d=False,
        label="samples", analysis_type="BAND",
    )
