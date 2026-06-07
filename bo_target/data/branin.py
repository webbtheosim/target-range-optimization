import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace

from bo_target.data._paths import base_path


RANDOM_STATE = 26
np.random.seed(RANDOM_STATE)

config_dir = base_path / "config"
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / "branin_config.json"


def _branin_raw(x):
    """Branin-Hoo function on [0,1]^2 (analytic benchmark, see SI S1)."""
    x = np.asarray(x, dtype=float)

    x1 = x[:, 0] * 15.0 - 5.0
    x2 = x[:, 1] * 15.0

    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    term1 = x2 - b * x1**2 + c * x1 - r
    y_clean = a * term1**2 + s * (1.0 - t) * np.cos(x1) + s

    return y_clean.reshape(-1, 1)


def branin_function():
    """Branin-Hoo function, raw output (no standardization)."""
    parameter_space = ParameterSpace([
        ContinuousParameter("x1", 0, 1),
        ContinuousParameter("x2", 0, 1),
    ])

    def f(x):
        return _branin_raw(x)

    return f, parameter_space, None


if __name__ == "__main__":
    from bo_target.data._pipeline import generate_config, run_epsilon_analysis

    np.random.seed(RANDOM_STATE)

    f, space, valid_configs = branin_function()

    config = generate_config(
        f,
        space,
        valid_configs,
        config_file,
        "branin",
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
        "branin",
        random_state=RANDOM_STATE,
        is_2d=False,
        label="samples",
        analysis_type="BAND",
    )
