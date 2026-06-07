import numpy as np
from typing import Optional

from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition


class BAX(Acquisition):
    """Mean Bayesian Algorithm Execution (BAX) acquisition.

    Reference: Neiswanger et al., NeurIPS 2021.
    """

    def __init__(
        self,
        model,
        space: ParameterSpace,
        exe_algorithm,
        n_algorithm_samples: int = 15,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.space = space
        self.exe_algorithm = exe_algorithm
        self.S = n_algorithm_samples
        self.rng = np.random.default_rng(seed)

    def _sample_posterior(self, x: np.ndarray):
        """Return predictive mean and standard deviation."""
        mean, var = self.model.predict(x)
        std = np.sqrt(var)
        mean = mean[None, :, :]
        std = std[None, :, :]
        return mean, std

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f_mean, f_std = self._sample_posterior(x)
        exec_cond = self.exe_algorithm(f_mean)

        f_product = exec_cond[..., None] * f_std
        f_product = np.mean(f_product, axis=(0, 2))

        if np.all(f_product == 0) or np.any(np.isnan(f_product)):
            f_product = np.mean(f_std, axis=(0, 2))

        return f_product

    @property
    def has_gradients(self) -> bool:
        return False
