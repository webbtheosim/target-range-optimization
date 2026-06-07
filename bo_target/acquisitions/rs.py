import numpy as np
from typing import Optional

from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition


class RS(Acquisition):
    """Random search acquisition: returns uniform random values."""

    def __init__(
        self,
        space: ParameterSpace = None,
        seed: Optional[int] = None,
    ) -> None:
        self.space = space
        self.rng = np.random.default_rng(seed)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.rng.uniform(low=0.0, high=1.0, size=(x.shape[0], 1))

    @property
    def has_gradients(self) -> bool:
        return False