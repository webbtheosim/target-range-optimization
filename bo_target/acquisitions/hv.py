from typing import Tuple, Union

import autograd.numpy as np
from autograd import elementwise_grad as egrad

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition

from bo_target.acquisitions._math import (
    assemble_gradient,
    broadcast_variances,
    ncx2_cdf_approximation,
    reshape_variance_gradients,
)


class HV(Acquisition):
    """Boundary-aware L2 Heaviside acquisition with smooth tanh transition."""

    def __init__(
        self,
        model: Union[IModel, IDifferentiable],
        target: np.ndarray,
        epsilon: Union[float, np.ndarray],
        transition_width: Union[float, np.ndarray] = None,
    ) -> None:
        self.model = model
        self.target = np.asarray(target, dtype=float).reshape(1, -1)
        self.epsilon_squared = np.asarray(epsilon, dtype=float) ** 2
        self.output_dim = self.target.shape[1]

        if transition_width is None:
            self.transition_width = 1e-3 * self.epsilon_squared
        else:
            self.transition_width = np.maximum(
                np.asarray(transition_width, dtype=float), 1e-12
            )

        self._gradient_function = None

    def _boundary_weight(self, squared_dist):
        """Smooth tanh transition from 0 (inside) to 1 (outside) the epsilon-ball."""
        tau = np.maximum(self.transition_width, 1e-12)
        normalized_margin = (squared_dist - self.epsilon_squared) / tau
        normalized_margin = np.clip(normalized_margin, -50.0, 50.0)
        weight = 0.5 * (1.0 + np.tanh(normalized_margin))
        return np.clip(weight, 0.0, 1.0)

    def _l2_mass(self, predictive_mean_and_variance):
        """Probability mass inside the epsilon-ball with smooth boundary transition."""
        means, variances = predictive_mean_and_variance

        gamma2 = np.maximum(variances.mean(axis=1), 1e-10)
        squared_dist = np.sum((self.target - means) ** 2, axis=1)

        nc = squared_dist / gamma2
        t_eps = self.epsilon_squared / gamma2

        t_clip = np.clip(t_eps, 1e-12, 1e9)
        lam_clip = np.clip(nc, 0.0, 1e9)

        prob = ncx2_cdf_approximation(t_clip, self.output_dim, lam_clip)
        weight = self._boundary_weight(squared_dist)

        result = (1.0 - weight) * 1.0 + weight * prob
        result = np.clip(result, 0.0, 1.0)

        return result

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        means, variances = self.model.predict(x)
        variances = broadcast_variances(variances, self.output_dim)
        values = self._l2_mass((means, variances))
        return np.atleast_2d(values).T

    def evaluate_with_gradients(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._gradient_function is None:
            self._gradient_function = egrad(self._l2_mass)

        means, variances = self.model.predict(x)

        variances = broadcast_variances(variances, self.output_dim)

        dmean_dx, dvariance_dx = self.model.model.predictive_gradients(x)
        dvariance_dx = reshape_variance_gradients(dvariance_dx, self.output_dim)

        values = self._l2_mass((means, variances))
        dval_dmean, dval_dvariance = self._gradient_function((means, variances))

        dval_dx = assemble_gradient(
            dmean_dx, dvariance_dx, dval_dmean, dval_dvariance
        )

        return np.atleast_2d(values).T, dval_dx

    @property
    def has_gradients(self) -> bool:
        return True
