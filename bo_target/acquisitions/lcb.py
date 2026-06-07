from typing import Tuple, Union

import autograd.numpy as np
from autograd import elementwise_grad as egrad

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition

from bo_target.acquisitions._math import (
    assemble_gradient,
    broadcast_variances,
    reshape_variance_gradients,
)


class LCB(Acquisition):
    """L2 Negative Lower Confidence Bound acquisition for target vector estimation.

    Reference: Uhrenholt & Jensen, AISTATS 2019.
    """

    def __init__(
        self,
        model: Union[IModel, IDifferentiable],
        target: np.ndarray,
        epsilon: np.ndarray,
        beta: float = 1.0
    ) -> None:
        self.model = model
        self.target = np.asarray(target).reshape(1, -1)
        self.epsilon = np.asarray(epsilon)
        self.beta = float(beta)
        self.output_dim = self.target.shape[1]
        self._gradient_function = None

    def _l2_negative_lower_bound_approximation(
        self, predictive_mean_and_variance: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Negative approximate lower confidence bound via Patnaik moment matching."""
        predictive_means, predictive_variances = predictive_mean_and_variance

        avg_variance = np.maximum(
            predictive_variances.mean(axis=1, keepdims=True), 1e-10
        )

        squared_distance = np.sum((self.target - predictive_means)**2, axis=1, keepdims=True)

        non_centrality = squared_distance / avg_variance
        non_centrality = np.clip(non_centrality, 0.0, 1e9)

        dof = self.output_dim
        r1 = dof + non_centrality
        r2 = 2.0 * (dof + 2.0 * non_centrality)
        r3 = 8.0 * (dof + 3.0 * non_centrality)

        r1_safe = np.maximum(r1, 1e-12)
        r2_safe = np.maximum(r2, 1e-12)
        r1_squared = r1_safe**2
        r1_fourth = r1_squared**2

        m_numerator = r1 * r3
        m_denominator = 3.0 * r2_safe**2
        m = 1.0 - m_numerator / m_denominator
        m = np.clip(m, 0.05, 2.5)

        term1 = r2_safe / (2.0 * r1_squared)
        term2_numerator = (2.0 - m) * (1.0 - 3.0 * m) * r2_safe**2
        term2_denominator = 8.0 * r1_fourth + 1e-300
        term2 = term2_numerator / term2_denominator

        alpha = 1.0 + m * (m - 1.0) * (term1 - term2)
        alpha = np.clip(alpha, -1e8, 1e8)

        rho_inner_numerator = (1.0 - m) * (1.0 - 3.0 * m) * r2_safe
        rho_inner_denominator = 4.0 * r1_squared + 1e-300
        rho_inner = rho_inner_numerator / rho_inner_denominator

        rho = (m * np.sqrt(r2_safe) / r1_safe) * (1.0 - rho_inner)
        rho = np.clip(rho, -1e8, 1e8)

        base = alpha - self.beta * rho
        base_safe = np.maximum(base, 1e-12)
        log_root = (1.0 / m) * np.log(base_safe)
        root = np.exp(log_root)
        root = np.clip(root, 1e-8, 1e8)

        bound = -root * r1_safe * avg_variance
        return bound.flatten()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        predictive_means, predictive_variances = self.model.predict(x)
        acquisition_values = self._l2_negative_lower_bound_approximation(
            (predictive_means, predictive_variances)
        )
        return np.atleast_2d(acquisition_values).T

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._gradient_function is None:
            self._gradient_function = egrad(self._l2_negative_lower_bound_approximation)

        predictive_means, predictive_variances = self.model.predict(x)

        predictive_variances = broadcast_variances(
            predictive_variances, self.output_dim
        )

        d_mean_dx, d_variance_dx = self.model.model.predictive_gradients(x)
        d_variance_dx = reshape_variance_gradients(
            d_variance_dx, self.output_dim
        )

        acquisition_value = self._l2_negative_lower_bound_approximation(
            (predictive_means, predictive_variances)
        )

        d_acq_d_mean, d_acq_d_variance = self._gradient_function((predictive_means, predictive_variances))

        acquisition_gradient = assemble_gradient(
            d_mean_dx, d_variance_dx, d_acq_d_mean, d_acq_d_variance
        )

        return np.atleast_2d(acquisition_value).T, acquisition_gradient

    @property
    def has_gradients(self) -> bool:
        return True