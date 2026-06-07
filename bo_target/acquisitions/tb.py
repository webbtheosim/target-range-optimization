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


class TB(Acquisition):
    """L2 Tolerance Band acquisition: P(||y - target|| < epsilon)."""

    def __init__(
        self,
        model: Union[IModel, IDifferentiable],
        target: np.ndarray,
        epsilon: np.ndarray,
    ) -> None:
        self.model = model
        self.target = np.asarray(target).reshape(1, -1)
        self.epsilon_squared = np.asarray(epsilon) ** 2
        self.output_dim = self.target.shape[1]
        self._gradient_function = None

    def _l2_probability_inside_band(
        self, predictive_mean_and_variance: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Probability that a sample lies inside the L2-band."""
        predictive_means, predictive_variances = predictive_mean_and_variance

        avg_variance = np.maximum(
            predictive_variances.mean(axis=1, keepdims=True), 1e-10
        )

        squared_distance = np.sum(
            (self.target - predictive_means) ** 2, axis=1, keepdims=True
        )

        non_centrality = squared_distance / avg_variance
        scaled_threshold = self.epsilon_squared / avg_variance

        t_clipped = np.clip(scaled_threshold, 1e-10, 1e9)
        lambda_clipped = np.clip(non_centrality, 0.0, 1e9)

        probability = ncx2_cdf_approximation(
            t_clipped, self.output_dim, lambda_clipped
        )
        probability = np.clip(probability, 0.0, 1.0)

        return probability.flatten()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        predictive_means, predictive_variances = self.model.predict(x)
        acquisition_values = self._l2_probability_inside_band((
            predictive_means,
            predictive_variances,
        ))
        return np.atleast_2d(acquisition_values).T

    def evaluate_with_gradients(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._gradient_function is None:
            self._gradient_function = egrad(self._l2_probability_inside_band)

        predictive_means, predictive_variances = self.model.predict(x)

        predictive_variances = broadcast_variances(
            predictive_variances, self.output_dim
        )

        d_mean_dx, d_variance_dx = self.model.model.predictive_gradients(x)
        d_variance_dx = reshape_variance_gradients(
            d_variance_dx, self.output_dim
        )

        acquisition_value = self._l2_probability_inside_band((
            predictive_means,
            predictive_variances,
        ))

        d_acq_d_mean, d_acq_d_variance = self._gradient_function((
            predictive_means,
            predictive_variances,
        ))
        acquisition_gradient = assemble_gradient(
            d_mean_dx, d_variance_dx, d_acq_d_mean, d_acq_d_variance
        )

        return np.atleast_2d(acquisition_value).T, acquisition_gradient

    @property
    def has_gradients(self) -> bool:
        return True
