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


class EI(Acquisition):
    """L2 Expected Improvement acquisition for target vector estimation.

    Reference: Uhrenholt & Jensen, AISTATS 2019.
    """

    def __init__(
        self,
        model: Union[IModel, IDifferentiable],
        target: np.ndarray,
        epsilon: np.ndarray,
    ) -> None:
        self.model = model
        self.target = np.asarray(target).reshape(1, -1)
        self.epsilon = np.asarray(epsilon)
        self.output_dim = self.target.shape[1]
        self._gradient_function = None

    def _ei(
        self,
        predictive_mean_and_variance: Tuple[np.ndarray, np.ndarray],
        y_min: float,
    ) -> np.ndarray:
        """Expected improvement in squared L2 distance to the target."""
        predictive_means, predictive_variances = predictive_mean_and_variance

        avg_variance = np.maximum(
            predictive_variances.mean(axis=1, keepdims=True), 1e-10
        )

        squared_distance = np.sum(
            (self.target - predictive_means) ** 2, axis=1, keepdims=True
        )

        non_centrality = squared_distance / avg_variance
        scaled_threshold = y_min / avg_variance

        t_clipped = np.clip(scaled_threshold, 1e-10, 1e9)
        lambda_clipped = np.clip(non_centrality, 0.0, 1e9)

        k = self.output_dim

        cdf_k = ncx2_cdf_approximation(t_clipped, k, lambda_clipped)
        cdf_kp2 = ncx2_cdf_approximation(t_clipped, k + 2, lambda_clipped)
        cdf_kp4 = ncx2_cdf_approximation(t_clipped, k + 4, lambda_clipped)

        t1 = y_min * cdf_k
        t2 = avg_variance * (k * cdf_kp2 + non_centrality * cdf_kp4)

        improvement = t1 - t2
        return improvement.flatten()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        predictive_means, predictive_variances = self.model.predict(x)
        y_min = ((self.model.Y - self.target) ** 2).sum(axis=1).min()
        acquisition_values = self._ei(
            (predictive_means, predictive_variances), y_min
        )
        return np.atleast_2d(acquisition_values).T

    def evaluate_with_gradients(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._gradient_function is None:
            self._gradient_function = egrad(self._ei)

        predictive_means, predictive_variances = self.model.predict(x)
        y_min = ((self.model.Y - self.target) ** 2).sum(axis=1).min()

        predictive_variances = broadcast_variances(
            predictive_variances, self.output_dim
        )

        d_mean_dx, d_variance_dx = self.model.model.predictive_gradients(x)
        d_variance_dx = reshape_variance_gradients(
            d_variance_dx, self.output_dim
        )

        acquisition_value = self._ei(
            (predictive_means, predictive_variances), y_min
        )

        d_acq_d_mean, d_acq_d_variance = self._gradient_function(
            (predictive_means, predictive_variances), y_min
        )

        acquisition_gradient = assemble_gradient(
            d_mean_dx, d_variance_dx, d_acq_d_mean, d_acq_d_variance
        )

        return np.atleast_2d(acquisition_value).T, acquisition_gradient

    @property
    def has_gradients(self) -> bool:
        return True
