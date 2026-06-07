import logging
from typing import Tuple

import numpy as np

from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.optimization.acquisition_optimizer import AcquisitionOptimizerBase
from emukit.core.optimization.context_manager import ContextManager

_log = logging.getLogger(__name__)


class BanditAcquisitionOptimizer(AcquisitionOptimizerBase):
    """Discrete optimizer over a fixed set of valid configurations.

    Masks previously selected indices with -inf to prevent reselection
    across iterations and within a batch.
    """

    def __init__(self, space: ParameterSpace, valid_config: np.ndarray, seed: int = None):
        super().__init__(space)
        self.valid_config = valid_config
        self.already_selected_index = []
        self._batch_selected_indices = []
        self._rng = np.random.default_rng(seed)

        _log.info(
            f"BanditAcquisitionOptimizer initialized with {len(valid_config)} valid configurations"
        )

    def _optimize(
        self, acquisition: Acquisition, context_manager: ContextManager
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mask previously selected indices with -inf, then argmax over remaining candidates."""
        _log.info(
            f"Starting bandit optimization with acquisition: {type(acquisition).__name__}"
        )

        acquisition_values = acquisition.evaluate(self.valid_config).ravel()

        _log.debug(
            f"Acquisition shape: {acquisition_values.shape}, valid configs: {len(self.valid_config)}"
        )
        _log.debug(
            f"Already selected indices (all iterations): {self.already_selected_index}"
        )
        _log.debug(
            f"Batch-selected indices (current batch): {self._batch_selected_indices}"
        )

        masks_applied = []
        if self.already_selected_index:
            acquisition_values[self.already_selected_index] = -np.inf
            masks_applied.append(
                f"previous-iterations: {len(self.already_selected_index)} indices"
            )

        if self._batch_selected_indices:
            acquisition_values[self._batch_selected_indices] = -np.inf
            masks_applied.append(
                f"current-batch: {len(self._batch_selected_indices)} indices"
            )

        if masks_applied:
            _log.debug(f"Masks applied: {', '.join(masks_applied)}")
            _log.debug(
                f"After masking, max acquisition value: {np.max(acquisition_values)}"
            )

        finite = np.isfinite(acquisition_values)
        if not np.any(finite):
            raise RuntimeError("All candidates are invalid or masked.")

        max_value = np.max(acquisition_values[finite])
        best = np.flatnonzero(
            np.isfinite(acquisition_values) & (acquisition_values >= max_value - 1e-12)
        )
        max_index = int(self._rng.choice(best))

        if max_index in self.already_selected_index:
            raise RuntimeError(
                f"CRITICAL: Selected index {max_index} is already in already_selected_index. "
                f"This should have been masked to -inf. Masking logic may be broken. "
                f"Already selected: {self.already_selected_index}"
            )

        if max_index in self._batch_selected_indices:
            raise RuntimeError(
                f"CRITICAL: Selected index {max_index} is already in batch. "
                f"This should have been masked to -inf. Masking logic may be broken. "
                f"Batch selected: {self._batch_selected_indices}"
            )

        selected_config = self.valid_config[max_index]
        for prev_idx in self.already_selected_index:
            prev_config = self.valid_config[prev_idx]
            if np.allclose(selected_config, prev_config, rtol=1e-10, atol=1e-10):
                _log.error("Value-based duplicate detected!")
                _log.error(
                    f"New selection (index {max_index}): {selected_config}"
                )
                _log.error(
                    f"Matches previous (index {prev_idx}): {prev_config}"
                )
                raise RuntimeError(
                    f"Value-based duplicate detected: index {max_index} has identical feature vector "
                    f"to previously selected index {prev_idx}. "
                    f"This indicates the valid_configs may have duplicate rows. "
                    f"Valid configs shape: {self.valid_config.shape}"
                )

        _log.info(f"Selected index: {max_index}, acquisition value: {max_value}")

        self.already_selected_index.append(max_index)
        self._batch_selected_indices.append(max_index)

        return self.valid_config[[max_index]], acquisition_values[[max_index]]

    def _begin_optimization(self):
        _log.debug(
            f"Beginning new BO iteration. Resetting batch tracking. "
            f"Total accumulated: {len(self.already_selected_index)}"
        )
        self._batch_selected_indices = []

    def get_selected_indices(self):
        """Indices of selected configurations from valid_config."""
        return self.already_selected_index.copy()

    def get_valid_config(self):
        """The valid configurations array."""
        return self.valid_config