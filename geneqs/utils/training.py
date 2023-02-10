import numpy as np
import jax.numpy as jnp
import jax

import netket as nk

from typing import Any, Optional, Union
from dataclasses import dataclass


def approximate_gs(hilbert: nk.hilbert.AbstractHilbert,
                   model: Any,
                   hamiltonian: nk.operator.AbstractOperator,
                   optimizer: Any,
                   preconditioner: Any,
                   n_iter: int,
                   n_chains: int,
                   n_samples: int,
                   n_discard_per_chain: int,
                   exact_sampling: bool = False):
    if not exact_sampling:
        sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains)
    else:
        sampler = nk.sampler.ExactSampler(hilbert)
    vqs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    gs_driver = nk.driver.VMC(hamiltonian, optimizer, variational_state=vqs, preconditioner=preconditioner)

    log = nk.logging.RuntimeLog()
    cb = CustomCallback(min_delta=0.1, patience=20, monitor="mean")
    if not exact_sampling:
        gs_driver.run(n_iter=n_iter, out=log, callback=cb)
    else:
        gs_driver.run(n_iter=n_iter, out=log)
    data = log.data

    return vqs, data


@dataclass
class CustomCallback:
    """A simple callback to stop NetKet if there are no more improvements in the training.
        based on `driver._loss_name`.
        Adapted from the default EarlyStopping callback implemented in NetKet, adds additional
        logging functionality."""

    min_delta: float = 0.0
    """Minimum change in the monitored quantity to qualify as an improvement."""
    min_step: int = 200
    """Minimum number of steps, does not stop before min_steps"""
    patience: Union[int, float] = 0
    """Number of epochs with no improvement after which training will be stopped."""
    baseline: Optional[float] = None
    """Baseline value for the monitored quantity. Training will stop if the driver hits the baseline."""
    monitor: str = "mean"
    """Loss statistic to monitor. Should be one of 'mean', 'variance', 'sigma'."""

    def __post_init__(self):
        self._best_val = np.infty
        self._best_iter = 0

    def __call__(self, step, log_data, driver):
        """
        A boolean function that determines whether to stop training.
        Args:
            step: An integer corresponding to the step (iteration or epoch) in training.
            log_data: A dictionary containing log data for training.
            driver: A NetKet variational driver.
        Returns:
            A boolean. If True, training continues, else, it does not.
        """
        if step == 1:
            log_data["n_params"] = driver._variational_state.n_parameters
        sampler_state = driver._variational_state.sampler_state
        log_data["acceptance_rate"] = sampler_state.acceptance.item()

        loss = np.real(getattr(log_data[driver._loss_name], self.monitor))
        if loss < self._best_val:
            self._best_val = loss
            self._best_iter = step
        if self.baseline is not None:
            if loss <= self.baseline:
                return False
        if (
                step >= self.min_step
                and step - self._best_iter >= self.patience
                and loss >= self._best_val - self.min_delta
        ):
            return False
        else:
            return True
