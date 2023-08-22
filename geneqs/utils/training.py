import numpy as np
import jax.numpy as jnp
import jax
import optax

import netket as nk

from functools import partial
from tqdm import tqdm
import time

from typing import Any, Optional, Union
from dataclasses import dataclass


# %% custom training loop to monitor performance of individual steps (timing)
def loop_gs(v_state: nk.vqs.MCState,
            _hamiltonian: nk.operator.AbstractOperator,
            optimizer: Any,
            preconditioner: Any,
            n_iter: int,
            min_steps: int = None,
            obs: dict = None,
            out: str = None):
    """
    Training function with some additional functionality, like timing of each step, early stopping, logging etc.
    Args:
        v_state: NetKet variational state. Contains the sampler, model, parameters.
        _hamiltonian: The hamiltonian to optimize.
        optimizer: Base optimizer, usually plain SGD.
        preconditioner: Gradient transformation like, for instance, stochastic reconfiguration.
        n_iter: Number of training iterations.
        min_steps: Minimum number of iterations before callback convergence criteria might cause early stopping.
        obs: Observables to be tracked during training. Values are saved in log file. {obs_name: obs_operator, ...}
        out: Where to save training_stats. None if no stats should be saved, else save_path.

    Returns:
        trained variational state, logger containing training information

    """

    hamiltonian = _hamiltonian.collect()
    if min_steps is None:
        min_steps = n_iter

    log = nk.logging.RuntimeLog()
    cb = LoopCallback(min_delta=0.1, min_steps=min_steps, patience=20, baseline=None, monitor="mean")

    opt_state = optimizer.init(v_state.parameters)

    with tqdm(total=n_iter) as pbar:
        for epoch in range(n_iter):
            times = {}
            v_state.reset()

            times["pre_sample"] = time.time()
            v_state.sample()
            times["pre_expectgrad"] = time.time()
            
            # the forces part of this crashes mpi when np > 2
            energy, gradients = v_state.expect_and_grad(hamiltonian)
            
            e_nan, e_inf = contains_naninf(energy.mean)
            g_nan, g_inf = contains_naninf(gradients)

            # adapt gradients according to stochastic reconfiguration
            times["pre_sr"] = time.time()
            sr_gradients = preconditioner(v_state, gradients, epoch)
            sr_nan, sr_inf = contains_naninf(sr_gradients)
            times["post_sr"] = time.time()

            # If parameters are real, then take only real part of the gradient (if it's complex)
            sr_gradients = jax.tree_map(
                lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
                sr_gradients,
                v_state.parameters)

            opt_state, new_params = apply_gradient(optimizer.update,
                                                   opt_state,
                                                   sr_gradients,
                                                   v_state.parameters)
            v_state.parameters = new_params
            times["final"] = time.time()

            # logging data
            sampling_time = times["pre_expectgrad"] - times["pre_sample"]
            expect_grad_time = times["pre_sr"] - times["pre_expectgrad"]
            sr_time = times["post_sr"] - times["pre_sr"]
            p_update_time = times["final"] - times["post_sr"]
            total_time = times["final"] - times["pre_sample"]

            log_data = {"Energy": energy}
            log_data["times"] = {"sampling": sampling_time,
                                 "expect_grad": expect_grad_time,
                                 "sr": sr_time,
                                 "p_update": p_update_time,
                                 "total": total_time}
            if obs is not None:
                for obs_name, obs_operator in obs.items():
                    log_data[obs_name] = v_state.expect(obs_operator)
            callback_stop = not cb(epoch, log_data, v_state)
            log(epoch, log_data, v_state)

            loss = log_data["Energy"]
            pbar.set_postfix_str(f"Energy = {loss}, time ratios:"
                                 f" sampling = {round(sampling_time/total_time, 2)},"
                                 f" ex_grad = {round(expect_grad_time/total_time, 2)},"
                                 f" sr = {round(sr_time/total_time, 2)}")

            if callback_stop:
                break

            if jnp.asarray([e_nan, e_inf, g_nan, g_inf, sr_nan, sr_inf]).any():
                print("nan or inf detected")
                print([e_nan, e_inf, g_nan, g_inf, sr_nan, sr_inf])
                break

            # reset time after first step to ignore compilation time
            if epoch == 0:
                pbar.unpause()

            pbar.update(1)

    if out is not None:
        log.serialize(out)
    return v_state, log.data


@partial(jax.jit, static_argnums=0)
def apply_gradient(optimizer_fun, optimizer_state, gradients, params):
    updates, new_optimizer_state = optimizer_fun(gradients, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_optimizer_state, new_params


def driver_gs(v_state: nk.vqs.VariationalState,
              hamiltonian: nk.operator.AbstractOperator,
              optimizer: Any,
              preconditioner: Any,
              n_iter: int,
              min_steps: int = None,
              obs: dict = None,
              out: str = None):
    if min_steps is None:
        min_steps = n_iter
    gs_driver = nk.driver.VMC(hamiltonian, optimizer, variational_state=v_state, preconditioner=preconditioner)

    cb = DriverCallback(min_delta=0.1, min_steps=min_steps, patience=20, monitor="mean")
    log = nk.logging.RuntimeLog()
    gs_driver.run(n_iter=n_iter, out=log, obs=obs, callback=cb)

    if out is not None:
        log.serialize(out)
    return v_state, log.data


@dataclass
class LoopCallback:
    """A simple callback to stop NetKet if there are no more improvements in the training.
    Adapted from the default EarlyStopping callback implemented in NetKet, adds additional
    logging functionality."""

    min_delta: float = 0.0
    """Minimum change in the monitored quantity to qualify as an improvement."""
    min_steps: int = 200
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

    def __call__(self, step, log_data, v_state):
        """
        A boolean function that determines whether to stop training.
        Args:
            step: An integer corresponding to the step (iteration or epoch) in training.
            log_data: A dictionary containing log data for training.
            v_state: A NetKet variational state.
        Returns:
            A boolean. If True, training continues, else, it does not.
        """
        if step == 0:
            log_data["n_params"] = v_state.n_parameters

        is_mc = isinstance(v_state, nk.vqs.MCState) and (not v_state.sampler.is_exact)
        if is_mc:
            sampler_state = v_state.sampler_state
            log_data["acceptance_rate"] = sampler_state.acceptance.item()

        loss = np.real(getattr(log_data["Energy"], self.monitor))
        if loss < self._best_val:
            self._best_val = loss
            self._best_iter = step
        if self.baseline is not None:
            if loss <= self.baseline:
                return False
        if (
                step >= self.min_steps
                and step - self._best_iter >= self.patience
                and loss >= self._best_val - self.min_delta
        ):
            return False
        else:
            return True


@dataclass
class DriverCallback:
    """A simple callback to stop NetKet if there are no more improvements in the training.
        based on `driver._loss_name`.
        Adapted from the default EarlyStopping callback implemented in NetKet, adds additional
        logging functionality."""

    min_delta: float = 0.0
    """Minimum change in the monitored quantity to qualify as an improvement."""
    min_steps: int = 200
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

        is_mc = isinstance(driver, nk.vqs.MCState) and (not driver._variational_state.sampler.is_exact)
        if is_mc:
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
                step >= self.min_steps
                and step - self._best_iter >= self.patience
                and loss >= self._best_val - self.min_delta
        ):
            return False
        else:
            return True


def contains_naninf(pytree):
    """
    Primitive check if some elements in a pytree are NaNs or Infs.
    Args:
        pytree:

    Returns:

    """
    nan, inf = False, False
    for vals in jax.tree_util.tree_flatten(pytree)[0]:
        nan = jnp.isnan(vals).any()
        inf = jnp.isinf(vals).any()
    return nan, inf
