import numpy as np
import jax.numpy as jnp
import jax

import netket as nk

from typing import Any


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
    if not exact_sampling:
        gs_driver.run(n_iter=n_iter, out=log, callback=custom_callback)
    else:
        gs_driver.run(n_iter=n_iter, out=log)
    data = log.data

    return vqs, data


def custom_callback(step: int, logdata: dict, driver):
    if step == 1:
        logdata["n_params"] = driver._variational_state.n_parameters
    sampler_state = driver._variational_state.sampler_state
    logdata["acceptance_rate"] = sampler_state.acceptance.item()
    return True


