import jax
import jax.numpy as jnp
import netket as nk

from typing import Tuple
from functools import partial

import geneqs

# %%
class Magnetization(nk.operator.DiscreteOperator):
    def __init__(self, hilbert: nk.hilbert.DiscreteHilbert):
        super().__init__(hilbert)

    @property
    def dtype(self):
        return float

    @property
    def is_hermitian(self):
        return True

    def get_conn_padded(self, sigma: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        See Netket repo for details. This method is called by <get_local_kernel_arguments> for discrete operators.
        Instead of implementing the flattened version, we can directly overwrite the padded method, which is more
        efficient.
        Args:
            sigma:

        Returns:

        """
        return magnetization_conns_and_mels(sigma)

    def get_conn_flattened(self, x, sections):
        eta, mels = magnetization_conns_and_mels(x)

        n_primes = eta.shape[1]  # number of connected states, same for alle possible configurations x
        n_visible = x.shape[-1]
        batch_size = x.shape[0]
        eta = eta.reshape(batch_size, n_primes*n_visible)  # flatten last dimension
        mels = mels.flatten()

        # must manipulate sections in place TODO: this doesnt work, sections are not modified by this piece of code
        for i in range(len(sections)):
            sections[i] = (i + 1) * n_primes

        return eta, mels


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: Magnetization):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: Magnetization):
    sigma = vstate.samples
    extra_args = magnetization_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size))
    return sigma, extra_args


@jax.jit
@partial(jax.vmap, in_axes=(0,))
def magnetization_conns_and_mels(sigma: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    For a given input spin configuration sigma, calculates all connected states eta and the corresponding non-zero
    matrix elements mels. See netket for further details.
    Args:
        sigma: Input state or "bra", acting from the left.

    Returns:
        connected states or "kets" eta, corresponding matrix elements mels

    """

    # connected states is sigma itself
    eta = jnp.expand_dims(sigma, axis=0)  # insert dimension such that dim of eta is three

    # connected matrix elements
    n_sites = sigma.shape[-1]
    mels = jnp.absolute(jnp.sum(sigma, axis=-1, keepdims=True)) / n_sites

    return eta, mels


def e_loc(logpsi, pars, sigma, extra_args):
    """
    Calculates the local estimate of the operator.
    Args:
        logpsi: Wavefunction, taking as input pars and some state, returning the log amplitude.
        pars: Parameters for logpsi model.
        sigma: VQS samples from which to calculate the estimate.
        extra_args: Connected states, non-zero matrix elements.

    Returns:
        local estimates: sum_{eta} <sigma|Operator|eta> * psi(eta)/psi(sigma) over different sigmas

    """
    eta, mels = extra_args
    assert sigma.ndim == 2, f"sigma dimensions should be (Nsamples, Nsites), but has dimensions {sigma.shape}"
    assert eta.ndim == 3, f"eta dimensions should be (Nsamples, Nconnected, Nsites), but has dimensions {eta.shape}"

    @partial(jax.vmap, in_axes=(0, 0, 0))
    def _loc_vals(sigma, eta, mels):
        return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)

    return _loc_vals(sigma, eta, mels)

def get_netket_wilsonob(hi, shape):
    wilsonob = nk.operator.LocalOperator(hi, dtype=float)
    for i in range(shape[0]):
        for j in range(shape[0]):
            wilsonob += geneqs.operators.toric_2d.get_netket_star(hi, jnp.array([i, j]), shape) * \
                        geneqs.operators.toric_2d.get_netket_star(hi, jnp.array([i, (j+1)%shape[1]]), shape) * \
                        geneqs.operators.toric_2d.get_netket_star(hi, jnp.array([(i+1)%shape[0], (j+1)%shape[1]]), shape) * \
                        geneqs.operators.toric_2d.get_netket_star(hi, jnp.array([(i+1)%shape[0], j]), shape) * \
                        geneqs.operators.toric_2d.get_netket_plaq(hi, jnp.array([i, j]), shape) * \
                        geneqs.operators.toric_2d.get_netket_plaq(hi, jnp.array([i, (j+1)%shape[1]]), shape) * \
                        geneqs.operators.toric_2d.get_netket_plaq(hi, jnp.array([(i+1)%shape[0], (j+1)%shape[1]]), shape) * \
                        geneqs.operators.toric_2d.get_netket_plaq(hi, jnp.array([(i+1)%shape[0], j]), shape)
    return wilsonob / hi.size
            