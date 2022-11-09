import jax
import jax.numpy as jnp
import netket as nk
import geneqs

from typing import Tuple
from functools import partial


# %%
class ToricCode2d(nk.operator.AbstractOperator):
    def __init__(self, hilbert: nk.hilbert.AbstractHilbert, shape: jax.Array):
        super().__init__(hilbert)
        self.shape = shape
        # get corresponding indices on which the operators act on
        positions = jnp.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
        self.plaqs = jnp.stack([geneqs.utils.indexing.position_to_plaq(p, shape) for p in positions])
        self.stars = jnp.stack([geneqs.utils.indexing.position_to_star(p, shape) for p in positions])

    @property
    def dtype(self):
        return float

    @property
    def is_hermitian(self):
        return True

    def conns_and_mels(self, sigma: jax.Array):
        return toric2d_conns_and_mels(sigma, self.plaqs, self.stars)


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: ToricCode2d):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: ToricCode2d):
    sigma = vstate.samples
    extra_args = toric2d_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size), op.plaqs, op.stars)
    return sigma, extra_args


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None))
def toric2d_conns_and_mels(sigma: jax.Array, plaqs: jax.Array, stars: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    For a given input spin configuration sigma, calculates all connected states eta and the corresponding non-zero
    matrix elements mels. See netket for further details.
    Args:
        sigma:
        plaqs:
        stars:

    Returns:

    """
    # repeat sigma for #stars times
    N = stars.shape[0]
    star_eta = jnp.tile(sigma, (N, 1))

    @jax.vmap
    def flip(x, idx):
        return x.at[idx].set(-x.at[idx].get())

    star_eta = flip(star_eta, stars)
    eta = jnp.vstack((star_eta, sigma.reshape(1, -1)))
    # now calcualte matrix elements
    mels = -jnp.ones(N + 1)
    # axis 0 of sigma.at[plaqs] corresponds to #N_plaqs and axis 1 to the 4 edges of one plaquette
    mels = mels.at[N].set(-jnp.sum(jnp.product(sigma.at[plaqs].get(), axis=1)))
    return eta, mels


def e_loc(logpsi, pars, sigma, extra_args):
    eta, mels = extra_args
    assert sigma.ndim == 2, f"sigma dimensions should be (Nsamples, Nsite), buthas dimensions {sigma.shape}"
    assert eta.ndim == 3, f"eta dimensions should be (Nsamples, Nconnected, Nsite), but has dimensions {eta.shape}"

    @partial(jax.vmap, in_axes=(0, 0, 0))
    def _loc_vals(sigma, eta, mels):
        return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)

    return _loc_vals(sigma, eta, mels)
