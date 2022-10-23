import jax
import jax.numpy as jnp
import numpy as np
import netket as nk

from typing import Tuple, Union
from functools import partial

from geneqs.utils.indexing import edge_to_index


# %%
def position_to_plaq(position: jax.Array, shape: jax.Array) -> jax.Array:
    right = (position + jnp.array([1, 0])) % shape[0]  # location to the right of position (PBC)
    top = (position + jnp.array([0, 1])) % shape[1]  # location to the top of position (PBC)
    indices = jnp.stack([
        edge_to_index(position, 1, shape),
        edge_to_index(top, 0, shape),
        edge_to_index(right, 1, shape),
        edge_to_index(position, 0, shape),
    ])
    return indices


def position_to_star(position: jax.Array, shape: jax.Array) -> jax.Array:
    # location to the left of position (PBC)
    left = position - jnp.array([1, 0]) if position[0] > 0 else jnp.array([shape[0] - 1, position[1]])
    #  location to the bottom of position (PBC)
    bot = position - jnp.array([0, 1]) if position[1] > 0 else jnp.array([position[0], shape[1] - 1])
    indices = jnp.stack([
        edge_to_index(position, 1, shape),
        edge_to_index(position, 0, shape),
        edge_to_index(bot, 1, shape),
        edge_to_index(left, 0, shape),
    ])
    return indices


@jax.jit
@partial(jax.vmap, in_axes=(0, None))
def plaqz2d_conns_and_mels(sigma: jax.Array, indices: jax.Array) -> Tuple[jax.Array, jax.Array]:
    # plaquette is diagonal in z basis, so eta is just the original
    eta = sigma.reshape(1, -1)
    mel = jnp.product(sigma.at[indices].get())
    return eta, mel


@jax.jit
@partial(jax.vmap, in_axes=(0, None))
def starz2d_conns_and_mels(sigma: jax.Array, indices: jax.Array) -> Tuple[jax.Array, jax.Array]:
    eta = sigma.at[indices].set(-sigma.at[indices].get()).reshape(1, -1)
    mel = jnp.ones(1)
    return eta, mel


class Plaq2d(nk.operator.AbstractOperator):
    def __init__(self, hilbert: nk.hilbert.AbstractHilbert, position: jax.Array, shape: jax.Array):
        super().__init__(hilbert)
        self.position = position
        self.shape = shape
        # get corresponding indices on which the operator acts on
        self.indices = position_to_plaq(self.position, self.shape)

    @property
    def dtype(self):
        return float

    @property
    def is_hermitian(self):
        return True


def e_loc(logpsi, pars, sigma, extra_args):
    eta, mels = extra_args
    assert sigma.ndim == 2, f"sigma dimensions should be (Nsamples, Nsite), buthas dimensions {sigma.shape}"
    assert eta.ndim == 3, f"eta dimensions should be (Nsamples, Nconnected, Nsite), but has dimensions {eta.shape}"

    @partial(jax.vmap, in_axes=(0, 0, 0))
    def _loc_vals(sigma, eta, mels):
        return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)

    return _loc_vals(sigma, eta, mels)


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: Plaq2d):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: Plaq2d):
    sigma = vstate.samples
    # get the connected elements. Reshape the samples because that code only works
    # if the input is a 2D matrix
    extra_args = plaqz2d_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size), op.indices)
    return sigma, extra_args


# %%
class ToricCode2d(nk.operator.AbstractOperator):
    def __init__(self, hilbert: nk.hilbert.AbstractHilbert, shape: jax.Array):
        super().__init__(hilbert)
        self.shape = shape
        # get corresponding indices on which the operators act on
        positions = jnp.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
        self.plaqs = jnp.stack([position_to_plaq(p, shape) for p in positions])
        self.stars = jnp.stack([position_to_star(p, shape) for p in positions])

    @property
    def dtype(self):
        return float

    @property
    def is_hermitian(self):
        return True


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: ToricCode2d):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: ToricCode2d):
    sigma = vstate.samples
    # get the connected elements. Reshape the samples because that code only works
    # if the input is a 2D matrix
    extra_args = toric2d_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size), op.plaqs, op.stars)
    return sigma, extra_args


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None))
def toric2d_conns_and_mels(sigma: jax.Array, plaqs: jax.Array, stars: jax.Array) -> Tuple[jax.Array, jax.Array]:
    # repeat sigma for #stars times
    N = stars.shape[0]
    eta = jnp.tile(sigma, (N + 1, 1))
    # indices where spins will be flipped by star operators
    ids = (jnp.arange(N).reshape(-1, 1), stars)
    eta = eta.at[ids].set(-eta.at[ids].get())

    # now calcualte matrix elements
    mels = -jnp.ones(N + 1)
    # axis 0 of sigma.at[plaqs] corresponds to #N_plaqs and axis 1 to the 4 edges of one plaquette
    mels = mels.at[N].set(-jnp.sum(jnp.product(sigma.at[plaqs].get(), axis=1)))
    return eta, mels


