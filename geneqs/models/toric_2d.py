import jax
import jax.numpy as jnp
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


@partial(jax.vmap, in_axes=(0, None))
def plaqz2d_conns_and_mels(sigma: jax.Array, indices: jax.Array) -> Tuple[jax.Array, jax.Array]:
    # plaquette is diagonal in z basis, so eta is just the original
    eta = sigma.reshape(1, -1)  # one dimension for number of connected states
    mel = sigma[indices[0]] * sigma[indices[1]] * sigma[indices[2]] * sigma[indices[3]]
    return eta, mel


@partial(jax.vmap, in_axes=(0, None))
def starz2d_conns_and_mels(sigma: jax.Array, indices: jax.Array) -> Tuple[jax.Array, jax.Array]:
    eta = sigma.at[indices].set(-sigma.at[indices].get()).reshape(1, -1)
    mel = jnp.ones(1)
    return eta, mel


class Plaq2d(nk.operator.AbstractOperator):
    def __init__(self, hilbert: nk.hilbert.AbstractHilbert, position: jax.Array, shape: Union[jax.Array, str] = "square"):
        super().__init__(hilbert)
        self.position = position
        # extract the shape
        if type(shape) == str:
            if shape == "square":
                self.shape = jnp.array([jnp.sqrt(hilbert.size), jnp.sqrt(hilbert.size)], dtype=jnp.integer)
            else:
                raise NotImplementedError("Unkown shape type, please provide explicit shape (jax.Array) instead.")
        else:
            assert jnp.product(shape) == hilbert.size, "shape and hilbert space size do not fit."
            self.shape = shape
        # get corresponding indices on which the operator acts on
        self.indices = position_to_plaq(self.position, self.shape)

    @property
    def dtype(self):
        return float


def e_loc(logpsi, pars, sigma, extra_args):
    eta, mels = extra_args
    assert sigma.ndim == 2, "sigma dimensions should be (Nsamples, Nsite)"
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
