import jax
import jax.numpy as jnp
import netket as nk
import geneqs

from typing import Tuple
from functools import partial


# %%
class ToricCode2d(nk.operator.AbstractOperator):
    def __init__(self, hilbert: nk.hilbert.AbstractHilbert, shape: jax.Array, h: Tuple[float, float, float] = None):
        super().__init__(hilbert)
        self.shape = shape
        if h is None:
            self.h = (0., 0., 0.)
        else:
            self.h = h
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
        return toric2d_conns_and_mels(sigma, self.plaqs, self.stars, self.h)


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: ToricCode2d):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: ToricCode2d):
    sigma = vstate.samples
    extra_args = toric2d_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size), op.plaqs, op.stars, op.h)
    return sigma, extra_args


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None))
def toric2d_conns_and_mels(sigma: jax.Array,
                           plaqs: jax.Array,
                           stars: jax.Array,
                           h: Tuple[float, float, float]) -> Tuple[jax.Array, jax.Array]:
    """
    For a given input spin configuration sigma, calculates all connected states eta and the corresponding non-zero
    matrix elements mels. See netket for further details.
    H = H_toric - hx * sum_i sigma_x - hy * sum_j sigma_y - hz * sum_k sigma_z
    Args:
        sigma: Input state or "bra", acting from the left.
        plaqs: Array of shape (num_plaquettes, 4) containing the indices of all plaquette operators.
        stars: Array of shape (num_stars, 4) containing the indices of all star operators.
        h: Tuple of field strengths of the external magentic field in x, y and z direction.

    Returns:
        connected states or "kets" eta, corresponding matrix elements mels

    """
    n_sites = stars.shape[0]
    hx, hy, hz = h

    @jax.vmap
    def flip(x, idx):
        return x.at[idx].set(-x.at[idx].get())

    # connected states by star operators
    star_eta = jnp.tile(sigma, (n_sites, 1))
    star_eta = flip(star_eta, stars)
    # connected states through external field (sigma_x and sigma_y)
    field_eta = jnp.tile(sigma, (2*n_sites, 1))
    field_eta = flip(field_eta, jnp.arange(2*n_sites))
    # stack connected mels (sigma.reshape corresponds to diagonal part, i.e. plaquettes, of the hamiltonian)
    eta = jnp.vstack((star_eta, sigma.reshape(1, -1), field_eta))

    # now calcualte matrix elements, first n_sites correspond to flipped stars
    star_mels = -jnp.ones(n_sites)
    # axis 0 of sigma.at[plaqs] corresponds to #N_plaqs and axis 1 to the 4 edges of one plaquette
    diag_mel = -jnp.sum(jnp.product(sigma.at[plaqs].get(), axis=1)) - hz * jnp.sum(sigma)
    # mel according to hx and hy
    field_mels = -hx * jnp.ones(2*n_sites) - hy * sigma * 1j
    mels = jnp.hstack((star_mels, diag_mel, field_mels))
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
