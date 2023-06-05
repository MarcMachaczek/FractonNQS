import jax
import jax.numpy as jnp
import netket as nk
import geneqs

import math
from typing import Tuple
from functools import partial


# %%
class Checkerboard(nk.operator.DiscreteOperator):
    def __init__(self, hilbert: nk.hilbert.DiscreteHilbert, shape: jax.Array, h: Tuple[float, float, float] = None):
        super().__init__(hilbert)
        self.shape = shape
        if h is None:
            self.h = (0., 0., 0.)
        else:
            self.h = h
        if h[1] != 0:
            print("Warning: hy field is currently disabled tor reduce memory cost")

        # get corresponding indices on which the operators act on
        positions = jnp.asarray([(x, y, z)
                                 for x in range(shape[0])
                                 for y in range(shape[1])
                                 for z in range(shape[2])
                                 if (x + y + z) % 2 == 0])

        self.cubes = jnp.stack([geneqs.utils.indexing.position_to_cube(p, shape) for p in positions])

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
            sigma: Input states or "bra", acting from the left.

        Returns:
            connected states or "kets" eta, corresponding matrix elements mels

        """
        return checkerboard_conns_and_mels(sigma, self.cubes, self.h)

    def get_conn_flattened(self, x, sections):
        eta, mels = checkerboard_conns_and_mels(x, self.cubes, self.h)

        n_primes = eta.shape[1]  # number of connected states, same for alle possible configurations x
        n_visible = x.shape[1]
        eta = eta.reshape(-1, n_visible)  # flatten last dimension
        mels = mels.flatten()

        # must manipulate sections in place
        for i in range(len(sections)):
            sections[i] = (i + 1) * n_primes

        return eta, mels


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: Checkerboard):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: Checkerboard):
    sigma = vstate.samples
    extra_args = checkerboard_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size), op.cubes, op.h)
    return sigma, extra_args


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None))
def checkerboard_conns_and_mels(sigma: jax.Array,
                                cubes: jax.Array,
                                h: Tuple[float, float, float]) -> Tuple[jax.Array, jax.Array]:
    """
    For a given input spin configuration sigma, calculates all connected states eta and the corresponding non-zero
    matrix elements mels. See netket for further details. This version is only faster than netket if external fields are
    non-zero, as it already includes corresponding conns and mels, even when fields are zero.
    H = H_checkerboard - hx * sum_i sigma_x - hy * sum_j sigma_y - hz * sum_k sigma_z
    Args:
        sigma: Input states or "bra", acting from the left.
        cubes: Array of shape (num_cubes, 8) containing the indices of all cube operators.
        h: Tuple of field strengths of the external magentic field in x, y and z direction.

    Returns:
        connected states or "kets" eta, corresponding matrix elements mels

    """
    n_sites = sigma.shape[-1]
    n_cubes = cubes.shape[0]
    hx, hy, hz = h

    @jax.vmap
    def flip(x, idx):
        return x.at[idx].set(-x.at[idx].get())

    # initialize connected states: 1 diagonal, n_cubes by checkerboard, n_sites by external field
    eta = jnp.tile(sigma, (1 + n_cubes + n_sites, 1))
    # connected states by cube operators, leave the first eta as is (diagonal connected state)
    eta = eta.at[1:n_cubes+1].set(flip(eta.at[1:n_cubes+1].get(), cubes))
    # connected states through external field (sigma_x and sigma_y)
    eta = eta.at[1+n_cubes:1+n_cubes+n_sites].set(
        flip(eta.at[1+n_cubes:1+n_cubes+n_sites].get(), jnp.arange(n_sites)))

    # now calcualte matrix elements
    # axis 0 of sigma.at[plaqs] corresponds to #n_cubes and axis 1 to the 8 sites in one cube
    diag_mel = -jnp.sum(jnp.product(sigma.at[cubes].get(), axis=1)) + hz * jnp.sum(sigma)
    # n_sites mels corresponding to flipped cubes
    cube_mels = -jnp.ones(n_cubes)
    # mel according to hx and hy
    field_mels = -hx * jnp.ones(n_sites) # TODO: include hy again if needed! - hy * sigma * 1j
    mels = jnp.hstack((diag_mel, cube_mels, field_mels))
    return eta, mels


def e_loc(logpsi, pars, sigma, extra_args):
    """
    Calculates the local estimate of the operator.
    Args:
        logpsi: Wavefunction, taking as input pars and some state, returning the log amplitude.
        pars: Parameters for logpsi model.
        sigma: Samples from which to calculate the estimate.
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


def get_netket_checkerboard(hilbert, shape: jax.Array, h: Tuple[float, float, float]):
    ha_netketlocal = nk.operator.LocalOperator(hilbert, dtype=complex)
    hx, hy, hz = h

    # get corresponding indices on which the operators act on
    positions = jnp.asarray([[x, y, z]
                             for x in range(shape[0])
                             for y in range(shape[1])
                             for z in range(shape[2])
                             if (x + y + z) % 2 == 0])

    for position in positions:
        ha_netketlocal -= get_netket_xcube(hilbert, position, shape) + get_netket_zcube(hilbert, position, shape)

    # adding external fields
    ha_netketlocal -= hz * sum([nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size)])
    ha_netketlocal -= hx * sum([nk.operator.spin.sigmax(hilbert, i) for i in range(hilbert.size)])
    ha_netketlocal -= hy * sum([nk.operator.spin.sigmay(hilbert, i) for i in range(hilbert.size)])

    return ha_netketlocal


def get_netket_xcube(hilbert, position: jax.Array, shape: jax.Array):
    """
    Create a cube operator (qubits on corners) consisting of the product of PauliX operators acting on hilbert space
    at position.
    Args:
        hilbert: NetKet spin hilbert space.
        position: Position of the operator.
        shape: Shape of the lattice.

    Returns:
        Cube operator.

    """
    cube_operator = nk.operator.LocalOperator(hilbert, dtype=float)
    cube = geneqs.utils.indexing.position_to_cube(position, shape)
    cube_operator += math.prod([nk.operator.spin.sigmax(hilbert, idx.item()) for idx in cube])
    return cube_operator


def get_netket_zcube(hilbert, position: jax.Array, shape: jax.Array):
    """
    Create a cube operator (qubits on corners) consisting of the product of PauliZ operators acting on hilbert space
    at position.
    Args:
        hilbert: NetKet spin hilbert space.
        position: Position of the operator.
        shape: Shape of the lattice.

    Returns:
        Cube operator.

    """
    cube_operator = nk.operator.LocalOperator(hilbert, dtype=float)
    cube = geneqs.utils.indexing.position_to_cube(position, shape)
    cube_operator += math.prod([nk.operator.spin.sigmaz(hilbert, idx.item()) for idx in cube])
    return cube_operator
