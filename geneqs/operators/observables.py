import jax
import jax.numpy as jnp
import netket as nk

from typing import Tuple
from functools import partial

from geneqs.utils.indexing import edge_to_index


# %%
class AbsZMagnetization(nk.operator.DiscreteOperator):
    """
    Operator corresponing to the absolute magnetization of a state. For regular magnetization simply use NetKet local
    operators.
    """

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
            sigma: Input states or "bra", acting from the left.

        Returns:
            connected states or "kets" eta, corresponding matrix elements mels

        """
        return abs_zmagnetization_conns_and_mels(sigma.reshape(-1, sigma.shape[-1]))

    def get_conn_flattened(self, x, sections):
        eta, mels = abs_zmagnetization_conns_and_mels(x)

        n_primes = eta.shape[1]  # number of connected states, same for all possible configurations x
        n_visible = x.shape[1]
        eta = eta.reshape(-1, n_visible)  # flatten last dimension
        mels = mels.flatten()

        # must manipulate sections in place
        for i in range(len(sections)):
            sections[i] = (i + 1) * n_primes

        return eta, mels


class AbsXMagnetization(nk.operator.DiscreteOperator):
    """
    Operator corresponing to the absolute magnetization of a state. For regular magnetization simply use NetKet local
    operators.
    """

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
            sigma: Input states or "bra", acting from the left.

        Returns:
            connected states or "kets" eta, corresponding matrix elements mels

        """
        return abs_xmagnetization_conns_and_mels(sigma.reshape(-1, sigma.shape[-1]))

    def get_conn_flattened(self, x, sections):
        eta, mels = abs_xmagnetization_conns_and_mels(x)

        n_primes = eta.shape[1]  # number of connected states, same for all possible configurations x
        n_visible = x.shape[1]
        eta = eta.reshape(-1, n_visible)  # flatten last dimension
        mels = mels.flatten()

        # must manipulate sections in place
        for i in range(len(sections)):
            sections[i] = (i + 1) * n_primes

        return eta, mels


class AbsYMagnetization(nk.operator.DiscreteOperator):
    """
    Operator corresponing to the absolute magnetization of a state. For regular magnetization simply use NetKet local
    operators.
    """

    def __init__(self, hilbert: nk.hilbert.DiscreteHilbert):
        super().__init__(hilbert)

    @property
    def dtype(self):
        return complex

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
        return abs_ymagnetization_conns_and_mels(sigma.reshape(-1, sigma.shape[-1]))

    def get_conn_flattened(self, x, sections):
        eta, mels = abs_ymagnetization_conns_and_mels(x)

        n_primes = eta.shape[1]  # number of connected states, same for all possible configurations x
        n_visible = x.shape[1]
        eta = eta.reshape(-1, n_visible)  # flatten last dimension
        mels = mels.flatten()

        # must manipulate sections in place
        for i in range(len(sections)):
            sections[i] = (i + 1) * n_primes

        return eta, mels


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: AbsZMagnetization):
    return abs_e_loc


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: AbsXMagnetization):
    return abs_e_loc


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: AbsYMagnetization):
    return abs_e_loc


@jax.jit
@partial(jax.vmap, in_axes=(0,))
def abs_zmagnetization_conns_and_mels(sigma: jax.Array) -> Tuple[jax.Array, jax.Array]:
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


@jax.jit
@partial(jax.vmap, in_axes=(0,))
def abs_xmagnetization_conns_and_mels(sigma: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    For a given input spin configuration sigma, calculates all connected states eta and the corresponding non-zero
    matrix elements mels. See netket for further details.
    Args:
        sigma: Input state or "bra", acting from the left.

    Returns:
        connected states or "kets" eta, corresponding matrix elements mels

    """
    n_sites = sigma.shape[-1]

    @jax.vmap
    def flip(x, idx):
        return x.at[idx].set(-x.at[idx].get())

    # connected states is sigma itself
    eta = jnp.tile(sigma, (n_sites, 1))
    eta = eta.at[:].set(flip(eta.at[:].get(), jnp.arange(n_sites)))

    # connected matrix elements
    mels = jnp.ones(n_sites) / n_sites

    return eta, mels


@jax.jit
@partial(jax.vmap, in_axes=(0,))
def abs_ymagnetization_conns_and_mels(sigma: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    For a given input spin configuration sigma, calculates all connected states eta and the corresponding non-zero
    matrix elements mels. See netket for further details.
    Args:
        sigma: Input state or "bra", acting from the left.

    Returns:
        connected states or "kets" eta, corresponding matrix elements mels

    """
    n_sites = sigma.shape[-1]

    @jax.vmap
    def flip(x, idx):
        return x.at[idx].set(-x.at[idx].get())

    # connected states is sigma itself
    eta = jnp.tile(sigma, (n_sites, 1))
    eta = eta.at[:].set(flip(eta.at[:].get(), jnp.arange(n_sites)))

    # connected matrix elements
    mels = 1j * sigma / n_sites

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


def abs_e_loc(logpsi, pars, sigma, extra_args):
    """
    Calculates the absolute local estimate of the operator.
    Args:
        logpsi: Wavefunction, taking as input pars and some state, returning the log amplitude.
        pars: Parameters for logpsi model.
        sigma: Samples from which to calculate the estimate.
        extra_args: Connected states, non-zero matrix elements.

    Returns:
        local estimates: | sum_{eta} <sigma|Operator|eta> * psi(eta)/psi(sigma) | over different sigmas

    """
    eta, mels = extra_args
    assert sigma.ndim == 2, f"sigma dimensions should be (Nsamples, Nsites), but has dimensions {sigma.shape}"
    assert eta.ndim == 3, f"eta dimensions should be (Nsamples, Nconnected, Nsites), but has dimensions {eta.shape}"

    @partial(jax.vmap, in_axes=(0, 0, 0))
    def _loc_vals(sigma, eta, mels):
        return jnp.abs(jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1))

    return _loc_vals(sigma, eta, mels)


def get_netket_wilsonob(hilbert, shape):
    wilson = nk.operator.LocalOperator(hilbert, dtype=float)
    x, y = shape[0], shape[1]
    for i in range(shape[0]):
        for j in range(shape[1]):
            # add pauliz loop corresponding to plaquette operators, clockwise
            wilson += nk.operator.spin.sigmaz(hilbert,
                                              edge_to_index(jnp.array([i, j]), 1, shape).item()) * \
                      nk.operator.spin.sigmaz(hilbert,
                                              edge_to_index(jnp.array([i, (j + 1) % y]), 1, shape).item()) * \
                      nk.operator.spin.sigmaz(hilbert,
                                              edge_to_index(jnp.array([i, (j + 2) % y]), 0, shape).item()) * \
                      nk.operator.spin.sigmaz(hilbert,
                                              edge_to_index(jnp.array([(i + 1) % x, (j + 2) % y]), 0, shape).item()) * \
                      nk.operator.spin.sigmaz(hilbert,
                                              edge_to_index(jnp.array([(i + 2) % x, (j + 1) % y]), 1, shape).item()) * \
                      nk.operator.spin.sigmaz(hilbert,
                                              edge_to_index(jnp.array([(i + 2) % x, j]), 1, shape).item()) * \
                      nk.operator.spin.sigmaz(hilbert,
                                              edge_to_index(jnp.array([(i + 1) % x, j]), 0, shape).item()) * \
                      nk.operator.spin.sigmaz(hilbert,
                                              edge_to_index(jnp.array([i, j]), 0, shape).item())

            # now add paulix loop corresponding to star operators, clockwise
            wilson += nk.operator.spin.sigmax(hilbert,
                                              edge_to_index(jnp.array([(i - 1) % x, j]), 0, shape).item()) * \
                      nk.operator.spin.sigmax(hilbert,
                                              edge_to_index(jnp.array([(i - 1) % x, (j + 1) % y]), 0, shape).item()) * \
                      nk.operator.spin.sigmax(hilbert,
                                              edge_to_index(jnp.array([i, (j + 1) % y]), 1, shape).item()) * \
                      nk.operator.spin.sigmax(hilbert,
                                              edge_to_index(jnp.array([(i + 1) % x, (j + 1) % y]), 1, shape).item()) * \
                      nk.operator.spin.sigmax(hilbert,
                                              edge_to_index(jnp.array([(i + 1) % x, (j + 1) % y]), 0, shape).item()) * \
                      nk.operator.spin.sigmax(hilbert,
                                              edge_to_index(jnp.array([(i + 1) % x, j]), 0, shape).item()) * \
                      nk.operator.spin.sigmax(hilbert,
                                              edge_to_index(jnp.array([(i + 1) % x, (j - 1) % y]), 1, shape).item()) * \
                      nk.operator.spin.sigmax(hilbert,
                                              edge_to_index(jnp.array([i, (j - 1) % y]), 1, shape).item())
    return wilson / hilbert.size
