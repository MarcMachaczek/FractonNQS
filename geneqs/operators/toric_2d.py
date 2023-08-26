import jax
import jax.numpy as jnp
import netket as nk
import geneqs

from typing import Tuple
from functools import partial


# %%
class ToricCode2d(nk.operator.DiscreteOperator):
    def __init__(self, hilbert: nk.hilbert.DiscreteHilbert, shape: jax.Array, h: Tuple[float, float, float] = None):
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
        return complex if self.h[1] != 0 else float

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
        return toric2d_conns_and_mels(sigma, self.plaqs, self.stars, self.h)

    def get_conn_flattened(self, x, sections):
        eta, mels = toric2d_conns_and_mels(x, self.plaqs, self.stars, self.h)

        n_primes = eta.shape[1]  # number of connected states, same for all possible configurations x
        n_visible = x.shape[1]
        eta = eta.reshape(-1, n_visible)  # flatten last dimension
        mels = mels.flatten()

        # must manipulate sections in place
        for i in range(len(sections)):
            sections[i] = (i + 1) * n_primes

        return eta, mels


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
    matrix elements mels. See netket for further details. This version is only faster than netket if external fields are
    non-zero, as it already includes corresponding conns and mels, even when fields are zero.
    H = H_toric - hx * sum_i sigma_x - hy * sum_j sigma_y - hz * sum_k sigma_z
    Args:
        sigma: Input states or "bra", acting from the left.
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

    # initialize connected states
    eta = jnp.tile(sigma, (1 + n_sites + 2*n_sites, 1))
    # connected states by star operators, leave the first eta as is (diagonal connected state)
    eta = eta.at[1:n_sites+1].set(flip(eta.at[1:n_sites+1].get(), stars))
    # connected states through external field (sigma_x and sigma_y)
    eta = eta.at[n_sites+1:3*n_sites+1].set(flip(eta.at[n_sites+1:3*n_sites+1].get(), jnp.arange(2*n_sites)))

    # calculate matrix elements
    # axis 0 of sigma.at[plaqs] corresponds to #N_plaqs and axis 1 to the 4 edges of one plaquette
    diag_mel = -jnp.sum(jnp.product(sigma.at[plaqs].get(), axis=1)) + hz * jnp.sum(sigma)
    # n_sites mels corresponding to flipped stars
    star_mels = -jnp.ones(n_sites)
    # mels according to hx and hy, NOTE: chunking with non-zero hy might not work (netket internal stuff)
    field_mels = -hx * jnp.ones(2*n_sites) - hy * sigma * 1j
    mels = jnp.hstack((diag_mel, star_mels, field_mels))
    return eta, mels


def e_loc(logpsi, pars, sigma, extra_args):
    """
    Calculates the local estimators of the operator.
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


def get_netket_toric2dh(hilbert, shape: jax.Array, h: Tuple[float, float, float]):
    """
    Retrieve a NetKet version of the 2d Toric code in external fields. (Constructed from NetKet local operators)
    Args:
        hilbert: The Hilbert space. Should be NetKet spin hilbert space.
        shape: Shape of the lattice.
        h: Tuple of field strengths of the external magentic field in x, y and z direction.

    Returns:
        NetKet discrete operator version of Toric Code.

    """
    ha_netketlocal = nk.operator.LocalOperator(hilbert, dtype=complex)
    hx, hy, hz = h
    # adding the plaquette terms:
    for i in range(shape[0]):
        for j in range(shape[1]):
            ha_netketlocal -= get_netket_plaq(hilbert, jnp.array([i, j]), shape)
            
    # adding the star terms
    for i in range(shape[0]):
        for j in range(shape[1]):
            ha_netketlocal -= get_netket_star(hilbert, jnp.array([i, j]), shape)

    # adding external fields
    ha_netketlocal -= hz * sum([nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size)])
    ha_netketlocal -= hx * sum([nk.operator.spin.sigmax(hilbert, i) for i in range(hilbert.size)])
    ha_netketlocal -= hy * sum([nk.operator.spin.sigmay(hilbert, i) for i in range(hilbert.size)])

    return ha_netketlocal


def get_netket_star(hilbert, position: jax.Array, shape: jax.Array):
    """
    Create a star (consisting of pauli_x operators) operator at position acting on hilbert space.
    Args:
        hilbert: NetKet spin hilbert space.
        position: Position of the operator.
        shape: Shape of the lattice.

    Returns:
        Star operator.

    """
    star = nk.operator.LocalOperator(hilbert, dtype=float)
    star_indices = geneqs.utils.indexing.position_to_star(position, shape)
    star += nk.operator.spin.sigmax(hilbert, star_indices[0].item()) * \
            nk.operator.spin.sigmax(hilbert, star_indices[1].item()) * \
            nk.operator.spin.sigmax(hilbert, star_indices[2].item()) * \
            nk.operator.spin.sigmax(hilbert, star_indices[3].item())
    return star
            
            
def get_netket_plaq(hilbert, position: jax.Array, shape: jax.Array):
    """
        Create a plaquette (consisting of pauli_z operators) operator at position acting on hilbert space.
        Args:
            hilbert: NetKet spin hilbert space.
            position: Position of the operator.
            shape: Shape of the lattice.

        Returns:
            Plaquette operator.

        """
    plaq = nk.operator.LocalOperator(hilbert, dtype=float)
    plaq_indices = geneqs.utils.indexing.position_to_plaq(position, shape)
    plaq += nk.operator.spin.sigmaz(hilbert, plaq_indices[0].item()) * \
            nk.operator.spin.sigmaz(hilbert, plaq_indices[1].item()) * \
            nk.operator.spin.sigmaz(hilbert, plaq_indices[2].item()) * \
            nk.operator.spin.sigmaz(hilbert, plaq_indices[3].item())
    return plaq

class ToricCode2d_xbasis(nk.operator.DiscreteOperator):
    def __init__(self, hilbert: nk.hilbert.DiscreteHilbert, shape: jax.Array, h: Tuple[float, float, float] = None):
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
        return complex if self.h[1] != 0 else float

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
        return toric2d_conns_and_mels(sigma, self.plaqs, self.stars, self.h)

    def get_conn_flattened(self, x, sections):
        eta, mels = toric2d_conns_and_mels(x, self.plaqs, self.stars, self.h)

        n_primes = eta.shape[1]  # number of connected states, same for all possible configurations x
        n_visible = x.shape[1]
        eta = eta.reshape(-1, n_visible)  # flatten last dimension
        mels = mels.flatten()

        # must manipulate sections in place
        for i in range(len(sections)):
            sections[i] = (i + 1) * n_primes

        return eta, mels


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: ToricCode2d_xbasis):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: ToricCode2d_xbasis):
    sigma = vstate.samples
    extra_args = toric2d_xbasis_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size), op.plaqs, op.stars, op.h)
    return sigma, extra_args


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None))
def toric2d_xbasis_conns_and_mels(sigma: jax.Array,
                                  plaqs: jax.Array,
                                  stars: jax.Array,
                                  h: Tuple[float, float, float]) -> Tuple[jax.Array, jax.Array]:
    """
    This is the toric code implementation in the x-basis! essentially swap sigma_x and sigma_z, sigma_y with -sigma_y
    For a given input spin configuration sigma, calculates all connected states eta and the corresponding non-zero
    matrix elements mels. See netket for further details. This version is only faster than netket if external fields are
    non-zero, as it already includes corresponding conns and mels, even when fields are zero.
    H = H_toric - hx * sum_i sigma_x - hy * sum_j sigma_y - hz * sum_k sigma_z
    Args:
        sigma: Input states or "bra", acting from the left.
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

    # initialize connected states
    eta = jnp.tile(sigma, (1 + n_sites + 2*n_sites, 1))
    # connected states by plaq operators, leave the first eta as is (diagonal connected state)
    eta = eta.at[1:n_sites+1].set(flip(eta.at[1:n_sites+1].get(), plaqs))
    # connected states through external field (sigma_z and sigma_y)
    eta = eta.at[n_sites+1:3*n_sites+1].set(flip(eta.at[n_sites+1:3*n_sites+1].get(), jnp.arange(2*n_sites)))

    # calculate matrix elements
    # axis 0 of sigma.at[plaqs] corresponds to #N_plaqs and axis 1 to the 4 edges of one plaquette
    diag_mel = -jnp.sum(jnp.product(sigma.at[stars].get(), axis=1)) + hx * jnp.sum(sigma)
    # n_sites mels corresponding to flipped plaqs
    plaq_mels = -jnp.ones(n_sites)
    # mels according to hz and hy, NOTE: chunking with non-zero hy might not work (netket internal stuff)
    field_mels = -hz * jnp.ones(2*n_sites) + hy * sigma * 1j
    mels = jnp.hstack((diag_mel, plaq_mels, field_mels))
    return eta, mels