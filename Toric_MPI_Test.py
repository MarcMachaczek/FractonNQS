import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
os.environ["MPI4JAX_USE_CUDA_MPI"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import jax
import jax.numpy as jnp

import netket as nk
from netket.utils import HashableArray
from netket import nn as nknn
from flax import linen as nn
from flax.linen.dtypes import promote_dtype

from typing import Union, Any, Callable, Sequence

# %%
class ToricCRBM(nn.Module):
    # permutations of lattice sites corresponding to symmetries
    symmetries: HashableArray
    # correlators that serve as additional input for the cRBM
    correlators: Sequence[HashableArray]
    # permutations of correlators corresponding to symmetries
    correlator_symmetries: Sequence[HashableArray]
    # The dtype of the weights
    param_dtype: Any = jnp.float64
    # The nonlinear activation function
    activation: Any = nknn.log_cosh
    # feature density. Number of features equal to alpha * input.shape[-1]
    alpha: Union[float, int] = 1

    # Initializer for the Dense layer matrix
    kernel_init: Callable = jax.nn.initializers.normal(0.01)
    # Initializer for the biases
    bias_init: Callable = jax.nn.initializers.normal(0.01)

    def setup(self):
        self.n_symm, self.n_sites = self.symmetries.wrapped.shape
        self.features = int(self.alpha * self.n_sites / self.n_symm)

    @nn.compact
    def __call__(self, x):
        n_batch = x.shape[0]
        # initialize bias and kernel for the "single spin correlators" analogous to GCNN
        hidden_bias = self.param("hidden_bias", self.bias_init, (self.features,), self.param_dtype)
        symm_kernel = self.param("symm_kernel", self.kernel_init, (self.features, self.n_sites), self.param_dtype)

        # take care of possibly different dtypes (e.g. x is float while parameters are complex)
        x, hidden_bias, symm_kernel = promote_dtype(x, hidden_bias, symm_kernel, dtype=None)

        # convert kernel to dense kernel of shape (features, n_symmetries, n_sites)
        symm_kernel = jnp.take(symm_kernel, self.symmetries.wrapped, axis=1)

        # x has shape (batch, n_sites), kernel has shape (features, n_symmetries, n_sites)
        # theta has shape (batch, features, n_symmetries)
        theta = jax.lax.dot_general(x, symm_kernel, (((1,), (2,)), ((), ())))
        theta += jnp.expand_dims(hidden_bias, 1)

        # here, use two visible biases according to the two spins in one unit cell
        visible_bias = self.param("visible_bias", self.bias_init, (2,), self.param_dtype)
        bias = jnp.sum(x.reshape(n_batch, -1, 2), axis=(1,)) @ visible_bias

        for i, (correlator, correlator_symmetry) in enumerate(zip(self.correlators, self.correlator_symmetries)):
            # initialize "visible" bias and kernel matrix for correlator
            correlator = correlator.wrapped  # convert hashable array to (usable) jax.Array
            corr_bias = self.param(f"corr{i}_bias", self.bias_init, (1,), self.param_dtype)
            corr_kernel = self.param(f"corr{i}_kernel", self.kernel_init, (self.features, len(correlator)),
                                     self.param_dtype)
            
            x, corr_bias, corr_kernel = promote_dtype(x, corr_bias, corr_kernel, dtype=None)

            # convert kernel to dense kernel of shape (features, n_correlator_symmetries, n_corrs)
            corr_kernel = jnp.take(corr_kernel, correlator_symmetry.wrapped, axis=1)

            # correlator has shape (n_corrs, degree), e.g. n_corrs=L²=n_site/2 and degree=4 for plaquettes
            corr_values = jnp.take(x, correlator, axis=1).prod(axis=2)  # shape (batch, n_corrs)

            # corr_values has shape (batch, n_corrs)
            # kernel has shape (features, n_correlator_symmetries, n_corrs)
            # theta has shape (batch, features, n_symmetries)
            theta += jax.lax.dot_general(corr_values, corr_kernel, (((1,), (2,)), ((), ())))
            bias += corr_bias * jnp.sum(corr_values, axis=(1,))

        out = self.activation(theta)
        out = jnp.sum(out, axis=(1, 2))  # sum over all symmetries and features = alpha * n_sites / n_symmetries
        out += bias

        for i, (loop_corr, loop_symmetries) in enumerate(zip(self.correlators, self.correlator_symmetries)):
            # initialize "visible" bias and kernel matrix for loop correlator
            loop_corr = loop_corr.wrapped  # convert hashable array to (usable) jax.Array
            loop_hidden_bias = self.param(f"loop{i}_hidden_bias", self.bias_init, (self.features,), self.param_dtype)
            loop_kernel = self.param(f"loop{i}_kernel", self.kernel_init, (self.features, len(loop_corr)),
                                     self.param_dtype)
            
            x, loop_hidden_bias, loop_kernel = promote_dtype(x, loop_hidden_bias, loop_kernel, dtype=None)

            # convert kernel to dense kernel of shape (features, n_correlator_symmetries, n_corrs)
            loop_kernel = jnp.take(loop_kernel, loop_symmetries.wrapped, axis=1)

            # loop_corr has shape (n_corrs, degree)
            loop_values = jnp.take(x, loop_corr, axis=1).prod(axis=2)  # shape (batch, n_corrs)

            # loop_values has shape (batch, n_corrs)
            # kernel has shape (features, n_loop_symmetries, n_corrs)
            # loop_theta has shape (batch, features, n_symmetries)
            loop_theta = jax.lax.dot_general(loop_values, loop_kernel, (((1,), (2,)), ((), ())))
            loop_theta += jnp.expand_dims(loop_hidden_bias, 1)
            loop_out = jnp.sum(self.activation(loop_theta), axis=(1, 2))
            # this last line causes mpi allreduce error code 15
            out += loop_out

        return out

# %%
L = 3  # size should be at least 3, else there are problems with pbc and indexing
N  = 2*L**2
hilbert = nk.hilbert.Spin(s=1 / 2, N=N)
ham = sum([nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size)])

perms = jnp.stack([(jnp.arange(int(N/2)) + i) % N for i in range(int(N/2))], axis=0)
perms = HashableArray(perms)
link_perms = jnp.stack([(jnp.arange(int(N)) + i) % N for i in range(int(N/2))], axis=0)
link_perms = HashableArray(link_perms)
plaqs = jnp.array([[1, 2, 3, 4] for _ in range(int(N/2))])
plaqs = HashableArray(plaqs)

correlators = (plaqs,)
correlator_symmetries = (perms,)

model = ToricCRBM(symmetries=link_perms,
                  correlators=correlators,
                  correlator_symmetries=correlator_symmetries,
                  param_dtype=complex)

single_rule = nk.sampler.rules.LocalRule()
sampler = nk.sampler.MetropolisSampler(hilbert, rule=single_rule)
variational_gs = nk.vqs.MCState(sampler, model)

exp, forces = variational_gs.expect_and_forces(ham)
