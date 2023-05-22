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
from netket import jax as nkjax
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch
from netket.vqs.mc import (
    get_local_kernel_arguments,
    get_local_kernel,
)

from flax import linen as nn
from flax.linen.dtypes import promote_dtype

from typing import Union, Any, Callable, Sequence
from functools import partial


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

    # Initializer for the Dense layer matrix
    kernel_init: Callable = jax.nn.initializers.normal(0.01)

    def setup(self):
        self.n_symm, self.n_sites = self.symmetries.wrapped.shape
        self.features = 2  # int(self.alpha * self.n_sites / self.n_symm)

    @nn.compact
    def __call__(self, x):
        # initialize bias and kernel for the "single spin correlators" analogous to GCNN
        symm_kernel = self.param("symm_kernel", self.kernel_init, (self.features, self.n_sites), self.param_dtype)

        # take care of possibly different dtypes (e.g. x is float while parameters are complex)
        x, symm_kernel = promote_dtype(x, symm_kernel, dtype=None)

        # convert kernel to dense kernel of shape (features, n_symmetries, n_sites)
        symm_kernel = jnp.take(symm_kernel, self.symmetries.wrapped, axis=1)

        # x has shape (batch, n_sites), kernel has shape (features, n_symmetries, n_sites)
        # theta has shape (batch, features, n_symmetries)
        theta = jax.lax.dot_general(x, symm_kernel, (((1,), (2,)), ((), ())))

        for i, (correlator, correlator_symmetry) in enumerate(zip(self.correlators[:-1], self.correlator_symmetries[:-1])):
            # initialize "visible" bias and kernel matrix for correlator
            # correlator = correlator.wrapped  # convert hashable array to (usable) jax.Array
            correlator = jnp.array([[1, 2, 3, 4] for _ in range(int(N/2))])
            corr_kernel = self.param(f"corr{i}_kernel", self.kernel_init, (self.features, len(correlator)),
                                     self.param_dtype)
            #jax.debug.print("{x}", x=corr_kernel.dtype)
            
            # convert kernel to dense kernel of shape (features, n_correlator_symmetries, n_corrs)
            # corr_kernel = jnp.take(corr_kernel, correlator_symmetry.wrapped, axis=1)
            corr_kernel = jnp.take(corr_kernel, jnp.stack([(jnp.arange(int(N/2)) + i) % int(N/2) for i in range(int(N/2))], axis=0), axis=1)
            

            # correlator has shape (n_corrs, degree), e.g. n_corrs=L²=n_site/2 and degree=4 for plaquettes
            corr_values = jnp.take(x, correlator, axis=1).prod(axis=2)  # shape (batch, n_corrs)

            # corr_values has shape (batch, n_corrs)
            # kernel has shape (features, n_correlator_symmetries, n_corrs)
            # theta has shape (batch, features, n_symmetries)
            theta += jax.lax.dot_general(corr_values, corr_kernel, (((1,), (2,)), ((), ())))

        out = self.activation(theta)
        out = jnp.sum(out, axis=(1, 2))  # sum over all symmetries and features = alpha * n_sites / n_symmetries

        for i, (loop_corr, loop_symmetries) in enumerate(zip(self.correlators[1:], self.correlator_symmetries[1:])):
            # initialize "visible" bias and kernel matrix for loop correlator
            # loop_corr = loop_corr.wrapped  # convert hashable array to (usable) jax.Array
            loop_corr = jnp.array([[1, 2, 3, 4] for _ in range(int(N/2))])
            loop_kernel = self.param(f"loop{i}_kernel", self.kernel_init, (self.features, len(loop_corr)),
                                     self.param_dtype)

            # convert kernel to dense kernel of shape (features, n_correlator_symmetries, n_corrs)
            # loop_kernel = jnp.take(loop_kernel, loop_symmetries.wrapped, axis=1)
            loop_kernel = jnp.take(loop_kernel, jnp.stack([(jnp.arange(int(N/2)) + i) % int(N/2) for i in range(int(N/2))], axis=0), axis=1)

            # loop_corr has shape (n_corrs, degree)
            loop_values = jnp.take(x, loop_corr, axis=1).prod(axis=2)  # shape (batch, n_corrs)

            # loop_values has shape (batch, n_corrs)
            # kernel has shape (features, n_loop_symmetries, n_corrs)
            # loop_theta has shape (batch, features, n_symmetries)
            loop_theta = jax.lax.dot_general(loop_values, loop_kernel, (((1,), (2,)), ((), ())))
            loop_out = jnp.sum(self.activation(loop_theta), axis=(1, 2))
            # this last line causes mpi allreduce error code 15
            out += loop_out
        return out
    

@partial(jax.jit, static_argnums=(0, 1, 2))
def forces_expect_hermitian(
    local_value_kernel,
    model_apply_fun,
    mutable,
    parameters,
    model_state,
    σ,
    local_value_args):

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples = σ.shape[0] * mpi.n_nodes

    O_loc = local_value_kernel(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
    )

    Ō = statistics(O_loc.reshape(σ_shape[:-1]).T)

    # O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]
    
    jax.debug.print("rank{0}, pars: {1}", mpi.rank, parameters)
    jax.debug.print("rank{0},O_loc: {1}", mpi.rank, O_loc)
    jax.debug.print("rank{0},O_grad: {1}", mpi.rank, Ō_grad)

    new_model_state = new_model_state[0] if is_mutable else None

    return Ō, jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad), new_model_state


# %%
L = 3  # size should be at least 3, else there are problems with pbc and indexing
N  = 2*L**2
hilbert = nk.hilbert.Spin(s=1 / 2, N=N)
ham = sum([nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size)])

perms = jnp.stack([(jnp.arange(int(N/2)) + i) % int(N/2) for i in range(int(N/2))], axis=0)
perms = HashableArray(perms)
perms2 = HashableArray(perms)
link_perms = jnp.stack([(jnp.arange(int(N)) + i) % N for i in range(int(N/2))], axis=0)
link_perms = HashableArray(link_perms)
plaqs = jnp.array([[1, 2, 3, 4] for _ in range(int(N/2))])
plaqs = HashableArray(plaqs)
plaqs2 = HashableArray(plaqs)

correlators = (plaqs, plaqs2)
correlator_symmetries = (perms, perms2)

model = ToricCRBM(symmetries=link_perms,
                  correlators=correlators,
                  correlator_symmetries=correlator_symmetries,
                  param_dtype=jnp.complex128)

single_rule = nk.sampler.rules.LocalRule()
sampler = nk.sampler.MetropolisSampler(hilbert, rule=single_rule)
variational_gs = nk.vqs.MCState(sampler, model, n_samples=4032)

σ, args = get_local_kernel_arguments(variational_gs, ham)
local_estimator_fun = get_local_kernel(variational_gs, ham)

sigma = jnp.ones(18)
sigma2 = sigma.at[1].set(-1)
sigmas  =jnp.stack([sigma, sigma2])
model.apply({"params": variational_gs.parameters}, sigmas)

exp, forces, _ = forces_expect_hermitian(local_estimator_fun, variational_gs._apply_fun, False, variational_gs.parameters, variational_gs.model_state, σ, args)
