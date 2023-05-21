import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from netket import nn as nknn
from netket.utils import HashableArray

from typing import Union, Any, Callable, Sequence

default_kernel_init = jax.nn.initializers.normal(stddev=0.01)


# %%
class ToricLoopCRBM(nn.Module):
    """
    Important: Network assumes the last two correlators provided are loop correlators to which additionall hidden units
    are attached exclusively.
    Difference to regular CorrelationRBM: Two separate bias terms for single spins according to the two sublattices
    of the Toric Code model; extra hidden units exclusively connected to the loop correlators.
    """
    # permutations of lattice sites corresponding to symmetries
    symmetries: HashableArray
    # correlators that serve as additional input for the cRBM
    correlators: Sequence[HashableArray]
    # permutations of correlators corresponding to symmetries
    correlator_symmetries: Sequence[HashableArray]
    # loops that serve as additional input for the cRBM, extra hidden units are connected to just them
    loops: Sequence[HashableArray]
    # permutations of loop correlators corresponding to symmetries
    loop_symmetries: Sequence[HashableArray]
    # The dtype of the weights
    param_dtype: Any = jnp.float64
    # The nonlinear activation function
    activation: Any = nknn.log_cosh
    # feature density. Number of features equal to alpha * input.shape[-1]
    alpha: Union[float, int] = 1
    # Numerical precision of the computation see :class:`jax.lax.Precision` for details
    precision: Any = None

    # Initializer for the Dense layer matrix
    kernel_init: Callable = default_kernel_init
    # Initializer for the biases
    bias_init: Callable = default_kernel_init

    def setup(self):
        self.n_symm, self.n_sites = self.symmetries.__array__().shape
        self.features = int(self.alpha * self.n_sites / self.n_symm)
        if self.alpha > 0 and self.features == 0:
            raise ValueError(
                f"RBMSymm: alpha={self.alpha} is too small "
                f"for {self.n_symm} permutations, alpha ≥ {self.n_symm / self.n_sites} is needed.")

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
        theta = jax.lax.dot_general(x, symm_kernel, (((1,), (2,)), ((), ())), precision=self.precision)
        theta += jnp.expand_dims(hidden_bias, 1)

        # here, use two visible biases according to the two spins in one unit cell
        visible_bias = self.param("visible_bias", self.bias_init, (2,), self.param_dtype)
        bias = jnp.sum(x.reshape(n_batch, -1, 2), axis=(1,)) @ visible_bias

        for i, (correlator, corr_symmetry) in enumerate(zip(self.correlators, self.correlator_symmetries)):
            # initialize "visible" bias and kernel matrix for correlator
            correlator = correlator.wrapped  # convert hashable array to (usable) jax.Array
            corr_bias = self.param(f"corr{i}_bias", self.bias_init, (1,), self.param_dtype)
            corr_kernel = self.param(f"corr{i}_kernel", self.kernel_init, (self.features, len(correlator)),
                                     self.param_dtype)

            # convert kernel to dense kernel of shape (features, n_correlator_symmetries, n_corrs)
            corr_kernel = jnp.take(corr_kernel, corr_symmetry.wrapped, axis=1)

            # correlator has shape (n_corrs, degree), e.g. n_corrs=L²=n_site/2 and degree=4 for plaquettes
            corr_values = jnp.take(x, correlator, axis=1).prod(axis=2)  # shape (batch, n_corrs)

            # corr_values has shape (batch, n_corrs)
            # kernel has shape (features, n_correlator_symmetries, n_corrs)
            # theta has shape (batch, features, n_symmetries)
            theta += jax.lax.dot_general(corr_values, corr_kernel, (((1,), (2,)), ((), ())), precision=self.precision)
            bias += corr_bias * jnp.sum(corr_values, axis=(1,))

        # add loop features corresponding to hidden units only connected to loops
        loop_out = 0
        for i, (loop_corr, loop_symmetry) in enumerate(zip(self.loops, self.loop_symmetries)):
            # initialize "visible" bias and kernel matrix for loop correlator
            loop_corr = loop_corr.wrapped  # convert hashable array to (usable) jax.Array
            loop_hidden_bias = self.param(f"loop{i}_hidden_bias", self.bias_init, (self.features,), self.param_dtype)
            loop_visible_bias = self.param(f"loop{i}_visible_bias", self.bias_init, (1,), self.param_dtype)
            loop_kernel = self.param(f"loop{i}_kernel", self.kernel_init, (self.features, len(loop_corr)),
                                     self.param_dtype)

            # convert kernel to dense kernel of shape (features, n_correlator_symmetries, n_corrs)
            loop_kernel = jnp.take(loop_kernel, loop_symmetry.wrapped, axis=1)

            # loop_corr has shape (n_corrs, degree)
            loop_values = jnp.take(x, loop_corr, axis=1).prod(axis=2)  # shape (batch, n_corrs)

            # loop_values has shape (batch, n_corrs)
            # kernel has shape (features, n_loop_symmetries, n_corrs)
            # loop_theta has shape (batch, features, n_symmetries)
            loop_theta = jax.lax.dot_general(loop_values, loop_kernel, (((1,), (2,)), ((), ())),
                                             precision=self.precision)

            # add loop features to original rbm input
            theta += loop_theta
            bias += loop_visible_bias * jnp.sum(loop_values, axis=(1,))

            # now for the extra factors
            loop_theta += jnp.expand_dims(loop_hidden_bias, 1)
            loop_out += jnp.sum(self.activation(loop_theta), axis=(1, 2))

        self.sow("intermediates", "activation_inputs", theta)
        out = self.activation(theta)
        self.sow("intermediates", "activation_outputs", out)
        out = jnp.sum(out, axis=(1, 2))  # sum over all symmetries and features = alpha * n_sites / n_symmetries
        out += loop_out + bias
        return out


# %%
class ToricCRBM(nn.Module):
    """
    Important: Network assumes the last two correlators provided are loop correlators to which additionall hidden units
    are attached exclusively.
    Difference to regular CorrelationRBM: Two separate bias terms for single spins according to the two sublattices
    of the Toric Code model; extra hidden units exclusively connected to the loop correlators.
    """
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
    # Numerical precision of the computation see :class:`jax.lax.Precision` for details
    precision: Any = None

    # Initializer for the Dense layer matrix
    kernel_init: Callable = default_kernel_init
    # Initializer for the biases
    bias_init: Callable = default_kernel_init

    def setup(self):
        self.n_symm, self.n_sites = self.symmetries.__array__().shape
        self.features = int(self.alpha * self.n_sites / self.n_symm)
        if self.alpha > 0 and self.features == 0:
            raise ValueError(
                f"RBMSymm: alpha={self.alpha} is too small "
                f"for {self.n_symm} permutations, alpha ≥ {self.n_symm / self.n_sites} is needed.")

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
        theta = jax.lax.dot_general(x, symm_kernel, (((1,), (2,)), ((), ())), precision=self.precision)
        theta += jnp.expand_dims(hidden_bias, 1)

        # here, use two visible biases according to the two spins in one unit cell
        visible_bias = self.param("visible_bias", self.bias_init, (2,), self.param_dtype)
        bias = jnp.sum(x.reshape(n_batch, -1, 2), axis=(1,)) @ visible_bias

        for i, correlator in enumerate(self.correlators):
            # initialize "visible" bias and kernel matrix for correlator
            correlator = correlator.wrapped  # convert hashable array to (usable) jax.Array
            corr_bias = self.param(f"corr{i}_bias", self.bias_init, (1,), self.param_dtype)
            corr_kernel = self.param(f"corr{i}_kernel", self.kernel_init, (self.features, len(correlator)),
                                     self.param_dtype)
            
            x, corr_bias, corr_kernel = promote_dtype(x, corr_bias, corr_kernel, dtype=None)

            # convert kernel to dense kernel of shape (features, n_correlator_symmetries, n_corrs)
            corr_kernel = jnp.take(corr_kernel, self.correlator_symmetries[i].wrapped, axis=1)

            # correlator has shape (n_corrs, degree), e.g. n_corrs=L²=n_site/2 and degree=4 for plaquettes
            corr_values = jnp.take(x, correlator, axis=1).prod(axis=2)  # shape (batch, n_corrs)

            # corr_values has shape (batch, n_corrs)
            # kernel has shape (features, n_correlator_symmetries, n_corrs)
            # theta has shape (batch, features, n_symmetries)
            theta += jax.lax.dot_general(corr_values, corr_kernel, (((1,), (2,)), ((), ())), precision=self.precision)
            bias += corr_bias * jnp.sum(corr_values, axis=(1,))

        self.sow("intermediates", "activation_inputs", theta)
        out = self.activation(theta)
        self.sow("intermediates", "activation_outputs", out)
        out = jnp.sum(out, axis=(1, 2))  # sum over all symmetries and features = alpha * n_sites / n_symmetries
        out += bias

        # add loop features corresponding to hidden units only connected to loops
        for i, (loop_corr, loop_symmetries) in enumerate(zip(self.correlators[-2:], self.correlator_symmetries[-2:])):
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
            loop_theta = jax.lax.dot_general(loop_values, loop_kernel, (((1,), (2,)), ((), ())),
                                             precision=self.precision)
            loop_theta += jnp.expand_dims(loop_hidden_bias, 1)
            loop_out = jnp.sum(self.activation(loop_theta), axis=(1, 2))
            # this last line causes mpi allreduce error code 15
            out += loop_out

        return out


# %%
class CorrelationRBM(nn.Module):
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
    # Numerical precision of the computation see :class:`jax.lax.Precision` for details
    precision: Any = None

    # Initializer for the Dense layer matrix
    kernel_init: Callable = default_kernel_init
    # Initializer for the biases
    bias_init: Callable = default_kernel_init

    def setup(self):
        self.n_symm, self.n_sites = self.symmetries.__array__().shape
        self.features = int(self.alpha * self.n_sites / self.n_symm)
        if self.alpha > 0 and self.features == 0:
            raise ValueError(
                f"RBMSymm: alpha={self.alpha} is too small "
                f"for {self.n_symm} permutations, alpha ≥ {self.n_symm / self.n_sites} is needed.")

    @nn.compact
    def __call__(self, x):
        # initialize bias and kernel for the "single spin correlators" analogous to GCNN
        hidden_bias = self.param("hidden_bias", self.bias_init, (self.features,), self.param_dtype)
        symm_kernel = self.param("symm_kernel", self.kernel_init, (self.features, self.n_sites), self.param_dtype)

        # take care of possibly different dtypes (e.g. x is float while parameters are complex)
        x, symm_kernel, hidden_bias = promote_dtype(x, symm_kernel, hidden_bias, dtype=None)

        # convert kernel to dense kernel of shape (features, n_symmetries, n_sites)
        symm_kernel = jnp.take(symm_kernel, self.symmetries.wrapped, axis=1)

        # x has shape (batch, n_sites), kernel has shape (features, n_symmetries, n_sites)
        # theta has shape (batch, features, n_symmetries)
        theta = jax.lax.dot_general(x, symm_kernel, (((1,), (2,)), ((), ())), precision=self.precision)
        theta += jnp.expand_dims(hidden_bias, 1)

        # for now, just stick with a single bias, irrespective of sublattice etc. (in contrast to Valenti et. al.)
        visible_bias = self.param("visible_bias", self.bias_init, (1,), self.param_dtype)
        bias = visible_bias * jnp.sum(x, axis=(1,))

        for i, correlator in enumerate(self.correlators):
            # initialize "visible" bias and kernel matrix for correlator
            correlator = correlator.wrapped  # convert hashable array to (usable) jax.Array
            corr_bias = self.param(f"corr{i}_bias", self.bias_init, (1,), self.param_dtype)
            corr_kernel = self.param(f"corr{i}_kernel", self.kernel_init, (self.features, len(correlator)),
                                     self.param_dtype)

            # convert kernel to dense kernel of shape (features, n_correlator_symmetries, n_corrs)
            corr_kernel = jnp.take(corr_kernel, self.correlator_symmetries[i].wrapped, axis=1)

            # correlator has shape (n_corrs, degree), e.g. n_corrs=L²=n_site/2 and degree=4 for plaquettes
            corr_values = jnp.take(x, correlator, axis=1).prod(axis=2)  # shape (batch, n_corrs)

            # corr_values has shape (batch, n_corrs)
            # kernel has shape (features, n_correlator_symmetries, n_corrs)
            # theta has shape (batch, features, n_symmetries)
            theta += jax.lax.dot_general(corr_values, corr_kernel, (((1,), (2,)), ((), ())), precision=self.precision)
            bias += corr_bias * jnp.sum(corr_values, axis=(1,))

        self.sow("intermediates", "activation_inputs", theta)
        theta = self.activation(theta)
        self.sow("intermediates", "activation_outputs", theta)
        theta = jnp.sum(theta, axis=(1, 2))  # sum over all symmetries and features = alpha * n_sites / n_symmetries
        theta += bias
        return theta


# %%
class ExplicitCorrelationRBM(nn.Module):
    """
    Difference to regular CorrelationRBM: Doesn't require the symmetries of the correlators features, they are
    explicitly constructed by applying symmetry operations to the raw configurations and recomputing the correlator
    features for each symmtry. This has strong impace on performance for large lattices but can be used as a
    sanity check.
    """
    # permutations of lattice sites corresponding to symmetries
    symmetries: HashableArray
    # correlators that serve as additional input for the cRBM
    correlators: Sequence[HashableArray]
    # The dtype of the weights
    param_dtype: Any = jnp.float64
    # The nonlinear activation function
    activation: Any = nknn.log_cosh
    # feature density. Number of features equal to alpha * input.shape[-1]
    alpha: Union[float, int] = 1
    # Numerical precision of the computation see :class:`jax.lax.Precision` for details
    precision: Any = None

    # Initializer for the Dense layer matrix
    kernel_init: Callable = default_kernel_init
    # Initializer for the biases
    bias_init: Callable = default_kernel_init

    def setup(self):
        self.n_symm, self.n_sites = jnp.asarray(self.symmetries).shape
        self.features = int(self.alpha * self.n_sites / self.n_symm)
        if self.alpha > 0 and self.features == 0:
            raise ValueError(
                f"RBMSymm: alpha={self.alpha} is too small "
                f"for {self.n_symm} permutations, alpha ≥ {self.n_symm / self.n_sites} is needed.")

    @nn.compact
    def __call__(self, x_in):
        # x_in shape (batch, n_sites)
        perm_x = jnp.take(x_in, self.symmetries.wrapped, axis=1)  # perm_x shape (batch, n_symmetries, n_sites)

        x = nn.Dense(name="Dense_SingleSpin",
                     features=self.features,
                     use_bias=True,
                     param_dtype=self.param_dtype,
                     precision=self.precision,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(perm_x)  # x shape (batch, n_symmetries, features/hidden_units)

        # for now, just stick with a single bias, irrespective of sublattice etc. (see Valenti et. al.)
        visible_bias = self.param("visible_bias", self.bias_init, (1,), self.param_dtype)
        bias = visible_bias * jnp.sum(perm_x, axis=(1, 2))

        correlator_biases = self.param("correlator_bias", self.bias_init, (len(self.correlators),), self.param_dtype)

        for i, correlator in enumerate(self.correlators):
            # before product, has shape (batch, n_symmetries, n_corrs, n_spins_in_corr)
            # where n_corrs corresponds to e.g. the number of plaquettes, bonds, loops etc. in one configuration
            corr_values = jnp.take(perm_x, correlator.wrapped, axis=2).prod(axis=3)
            x += nn.Dense(name=f"Dense_Correlator{i}",
                          features=self.features,
                          use_bias=False,
                          param_dtype=self.param_dtype,
                          precision=self.precision,
                          kernel_init=self.kernel_init,
                          bias_init=self.bias_init)(corr_values)

            bias += correlator_biases[i] * jnp.sum(corr_values, axis=(1, 2))

        x = self.activation(x)
        x = jnp.sum(x, axis=(1, 2))  # sum over all symmetries and features=alpha * n_sites / n_symmetries
        x += bias
        return x


# %%
class RBMModPhaseSymm(nn.Module):
    """
    Essentially the same as the RBMModPhase model in the netket library, however, it additionally utilizes equivariant
    layers, aka GCNNs, for translational symmetry.
    """
    # Array of permutation indices corresponding to symmetry opeartions incl identity.
    # in order to jit model.apply, attributes/data fiels must be hashable (model apply function is static arg in vqs)
    symmetries: HashableArray
    # The dtype of the weights
    param_dtype: Any = jnp.float64
    # The nonlinear activation function
    activation: Any = nknn.log_cosh
    # feature density. Number of features equal to alpha * input.shape[-1]
    alpha: Union[float, int] = 1
    # if True uses a bias in the dense layer (hidden layer bias)
    use_hidden_bias: bool = True
    # Numerical precision of the computation see :class:`jax.lax.Precision` for details
    precision: Any = None

    # Initializer for the Dense layer matrix
    kernel_init: Callable = default_kernel_init
    # Initializer for the hidden bias
    hidden_bias_init: Callable = default_kernel_init

    def setup(self):
        self.n_symm, self.n_sites = self.symmetries.shape
        self.features = int(self.alpha * self.n_sites / self.n_symm)
        if self.alpha > 0 and self.features == 0:
            raise ValueError(
                f"RBMSymm: alpha={self.alpha} is too small "
                f"for {self.n_symm} permutations, alpha ≥ {self.n_symm / self.n_sites} is needed.")

    @nn.compact
    def __call__(self, x_in):
        x = x_in
        # input for nknn.DenseSymm must be of shape (batch, in_features, n_sites)
        if x.ndim < 3:
            x = jnp.expand_dims(x, -2)

        re = nknn.DenseSymm(
            name="reDense",
            mode="matrix",
            symmetries=self.symmetries,
            features=self.features,
            param_dtype=self.param_dtype,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
            precision=self.precision)(x)

        re = self.activation(re)
        re = re.reshape(-1, self.features * self.n_symm)  # flatten output_feature dimension
        re = jnp.sum(re, axis=-1)

        im = nknn.DenseSymm(
            name="imDense",
            mode="matrix",
            symmetries=self.symmetries,
            features=self.features,
            param_dtype=self.param_dtype,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
            precision=self.precision)(x)

        im = self.activation(im)
        im = im.reshape(-1, self.features * self.n_symm)  # flatten output_feature dimension
        im = jnp.sum(im, axis=-1)

        return re + 1j * im
