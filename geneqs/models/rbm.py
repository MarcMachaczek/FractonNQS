import jax
from jax import numpy as jnp
from flax import linen as nn
from netket import nn as nknn
from netket.utils import HashableArray

from typing import Union, Any, Callable, Sequence

default_kernel_init = jax.nn.initializers.normal(stddev=0.01)


# %%
class CorrelationRBM(nn.Module):
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

        self.layers = [nn.Dense(feat,
                                use_bias=self.use_bias,
                                kernel_init=self.kernel_init,
                                bias_init=self.bias_init,
                                precision=self.precision,
                                param_dtype=self.param_dtype)
                       for feat in self.features[1:]]

    @nn.compact
    def __call__(self, x_in):
        perm_x = jnp.take(x_in, self.symmetries, axis=1)

        x = nn.Dense(features=self.features,
                     use_bias=self.use_bias,
                     param_dtype=self.param_dtype,
                     precision=self.precision,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(perm_x)

        for correlator in self.correlators:
            x +=

        return x


# %%
a = jnp.arange(12).reshape(4, 3)
perms = jnp.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
perms_a = jnp.take(a, perms, axis=1)


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
