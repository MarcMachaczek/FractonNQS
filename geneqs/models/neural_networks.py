import jax
from jax import numpy as jnp
from flax import linen as nn
from netket import nn as nknn
from netket.utils import HashableArray

from typing import Optional, Any, Callable, Sequence, Tuple

lecun = nn.initializers.lecun_normal(in_axis=1, out_axis=0)
zeros = jax.nn.initializers.zeros


class SimpleNN(nn.Module):
    """
    A simple FFNN architecture.
    """
    features: Sequence[int]
    # The nonlinear activation function
    activation: Any = nknn.gelu
    # bias for the all layers
    use_bias: bool = True
    # The dtype of the weights
    param_dtype: Any = jnp.float64
    # Numerical precision of the computation see :class:`jax.lax.Precision` for details
    precision: Any = None

    # Initializer for the Dense layer matrix
    kernel_init: callable = lecun
    bias_init: Callable = zeros

    def setup(self):
        self.layers = [nn.Dense(features=feat,
                                use_bias=self.use_bias,
                                kernel_init=self.kernel_init,
                                bias_init=self.bias_init,
                                param_dtype=self.param_dtype,
                                precision=self.precision)
                       for feat in self.features]

    def __call__(self, x_in):
        x = x_in
        # jax.debug.print("{0}", self.layers)
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)  # remove last axis
        return x


class SymmetricNN(nn.Module):
    """
    A symmetric FFNN. The first layer constructs invariant features similar to a GCNN.
    """
    # permutations of lattice sites corresponding to symmetries
    symmetries: HashableArray
    # features for each layer in the NN
    features: Tuple[int, ...]
    # kernel mask (aka filter size for CNN)
    mask: Optional[HashableArray] = None
    # The nonlinear activation function
    activation: Any = jax.nn.gelu
    # bias for the all layers
    use_bias: bool = True
    # The dtype of the weights
    param_dtype: Any = jnp.float64
    # Numerical precision of the computation see :class:`jax.lax.Precision` for details
    precision: Any = None

    # Initializer for the Dense layer matrix
    kernel_init: Callable = lecun
    bias_init: Callable = zeros

    def setup(self):
        self.n_symm, self.n_sites = self.symmetries.shape
        self.symm_layer = nknn.DenseSymm(name="Symm",
                                         mode="matrix",
                                         symmetries=self.symmetries,
                                         features=self.features[0],
                                         use_bias=self.use_bias,
                                         mask=self.mask,
                                         kernel_init=self.kernel_init,
                                         bias_init=self.bias_init,
                                         param_dtype=self.param_dtype,
                                         precision=self.precision)

        self.layers = [nn.Dense(features=feat,
                                use_bias=self.use_bias,
                                kernel_init=self.kernel_init,
                                bias_init=self.bias_init,
                                precision=self.precision,
                                param_dtype=self.param_dtype)
                       for feat in self.features[1:]]

    @nn.compact
    def __call__(self, x_in):
        x = x_in
        # input for nknn.DenseSymm must be of shape (batch, in_features, n_sites)
        if x.ndim < 3:
            x = jnp.expand_dims(x, -2)

        x = self.symm_layer(x)
        x = self.activation(x)
        x = jnp.sum(x, axis=-1)  # sum over symmetries to make features invariant
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)  # final sum layer
        return x
