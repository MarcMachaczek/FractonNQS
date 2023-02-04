import jax
from jax import numpy as jnp
from flax import linen as nn
from netket import nn as nknn
from netket.utils import HashableArray

from typing import Union, Any, Callable, Sequence

default_kernel_init = jax.nn.initializers.normal(stddev=0.01)  # nn.initializers.lecun_normal()


class SimpleMLP(nn.Module):
    features: Sequence[int]
    activation: Any = nn.relu
    alpa = Union[float, int] = 1
    param_dtype: Any = jnp.float64

    kernel_init: callable = default_kernel_init

    def setup(self):
        self.layers = [nn.Dense(feat,
                                kernel_init=self.default_kernel_init,
                                param_dtype=self.param_dtype)
                       for feat in self.features]

    def __call__(self, x_in):
        x = x_in
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x


class SymmetricNN(nn.module):
    # permutations of laatice sites corresponding to symmetries
    symmetries: HashableArray
    # The nonlinear activation function
    activation: Any = nn.relu
    # feature density. Number of features equal to alpha * input.shape[-1]
    alpha: Union[float, int] = 1
    # The dtype of the weights
    param_dtype: Any = jnp.float64
    # Numerical precision of the computation see :class:`jax.lax.Precision` for details
    precision: Any = None

    # Initializer for the Dense layer matrix
    kernel_init: Callable = default_kernel_init

    def __init__(self):
        pass
