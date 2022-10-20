# %%
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random
import flax

import netket as nk

from functools import partial


# %%
def edge_to_index(position: jax.Array, direction: int, size: jax.Array) -> jax.Array:
    dimension = size.shape[0]  # extract spatial dimension
    index = 0
    for d in range(dimension - 1):
        index += position[d] * jnp.product(size[d + 1:])
    index += position[dimension - 1]
    index = dimension * index + direction
    return index


def index_to_edge(index: int, size: jax.Array) -> Tuple[jax.Array, jax.Array]:
    dimension = size.shape[0]
    position = jnp.zeros_like(size)
    remainder = index
    for d in range(dimension - 1):
        position = position.at[d].add(remainder // (jnp.product(size[d + 1:]) * dimension))
        remainder = remainder % (jnp.product(size[d + 1:]) * dimension)
    position = position.at[dimension - 1].add(remainder // dimension)
    return position, (remainder % dimension)


def position_to_plaq(position: jax.Array, size: jax.Array) -> jax.Array:
    right = (position + jnp.array([1, 0])) % size[0]  # location to the right of position (PBC)
    top = (position + jnp.array([0, 1])) % size[1]  # location to the top of position (PBC)
    indices = jnp.stack([
        edge_to_index(position, 1, size),
        edge_to_index(top, 0, size),
        edge_to_index(right, 1, size),
        edge_to_index(position, 0, size),
    ])
    return indices


def position_to_star(position: jax.Array, size: jax.Array) -> jax.Array:
    # location to the left of position (PBC)
    left = position - jnp.array([1, 0]) if position[0] > 0 else jnp.array([size[0] - 1, position[1]])
    #  location to the bottom of position (PBC)
    bot = position - jnp.array([0, 1]) if position[1] > 0 else jnp.array([position[0], size[1] - 1])
    indices = jnp.stack([
        edge_to_index(position, 1, size),
        edge_to_index(position, 0, size),
        edge_to_index(bot, 1, size),
        edge_to_index(left, 0, size),
    ])
    return indices


# %%
L = 3  # size should be at least 3, else there are problems with pbc and indexing
size = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)

hi = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)
ha = nk.operator.LocalOperator(hi)

# %%
# construct the hamiltonian, first by adding the plaquette terms:
for i in range(L):
    for j in range(L):
        plaq_indices = position_to_plaq(jnp.array([i, j]), size)
        ha -= nk.operator.spin.sigmaz(hi, plaq_indices[0].item()) * \
            nk.operator.spin.sigmaz(hi, plaq_indices[1].item()) * \
            nk.operator.spin.sigmaz(hi, plaq_indices[2].item()) * \
            nk.operator.spin.sigmaz(hi, plaq_indices[3].item())

# now add the star terms
for i in range(L):
    for j in range(L):
        star_indices = position_to_star(jnp.array([i, j]), size)
        ha -= nk.operator.spin.sigmax(hi, star_indices[0].item()) * \
            nk.operator.spin.sigmax(hi, star_indices[1].item()) * \
            nk.operator.spin.sigmax(hi, star_indices[2].item()) * \
            nk.operator.spin.sigmax(hi, star_indices[3].item())


# %%
class Toric2D(nk.operator.DiscreteOperator):
    def __init__(self, hilbert: nk.hilbert.DiscreteHilbert):
        super().__init__(hilbert)

    @property
    def dtype(self):
        return float


def plaqz2d_conns_and_mels(sigma, position: jax.Array, size: jax.Array):
    L = size[0]
    # plaquette is diagonal in z basis, so eta is just the original
    eta = sigma * \
        sigma[edge_to_index(position, 0, size)] * \
        sigma[edge_to_index(position, 1, size)] * \
        sigma[edge_to_index((position + jnp.array([0, 1])) % L, 0, size)] * \
        sigma[edge_to_index((position + jnp.array([1, 0])) % L, 1, size)]


    eta = jnp.expand_dims(eta, axis=0)
    return eta, jnp.ones()


# %%
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# %%
size = jnp.array([3, 3])
idx = edge_to_index(jnp.array([1, 2]), 1, size)
print(idx)
pos, dir = index_to_edge(idx.item(), size=size)
