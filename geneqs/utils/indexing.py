import jax
import jax.numpy as jnp

from typing import Tuple


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
