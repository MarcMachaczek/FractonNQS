import jax
import jax.numpy as jnp

from typing import Tuple


# %%
def edge_to_index(position: jax.Array, direction: int, shape: jax.Array) -> jax.Array:
    """
    Indexes the edge of a cubical lattice with periodic boundary conditions.
    An edge is described by the point it is starting from (position) and its direction.
    Note: Only works correctly if extend in every direction (see size) is at least three.
    Args:
        position: Specifies the point on the lattice the edge originates from. Array with entries [x_0, x_1, ...]
        direction: Specifies the direction. Starts from zero = x_0 direction, then one = x_1 direction etc.
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend, ...]

    Returns:
        A single element array containing the integer index.

    """
    dimension = shape.shape[0]  # extract spatial dimension
    index = 0
    for d in range(dimension - 1):
        index += position[d] * jnp.product(shape[d + 1:])
    index += position[dimension - 1]
    index = dimension * index + direction
    return index


def index_to_edge(index: int, shape: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Converts the index of an edge back to the position, direction representation.
    Reverses the maaping of edge_to_index.
    Args:
        index: Index of the edge.
        shape: Size of the lattice: [x_0 extend, x_1 extend, ...]

    Returns:
        Position and direction of the edge with index.

    """
    dimension = shape.shape[0]
    position = jnp.zeros_like(shape)
    remainder = index
    for d in range(dimension - 1):
        position = position.at[d].add(remainder // (jnp.product(shape[d + 1:]) * dimension))
        remainder = remainder % (jnp.product(shape[d + 1:]) * dimension)
    position = position.at[dimension - 1].add(remainder // dimension)
    return position, (remainder % dimension)


# %% Utilities specifically for the 2 dimensional toric code
def position_to_plaq(position: jax.Array, shape: jax.Array) -> jax.Array:
    """
    From a position on a cubical lattice with PBC, returns the indices of the edges forming the plaquette operator.
    The indices are chosen sucht that position is in the lower left corner of the plaquette.
    Args:
        position: Position of the plaquette. Array with entries [x_0 index, x_1 index, ...]
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend, ...]

    Returns:
        The four indices forming the plaquette.

    """
    right = (position + jnp.array([1, 0])) % shape[0]  # location to the right of position (PBC)
    top = (position + jnp.array([0, 1])) % shape[1]  # location to the top of position (PBC)
    indices = jnp.stack([
        edge_to_index(position, 1, shape),
        edge_to_index(top, 0, shape),
        edge_to_index(right, 1, shape),
        edge_to_index(position, 0, shape)])
    return indices


def position_to_star(position: jax.Array, shape: jax.Array) -> jax.Array:
    """
    From a position on a cubical lattice with PBC, returns the indices of the edges forming the star operator.
    The indices are chosen sucht that position is in the center of the star.
    Args:
        position: Position of the star. Array with entries [x_0 index, x_1 index, ...]
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend, ...]

    Returns:
        The four indices forming the star.

    """
    # location to the left of position (PBC)
    left = position - jnp.array([1, 0]) if position[0] > 0 else jnp.array([shape[0] - 1, position[1]])
    #  location to the bottom of position (PBC)
    bot = position - jnp.array([0, 1]) if position[1] > 0 else jnp.array([position[0], shape[1] - 1])
    indices = jnp.stack([
        edge_to_index(position, 1, shape),
        edge_to_index(position, 0, shape),
        edge_to_index(bot, 1, shape),
        edge_to_index(left, 0, shape)])
    return indices
