import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

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
    The indices are chosen sucht that position is in the lower left corner of the plaquette. See Toric Code model.
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
    The indices are chosen sucht that position is in the center of the star. See Toric Code model.
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


# %%
def cubical_translation(arr: np.ndarray, shape: ArrayLike, dimension: int, shift: int = 1) -> np.ndarray:
    """
    Applies a translation to the input array, permuting the features of arr accordingly.

    Args:
        arr: Array of shape (n_sites, features).
        shape: The shape of a cubical lattice with arbitrary dimension.
        dimension: Dimension along which to apply the shift/translation.
        shift: The "step size" for the translation

    Returns:
        Permuted version of arr, where the first axis of arr was permuted according to the specified translation.

    """
    n_dim = shape[dimension]
    perm = (np.arange(n_dim) - shift) % n_dim  # create index permutation along dim
    perm_array = np.copy(arr).reshape(*shape, *arr.shape[1:])  # unflatten spatial dimensions
    perm_array = np.take(perm_array, perm, axis=dimension)  # apply permutation along spatial dimension
    return perm_array.reshape(-1, *arr.shape[1:])  # flatten spatial dimensions again


# %%
def get_translations_cubical2d(shape: ArrayLike) -> np.ndarray:
    """
    Retrieve permutations according to all translations of a 2d cubical lattice with PBC.
    Args:
        shape: Shape of the 2d lattice in the form (x0_extend, x1_extend)

    Returns:
        Array with permutations of the lattice sites with dimensions (#permutations, n_sites)

    """
    base = np.arange(np.product(shape)).reshape(-1, 1)
    permutations = np.zeros(shape=(np.product(shape), np.product(shape)), dtype=int)
    p = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            dum = cubical_translation(base, shape, 0, i)  # apply x translation
            dum = cubical_translation(dum, shape, 1, j)  # apply y translation
            permutations[p] = dum.flatten()
            p += 1
    return permutations


# %%
def get_linkperms_cubical2d(permutations: np.ndarray) -> np.ndarray:
    # note: use netket graph stuff to get complete graph automorphisms, but there we have less control over symmetries
    # now get permutations on the link level
    n_spins = 2*permutations.shape[0]
    link_perms = np.zeros(shape=(permutations.shape[0], n_spins))
    for i, perm in enumerate(permutations):
        link_perm = [[p * 2, p * 2 + 1] for p in perm]
        link_perms[i] = np.asarray(link_perm, dtype=int).flatten()

    return link_perms
