import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from typing import Tuple


# %%
def position_to_index(position: jax.Array, shape: jax.Array) -> jax.Array:
    """
    Indexes the position of a cubical lattice with periodic boundary conditions.
    Note: Only works correctly if extend in every direction (see shape) is at least three.
    Args:
        position: Specifies the point on the lattice. Array with entries [x_0, x_1, ...]
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend, ...]

    Returns:
        A single element array containing the integer index.

    """
    assert jnp.alltrue(position < shape), f"position {position} is out of bounds for lattice of shape {shape}"
    dimension = shape.shape[0]  # extract spatial dimension
    index = 0
    for d in range(dimension - 1):
        index += position[d] * jnp.product(shape[d + 1:])
    index += position[dimension - 1]
    return index


def edge_to_index(position: jax.Array, direction: int, shape: jax.Array) -> jax.Array:
    """
    Indexes the edge of a cubical lattice with periodic boundary conditions.
    An edge is described by the point it is starting from (position) and its direction.
    Note: Only works correctly if extend in every direction (see shape) is at least three.
    Args:
        position: Specifies the point on the lattice the edge originates from. Array with entries [x_0, x_1, ...]
        direction: Specifies the direction. Starts from zero = x_0 direction, then one = x_1 direction etc.
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend, ...]

    Returns:
        A single element array containing the integer index.

    """
    assert jnp.alltrue(position < shape), f"position {position} is out of bounds for lattice of shape {shape}"
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
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend, ...]

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


def cubical_translation(arr: np.ndarray, shape: ArrayLike, direction: int, shift: int) -> np.ndarray:
    """
    Applies a translation to the input array, permuting the features of arr accordingly.

    Args:
        arr: Array of shape (n_sites, features).
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend, ...]
        direction: Dimension along which to apply the shift/translation.
        shift: The "step size" for the translation

    Returns:
        Permuted version of arr, where the first axis of arr was permuted according to the specified translation.

    """
    n_dim = shape[direction]
    perm = (np.arange(n_dim) - shift) % n_dim  # create index permutation along dim
    perm_array = np.copy(arr).reshape(*shape, *arr.shape[1:])  # unflatten spatial dimensions
    perm_array = np.take(perm_array, perm, axis=direction)  # apply permutation along spatial dimension
    return perm_array.reshape(-1, *arr.shape[1:])  # flatten spatial dimensions again


def get_translations_cubical2d(shape: ArrayLike, shift: int) -> np.ndarray:
    """
    Retrieve permutations according to all translations of a 2d cubical lattice with PBC.
    Args:
        shape: Size of the 2d lattice. Array with entries [x_0 extend, x_1 extend]
        shift: Only include translations by n=shift steps

    Returns:
        Array with permutations of the lattice sites with dimensions (n_permutations, n_sites)

    """
    assert np.all(shape % shift == 0), f"Provided shape {shape} and shift {shift} are not compatible"
    base = np.arange(np.product(shape)).reshape(-1, 1)
    permutations = np.zeros(shape=(int(np.product(shape) / shift ** len(shape)), np.product(shape)), dtype=int)
    p = 0
    for i in range(0, shape[0], shift):
        for j in range(0, shape[1], shift):
            dum = cubical_translation(base, shape, 0, i)  # apply x translation
            dum = cubical_translation(dum, shape, 1, j)  # apply y translation
            permutations[p] = dum.flatten()  # flatten feature dimension
            p += 1
    return permutations


def get_translations_cubical3d(shape: ArrayLike, shift: int) -> np.ndarray:
    """
    Retrieve permutations according to all translations of a 3d cubical lattice with PBC.
    Args:
        shape: Size of the 3d lattice. Array with entries [x_0 extend, x_1 extend, x_2 extend]
        shift: Only include translations by n=shift steps

    Returns:
        Array with permutations of the lattice sites with dimensions (n_permutations, n_sites)

    """
    assert np.all(shape % shift == 0), f"Provided shape {shape} and shift {shift} are not compatible"
    base = np.arange(np.product(shape)).reshape(-1, 1)
    permutations = np.zeros(shape=(int(np.product(shape) / shift ** len(shape)), np.product(shape)), dtype=int)
    p = 0
    for i in range(0, shape[0], shift):
        for j in range(0, shape[1], shift):
            for k in range(0, shape[2], shift):
                dum = cubical_translation(base, shape, 0, i)  # apply x translation
                dum = cubical_translation(dum, shape, 1, j)  # apply y translation
                dum = cubical_translation(dum, shape, 2, k)  # apply z translation
                permutations[p] = dum.flatten()  # flatten feature dimension
                p += 1
    return permutations


# %% Utilities specifically for the checkerboard model
def position_to_cube(position: jax.Array, shape: jax.Array) -> jax.Array:
    """
    From a position on a cubical lattice with PBC, returns the indices of the 8 corners forming the cube.
    The indices are chosen sucht that the extent of the cube is positive in all directions.
    Args:
        position: Position of the cube. Array with entries [x_0 index, x_1 index, ...]
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend, ...]

    Returns:
        The four indices forming the plaquette.

    """
    # notation: b: bottom, t:top -> tbt means x_top=pos_x+1, y_bot=pos_y, z_top=pos_z+1. position is always bbb
    bbb = position
    bbt = position.at[2].set((position.at[2].get() + 1) % shape[2])
    btb = position.at[1].set((position.at[1].get() + 1) % shape[1])
    btt = btb.at[2].set((btb.at[2].get() + 1) % shape[2])
    tbb = position.at[0].set((position.at[0].get() + 1) % shape[0])
    tbt = tbb.at[2].set((tbb.at[2].get() + 1) % shape[2])
    ttb = tbb.at[1].set((tbb.at[1].get() + 1) % shape[1])
    ttt = ttb.at[2].set((ttb.at[2].get() + 1) % shape[2])

    indices = jnp.stack([position_to_index(bbb, shape),
                         position_to_index(btb, shape),
                         position_to_index(ttb, shape),
                         position_to_index(tbb, shape),
                         position_to_index(bbt, shape),
                         position_to_index(btt, shape),
                         position_to_index(ttt, shape),
                         position_to_index(tbt, shape)])
    return indices


def get_cubes_cubical3d(shape: jax.Array, shift: int) -> jax.Array:
    """
    Get the indices of all cubes formed by corners on the three-dimensional lattice (with PBC).
    Args:
        shift: Shift between cubes: 1 means every cube, 2 corresponds to a checkerboard, etc.
        shape: Size of the 3d lattice. Array with entries [x_0 extend, x_1 extend, x_2 extend]

    Returns:
        Array with all indices of shape (n_plaquettes, 8)

    """
    positions = jnp.asarray([(x, y, z)
                             for x in range(shape[0])
                             for y in range(shape[1])
                             for z in range(shape[2])
                             if (x + y + z) % shift == 0])
    return jnp.stack([position_to_cube(p, shape) for p in positions])


def get_cubeperms_cubical3d(shape: jax.Array, shift: int) -> jax.Array:
    """
        Retrieve the permutations of cube correlators induced by permutations on sites with shift on a 3d
        cubical lattice with PBC.

        Args:
            shape: Size of the 3d lattice. Array with entries [x_0 extend, x_1 extend, x_2 extend]
            shift: Only include translations by n=shift steps

        Returns:
            Array of shape (n_symmetries, n_cubes)

        """
    positions = jnp.asarray([(x, y, z)
                             for x in range(shape[0])
                             for y in range(shape[1])
                             for z in range(shape[2])
                             if (x + y + z) % shift == 0])

    position_idxs = [position_to_index(p, shape).item() for p in positions]  # indices where cubes are constructed

    # create permutation dictionary to convert from positions_idxs back to cube indices
    # this approach works because lattice translations are isomorphic to cube translations
    perm_dict = {}
    for i, idx in enumerate(position_idxs):
        perm_dict[idx] = i

    perms = jnp.take(get_translations_cubical3d(shape, shift), jnp.asarray(position_idxs), axis=1)

    cube_perms = [[perm_dict[idx.item()] for idx in perm] for perm in perms]

    return jnp.asarray(cube_perms)


def position_to_string_sites(position: jax.Array, direction: int, shape: jax.Array) -> jax.Array:
    """
    From a position and direction on a cubical lattice, returns the indices of the sites forming a string (with PBC)
    along the specified direction.
    Args:
        position: Position of the star. Array with entries [x_0 index, x_1 index, x_2_extend]
        direction: Specifies the direction. Starts from zero = x_0 direction, then one = x_1 direction etc.
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend, x_2_extend]

    Returns:
        The indices forming the string.

    """
    assert direction < len(position), f"direction not maching with dimension, should be smaller than {len(position)}"
    indices = []
    for i in range(shape[direction]):
        next_pos = position.at[direction].set((position.at[direction].get() + i) % shape[direction])
        indices.append(position_to_index(next_pos, shape))
    return jnp.array(indices)


def get_strings_cubical3d(direction: int, shape: jax.Array) -> jax.Array:
    """
    Get the indices of all strings formed by sites around the three-dimensional lattice (with PBC)
    into the specified direction.
    Args:
        direction: Specifies the direction. Starts from zero = x_0 direction, then one = x_1 direction etc.
        shape: Size of the 3d lattice. Array with entries [x_0 extend, x_1 extend, x_2 extend]

    Returns:
        Array with all indices of shape (n_strings, string_length)

    """
    string_indices = []
    position = jnp.zeros_like(shape)
    sub_shape = jnp.delete(shape, direction)
    sub_dims = jnp.delete(jnp.arange(3), direction)
    for i in range(sub_shape[0]):
        for j in range(sub_shape[1]):
            position = position.at[sub_dims[0]].set(i)
            position = position.at[sub_dims[1]].set(j)
            string_indices.append(position_to_string_sites(position, direction, shape))

    return jnp.stack(string_indices)


def get_xstring_perms3d(shape: jax.Array, shift: int = 2) -> jax.Array:
    """
    Get the permutations on x-strings / loops on the Checkerboard model induced by translational symmetries.
    Args:
        shape: Size of the 3d lattice. Array with entries [x_0 extend, x_1 extend, x_2 extend]
        shift: Only include translations by n=shift steps

    Returns:
        Array of shape (n_symmetries, n_xstrings)

    """
    # non-trivial permutations, corresponding to translations in y and z directions
    proper_perms = get_translations_cubical2d(shape=shape[1:], shift=shift)
    # stack them as many times as we have x translations, x_translation leave the correlators invariant
    xstring_perms = jnp.tile(proper_perms, (int(shape[0] / shift), 1))
    return xstring_perms


def get_ystring_perms3d(shape: jax.Array, shift: int = 2) -> jax.Array:
    """
    Get the permutations on y-strings / loops on the Checkerboard model induced by translational symmetries.
    Args:
        shape: Size of the 3d lattice. Array with entries [x_0 extend, x_1 extend, x_2 extend]
        shift: Only include translations by n=shift steps

    Returns:
        Array of shape (n_symmetries, n_xstrings)

    """
    xz_shape = shape[jnp.array([0, 2])]
    base = np.arange(np.product(xz_shape))
    x_perms = []
    for i in range(0, shape[0], shift):
        # append n=<translations in y direction> times to account for y translations that leave the strings invariant
        for _ in range(0, shape[1], shift):
            x_perms.append(cubical_translation(base.reshape(-1, 1), xz_shape, 0, i).flatten())
    xz_perms = []
    for x_perm in x_perms:
        for j in range(0, shape[2], shift):
            xz_perms.append(cubical_translation(x_perm.reshape(-1, 1), xz_shape, 1, j).flatten())
    ystring_perms = np.stack(xz_perms)
    return jnp.asarray(ystring_perms)


def get_zstring_perms3d(shape: jax.Array, shift: int = 2) -> jax.Array:
    """
    Get the permutations on z-strings / loops on the Checkerboard model induced by translational symmetries.
    Args:
        shape: Size of the 3d lattice. Array with entries [x_0 extend, x_1 extend, x_2 extend]
        shift: Only include translations by n=shift steps

    Returns:
        Array of shape (n_symmetries, n_xstrings)

    """
    # non-trivial permutations, corresponding to translations in x and y directions
    proper_perms = get_translations_cubical2d(shape=shape[:-1], shift=shift)
    # repeat them as many times as we have z translations, z_translation leave the correlators invariant
    zstring_perms = jnp.repeat(proper_perms, int(shape[2] / shift), axis=0)
    return zstring_perms


def get_bonds_cubical3d(shape: jax.Array) -> jax.Array:
    """
    Retrieve all neirest neighbor bonds on a 3d cubical lattice with PBC.
    Args:
        shape: Size of the 3d lattice. Array with entries [x_0 extend, x_1 extend, x_2 extend]

    Returns:
        Array of shape (n_bonds, 2)

    """
    positions = jnp.array([[x, y, z]
                           for x in range(shape[0])
                           for y in range(shape[1])
                           for z in range(shape[2])])
    bonds = []
    for p in positions:
        px = p.at[0].set((p.at[0].get() + 1) % shape[0])
        py = p.at[1].set((p.at[1].get() + 1) % shape[1])
        pz = p.at[2].set((p.at[2].get() + 1) % shape[2])
        bonds.append([position_to_index(p, shape), position_to_index(px, shape)])
        bonds.append([position_to_index(p, shape), position_to_index(py, shape)])
        bonds.append([position_to_index(p, shape), position_to_index(pz, shape)])
    return jnp.array(bonds)


def get_bondperms_cubical3d(shape: jax.Array, shift: int = 2) -> jax.Array:
    """
    Retrieve the permutations of neirest neighbor bonds induced by permutations on sites with shift on a 3d
    cubical lattice with PBC.

    Args:
        shape: Size of the 3d lattice. Array with entries [x_0 extend, x_1 extend, x_2 extend]
        shift: Only include translations by n=shift steps

    Returns:
        Array of shape (n_symmetries, n_bonds)

    """
    perms = get_translations_cubical3d(shape, shift=shift)
    bond_perms = []
    for perm in perms:
        bond_perm = jnp.array([[3 * p, 3 * p + 1, 3 * p + 2] for p in perm], dtype=jnp.int32).flatten()
        bond_perms.append(bond_perm)
    return jnp.vstack(bond_perms)


# %%  Utilities specifically for the 2 dimensional toric code
def position_to_plaq(position: jax.Array, shape: jax.Array) -> jax.Array:
    """
    From a position on a cubical lattice with PBC, returns the indices of the edges forming the plaquette operator.
    The indices are chosen sucht that position is in the lower left corner of the plaquette. See Toric Code model.
    Args:
        position: Position of the plaquette. Array with entries [x_0 index, x_1 index]
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend]

    Returns:
        The four indices forming the plaquette.

    """
    right = position.at[0].set((position.at[0].get() + 1) % shape[0])
    top = position.at[1].set((position.at[1].get() + 1) % shape[1])
    # right = (position + jnp.array([1, 0])) % shape[0]  # location to the right of position (PBC)
    # top = (position + jnp.array([0, 1])) % shape[1]  # location to the top of position (PBC)
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
        position: Position of the star. Array with entries [x_0 index, x_1 index]
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend]

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


def position_to_string_edges(position: jax.Array, direction: int, shape: jax.Array) -> jax.Array:
    """
    From a position and direction on a cubical lattice, returns the indices of the edges forming a string (with PBC)
    along the specified direction.
    Args:
        position: Position of the star. Array with entries [x_0 index, x_1 index]
        direction: Specifies the direction. Starts from zero = x_0 direction, then one = x_1 direction etc.
        shape: Size of the lattice. Array with entries [x_0 extend, x_1 extend]

    Returns:
        The indices forming the string.

    """
    assert direction < len(position), f"direction not maching with dimension, should be smaller than {len(position)}"
    indices = []
    for i in range(shape[direction]):
        next_pos = position.at[direction].set((position.at[direction].get() + i) % shape[direction])
        indices.append(edge_to_index(next_pos, direction, shape))
    return jnp.array(indices)


def get_strings_cubical2d(direction: int, shape: jax.Array) -> jax.Array:
    """
    Get the indices of all strings formed by edges around the two-dimensional lattice (with PBC)
    into the specified direction.
    Args:
        direction: Specifies the direction. Starts from zero = x_0 direction, then one = x_1 direction etc.
        shape: Size of the 2d lattice. Array with entries [x_0 extend, x_1 extend]

    Returns:
        Array with all indices of shape (n_strings, string_length)

    """
    string_indices = []
    position = jnp.zeros_like(shape)
    for _ in range(shape[(direction + 1) % 2]):
        string_indices.append(position_to_string_edges(position, direction, shape))
        position = position.at[(direction + 1) % 2].add(1)

    return jnp.stack(string_indices)


def get_plaquettes_cubical2d(shape: jax.Array) -> jax.Array:
    """
    Get the indices of all plaquettes formed by edges on the two-dimensional lattice (with PBC).
    Args:
        shape: Size of the 2d lattice. Array with entries [x_0 extend, x_1 extend]

    Returns:
        Array with all indices of shape (n_plaquettes, 4)

    """
    positions = jnp.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
    return jnp.stack([position_to_plaq(p, shape) for p in positions])


def get_stars_cubical2d(shape: jax.Array) -> jax.Array:
    """
    Get the indices of all stars formed by edges on the two-dimensional lattice (with PBC).
    Args:
        shape: Size of the 2d lattice. Array with entries [x_0 extend, x_1 extend]

    Returns:
        Array with all indices of shape (n_stars, 4)

    """
    positions = jnp.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
    return jnp.stack([position_to_star(p, shape) for p in positions])


def get_bonds_cubical2d(shape: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Get indices of all bonds on the two-dimensional lattice where qubits are placed on the edges (with PBC).
    Args:
        shape: Size of the 2d lattice. Array with entries [x_0 extend, x_1 extend]

    Returns:
        4 arrays containing the bond indices corresponding to the 4 distinct "bond-sublattices", aka the 4 subgroups
        of bonds connected by the symmetries, that is bottom-left, left-top, top-right, right-bottom within one
        plaquette structure (returned in this order).

    """
    bl_indices = []
    lt_indices = []
    tr_indices = []
    rb_indices = []
    positions = jnp.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
    for position in positions:
        right = (position + jnp.array([1, 0])) % shape[0]  # location to the right of position (PBC)
        top = (position + jnp.array([0, 1])) % shape[1]  # location to the top of position (PBC)

        bl_indices.append([edge_to_index(position, 0, shape), edge_to_index(position, 1, shape)])  # bot * left
        lt_indices.append([edge_to_index(position, 1, shape), edge_to_index(top, 0, shape)])  # left * top
        tr_indices.append([edge_to_index(top, 0, shape), edge_to_index(right, 1, shape)])  # top * right
        rb_indices.append([edge_to_index(right, 1, shape), edge_to_index(position, 0, shape)])  # right * bot

    return jnp.array(bl_indices), jnp.array(lt_indices), jnp.array(tr_indices), jnp.array(rb_indices)


def get_linkperms_cubical2d(shape: jax.Array, shift: int = 1) -> jax.Array:
    """
    Retrieve the permutations of links between sites induced by the permutations of the sites.
    Args:
        shape: Size of the 2d lattice. Array with entries [x_0 extend, x_1 extend]
        shift: Only include translations by n=shift steps

    Returns:
        Array with permutations of the links/edges with dimensions (n_permutations, n_edges)

    """
    # note: use netket graph stuff to get complete graph automorphisms, but there we have less control over symmetries
    # now get permutations on the link level
    permutations = get_translations_cubical2d(shape, shift)
    n_spins = 2 * jnp.product(shape)
    link_perms = jnp.zeros(shape=(permutations.shape[0], n_spins), dtype=jnp.int32)
    for i, perm in enumerate(permutations):
        link_perm = jnp.array([[p * 2, p * 2 + 1] for p in perm], dtype=jnp.int32).flatten()
        link_perms = link_perms.at[i, :].set(link_perm)
    return link_perms


def get_xstring_perms(shape: jax.Array) -> jax.Array:
    """
    Get the permutations on x-strings / loops on the Toric Code model induced by translational symmetries.
    Args:
        shape: Size of the 2d lattice. Array with entries [x_0 extend, x_1 extend]

    Returns:
        Array of shape (n_symmetries, n_xstrings)

    """
    base = jnp.arange(shape[1])  # identity permutation, number of x-strings is equal to extend in y dimension
    # proper (not identity) permutations, corresponding to translations in d=1 / y direction
    proper_perms = jnp.stack([(base - i) % shape[1] for i in range(shape[1])])
    # stack them as many times as we have x translations, x_translation leave the correlators invariant
    # recall that translation perms start with trailing dimension ([0,0], [0,1]... ,[0,y_dim-1], [1,0], [1,1]...)
    xstring_perms = jnp.tile(proper_perms, (shape[0], 1))
    return xstring_perms


def get_ystring_perms(shape: jax.Array) -> jax.Array:
    """
        Get the permutations on y-strings / loops on the Toric Code model induced by translational symmetries.
        Args:
            shape: Size of the 2d lattice. Array with entries [x_0 extend, x_1 extend]

        Returns:
            Array of shape (n_symmetries, n_ystrings)

        """
    base = jnp.arange(shape[0])  # identity permutation, number of y-strings is equal to extend in x dimension
    # proper (not identity) permutations, corresponding to translations in d=0 / x direction
    proper_perms = jnp.stack([(base - i) % shape[0] for i in range(shape[0])])
    # only changes every y_dim translations because only x-translations permute the correlators
    ystring_perms = jnp.repeat(proper_perms, shape[1], axis=0)
    return ystring_perms


def get_bondperms_cubical2d(shape: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Retrieve the permutations of nearest neighbor bonds induced by the permutations of the sites.
    Args:
        shape: Size of the 2d lattice. Array with entries [x_0 extend, x_1 extend]

    Returns:
        4 Arrays with permutations of the bl, lt, tr, rb bonds with dimensions (n_permutations, n_bonds)
        Each of the 4 bond sublattices actually permutes just according to the translations so this function is
        trivial, however, it serves to create a uniform UI.

    """
    perms = jnp.asarray(get_translations_cubical2d(shape, shift=1))
    return perms, perms, perms, perms
