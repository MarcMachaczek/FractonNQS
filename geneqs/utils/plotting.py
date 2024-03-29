import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import netket as nk
import geneqs
from typing import Optional


# %%
def plot_checkerboard(ax: plt.axis, L: int, sigma: Optional[jax.Array] = None):
    """
    Plots the checkerboard model of extend L (shape=[L, L, L]) on a provided axis.
    Args:
        ax: Pyplot axis
        L: Extend of the system.
        sigma: If sigma is provided, the colors are adapted

    Returns:

    """
    shape = jnp.array([L, L, L])
    cube_graph_obc = nk.graph.Hypercube(length=shape[0].item() + 1, n_dim=3, pbc=False)
    cube_graph_pbc = nk.graph.Hypercube(length=shape[0].item(), n_dim=3, pbc=True)

    if sigma is None:
        colors = "royalblue"
    else:
        pos_idxs = np.array([geneqs.utils.indexing.position_to_index(p, shape) for p in cube_graph_pbc.positions],
                            dtype=int)
        color_code = sigma[pos_idxs]
        colors = []
        for c in color_code:
            if c == -1:
                color = "royalblue"
            elif c == 1:
                color = "goldenrod"
            else:
                color = "grey"
            colors.append(color)

    ax.scatter3D(cube_graph_pbc.positions[:, 0], cube_graph_pbc.positions[:, 1], cube_graph_pbc.positions[:, 2],
                 s=600 / L ** (1 / 3), alpha=0.7, c=colors)

    for i, pos in enumerate(cube_graph_pbc.positions):
        delta = 0.03
        ax.text(pos[0] - delta, pos[1] - delta, pos[2] - delta, f"{i}", fontsize=14 / L ** (1 / 3))

    positions = jnp.asarray([(x, y, z)
                             for x in range(shape[0]-1)
                             for y in range(shape[1]-1)
                             for z in range(shape[2]-1)
                             if (x + y + z) % 2 == 0])

    cubes = jnp.stack([geneqs.utils.indexing.position_to_cube(p, shape + jnp.array([1, 1, 1])) for p in positions])

    for edge in cube_graph_obc.edges():
        is_in_cube = False
        for cube in np.asarray(cubes):
            if set(edge).issubset(set(cube)):
                is_in_cube = True
        if is_in_cube is False:
            continue
        start = cube_graph_obc.positions[edge[0]]
        end = cube_graph_obc.positions[edge[1]]
        ax.plot3D((start[0], end[0]), (start[1], end[1]), (start[2], end[2]), color="black", alpha=0.5)
