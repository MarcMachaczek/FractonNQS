# A very simple script to test training functionality
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import jax
import jax.numpy as jnp
import optax
import netket as nk

import geneqs
from geneqs.utils.training import loop_gs, driver_gs

stddev = 0.05
default_kernel_init = jax.nn.initializers.normal(stddev)

# %%
L = 3  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L, L])
cube_graph = nk.graph.Hypercube(length=L, n_dim=3, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=cube_graph.n_nodes)

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(projection='3d')
ax.scatter3D(cube_graph.positions[:, 0], cube_graph.positions[:, 1], cube_graph.positions[:, 2], s=600, alpha=0.8)

for i, pos in enumerate(cube_graph.positions):
    delta = 0.03
    ax.text(pos[0]-delta, pos[1]-delta, pos[2]-delta, f"{i}", fontsize=14)

for edge in cube_graph.edges():
    start = cube_graph.positions[edge[0]]
    end = cube_graph.positions[edge[1]]
    ax.plot3D((start[0], end[0]), (start[1], end[1]), (start[2], end[2]), color="black", alpha=0.3)

# ax.invert_xaxis()

plt.show()

