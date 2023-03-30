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

stddev = 0.01
default_kernel_init = jax.nn.initializers.normal(stddev)

# %%
L = 4  # this means L+1 without PBC
shape = jnp.array([L, L, L])
cube_graph = nk.graph.Hypercube(length=L, n_dim=3, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)

h = (1.0, 0.0, 0.0)
checkerboard_nk = geneqs.operators.checkerboard.get_netket_checkerboard(hilbert, shape, h)
checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)

# %%
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(projection='3d')

geneqs.utils.plotting.plot_checkerboard(ax, 2)

plt.show()

# %%
sigma = jnp.asarray([jnp.ones(cube_graph.n_nodes)])
conns, mels = checkerboard_nk.get_conn_padded(sigma)
