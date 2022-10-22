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
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# %%
size = jnp.array([3, 3])
idx = edge_to_index(jnp.array([1, 2]), 1, size)
print(idx)
pos, dir = index_to_edge(idx.item(), size=size)
