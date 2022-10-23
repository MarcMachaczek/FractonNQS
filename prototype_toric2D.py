# %%
import numpy as np
import scipy
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import flax

import netket as nk
import geneqs

import time

# %% Define graph and visualize it
L = 3  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)

fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# %% define the hilbert space and operators of interest
hi = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)

position = jnp.array([0, 0])
p0 = geneqs.operators.toric_2d.Plaq2d(hi, position, shape)
p_ids = geneqs.operators.toric_2d.position_to_plaq(position, shape)

p1 = nk.operator.spin.sigmaz(hi, p_ids[0].item()) * \
     nk.operator.spin.sigmaz(hi, p_ids[1].item()) * \
     nk.operator.spin.sigmaz(hi, p_ids[2].item()) * \
     nk.operator.spin.sigmaz(hi, p_ids[3].item())

vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), nk.models.RBM())
vs.n_samples = 2016
# %%
t0 = time.perf_counter()
print(vs.expect(p0))
print(time.perf_counter() - t0, "needed for p0")

t0 = time.perf_counter()
print(vs.expect(p1))
print(time.perf_counter() - t0, "needed for p1")

# %%
ha = nk.operator.LocalOperator(hi)
# construct the hamiltonian, first by adding the plaquette terms:
for i in range(L):
    for j in range(L):
        plaq_indices = geneqs.operators.toric_2d.position_to_plaq(jnp.array([i, j]), shape)
        ha -= nk.operator.spin.sigmaz(hi, plaq_indices[0].item()) * \
            nk.operator.spin.sigmaz(hi, plaq_indices[1].item()) * \
            nk.operator.spin.sigmaz(hi, plaq_indices[2].item()) * \
            nk.operator.spin.sigmaz(hi, plaq_indices[3].item())

# now add the star terms
for i in range(L):
    for j in range(L):
        star_indices = geneqs.operators.toric_2d.position_to_star(jnp.array([i, j]), shape)
        ha -= nk.operator.spin.sigmax(hi, star_indices[0].item()) * \
            nk.operator.spin.sigmax(hi, star_indices[1].item()) * \
            nk.operator.spin.sigmax(hi, star_indices[2].item()) * \
            nk.operator.spin.sigmax(hi, star_indices[3].item())

toric = geneqs.operators.toric_2d.ToricCode2d(hi, shape)
# %% for small system sizes, diagonalize hamiltonian directly to be able to check varitional result
sparse_ha = ha.to_sparse()
eig_vals, eig_vecs = scipy.sparse.linalg.eigs(sparse_ha, k=12, which="SR")

print("eigenvalues with scipy sparse:", eig_vals)
# as expected, the energy of the ground state is -18 and the GS is 4-fold degenerate
# first excited state has energy +4, as flipping one spin reverses the contribution of one plaq and star operator each


# %%
t0 = time.perf_counter()
print(vs.expect(toric))
print(time.perf_counter() - t0, "needed for toric")

t0 = time.perf_counter()
print(vs.expect(ha))
print(time.perf_counter() - t0, "needed for ha")

