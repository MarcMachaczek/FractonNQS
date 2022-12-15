# %%
import numpy as np
from matplotlib import pyplot as plt

import jax.numpy as jnp
import netket as nk

import geneqs

# %% Define graph/lattice and hilbert space
L = 3  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)

# own custom hamiltonian
toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape)

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# %% get (specific) symmetries of the model, in our case translations
permutations = geneqs.utils.indexing.get_translations_cubical2d(shape)

# note: use netket graph stuff to get complete graph automorphisms, but there we have less control over symmetries
# now get permutations on the link level
link_perms = np.zeros(shape=(np.product(shape), hilbert.size))
for i, perm in enumerate(permutations):
    link_perm = [[p * 2, p * 2 + 1] for p in perm]
    link_perms[i] = np.asarray(link_perm, dtype=int).flatten()

# must be hashable to be included as flax.module attribute
link_perms = nk.utils.HashableArray(link_perms.astype(int))

# start with exact sampling to avoid MCMC issues
#%% train regular RBM
sampler = nk.sampler.ExactSampler(hilbert)
RBM = nk.models.RBM(alpha=2)
sgd = nk.optimizer.Sgd(learning_rate=0.1)
vqs = nk.vqs.MCState(sampler, RBM, n_samples=1024)

gs = nk.driver.VMC(toric, sgd, variational_state=vqs, preconditioner=nk.optimizer.SR(diag_shift=0.01))

log = nk.logging.RuntimeLog()
gs.run(n_iter=300, out=log)
data_rbm = log.data

# %% train symmetric RBM
sampler = nk.sampler.ExactSampler(hilbert)
RBM_symm = nk.models.RBMSymm(symmetries=link_perms, alpha=2)
sgd_symm = nk.optimizer.Sgd(learning_rate=0.05)
vqs_symm = nk.vqs.MCState(sampler, RBM_symm, n_samples=1024)

vqs_symm.init_parameters()
gs_symm = nk.driver.VMC(toric, sgd_symm, variational_state=vqs_symm, preconditioner=nk.optimizer.SR(diag_shift=0.01))

log = nk.logging.RuntimeLog()
gs_symm.run(n_iter=300, out=log)
data_symm = log.data

# %% train complex RBM
sampler = nk.sampler.ExactSampler(hilbert)
RBM_mp = nk.models.RBMModPhase(alpha=2)
sgd_mp = nk.optimizer.Sgd(learning_rate=0.1)
vqs_mp = nk.vqs.MCState(sampler, RBM_mp, n_samples=1024)

vqs_mp.init_parameters()
gs_mp = nk.driver.VMC(toric, sgd_mp, variational_state=vqs_mp, preconditioner=nk.optimizer.SR(diag_shift=0.01))

log = nk.logging.RuntimeLog()
gs_mp.run(n_iter=300, out=log)
data_mp = log.data

# %% train symmetric complex RBM
sampler = nk.sampler.ExactSampler(hilbert)
RBM_symm_mp = geneqs.models.RBMModPhaseSymm(symmetries=link_perms, alpha=2)
sgd_symm_mp = nk.optimizer.Sgd(learning_rate=0.08)
vqs_symm_mp = nk.vqs.MCState(sampler, RBM_symm_mp, n_samples=1024*4)

vqs_symm_mp.init_parameters()
gs_symm_mp = nk.driver.VMC(toric, sgd_symm_mp, variational_state=vqs_symm_mp, preconditioner=nk.optimizer.SR(diag_shift=0.01))

log = nk.logging.RuntimeLog()
gs_symm_mp.run(n_iter=300, out=log)
data_symm_mp = log.data

# %%
ffn_energy = vs.expect(toric)
error = abs((ffn_energy.mean+2*square_graph.n_nodes)/(2*square_graph.n_nodes))
print("Optimized energy and relative error: ", ffn_energy, error)

data_FFN = log.data
plt.errorbar(data_FFN["Energy"].iters, data_FFN["Energy"].Mean, yerr=data_FFN["Energy"].Sigma, label="FFN")
plt.show()
