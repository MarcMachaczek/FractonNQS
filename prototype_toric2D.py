# %%
import numpy as np
import scipy
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import flax

import netket as nk
import geneqs

# %% Define graph/lattice and hilbert space
L = 3  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# %% create variational state
vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), nk.models.RBM(alpha=2))

# own custom hamiltonian
toric = geneqs.operators.toric_2d.ToricCode2d(hi, shape)

# %%
vs.init_parameters()
vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), nk.models.RBM(alpha=2))
optimizer = nk.optimizer.Adam(learning_rate=0.001)

# Notice the use, again of Stochastic Reconfiguration, which considerably improves the optimisation
gs = nk.driver.VMC(toric, optimizer, variational_state=vs, preconditioner=nk.optimizer.SR(diag_shift=0.1))

log = nk.logging.RuntimeLog()
gs.run(n_iter=500, out=log)

ffn_energy = vs.expect(toric)
error = abs((ffn_energy.mean-2*square_graph.n_nodes)/(2*square_graph.n_nodes))
print("Optimized energy and relative error: ", ffn_energy, error)

# %%
data_FFN = log.data
plt.errorbar(data_FFN["Energy"].iters, data_FFN["Energy"].Mean, yerr=data_FFN["Energy"].Sigma, label="FFN")
