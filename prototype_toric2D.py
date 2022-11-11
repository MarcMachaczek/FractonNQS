# %%
import numpy as np
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
# own custom hamiltonian
toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape)

ha_netketlocal = nk.operator.LocalOperator(hilbert)
# adding the plaquette terms:
for i in range(L):
    for j in range(L):
        plaq_indices = geneqs.utils.indexing.position_to_plaq(jnp.array([i, j]), shape)
        ha_netketlocal -= nk.operator.spin.sigmaz(hilbert, plaq_indices[0].item()) * \
                          nk.operator.spin.sigmaz(hilbert, plaq_indices[1].item()) * \
                          nk.operator.spin.sigmaz(hilbert, plaq_indices[2].item()) * \
                          nk.operator.spin.sigmaz(hilbert, plaq_indices[3].item())
# adding the star terms
for i in range(L):
    for j in range(L):
        star_indices = geneqs.utils.indexing.position_to_star(jnp.array([i, j]), shape)
        ha_netketlocal -= nk.operator.spin.sigmax(hilbert, star_indices[0].item()) * \
                          nk.operator.spin.sigmax(hilbert, star_indices[1].item()) * \
                          nk.operator.spin.sigmax(hilbert, star_indices[2].item()) * \
                          nk.operator.spin.sigmax(hilbert, star_indices[3].item())

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# %%
model = nk.models.RBMModPhase(alpha=2)
sampler = nk.sampler.MetropolisLocal(hilbert)  # nk.sampler.MetropolisLocal(hilbert)
optimizer = nk.optimizer.Sgd(learning_rate=0.15)
vs = nk.vqs.MCState(sampler, model)  # nk.sampler.MetropolisLocal(hilbert)
hi = toric # ha_netketlocal

vs.init_parameters()
# Notice the use of Stochastic Reconfiguration, which considerably improves the optimisation
gs = nk.driver.VMC(hi, optimizer, variational_state=vs, preconditioner=nk.optimizer.SR(diag_shift=0.01))

log = nk.logging.RuntimeLog()
gs.run(n_iter=300, out=log)

ffn_energy = vs.expect(hi)
error = abs((ffn_energy.mean-2*square_graph.n_nodes)/(2*square_graph.n_nodes))
print("Optimized energy and relative error: ", ffn_energy, error)

data_FFN = log.data
plt.errorbar(data_FFN["Energy"].iters, data_FFN["Energy"].Mean, yerr=data_FFN["Energy"].Sigma, label="FFN")
plt.show()
