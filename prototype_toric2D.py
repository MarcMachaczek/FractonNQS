# %%
import numpy as np
from matplotlib import pyplot as plt

import jax.numpy as jnp
import netket as nk

import geneqs
from global_variables import RESULTS_PATH

# %% setting hyper-parameters
n_iter = 300
n_chains = 1024  # total number of MCMC chains, should be large when runnning on GPU  ~O(1000)
n_samples = n_chains*20  # each chain will generate 32 samples
n_discard_per_chain = 24  # should be small for when using many chains, default is 10% of n_samples (too big)
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

alpha = 2
lr_multiplier = 1
lr_rbm = 0.1 * lr_multiplier
lr_rbm_symm = 0.09 * lr_multiplier
lr_rbm_mp = 0.1 * lr_multiplier
lr_rbm_mp_symm = 0.09 * lr_multiplier
preconditioner = nk.optimizer.SR(diag_shift=0.008)


# %% Define graph/lattice and hilbert space
L = 4  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)

# own custom hamiltonian
toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape)

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
# plt.show()

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
# sampler = nk.sampler.ExactSampler(hilbert)
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains)
RBM = nk.models.RBM(alpha=alpha)
sgd = nk.optimizer.Sgd(learning_rate=lr_rbm)
vqs = nk.vqs.MCState(sampler, RBM, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

gs = nk.driver.VMC(toric, sgd, variational_state=vqs, preconditioner=preconditioner)

log = nk.logging.RuntimeLog()
gs.run(n_iter=n_iter, out=log)
data_rbm = log.data

# %% train symmetric RBM
# sampler = nk.sampler.ExactSampler(hilbert)
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains)
RBM_symm = nk.models.RBMSymm(symmetries=link_perms, alpha=alpha)
sgd_symm = nk.optimizer.Sgd(learning_rate=0.09)  # lr_rbm_symm
# n_samples is divided by n_chains for the length of any MCMC chain
vqs_symm = nk.vqs.MCState(sampler, RBM_symm, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

vqs_symm.init_parameters()
gs_symm = nk.driver.VMC(toric, sgd_symm, variational_state=vqs_symm, preconditioner=preconditioner)

log = nk.logging.RuntimeLog()
gs_symm.run(n_iter=n_iter, out=log)
data_symm = log.data

# %% train complex RBM
# sampler = nk.sampler.ExactSampler(hilbert)
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains)
RBM_mp = nk.models.RBMModPhase(alpha=alpha)
sgd_mp = nk.optimizer.Sgd(learning_rate=lr_rbm_mp)
vqs_mp = nk.vqs.MCState(sampler, RBM_mp, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

vqs_mp.init_parameters()
gs_mp = nk.driver.VMC(toric, sgd_mp, variational_state=vqs_mp, preconditioner=preconditioner)

log = nk.logging.RuntimeLog()
gs_mp.run(n_iter=n_iter, out=log)
data_mp = log.data

# %% train symmetric complex RBM
# sampler = nk.sampler.ExactSampler(hilbert)
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains)
RBM_symm_mp = geneqs.models.RBMModPhaseSymm(symmetries=link_perms, alpha=alpha)
sgd_symm_mp = nk.optimizer.Sgd(learning_rate=lr_rbm_mp_symm)
vqs_symm_mp = nk.vqs.MCState(sampler, RBM_symm_mp, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

vqs_symm_mp.init_parameters()
gs_symm_mp = nk.driver.VMC(toric, sgd_symm_mp, variational_state=vqs_symm_mp, preconditioner=preconditioner)

log = nk.logging.RuntimeLog()
gs_symm_mp.run(n_iter=n_iter, out=log)
data_symm_mp = log.data

# %%
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

plot.errorbar(data_rbm["Energy"].iters, data_rbm["Energy"].Mean, yerr=data_rbm["Energy"].Sigma,
              label=f"RBM, lr={lr_rbm}")
plot.errorbar(data_symm["Energy"].iters, data_symm["Energy"].Mean, yerr=data_symm["Energy"].Sigma,
              label=f"RBMSymm, lr={lr_rbm_symm}")
plot.errorbar(data_mp["Energy"].iters, data_mp["Energy"].Mean, yerr=data_mp["Energy"].Sigma,
              label=f"RBMModPhase, lr={lr_rbm_mp}")
plot.errorbar(data_symm_mp["Energy"].iters, data_symm_mp["Energy"].Mean, yerr=data_symm_mp["Energy"].Sigma,
              label=f"RBMModPhaseSymm, lr={lr_rbm_mp_symm}")

fig.suptitle(f"ToricCode2d Exact: size={shape},"
             f" n_chains={n_chains},"
             f" n_samples={n_samples},"
             f" n_sweeps={L**2*2},"
             f" single spin flip updates,"
             f" alpha={alpha}")

plot.set_xlabel("iterations")
plot.set_ylabel("energy")

plot.set_title(f"using stochastic gradient descent with stochastic reconfiguration, diag_shift=0.01")
plot.legend()

plt.show()
# fig.savefig(f"{RESULTS_PATH}/toric2d/Exact_lattice{shape}_2.pdf")
