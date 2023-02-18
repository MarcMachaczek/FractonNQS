# A very simple script to test training functionality
import jax
from matplotlib import pyplot as plt

import jax.numpy as jnp
import netket as nk

import geneqs
from geneqs.utils.training import loop_gs, driver_gs

default_kernel_init = jax.nn.initializers.normal(0.001)

param_seed = jax.random.PRNGKey(2345)
sampler_seed = jax.random.PRNGKey(1234)

# %%
L = 6  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# get (specific) symmetries of the model, in our case translations
perms = geneqs.utils.indexing.get_translations_cubical2d(shape)
link_perms = geneqs.utils.indexing.get_linkperms_cubical2d(perms)
# must be hashable to be included as flax.module attribute
link_perms = nk.utils.HashableArray(link_perms.astype(int))

# %%
# setting hyper-parameters
n_iter = 100
n_chains = 256 * 2  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = 256 * 8
n_discard_per_chain = 8  # should be small for using many chains, default is 10% of n_samples
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.01
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTOnTheFly, diag_shift=diag_shift)

# define model parameters
alpha = 2
RBMSymm = nk.models.RBMSymm(symmetries=link_perms, alpha=alpha, kernel_init=default_kernel_init)

learning_rate = 0.03

g = 0

# %%
ltoric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, g)
loptimizer = nk.optimizer.Sgd(learning_rate)
lsampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
lvqs = nk.vqs.MCState(lsampler, RBMSymm, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain,
                      seed=param_seed, sampler_seed=sampler_seed)

lvqs, ldata = loop_gs(lvqs, ltoric, loptimizer, preconditioner, n_iter)

# %%
dtoric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, g)
doptimizer = nk.optimizer.Sgd(learning_rate)
dsampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
dvqs = nk.vqs.MCState(dsampler, RBMSymm, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain,
                      seed=param_seed, sampler_seed=sampler_seed)

dvqs, ddata = driver_gs(dvqs, dtoric, doptimizer, preconditioner, n_iter)

