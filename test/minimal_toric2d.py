# A very simple script to test training functionality
import numpy as np
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import optax
import netket as nk

import geneqs
from geneqs.utils.training import loop_gs

stddev = 0.1
default_kernel_init = jax.nn.initializers.normal(stddev)

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
n_iter = 120
n_chains = 256 * 2  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 4
n_discard_per_chain = 8  # should be small for using many chains, default is 10% of n_samples
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.01
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTOnTheFly, diag_shift=diag_shift)

# define model parameters
alpha = 2
RBMSymm = nk.models.RBMSymm(symmetries=link_perms, alpha=alpha,
                            kernel_init=default_kernel_init,
                            hidden_bias_init=default_kernel_init,
                            visible_bias_init=default_kernel_init)

lr_init = 0.02
lr_end = 0.008
transition_begin = 50
transition_steps = n_iter - transition_begin - 20
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

h = 0

# %%
toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
optimizer = optax.sgd(lr_schedule)
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
vqs = nk.vqs.MCState(sampler, RBMSymm, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

vqs, data = loop_gs(vqs, toric, optimizer, preconditioner, n_iter)