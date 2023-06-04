# A very simple script to test training functionality
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import optax
import netket as nk
from netket.utils import HashableArray

import geneqs
from geneqs.utils.training import loop_gs, driver_gs

stddev = 0.01
default_kernel_init = jax.nn.initializers.normal(stddev)

# %%
L = 3  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# get (specific) symmetries of the model, in our case translations
perms = geneqs.utils.indexing.get_translations_cubical2d(shape, shift=1)
# noinspection PyArgumentList
link_perms = HashableArray(geneqs.utils.indexing.get_linkperms_cubical2d(shape, shift=1))

bl_bonds, lt_bonds, tr_bonds, rb_bonds = geneqs.utils.indexing.get_bonds_cubical2d(shape)
bl_perms, lt_perms, tr_perms, rb_perms = geneqs.utils.indexing.get_bondperms_cubical2d(shape)
# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_plaquettes_cubical2d(shape)),  # plaquette correlators
               HashableArray(bl_bonds), HashableArray(lt_bonds), HashableArray(tr_bonds), HashableArray(rb_bonds))

# noinspection PyArgumentList
correlator_symmetries = (HashableArray(jnp.asarray(perms)),  # plaquettes permute like sites
                         HashableArray(bl_perms), HashableArray(lt_perms),
                         HashableArray(tr_perms), HashableArray(rb_perms))
# noinspection PyArgumentList
loops = (HashableArray(geneqs.utils.indexing.get_strings_cubical2d(0, shape)),  # x-string correlators
         HashableArray(geneqs.utils.indexing.get_strings_cubical2d(1, shape)))  # y-string correlators
# noinspection PyArgumentList
loop_symmetries = (HashableArray(geneqs.utils.indexing.get_xstring_perms(shape)),
                   HashableArray(geneqs.utils.indexing.get_ystring_perms(shape)))

# %%
# setting hyper-parameters
n_iter = 120
n_chains = 512  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 4
n_discard_per_chain = 12  # should be small for using many chains, default is 10% of n_samples
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.01
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense, diag_shift=diag_shift)

# define model parameters
alpha = 1
RBMSymm = nk.models.RBMSymm(symmetries=link_perms, alpha=alpha,
                            kernel_init=default_kernel_init,
                            hidden_bias_init=default_kernel_init,
                            visible_bias_init=default_kernel_init,
                            param_dtype=float)

cRBM = geneqs.models.ToricLoopCRBM(symmetries=link_perms,
                                   correlators=correlators,
                                   correlator_symmetries=correlator_symmetries,
                                   loops=loops,
                                   loop_symmetries=loop_symmetries,
                                   alpha=alpha,
                                   kernel_init=default_kernel_init,
                                   bias_init=default_kernel_init,
                                   param_dtype=complex)

mask = jnp.vstack([geneqs.utils.indexing.position_to_plaq(jnp.asarray([0, 0]), shape), ])

mask = nk.utils.HashableArray(mask)
features = (4, 4, 4)
SymmNN = geneqs.models.neural_networks.SymmetricNN(link_perms, features)

lr_init = 0.03
lr_end = 0.03
transition_begin = 50
transition_steps = n_iter - transition_begin - 20
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

h = (0., 0., 0.)  # (hx, hy, hz)

# %%
toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
netket_toric = geneqs.operators.toric_2d.get_netket_toric2dh(hilbert, shape, h)
optimizer = optax.sgd(lr_schedule)
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
vqs = nk.vqs.MCState(sampler, cRBM, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

sigma1 = jnp.ones(hilbert.size)
sigma2 = sigma1.at[0].set(-1)
sigma = jnp.stack((sigma1, sigma2), axis=0)

# print("test begins here")
# y = cRBM.apply({"params": vqs.parameters}, sigma)

vqs, data = loop_gs(vqs, toric, optimizer, preconditioner, n_iter)

