import numpy as np
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import optax
import netket as nk

import geneqs
from geneqs.utils.training import loop_gs

from tqdm import tqdm
from global_variables import BENCHMARK_PATH

RUN_PATH = f"{BENCHMARK_PATH}/toric_code/"
SAVE = True

stddev = 0.01
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
n_samples = n_chains * 4 * 4  # TODO: remove *4
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

h = (0, 0, 0)

# %%
runs = 100
energies = []
errors = []
for i in tqdm(range(runs)):
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
    vqs = nk.vqs.MCState(sampler, RBMSymm, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    vqs, data = loop_gs(vqs, toric, optimizer, preconditioner, n_iter)

    if len(data["energy"].Mean) == n_iter:  # in case training was interrupted by nans or infs
        energies.append(data["energy"].Mean)
        errors.append(data["energy"].Sigma)

# %%
energies = np.asarray(energies)
errors = np.asarray(errors)

if SAVE:
    np.save(f"{RUN_PATH}/data/L{shape}_h{h}_sd{stddev}_S{n_samples}_energies_2", energies)
    np.save(f"{RUN_PATH}//data/L{shape}_h{h}_sd{stddev}_S{n_samples}_errors_2", errors)

exact_gse = -2*L**2
con_bound = 5e-4
converged = np.sum(np.abs(energies[:, -1] - exact_gse)/np.abs(exact_gse) < con_bound)

# %%
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

for i in range(len(energies)):
    plot.errorbar(range(n_iter), (energies[i]-exact_gse)/np.abs(exact_gse),  # yerr=np.abs(errors[i]/exact_gse),
                  color="blue", alpha=0.1)

plot.set_ylim(-0.1, 1.1)
plot.set_xlim(0, n_iter)

fig.suptitle(f" ToricCode2d h={h}: size={shape},"
             f" real symmetric RBM with alpha={alpha},"
             f" lr_init={lr_init} to lr_end={lr_end} from"
             f" epoch {transition_begin} to {transition_begin+transition_steps} \n"
             f" n_chains={n_chains},"
             f" n_samples={n_samples} \n"
             f" converged (err_rel < {con_bound}): {converged} of {runs}")

plot.set_xlabel("iterations")
plot.set_ylabel("(E_vqs - E_exact) / |E_exact|")
plot.set_title(f"relative energy error over {n_iter} iterations for {runs} training runs")
plt.show()

if SAVE:
    fig.savefig(f"{RUN_PATH}/plots/RelativeErr_L{shape}_h{h}_sd{stddev}_S{n_samples}_2.pdf")

# %%
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

for i in range(len(energies)):
    plot.plot(range(n_iter), np.abs((energies[i]-exact_gse)/exact_gse),  # yerr=np.abs(errors[i]/exact_gse),
              color="blue", alpha=0.1)

plot.set_yscale("log")
plot.set_ylim(01e-7, 1)
plot.set_xlim(0, n_iter)

fig.suptitle(f" ToricCode2d h={h}: size={shape},"
             f" real symmetric RBM with alpha={alpha},"
             f" lr_init={lr_init} to lr_end={lr_end} from"
             f" epoch {transition_begin} to {transition_begin+transition_steps} \n"
             f" n_chains={n_chains},"
             f" n_samples={n_samples} \n"
             f" converged (err_rel < {con_bound}): {converged} of {runs}")

plot.set_xlabel("iterations")
plot.set_ylabel("|E_vqs - E_exact| / |E_exact|")
plot.set_title(f"relative energy error over {n_iter} iterations for {runs} training runs")
plt.show()

if SAVE:
    fig.savefig(f"{RUN_PATH}/plots/LogRelativeErr_L{shape}_h{h}_sd{stddev}_S{n_samples}_2.pdf")

