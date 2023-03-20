from matplotlib import pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import optax
import netket as nk
from netket.utils import HashableArray

import geneqs
from geneqs.utils.training import loop_gs
from global_variables import RESULTS_PATH

from tqdm import tqdm

stddev = 0.01
default_kernel_init = jax.nn.initializers.normal(stddev)
save_results = False

# %%
L = 8  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)
magnetization = 1 / hilbert.size * sum(nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size))

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# get (specific) symmetries of the model, in our case translations
perms = geneqs.utils.indexing.get_translations_cubical2d(shape)
link_perms = geneqs.utils.indexing.get_linkperms_cubical2d(perms)
# must be hashable to be included as flax.module attribute
# noinspection PyArgumentList
link_perms = HashableArray(link_perms.astype(int))

# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_plaquettes_cubical2d(shape)),  # plaquette correlators
               HashableArray(geneqs.utils.indexing.get_strings_cubical2d(0, shape)),  # x-string correlators
               HashableArray(geneqs.utils.indexing.get_strings_cubical2d(1, shape)),  # y-string correlators
               HashableArray(geneqs.utils.indexing.get_bonds_cubical2d(shape)))  # bond correlators

# noinspection PyArgumentList
correlator_symmetries = (HashableArray(jnp.asarray(perms)),  # plaquettes permute like sites
                         HashableArray(geneqs.utils.indexing.get_xstring_perms(shape)),
                         HashableArray(geneqs.utils.indexing.get_ystring_perms(shape)),
                         HashableArray(geneqs.utils.indexing.get_bondperms_cubical2d(perms)))

# h_c at 0.328474, for L=10 compute sigma_z average over different h
hx = 0.3
field_strengths = ((hx, 0., 0.0),
                   (hx, 0., 0.1),
                   (hx, 0., 0.2),
                   (hx, 0., 0.3),
                   (hx, 0., 0.31),
                   (hx, 0., 0.32),
                   (hx, 0., 0.33),
                   (hx, 0., 0.34),
                   (hx, 0., 0.35),
                   (hx, 0., 0.36),
                   (hx, 0., 0.37),
                   (hx, 0., 0.38),
                   (hx, 0., 0.39),
                   (hx, 0., 0.4),
                   (hx, 0., 0.45),
                   (hx, 0., 0.5))

magnetizations = {}

# %%  setting hyper-parameters
n_iter = 600
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 512  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 4
n_discard_per_chain = 8  # should be small for using many chains, default is 10% of n_samples
chunk_size = 1024 * 8  # doesn't work for gradient operations, need to check why!
n_expect = chunk_size * 12  # number of samples to estimate observables, must be dividable by chunk_size
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.01
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense, diag_shift=diag_shift)  # holomorphic=True)

# define correlation enhanced RBM
alpha = 1
cRBM = geneqs.models.CorrelationRBM(symmetries=link_perms,
                                    correlators=correlators,
                                    correlator_symmetries=correlator_symmetries,
                                    alpha=alpha,
                                    kernel_init=default_kernel_init,
                                    bias_init=default_kernel_init,
                                    param_dtype=complex)

model = cRBM
eval_model = "cRBM"

# learning rate scheduling
lr_init = 0.01
lr_end = 0.001
transition_begin = int(n_iter / 3)
transition_steps = int(n_iter / 3)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

# %%
for h in tqdm(field_strengths, "external_field"):
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    variational_gs, training_data = loop_gs(variational_gs, toric, optimizer, preconditioner, n_iter, min_iter)

    # save magnetization
    variational_gs.chunk_size = chunk_size
    variational_gs.n_samples = n_expect
    magnetizations[h] = variational_gs.expect(magnetization)

    # plot and save training data
    fig = plt.figure(dpi=300, figsize=(10, 10))
    plot = fig.add_subplot(111)

    n_params = int(training_data["n_params"].value)
    plot.errorbar(training_data["energy"].iters, training_data["energy"].Mean, yerr=training_data["energy"].Sigma,
                  label=f"{eval_model}, lr_init={lr_init}, #p={n_params}")

    E0 = training_data["energy"].Mean[-1].item().real
    err = training_data["energy"].Sigma[-1].item().real

    fig.suptitle(f" ToricCode2d hz={h}: size={shape},"
                 f" {eval_model} with alpha={alpha},"
                 f" n_sweeps={L ** 2 * 2},"
                 f" n_chains={n_chains},"
                 f" n_samples={n_samples} \n"
                 f" E0 = {round(E0, 5)} +- {round(err, 5)}")

    plot.set_xlabel("iterations")
    plot.set_ylabel("energy")
    plot.set_title(f"using stochastic reconfiguration with diag_shift={diag_shift}")
    plot.legend()
    if save_results:
        fig.savefig(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_h{h}.pdf")

# %%
mags = []
for h, mag in magnetizations.items():
    mags.append([*h] + [mag.Mean.item().real, mag.Sigma.item().real])
mags = np.asarray(mags)

if save_results:
    np.savetxt(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_magvals", mags)
# mags = np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_magvals")

# %%
# create and save magnetization plot
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

c = "red" if mags[0, 0] == 0. else "blue"
for mag in mags:
    plot.errorbar(mag[2], np.abs(mag[3]), yerr=mag[4], marker="o", markersize=2, color=c)

plot.plot(mags[:, 2], np.abs(mags[:, 3]), marker="o", markersize=2, color=c)

plot.set_xlabel("external field hz")
plot.set_ylabel("magnetization")
plot.set_title(f"Magnetization vs external field in z-direction for ToricCode2d of size={shape} "
               f"and hx={mags[0, 0]}")

plt.show()

if save_results:
    fig.savefig(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_magnetizations.pdf")

# took 02:50 till magnetization was calculated
