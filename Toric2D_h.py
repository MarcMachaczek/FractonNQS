from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import optax
import netket as nk

import geneqs
from geneqs.utils.training import loop_gs
from global_variables import RESULTS_PATH

from tqdm import tqdm

stddev = 0.01
default_kernel_init = jax.nn.initializers.normal(stddev)

# %%
L = 6  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)
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
link_perms = nk.utils.HashableArray(link_perms.astype(int))

# h_c at 0.328474, for L=10 compute sigma_z average over different h
# field_strengths = (0., 0.1, 0.2, 0.3, 0.31, 0.328474, 0.33, 0.35, 0.375, 0.4, 0.45, 0.5)
field_strengths = ((0.0, 0., 0.33),)

magnetizations = {}

# %%
# setting hyper-parameters
n_iter = 400
n_expect = 300_000  # number of samples to estimate observables from the trained vqs
n_chains = 1024  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 2
n_discard_per_chain = 8  # should be small for using many chains, default is 10% of n_samples
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.01
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTOnTheFly, diag_shift=diag_shift)

# define model parameters
alpha = 2
RBMSymm = nk.models.RBMSymm(symmetries=link_perms, alpha=alpha, kernel_init=default_kernel_init, dtype=float)

# be careful that keys of model and hyperparameters dicts match
models = {"rbm_symm": RBMSymm}

learning_rates = {"rbm_symm": 0.01}

for h in tqdm(field_strengths, "external_field"):
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
    variational_gs = {}
    training_records = {}
    for name, model in models.items():
        optimizer = optax.sgd(learning_rates[name])
        sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
        vqs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

        vqs, training_data = loop_gs(vqs, toric, optimizer, preconditioner, n_iter, min_steps=200)

        variational_gs[name] = vqs
        training_records[name] = training_data
    # save magnetization
    variational_gs["rbm_symm"].n_samples = n_expect
    magnetizations[h] = variational_gs["rbm_symm"].expect(magnetization)
    energy = variational_gs["rbm_symm"].expect(toric)

    # plot and save training data
    fig = plt.figure(dpi=300, figsize=(10, 10))
    plot = fig.add_subplot(111)
    for name, record in training_records.items():
        n_params = int(record["n_params"].value)
        plot.errorbar(record["Energy"].iters, record["Energy"].Mean, yerr=record["Energy"].Sigma,
                      label=f"{name}, lr={learning_rates[name]}, #p={n_params}")

    E0 = energy.mean[-1].real
    err = energy.sigma[-1].real

    fig.suptitle(f" ToricCode2d hz={h}: size={shape},"
                 f" real symmetric RBM with alpha={alpha},"
                 f" n_sweeps={L ** 2 * 2},"
                 f" n_chains={n_chains},"
                 f" n_samples={n_samples} \n"
                 f" RBM_symm E0 = {round(E0, 5)} +- {round(err, 5)}")

    plot.set_xlabel("iterations")
    plot.set_ylabel("energy")
    plot.set_title(f"using stochastic reconfiguration with diag_shift={diag_shift}")
    plot.legend()
    # fig.savefig(f"{RESULTS_PATH}/toric2d_h/VMC_lattice{shape}_h{g}.pdf")

# %%
# create and save magnetization plot
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)
for h, m in magnetizations.items():
    plot.errorbar(h, m.Mean.item().real, yerr=m.error_of_mean.item().real, marker="o", markersize=2, color="blue")
plot.set_xlabel("external field")
plot.set_ylabel("magnetization")
plot.set_title(f"Magnetization vs external field in z-direction for ToricCode2d with size={shape}")
plt.show()
# fig.savefig(f"{RESULTS_PATH}/toric2d_h/VMC_lattice{shape}_magnetizations.pdf")
