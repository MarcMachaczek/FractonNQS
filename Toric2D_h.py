from matplotlib import pyplot as plt
import numpy as np

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
L = 10  # size should be at least 3, else there are problems with pbc and indexing
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
# field_strengths = ((0., 0., 0.0),
#                    (0., 0., 0.1),
#                    (0., 0., 0.2),
#                    (0., 0., 0.3),
#                    (0., 0., 0.32),
#                    (0., 0., 0.34),
#                    (0., 0., 0.36),
#                    (0., 0., 0.38),
#                    (0., 0., 0.4),
#                    (0., 0., 0.45),
#                    (0., 0., 0.5))

magnetizations = {}

# %%
field_strengths = ((0., 0., 0.3),
                   (0., 0., 0.34),
                   (0., 0., 0.38),
                   (0., 0., 0.45))
# setting hyper-parameters
n_iter = 300
n_chains = 512  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 4
n_discard_per_chain = 8  # should be small for using many chains, default is 10% of n_samples
chunk_size = 1024 * 8  # doesn't work for gradient operations, need to check why!
n_expect = chunk_size * 12  # number of samples to estimate observables, must be dividable by chunk_size
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.01
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense, diag_shift=diag_shift,)  # holomorphic=True)

# define model parameters
alpha = 2
RBMSymm = nk.models.RBMSymm(symmetries=link_perms, alpha=alpha, kernel_init=default_kernel_init, param_dtype=float)

# be careful that keys of model and hyperparameters dicts match
eval_model = "real_rbm_symm"
models = {"real_rbm_symm": RBMSymm}

learning_rates = {"real_rbm_symm": 0.01}

# %%
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
    variational_gs[eval_model].chunk_size = chunk_size
    variational_gs[eval_model].n_samples = n_expect
    magnetizations[h] = variational_gs[eval_model].expect(magnetization)

    # plot and save training data
    fig = plt.figure(dpi=300, figsize=(10, 10))
    plot = fig.add_subplot(111)
    for name, record in training_records.items():
        n_params = int(record["n_params"].value)
        plot.errorbar(record["energy"].iters, record["energy"].Mean, yerr=record["energy"].Sigma,
                      label=f"{name}, lr={learning_rates[name]}, #p={n_params}")

    E0 = training_records[eval_model]["energy"].Mean[-1].item().real
    err = training_records[eval_model]["energy"].Sigma[-1].item().real

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
    fig.savefig(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_h{h}.pdf")

# %%
mags = []
for h, mag in magnetizations.items():
    mags.append([*h] + [mag.Mean.item().real, mag.Sigma.item().real])

np.savetxt(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_magvals", np.asarray(mags))

# %%
# create and save magnetization plot
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)
for h, m in magnetizations.items():
    hz = h[2]
    c = "red" if h[0] == 0. else "blue"
    plot.errorbar(hz, np.abs(m.Mean.item().real), yerr=m.error_of_mean.item().real, marker="o", markersize=2, color=c)
    plot.plot(hz, np.abs(m.Mean.item().real), marker="o", markersize=2, color=c) # TODO: correct plot

plot.set_xlabel("external field hz")
plot.set_ylabel("magnetization")
plot.set_title(f"Magnetization vs external field in z-direction for ToricCode2d of size={shape} "
               f"and hx={field_strengths[0][0]}")
plt.show()
# fig.savefig(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_magnetizations.pdf")
