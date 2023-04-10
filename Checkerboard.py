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

save_results = False
pre_train = False

random_key = jax.random.PRNGKey(42)  # this can be used to make results deterministic, but so far is not used

# %%
L = 4  # this translates to L+1 without PBC
shape = jnp.array([L, L, L])
cube_graph = nk.graph.Hypercube(length=L, n_dim=3, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)
magnetization = 1 / hilbert.size * sum(nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size))

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(projection='3d')
geneqs.utils.plotting.plot_checkerboard(ax, L)
plt.show()

# get (specific) symmetries of the model, in our case translations
perms = geneqs.utils.indexing.get_translations_cubical3d(shape, shift=2)
# must be hashable to be included as flax.module attribute
# noinspection PyArgumentList
perms = HashableArray(perms.astype(int))

# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_cubes_cubical3d(shape, shift=2)),)

# noinspection PyArgumentList
correlator_symmetries = (HashableArray(geneqs.utils.indexing.get_cubeperms_cubical3d(shape, shift=2)),)

# h_c at 0.328474, for L=10 compute sigma_z average over different h
hx = 0.0
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

field_strengths = ((hx, 0., 0.0),)

observables = {}

# %%  setting hyper-parameters
n_iter = 200
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 512  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 16
n_discard_per_chain = 128  # should be small for using many chains, default is 10% of n_samples
chunk_size = 1024 * 8  # doesn't work for gradient operations, need to check why!
n_expect = chunk_size * 12  # number of samples to estimate observables, must be dividable by chunk_size
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.001
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense, diag_shift=diag_shift,)  # holomorphic=True)

# define correlation enhanced RBM
stddev = 0.001
default_kernel_init = jax.nn.initializers.normal(stddev)

alpha = 0.5
cRBM = geneqs.models.CorrelationRBM(symmetries=perms,
                                    correlators=correlators,
                                    correlator_symmetries=correlator_symmetries,
                                    alpha=alpha,
                                    kernel_init=default_kernel_init,
                                    bias_init=default_kernel_init,
                                    param_dtype=float)

model = cRBM
eval_model = "cRBM"

# learning rate scheduling
lr_init = 0.02
lr_end = lr_init
transition_begin = int(n_iter / 3)
transition_steps = int(n_iter / 3)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

# %%
if pre_train:
    print(f"pre-training for {n_iter} iterations on the pure model")

    hamiltonian = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h=(0., 0., 0.))
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    variational_gs, training_data = loop_gs(variational_gs, hamiltonian, optimizer, preconditioner, n_iter, min_iter)
    pretrained_parameters = variational_gs.parameters

    print("\n pre-training finished")

for h in tqdm(field_strengths, "external_field"):
    hamiltonian = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    if pre_train:
        noise_generator = jax.nn.initializers.normal(stddev/2)
        random_key, noise_key = jax.random.split(random_key, 2)
        variational_gs.parameters = jax.tree_util.tree_map(lambda x: x + noise_generator(noise_key, x.shape),
                                                           pretrained_parameters)

    variational_gs, training_data = loop_gs(variational_gs, hamiltonian, optimizer, preconditioner, n_iter, min_iter)

    # calculate observables, therefore set some params of vqs
    variational_gs.chunk_size = chunk_size
    variational_gs.n_samples = n_expect
    observables[h] = {}

    # calculate energy and specific heat / variance of energy
    observables[h]["energy"] = variational_gs.expect(hamiltonian)

    # calculate magnetization
    observables[h]["mag"] = variational_gs.expect(magnetization)

    # calculate susceptibility / variance of magnetization
    m = observables[h]["mag"].Mean.item().real
    chi = 1 / hilbert.size * sum((nk.operator.spin.sigmaz(hilbert, i) - m) *
                                 (nk.operator.spin.sigmaz(hilbert, i) - m).H
                                 for i in range(hilbert.size))
    observables[h]["sus"] = variational_gs.expect(chi)

    # plot and save training data
    fig = plt.figure(dpi=300, figsize=(10, 10))
    plot = fig.add_subplot(111)

    n_params = int(training_data["n_params"].value)
    plot.errorbar(training_data["energy"].iters, training_data["energy"].Mean, yerr=training_data["energy"].Sigma,
                  label=f"{eval_model}, lr_init={lr_init}, #p={n_params}")

    E0 = observables[h]["energy"].Mean.item().real
    err = observables[h]["energy"].Sigma.item().real

    fig.suptitle(f" Checkerboard h={h}: size={shape},"
                 f" {eval_model} with alpha={alpha},"
                 f" n_sweeps={hilbert.size},"
                 f" n_chains={n_chains},"
                 f" n_samples={n_samples} \n"
                 f" E0 = {round(E0, 5)} +- {round(err, 5)}")

    plot.set_xlabel("iterations")
    plot.set_ylabel("energy")
    plot.set_title(f"using stochastic reconfiguration with diag_shift={diag_shift}")
    plot.legend()
    if save_results:
        fig.savefig(f"{RESULTS_PATH}/checkerboard/L{shape}_{eval_model}_a{alpha}_h{h}.pdf")

# %%
obs_to_array = []
for h, obs in observables.items():
    obs_to_array.append([*h] +
                        [obs["mag"].Mean.item().real, obs["mag"].Sigma.item().real] +
                        [obs["sus"].Mean.item().real, obs["sus"].Sigma.item().real] +
                        [obs["energy"].Mean.item().real, obs["energy"].Sigma.item().real])
obs_to_array = np.asarray(obs_to_array)

if save_results:
    np.savetxt(f"{RESULTS_PATH}/checkerboard/L{shape}_{eval_model}_a{alpha}_observables", obs_to_array,
               header="hx, hy, hz, mag, mag_var, susceptibility, sus_var, energy, energy_var")

# %%
# create and save magnetization plot
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

c = "red" if hx == 0. else "blue"
for obs in obs_to_array:
    plot.errorbar(obs[2], np.abs(obs[3]), yerr=obs[4], marker="o", markersize=2, color=c)

plot.plot(obs_to_array[:, 2], np.abs(obs_to_array[:, 3]), marker="o", markersize=2, color=c)

plot.set_xlabel("external field hz")
plot.set_ylabel("magnetization")
plot.set_title(f"Magnetization vs external field in z-direction for Checkerboard of size={shape} "
               f"and hx={hx}")

plot.set_xlim(0, 0.5)
plot.set_ylim(0, 1.)
plt.show()

if save_results:
    fig.savefig(f"{RESULTS_PATH}/checkerboard/L{shape}_{eval_model}_a{alpha}_magnetizations.pdf")

