import os
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
# supress warning about no cuda mpi version
# we don't need that because jax handles that, we only want to run copies of a process with some communication
os.environ["MPI4JAX_USE_CUDA_MPI"] = "0"
# set only one visible device
os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
# force to use gpu
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import jax
# jax.distributed.initialize()
import jax.numpy as jnp
import optax

import netket as nk
from netket.utils import HashableArray

import geneqs
from geneqs.utils.training import loop_gs
from global_variables import RESULTS_PATH

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

save_results = True
pre_train = False

random_key = jax.random.PRNGKey(42)  # this can be used to make results deterministic, but so far is not used

# %%
L = 10  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)
magnetization = 1 / hilbert.size * sum(nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size))

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

observables = {}

# %%  setting hyper-parameters
n_iter = 700
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 512 * 2  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 6
n_discard_per_chain = 16  # should be small for using many chains, default is 10% of n_samples
chunk_size = 1024 * 16  # doesn't work for gradient operations, need to check why!
n_expect = chunk_size * 16  # number of samples to estimate observables, must be dividable by chunk_size
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.0001
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense, diag_shift=diag_shift, holomorphic=True)

# define correlation enhanced RBM
stddev = 0.01
default_kernel_init = jax.nn.initializers.normal(stddev)

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
lr_end = 0.0005
transition_begin = int(n_iter / 3)
transition_steps = int(n_iter / 3)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

# %%
if pre_train:
    print(f"pre-training for {n_iter} iterations on the pure model")

    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h=(0., 0., 0.))
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    variational_gs, training_data = loop_gs(variational_gs, toric, optimizer, preconditioner, n_iter, min_iter)
    pretrained_parameters = variational_gs.parameters

    print("\n pre-training finished")

for h in tqdm(field_strengths, "external_field"):
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    if pre_train:
        noise_generator = jax.nn.initializers.normal(stddev / 2)
        random_key, noise_key = jax.random.split(random_key, 2)
        variational_gs.parameters = jax.tree_util.tree_map(lambda x: x + noise_generator(noise_key, x.shape),
                                                           pretrained_parameters)

    variational_gs, training_data = loop_gs(variational_gs, toric, optimizer, preconditioner, n_iter, min_iter)

    # calculate observables, therefore set some params of vqs
    variational_gs.chunk_size = chunk_size
    variational_gs.n_samples = n_expect
    observables[h] = {}

    # calculate energy and specific heat / variance of energy
    observables[h]["energy"] = variational_gs.expect(toric)

    # calculate magnetization
    observables[h]["mag"] = variational_gs.expect(magnetization)

    # calculate susceptibility / variance of magnetization
    m = observables[h]["mag"].Mean.item().real
    chi = 1 / hilbert.size * sum((nk.operator.spin.sigmaz(hilbert, i) - m) *
                                 (nk.operator.spin.sigmaz(hilbert, i) - m).H
                                 for i in range(hilbert.size))
    observables[h]["sus"] = variational_gs.expect(chi)

    # plot and save training data
    if rank == 0:
        fig = plt.figure(dpi=300, figsize=(10, 10))
        plot = fig.add_subplot(111)

        n_params = int(training_data["n_params"].value)
        plot.errorbar(training_data["energy"].iters, training_data["energy"].Mean, yerr=training_data["energy"].Sigma,
                      label=f"{eval_model}, lr_init={lr_init}, #p={n_params}")

        E0 = observables[h]["energy"].Mean[-1].item().real
        err = observables[h]["energy"].Sigma[-1].item().real

        fig.suptitle(f" ToricCode2d h={h}: size={shape},"
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
obs_to_array = []
for h, obs in observables.items():
    obs_to_array.append([*h] +
                        [obs["mag"].Mean.item().real, obs["mag"].Sigma.item().real] +
                        [obs["sus"].Mean.item().real, obs["sus"].Sigma.item().real] +
                        [obs["energy"].Mean.item().real, obs["energy"].Sigma.item().real])
obs_to_array = np.asarray(obs_to_array)

if rank == 0:
    if save_results:
        np.savetxt(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_observables", obs_to_array,
                   header="hx, hy, hz, mag, mag_var, susceptibility, sus_var, energy, energy_var")
# mags = np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_magvals")

# %%
# create and save magnetization plot
if rank == 0:
    fig = plt.figure(dpi=300, figsize=(10, 10))
    plot = fig.add_subplot(111)

    c = "red" if hx == 0. else "blue"
    for obs in obs_to_array:
        plot.errorbar(obs[2], np.abs(obs[3]), yerr=obs[4], marker="o", markersize=2, color=c)

    plot.plot(obs_to_array[:, 2], np.abs(obs_to_array[:, 3]), marker="o", markersize=2, color=c)

    plot.set_xlabel("external field hz")
    plot.set_ylabel("magnetization")
    plot.set_title(f"Magnetization vs external field in z-direction for ToricCode2d of size={shape} "
                   f"and hx={hx}")

    plot.set_xlim(0, field_strengths[-1][-1])
    plot.set_ylim(0, 1.)
    plt.show()

    if save_results:
        fig.savefig(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_magnetizations.pdf")
# took 02:50 till magnetization was calculated