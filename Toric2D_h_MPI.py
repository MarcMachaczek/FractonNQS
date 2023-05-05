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
from functools import partial

save_results = True
pre_train = False

random_key = jax.random.PRNGKey(421)  # this can be used to make results deterministic, but so far is not used

# %%
L = 8  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)
magnetization = geneqs.operators.observables.Magnetization(hilbert)
wilsonob = geneqs.operators.observables.get_netket_wilsonob(hilbert, shape)

# get (specific) symmetries of the model, in our case translations
perms = geneqs.utils.indexing.get_translations_cubical2d(shape, shift=1)
link_perms = geneqs.utils.indexing.get_linkperms_cubical2d(perms)
# must be hashable to be included as flax.module attribute
# noinspection PyArgumentList
link_perms = HashableArray(link_perms.astype(int))

# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_plaquettes_cubical2d(shape)),  # plaquette correlators
               HashableArray(geneqs.utils.indexing.get_bonds_cubical2d(shape)),  # bond correlators
               HashableArray(geneqs.utils.indexing.get_strings_cubical2d(0, shape)),  # x-string correlators
               HashableArray(geneqs.utils.indexing.get_strings_cubical2d(1, shape)))  # y-string correlators

# noinspection PyArgumentList
correlator_symmetries = (HashableArray(jnp.asarray(perms)),  # plaquettes permute like sites
                         HashableArray(geneqs.utils.indexing.get_bondperms_cubical2d(perms)),
                         HashableArray(geneqs.utils.indexing.get_xstring_perms(shape)),
                         HashableArray(geneqs.utils.indexing.get_ystring_perms(shape)))

# h_c at 0.328474, for L=10 compute sigma_z average over different h
direction = np.array([0.8, 0, 0]).reshape(-1, 1)
field_strengths = (np.linspace(0, 1, 20) * direction).T

observables = {}

# %%  setting hyper-parameters
n_iter = 500
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 256 * 2  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 32
n_discard_per_chain = 64  # should be small for using many chains, default is 10% of n_samples
chunk_size = 1024 * 8  # doesn't work for gradient operations, need to check why!
n_expect = chunk_size * 32  # number of samples to estimate observables, must be dividable by chunk_size
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift_init = 1e-4
diag_shift_end = 1e-5
diag_shift_begin = int(n_iter / 3)
diag_shift_steps = int(n_iter / 3)
diag_shift_schedule = optax.linear_schedule(diag_shift_init, diag_shift_end, diag_shift_steps, diag_shift_begin)

preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense, solver=partial(jax.scipy.sparse.linalg.cg, tol=1e-6), diag_shift=diag_shift_schedule, holomorphic=True)

# define correlation enhanced RBM
stddev = 0.01
default_kernel_init = jax.nn.initializers.normal(stddev)

alpha = 1
cRBM = geneqs.models.ToricCRBM(symmetries=link_perms,
                               correlators=correlators,
                               correlator_symmetries=correlator_symmetries,
                               alpha=alpha,
                               kernel_init=default_kernel_init,
                               bias_init=default_kernel_init,
                               param_dtype=complex)

model = cRBM
eval_model = "ToricCRBM"

# create custom update rule
single_rule = nk.sampler.rules.LocalRule()
vertex_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_stars_cubical2d(shape))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(0, shape))
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.5, 0.25, 0.25), [single_rule, vertex_rule, xstring_rule])

# learning rate scheduling
lr_init = 0.01
lr_end = 0.001
transition_begin = int(n_iter / 3)
transition_steps = int(n_iter / 3)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

# %%
if pre_train:
    print(f"pre-training for {n_iter} iterations on the pure model")

    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h=(0., 0., 0.))
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    # exact ground state parameters for the 2d toric code
    noise_generator = jax.nn.initializers.normal(stddev)
    random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
    gs_params = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), variational_gs.parameters)
    plaq_idxs = toric.plaqs[0].reshape(1, -1)
    star_idxs = toric.stars[0].reshape(1, -1)
    exact_weights = jnp.zeros_like(variational_gs.parameters["symm_kernel"], dtype=complex)
    exact_weights = exact_weights.at[0, plaq_idxs].set(1j * jnp.pi/4)
    exact_weights = exact_weights.at[1, star_idxs].set(1j * jnp.pi/2)
    # add noise to non-zero parameters
    gs_params = gs_params.copy({"symm_kernel": exact_weights})
    gs_params = jax.tree_util.tree_map(lambda p: p + noise_generator(noise_key_real, p.shape) + 1j * noise_generator(noise_key_complex, p.shape), gs_params)
    pretrained_parameters = gs_params

    # variational_gs, training_data = loop_gs(variational_gs, toric, optimizer, preconditioner, n_iter, min_iter)
    # pretrained_parameters = variational_gs.parameters

    print("\n pre-training finished")

for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    if pre_train:
        variational_gs.parameters = pretrained_parameters
        # print(f"pre_train energy after adding noise:{variational_gs.expect(toric).Mean.item().real}")

    variational_gs, training_data = loop_gs(variational_gs, toric, optimizer, preconditioner, n_iter, min_iter)

    # calculate observables, therefore set some params of vqs
    variational_gs.chunk_size = chunk_size
    variational_gs.n_samples = n_expect
    observables[h] = {}

    # calculate energy and specific heat / variance of energy
    observables[h]["energy"] = variational_gs.expect(toric)

    # calculate magnetization
    observables[h]["mag"] = variational_gs.expect(magnetization)
    
    # calcualte wilson loop operator, doesnt work yet, too many entries
    # observables[h]["wilson"] = variational_gs.expect(wilsonob)

    # plot and save training data, save observables
    if rank == 0:
        fig = plt.figure(dpi=300, figsize=(12, 12))
        plot = fig.add_subplot(111)

        n_params = int(training_data["n_params"].value)
        plot.errorbar(training_data["energy"].iters, training_data["energy"].Mean, yerr=training_data["energy"].Sigma,
                      label=f"{eval_model}, lr_init={lr_init}, #p={n_params}")

        E0 = observables[h]["energy"].Mean.item().real
        err = observables[h]["energy"].Sigma.item().real

        fig.suptitle(f" ToricCode2d h={tuple([round(hi, 3) for hi in h])}: size={shape},"
                 f" {eval_model}, alpha={alpha},"
                 f" n_sweeps={L ** 2 * 2},"
                 f" n_chains={n_chains},"
                 f" n_samples={n_samples} \n"
                 f" pre_train={pre_train}, stddev={stddev}")

        plot.set_xlabel("iterations")
        plot.set_ylabel("energy")
        plot.set_title(f"E0 = {round(E0, 5)} +- {round(err, 5)} using SR with diag_shift={diag_shift_init}-{diag_shift_end}")
        plot.legend()
        if save_results:
            fig.savefig(
                f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_h{tuple([round(hi, 3) for hi in h])}.pdf")

        # save observables to file
        if save_results:
            obs = observables[h]
            obs_to_write = np.asarray([[*h] +
                                      [obs["mag"].Mean.item().real, obs["mag"].Sigma.item().real] +
                                      [obs["energy"].Mean.item().real, obs["energy"].Sigma.item().real]])

            with open(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_observables.txt", "ab") as f:
                if os.path.getsize(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_observables.txt") == 0:
                    np.savetxt(f, obs_to_write,
                               header="hx, hy, hz, mag, mag_var, energy, energy_var")
                else:
                    np.savetxt(f, obs_to_write)

# %%
# create and save magnetization plot
if rank == 0:
    obs_to_array = np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_observables.txt")

    fig = plt.figure(dpi=300, figsize=(10, 10))
    plot = fig.add_subplot(111)

    c = "red"
    for obs in obs_to_array:
        plot.errorbar(obs[0], np.abs(obs[3]), yerr=obs[4], marker="o", markersize=2, color=c)

    plot.plot(obs_to_array[:, 0], np.abs(obs_to_array[:, 3]), marker="o", markersize=2, color=c)

    plot.set_xlabel("external field hx")
    plot.set_ylabel("magnetization")
    plot.set_title(f"Magnetization vs external field in {direction.flatten()}-direction for ToricCode2d of size={shape}")

    plot.set_xlim(0, field_strengths[-1][2])
    plt.show()

    if save_results:
        fig.savefig(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_magnetizations.pdf")
# took 02:50 till magnetization was calculated
