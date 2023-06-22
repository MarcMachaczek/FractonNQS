import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
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
import flax

import netket as nk
from netket.utils import HashableArray

import geneqs
from geneqs.utils.training import loop_gs
from geneqs.utils.eval_obs import get_locests_mixed
from global_variables import RESULTS_PATH

from matplotlib import pyplot as plt
import matplotlib
import numpy as np

from tqdm import tqdm
from functools import partial

matplotlib.rcParams.update({'font.size': 12})

# %% training configuration
save_results = True
save_stats = True  # whether to save stats logged during training to drive
save_path = f"{RESULTS_PATH}/checkerboard"
pre_init = False  # True only has effect when swip=="independent"
swipe = "independent"  # viable options: "independent", "left_right", "right_left"
checkpoint = None  # f"{RESULTS_PATH}/checkerbaord/vqs_ToricCRBM_L[8 8]_h(0.0, 0.0, 0.33).mpack"
# options are either None or the path to an .mpack file containing a VQSs

random_key = jax.random.PRNGKey(4214564359)  # so far only used for weightinit

# define fields for which to trian the NQS and get observables
direction_index = 0  # 0 for x, 1 for y, 2 for z;
direction = np.array([0.8, 0., 0.]).reshape(-1, 1)
field_strengths = (np.linspace(0, 1, 9) * direction).T
field_strengths = np.vstack((field_strengths, np.array([[0.31, 0, 0],
                                                        [0.32, 0, 0],
                                                        [0.33, 0, 0],
                                                        [0.34, 0, 0],
                                                        [0.35, 0, 0],
                                                        [0.36, 0, 0]])))
# for which fields indices histograms are created
hist_fields = np.array([[0.39, 0, 0.]])
save_fields = field_strengths  # field values for which vqs is serialized

# %% operators on hilbert space
L = 6  # this translates to L+1 without PBC
shape = jnp.array([L, L, L])
cube_graph = nk.graph.Hypercube(length=L, n_dim=3, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)

# define some observables
if direction_index == 0:
    abs_magnetization = geneqs.operators.observables.AbsXMagnetization(hilbert)
    magnetization = 1 / hilbert.size * sum([nk.operator.spin.sigmax(hilbert, i) for i in range(hilbert.size)])
elif direction_index == 1:
    abs_magnetization = geneqs.operators.observables.AbsYMagnetization(hilbert)
    magnetization = 1 / hilbert.size * sum([nk.operator.spin.sigmay(hilbert, i) for i in range(hilbert.size)])
elif direction_index == 2:
    abs_magnetization = geneqs.operators.observables.AbsZMagnetization(hilbert)
    magnetization = 1 / hilbert.size * sum([nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size)])

# %%  setting hyper-parameters and model
n_iter = 1200
min_iter = 1000  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 2 * 256 * n_ranks  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = int(n_chains * 32 / n_ranks)
n_discard_per_chain = 24  # should be small for using many chains, default is 10% of n_samples
chunk_size = int(n_samples / n_ranks / 2)
n_expect = chunk_size * 48 * n_ranks  # number of samples to estimate observables, must be dividable by chunk_size
n_bins = 20  # number of bins for calculating histograms

diag_shift_init = 1e-4
diag_shift_end = 1e-5
diag_shift_begin = int(n_iter * 2 / 5)
diag_shift_steps = int(n_iter * 1 / 5)
diag_shift_schedule = optax.linear_schedule(diag_shift_init, diag_shift_end, diag_shift_steps, diag_shift_begin)

preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense,
                                 solver=partial(jax.scipy.sparse.linalg.cg, tol=1e-6),
                                 diag_shift=diag_shift_schedule,
                                 holomorphic=True)

# learning rate scheduling
lr_init = 0.01
lr_end = 0.001
transition_begin = int(n_iter * 3 / 5)
transition_steps = int(n_iter * 1 / 3)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

# define correlation enhanced RBM
stddev = 0.01
trans_dev = stddev / 10  # standard deviation for transfer learning noise
default_kernel_init = jax.nn.initializers.normal(stddev)

perms = geneqs.utils.indexing.get_translations_cubical3d(shape, shift=2)
perms = nk.utils.HashableArray(perms.astype(int))

# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_cubes_cubical3d(shape, 2)),
               HashableArray(geneqs.utils.indexing.get_bonds_cubical3d(shape)))
# noinspection PyArgumentList
correlator_symmetries = (HashableArray(geneqs.utils.indexing.get_cubeperms_cubical3d(shape, 2)),
                         HashableArray(geneqs.utils.indexing.get_bondperms_cubical3d(shape, 2)))
# noinspection PyArgumentList
loops = (HashableArray(geneqs.utils.indexing.get_strings_cubical3d(0, shape)),
         HashableArray(geneqs.utils.indexing.get_strings_cubical3d(1, shape)),
         HashableArray(geneqs.utils.indexing.get_strings_cubical3d(2, shape)))
# noinspection PyArgumentList
loop_symmetries = (HashableArray(geneqs.utils.indexing.get_xstring_perms3d(shape, 2)),
                   HashableArray(geneqs.utils.indexing.get_ystring_perms3d(shape, 2)),
                   HashableArray(geneqs.utils.indexing.get_zstring_perms3d(shape, 2)))

alpha = 1 / 4
cRBM = geneqs.models.CheckerLoopCRBM(symmetries=perms,
                                     correlators=correlators,
                                     correlator_symmetries=correlator_symmetries,
                                     loops=loops,
                                     loop_symmetries=loop_symmetries,
                                     alpha=alpha,
                                     kernel_init=default_kernel_init,
                                     bias_init=default_kernel_init,
                                     param_dtype=complex)

model = cRBM
eval_model = "CheckerCRBM"

# create custom update rule
single_rule = nk.sampler.rules.LocalRule()
cube_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_cubes_cubical3d(shape, shift=2))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(0, shape))
ystring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(1, shape))
zstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(2, shape))
# noinspection PyArgumentList
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.51, 0.25, 0.08, 0.08, 0.08),
                                                          [single_rule,
                                                           cube_rule,
                                                           xstring_rule,
                                                           ystring_rule,
                                                           zstring_rule])

# make sure hist and save fields are contained in field_strengths and sort final field array
field_strengths = np.unique(np.round(np.vstack((field_strengths, hist_fields, save_fields)), 3), axis=0)
field_strengths = field_strengths[field_strengths[:, direction_index].argsort()]
if swipe == "right_left":
    field_strengths = field_strengths[::-1]
observables = geneqs.utils.eval_obs.ObservableCollector(key_names=("hx", "hy", "hz"))

# %% training
if pre_init:
    checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h=(0., 0., 0.))
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    vqs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    # exact ground state parameters for the checkerboard model, start with just noisy parameters
    random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
    real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, vqs.parameters, stddev)
    complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, vqs.parameters, stddev)
    gs_params = jax.tree_util.tree_map(lambda real, comp: real + 1j * comp, real_noise, complex_noise)
    # now set the exact parameters, this way noise is only added to all but the non-zero exact params
    cube_idx = checkerboard.cubes[jnp.array([0, 2, 8, 10])]
    exact_weights = jnp.zeros_like(vqs.parameters["symm_kernel"], dtype=complex)
    exact_weights = exact_weights.at[jnp.arange(4).reshape(-1, 1), cube_idx].set(1j * jnp.pi / 4)
    exact_weights = exact_weights.at[(jnp.arange(4) + 4).reshape(-1, 1), cube_idx].set(1j * jnp.pi / 4)

    gs_params = gs_params.copy({"symm_kernel": exact_weights})
    pre_init_parameters = gs_params

    vqs.parameters = pre_init_parameters
    print("init energy", vqs.expect(checkerboard))

if checkpoint is not None:
    checkpoint_sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    checkpoint_vqs = nk.vqs.MCState(checkpoint_sampler, model, n_samples=n_samples,
                                    n_discard_per_chain=n_discard_per_chain)
    with open(checkpoint, 'rb') as file:
        checkpoint_vqs = flax.serialization.from_bytes(checkpoint_vqs, file.read())
    print(f"checkpoint {checkpoint} loaded.")
last_trained_params = None if checkpoint is None else checkpoint_vqs.parameters
last_sampler_state = None if checkpoint is None else checkpoint_vqs.sampler_state

for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    print(f"training for field={h}")
    checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    vqs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    if swipe != "independent":
        if last_trained_params is not None:
            random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
            real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, vqs.parameters, trans_dev)
            complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, vqs.parameters, trans_dev)
            vqs.parameters = jax.tree_util.tree_map(lambda ltp, rn, cn: ltp + rn + 1j * cn,
                                                    last_trained_params, real_noise, complex_noise)
        # if last_sampler_state is not None:
        #     vqs.sampler_state = last_sampler_state
        vqs.sample(chain_length=256)  # let mcmc chains adapt to noisy initial paramters

    if pre_init and swipe == "independent":
        vqs.parameters = pre_init_parameters

    if save_stats:
        out_path = f"{save_path}/stats_L{shape}_{eval_model}_h{tuple([round(hi, 3) for hi in h])}.json"
    else:
        out_path = None
    vqs, training_data = loop_gs(vqs, checkerboard, optimizer, preconditioner, n_iter, min_iter, out=out_path)
    last_trained_params = vqs.parameters
    last_sampler_state = vqs.sampler_state

    # calculate observables, therefore set some params of vqs
    vqs.chunk_size = chunk_size
    vqs.n_samples = n_expect

    # calculate energy and specific heat / variance of energy
    energy_nk = vqs.expect(checkerboard)
    observables.add_nk_obs("energy", h, energy_nk)
    # calculate magnetization
    magnetization_nk = vqs.expect(magnetization)
    observables.add_nk_obs("mag", h, magnetization_nk)
    # calculate absolute magnetization
    abs_magnetization_nk = vqs.expect(abs_magnetization)
    observables.add_nk_obs("abs_mag", h, abs_magnetization_nk)

    if rank == 0:
        if np.any((h == save_fields).all(axis=1)) and save_results:
            filename = f"{eval_model}_L{shape}_h{tuple([round(hi, 3) for hi in h])}"
            with open(f"{save_path}/vqs_{filename}.mpack", 'wb') as file:
                file.write(flax.serialization.to_bytes(vqs))
            geneqs.utils.model_surgery.params_to_txt(vqs, f"{save_path}/params_{filename}.txt")

    # gather local estimators as each rank calculates them based on their own samples_per_rank
    if np.any((h == hist_fields).all(axis=1)):
        vqs.n_samples = int(n_samples / 2)
        random_key, init_state_key = jax.random.split(random_key)
        energy_locests = comm.gather(get_locests_mixed(init_state_key, vqs, checkerboard), root=0)
        mag_locests = comm.gather(get_locests_mixed(init_state_key, vqs, magnetization), root=0)
        abs_mag_locests = comm.gather(get_locests_mixed(init_state_key, vqs, abs_magnetization), root=0)

    # plot and save training data, save observables
    if rank == 0:
        fig = plt.figure(dpi=300, figsize=(12, 12))
        plot = fig.add_subplot(111)

        n_params = int(training_data["n_params"].value)
        plot.errorbar(training_data["Energy"].iters, training_data["Energy"].Mean, yerr=training_data["Energy"].Sigma,
                      label=f"Energy")

        fig.suptitle(f" Checkerboard h={tuple([round(hi, 3) for hi in h])}: size={shape} \n"
                     f" {eval_model}, alpha={alpha}, #p={n_params}, lr from {lr_init} to {lr_end} \n"
                     f" n_discard={n_discard_per_chain},"
                     f" n_chains={n_chains},"
                     f" n_samples={n_samples} \n"
                     f" pre_init={pre_init}, stddev={stddev}, trans_dev={trans_dev}, swipe={swipe}")

        plot.set_xlabel("Training Iterations")
        plot.set_ylabel("Observables")

        E0, err = energy_nk.Mean.item().real, energy_nk.Sigma.item().real
        plot.set_title(f"E0 = {round(E0, 5)} +- {round(err, 5)} using SR with diag_shift={diag_shift_init}"
                       f" down to {diag_shift_end}")
        plot.legend()
        if save_results:
            fig.savefig(
                f"{save_path}/L{shape}_{eval_model}_h{tuple([round(hi, 3) for hi in h])}.pdf")

        # create histograms
        if np.any((h == hist_fields).all(axis=1)):
            observables.add_hist("energy", h, np.histogram(np.asarray(energy_locests) / hilbert.size, n_bins))
            observables.add_hist("mag", h, np.histogram(np.asarray(mag_locests), n_bins))
            observables.add_hist("abs_mag", h, np.histogram(np.asarray(abs_mag_locests), n_bins))

        # save observables to file
        if save_results:
            save_array = observables.obs_to_array(separate_keys=False)[-1].reshape(1, -1)
            with open(f"{save_path}/L{shape}_{eval_model}_observables.txt", "ab") as f:
                if os.path.getsize(f"{save_path}/L{shape}_{eval_model}_observables.txt") == 0:
                    np.savetxt(f, save_array, header=", ".join(observables.key_names + observables.obs_names))
                else:
                    np.savetxt(f, save_array)

# %% save histograms
if rank == 0:
    for hist_name, _ in observables.histograms.items():
        np.save(f"{save_path}/hists_{hist_name}_L{shape}_{eval_model}.npy",
                observables.hist_to_array(hist_name))

# %% create and save magnetization plot
if rank == 0:
    obs_to_array = np.loadtxt(f"{save_path}/L{shape}_{eval_model}_observables.txt")

    fig = plt.figure(dpi=300, figsize=(10, 10))
    plot = fig.add_subplot(111)

    c = "red"
    for obs in obs_to_array:
        plot.errorbar(obs[direction_index], np.abs(obs[5]), yerr=obs[6], marker="o", markersize=2, color=c)

    plot.plot(obs_to_array[:, direction_index], np.abs(obs_to_array[:, 5]), marker="o", markersize=2, color=c)

    plot.set_xlabel("external magnetic field")
    plot.set_ylabel("magnetization")
    plot.set_title(
        f"Magnetization vs external field in {direction.flatten()}-direction for Checkerboard of size={shape}")

    plot.set_xlim(0, field_strengths[-1][direction_index])

    if save_results:
        fig.savefig(f"{save_path}/Magnetizations_L{shape}_{eval_model}_hdir{direction.flatten()}.pdf")
