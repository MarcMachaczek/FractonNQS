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
import numpy as np

from tqdm import tqdm
from functools import partial

# %% training configuration
save_results = True
save_path = f"{RESULTS_PATH}/toric2d_h/mpi"
pre_init = False  # True only has effect when swip=="independent"
swipe = "left_right"  # viable options: "independent", "left_right", "right_left"
# if pre_init==True and swipe!="independent", pre_init only applies to the first training run

random_key = jax.random.PRNGKey(4214564359)  # can be used to make results deterministic, so far only used for weightinit

# define fields for which to trian the NQS and get observables
direction_index = 2  # 0 for x, 1 for y, 2 for z;
direction = np.array([0., 0., 0.8]).reshape(-1, 1)
field_strengths = (np.linspace(0, 1, 9) * direction).T
field_strengths = np.vstack((field_strengths, np.array([[0., 0, 0.31],
                                                        [0., 0, 0.33],
                                                        [0., 0, 0.35]])))
# for which fields indices histograms are created
hist_fields = np.array([[0., 0, 0.2],
                        [0., 0, 0.33],
                        [0., 0, 0.5]])
save_fields = hist_fields  # field values for which vqs is serialized

# %% operators on hilbert space
L = 8  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)

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
wilsonob = geneqs.operators.observables.get_netket_wilsonob(hilbert, shape)

positions = jnp.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
A_B = 1 / hilbert.size * sum([geneqs.operators.toric_2d.get_netket_star(hilbert, p, shape) for p in positions]) - \
      1 / hilbert.size * sum([geneqs.operators.toric_2d.get_netket_plaq(hilbert, p, shape) for p in positions])

# %%  setting hyper-parameters and model
n_iter = 800
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 256 * n_ranks  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = int(n_chains * 64 / n_ranks)
n_discard_per_chain = 24  # should be small for using many chains, default is 10% of n_samples
chunk_size = n_samples  # doesn't work for gradient operations, need to check why!
n_expect = chunk_size * 48  # number of samples to estimate observables, must be dividable by chunk_size
n_bins = 20  # number of bins for calculating histograms

diag_shift_init = 1e-4
diag_shift_end = 1e-5
diag_shift_begin = int(n_iter / 3)
diag_shift_steps = int(n_iter / 3)
diag_shift_schedule = optax.linear_schedule(diag_shift_init, diag_shift_end, diag_shift_steps, diag_shift_begin)

preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense,
                                 solver=partial(jax.scipy.sparse.linalg.cg, tol=1e-6),
                                 diag_shift=diag_shift_schedule,
                                 holomorphic=True)

# learning rate scheduling
lr_init = 0.01
lr_end = 0.01
transition_begin = int(n_iter / 3)
transition_steps = int(n_iter / 3)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

# define correlation enhanced RBM
stddev = 0.01
default_kernel_init = jax.nn.initializers.normal(stddev)

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

alpha = 1
cRBM = geneqs.models.ToricLoopCRBM(symmetries=link_perms,
                                   correlators=correlators,
                                   correlator_symmetries=correlator_symmetries,
                                   loops=loops,
                                   loop_symmetries=loop_symmetries,
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
ystring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(1, shape))
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.55, 0.25, 0.1, 0.1),
                                                          [single_rule, vertex_rule, xstring_rule, ystring_rule])

# make sure hist and save fields are contained in field_strengths and sort final field array
field_strengths = np.unique(np.round(np.vstack((field_strengths, hist_fields, save_fields)), 3), axis=0)
field_strengths = field_strengths[field_strengths[:, direction_index].argsort()]
if swipe == "right_left":
    field_strengths = field_strengths[::-1]
observables = geneqs.utils.eval_obs.ObservableCollector(key_names=("hx", "hy", "hz"))

# %% training
if pre_init:
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h=(0., 0., 0.))
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    vqs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    # exact ground state parameters for the 2d toric code, start with just noisy parameters
    random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
    real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, vqs.parameters, stddev / 10)
    complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, vqs.parameters, stddev / 10)
    gs_params = jax.tree_util.tree_map(lambda real, comp: real + 1j * comp, real_noise, complex_noise)
    # now set the exact parameters, this way noise is only added to all but the non-zero exact params
    plaq_idxs = toric.plaqs[0].reshape(1, -1)
    star_idxs = toric.stars[0].reshape(1, -1)
    exact_weights = jnp.zeros_like(vqs.parameters["symm_kernel"], dtype=complex)
    exact_weights = exact_weights.at[0, plaq_idxs].set(1j * jnp.pi / 4)
    exact_weights = exact_weights.at[1, star_idxs].set(1j * jnp.pi / 2)

    gs_params = gs_params.copy({"symm_kernel": exact_weights})
    pre_init_parameters = gs_params

    vqs.parameters = pre_init_parameters
    print("init energy", vqs.expect(toric))

last_trained_params = None
for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    vqs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    if swipe != "independent":
        if last_trained_params is not None:
            random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
            real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, vqs.parameters, stddev / 10)
            complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, vqs.parameters, stddev / 10)
            vqs.parameters = jax.tree_util.tree_map(lambda ltp, r, c: ltp + r + 1j * c,
                                                    last_trained_params, real_noise, complex_noise)
        elif pre_init:
            vqs.parameters = pre_init_parameters

    if pre_init and swipe == "independent":
        vqs.parameters = pre_init_parameters

    vqs, training_data = loop_gs(vqs, toric, optimizer, preconditioner, n_iter, min_iter)
    last_trained_params = vqs.parameters

    # calculate observables, therefore set some params of vqs
    vqs.chunk_size = chunk_size
    vqs.n_samples = n_expect

    # calculate energy and specific heat / variance of energy
    energy_nk = vqs.expect(toric)
    observables.add_nk_obs("energy", h, energy_nk)
    # calculate magnetization
    magnetization_nk = vqs.expect(magnetization)
    observables.add_nk_obs("mag", h, magnetization_nk)
    # calculate absolute magnetization
    abs_magnetization_nk = vqs.expect(abs_magnetization)
    observables.add_nk_obs("abs_mag", h, abs_magnetization_nk)
    # calcualte wilson loop operator
    wilsonob_nk = vqs.expect(wilsonob)
    observables.add_nk_obs("wilson", h, wilsonob_nk)

    if rank == 0:
        if np.any((h == save_fields).all(axis=1)) and save_results:
            filename = f"{eval_model}_L{shape}_h{tuple([round(hi, 3) for hi in h])}"
            with open(f"{save_path}/vqs_{filename}.mpack", 'wb') as file:
                file.write(flax.serialization.to_bytes(vqs))
            geneqs.utils.model_surgery.params_to_txt(vqs, f"{save_path}/params_{filename}.txt")

    # gather local estimators as each rank calculates them based on their own samples_per_rank
    if np.any((h == hist_fields).all(axis=1)):
        vqs.n_samples = n_samples
        random_key, init_state_key = jax.random.split(random_key)
        energy_locests = comm.gather(get_locests_mixed(init_state_key, vqs, toric), root=0)
        mag_locests = comm.gather(get_locests_mixed(init_state_key, vqs, magnetization), root=0)
        abs_mag_locests = comm.gather(get_locests_mixed(init_state_key, vqs, abs_magnetization), root=0)
        A_B_locests = comm.gather(get_locests_mixed(init_state_key, vqs, A_B), root=0)

    # plot and save training data, save observables
    if rank == 0:
        fig = plt.figure(dpi=300, figsize=(12, 12))
        plot = fig.add_subplot(111)

        n_params = int(training_data["n_params"].value)
        plot.errorbar(training_data["Energy"].iters, training_data["Energy"].Mean, yerr=training_data["Energy"].Sigma,
                      label=f"{eval_model}, lr_init={lr_init}, #p={n_params}")

        fig.suptitle(f" ToricCode2d h={tuple([round(hi, 3) for hi in h])}: size={shape},"
                     f" {eval_model}, alpha={alpha},"
                     f" n_discard={n_discard_per_chain},"
                     f" n_chains={n_chains},"
                     f" n_samples={n_samples} \n"
                     f" pre_init={pre_init}, stddev={stddev}, swipe={swipe}")

        plot.set_xlabel("iterations")
        plot.set_ylabel("energy")

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
            observables.add_hist("A_B", h, np.histogram(np.asarray(A_B_locests), n_bins))

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
        plot.errorbar(obs[direction_index], np.abs(obs[3]), yerr=obs[4], marker="o", markersize=2, color=c)

    plot.plot(obs_to_array[:, direction_index], np.abs(obs_to_array[:, 3]), marker="o", markersize=2, color=c)

    plot.set_xlabel("external magnetic field")
    plot.set_ylabel("magnetization")
    plot.set_title(
        f"Magnetization vs external field in {direction.flatten()}-direction for ToricCode2d of size={shape}")

    plot.set_xlim(0, field_strengths[-1][direction_index])

    if save_results:
        fig.savefig(f"{save_path}/Magnetizations_L{shape}_{eval_model}_hdir{direction.flatten()}.pdf")
