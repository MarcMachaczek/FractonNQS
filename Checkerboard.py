import jax
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
L = 4  # this translates to L+1 without PBC
shape = jnp.array([L, L, L])
cube_graph = nk.graph.Hypercube(length=L, n_dim=3, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)

# define some observables
magnetization = 1 / hilbert.size * sum([nk.operator.spin.sigmax(hilbert, i) for i in range(hilbert.size)])
abs_magnetization = geneqs.operators.observables.AbsXMagnetization(hilbert)

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(projection='3d')
geneqs.utils.plotting.plot_checkerboard(ax, L)
plt.show()

perms = geneqs.utils.indexing.get_translations_cubical3d(shape, shift=2)
perms = nk.utils.HashableArray(perms.astype(int))

# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_cubes_cubical3d(shape, 2)),
               HashableArray(geneqs.utils.indexing.get_bonds_cubical3d(shape)))
# noinspection PyArgumentList
correlators_symmetries = (HashableArray(geneqs.utils.indexing.get_cubeperms_cubical3d(shape, 2)),
                          HashableArray(geneqs.utils.indexing.get_bondperms_cubical3d(shape, 2)))
# noinspection PyArgumentList
loops = (HashableArray(geneqs.utils.indexing.get_strings_cubical3d(0, shape)),
         HashableArray(geneqs.utils.indexing.get_strings_cubical3d(1, shape)),
         HashableArray(geneqs.utils.indexing.get_strings_cubical3d(2, shape)))
# noinspection PyArgumentList
loop_symmetries = (HashableArray(geneqs.utils.indexing.get_xstring_perms3d(shape, 2)),
                   HashableArray(geneqs.utils.indexing.get_ystring_perms3d(shape, 2)),
                   HashableArray(geneqs.utils.indexing.get_zstring_perms3d(shape, 2)))

# %%  setting hyper-parameters
n_iter = 1000
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 512 * 1  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 8
n_discard_per_chain = 12  # should be small for using many chains, default is 10% of n_samples
chunk_size = 1024 * 8  # doesn't work for gradient operations, need to check why!
n_expect = chunk_size * 12  # number of samples to estimate observables, must be dividable by chunk_size
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

# define correlation enhanced RBM
stddev = 0.008
default_kernel_init = jax.nn.initializers.normal(stddev)

alpha = 1 / 4
cRBM = geneqs.models.CheckerLoopCRBM(symmetries=perms,
                                     correlators=correlators,
                                     correlator_symmetries=correlators_symmetries,
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
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.7, 0.25, 0.05, 0.05, 0.05),
                                                          [single_rule,
                                                           cube_rule,
                                                           xstring_rule,
                                                           ystring_rule,
                                                           zstring_rule])
# learning rate scheduling
lr_init = 0.01
lr_end = 0.001
transition_begin = int(n_iter / 3)
transition_steps = int(n_iter / 3)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

# define fields for which to trian the NQS and get observables
direction = np.array([0.8, 0., 0.]).reshape(-1, 1)
field_strengths = (np.linspace(0, 1, 20) * direction).T
# field_strengths = np.vstack((field_strengths, np.array([[0.31, 0, 0],
#                                                         [0.32, 0, 0],
#                                                         [0.33, 0, 0],
#                                                         [0.34, 0, 0],
#                                                         [0.35, 0, 0]])))
# for which fields indices histograms are created
hist_fields = np.array([[0.3, 0, 0.],
                        [0.43, 0, 0.],
                        [0.44, 0, 0.],
                        [0.6, 0, 0.]])
# make sure hist fields are contained in field_strengths and sort final field array
field_strengths = np.unique(np.round(np.vstack((field_strengths, hist_fields)), 3), axis=0)

field_strengths = field_strengths[field_strengths[:, 0].argsort()]

observables = geneqs.utils.eval_obs.ObservableCollector(key_names=("hx", "hy", "hz"))

# %%
if pre_train:
    checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h=(0., 0., 0.))
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    # exact ground state parameters for the 2d toric code, start with just noisy parameters
    random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
    real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, variational_gs.parameters, stddev)
    complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, variational_gs.parameters, stddev)
    gs_params = jax.tree_util.tree_map(lambda real, comp: real + 1j * comp, real_noise, complex_noise)
    # now set the exact parameters, this way noise is only added to all but the non-zero exact params

    gs_params = gs_params.copy({"symm_kernel": exact_weights})
    pretrained_parameters = gs_params

    variational_gs.parameters = pretrained_parameters
    print("init energy", variational_gs.expect(checkerboard))

for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    if pre_train:
        variational_gs.parameters = pretrained_parameters

    variational_gs, training_data = loop_gs(variational_gs, checkerboard, optimizer, preconditioner, n_iter, min_iter)

    # calculate observables, therefore set some params of vqs
    variational_gs.chunk_size = chunk_size
    variational_gs.n_samples = n_expect

    # calculate energy and specific heat / variance of energy
    energy_nk = variational_gs.expect(checkerboard)
    observables.add_nk_obs("energy", h, energy_nk)
    # calculate magnetization
    magnetization_nk = variational_gs.expect(magnetization)
    observables.add_nk_obs("mag", h, magnetization_nk)
    # calculate absolute magnetization
    abs_magnetization_nk = variational_gs.expect(abs_magnetization)
    observables.add_nk_obs("abs_mag", h, abs_magnetization_nk)

    if np.any((h == hist_fields).all(axis=1)):
        variational_gs.n_samples = n_samples
        # calculate histograms, CAREFUL: if run with mpi, local_estimators produces rank-dependent output!
        e_locs = np.asarray((variational_gs.local_estimators(checkerboard)), dtype=np.float64)
        observables.add_hist("energy", h, np.histogram(e_locs / hilbert.size, n_bins, density=False))

        mag_locs = np.asarray((variational_gs.local_estimators(magnetization)), dtype=np.float64)
        observables.add_hist("mag", h, np.histogram(mag_locs, n_bins, density=False))

        abs_mag_locs = np.asarray((variational_gs.local_estimators(abs_magnetization)), dtype=np.float64)
        observables.add_hist("abs_mag", h, np.histogram(abs_mag_locs, n_bins, density=False))

    # plot and save training data
    fig = plt.figure(dpi=300, figsize=(12, 12))
    plot = fig.add_subplot(111)

    n_params = int(training_data["n_params"].value)
    plot.errorbar(training_data["energy"].iters, training_data["energy"].Mean, yerr=training_data["energy"].Sigma,
                  label=f"{eval_model}, lr_init={lr_init}, #p={n_params}")

    fig.suptitle(f" Checkerboard h={tuple([round(hi, 3) for hi in h])}: size={shape},"
                 f" {eval_model}, alpha={alpha},"
                 f" n_discard={n_discard_per_chain},"
                 f" n_chains={n_chains},"
                 f" n_samples={n_samples} \n"
                 f" pre_train={pre_train}, stddev={stddev}")

    plot.set_xlabel("iterations")
    plot.set_ylabel("energy")

    E0, err = energy_nk.Mean.item().real, energy_nk.Sigma.item().real
    plot.set_title(f"E0 = {round(E0, 5)} +- {round(err, 5)} using SR with diag_shift={diag_shift_init}"
                   f" down to {diag_shift_end}")
    plot.legend()
    if save_results:
        fig.savefig(
            f"{RESULTS_PATH}/checkerboard/L{shape}_{eval_model}_h{tuple([round(hi, 3) for hi in h])}.pdf")

# %%
if save_results:
    save_array = observables.obs_to_array(separate_keys=False)
    np.savetxt(f"{RESULTS_PATH}/checkerboard/L{shape}_{eval_model}_observables.txt", save_array,
               header=", ".join(observables.key_names + observables.obs_names))

    for hist_name, _ in observables.histograms.items():
        np.save(f"{RESULTS_PATH}/checkerboard/hists_{hist_name}_L{shape}_{eval_model}.npy",
                observables.hist_to_array(hist_name))

# %%
# create and save magnetization plot
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)
c = "red"

fields, mags = observables.obs_to_array(["mag", "mag_var"])

for field, mag in zip(fields, mags):
    plot.errorbar(field[0], np.abs(mag[0]), yerr=mag[1], marker="o", markersize=2, color=c)

plot.plot(fields[:, 0], np.abs(mags[:, 0]), marker="o", markersize=2, color=c)

plot.set_xlabel("external magnetic field")
plot.set_ylabel("magnetization")
plot.set_title(f"Magnetization vs external field in {direction.flatten()}-direction for Checkerboard of size={shape}")

plot.set_xlim(0, field_strengths[-1][0])

if save_results:
    fig.savefig(f"{RESULTS_PATH}/checkerboard/Magnetizations_L{shape}_{eval_model}_hdir{direction.flatten()}.pdf")
