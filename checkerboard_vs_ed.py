import jax
import jax.numpy as jnp
import optax
import flax

import netket as nk
from netket.utils import HashableArray

import geneqs
from geneqs.utils.training import loop_gs, driver_gs
from global_variables import RESULTS_PATH

from matplotlib import pyplot as plt
import matplotlib
import numpy as np

from tqdm import tqdm
from functools import partial

matplotlib.rcParams.update({'font.size': 12})

# %% training configuration
save_results = True
save_path = f"{RESULTS_PATH}/checkerboard"
pre_init = False  # True only has effect when swip=="independent"
swipe = "right_left"  # viable options: "independent", "left_right", "right_left"
# if pre_init==True and swipe!="independent", pre_init only applies to the first training run

random_key = jax.random.PRNGKey(1234567)  # this can be used to make results deterministic, but so far is not used

# define fields for which to trian the NQS and get observables
direction_index = 2  # 0 for x, 1 for y, 2 for z;
direction = np.array([0., 0., 0.7]).reshape(-1, 1)
field_strengths = (np.linspace(0, 1, 8) * direction).T

field_strengths = np.vstack((field_strengths, np.array([[0., 0., 0.42],
                                                        [0., 0., 0.44],
                                                        [0., 0., 0.46]])))

save_fields = np.array([[0., 0, 0.1],
                        [0., 0, 0.43],
                        [0., 0, 0.7]])

# %% operators on hilbert space
shape = jnp.array([4, 2, 2])
hilbert = nk.hilbert.Spin(s=1 / 2, N=jnp.product(shape).item())

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

# %%  setting hyper-parameters
n_iter = 1500
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 512  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 40
n_discard_per_chain = 48  # should be small for using many chains, default is 10% of n_samples
n_expect = n_samples * 12  # number of samples to estimate observables, must be dividable by chunk_size
chunk_size = n_samples

diag_shift_init = 1e-4
diag_shift_end = 1e-6
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
# define correlation enhanced RBM
stddev = 0.01
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

RBMSymm = nk.models.RBMSymm(symmetries=perms,
                            alpha=alpha,
                            kernel_init=default_kernel_init,
                            hidden_bias_init=default_kernel_init,
                            visible_bias_init=default_kernel_init,
                            param_dtype=complex)

model = cRBM
eval_model = "CheckerCRBM"

field_strengths = np.unique(np.round(np.vstack((field_strengths, save_fields)), 3), axis=0)
field_strengths = field_strengths[field_strengths[:, direction_index].argsort()]
if swipe == "right_left":
    field_strengths = field_strengths[::-1]
observables = geneqs.utils.eval_obs.ObservableCollector(key_names=("hx", "hy", "hz"))
exact_energies = []

# %% training
if pre_init:
    checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h=(0., 0., 0.))
    optimizer = optax.sgd(lr_schedule)
    sampler_exact = nk.sampler.ExactSampler(hilbert)
    vqs_exact_samp = nk.vqs.MCState(sampler_exact, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    random_key, init_key = jax.random.split(random_key)  # this makes everything deterministic
    vqs_full = nk.vqs.ExactState(hilbert, model, seed=init_key)
    vqs = vqs_full

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

last_trained_params = None
for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    sampler_exact = nk.sampler.ExactSampler(hilbert)
    vqs_exact_samp = nk.vqs.MCState(sampler_exact, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    random_key, init_key = jax.random.split(random_key)  # this makes everything deterministic
    vqs_full = nk.vqs.ExactState(hilbert, model, seed=random_key)
    vqs = vqs_full

    if swipe != "independent":
        if last_trained_params is not None:
            random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
            real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, vqs.parameters, stddev/10)
            complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, vqs.parameters, stddev/10)
            vqs.parameters = jax.tree_util.tree_map(lambda ltp, r, c: ltp + r + 1j * c,
                                                    last_trained_params, real_noise, complex_noise)

    if pre_init and swipe == "independent":
        vqs.parameters = pre_init_parameters

    # use driver gs if vqs is exact_state aka full_summation_state
    vqs, training_data = driver_gs(vqs, checkerboard, optimizer, preconditioner, n_iter, min_iter)
    last_trained_params = vqs.parameters

    # calculate observables, therefore set some params of vqs
    # vqs.n_samples = n_expect
    # vqs.chunk_size = chunk_size

    # calculate energy and specific heat / variance of energy
    energy_nk = vqs.expect(checkerboard)
    observables.add_nk_obs("energy", h, energy_nk)
    # exactly diagonalize hamiltonian, find exact E0 and save it
    E0_exact = nk.exact.lanczos_ed(checkerboard, compute_eigenvectors=False)[0]
    exact_energies.append(E0_exact)

    # calculate magnetization
    magnetization_nk = vqs.expect(magnetization)
    observables.add_nk_obs("mag", h, magnetization_nk)
    # calculate absolute magnetization
    abs_magnetization_nk = vqs.expect(abs_magnetization)
    observables.add_nk_obs("abs_mag", h, abs_magnetization_nk)

    if np.any((h == save_fields).all(axis=1)) and save_results:
        filename = f"{eval_model}_L{shape}_h{tuple([round(hi, 3) for hi in h])}"
        with open(f"{save_path}/vqs_{filename}.mpack", 'wb') as file:
            file.write(flax.serialization.to_bytes(vqs))
        geneqs.utils.model_surgery.params_to_txt(vqs, f"{save_path}/params_{filename}.txt")

    # plot and save training data, save observables
    fig = plt.figure(dpi=300, figsize=(12, 12))
    plot = fig.add_subplot(111)

    n_params = int(training_data["n_params"].value)
    plot.errorbar(training_data["Energy"].iters, training_data["Energy"].Mean, yerr=training_data["Energy"].Sigma,
                  label=f"{eval_model}, lr_init={lr_init}, #p={n_params}")

    fig.suptitle(f" Checkerboard h={tuple([round(hi, 3) for hi in h])}: size={shape},"
                 f" {eval_model}, alpha={alpha},"
                 f" n_discard={n_discard_per_chain},"
                 f" n_chains={n_chains},"
                 f" n_samples={n_samples} \n"
                 f" pre_init={pre_init}, stddev={stddev}, swipe={swipe}")

    plot.set_xlabel("iterations")
    plot.set_ylabel("energy")

    E0, err = energy_nk.Mean.item().real, energy_nk.Sigma.real
    plot.set_title(f"E0 = {round(E0, 5)} +- {round(err, 5)} using SR with diag_shift={diag_shift_init}"
                   f" down to {diag_shift_end}")
    plot.legend()
    if save_results:
        fig.savefig(
            f"{save_path}/L{shape}_{eval_model}_h{tuple([round(hi, 3) for hi in h])}.pdf")

# %% save results
exact_energies = np.array(exact_energies).reshape(-1, 1)
if save_results:
    save_array = np.concatenate((observables.obs_to_array(separate_keys=False), exact_energies), axis=1)
    np.savetxt(f"{save_path}/L{shape}_{eval_model}_observables.txt", save_array,
               header=", ".join(observables.key_names + observables.obs_names + ["exact_energy"]))

# %% create and save relative error plot
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

fields, energies = observables.obs_to_array("energy", separate_keys=True)
rel_errors = np.abs(exact_energies - energies) / np.abs(exact_energies)

plot.plot(fields[:, direction_index], rel_errors, marker="o", markersize=2)

plot.set_yscale("log")
plot.set_ylim(1e-7, 1e-1)
plot.set_xlabel("external field")
plot.set_ylabel("relative error")
plot.set_title(f"Relative error of {eval_model} for the checkerboard model vs external field in {direction.flatten()} "
               f"direction on a {shape} lattice")

plt.show()

if save_results:
    fig.savefig(f"{save_path}/Relative_Error_L{shape}_{eval_model}_hdir{direction.flatten()}.pdf")
