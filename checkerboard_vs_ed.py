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
save_stats = True  # whether to save stats logged during training to drive
save_path = f"{RESULTS_PATH}/checkerboard/vsed_final/vsed_full_hx_independent"
pre_init = False  # True only has effect when sweep=="independent"
sweep = "independent"  # viable options: "independent", "left_right", "right_left"
# if pre_init==True and sweep!="independent", pre_init only applies to the first training run

random_key = jax.random.PRNGKey(1234567)  # this can be used to make results deterministic, but so far is not used

# define fields for which to trian the NQS and get observables
direction_index = 0  # 0 for x, 1 for y, 2 for z;
direction = np.array([0.9, 0., 0.]).reshape(-1, 1)
field_strengths = (np.linspace(0, 1, 10) * direction).T

field_strengths = np.vstack((field_strengths, np.array([[0.15, 0., 0.],
                                                        [0.25, 0., 0.],
                                                        [0.35, 0., 0.],
                                                        [0.45, 0., 0.]])))

# field_strengths[:, [0, 1]] = field_strengths[:, [1, 0]]
print(f"training for field strengths: {field_strengths}")

save_fields = field_strengths  # field values for which vqs is serialized

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
n_iter = 1200
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 512  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 40
n_discard_per_chain = 48  # should be small for using many chains, default is 10% of n_samples
n_expect = n_samples * 12  # number of samples to estimate observables, must be dividable by chunk_size
chunk_size = n_samples

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
transition_steps = int(n_iter * 1 / 5)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

# define correlation enhanced RBM
stddev = 0.01
trans_dev = 0  # previously stddev / 10  # standard deviation for transfer learning noise
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
cRBM = geneqs.models.CheckerLoopCRBM_2(symmetries=perms,
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
eval_model = "CheckerCRBM_2"

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
save_fields = np.round(save_fields, 3)
field_strengths = np.unique(np.round(np.vstack((field_strengths, save_fields)), 3), axis=0)
field_strengths = field_strengths[field_strengths[:, direction_index].argsort()]
if sweep == "right_left":
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
    print(f"training for field={h}")
    checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    sampler_exact = nk.sampler.ExactSampler(hilbert)
    vqs_exact_samp = nk.vqs.MCState(sampler_exact, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    random_key, init_key = jax.random.split(random_key)  # this makes everything deterministic
    vqs_full = nk.vqs.ExactState(hilbert, model, seed=random_key)
    vqs = vqs_full

    if sweep != "independent":
        if last_trained_params is not None:
            random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
            real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, vqs.parameters, trans_dev)
            complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, vqs.parameters, trans_dev)
            vqs.parameters = jax.tree_util.tree_map(lambda ltp, rn, cn: ltp + rn + 1j * cn,
                                                    last_trained_params, real_noise, complex_noise)
        # if last_sampler_state is not None:
        #     vqs.sampler_state = last_sampler_state
        # vqs.sample(chain_length=256)  # let mcmc chains adapt to noisy initial paramters

    if pre_init and sweep == "independent":
        vqs.parameters = pre_init_parameters

    if save_stats:
        out_path = f"{save_path}/stats_L{shape}_{eval_model}_h{tuple([round(hi, 3) for hi in h])}.json"
    else:
        out_path = None
    # use driver gs if vqs is exact_state aka full_summation_state
    vqs, training_data = driver_gs(vqs, checkerboard, optimizer, preconditioner, n_iter, min_iter, out=out_path)
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
                  label=f"Energy")

    fig.suptitle(f" Checkerboard h={tuple([round(hi, 3) for hi in h])}: size={shape} \n"
                 f" {eval_model}, alpha={alpha}, #p={n_params}, lr from {lr_init} to {lr_end} \n"
                 f" n_discard={n_discard_per_chain},"
                 f" n_chains={n_chains},"
                 f" n_samples={n_samples} \n"
                 f" pre_init={pre_init}, stddev={stddev}, trans_dev={trans_dev}, sweep={sweep}")

    plot.set_xlabel("Training Iterations")
    plot.set_ylabel("Observables")

    E0, err = energy_nk.Mean.item().real, energy_nk.Sigma.real
    psi = vqs.to_array()
    full_probabilities = psi * jnp.conjugate(psi) / jnp.sum(psi * jnp.conjugate(psi))
    plot.set_title(f"E0 = {round(E0, 5)} +- {round(err, 5)} using SR with diag_shift={diag_shift_init}"
                   f" down to {diag_shift_end}, max_prob = {jnp.max(full_probabilities)}")
    plot.legend()
    if save_results:
        fig.savefig(
            f"{save_path}/L{shape}_{eval_model}_h{tuple([round(hi, 3) for hi in h])}.pdf")

# %% save results
exact_energies = np.array(exact_energies).reshape(-1, 1)
if save_results:
    save_array = np.concatenate((observables.obs_to_array(separate_keys=False), exact_energies), axis=1)
    np.savetxt(f"{save_path}/L{shape}_{eval_model}_observables.txt", save_array,
               header=" ".join(observables.key_names + observables.obs_names + ["exact_energy"]), comments="")

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
