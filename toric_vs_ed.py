import jax
import jax.numpy as jnp
import optax

import netket as nk
from netket.utils import HashableArray

import geneqs
from geneqs.utils.training import loop_gs, driver_gs
from global_variables import RESULTS_PATH

from matplotlib import pyplot as plt
import numpy as np

from tqdm import tqdm
from functools import partial

save_results = False
pre_train = False

random_key = jax.random.PRNGKey(12344567)  # this can be used to make results deterministic, but so far is not used

# %%
L = 3  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)

# define some observables
magnetization = 1 / hilbert.size * sum([nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size)])
abs_magnetization = geneqs.operators.observables.AbsZMagnetization(hilbert)
wilsonob = geneqs.operators.observables.get_netket_wilsonob(hilbert, shape)

positions = jnp.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
A_B = 1 / hilbert.size * sum([geneqs.operators.toric_2d.get_netket_star(hilbert, p, shape) for p in positions]) - \
      1 / hilbert.size * sum([geneqs.operators.toric_2d.get_netket_plaq(hilbert, p, shape) for p in positions])

# get (specific) symmetries of the model, in our case translations
perms = geneqs.utils.indexing.get_translations_cubical2d(shape, shift=1)
# noinspection PyArgumentList
link_perms = HashableArray(geneqs.utils.indexing.get_linkperms_cubical2d(shape, shift=1))

bl_bonds, lt_bonds, tr_bonds, rb_bonds = geneqs.utils.indexing.get_bonds_cubical2d(shape)
bl_perms, lt_perms, tr_perms, rb_perms = geneqs.utils.indexing.get_bondperms_cubical2d(shape)
# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_plaquettes_cubical2d(shape)),  # plaquette correlators
               HashableArray(geneqs.utils.indexing.get_stars_cubical2d(shape)),
               HashableArray(bl_bonds), HashableArray(lt_bonds), HashableArray(tr_bonds), HashableArray(rb_bonds))

# noinspection PyArgumentList
correlator_symmetries = (HashableArray(jnp.asarray(perms)),  # plaquettes permute like sites
                         HashableArray(jnp.asarray(perms)),
                         HashableArray(bl_perms), HashableArray(lt_perms),
                         HashableArray(tr_perms), HashableArray(rb_perms))
# noinspection PyArgumentList
loops = (HashableArray(geneqs.utils.indexing.get_strings_cubical2d(0, shape)),  # x-string correlators
         HashableArray(geneqs.utils.indexing.get_strings_cubical2d(1, shape)))  # y-string correlators
# noinspection PyArgumentList
loop_symmetries = (HashableArray(geneqs.utils.indexing.get_xstring_perms(shape)),
                   HashableArray(geneqs.utils.indexing.get_ystring_perms(shape)))

# %%  setting hyper-parameters
n_iter = 800
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 512  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 20
n_discard_per_chain = 48  # should be small for using many chains, default is 10% of n_samples
n_expect = n_samples * 12  # number of samples to estimate observables, must be dividable by chunk_size
chunk_size = n_samples
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

RBMSymm = nk.models.RBMSymm(symmetries=link_perms,
                            alpha=alpha,
                            kernel_init=default_kernel_init,
                            hidden_bias_init=default_kernel_init,
                            visible_bias_init=default_kernel_init,
                            param_dtype=complex)

model = cRBM
eval_model = "ToricCRBM"

# create custom update rule
single_rule = nk.sampler.rules.LocalRule()
vertex_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_stars_cubical2d(shape))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(0, shape))
ystring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(1, shape))
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.5, 0.25, 0.125, 0.125),
                                                          [single_rule, vertex_rule, xstring_rule, ystring_rule])

# learning rate scheduling
lr_init = 0.01
lr_end = 0.001
transition_begin = int(n_iter / 3)
transition_steps = int(n_iter / 3)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)

# define fields for which to trian the NQS and get observables
direction = np.array([0., 0., 0.8]).reshape(-1, 1)
field_strengths = (np.linspace(0, 1, 9) * direction).T

field_strengths = np.vstack((field_strengths, np.array([[0., 0., 0.31],
                                                        [0., 0., 0.33],
                                                        [0., 0., 0.35]])))

field_strengths = field_strengths[field_strengths[:, 2].argsort()]

observables = geneqs.utils.eval_obs.ObservableCollector(key_names=("hx", "hy", "hz"))
exact_energies = []

# %%
if pre_train:
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h=(0., 0., 0.))
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    sampler_exact = nk.sampler.ExactSampler(hilbert)
    variational_gs = nk.vqs.MCState(sampler_exact, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    # exact ground state parameters for the 2d toric code, start with just noisy parameters
    random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
    real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, variational_gs.parameters, stddev)
    complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, variational_gs.parameters, stddev)
    gs_params = jax.tree_util.tree_map(lambda real, comp: real + 1j * comp, real_noise, complex_noise)
    # now set the exact parameters, this way noise is only added to all but the non-zero exact params
    plaq_idxs = toric.plaqs[0].reshape(1, -1)
    star_idxs = toric.stars[0].reshape(1, -1)
    exact_weights = jnp.zeros_like(variational_gs.parameters["symm_kernel"], dtype=complex)
    exact_weights = exact_weights.at[0, plaq_idxs].set(1j * jnp.pi / 4)
    exact_weights = exact_weights.at[1, star_idxs].set(1j * jnp.pi / 2)

    gs_params = gs_params.copy({"symm_kernel": exact_weights})
    pretrained_parameters = gs_params
    
    variational_gs.parameters = pretrained_parameters
    print("init energy", variational_gs.expect(toric))

for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
    toric_nk = geneqs.operators.toric_2d.get_netket_toric2dh(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    sampler_exact = nk.sampler.ExactSampler(hilbert)
    vqs_exact_samp = nk.vqs.MCState(sampler_exact, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    variational_gs = nk.vqs.ExactState(hilbert, model)

    if pre_train:
        variational_gs.parameters = pretrained_parameters

    # use driver gs if vqs is exact_state aka full_summation_state
    variational_gs, training_data = driver_gs(variational_gs, toric, optimizer, preconditioner, n_iter, min_iter)

    # calculate observables, therefore set some params of vqs
    # variational_gs.n_samples = n_expect
    # variational_gs.chunk_size = chunk_size

    # calculate energy and specific heat / variance of energy
    energy_nk = variational_gs.expect(toric)
    observables.add_nk_obs("energy", h, energy_nk)
    # exactly diagonalize hamiltonian, find exact E0 and save it
    E0_exact = nk.exact.lanczos_ed(toric, compute_eigenvectors=False)[0]
    exact_energies.append(E0_exact)

    # calculate magnetization
    magnetization_nk = variational_gs.expect(magnetization)
    observables.add_nk_obs("mag", h, magnetization_nk)
    # calculate absolute magnetization
    abs_magnetization_nk = variational_gs.expect(abs_magnetization)
    observables.add_nk_obs("abs_mag", h, abs_magnetization_nk)
    # calcualte wilson loop operator
    wilsonob_nk = variational_gs.expect(wilsonob)
    observables.add_nk_obs("wilson", h, wilsonob_nk)

    # plot and save training data, save observables
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
                 f" pre_train={pre_train}, stddev={stddev}")

    plot.set_xlabel("iterations")
    plot.set_ylabel("energy")

    E0, err = energy_nk.Mean.item().real, energy_nk.Sigma.real
    plot.set_title(f"E0 = {round(E0, 5)} +- {round(err, 5)} using SR with diag_shift={diag_shift_init}"
                   f" down to {diag_shift_end}")
    plot.legend()
    if save_results:
        fig.savefig(
            f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_h{tuple([round(hi, 3) for hi in h])}.pdf")

# %%
exact_energies = np.array(exact_energies).reshape(-1, 1)
if save_results:
    save_array = np.concatenate((observables.obs_to_array(separate_keys=False), exact_energies), axis=1)
    np.savetxt(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_observables.txt", save_array,
               header=", ".join(observables.key_names + observables.obs_names + ["exact_energy"]))

# %%
# create and save relative error plot
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

fields, energies = observables.obs_to_array("energy", separate_keys=True)
rel_errors = np.abs(exact_energies - energies) / np.abs(exact_energies)

plot.plot(fields[:, 2], rel_errors, marker="o", markersize=2)

plot.set_yscale("log")
plot.set_ylim(1e-7, 1e-1)
plot.set_xlabel("external field")
plot.set_ylabel("relative error")
plot.set_title(f"Relative error of {eval_model} for the 2d Toric code vs external field in {direction.flatten()} "
               f"direction on a 3x3 lattice")

plt.show()

if save_results:
    fig.savefig(f"{RESULTS_PATH}/toric2d_h/Relative_Error_L{shape}_{eval_model}_hdir{direction.flatten()}.pdf")