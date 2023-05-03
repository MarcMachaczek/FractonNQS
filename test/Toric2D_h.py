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
L = 3  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)
magnetization = geneqs.operators.observables.Magnetization(hilbert)

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

direction = np.array([1, 0, 1]).reshape(-1, 1)
field_strengths = (np.linspace(0, 1, 22) * direction).T

observables = {}

# %%  setting hyper-parameters
n_iter = 400
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 512 * 1  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 16
n_discard_per_chain = 24  # should be small for using many chains, default is 10% of n_samples
chunk_size = 1024 * 8  # doesn't work for gradient operations, need to check why!
n_expect = chunk_size * 12  # number of samples to estimate observables, must be dividable by chunk_size
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.0001
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense, diag_shift=diag_shift, )  # holomorphic=True)

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
                               param_dtype=float)

model = cRBM
eval_model = "ToricCRBM"

# create custom update rule
single_rule = nk.sampler.rules.LocalRule()
vertex_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_stars_cubical2d(shape))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(0, shape))
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.6, 0.2, 0.2), [single_rule, vertex_rule, xstring_rule])

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
    gs_params = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), variational_gs.parameters)
    plaq_idxs = toric.plaqs[0].reshape(1, -1)
    star_idxs = toric.stars[0].reshape(1, -1)
    exact_weights = jnp.zeros_like(variational_gs.parameters["symm_kernel"], dtype=complex)
    exact_weights = exact_weights.at[0, plaq_idxs].set(1j * jnp.pi / 4)
    exact_weights = exact_weights.at[1, star_idxs].set(1j * jnp.pi / 2)
    gs_params = gs_params.copy({"symm_kernel": exact_weights})
    pretrained_parameters = gs_params

    # variational_gs, training_data = loop_gs(variational_gs, toric, optimizer, preconditioner, n_iter, min_iter)
    # pretrained_parameters = variational_gs.parameters

    print("\n pre-training finished")

for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
    optimizer = optax.sgd(lr_schedule)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)
    sampler_exact = nk.sampler.ExactSampler(hilbert)
    variational_gs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    if pre_train:
        noise_generator = jax.nn.initializers.normal(0.1 * stddev)
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

    # exactly diagonalize hamiltonian, find exact E0 and save it 
    toric_nk = geneqs.operators.toric_2d.get_netket_toric2dh(hilbert, shape, h)
    E0_exact = nk.exact.lanczos_ed(toric_nk, compute_eigenvectors=False)[0]
    observables[h]["energy_exact"] = E0_exact

    # calculate magnetization
    observables[h]["mag"] = variational_gs.expect(magnetization)

    # plot and save training data
    fig = plt.figure(dpi=300, figsize=(10, 10))
    plot = fig.add_subplot(111)

    n_params = int(training_data["n_params"].value)
    plot.errorbar(training_data["energy"].iters, training_data["energy"].Mean, yerr=training_data["energy"].Sigma,
                  label=f"{eval_model}, lr_init={lr_init}, #p={n_params}")

    E0 = observables[h]["energy"].Mean.item().real
    err = observables[h]["energy"].Sigma.item().real

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
                        [obs["energy"].Mean.item().real, obs["energy"].Sigma.item().real])
obs_to_array = np.asarray(obs_to_array)

if save_results:
    np.savetxt(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_observables", obs_to_array,
               header="hx, hy, hz, mag, mag_var, energy, energy_var")
# mags = np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L{shape}_{eval_model}_a{alpha}_magvals")

# %%
# create and save relative error plot
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

rel_errors = np.asarray([np.abs(observables[tuple(h)]["energy_exact"] - observables[tuple(h)]["energy"]) / np.abs(observables[tuple(h)]["energy_exact"])] for h in field_strengths)

plot.plot(obs_to_array[:, 2], rel_errors, marker="o", markersize=2)

plot.set_xlabel("external field")
plot.set_ylabel("relative error")
plot.set_title(f"Relative error of cRBM for the 2d Toric code vs external field on a 3x3 lattice")

plt.show()

if save_results:
    fig.savefig(f"{RESULTS_PATH}/toric2d_h/Relative_Error_{eval_model}_a{alpha}.pdf")
