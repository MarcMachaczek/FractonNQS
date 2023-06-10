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
matplotlib.rcParams.update({'font.size': 12})

import numpy as np

from tqdm import tqdm
from functools import partial

# %% training configuration
save_results = True
save_path = f"{RESULTS_PATH}/toric2d_h"
# if pre_init==True and swipe!="independent", pre_init only applies to the first training run

random_key = jax.random.PRNGKey(144567)  # this can be used to make results deterministic, but so far is not used

# %% operators on hilbert space
L = 5  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)
h = (0., 0., 0.)
toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
# exactly diagonalize hamiltonian, find exact E0 and save it
E0_exact = - L**2 * 2

# %%  setting hyper-parameters
n_iter = 200
min_iter = n_iter  # after min_iter training can be stopped by callback (e.g. due to no improvement of gs energy)
n_chains = 256  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
n_samples = n_chains * 32
n_discard_per_chain = 20  # should be small for using many chains, default is 10% of n_samples
n_expect = n_samples * 12  # number of samples to estimate observables, must be dividable by chunk_size
chunk_size = n_samples

diag_shift_init = 1e-4
diag_shift_end = 1e-5
diag_shift_begin = int(n_iter / 3)
diag_shift_steps = int(n_iter / 3)
diag_shift_schedule = optax.linear_schedule(diag_shift_init, diag_shift_end, diag_shift_steps, diag_shift_begin)

preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense,
                                 solver=partial(jax.scipy.sparse.linalg.cg, tol=1e-6),
                                 diag_shift=diag_shift_schedule,
                                 holomorphic=False)

# learning rate scheduling
lr_init = 0.01
lr_end = 0.01
transition_begin = int(n_iter / 3)
transition_steps = int(n_iter / 3)
lr_schedule = optax.linear_schedule(lr_init, lr_end, transition_steps, transition_begin)
optimizer = optax.sgd(lr_schedule)

# create custom update rule
single_rule = nk.sampler.rules.LocalRule()
vertex_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_stars_cubical2d(shape))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(0, shape))
ystring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(1, shape))
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.5, 0.25, 0.125, 0.125),
                                                          [single_rule, vertex_rule, xstring_rule, ystring_rule])

# define correlation enhanced RBM
stddev = 0.1
default_kernel_init = jax.nn.initializers.normal(stddev)

# get (specific) symmetries of the model, in our case translations
perms = geneqs.utils.indexing.get_translations_cubical2d(shape, shift=1)
# noinspection PyArgumentList
link_perms = HashableArray(geneqs.utils.indexing.get_linkperms_cubical2d(shape, shift=1))

bl_bonds, lt_bonds, tr_bonds, rb_bonds = geneqs.utils.indexing.get_bonds_cubical2d(shape)
bl_perms, lt_perms, tr_perms, rb_perms = geneqs.utils.indexing.get_bondperms_cubical2d(shape)
# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_plaquettes_cubical2d(shape)),  # plaquette correlators,
               HashableArray(bl_bonds), HashableArray(lt_bonds), HashableArray(tr_bonds), HashableArray(rb_bonds))

# noinspection PyArgumentList
correlator_symmetries = (HashableArray(jnp.asarray(perms)),  # plaquettes permute like sites,
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
                                   correlators=(correlators[0],),
                                   correlator_symmetries=(correlator_symmetries[0],),
                                   loops=(),
                                   loop_symmetries=(),
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

RBM = nk.models.RBM(alpha=alpha,
                    kernel_init=default_kernel_init,
                    hidden_bias_init=default_kernel_init,
                    visible_bias_init=default_kernel_init,
                    param_dtype=complex)

features = (2, 4)  # first number sets the invariant features
SymmNN = geneqs.models.neural_networks.SymmetricNN(symmetries=link_perms,
                                                   features=features,
                                                   kernel_init=default_kernel_init,
                                                   bias_init=default_kernel_init,
                                                   param_dtype=complex)
FFNN = geneqs.models.neural_networks.SimpleNN(features=features,
                                              kernel_init=default_kernel_init,
                                              bias_init=default_kernel_init,
                                              param_dtype=complex)

models = {f"FFNN_f{features}": FFNN,
          "RBM": RBM,
          "RBMSymm": RBMSymm,
          f"SymmNN_f{features}": SymmNN,
          "ToricCRBM": cRBM}

observables = geneqs.utils.eval_obs.ObservableCollector(key_names="eval_model")

# %% training
training_data = {}
for eval_model, model in tqdm(models.items()):
    sampler_mc = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    vqs_mc = nk.vqs.MCState(sampler_mc, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    if L <= 3:
        sampler_exact = nk.sampler.ExactSampler(hilbert)
        vqs_exact_samp = nk.vqs.MCState(sampler_exact, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
        random_key, init_key = jax.random.split(random_key)  # this makes everything deterministic
        vqs_full = nk.vqs.ExactState(hilbert, model, seed=init_key)
    vqs = vqs_mc

    # use driver gs if vqs is exact_state aka full_summation_state
    vqs, data = driver_gs(vqs, toric, optimizer, preconditioner, n_iter, min_iter)
    training_data[f"{eval_model}"] = data

    # calculate observables, therefore set some params of vqs
    # vqs.n_samples = n_expect
    # vqs.chunk_size = chunk_size

    energy_nk = vqs.expect(toric)
    observables.add_nk_obs("energy", (eval_model,), energy_nk)

# %% plot and save training data, save observables
fig = plt.figure(dpi=300, figsize=(12, 12))
plot = fig.add_subplot(111)

obs = observables.observables
for eval_model, data in training_data.items():
    n_params = int(data["n_params"].value)
    E0, var = round(obs["energy"][(eval_model,)], 4), round(obs["energy_var"][(eval_model,)], 4)
    rel_error = np.abs(E0 - E0_exact) / np.abs(E0)
    rel_error = '{:.2e}'.format(rel_error)
    plot.errorbar(data["Energy"].iters, data["Energy"].Mean, yerr=data["Energy"].Sigma,
                  label=f"{eval_model}, #p={n_params}, E={E0}+-{var}, delta= {rel_error}")


fig.suptitle(f" ToricCode2d h={tuple([round(hi, 3) for hi in h])}: size={shape},"
             f" n_discard={n_discard_per_chain},"
             f" n_chains={n_chains},"
             f" n_samples={n_samples} \n"
             f" using SR with diag_shift={diag_shift_init} down to {diag_shift_end}"
             f" and lr from {lr_init} to {lr_end}")

plot.set_xlabel("iterations")
plot.set_ylabel("energy")

plot.legend()
plt.show()
if save_results:
    fig.savefig(
        f"{save_path}/L{shape}_comparison_h{tuple([round(hi, 3) for hi in h])}.pdf")
