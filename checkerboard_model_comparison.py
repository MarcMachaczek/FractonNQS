import jax
import jax.numpy as jnp
import optax

import netket as nk
from netket.utils import HashableArray

import geneqs
from geneqs.utils.training import driver_gs
from global_variables import RESULTS_PATH

from matplotlib import pyplot as plt
import matplotlib

import numpy as np

from tqdm import tqdm
from functools import partial

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 24})

# %% training configuration
save_results = False
save_path = f"{RESULTS_PATH}/checkerboard"
# if pre_init==True and swipe!="independent", pre_init only applies to the first training run

random_key = jax.random.PRNGKey(144567)  # this can be used to make results deterministic, but so far is not used

# %% operators on hilbert space
shape = jnp.array([4, 2, 2])
hilbert = nk.hilbert.Spin(s=1 / 2, N=jnp.product(shape).item())
h = (0., 0., 0.)
checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
# exactly diagonalize hamiltonian, find exact E0 and save it
E0_exact = - jnp.prod(shape)

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
stddev = 0.1
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
                                     correlators=(correlators[0],),
                                     correlator_symmetries=(correlator_symmetries[0],),
                                     loops=(),
                                     loop_symmetries=(),
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

RBM = nk.models.RBM(alpha=alpha,
                    kernel_init=default_kernel_init,
                    hidden_bias_init=default_kernel_init,
                    visible_bias_init=default_kernel_init,
                    param_dtype=complex)

features = (2, 4)  # first number sets the invariant features
SymmNN = geneqs.models.neural_networks.SymmetricNN(symmetries=perms,
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
          "cRBM": cRBM}

observables = geneqs.utils.eval_obs.ObservableCollector(key_names="eval_model")

# %% training
training_data = {}
for eval_model, model in tqdm(models.items()):
    sampler_mc = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    vqs_mc = nk.vqs.MCState(sampler_mc, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    if L <= 3:
        sampler_exact = nk.sampler.ExactSampler(hilbert)
        vqs_exact_samp = nk.vqs.MCState(sampler_exact, model, n_samples=n_samples,
                                        n_discard_per_chain=n_discard_per_chain)
        random_key, init_key = jax.random.split(random_key)  # this makes everything deterministic
        vqs_full = nk.vqs.ExactState(hilbert, model, seed=init_key)
    vqs = vqs_full

    # use driver gs if vqs is exact_state aka full_summation_state
    vqs, data = driver_gs(vqs, checkerboard, optimizer, preconditioner, n_iter, min_iter)
    training_data[f"{eval_model}"] = data

    # calculate observables, therefore set some params of vqs
    # vqs.n_samples = n_expect
    # vqs.chunk_size = chunk_size

    energy_nk = vqs.expect(checkerboard)
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


fig.suptitle(f" Checkerboard h={tuple([round(hi, 3) for hi in h])}: size={shape},"
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
