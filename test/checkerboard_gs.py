import jax
import jax.numpy as jnp
import netket as nk
import optax
from netket.utils import HashableArray

import geneqs

stddev = 0.0001
default_kernel_init = jax.nn.initializers.normal(stddev)
random_key = jax.random.PRNGKey(420)
# some common parameters
n_chains = 512
n_samples = n_chains * 8
n_discard_per_chain = 48

h = (0., 0., 0.)  # external field

# %% Test for the Checkerboard model
L = 4  # this translates to L+1 without PBC
shape = jnp.array([L, L, L])
cube_graph = nk.graph.Hypercube(length=L, n_dim=3, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)

perms = geneqs.utils.indexing.get_translations_cubical3d(shape, shift=2)
# noinspection PyArgumentList
perms = nk.utils.HashableArray(perms.astype(int))

alpha = 1
cRBM = geneqs.models.CorrelationRBM(symmetries=perms,
                                    correlators=(),
                                    correlator_symmetries=(),
                                    alpha=alpha,
                                    kernel_init=default_kernel_init,
                                    bias_init=default_kernel_init,
                                    param_dtype=complex)

checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
nk_checkerboard = geneqs.operators.checkerboard.get_netket_checkerboard(hilbert, shape, h)

n_chains = 256
n_samples = n_chains * 32
n_discard_per_chain = 48 * 32

single_rule = nk.sampler.rules.LocalRule()

cube_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_cubes_cubical3d(shape, shift=2))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(0, shape))
ystring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(1, shape))
zstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(2, shape))
# noinspection PyArgumentList
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.8, 0.15, 0.05, 0.05, 0.05),
                                                          [single_rule,
                                                           cube_rule,
                                                           xstring_rule,
                                                           ystring_rule,
                                                           zstring_rule])
sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)

vqs = nk.vqs.MCState(sampler, cRBM, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
e_random = vqs.expect(nk_checkerboard)

print(jax.tree_util.tree_map(lambda x: x.shape, vqs.parameters))

# exact ground state parameters for the checkerboard model, start with just noisy parameters
random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, vqs.parameters, stddev)
complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, vqs.parameters, stddev)
gs_params = jax.tree_util.tree_map(lambda real, comp: real + 1j * comp, real_noise, complex_noise)
# now set the exact parameters, this way noise is only added to all but the non-zero exact params
cube_idx = checkerboard.cubes[jnp.array([0, 2, 8, 10])]
exact_weights = jnp.zeros_like(vqs.parameters["symm_kernel"], dtype=complex)
exact_weights = exact_weights.at[jnp.arange(4).reshape(-1, 1), cube_idx].set(1j * jnp.pi / 4)
exact_weights = exact_weights.at[(jnp.arange(4)+4).reshape(-1, 1), cube_idx].set(1j * jnp.pi / 4)

# add noise to non-zero parameters
gs_params = gs_params.copy({"symm_kernel": exact_weights})
vqs.parameters = gs_params

samples = vqs.sample().reshape(-1, hilbert.size)

e_exact = vqs.expect(nk_checkerboard)
print(e_exact)

# %% Test for the Checkerboard model
L = 4  # this translates to L+1 without PBC
shape = jnp.array([L, L, L])
cube_graph = nk.graph.Hypercube(length=L, n_dim=3, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)

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

alpha = 1 / 2
cRBM = geneqs.models.CheckerLoopCRBM(symmetries=perms,
                                     correlators=correlators,
                                     correlator_symmetries=correlators_symmetries,
                                     loops=loops,
                                     loop_symmetries=loop_symmetries,
                                     alpha=alpha,
                                     kernel_init=default_kernel_init,
                                     bias_init=default_kernel_init,
                                     param_dtype=complex)

checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)

n_chains = 256
n_samples = n_chains * 32
n_discard_per_chain = 24 * 20

# careful with sampling, many chains get stuck if starting from random configurations and exact gs
# solution 1: actually train a vqs and wait till converged, use these samples to check exact parametrization
# solution 2: don't start from random but somehow ordered states, but must be careful with that too
single_rule = nk.sampler.rules.LocalRule()
cube_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_cubes_cubical3d(shape, shift=2))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(0, shape))
ystring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(1, shape))
zstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(2, shape))
# noinspection PyArgumentList
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.5, 0.5),
                                                          [single_rule,
                                                           cube_rule])
sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)

vqs = nk.vqs.MCState(sampler, cRBM, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
# e_random = vqs.expect(checkerboard)

# %%
preconditioner = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianDense, holomorphic=True)
optimizer = optax.sgd(0.01)
trained_vqs, training_data = geneqs.utils.training.loop_gs(vqs, checkerboard, optimizer, preconditioner, n_iter=150)
trained_params = trained_vqs.parameters

# %%
print(jax.tree_util.tree_map(lambda x: x.shape, vqs.parameters))

stddev = 0.00001
# exact ground state parameters for the checkerboard model, start with just noisy parameters
random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, vqs.parameters, stddev)
complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, vqs.parameters, stddev)
gs_params = jax.tree_util.tree_map(lambda real, comp: real + 1j * comp, real_noise, complex_noise)
# now set the exact parameters, this way noise is only added to all but the non-zero exact params
cube_idx = jnp.array([0, 2, 8, 10])
exact_weights = jnp.zeros_like(vqs.parameters["corr0_kernel"], dtype=complex)
exact_weights = exact_weights.at[jnp.arange(4), cube_idx].set(- 1j * jnp.pi / 4)
exact_bias = jnp.zeros_like(vqs.parameters["hidden_bias"], dtype=complex)
exact_bias = exact_bias.at[jnp.arange(4)].set(1j * jnp.pi / 4)

# add noise to non-zero parameters
gs_params = gs_params.copy({"corr0_kernel": exact_weights})
gs_params = gs_params.copy({"hidden_bias": exact_bias})
vqs.parameters = gs_params

e_exact = vqs.expect(checkerboard)
print(e_exact)

# %%
sigmas = jnp.ones((2, hilbert.size))
sigmas = sigmas.at[1, 0].set(-1)
conns, mels = checkerboard.get_conn_padded(sigmas)

jnp.exp(vqs.log_value(conns[0]))
