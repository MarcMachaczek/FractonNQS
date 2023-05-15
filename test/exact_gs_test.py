import jax
import jax.numpy as jnp
import netket as nk
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

# %% Test for the Toric code in two dimensions
L = 3  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)

perms = geneqs.utils.indexing.get_translations_cubical2d(shape, shift=1)
link_perms = HashableArray(geneqs.utils.indexing.get_linkperms_cubical2d(shape, shift=1))

bl_bonds, lt_bonds, tr_bonds, rb_bonds = geneqs.utils.indexing.get_bonds_cubical2d(shape)
bl_perms, lt_perms, tr_perms, rb_perms = geneqs.utils.indexing.get_bondperms_cubical2d(shape)
# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_plaquettes_cubical2d(shape)),  # plaquette correlators
               HashableArray(bl_bonds), HashableArray(lt_bonds), HashableArray(tr_bonds), HashableArray(rb_bonds),
               HashableArray(geneqs.utils.indexing.get_strings_cubical2d(0, shape)),  # x-string correlators
               HashableArray(geneqs.utils.indexing.get_strings_cubical2d(1, shape)))  # y-string correlators)

# noinspection PyArgumentList
correlator_symmetries = (HashableArray(jnp.asarray(perms)),  # plaquettes permute like sites
                         HashableArray(bl_perms), HashableArray(lt_perms),
                         HashableArray(tr_perms), HashableArray(rb_perms),
                         HashableArray(geneqs.utils.indexing.get_xstring_perms(shape)),
                         HashableArray(geneqs.utils.indexing.get_ystring_perms(shape)))

alpha = 1
cRBM = geneqs.models.ToricCRBM(symmetries=link_perms,
                               correlators=correlators,
                               correlator_symmetries=correlator_symmetries,
                               alpha=alpha,
                               kernel_init=default_kernel_init,
                               bias_init=default_kernel_init,
                               param_dtype=complex)

toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
plaq_idx = toric.plaqs[0].reshape(1, -1)
star_idx = toric.stars[0].reshape(1, -1)

single_rule = nk.sampler.rules.LocalRule()
vertex_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_stars_cubical2d(shape))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(0, shape))
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.5, 0.25, 0.25), [single_rule, vertex_rule, xstring_rule])
sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)

vqs = nk.vqs.MCState(sampler, cRBM, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
e_random = vqs.expect(toric)

print(jax.tree_util.tree_map(lambda x: x.shape, vqs.parameters))

# exact ground state parameters for the 2d toric code, start with just noisy parameters
random_key, noise_key_real, noise_key_complex = jax.random.split(random_key, 3)
real_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_real, vqs.parameters, stddev)
complex_noise = geneqs.utils.jax_utils.tree_random_normal_like(noise_key_complex, vqs.parameters, stddev)
gs_params = jax.tree_util.tree_map(lambda real, comp: real + 1j * comp, real_noise, complex_noise)
# now set the exact parameters, this way noise is only added to all but the non-zero exact params
plaq_idxs = toric.plaqs[0].reshape(1, -1)
star_idxs = toric.stars[0].reshape(1, -1)
exact_weights = jnp.zeros_like(vqs.parameters["symm_kernel"], dtype=complex)
exact_weights = exact_weights.at[0, plaq_idxs].set(1j * jnp.pi / 4)
exact_weights = exact_weights.at[1, star_idxs].set(1j * jnp.pi / 2)

# add noise to non-zero parameters
gs_params = gs_params.copy({"symm_kernel": exact_weights})
vqs.parameters = gs_params

samples = vqs.sample().reshape(-1, 2 * L ** 2)

psi, state = cRBM.apply({"params": vqs.parameters}, samples, mutable=["intermediates"])
e_exact = vqs.expect(toric)

# %% Test for the Checkerboard model
L = 4  # this translates to L+1 without PBC
shape = jnp.array([L, L, L])
cube_graph = nk.graph.Hypercube(length=L, n_dim=3, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)

perms = geneqs.utils.indexing.get_translations_cubical3d(shape, shift=2)
# must be hashable to be included as flax.module attribute
# noinspection PyArgumentList
perms = nk.utils.HashableArray(perms.astype(int))

alpha = 1 / 4
cRBM = geneqs.models.CorrelationRBM(symmetries=perms,
                                    correlators=(),
                                    correlator_symmetries=(),
                                    alpha=alpha,
                                    kernel_init=default_kernel_init,
                                    bias_init=default_kernel_init,
                                    param_dtype=complex)

checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
cube_idx = checkerboard.cubes[0].reshape(1, -1)

n_chains = 512 * 2
n_samples = n_chains * 16
n_discard_per_chain = 48 * 16
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)  # TODO: need better sampler

vqs = nk.vqs.MCState(sampler, cRBM, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
e_random = vqs.expect(checkerboard)

print(jax.tree_util.tree_map(lambda x: x.shape, vqs.parameters))

# set exact wights for the rbm, noise avoids inf/nan when applying log_cosh activation
exact_hidden_bias = jnp.zeros_like(vqs.parameters["hidden_bias"]) \
                    + jax.random.normal(jax.random.PRNGKey(0), vqs.parameters["hidden_bias"].shape) * 0.0001
exact_visible_bias = jnp.zeros_like(vqs.parameters["visible_bias"])
exact_weights = jnp.zeros_like(vqs.parameters["symm_kernel"], dtype=complex)
exact_weights = exact_weights.at[0, cube_idx].set(1j * jnp.pi / 4)
exact_weights = exact_weights.at[1, cube_idx].set(1j * jnp.pi / 4)

vqs.parameters = {"hidden_bias": exact_hidden_bias, "symm_kernel": exact_weights, "visible_bias": exact_visible_bias}

samples = vqs.sample().reshape(-1, hilbert.size)

psi, state = cRBM.apply({"params": vqs.parameters}, samples, mutable=["intermediates"])
e_exact = vqs.expect(checkerboard)
