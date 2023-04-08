import jax
import jax.numpy as jnp
import netket as nk

import geneqs

stddev = 0.05
default_kernel_init = jax.nn.initializers.normal(stddev)

# %%
L = 6  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)

perms = geneqs.utils.indexing.get_translations_cubical2d(shape)
link_perms = geneqs.utils.indexing.get_linkperms_cubical2d(perms)
link_perms = nk.utils.HashableArray(link_perms.astype(int))

alpha = 0.5
cRBM = geneqs.models.CorrelationRBM(symmetries=link_perms,
                                    correlators=(),
                                    correlator_symmetries=(),
                                    alpha=alpha,
                                    kernel_init=default_kernel_init,
                                    bias_init=default_kernel_init,
                                    param_dtype=complex)

h = (0., 0., 0.)
toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
plaq_idx = toric.plaqs[0].reshape(1, -1)
toric = geneqs.operators.toric_2d.get_netket_toric2dh(hilbert, shape, h)

n_chains = 512
n_samples = n_chains * 8
n_discard_per_chain = 12
sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains, dtype=jnp.int8)

vqs = nk.vqs.MCState(sampler, cRBM, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
e_random = vqs.expect(toric)

print(jax.tree_util.tree_map(lambda x: x.shape, vqs.parameters))
# needed to avoid inf when applying log_cosh activation
exact_hidden_bias = jnp.zeros_like(vqs.parameters["hidden_bias"])\
                    + jax.random.normal(jax.random.PRNGKey(0), vqs.parameters["hidden_bias"].shape) * 0.00000001
exact_visible_bias = jnp.zeros_like(vqs.parameters["visible_bias"])
exact_weights = jnp.zeros_like(vqs.parameters["symm_kernel"], dtype=complex)
exact_weights = exact_weights.at[0, plaq_idx].set(1j * jnp.pi/4)

vqs.parameters = {"hidden_bias": exact_hidden_bias, "symm_kernel": exact_weights, "visible_bias": exact_visible_bias}

samples = vqs.sample().reshape(-1, 2*L**2)

psi, state = cRBM.apply({"params": vqs.parameters}, samples, mutable=["intermediates"])
e_exact = vqs.expect(toric)

