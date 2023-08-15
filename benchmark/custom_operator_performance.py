import jax.numpy as jnp
import netket as nk

import geneqs

# %% define toric code operator
toric_shape = jnp.array([8, 8])
square_graph = nk.graph.Square(length=toric_shape[0].item(), pbc=True)
toric_hilbert = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)
custom_toric = geneqs.operators.toric_2d.ToricCode2d(toric_hilbert, toric_shape, h=(1, 0, 0))
netket_toric = geneqs.operators.toric_2d.get_netket_toric2dh(toric_hilbert, toric_shape, h=(1, 0, 0))

#%% define checkerboard operator
checkerboard_shape = jnp.array([4, 4, 4])
cube_graph = nk.graph.Hypercube(length=checkerboard_shape[0].item(), n_dim=3, pbc=True)
checkerboard_hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)
custom_checkerboard = geneqs.operators.checkerboard.Checkerboard(checkerboard_hilbert, checkerboard_shape, h=(1, 0, 0))
netket_checkerboard = geneqs.operators.checkerboard.get_netket_checkerboard(checkerboard_hilbert, checkerboard_shape, h=(1, 0, 0))

# %%
# construct some dummy samples
n_samples = 4096
toric_samples = jnp.ones(shape=(n_samples, toric_hilbert.size))
checkerboard_samples = jnp.ones(shape=(n_samples, checkerboard_hilbert.size))

# run everything once so jax compilation is triggered
ctconns, ctmels = custom_toric.get_conn_padded(toric_samples)
ntconns, ntmels = netket_toric.get_conn_padded(toric_samples)
ccconns, ccmels = custom_checkerboard.get_conn_padded(checkerboard_samples)
ncconns, ncmels = netket_checkerboard.get_conn_padded(checkerboard_samples)

# %% get times
_, custom_toric_time = geneqs.utils.benchmarking.time_function(custom_toric.get_conn_padded, 50, toric_samples)
_, netket_toric_time = geneqs.utils.benchmarking.time_function(netket_toric.get_conn_padded, 50, toric_samples)
_, custom_checkerboard_time = geneqs.utils.benchmarking.time_function(custom_checkerboard.get_conn_padded, 50, checkerboard_samples)
_, netket_checkerboard_time = geneqs.utils.benchmarking.time_function(netket_checkerboard.get_conn_padded, 50, checkerboard_samples)

# %%
print(f"custom toric time to calculate conns, mels for {n_samples} samples on shape={toric_shape}: {custom_toric_time} \n"
      f"netket toric time to calculate conns, mels for {n_samples} samples on shape={toric_shape}: {netket_toric_time} \n"
      f"custom checkerboard time to calculate conns, mels for {n_samples} samples on shape={checkerboard_shape}: {custom_checkerboard_time} \n"
      f"netket checkerboard time to calculate conns, mels for {n_samples} samples on shape={checkerboard_shape}: {netket_checkerboard_time} \n")