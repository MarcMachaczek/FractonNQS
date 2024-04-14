import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import netket as nk
import geneqs
from global_variables import BENCHMARK_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 30})

toric_shapes = []  # jnp.array([[4, 4], [6, 6], [8, 8]])
checkerboard_shapes = jnp.array([[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]])
n_samples = 2048 * 2

# custom_toric_times = []
# netket_toric_times = []
# for toric_shape in toric_shapes:
#     # define toric code operator
#     square_graph = nk.graph.Square(length=toric_shape[0].item(), pbc=True)
#     toric_hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)
#     custom_toric = geneqs.operators.toric_2d.ToricCode2d(toric_hilbert, toric_shape, h=(1, 0, 0))
#     netket_toric = geneqs.operators.toric_2d.get_netket_toric2dh(toric_hilbert, toric_shape, h=(1, 0, 0))

#     # construct some dummy samples
#     toric_samples = jnp.ones(shape=(n_samples, toric_hilbert.size))

#     # run everything once so jax compilation is triggered
#     ctconns, ctmels = custom_toric.get_conn_padded(toric_samples)
#     ntconns, ntmels = netket_toric.get_conn_padded(toric_samples)

#     # get times
#     custom_time = geneqs.utils.benchmarking.time_function(custom_toric.get_conn_padded, 40, toric_samples)
#     netket_time = geneqs.utils.benchmarking.time_function(netket_toric.get_conn_padded, 40, toric_samples)

#     custom_toric_times.append(custom_time)
#     netket_toric_times.append(netket_time)

custom_checkerboard_times = []
netket_checkerboard_times = []
for checkerboard_shape in checkerboard_shapes:
    # define checkerboard operator
    cube_graph = nk.graph.Hypercube(length=checkerboard_shape[0].item(), n_dim=3, pbc=True)
    checkerboard_hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)
    custom_checkerboard = geneqs.operators.checkerboard.Checkerboard(checkerboard_hilbert, checkerboard_shape,
                                                                     h=(1, 0, 0))
    netket_checkerboard = geneqs.operators.checkerboard.get_netket_checkerboard(checkerboard_hilbert,
                                                                                checkerboard_shape, h=(1, 0, 0))

    # construct some dummy samples
    checkerboard_samples = jnp.ones(shape=(n_samples, checkerboard_hilbert.size))

    # run everything once so jax compilation is triggered
    a, b = custom_checkerboard.get_conn_padded(checkerboard_samples)
    del a, b
    a, b = netket_checkerboard.get_conn_padded(checkerboard_samples)
    del a, b

    # %% get times
    custom_time = geneqs.utils.benchmarking.time_function(custom_checkerboard.get_conn_padded, 100,
                                                             checkerboard_samples)
    netket_time = geneqs.utils.benchmarking.time_function(netket_checkerboard.get_conn_padded, 100,
                                                             checkerboard_samples)

    custom_checkerboard_times.append(custom_time)
    netket_checkerboard_times.append(netket_time)

fig = plt.figure(dpi=300, figsize=(20, 10))
# toric_plot = fig.add_subplot(121)
# toric_plot.scatter(toric_shapes[:, 0], custom_toric_times, label="custom operator", c="green")
# toric_plot.scatter(toric_shapes[:, 0], netket_toric_times, label="NetKet operator", c="red")
# toric_plot.legend()
# toric_plot.set_xlabel("linear system size \$L\$")
# toric_plot.set_ylabel("time in ns")
# toric_plot.set_title("2d Toric Code")

checkerboard_plot = fig.add_subplot(111)
checkerboard_plot.scatter(checkerboard_shapes[:, 0], custom_checkerboard_times, label="Custom", c="green", s=16, marker="o")
checkerboard_plot.scatter(checkerboard_shapes[:, 0], netket_checkerboard_times, label="NetKet", c="red", s=16, marker="o")
checkerboard_plot.legend()
checkerboard_plot.set_xlabel("Linear system size \$L\$")
checkerboard_plot.set_ylabel("Time in ns")

fig.savefig(f"{BENCHMARK_PATH}/cb_operator_performance.svg")

np.savetxt(f"{BENCHMARK_PATH}/cb_operator_performance.txt",
           np.asarray([custom_checkerboard_times, netket_checkerboard_times]).T,
           header="custom checkerboard, netket checkerboard")
