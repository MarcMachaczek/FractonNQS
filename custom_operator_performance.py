import jax.numpy as jnp
import numpy as np

import netket as nk
import geneqs
from global_variables import BENCHMARK_PATH

checkerboard_shapes = jnp.array([[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]])
n_samples = 2048

custom_checkerboard_times = []
custom_checkerboard_stdevs = []
netket_checkerboard_times = []
netket_checkerboard_stdevs = []
for checkerboard_shape in checkerboard_shapes:
    # define checkerboard operator
    cube_graph = nk.graph.Hypercube(length=checkerboard_shape[0].item(), n_dim=3, pbc=True)
    checkerboard_hilbert = nk.hilbert.Spin(s=1 / 2, N=cube_graph.n_nodes)
    custom_checkerboard = geneqs.operators.checkerboard.Checkerboard(checkerboard_hilbert, checkerboard_shape,
                                                                     h=(1, 0, 0))
    netket_checkerboard = geneqs.operators.checkerboard.get_netket_checkerboard(checkerboard_hilbert,
                                                                                checkerboard_shape, h=(1, 0, 0))

    # construct some dummy samples
    checkerboard_samples = jnp.ones(shape=(n_samples, checkerboard_hilbert.size), dtype=jnp.int8)

    # run everything once so jax compilation is triggered
    a, b = custom_checkerboard.get_conn_padded(checkerboard_samples)
    del a, b
    a, b = netket_checkerboard.get_conn_padded(checkerboard_samples)
    del a, b

    # %% get times
    custom_time, custom_stdev = geneqs.utils.benchmarking.time_function(custom_checkerboard.get_conn_padded, 500,
                                                                        checkerboard_samples)
    netket_time, netket_stdev = geneqs.utils.benchmarking.time_function(netket_checkerboard.get_conn_padded, 100,
                                                                        checkerboard_samples)

    custom_checkerboard_times.append(custom_time)
    custom_checkerboard_stdevs.append(custom_stdev)
    netket_checkerboard_times.append(netket_time)
    netket_checkerboard_stdevs.append(netket_stdev)

np.savetxt(f"{BENCHMARK_PATH}/cb_operator_performance.txt",
           np.array([custom_checkerboard_times, custom_checkerboard_stdevs, netket_checkerboard_times,
                     netket_checkerboard_stdevs]).T,
           header="custom checkerboard, custom_std, netket checkerboard, netket_std")

# %%
import numpy as np
from matplotlib import pyplot as plt
from global_variables import BENCHMARK_PATH

checkerboard_shapes = np.array([[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]])

data = np.loadtxt(f"{BENCHMARK_PATH}/cb_operator_performance.txt", )
custom_checkerboard_times, custom_checkerboard_stdevs = data[:, 0], data[:, 1]
netket_checkerboard_times, netket_checkerboard_stdevs = data[:, 2], data[:, 3]

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 40,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"]
})
# matplotlib.rcParams['svg.fonttype'] = 'none'
# matplotlib.rcParams.update({'font.size': 30})

fig = plt.figure(dpi=300, figsize=(12, 12))

checkerboard_plot = fig.add_subplot(111)
checkerboard_plot.errorbar(checkerboard_shapes[:, 0], custom_checkerboard_times,
                           yerr=custom_checkerboard_stdevs, label="Custom", c="green", markersize=20, marker="x")
checkerboard_plot.errorbar(checkerboard_shapes[:, 0], netket_checkerboard_times,
                           yerr=netket_checkerboard_stdevs, label="NetKet", c="red", markersize=20, marker="x")

checkerboard_plot.legend()
checkerboard_plot.set_xlabel(r"Linear system size $L$")
checkerboard_plot.set_ylabel(r"Time in ns")
checkerboard_plot.set_yscale("log")

fig.subplots_adjust(left=0.15)
fig.savefig(f"{BENCHMARK_PATH}/cb_operator_performance.pdf")
