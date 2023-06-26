import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams.update({'font.size': 12})

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/checkerboard"

# %%
field_direction = [1]  # 0=x, 1=y, 2=z
shape = [[4, 2, 2]]
labels = ["L=[4, 2, 2]"]
eval_model = "CheckerCRBM"
obs_list = []
# append multiple data to compare them each within one plot
obs_list.append(
    np.loadtxt(f"{save_dir}/L[4 2 2]_{eval_model}_observables.txt"))

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs[obs[:, field_direction[i]].argsort()]
direction = obs_list[0][-1, :3]

# %% magnetizations comparison
# obs columns: "hx, hy, hz, energy, energy_var, mag, mag_var, abs_mag, abs_mag_var, (exact_energy)"
fig = plt.figure(dpi=300, figsize=(34, 22))
plot_mag = fig.add_subplot(231)

for i, obs in enumerate(obs_list):
    for ob in obs:
        plot_mag.errorbar(ob[field_direction[i]], ob[5], yerr=ob[6], marker="o", markersize=2, color=cmap(i))

    plot_mag.plot(obs[:, field_direction[i]], obs[:, 5], marker="o", markersize=2, color=cmap(i), label=labels[i])
plot_mag.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_mag.set_ylabel("magnetization")
plot_mag.set_title(
    f"magnetization vs external field in {direction}-direction for Checkerboard")
plot_mag.legend()

# %% absolute magnetization
plot_abs_mag = fig.add_subplot(232)

for i, obs in enumerate(obs_list):
    for ob in obs:
        plot_abs_mag.errorbar(ob[field_direction[i]], ob[7], yerr=ob[8], marker="o", markersize=2, color=cmap(i))

    plot_abs_mag.plot(obs[:, field_direction[i]], obs[:, 7], marker="o", markersize=2, color=cmap(i),
                      label=labels[i])

plot_abs_mag.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_abs_mag.set_ylabel("absolute magnetization")
plot_abs_mag.set_title(
    f" absolute magnetization vs external field in {direction}-direction for Checkerboard")
plot_abs_mag.legend()

# %% susceptibility
plot_sus = fig.add_subplot(233)

for i, obs in enumerate(obs_list):
    sus, sus_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs[:, 5], fields=obs[:, :3])
    plot_sus.plot(sus_fields[:, field_direction[i]], sus, marker="o", markersize=2, color=cmap(i),
                  label=labels[i])

plot_sus.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_sus.set_ylabel("susceptibility of magnetization")
plot_sus.set_title(
    f"susceptibility vs external field in {direction}-direction for Checkerboard")
plot_sus.legend()

# %% energy per site
plot_energy = fig.add_subplot(234)

for i, obs in enumerate(obs_list):
    hilbert_size = np.prod(shape[i])
    plot_energy.plot(obs[:, field_direction[i]], obs[:, 3] / hilbert_size, marker="o", markersize=2, color=cmap(i),
                     label=labels[i])

plot_energy.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_energy.set_ylabel("energy per site")
plot_energy.set_title(
    f"energy per site vs external field in {direction}-direction for Checkerboard")

plot_energy.legend()

# %% energy derivative dE/dh
plot_dEdh = fig.add_subplot(235)

for i, obs in enumerate(obs_list):
    dEdh, fd_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs[:, 3], fields=obs[:, :3])
    plot_dEdh.plot(fd_fields[:, field_direction[i]], dEdh, marker="o", markersize=2, color=cmap(i),
                   label=labels[i])

plot_dEdh.set_xlabel(f"external field h{f_dict[field_direction[i]]}")
plot_dEdh.set_ylabel("dE / dh")
plot_dEdh.set_title(
    f"energy derivative vs external field in {direction}-direction for Checkerboard")
plot_dEdh.legend()

# %% specific heat / variance of the energy
plot_spheat = fig.add_subplot(236)

for i, obs in enumerate(obs_list):
    plot_spheat.plot(obs[:, field_direction[i]], np.abs(obs[:, 4]), marker="o", markersize=2, color=cmap(i),
                     label=labels[i])

plot_spheat.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_spheat.set_ylabel("specific heat")
plot_spheat.set_title(
    f"specific heat vs external field in {direction}-direction for Checkerboard")

plot_spheat.legend()
plt.show()
fig.savefig(f"{save_dir}/obs_comparison_L{shape}_cRBM.pdf")
