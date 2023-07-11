import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams.update({'font.size': 12})

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/checkerboard"

# %%
field_direction = [0]  # 0=x, 1=y, 2=z
shape = [[4, 4, 4]]
labels = ["L=[4, 4, 4]"]
eval_model = "CheckerCRBM"
obs_list = []
# append multiple data to compare them each within one plot
obs_list.append(
    pd.read_csv(f"{save_dir}/L[4 4 4]_{eval_model}_observables.txt", sep=" ", header=0))

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs.sort_values(by=[f"h{f_dict[field_direction[i]]}"])
direction = [obs.iloc[-1, :3].values for obs in obs_list]

# %% magnetizations comparison
# obs columns: "hx, hy, hz, energy, energy_var, mag, mag_var, abs_mag, abs_mag_var, (exact_energy)"
fig = plt.figure(dpi=300, figsize=(34, 22))
plot_mag = fig.add_subplot(231)

for i, obs in enumerate(obs_list):
    plot_mag.errorbar(obs.iloc[:, field_direction], obs["mag"], yerr=obs["mag_err"], marker="o", markersize=2,
                      color=cmap(i))

    plot_mag.plot(obs.iloc[:, field_direction], obs["mag"], marker="o", markersize=2, color=cmap(i),
                  label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_mag.set_xlabel(f"external field")
plot_mag.set_ylabel("magnetization")
plot_mag.set_title(
    f"magnetization vs external field for Checkerboard")
plot_mag.legend()

# %% absolute magnetization
plot_abs_mag = fig.add_subplot(232)

for i, obs in enumerate(obs_list):
    plot_abs_mag.errorbar(obs.iloc[:, field_direction], obs["abs_mag"], yerr=obs["abs_mag_err"], marker="o",
                          markersize=2, color=cmap(i))

    plot_abs_mag.plot(obs.iloc[:, field_direction], obs["abs_mag"], marker="o", markersize=2, color=cmap(i),
                      label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_abs_mag.set_xlabel(f"external field")
plot_abs_mag.set_ylabel("absolute magnetization")
plot_abs_mag.set_title(
    f" absolute magnetization vs external field for Checkerboard")
plot_abs_mag.legend()

# %% susceptibility
plot_sus = fig.add_subplot(233)

for i, obs in enumerate(obs_list):
    sus, sus_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs["mag"].values, fields=obs.iloc[:, :3].values)
    plot_sus.plot(sus_fields[:, field_direction[i]], sus, marker="o", markersize=2, color=cmap(i),
                  label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_sus.set_xlabel(f"external field")
plot_sus.set_ylabel("susceptibility of magnetization")
plot_sus.set_title(
    f"susceptibility vs external field for Checkerboard")
plot_sus.legend()

# %% energy per site
plot_energy = fig.add_subplot(234)

for i, obs in enumerate(obs_list):
    hilbert_size = np.prod(shape[i])
    plot_energy.errorbar(obs.iloc[:, field_direction], obs["energy"].values / hilbert_size,
                          yerr=obs["energy_err"].values / hilbert_size, marker="o", markersize=2, color=cmap(i))

    plot_energy.plot(obs.iloc[:, field_direction], obs["energy"].values / hilbert_size, marker="o", markersize=2,
                     color=cmap(i),
                     label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_energy.set_xlabel(f"external field")
plot_energy.set_ylabel("energy per site")
plot_energy.set_title(
    f"energy per site vs external field for Checkerboard")

plot_energy.legend()

# %% energy derivative dE/dh
plot_dEdh = fig.add_subplot(235)

for i, obs in enumerate(obs_list):
    dEdh, fd_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs["energy"].values,
                                                          fields=obs.iloc[:, :3].values)
    plot_dEdh.plot(fd_fields[:, field_direction[i]], dEdh, marker="o", markersize=2, color=cmap(i),
                   label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_dEdh.set_xlabel(f"external field")
plot_dEdh.set_ylabel("dE / dh")
plot_dEdh.set_title(
    f"energy derivative vs external field for Checkerboard")
plot_dEdh.legend()

# %% specific heat / variance of the energy
plot_spheat = fig.add_subplot(236)

for i, obs in enumerate(obs_list):
    plot_spheat.plot(obs.iloc[:, field_direction], np.abs(obs["energy_var"].values), marker="o", markersize=2,
                     color=cmap(i),
                     label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_spheat.set_xlabel(f"external field")
plot_spheat.set_ylabel("specific heat")
plot_spheat.set_title(
    f"specific heat vs external field for Checkerboard")

plot_spheat.legend()
plt.show()
fig.savefig(f"{save_dir}/obs_comparison_L{shape}_cRBM.pdf")
