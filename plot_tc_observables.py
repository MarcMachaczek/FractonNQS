import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 24})
cmap = matplotlib.colormaps["Set1"]

f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/toric2d_h"

# %%
field_direction = [2,]
shape = [[3, 3]]
labels = ["L=[3, 3]"]
eval_model = "ToricCRBM"
obs_list = []
# append multiple data to compare them each within one plot
obs_list.append(
    pd.read_csv(f"{save_dir}/L[3 3]_{eval_model}_observables.txt", sep=" ", header=0))

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs.sort_values(by=[f"h{f_dict[field_direction[i]]}"])
direction = [obs.iloc[-1, :3].values for obs in obs_list]

# %% magnetizations comparison
# obs columns: "hx, hy, hz, energy, energy_var, mag, mag_var, abs_mag, abs_mag_var, wilson, wilson_var, exact_energy"
fig = plt.figure(dpi=300, figsize=(22, 30))
plot_mag = fig.add_subplot(321)

for i, obs in enumerate(obs_list):
    plot_mag.errorbar(obs.iloc[:, field_direction], obs["mag"], yerr=obs["mag_err"], marker="o", markersize=2,
                      color=cmap(i))

    plot_mag.plot(obs.iloc[:, field_direction], obs["mag"], marker="o", markersize=2, color=cmap(i),
                  label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_mag.set_xlabel(f"external field")
plot_mag.set_ylabel("magnetization")
plot_mag.set_title(
    f"magnetization vs external field for ToricCode2d")
plot_mag.legend()

# %% absolute magnetization
plot_abs_mag = fig.add_subplot(322)

for i, obs in enumerate(obs_list):
    plot_abs_mag.errorbar(obs.iloc[:, field_direction], obs["abs_mag"], yerr=obs["abs_mag_err"], marker="o",
                          markersize=2, color=cmap(i))

    plot_abs_mag.plot(obs.iloc[:, field_direction], obs["abs_mag"], marker="o", markersize=2, color=cmap(i),
                      label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_abs_mag.set_xlabel(f"external field")
plot_abs_mag.set_ylabel("absolute magnetization")
plot_abs_mag.set_title(
    f" absolute magnetization vs external field for ToricCode2d")
plot_abs_mag.legend()

# %% susceptibility
plot_sus = fig.add_subplot(323)

for i, obs in enumerate(obs_list):
    sus, sus_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs["mag"].values, fields=obs.iloc[:, :3].values)
    plot_sus.plot(sus_fields[:, field_direction[i]], sus, marker="o", markersize=2, color=cmap(i),
                  label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_sus.set_xlabel(f"external field")
plot_sus.set_ylabel("susceptibility of magnetization")
plot_sus.set_title(
    f"susceptibility vs external field for ToricCode2d")
plot_sus.legend()

# %% wilson loop (see valenti et al)
plot_wilson = fig.add_subplot(324)

for i, obs in enumerate(obs_list):
    plot_wilson.errorbar(obs.iloc[:, field_direction[i]], obs["wilson"], yerr=obs["wilson_err"], marker="o",
                         markersize=2, color=cmap(i))

    plot_wilson.plot(obs.iloc[:, field_direction[i]], obs["wilson"], marker="o", markersize=2, color=cmap(i),
                     label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_wilson.set_xlabel(f"external field")
plot_wilson.set_ylabel("wilson loop < prod A_s * prod B_p >")
plot_wilson.set_title(
    f"wilson loop (see Valenti et al) vs external field for ToricCode2d")
plot_wilson.legend()

# %% energy per site
plot_energy = fig.add_subplot(325)

for i, obs in enumerate(obs_list):
    hilbert_size = np.prod(shape[i]) * 2

    plot_energy.plot(obs.iloc[:, field_direction], obs["energy"].values / hilbert_size, marker="o", markersize=2,
                     color=cmap(i), label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_energy.set_xlabel(f"external field")
plot_energy.set_ylabel("energy per site")
plot_energy.set_title(
    f"energy per site vs external field for ToricCode2d")

plot_energy.legend()

# %% variance of the energy
plot_evar = fig.add_subplot(326)

for i, obs in enumerate(obs_list):
    plot_evar.plot(obs.iloc[:, field_direction[i]], np.abs(obs["energy_var"].values), marker="o",
                     markersize=2, color=cmap(i), label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

plot_evar.set_xlabel(f"external field")
plot_evar.set_ylabel("specific heat")
plot_evar.set_title(
    f"specific heat vs external field for ToricCode2d")

plot_evar.legend()
plt.show()
fig.savefig(f"{save_dir}/obs_comparison_L[{shape}_cRBM.pdf")
