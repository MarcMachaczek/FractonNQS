import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams.update({'font.size': 12})

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}

# %%
field_direction = [0, 1]
L = 8
hilbert_size = 2 * L ** 2
eval_model = "ToricCRBM"
save_dir = f"{RESULTS_PATH}/toric2d_h"
obs_list = []
# append multiple data to compare them each within one plot
obs_list.append(
    np.loadtxt(f"{save_dir}/L[{L} {L}]_{eval_model}_observables.txt"))
obs_list.append(
    np.loadtxt(f"{save_dir}/L[{L} {L}]_{eval_model}_observables_hz.txt"))

labels = ["x-direction", "z-direction"]

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs[obs[:, field_direction[i]].argsort()]
direction = obs_list[0][-1, :3]

# %% magnetizations comparison
# obs columns: "hx, hy, hz, energy, energy_var, mag, mag_var, abs_mag, abs_mag_var, wilson, wilson_var, exact_energy"
fig = plt.figure(dpi=300, figsize=(22, 30))
plot_mag = fig.add_subplot(321)

for i, obs in enumerate(obs_list):
    for ob in obs:
        plot_mag.errorbar(ob[field_direction[i]], ob[5], yerr=ob[6], marker="o", markersize=2, color=cmap(i))

    plot_mag.plot(obs[:, field_direction[i]], obs[:, 5], marker="o", markersize=2, color=cmap(i), label=labels[i])
plot_mag.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_mag.set_ylabel("magnetization")
plot_mag.set_title(
    f"magnetization vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")
plot_mag.legend()

# %% absolute magnetization
plot_abs_mag = fig.add_subplot(322)

for i, obs in enumerate(obs_list):
    for ob in obs:
        plot_abs_mag.errorbar(ob[field_direction], ob[7], yerr=ob[8], marker="o", markersize=2, color=cmap(i))

    plot_abs_mag.plot(obs[:, field_direction], obs[:, 7], marker="o", markersize=2, color=cmap(i),
                      label=labels[i])

plot_abs_mag.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_abs_mag.set_ylabel("absolute magnetization")
plot_abs_mag.set_title(
    f" absolute magnetization vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")
plot_abs_mag.legend()

# %% susceptibility
plot_sus = fig.add_subplot(323)

for i, obs in enumerate(obs_list):
    sus, sus_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs[:, 5], fields=obs[:, :3])
    plot_sus.plot(sus_fields[:, field_direction], sus, marker="o", markersize=2, color=cmap(i),
                  label=labels[i])

plot_sus.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_sus.set_ylabel("susceptibility of magnetization")
plot_sus.set_title(
    f"susceptibility vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")
plot_sus.legend()

# %% wilson loop (see valenti et al)
plot_wilson = fig.add_subplot(324)

for i, obs in enumerate(obs_list):
    for ob in obs:
        plot_wilson.errorbar(ob[field_direction], ob[9], yerr=ob[10], marker="o", markersize=2, color=cmap(i))

    plot_wilson.plot(obs[:, field_direction], obs[:, 9], marker="o", markersize=2, color=cmap(i),
                     label=labels[i])

plot_wilson.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_wilson.set_ylabel("wilson loop < prod A_s * prod B_p >")
plot_wilson.set_title(
    f"wilson loop (see Valenti et al) vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")
plot_wilson.legend()

# %% energy per site
plot_energy = fig.add_subplot(325)

for i, obs in enumerate(obs_list):
    n_sites = L ** 2 * 2
    plot_energy.plot(obs[:, field_direction], obs[:, 3] / n_sites, marker="o", markersize=2, color=cmap(i),
                     label=labels[i])

plot_energy.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_energy.set_ylabel("energy per site")
plot_energy.set_title(
    f"energy per site vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")

plot_energy.legend()

# %% specific heat / variance of the energy
plot_spheat = fig.add_subplot(326)

for i, obs in enumerate(obs_list):
    plot_spheat.plot(obs[:, field_direction], np.abs(obs[:, 4]), marker="o", markersize=2, color=cmap(i),
                     label=labels[i])

plot_spheat.set_xlabel(f"external field h{f_dict[field_direction[0]]}")
plot_spheat.set_ylabel("specific heat")
plot_spheat.set_title(
    f"specific heat vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")

plot_spheat.legend()
plt.show()
fig.savefig(f"{save_dir}/obs_comparison_L[{L} {L}]_cRBM.pdf")
