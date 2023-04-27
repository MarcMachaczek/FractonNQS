import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}
# %%
L = 8
field_direction = 2  # 0=x, 1=y, 2=z
obs_list = []  # append multiple data to compare them each within one plot
# obs_list.append(np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L[{L} {L}]_cRBM_a1_observables"))
obs_list.append(np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L={L}_complex_crbm_sd_5/L[{L} {L}]_cRBM_a1_observables.txt"))
#obs_list.append(np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L={L}_complex_crbm_hx03_4/L[{L} {L}]_cRBM_a1_observables"))


# %% magnetizations comparison
# obs columns: "hx, hy, hz, mag, mag_var, energy, energy_var"
fig = plt.figure(dpi=300, figsize=(30, 12))
plot_mag = fig.add_subplot(131)

for i, obs in enumerate(obs_list):
    for ob in obs:
        plot_mag.errorbar(ob[field_direction], ob[3], yerr=ob[4], marker="o", markersize=2, color=cmap(i))

    plot_mag.plot(obs[:, field_direction], obs[:, 3], marker="o", markersize=2, color=cmap(i), label=f"h={obs[-1][:3]}")

plot_mag.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_mag.set_ylabel("magnetization")
plot_mag.set_title(f"magnetization vs external field in {f_dict[field_direction]}-direction for ToricCode2d of shape=[{L},{L}]")
plot_mag.legend()


# %% susceptibility
plot_sus = fig.add_subplot(132)

for i, obs in enumerate(obs_list):
    sus, sus_fields = geneqs.utils.eval_obs.susc_from_mag(magnetizations=obs[:, 3], fields=obs[:, :3])
    plot_sus.plot(sus_fields[:,2], sus, marker="o", markersize=2, color=cmap(i), label=f"h={obs[-1][:3]}")

plot_sus.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_sus.set_ylabel("susceptibility")
plot_sus.set_title(f"susceptibility vs external field in {f_dict[field_direction]}-direction for ToricCode2d of shape=[{L},{L}]")
plot_sus.legend()


# %%
plot_spheat = fig.add_subplot(133)

for i, obs in enumerate(obs_list):
    plot_spheat.plot(obs[:, field_direction], np.abs(obs[:, 6]), marker="o", markersize=2, color=cmap(i), label=f"h={obs[-1][:3]}")

plot_spheat.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_spheat.set_ylabel("specific heat")
plot_spheat.set_title(f"specific heat vs external field in {f_dict[field_direction]}-direction for ToricCode2d of shape=[{L},{L}]")


plot_spheat.legend()
plt.show()
fig.savefig(f"{RESULTS_PATH}/toric2d_h/L[{L} {L}]_cRBM_a1_obs_comparison.pdf")
