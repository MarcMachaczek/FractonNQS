import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}
# %%
L = 3
field_direction = 0  # 0=x, 1=y, 2=z
obs_list = []  # append multiple data to compare them each within one plot
# obs_list.append(np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L[{L} {L}]_cRBM_a1_observables"))
# obs_list.append(
#     np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L={L}_complex_crbm_hx/L[{L} {L}]_ToricCRBM_a1_observables.txt"))

obs_list.append(
    np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L={L}_crbm_ed_test_hxhz_mc/ed_test_ToricCRBM_hdir[0.8 0.  0.8]_observables"))

# obs_list.append(np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L={L}_complex_crbm_hx03_4/L[{L} {L}]_cRBM_a1_observables"))

direction = obs_list[0][-1, :3]
# %% magnetizations comparison
# obs columns: "hx, hy, hz, mag, mag_var, energy, energy_var, wilson, wilson_var"
fig = plt.figure(dpi=300, figsize=(26, 24))
plot_mag = fig.add_subplot(221)

for i, obs in enumerate(obs_list):
    for ob in obs:
        plot_mag.errorbar(ob[field_direction], ob[3], yerr=ob[4], marker="o", markersize=2, color=cmap(i))

    plot_mag.plot(obs[:, field_direction], obs[:, 3], marker="o", markersize=2, color=cmap(i), label=f"h={obs[-1][:3]}")

plot_mag.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_mag.set_ylabel("magnetization")
plot_mag.set_title(
    f"magnetization vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")
plot_mag.legend()

# %% susceptibility
plot_sus = fig.add_subplot(222)

for i, obs in enumerate(obs_list):
    sus, sus_fields = geneqs.utils.eval_obs.susc_from_mag(magnetizations=obs[:, 3], fields=obs[:, :3])
    plot_sus.plot(sus_fields[:, field_direction], sus, marker="o", markersize=2, color=cmap(i), label=f"h={obs[-1][:3]}")

plot_sus.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_sus.set_ylabel("susceptibility")
plot_sus.set_title(
    f"susceptibility vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")
plot_sus.legend()

# %% energy per site
plot_energy = fig.add_subplot(223)

for i, obs in enumerate(obs_list):
    n_sites = L ** 2 * 2
    plot_energy.plot(obs[:, field_direction], obs[:, 5] / n_sites, marker="o", markersize=2, color=cmap(i),
                     label=f"h={obs[-1][:3]}")

plot_energy.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_energy.set_ylabel("specific heat")
plot_energy.set_title(
    f"specific heat vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")

plot_energy.legend()

# %% specific heat / variance of the energy
plot_spheat = fig.add_subplot(224)

for i, obs in enumerate(obs_list):
    plot_spheat.plot(obs[:, field_direction], np.abs(obs[:, 6]), marker="o", markersize=2, color=cmap(i),
                     label=f"h={obs[-1][:3]}")

plot_spheat.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_spheat.set_ylabel("specific heat")
plot_spheat.set_title(
    f"specific heat vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")

plot_spheat.legend()
plt.show()
fig.savefig(f"{RESULTS_PATH}/toric2d_h/L[{L} {L}]_cRBM_obs_comparison.pdf")
