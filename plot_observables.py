import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}
# %%
L = 4
eval_model = "ToricCRBM"
save_dir = f"{RESULTS_PATH}/toric2d_h/test"
obs_list = []
# append multiple data to compare them each within one plot
obs_list.append(
    np.loadtxt(f"{save_dir}/L[{L} {L}]_{eval_model}_a1_observables.txt"))

direction = obs_list[0][-1, :3]
field_direction = 0  # 0=x, 1=y, 2=z
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

# %% histograms
# shape is (n_hist_fields, 3), where 3 = field_value + hist_values + bin_edges
energy_histograms = np.load(f"{save_dir}/hists_energy_L[{L} {L}]_{eval_model}.npy", allow_pickle=True)
mag_histograms = np.load(f"{save_dir}/hists_mag_L[{L} {L}]_{eval_model}.npy", allow_pickle=True)
abs_mag_histograms = np.load(f"{save_dir}/hists_abs_mag_L[{L} {L}]_{eval_model}.npy", allow_pickle=True)
A_B_histograms = np.load(f"{save_dir}/hists_A_B_L[{L} {L}]_{eval_model}.npy", allow_pickle=True)

fig = plt.figure(dpi=300, figsize=(13, 12))
fig.suptitle("Probability (histogram) plots for different observables and external field for the Toric Code")

energy_hist = fig.add_subplot(221)
mag_hist = fig.add_subplot(222)
abs_mag_hist = fig.add_subplot(223)
A_B_hist = fig.add_subplot(224)

for hist in energy_histograms:
    field, hist_values, bin_edges = hist[0], hist[1], hist[2]
    edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    energy_hist.plot(edges, hist_values, label=f"h={tuple([round(h, 3) for h in field])}")
    energy_hist.set_xlabel("energy per site")

for hist in mag_histograms:
    field, hist_values, bin_edges = hist[0], hist[1], hist[2]
    edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    mag_hist.plot(edges, hist_values, label=f"h={tuple([round(h, 3) for h in field])}")
    mag_hist.set_xlabel("magnetization")

for hist in abs_mag_histograms:
    field, hist_values, bin_edges = hist[0], hist[1], hist[2]
    edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    abs_mag_hist.plot(edges, hist_values, label=f"h={tuple([round(h, 3) for h in field])}")
    abs_mag_hist.set_xlabel("absolute magnetization")

for hist in A_B_histograms:
    field, hist_values, bin_edges = hist[0], hist[1], hist[2]
    edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    A_B_hist.plot(edges, hist_values, label=f"h={tuple([round(h, 3) for h in field])}")
    A_B_hist.set_xlabel("A_star - B_plaq")

energy_hist.legend()
mag_hist.legend()
abs_mag_hist.legend()
A_B_hist.legend()

plt.show()
