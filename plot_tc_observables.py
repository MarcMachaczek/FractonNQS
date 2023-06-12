import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams.update({'font.size': 12})

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}

# %%
field_direction = 2
L = 8
hilbert_size = 2 * L ** 2
eval_model = "ToricCRBM"
save_dir = f"{RESULTS_PATH}/toric2d_h"
obs_list = []
# append multiple data to compare them each within one plot
obs_list.append(
    np.loadtxt(f"{save_dir}/L[{L} {L}]_{eval_model}_observables.txt"))

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs[obs[:, field_direction].argsort()]
direction = obs_list[0][-1, :3]

# %% magnetizations comparison
# obs columns: "hx, hy, hz, energy, energy_var, mag, mag_var, abs_mag, abs_mag_var, wilson, wilson_var, exact_energy"
fig = plt.figure(dpi=300, figsize=(22, 30))
plot_mag = fig.add_subplot(321)

for i, obs in enumerate(obs_list):
    for ob in obs:
        plot_mag.errorbar(ob[field_direction], ob[5], yerr=ob[6], marker="o", markersize=2, color=cmap(i))

    plot_mag.plot(obs[:, field_direction], obs[:, 5], marker="o", markersize=2, color=cmap(i), label=f"h={obs[-1][:3]}")

plot_mag.set_xlabel(f"external field h{f_dict[field_direction]}")
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
                      label=f"h={obs[-1][:3]}")

plot_abs_mag.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_abs_mag.set_ylabel("absolute magnetization")
plot_abs_mag.set_title(
    f" absolute magnetization vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")
plot_abs_mag.legend()

# %% susceptibility
plot_sus = fig.add_subplot(323)

for i, obs in enumerate(obs_list):
    sus, sus_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs[:, 5], fields=obs[:, :3])
    plot_sus.plot(sus_fields[:, field_direction], sus, marker="o", markersize=2, color=cmap(i),
                  label=f"h={obs[-1][:3]}")

plot_sus.set_xlabel(f"external field h{f_dict[field_direction]}")
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
                     label=f"h={obs[-1][:3]}")

plot_wilson.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_wilson.set_ylabel("wilson loop < prod A_s * prod B_p >")
plot_wilson.set_title(
    f"wilson loop (see Valenti et al) vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")
plot_wilson.legend()

# %% energy per site
plot_energy = fig.add_subplot(325)

for i, obs in enumerate(obs_list):
    n_sites = L ** 2 * 2
    plot_energy.plot(obs[:, field_direction], obs[:, 3] / n_sites, marker="o", markersize=2, color=cmap(i),
                     label=f"h={obs[-1][:3]}")

plot_energy.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_energy.set_ylabel("energy per site")
plot_energy.set_title(
    f"energy per site vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")

plot_energy.legend()

# %% specific heat / variance of the energy
plot_spheat = fig.add_subplot(326)

for i, obs in enumerate(obs_list):
    plot_spheat.plot(obs[:, field_direction], np.abs(obs[:, 4]), marker="o", markersize=2, color=cmap(i),
                     label=f"h={obs[-1][:3]}")

plot_spheat.set_xlabel(f"external field h{f_dict[field_direction]}")
plot_spheat.set_ylabel("specific heat")
plot_spheat.set_title(
    f"specific heat vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")

plot_spheat.legend()
plt.show()
fig.savefig(f"{RESULTS_PATH}/toric2d_h/obs_comparison_L[{L} {L}]_cRBM.pdf")

# %%%%%%%%%%%%%%% HISTOGRAMS %%%%%%%%%%%%%%% #
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

obs = obs_list[0]
for hist in energy_histograms:
    field, hist_values, bin_edges = np.round(hist[0], 3), hist[1], hist[2]
    e_mean = obs[np.argwhere(np.all(np.round(obs[:, :3], 3) == field, axis=1)), 3].item()
    edges = (bin_edges[1:] + bin_edges[:-1]) / 2 - e_mean / hilbert_size
    hist_values_rescaled = hist_values / np.sum(hist_values)
    energy_hist.plot(edges, hist_values_rescaled, label=f"h={tuple([round(h, 3) for h in field])}")
    energy_hist.set_xlabel("energy per site with mean subtracted")

for hist in mag_histograms:
    field, hist_values, bin_edges = np.round(hist[0], 3), hist[1], hist[2]
    edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    hist_values_rescaled = hist_values / np.sum(hist_values)
    mag_hist.plot(edges, hist_values_rescaled, label=f"h={tuple([round(h, 3) for h in field])}")
    mag_hist.set_xlabel("magnetization")

for hist in abs_mag_histograms:
    field, hist_values, bin_edges = np.round(hist[0], 3), hist[1], hist[2]
    edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    hist_values_rescaled = hist_values / np.sum(hist_values)
    abs_mag_hist.plot(edges, hist_values_rescaled, label=f"h={tuple([round(h, 3) for h in field])}")
    abs_mag_hist.set_xlabel("absolute magnetization")

for hist in A_B_histograms:
    field, hist_values, bin_edges = np.round(hist[0], 3), hist[1], hist[2]
    edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    hist_values_rescaled = hist_values / np.sum(hist_values)
    A_B_hist.plot(edges, hist_values_rescaled, label=f"h={tuple([round(h, 3) for h in field])}")
    A_B_hist.set_xlabel("A_star - B_plaq")

energy_hist.legend()
mag_hist.legend()
abs_mag_hist.legend()
A_B_hist.legend()

plt.show()
fig.savefig(f"{RESULTS_PATH}/toric2d_h/hist_comparison_L[{L} {L}]_cRBM.pdf")
