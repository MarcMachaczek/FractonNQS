import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 24})
cmap = matplotlib.colormaps["Set1"]

f_dict = {0: "x", 1: "y", 2: "z"}

# %%
field_direction = 2
L = 8
hilbert_size = 2 * L ** 2
eval_model = "ToricCRBM"
save_dir = f"{RESULTS_PATH}/toric2d_h"

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
fig.savefig(f"{save_dir}/hist_comparison_L[{L} {L}]_cRBM.pdf")
