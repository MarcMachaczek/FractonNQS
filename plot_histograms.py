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
shapes = [[4, 4, 4], [6, 6, 6]]
eval_model = "CheckerCRBM"
label = "right-left"
save_dir = f"{RESULTS_PATH}/checkerboard"

# %%%%%%%%%%%%%%% HISTOGRAMS %%%%%%%%%%%%%%% #
epsite_fig = plt.figure(dpi=300, figsize=(10, 10))
epsite_hist = epsite_fig.add_subplot(111)

mag_fig = plt.figure(dpi=300, figsize=(10, 10))
mag_hist = mag_fig.add_subplot(111)

h_idx = 0

for i, shape in enumerate(shapes):
    # shape is (n_hist_fields, 3), where 3 = field_value + hist_values + bin_edges
    epsite_histograms = np.load(f"{save_dir}/hist_epsite_L[{shape}]_{eval_model}.npy", allow_pickle=True)
    
    hist = epsite_histograms[h_idx]
    
    field, hist_values, bin_edges = np.round(hist[0], 3), hist[1], hist[2]
    mids = (bins[1:] + bins[:-1]) / 2
    mean = np.average(mids, weights=hist_values)
    # edges = mids - mean
    # hist_values_rescaled = hist_values / np.sum(hist_values)
    epsite_hist.plot(edges, hist_values, label=f"{shape}", color=cmap(i))

for i, shape in enumerate(shapes):
    # shape is (n_hist_fields, 3), where 3 = field_value + hist_values + bin_edges
    mag_histograms = np.load(f"{save_dir}/hist_mag_L[{shape}]_{eval_model}.npy", allow_pickle=True)
    
    hist = mag_histograms[h_idx]
    
    field, hist_values, bin_edges = np.round(hist[0], 3), hist[1], hist[2]
    mids = (bins[1:] + bins[:-1]) / 2
    mean = np.average(mids, weights=hist_values)
    # edges = mids - mean
    # hist_values_rescaled = hist_values / np.sum(hist_values)
    mag_hist.plot(edges, hist_values, label=f"{shape}", color=cmap(i))

epsite_hist.legend()
mag_hist.legend()

epsite_hist.xlabel(f"\$ h={field} \$, {label}")
epsite_hist.xlabel(f"\$ h={field} \$, {label}")

fig.savefig(f"{save_dir}/epsite_hists_L{shapes}_{field}.svg")
fig.savefig(f"{save_dir}/mag_hists_L{shapes}_{field}.svg")