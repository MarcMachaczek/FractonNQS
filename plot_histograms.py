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
shapes = [[8, 8, 8]]
eval_model = "CheckerCRBM"
label = "hx_right_left_chaintrans_nonoise"
save_dir = f"{RESULTS_PATH}/checkerboard/L=8_final"

# %%%%%%%%%%%%%%% HISTOGRAMS %%%%%%%%%%%%%%% #
epsite_fig = plt.figure(dpi=300, figsize=(10, 10))
epsite_hist = epsite_fig.add_subplot(111)

mag_fig = plt.figure(dpi=300, figsize=(10, 10))
mag_hist = mag_fig.add_subplot(111)

h_idx = 1

for i, shape in enumerate(shapes):
    shape_string = " ".join(map(str, shape))
    # shape is (n_hist_fields, 3), where 3 = field_value + hist_values + bin_edges
    epsite_histograms = np.load(f"{save_dir}/hist_epsite_L[{shape_string}]_{eval_model}_{label}.npy", allow_pickle=True)
    
    hist = epsite_histograms[h_idx]
    
    field, hist_values, bin_edges = np.round(hist[0], 3), hist[1], hist[2]
    print(hist_values, bin_edges)
    mids = (bin_edges[1:] + bin_edges[:-1]) / 2
    mean = np.average(mids, weights=hist_values)
    # edges = mids - mean
    # hist_values_rescaled = hist_values / np.sum(hist_values)
    epsite_hist.plot(mids, hist_values, label=f"{shape}_{field}", color=cmap(i))

for i, shape in enumerate(shapes):
    shape_string = " ".join(map(str, shape))
    # shape is (n_hist_fields, 3), where 3 = field_value + hist_values + bin_edges
    mag_histograms = np.load(f"{save_dir}/hist_mag_L[{shape_string}]_{eval_model}_{label}.npy", allow_pickle=True)
    
    hist = mag_histograms[h_idx]
    
    field, hist_values, bin_edges = np.round(hist[0], 3), hist[1], hist[2]
    print(hist_values, bin_edges)
    mids = (bin_edges[1:] + bin_edges[:-1]) / 2
    mean = np.average(mids, weights=hist_values)
    # edges = mids - mean
    # hist_values_rescaled = hist_values / np.sum(hist_values)
    mag_hist.plot(mids, hist_values, label=f"{shape}_{field}", color=cmap(i))

epsite_hist.legend()
mag_hist.legend()

epsite_hist.set_xlabel(f"\$ h={field} \$, {label}")
epsite_hist.set_xlabel(f"\$ h={field} \$, {label}")

epsite_fig.savefig(f"{save_dir}/epsite_hists_L{shapes}_{field}.svg")
mag_fig.savefig(f"{save_dir}/mag_hists_L{shapes}_{field}.svg")