import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json

from global_variables import RESULTS_PATH

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 40,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman"  # ["Computer Modern Serif"]
})

# matplotlib.rcParams['svg.fonttype'] = 'none'
# matplotlib.rcParams.update({'font.size': 30})
cmap = matplotlib.colormaps["Set1"]
line_styles = ["dashed", "dotted"]
markers = ["<", ">"]
markersize = 12

f_dict = {0: "x", 1: "y", 2: "z"}

# %%
L = 4
shapes = 2 * [[L, L, L]]
field_directions = 2 * [2]
labels = ["right_left_chaintransfer_nonoise_dense_lowlr", "left_right_chaintransfer_nonoise_dense_lowlr"]
legend_labels = ["right_left", "left_right"]  # ["hx03_independent", "hx03_right_left", "left_right"]  # ["CRBM, alpha=1", "CRBM, alpha=2", "RBMSymm"]
eval_model = "CheckerCRBM_2"
save_dir = f"{RESULTS_PATH}/checkerboard/L={L}_final"

rhat_fig = plt.figure(dpi=300, figsize=(12, 12))
tau_fig = plt.figure(dpi=300, figsize=(12, 12))
arate_fig = plt.figure(dpi=300, figsize=(12, 12))

rhat_plot = rhat_fig.add_subplot(111)
tau_plot = tau_fig.add_subplot(111)
arate_plot = arate_fig.add_subplot(111)

for i, (fdir, label) in enumerate(zip(field_directions, labels)):
    shape_string = " ".join(map(str, shapes[i]))
    fname = f"{save_dir}/L={L}_mc_crbm_h{f_dict[field_directions[i]]}_{label}"
    obs = pd.read_csv(f"{fname}/L[{shape_string}]_{eval_model}_observables.txt",
                      sep=" ", header=0)
    obs = obs.sort_values(by=[f"h{f_dict[field_directions[i]]}"])
    fields = obs.iloc[:, :3].values

    rhats, taus, arates = [], [], []

    for field in fields:
        stats = json.load(open(f"{fname}/stats_L[{shape_string}]_{eval_model}_h{tuple(field)}.json"))
        rhats.append(np.mean(stats["Energy"]["R_hat"][-200:]))
        taus.append(np.mean(stats["Energy"]["TauCorr"][-200:]))
        arates.append(np.mean(stats["acceptance_rate"]["value"][-200:]))

    iters = fields[:, field_directions[i]]

    rhat_plot.plot(iters, np.asarray(rhats), marker=markers[i], markersize=markersize,
                   color=cmap(i+1), label=legend_labels[i].replace("_", "-"), linestyle=line_styles[i], linewidth=3.0)
    tau_plot.plot(iters, np.asarray(taus), marker=markers[i], markersize=markersize,
                  color=cmap(i+1), label=legend_labels[i].replace("_", "-"), linestyle=line_styles[i], linewidth=3.0)
    arate_plot.plot(iters, np.asarray(arates), marker=markers[i], markersize=markersize,
                    color=cmap(i+1), label=legend_labels[i].replace("_", "-"), linestyle=line_styles[i], linewidth=3.0)

rhat_plot.set_xlabel(
    rf"Field strength in $ {f_dict[field_directions[0]]} $-direction $ h_{f_dict[field_directions[0]]} $ ")
tau_plot.set_xlabel(
    rf"Field strength in $ {f_dict[field_directions[0]]} $-direction $ h_{f_dict[field_directions[0]]} $ ")
arate_plot.set_xlabel(
    rf"Field strength in $ {f_dict[field_directions[0]]} $-direction $ h_{f_dict[field_directions[0]]} $ ")

rhat_plot.set_ylabel(r"Split $\hat{R}$")
tau_plot.set_ylabel(r"Auto-correlation time $\tau $")
arate_plot.set_ylabel(r"Update acceptance rate")

rhat_plot.set_ylim(1, 1.4)
arate_plot.set_ylim(0, 1)

rhat_plot.legend()
tau_plot.legend()
arate_plot.legend()

rhat_fig.subplots_adjust(left=0.15)
tau_fig.subplots_adjust(left=0.15)
arate_fig.subplots_adjust(left=0.15)
rhat_fig.savefig(f"{save_dir}/rhats_L[{shape_string}]_{eval_model}_h{f_dict[field_directions[0]]}.pdf")
tau_fig.savefig(f"{save_dir}/taus_L[{shape_string}]_{eval_model}_h{f_dict[field_directions[0]]}.pdf")
arate_fig.savefig(f"{save_dir}/arates_L[{shape_string}]_{eval_model}_h{f_dict[field_directions[0]]}.pdf")
