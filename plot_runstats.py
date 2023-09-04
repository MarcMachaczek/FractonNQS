import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json

from global_variables import RESULTS_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 24})
cmap = matplotlib.colormaps["Set1"]
line_styles = ["solid", "dashed", "dotted"]
markers = ["o", "<", ">"]

f_dict = {0: "x", 1: "y", 2: "z"}


# %%
L = 8
shapes = 2*[[L, L]]
field_directions = 2*[0,]
labels = ["independent",
          "right_left_chaintrans_nonoise"]
legend_labels = ["independent", "right_left"]  # ["CRBM, alpha=1", "CRBM, alpha=2", "RBMSymm"]
eval_model = "ToricCRBM"
save_dir = f"{RESULTS_PATH}/toric2d_h/L={L}_final"

rhat_fig = plt.figure(dpi=300, figsize=(10, 10))
tau_fig = plt.figure(dpi=300, figsize=(10, 10))
arate_fig = plt.figure(dpi=300, figsize=(10, 10))

rhat_plot = rhat_fig.add_subplot(111)
tau_plot = tau_fig.add_subplot(111)
arate_plot = arate_fig.add_subplot(111)

for i, (fdir, label) in enumerate(zip(field_directions, labels)):
    shape_string = " ".join(map(str, shapes[i]))
    fname = f"{save_dir}/L={L}_mc_h{f_dict[field_directions[i]]}_{label}"
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

    rhat_plot.plot(iters, np.asarray(rhats), marker=markers[i], markersize=7,
                 color=cmap(i), label=legend_labels[i].replace("_","-"), linestyle=line_styles[i])
    tau_plot.plot(iters, np.asarray(taus), marker=markers[i], markersize=7,
                color=cmap(i), label=legend_labels[i].replace("_","-"), linestyle=line_styles[i])
    arate_plot.plot(iters, np.asarray(arates), marker=markers[i], markersize=7,
                  color=cmap(i), label=legend_labels[i].replace("_","-"), linestyle=line_styles[i])   

rhat_plot.set_xlabel(f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
tau_plot.set_xlabel(f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
arate_plot.set_xlabel(f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")

rhat_plot.set_ylabel("Split \$\\hat{R} \$")
tau_plot.set_ylabel("Auto-correlation time \$\\tau \$")
arate_plot.set_ylabel("Update acceptance rate")

rhat_plot.set_ylim(1, 1.4)
arate_plot.set_ylim(0, 1)

rhat_plot.legend()
tau_plot.legend()
arate_plot.legend()

rhat_fig.savefig(f"{save_dir}/rhats_L[{shape_string}]_{eval_model}_h{f_dict[field_directions[0]]}.svg")
tau_fig.savefig(f"{save_dir}/taus_L[{shape_string}]_{eval_model}_h{f_dict[field_directions[0]]}.svg")
arate_fig.savefig(f"{save_dir}/arates_L[{shape_string}]_{eval_model}_h{f_dict[field_directions[0]]}.svg")
