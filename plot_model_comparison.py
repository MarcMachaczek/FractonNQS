import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import json

from global_variables import RESULTS_PATH

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 40,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"]
})

# matplotlib.rcParams['svg.fonttype'] = 'none'
# matplotlib.rcParams.update({'font.size': 20})
cmap = matplotlib.colormaps["Set1"]
line_styles = ["solid", "dashed", (0, (5, 10)), "dashdot", "dotted"]

save_dir = f"{RESULTS_PATH}/checkerboard/model_comparison"

# %%
shape = np.array([4, 2, 2])
eval_models = ["FFNN(8, 8)", "RBM", "RBMSymm", "SymmNN(4, 4)", "cRBM"]
h = (0., 0., 0.)
E0 = -np.prod(shape)

# %% energy
fig = plt.figure(dpi=300, figsize=(12, 12))
plot_en = fig.add_subplot(111)

for i, eval_model in enumerate(eval_models):
    stats = json.load(open(f"{save_dir}/stats_L{shape}_{eval_model}_h{h}.json"))
    plot_en.errorbar(stats["Energy"]["iters"],
                     stats["Energy"]["Mean"]["real"],
                     yerr=stats["Energy"]["Sigma"],
                     label=rf"{eval_model}",
                     linestyle=line_styles[i],
                     linewidth=6.0)

plot_en.axhline(E0, color="dimgrey", linestyle="dashed")
plot_en.set_ylim(top=-7, bottom=E0 - 0.5)
plot_en.set_xlim(left=0., right=stats["Energy"]["iters"][-1] + 1)
plot_en.set_xlabel(r"Training iterations")
plot_en.set_ylabel(r"Energy $E$")
plot_en.legend(fontsize=30)

fig.subplots_adjust(left=0.15)
fig.savefig(f"{save_dir}/comparison_plot_L{shape}_{eval_model}_h{h}.pdf")

# %% auxilliary
# eval_models = ["FFNNf(2, 4)", "RBM", "RBMSymm", "SymmNNf(2, 4)", "ToricCRBM"]
# fig = plt.figure(dpi=300, figsize=(20, 10))
# plot_aux = fig.add_subplot(111)

# for i, eval_model in enumerate(eval_models):
#     stats = json.load(open(f"{save_dir}/stats_L{shape}_{eval_model}_h{h}.json"))
#     plot_aux.errorbar(stats["stars"]["iters"],
#                       -np.asarray(stats["stars"]["Mean"]["real"]) + np.asarray(stats["plaqs"]["Mean"]["real"]),
#                       yerr=np.asarray(stats["stars"]["Sigma"]) + np.asarray(stats["plaqs"]["Sigma"]),
#                       label=eval_model)

# plot_aux.set_ylim(top=-E0 / 2 + 1, bottom=0)
# plot_aux.set_xlim(left=0., right=stats["Energy"]["iters"][-1] + 1)
# plot_aux.set_xlabel("Training Iterations")
# plot_aux.set_ylabel("\$\sum_\mathcal{V} A_{\mathcal{V}} - \sum_\mathcal{F} B_{\mathcal{F}}\$")
# plot_aux.legend(loc="right")
# fig.savefig(f"{save_dir}/comparison_aux_plot_L{shape}_{eval_model}_h{h}.svg")
