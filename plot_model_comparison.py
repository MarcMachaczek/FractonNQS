import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp
import json

from global_variables import RESULTS_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 12})
cmap = matplotlib.colormaps["Set1"]

save_dir = f"{RESULTS_PATH}/checkerboard/model_comparison"

# %%
shape = jnp.array([5, 5])
eval_models = ["FFNNf(2, 4)", "RBM", "RBMSymm", "SymmNNf(2, 4)", "CheckerCRBM"]
h = (0., 0., 0.)
E0 = -jnp.prod(shape)

# %% energy
fig = plt.figure(dpi=300, figsize=(10, 14))
plot_en = fig.add_subplot(311)

for i, eval_model in enumerate(eval_models):
    stats = json.load(open(f"{save_dir}/stats_L{shape}_{eval_model}_h{h}.json"))
    plot_en.errorbar(stats["Energy"]["iters"],
                     stats["Energy"]["Mean"]["real"],
                     yerr=stats["Energy"]["Sigma"],
                     label=eval_model)

plot_en.axhline(E0, color="dimgrey", linestyle="dashed")
plot_en.set_ylim(top=0., bottom=E0-1)
plot_en.set_xlim(left=0., right=stats["Energy"]["iters"][-1] + 1)
plot_en.set_xlabel("Training Iterations")
plot_en.set_ylabel("Energy")
plot_en.legend()
fig.savefig(f"{save_dir}/comparison_plot_L{shape}_{eval_model}_h{h}.svg")

# %% auxilliary
# eval_models = ["FFNNf(2, 4)", "RBM", "RBMSymm", "SymmNNf(2, 4)", "ToricCRBM"]
# fig = plt.figure(dpi=300, figsize=(10, 14))
# plot_aux = fig.add_subplot(311)

# for i, eval_model in enumerate(eval_models):
#     stats = json.load(open(f"{save_dir}/stats_L{shape}_{eval_model}_h{h}.json"))
#     plot_aux.errorbar(stats["stars"]["iters"],
#                      np.asarray(stats["stars"]["Mean"]["real"]) - np.asarray(stats["plaqs"]["Mean"]["real"]),
#                      yerr=np.asarray(stats["stars"]["Sigma"]) + np.asarray(stats["plaqs"]["Sigma"]),
#                      label=eval_model)

# plot_aux.set_ylim(top=0., bottom=E0/2-1)
# plot_aux.set_xlim(left=0., right=stats["Energy"]["iters"][-1] + 1)
# plot_aux.set_xlabel("Training Iterations")
# plot_aux.set_ylabel("\$\sum_\mathcal{V} A_{\mathcal{V}} - \sum_\mathcal{F} B_{\mathcal{F}}\$")
# plot_aux.legend()
# fig.savefig(f"{save_dir}/comparison_aux_plot_L{shape}_{eval_model}_h{h}.svg")