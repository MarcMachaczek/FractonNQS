import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp
import json

from global_variables import RESULTS_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 12})  # 24
cmap = matplotlib.colormaps["Set1"]

save_dir = f"{RESULTS_PATH}/toric2d_h"

# %%
shape = jnp.array([3, 3])
eval_models = ["FFNNf(2, 4)", "RBM", "RBMSymm", "SymmNNf(2, 4)", "ToricCRBM"]
h = (0., 0., 0.)
E0 = -jnp.prod(shape)*2

# %% energy
fig = plt.figure(dpi=300, figsize=(10, 14))
plot_en = fig.add_subplot(311)

for i, eval_model in enumerate(eval_models):
    stats = json.load(open(f"{save_dir}/stats_L{shape}_{eval_model}_h{h}.json"))
    if i==3:
        plot_en.plot(stats["Energy"]["iters"], stats["Energy"]["Mean"]["real"], label=eval_model, linestyle="dashdot")
    else:
        plot_en.plot(stats["Energy"]["iters"], stats["Energy"]["Mean"]["real"], label=eval_model)

plot_en.axhline(E0, color="dimgrey", linestyle="dashed")
plot_en.set_ylim(top=0., bottom=E0-1)
plot_en.set_xlim(left=0., right=stats["Energy"]["iters"][-1] + 1)
plot_en.set_xlabel("Training Iterations")
plot_en.set_ylabel("Energy")
plot_en.legend()
# %%
fig.savefig(f"{save_dir}/comparison_plot_L{shape}_{eval_model}_h{h}.svg")

