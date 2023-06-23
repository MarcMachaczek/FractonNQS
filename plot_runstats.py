import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp
import json

from global_variables import RESULTS_PATH

matplotlib.rcParams.update({'font.size': 12})

cmap = matplotlib.colormaps["Set1"]
save_dir = f"{RESULTS_PATH}/toric2d_h"

# %%
shape = jnp.array([3, 3])
eval_model = "ToricCRBM"
h = (0., 0., 0.)
stats = json.load(open(f"{save_dir}/stats_L{shape}_{eval_model}_h{h}.json"))

# %% acceptance rate
fig = plt.figure(dpi=300, figsize=(10, 30))
plot_ar = fig.add_subplot(311)

plot_ar.plot(stats["acceptance_rate"]["iters"], stats["acceptance_rate"]["value"])

plot_ar.set_xlabel("Training Iterations")
plot_ar.set_ylabel("Acceptance Rate")
plot_ar.set_ylim(0, 1)

# %% R_hat
plot_rh = fig.add_subplot(312)

plot_rh.plot(stats["Energy"]["iters"], stats["Energy"]["R_hat"])

plot_rh.set_xlabel("Training Iterations")
plot_rh.set_ylabel("R_hat")
plot_ar.set_ylim(1, 1.4)

# %% auto-correlation time
plot_act = fig.add_subplot(313)

plot_act.plot(stats["Energy"]["iters"], stats["Energy"]["TauCorr"])

plot_act.set_xlabel("Training Iterations")
plot_act.set_ylabel("Auto-correlation")

plt.show()
# %%
fig.savefig(f"{save_dir}/stats_plot_L{shape}_{eval_model}_h{h}.pdf")

