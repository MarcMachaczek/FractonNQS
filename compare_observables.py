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
L = 3
hilbert_size = 2 * L ** 2
eval_model = "ToricCRBM"
save_dir = f"{RESULTS_PATH}/toric2d_h"
obs_list = []
# append multiple data to compare them each within one plot
obs_list.append(
    np.loadtxt(f"{save_dir}/L[{L} {L}]_{eval_model}_observables.txt"))
obs_list.append(
    np.loadtxt(f"{save_dir}/L[{L} {L}]_{eval_model}_observables_lr.txt"))

labels = ["independent", "left_right"]

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs[obs[:, field_direction].argsort()]
direction = obs_list[0][-1, :3]

# %% error comparison
# obs columns: "hx, hy, hz, energy, energy_var, mag, mag_var, abs_mag, abs_mag_var, wilson, wilson_var, exact_energy"
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    energies = obs[:, 3]
    exact_energies = obs[:, -1]
    rel_errors = np.abs(exact_energies - energies) / np.abs(exact_energies)

    plot.plot(obs[:, field_direction], rel_errors, marker="o", markersize=2, color=cmap(i), label=labels[i])
plot.set_xlabel(f"external field h{f_dict[field_direction]}")
plot.set_ylabel("relative energy error")
plot.set_title(
    f"energy error vs external field in {direction}-direction for ToricCode2d of shape=[{L},{L}]")
plot.legend()

plot.set_yscale("log")
plot.set_ylim(1e-7, 1e-1)

plt.show()
fig.savefig(f"{save_dir}/error_comparison_L[{L} {L}]_cRBM.pdf")
