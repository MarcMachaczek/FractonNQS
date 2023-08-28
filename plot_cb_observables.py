import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 12})
cmap = matplotlib.colormaps["Set1"]
line_styles = ["solid", "dashed", "dotted"]

f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/checkerboard/L=4_final"

# %%
field_directions = 3*[0]  # 0=x, 1=y, 2=z
shapes = 3*[[4, 4, 4]]
labels = ["independent", "right_left", "left_right"]
legend_labels = labels  # ["CRBM, alpha=1", "CRBM, alpha=2", "RBMSymm"]
eval_model = "CheckerCRBM"
obs_list = []

# append multiple data to compare them each within one plot
for i, shape in enumerate(shapes):
    shape_string = " ".join(map(str, shape))
    obs_list.append(
        pd.read_csv(f"{save_dir}/L[{shape_string}]_{eval_model}_observables_h{f_dict[field_directions[i]]}_{labels[i]}.txt", sep=" ", header=0))

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs.sort_values(by=[f"h{f_dict[field_directions[i]]}"])
direction = [obs.iloc[-1, :3].values for obs in obs_list]

# %% magnetizations comparison
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_mag = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    plot_mag.errorbar(obs.iloc[:, field_directions[i]], obs["mag"], yerr=obs["mag_err"], marker="o", markersize=2,
                      color=cmap(i), label=legend_labels[i], linestyle=line_styles[i])

plot_mag.set_xlabel("Field strength in \$ x \$-direction \$ h_x \$ ")
plot_mag.set_ylabel("Magnetization \$ m \$ ")
plot_mag.legend()
fig.savefig(f"{save_dir}/mag_comparison_L{shape}_cRBM.svg")

# %% susceptibility
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_sus = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    sus, sus_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs["mag"].values, fields=obs.iloc[:, :3].values)
    plot_sus.plot(sus_fields[:, field_directions[i]], sus, marker="o", markersize=2,
                  color=cmap(i), label=legend_labels[i], linestyle=line_styles[i])

plot_sus.set_xlabel("Field strength in \$ x \$-direction \$ h_x \$ ")
plot_sus.set_ylabel("Susceptibility \$ \\xi \$ ")
plot_sus.legend()
fig.savefig(f"{save_dir}/susc_comparison_L{shape}_cRBM.svg")

# %% energy per site
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_energy = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    hilbert_size = np.prod(shapes[i])
    plot_energy.errorbar(obs.iloc[:, field_directions[i]], obs["energy"].values / hilbert_size,
                         yerr=obs["energy_err"].values / hilbert_size, marker="o", markersize=2,
                         color=cmap(i), label=legend_labels[i], linestyle=line_styles[i])

plot_energy.set_xlabel("Field strength in \$ x \$-direction \$ h_x \$ ")
plot_energy.set_ylabel("Energy per spin")
plot_energy.legend()
fig.savefig(f"{save_dir}/epsite_comparison_L{shape}_cRBM.svg")

# %% variance of the energy
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_evar = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    plot_evar.plot(obs.iloc[:, field_directions[i]], np.abs(obs["energy_var"].values), marker="o", markersize=2,
                     color=cmap(i), label=legend_labels[i], linestyle=line_styles[i])

plot_evar.set_xlabel("Field strength in \$ x \$-direction \$ h_x \$ ")
plot_evar.set_ylabel("Var(\$ E \$)")
plot_evar.legend()
fig.savefig(f"{save_dir}/evar_comparison_L{shape}_cRBM.svg")

# # %% energy derivative dE/dh
# plot_dEdh = fig.add_subplot(235)

# for i, obs in enumerate(obs_list):
#     dEdh, fd_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs["energy"].values,
#                                                           fields=obs.iloc[:, :3].values)
#     plot_dEdh.plot(fd_fields[:, field_direction[i]], dEdh, marker="o", markersize=2, color=cmap(i),
#                    label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

# plot_dEdh.set_xlabel(f"external field")
# plot_dEdh.set_ylabel("dE / dh")
# plot_dEdh.set_title(
#     f"energy derivative vs external field for Checkerboard")
# plot_dEdh.legend()



# # %% absolute magnetization
# plot_abs_mag = fig.add_subplot(232)

# for i, obs in enumerate(obs_list):
#     plot_abs_mag.errorbar(obs.iloc[:, field_direction], obs["abs_mag"], yerr=obs["abs_mag_err"], marker="o",
#                           markersize=2, color=cmap(i))

#     plot_abs_mag.plot(obs.iloc[:, field_direction], obs["abs_mag"], marker="o", markersize=2, color=cmap(i),
#                       label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

# plot_abs_mag.set_xlabel(f"external field")
# plot_abs_mag.set_ylabel("absolute magnetization")
# plot_abs_mag.set_title(
#     f" absolute magnetization vs external field for Checkerboard")
# plot_abs_mag.legend()