import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 20})
cmap = matplotlib.colormaps["Set1"]
line_styles = ["solid", "dashed", "dotted"]

colors = [cmap(0), cmap(0), cmap(1), cmap(1), cmap(2), cmap(2)]
line_styles = ["dotted", "dotted", "dashed", "dashed", "solid", "solid"]
markerstyles = ["<", ">"] + [4, 5] + [8, 9]
alpha = 0.7
ms = 10

f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/checkerboard/obs_comparison"

# %%
field_directions = 6*[0]  # 0=x, 1=y, 2=z
shapes = [[4, 4, 4], [4, 4, 4], [6, 6, 6], [6, 6, 6], [8, 8, 8], [8, 8, 8]]
labels = 3*["right_left", "left_right"]
legend_labels = ["\$L=4\$ right_left", "\$L=4\$ left_right",
                 "\$L=6\$ right_left", "\$L=6\$ left_right",
                 "\$L=8\$ right_left", "\$L=8\$ left_right"]
eval_model = "CheckerCRBM"
obs_list = []

# append multiple data to compare them within one plot
for i, shape in enumerate(shapes):
    shape_string = " ".join(map(str, shape))
    obs_list.append(
        pd.read_csv(f"{save_dir}/L[{shape_string}]_{eval_model}_observables_h{f_dict[field_directions[i]]}_{labels[i]}.txt", sep=" ", header=0))

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs.sort_values(by=[f"h{f_dict[field_directions[i]]}"])
direction = [obs.iloc[-1, :3].values for obs in obs_list]

# %%%%%%%%%%%%%%%%%%%%%% magnetizations comparison
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_mag = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    color = colors[i]
    plot_mag.errorbar(obs.iloc[:, field_directions[i]], obs["mag"], yerr=obs["mag_err"], marker=markerstyles[i], markersize=ms,
                      color=color, label=legend_labels[i].replace("_","-"), linestyle=line_styles[i], alpha=alpha)

plot_mag.set_xlabel(f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
plot_mag.set_ylabel("Magnetization \$ m \$ ")
plot_mag.legend()
fig.savefig(f"{save_dir}/mag_comparison_L{shape}_cRBM_{f_dict[field_directions[0]]}2.svg")

# %%%%%%%%%%%%%%%%%%%%%% susceptibility
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_sus = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    color = colors[i]
    sus, sus_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs["mag"].values, fields=obs.iloc[:, :3].values)
    plot_sus.plot(sus_fields[:, field_directions[i]], sus, marker=markerstyles[i], markersize=ms,
                  color=color, label=legend_labels[i].replace("_","-"), linestyle=line_styles[i], alpha=alpha)

plot_sus.set_xlabel(f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
plot_sus.set_ylabel("Susceptibility \$ \\chi \$ ")
plot_sus.legend()
fig.savefig(f"{save_dir}/susc_comparison_L{shape}_cRBM_{f_dict[field_directions[0]]}2.svg")

# %%%%%%%%%%%%%%%%%%%%%% energy per spin
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_energy = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    color = colors[i]
    hilbert_size = np.prod(shapes[i])
    plot_energy.errorbar(obs.iloc[:, field_directions[i]], obs["energy"].values / hilbert_size,
                         yerr=obs["energy_err"].values / hilbert_size, marker=markerstyles[i], markersize=ms,
                         color=color, label=legend_labels[i].replace("_","-"), linestyle=line_styles[i], alpha=alpha)

plot_energy.set_xlabel(f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
plot_energy.set_ylabel("Energy per spin \$ E/N \$")
plot_energy.legend()
fig.savefig(f"{save_dir}/epsite_comparison_L{shape}_cRBM_{f_dict[field_directions[0]]}2.svg")

# %%%%%%%%%%%%%%%%%%%%%% v-score of energy, see Wu et al:
# "Variational Benchmarks for Quantum Many-Body Problems"
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_vscore = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    color = colors[i]
    hilbert_size = np.prod(shapes[i])
    vscore = hilbert_size * np.abs(obs["energy_var"].values / obs["energy"].values**2)
    plot_vscore.plot(obs.iloc[:, field_directions[i]], vscore, marker=markerstyles[i], markersize=ms,
                   color=color, label=legend_labels[i].replace("_","-"), linestyle=line_styles[i], alpha=alpha)

plot_vscore.set_xlabel(f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
plot_vscore.set_ylabel("V-score = \$ N \\text{Var}( E ) / \\langle E \\rangle^2 \$")
plot_vscore.set_yscale("log")
plot_vscore.set_ylim(top=0.5, bottom=1e-7)
plot_vscore.legend()
fig.savefig(f"{save_dir}/vscore_comparison_L{shape}_cRBM_{f_dict[field_directions[0]]}2.svg")


# fig = plt.figure(dpi=300, figsize=(10, 10))
# plot_comps = fig.add_subplot(111)

# for i, obs in enumerate(obs_list):
#     field_comp = np.abs(obs["mag"] * np.prod(shapes[i]) / obs["energy"] * obs[f"h{f_dict[field_directions[i]]}"])
#     zcubes_comp = np.abs(obs["zcubes"] / obs["energy"])
#     xcubes_comp = np.abs(obs["xcubes"] / obs["energy"])
#     plot_comps.plot(obs.iloc[:, field_directions[i]], xcubes_comp, marker="o", markersize=3,
#                   color=cmap(i), label="x-cubes", linestyle=line_styles[i])
#     plot_comps.plot(obs.iloc[:, field_directions[i]], zcubes_comp, marker="o", markersize=3,
#                   color=cmap(i+1), label="z-cubes", linestyle=line_styles[i])
#     plot_comps.plot(obs.iloc[:, field_directions[i]], field_comp, marker="o", markersize=3,
#                   color=cmap(i+2), label="field", linestyle=line_styles[i])

# plot_comps.set_xlabel(f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
# plot_comps.set_ylabel("Contribution to \$ \\langle E \\rangle \$")
# # plot_comps.set_ylim(-1, 1)
# plot_comps.legend()
# fig.savefig(f"{save_dir}/hparts_comparison_L{shape}_cRBM_{f_dict[field_directions[0]]}.svg")
