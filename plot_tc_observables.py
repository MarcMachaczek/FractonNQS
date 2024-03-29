import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 24})
cmap = matplotlib.colormaps["Set1"]
line_styles = ["solid", "dashed", "dotted"]

f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/toric2d_h/L=10_final"

# %%
field_directions = 3*[2]
shapes = 3*[[10, 10]]
labels = ["independent", "right_left", "left_right"]
legend_labels = ["independent", "right_left", "left_right"]  # ["CRBM, alpha=1", "CRBM, alpha=2", "RBMSymm"]
eval_model = "ToricCRBM"
obs_list = []

# append multiple data to compare them each within one plot
for i, shape in enumerate(shapes):
    shape_string = " ".join(map(str, shape))
    obs_list.append(
        pd.read_csv(
            f"{save_dir}/L[{shape_string}]_{eval_model}_observables_h{f_dict[field_directions[i]]}_{labels[i]}.txt",
            sep=" ", header=0))

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs.sort_values(by=[f"h{f_dict[field_directions[i]]}"])
direction = [obs.iloc[-1, :3].values for obs in obs_list]

# %%%%%%%%%%%%%%%%%%%%%% magnetizations comparison
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_mag = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    plot_mag.errorbar(obs.iloc[:, field_directions[i]], obs["mag"], yerr=obs["mag_err"], marker="o", markersize=3,
                      color=cmap(i), label=legend_labels[i].replace("_", "-"), linestyle=line_styles[i])

plot_mag.set_xlabel(
    f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
plot_mag.set_ylabel("Magnetization \$ m \$ ")
plot_mag.legend()
fig.savefig(f"{save_dir}/mag_comparison_L{shape}_cRBM_{f_dict[field_directions[0]]}.svg")

# %%%%%%%%%%%%%%%%%%%%%% susceptibility
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_sus = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    sus, sus_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs["mag"].values, fields=obs.iloc[:, :3].values)
    plot_sus.plot(sus_fields[:, field_directions[i]], sus, marker="o", markersize=3,
                  color=cmap(i), label=legend_labels[i].replace("_", "-"), linestyle=line_styles[i])

plot_sus.set_xlabel(
    f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
plot_sus.set_ylabel("Susceptibility \$ \\chi \$ ")
plot_sus.legend()
fig.savefig(f"{save_dir}/susc_comparison_L{shape}_cRBM_{f_dict[field_directions[0]]}.svg")

# %%%%%%%%%%%%%%%%%%%%%% energy per spin
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_energy = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    hilbert_size = 2 * np.prod(shapes[i])
    plot_energy.errorbar(obs.iloc[:, field_directions[i]], obs["energy"].values / hilbert_size,
                         yerr=obs["energy_err"].values / hilbert_size, marker="o", markersize=3,
                         color=cmap(i), label=legend_labels[i].replace("_", "-"), linestyle=line_styles[i])

plot_energy.set_xlabel(
    f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
plot_energy.set_ylabel("Energy per spin \$ E/N \$")
plot_energy.legend()
fig.savefig(f"{save_dir}/epsite_comparison_L{shape}_cRBM_{f_dict[field_directions[0]]}.svg")

# %%%%%%%%%%%%%%%%%%%%%% v-score of energy, see Wu et al:
# "Variational Benchmarks for Quantum Many-Body Problems"
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_vscore = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    hilbert_size = 2 * np.prod(shapes[i])
    vscore = hilbert_size * np.abs(obs["energy_var"].values / obs["energy"].values ** 2)
    plot_vscore.plot(obs.iloc[:, field_directions[i]], vscore, marker="o", markersize=3,
                     color=cmap(i), label=legend_labels[i].replace("_", "-"), linestyle=line_styles[i])

plot_vscore.set_xlabel(
    f"Field strength in \$ {f_dict[field_directions[0]]} \$-direction \$ h_{f_dict[field_directions[0]]} \$ ")
plot_vscore.set_ylabel("V-score = \$ N \\text{Var}( E ) / \\langle E \\rangle^2 \$")
plot_vscore.set_yscale("log")
plot_vscore.set_ylim(top=0.5, bottom=1e-7)
plot_vscore.legend()
fig.savefig(f"{save_dir}/vscore_comparison_L{shape}_cRBM_{f_dict[field_directions[0]]}.svg")

# %% wilson loop (see valenti et al)
# plot_wilson = fig.add_subplot(324)

# for i, obs in enumerate(obs_list):
#     plot_wilson.errorbar(obs.iloc[:, field_direction[i]], obs["wilson"], yerr=obs["wilson_err"], marker="o",
#                          markersize=2, color=cmap(i))

#     plot_wilson.plot(obs.iloc[:, field_direction[i]], obs["wilson"], marker="o", markersize=2, color=cmap(i),
#                      label=f"hdir={f_dict[field_direction[i]]}_{labels[i]}")

# plot_wilson.set_xlabel(f"external field")
# plot_wilson.set_ylabel("wilson loop < prod A_s * prod B_p >")
# plot_wilson.set_title(
#     f"wilson loop (see Valenti et al) vs external field for ToricCode2d")
# plot_wilson.legend()
