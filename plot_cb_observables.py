import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from global_variables import RESULTS_PATH

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 32,  # 32
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})
# matplotlib.rcParams['svg.fonttype'] = 'none'
# matplotlib.rcParams.update({'font.size': 30})
cmap = matplotlib.colormaps["Set1"]

colors = [cmap(1), cmap(1), cmap(2), cmap(2)]  # [cmap(0), cmap(0), cmap(1), cmap(1), cmap(2), cmap(2)]
line_styles = ["dashed", "dashed", "solid", "solid"]  # ["dotted", "dotted", "dashed", "dashed", "solid", "solid"]  # ["dotted", "dashed"]
markerstyles = 2 * ["<", ">"]  # ["<", ">"] + [4, 5] + [8, 9]
alpha = 0.8  # 0.8 transparency
ms = 12  # 12 marker size

f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/checkerboard/obs_comparison"  # obs_comparison

# %%
field_directions = 2 * [0] + 2 * [2]  # 6 * [0]  # 0=x, 1=y, 2=z
shapes = 4 * [[4, 4, 4]]  # [[4, 4, 4], [4, 4, 4], [6, 6, 6], [6, 6, 6], [8, 8, 8], [8, 8, 8]]
labels = 2 * ["right_left", "left_right"]  # 3 * ["right_left", "left_right"]
# legend_labels = [r"$L=4$ right_left", r"$L=4$ left_right",
#                  r"$L=6$ right_left", r"$L=6$ left_right",
#                  r"$L=8$ right_left", r"$L=8$ left_right"]

legend_labels = [r"$h_x$ right_left", r"$h_x$ left_right", r"$h_z$ right_left", r"$h_z$ left_right"]

line_styles = line_styles[::-1]
colors = colors[::-1]
markerstyles = markerstyles[::-1]
shapes = shapes[::-1]
labels = labels[::-1]
legend_labels = legend_labels[::-1]

suffix = "xz"
# zoom
# x_limits = (0.25, 0.55)
# y_limits = (-1.09, -1.0)  # (-1.123, -0.956)
# no zoom
x_limits = (-0.025, 0.925)
y_limits = (-1.45, -0.925)
eval_model = "CheckerCRBM_2"
obs_list = []

# append multiple data to compare them within one plot
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
fig = plt.figure(dpi=300, figsize=(12, 12))
plot_mag = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    color = colors[i]
    plot_mag.errorbar(obs.iloc[:, field_directions[i]], obs["mag"], yerr=obs["mag_err"], marker=markerstyles[i], markersize=ms,
                      color=color, label=legend_labels[i].replace("_", "_"), linestyle=line_styles[i], linewidth=2.5, alpha=alpha)

plot_mag.set_xlim(*x_limits)
plot_mag.set_xlabel(r"Field strength $|h|$")
    # rf"Field strength in ${f_dict[field_directions[0]]}$-direction $h_{f_dict[field_directions[0]]}$")
plot_mag.set_ylabel(r"Magnetization $m$")
plot_mag.legend(fontsize=26)

fig.subplots_adjust(left=0.15)
fig.savefig(f"{save_dir}/mag_comparison_L{shapes[0]}_{eval_model}_{f_dict[field_directions[0]]}_{suffix}.pdf")

# %%%%%%%%%%%%%%%%%%%%%% susceptibility
# fig = plt.figure(dpi=300, figsize=(11, 11))
# plot_sus = fig.add_subplot(111)
#
# for i, obs in enumerate(obs_list):
#     color = colors[i]
#     sus, sus_fields = geneqs.utils.eval_obs.derivative_fd(observable=obs["mag"].values, fields=obs.iloc[:, :3].values)
#     plot_sus.plot(sus_fields[:, field_directions[i]], sus, marker=markerstyles[i], markersize=ms,
#                   color=color, label=legend_labels[i].replace("_", "-"), linestyle=line_styles[i], linewidth=2.0, alpha=alpha)
#
# plot_sus.set_xlim(*x_limits)
# plot_sus.set_xlabel(
#     rf"Field strength in ${f_dict[field_directions[0]]}$-direction $h_{f_dict[field_directions[0]]}$")
# plot_sus.set_ylabel(r"Susceptibility $\chi$")
# plot_sus.legend(fontsize=26)
#
# fig.subplots_adjust(left=0.15)
# fig.savefig(f"{save_dir}/susc_comparison_L468_{eval_model}_{f_dict[field_directions[0]]}_{suffix}.pdf")

# %%%%%%%%%%%%%%%%%%%%%% energy per spin
fig = plt.figure(dpi=300, figsize=(11, 11))
plot_energy = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    color = colors[i]
    hilbert_size = np.prod(shapes[i])
    plot_energy.errorbar(obs.iloc[:, field_directions[i]], obs["energy"].values / hilbert_size,
                         yerr=obs["energy_err"].values / hilbert_size, marker=markerstyles[i], markersize=ms,
                         color=color, label=legend_labels[i].replace("_", "_"), linestyle=line_styles[i], linewidth=2.5, alpha=alpha)

plot_energy.set_xlim(*x_limits)
plot_energy.set_ylim(*y_limits)
plot_energy.set_xlabel(r"Field strength $|h|$")
    # rf"Field strength in ${f_dict[field_directions[0]]}$-direction $h_{f_dict[field_directions[0]]}$")
plot_energy.set_ylabel(r"Energy per spin $E/N$")
plot_energy.legend(fontsize=26)
plot_energy.locator_params(nbins=7, axis='x')

fig.subplots_adjust(left=0.15)
fig.savefig(f"{save_dir}/epsite_comparison_L{shapes[0]}_{eval_model}_{f_dict[field_directions[0]]}_{suffix}.pdf")

# %%%%%%%%%%%%%%%%%%%%%% v-score of energy, see Wu et al:
# "Variational Benchmarks for Quantum Many-Body Problems"
fig = plt.figure(dpi=300, figsize=(12, 12))
plot_vscore = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    color = colors[i]
    hilbert_size = np.prod(shapes[i])
    vscore = hilbert_size * np.abs(obs["energy_var"].values / obs["energy"].values**2)
    plot_vscore.plot(obs.iloc[:, field_directions[i]], vscore, marker=markerstyles[i], markersize=ms,
                     color=color, label=legend_labels[i].replace("_", "_"), linestyle=line_styles[i], linewidth=3, alpha=alpha)

plot_vscore.set_xlim(*x_limits)
plot_vscore.set_xlabel(#r"Field strength $|h|$")
    rf"Field strength in ${f_dict[field_directions[0]]}$-direction $h_{f_dict[field_directions[0]]}$")
plot_vscore.set_ylabel(r"V-score = $N \mathrm{Var}(E) / E^2 $")
plot_vscore.set_yscale("log")
plot_vscore.set_ylim(top=0.5, bottom=1e-7)
plot_vscore.legend()

fig.subplots_adjust(left=0.15)
fig.savefig(f"{save_dir}/vscore_L{shapes[0]}_{eval_model}_{f_dict[field_directions[0]]}_{suffix}.pdf")

# %% Relative energy contributions to Hamiltonian
fig = plt.figure(dpi=300, figsize=(12, 12))
plot_comps = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    field_comp = np.abs(obs["mag"] * np.prod(shapes[i]) / obs["energy"] * obs[f"h{f_dict[field_directions[i]]}"])
    zcubes_comp = np.abs(obs["zcubes"] / obs["energy"])
    xcubes_comp = np.abs(obs["xcubes"] / obs["energy"])
    plot_comps.plot(obs.iloc[:, field_directions[i]], xcubes_comp, marker="o", markersize=ms,
                    color=cmap(i), label=r"$\sum_\mathcal{C}A_\mathcal{C} / E$", linestyle=line_styles[i], linewidth=3.0, alpha=alpha)
    plot_comps.plot(obs.iloc[:, field_directions[i]], zcubes_comp, marker="o", markersize=ms,
                    color=cmap(i+1), label=r"$\sum_\mathcal{C}B_\mathcal{C} / E$", linestyle=line_styles[i], linewidth=3.0, alpha=alpha)
    plot_comps.plot(obs.iloc[:, field_directions[i]], field_comp, marker="o", markersize=ms,
                    color=cmap(i+2), label=r"$h_x \sum_i \sigma_i^{x} / E$", linestyle=line_styles[i], linewidth=3.0, alpha=alpha)

plot_comps.set_xlabel(
    rf"Field strength in ${f_dict[field_directions[0]]}$-direction $h_{f_dict[field_directions[0]]}$")
plot_comps.set_ylabel("Relative energy contribution")
# plot_comps.set_ylim(-1, 1)
plot_comps.legend()

fig.subplots_adjust(left=0.15)
fig.savefig(f"{save_dir}/hparts_comparison_L{shapes[0]}_cRBM_{f_dict[field_directions[0]]}_{suffix}.pdf")
