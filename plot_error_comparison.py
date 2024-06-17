import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from global_variables import RESULTS_PATH

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 40,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})
# matplotlib.rcParams['svg.fonttype'] = 'none'
# matplotlib.rcParams.update({'font.size': 30})  # 36

cmap = matplotlib.colormaps["Set1"]
line_styles = ["solid", "dashed", "dotted"]
markerstyles = ["o", "<", ">"]

f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/checkerboard/vsed_final"
# save_dir = f"{RESULTS_PATH}/toric2d_h/L=3_final"

# %%
field_directions = 3*[2]  # [0, 0, 0]  # 0=x, 1=y, 2=z
shapes = 3*[[4, 2, 2]]
# shapes = 3*[[3, 3]]
labels = ["independent", "right_left", "left_right"]
# labels = ["independent", "independent", "independent"]
# legend_labels = [r"x-direction", r"y-direction", r"z-direction"]
legend_labels = labels
eval_models = 3*["CheckerCRBM_2",]
# eval_models = 3*["ToricCRBM_2",]
obs_list = []

# append multiple data to compare them each within one plot
for i, shape in enumerate(shapes):
    shape_string = " ".join(map(str, shape))
    obs_list.append(
        pd.read_csv(f"{save_dir}/L[{shape_string}]_{eval_models[i]}_observables_h{f_dict[field_directions[i]]}_{labels[i]}.txt", sep=" ", header=0))

# order by increasing field strength
for i, obs in enumerate(obs_list):
    obs_list[i] = obs.sort_values(by=[f"h{f_dict[field_directions[i]]}"])
direction = [obs.iloc[-1, :3].values for obs in obs_list]

# %% energy error comparison
fig = plt.figure(dpi=300, figsize=(12, 12))
plot_mag = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    c = cmap(i)
    rel_errors = np.abs(obs["exact_energy"] - obs["energy"]) / np.abs(obs["exact_energy"])
    plot_mag.plot(obs.iloc[:, field_directions[i]], rel_errors, markersize=12, marker=markerstyles[i],
                  color=c, label=legend_labels[i].replace("_", "-"),
                  linewidth=3.0, linestyle=line_styles[i])

plot_mag.set_xlabel(rf"Magnetic field in {f_dict[field_directions[i]]}-direction $h_{f_dict[field_directions[i]]}$")  # f"Field strength \$|h|\$"
#plot_mag.set_ylabel(r"$|E_{\boldsymbol{\theta}}-E_\mathrm{exact}| / |E_\mathrm{exact}| $ ")
plot_mag.set_ylabel(r"$|E_{\boldsymbol{\theta}} - E_\mathrm{exact}| / |E_\mathrm{exact}|$ ")
plot_mag.set_yscale("log")
#plot_mag.set_ylim(1e-9, 1e-3)
plot_mag.legend(fontsize=40)

fig.subplots_adjust(left=0.16)
fig.savefig(f"{save_dir}/error_comparison_L{shape}_cRBM_h{f_dict[field_directions[0]]}.pdf")
