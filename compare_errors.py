import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import geneqs.utils.eval_obs
from global_variables import RESULTS_PATH

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams.update({'font.size': 36})
cmap = matplotlib.colormaps["Set1"]
line_styles = ["solid", "dashed", "dotted"]

f_dict = {0: "x", 1: "y", 2: "z"}
# save_dir = f"{RESULTS_PATH}/checkerboard/vsed_final"
save_dir = f"{RESULTS_PATH}/toric2d_h/L=3_final"

# %%
field_directions = [0, 1, 2]  # 0=x, 1=y, 2=z
# shapes = 3*[[4, 2, 2]]
shapes = 3*[[3, 3]]
# labels = ["independent", "right_left", "left_right"]
labels = ["independent", "independent", "independent"]
legend_labels = ["x-direction", "y-direction", "z-direction"]  # =labels
# eval_model = "CheckerCRBM"
eval_models = 3*["ToricCRBM",]
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
fig = plt.figure(dpi=300, figsize=(10, 10))
plot_mag = fig.add_subplot(111)

for i, obs in enumerate(obs_list):
    c=cmap(0)
    if i == 1:
        c = cmap(2)
    elif i == 2:
        c = cmap(1)
    rel_errors = np.abs(obs["exact_energy"] - obs["energy"]) / np.abs(obs["exact_energy"])
    plot_mag.plot(obs.iloc[:, field_directions[i]], rel_errors, marker="o", markersize=2,
                  color=c, label=legend_labels[i].replace("_","-"), linestyle=line_styles[i])

plot_mag.set_xlabel("Field strength \$|h|\$")
plot_mag.set_ylabel("\$ |E_{\\boldsymbol{\\theta}}-E_\\mathrm{exact}| / |E_\\mathrm{exact}| \$ ")
plot_mag.set_yscale("log")
#plot_mag.set_ylim(1e-9, 1e-3)
plot_mag.legend()
fig.savefig(f"{save_dir}/error_comparison_L{shape}_tcdirections_h{f_dict[field_directions[0]]}.svg")