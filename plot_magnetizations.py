from matplotlib import pyplot as plt
import numpy as np
from global_variables import RESULTS_PATH

# %%
L = 10
mags = np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L={L}_complex_crbm_hx0_1/L[{L} {L}]_cRBM_a1_magvals")
mags_hx = np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L={L}_complex_crbm_hx03_4/L[{L} {L}]_cRBM_a1_magvals")


fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)
for mag in mags:
    plot.errorbar(mag[2], np.abs(mag[3]), yerr=mag[4], marker="o", markersize=2, color="red")

for mag in mags_hx:
    plot.errorbar(mag[2], np.abs(mag[3]), yerr=mag[4], marker="o", markersize=2, color="blue")

plot.plot(mags[:, 2], np.abs(mags[:, 3]), marker="o", markersize=2, color="red", label="hx=0")
plot.plot(mags_hx[:, 2], np.abs(mags_hx[:, 3]), marker="o", markersize=2, color="blue", label="hx=0.3")

plot.set_xlabel("external field hz")
plot.set_ylabel("magnetization")
plot.set_title(f"magnetization vs external field in z-direction for ToricCode2d of shape=[{L},{L}]")
plot.legend()

plt.show()

fig.savefig(f"{RESULTS_PATH}/toric2d_h/L[{L} {L}]_cRBM_a1_mags_comparison.pdf")
