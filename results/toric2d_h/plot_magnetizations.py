from matplotlib import pyplot as plt
import numpy as np
from global_variables import RESULTS_PATH

# %%
mags = np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L=10_real_rbm3/L[10 10]_real_rbm_symm_a2_magvals")
mags_hx = np.loadtxt(f"{RESULTS_PATH}/toric2d_h/L=10_real_rbm_hx03/L[10 10]_real_rbm_symm_a2_magvals")


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
plot.set_title(f"magnetization vs external field in z-direction for ToricCode2d of shape=[10,10]")
plot.legend()

plt.show()

fig.savefig(f"{RESULTS_PATH}/toric2d_h/L[10,10]_real_rbm_symm_a2_mags_comparison.pdf")
