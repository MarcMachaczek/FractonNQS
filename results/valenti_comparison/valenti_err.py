import numpy as np
from matplotlib import pyplot as plt

vmc_ens = np.loadtxt("./valenti_sd_energies_crbm.txt")
fields = vmc_ens[:, 0]
crbm_ens = vmc_ens[:, 1]
exact_ens = np.loadtxt("./valenti_sd_energies_exact.txt")
exact_ens = exact_ens[:, 0]

rel_errs = np.abs(crbm_ens - exact_ens) / np.abs(exact_ens)
err_fig = plt.figure()
err_plot = err_fig.add_subplot(111)

err_plot.plot(fields, rel_errs)
err_plot.set_yscale("log")

err_fig.savefig("./rel_errs.pdf")