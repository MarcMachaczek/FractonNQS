import jax.random
from matplotlib import pyplot as plt

import jax.numpy as jnp
import netket as nk

import geneqs
from geneqs.utils.training import approximate_gs
from global_variables import RESULTS_PATH

# %% Define graph/lattice and hilbert space
L = 3  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)

# own custom hamiltonian
field_strength = -1.
toric = geneqs.operators.toric_2d.ToricCode2d_H(hilbert, shape, field_strength)

# visualize the graph
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = fig.add_subplot(111)
square_graph.draw(ax)
plt.show()

# get (specific) symmetries of the model, in our case translations
perms = geneqs.utils.indexing.get_translations_cubical2d(shape)
link_perms = geneqs.utils.indexing.get_linkperms_cubical2d(perms)
# must be hashable to be included as flax.module attribute
link_perms = nk.utils.HashableArray(link_perms.astype(int))

training_records = {}
# %% setting hyper-parameters
n_iter = 300
sampler_args = {"n_chains": 256*2,  # total number of MCMC chains, can be large when runnning on GPU  ~O(1000)
                "n_samples": 256*8,
                "n_discard_per_chain": 8}  # should be small when using many chains, default is 10% of n_samples
# n_sweeps will default to n_sites, every n_sweeps (updates) a sample will be generated

diag_shift = 0.01
preconditioner = nk.optimizer.SR(diag_shift=diag_shift)

alpha = 2
features = (int(alpha*hilbert.size/link_perms.shape[0]), 1)

learning_rates = {"symmetric_mlp": 0.01,
                  "rbm": 0.01,
                  "symm_rbm": 0.01,
                  "rbm_modphase": 0.02,
                  "rbm_symm_modphase": 0.01}

default_kernel_init = jax.nn.initializers.normal(0.01)

# %% simple (symmetric) FFNN ansatz
model = geneqs.models.symmetric_networks.SymmetricNN(link_perms, features)
optimizer = nk.optimizer.Sgd(learning_rates["symmetric_mlp"])
vqs_mlp, data_mlp = approximate_gs(hilbert, model, toric, optimizer, preconditioner, n_iter, **sampler_args)
training_records["symmetric_mlp"] = data_mlp

#%% train regular RBM
model = nk.models.RBM(alpha=alpha)
optimizer = nk.optimizer.Sgd(learning_rates["rbm"])
vqs_rbm, data_rbm = approximate_gs(hilbert, model, toric, optimizer, preconditioner, n_iter, **sampler_args)
training_records["rbm"] = data_rbm

# %% train symmetric RBM
model = nk.models.RBMSymm(symmetries=link_perms, alpha=alpha, kernel_init=default_kernel_init)
optimizer = nk.optimizer.Sgd(learning_rates["symm_rbm"])
vqs_symm, data_symm = approximate_gs(hilbert, model, toric, optimizer, preconditioner, n_iter, **sampler_args)
training_records["symm_rbm"] = data_symm

# %% train complex RBM
model = nk.models.RBMModPhase(alpha=alpha)
optimizer = nk.optimizer.Sgd(learning_rates["rbm_modphase"])
vqs_mp, data_mp = approximate_gs(hilbert, model, toric, optimizer, preconditioner, n_iter, **sampler_args)
training_records["rbm_modphase"] = data_mp

# %% train symmetric complex RBM
model = geneqs.models.RBMModPhaseSymm(symmetries=link_perms, alpha=alpha)
optimizer = nk.optimizer.Sgd(learning_rates["rbm_symm_modphase"])
vqs_symm_mp, data_symm_mp = approximate_gs(hilbert, model, toric, optimizer, preconditioner, n_iter, **sampler_args)
training_records["rbm_symm_modphase"] = data_symm_mp

# %%
fig = plt.figure(dpi=300, figsize=(10, 10))
plot = fig.add_subplot(111)

for model, record in training_records.items():
    n_params = int(record["n_params"].value)
    plot.errorbar(record["Energy"].iters, record["Energy"].Mean, yerr=record["Energy"].Sigma,
                  label=f"{model}, lr={learning_rates[model]}, #p={n_params}")


n_chains = sampler_args["n_chains"]
n_samples = sampler_args["n_samples"]
fig.suptitle(f"ToricCode2d VMC: size={shape},"
             f" single spin flip updates,"
             f" alpha={alpha},"
             f" n_sweeps={L**2*2},"
             f" n_chains={n_chains},"
             f" n_samples={n_samples}")

plot.set_xlabel("iterations")
plot.set_ylabel("energy")

plot.set_title(f"using stochastic gradient descent with stochastic reconfiguration, diag_shift={diag_shift}")
plot.legend()

plt.show()

fig.savefig(f"{RESULTS_PATH}/toric2d_h/VMC_lattice{shape}_h{field_strength}.pdf")

# %%
# max acceptance rates for symm_mlp, symm_rbm, symm_rbm_mp
# L=3: 0.999, 0.416, 0.995
# L=4: 0.999, 0.674, 0.992
# L=6: 0.999, 0.651, 0.982
# L=8: 0.999, 0.903, 0.984
a = data_symm_mp["acceptance_rate"].values
print(a, jnp.max(a))

# notes for later: decrease learning rate with system size, acceptance pretty low for symm_rbm
# learning rate scheduler?, early stopping?, gradient clipping?
