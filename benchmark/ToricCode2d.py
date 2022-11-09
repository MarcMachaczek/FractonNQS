import jax.numpy as jnp
import netket as nk
import geneqs

# %% get benchmarking settings
config = geneqs.utils.benchmarking.parse_settings()
config = config["OPERATOR_BENCHMARK"]
print("operator benchmark settings: ", config, "\n")

# %% define graph, hilbert space and vqs
L = config["lattice_size"]  # size should be at least 3, else there are problems with pbc and indexing
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=square_graph.n_edges)

# define a basic variational quantum state
# TODO: possibly provide vqs settings in the settings.yml file
vqs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), nk.models.RBM(alpha=2))
vqs.n_samples = config["n_samples"]
n_runs = config["n_runs"]

# %% define Hamiltonians
# start with the netket hamiltonian constructed from local operators
ha_netketlocal = nk.operator.LocalOperator(hi)
# adding the plaquette terms:
for i in range(L):
    for j in range(L):
        plaq_indices = geneqs.operators.toric_2d.position_to_plaq(jnp.array([i, j]), shape)
        ha_netketlocal -= nk.operator.spin.sigmaz(hi, plaq_indices[0].item()) * \
                          nk.operator.spin.sigmaz(hi, plaq_indices[1].item()) * \
                          nk.operator.spin.sigmaz(hi, plaq_indices[2].item()) * \
                          nk.operator.spin.sigmaz(hi, plaq_indices[3].item())
# adding the star terms
for i in range(L):
    for j in range(L):
        star_indices = geneqs.operators.toric_2d.position_to_star(jnp.array([i, j]), shape)
        ha_netketlocal -= nk.operator.spin.sigmax(hi, star_indices[0].item()) * \
                          nk.operator.spin.sigmax(hi, star_indices[1].item()) * \
                          nk.operator.spin.sigmax(hi, star_indices[2].item()) * \
                          nk.operator.spin.sigmax(hi, star_indices[3].item())

# own custom hamiltonian
ha_custom = geneqs.operators.toric_2d.ToricCode2d(hi, shape)

# %%
print("Comparing ToricCode2d implementations:")

# compare timing between these two operators
t_netket = geneqs.utils.benchmarking.time_expect(vqs, ha_netketlocal, n_runs)
t_custom = geneqs.utils.benchmarking.time_expect(vqs, ha_custom, n_runs)

print(f"Implementation via NetKet local operators required {t_netket/n_runs}", "\n")
print(f"Implementation as custom operator required {t_custom/n_runs}", "\n")
print(f"ratio custom/netket = {t_custom/t_netket}")
