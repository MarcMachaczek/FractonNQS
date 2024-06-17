import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()
# supress warning about no cuda mpi version
# we don't need that because jax handles that, we only want to run copies of a process with some communication
os.environ["MPI4JAX_USE_CUDA_MPI"] = "0"
# set only one visible device
os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
# force to use gpu
os.environ["JAX_PLATFORM_NAME"] = "gpu"
# make netket compute the exact autocorrelation time
# os.environ["NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"] = "1"
# make netket use the split-Rhat diagnostic, not just the plain one
# os.environ["NETKET_USE_PLAIN_RHAT"] = "0"

import jax
from jax import numpy as jnp

import netket as nk
from netket.utils import HashableArray
import flax

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt

import geneqs
from global_variables import RESULTS_PATH

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}
L = 8
label = "hx_left_right_chaintransfer_nonoise"
save_dir = f"{RESULTS_PATH}/checkerboard/L={L}_final/L={L}_mc_crbm_{label}"

# %%
shape = jnp.array([L, L, L])
hilbert = nk.hilbert.Spin(s=1 / 2, N=jnp.prod(shape).item())
eval_model = "CheckerCRBM_2"
# get fields
direction_index = 0  # 0 for x, 1 for y, 2 for z;
obs = pd.read_csv(f"{save_dir}/L{shape}_{eval_model}_observables.txt", sep=" ", header=0)
field_strengths = obs.iloc[:, :3].values[8:]
print(field_strengths)

# hist_fields = np.array([[0.20, 0, 0],
#                         [0.40, 0, 0],
#                         [0.44, 0, 0],
#                         [0.45, 0, 0],
#                         [0.50, 0, 0],
#                         [0.70, 0, 0]])

n_chains = 256 * 4  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
chunk_size = 1 * 1024
n_samples = 16 * 1024 * 24
n_discard_per_chain = 0
n_bins = 20

# %% define observables
if direction_index == 0:
    abs_magnetization = geneqs.operators.observables.AbsXMagnetization(hilbert)
    magnetization = 1 / hilbert.size * sum([nk.operator.spin.sigmax(hilbert, i) for i in range(hilbert.size)])
elif direction_index == 1:
    abs_magnetization = geneqs.operators.observables.AbsYMagnetization(hilbert)
    magnetization = 1 / hilbert.size * sum([nk.operator.spin.sigmay(hilbert, i) for i in range(hilbert.size)])
elif direction_index == 2:
    abs_magnetization = geneqs.operators.observables.AbsZMagnetization(hilbert)
    magnetization = 1 / hilbert.size * sum([nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size)])
    
positions = jnp.asarray([[x, y, z]
                         for x in range(shape[0])
                         for y in range(shape[1])
                         for z in range(shape[2])
                         if (x + y + z) % 2 == 0])
xcubes = nk.operator.LocalOperator(hilbert, dtype=complex)
zcubes = nk.operator.LocalOperator(hilbert, dtype=complex)

for pos in positions:
    xcubes += geneqs.operators.checkerboard.get_netket_xcube(hilbert, pos, shape)
    zcubes += geneqs.operators.checkerboard.get_netket_zcube(hilbert, pos, shape)
    
# %% create model and update rule
perms = geneqs.utils.indexing.get_translations_cubical3d(shape, shift=2)
perms = nk.utils.HashableArray(perms.astype(int))

# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_cubes_cubical3d(shape, 2)),
               HashableArray(geneqs.utils.indexing.get_bonds_cubical3d(shape)))
# noinspection PyArgumentList
correlator_symmetries = (HashableArray(geneqs.utils.indexing.get_cubeperms_cubical3d(shape, 2)),
                         HashableArray(geneqs.utils.indexing.get_bondperms_cubical3d(shape, 2)))
# noinspection PyArgumentList
loops = (HashableArray(geneqs.utils.indexing.get_strings_cubical3d(0, shape)),
         HashableArray(geneqs.utils.indexing.get_strings_cubical3d(1, shape)),
         HashableArray(geneqs.utils.indexing.get_strings_cubical3d(2, shape)))
# noinspection PyArgumentList
loop_symmetries = (HashableArray(geneqs.utils.indexing.get_xstring_perms3d(shape, 2)),
                   HashableArray(geneqs.utils.indexing.get_ystring_perms3d(shape, 2)),
                   HashableArray(geneqs.utils.indexing.get_zstring_perms3d(shape, 2)))

alpha = 1 / 4
default_kernel_init = jax.nn.initializers.normal(0.01)
cRBM = geneqs.models.CheckerLoopCRBM_2(symmetries=perms,
                                       correlators=correlators,
                                       correlator_symmetries=correlator_symmetries,
                                       loops=loops,
                                       loop_symmetries=loop_symmetries,
                                       alpha=alpha,
                                       kernel_init=default_kernel_init,
                                       bias_init=default_kernel_init,
                                       param_dtype=complex)
model = cRBM

# create custom update rule
single_rule = nk.sampler.rules.LocalRule()
cube_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_cubes_cubical3d(shape, shift=2))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(0, shape))
ystring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(1, shape))
zstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical3d(2, shape))
# noinspection PyArgumentList
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.51, 0.25, 0.08, 0.08, 0.08),
                                                          [single_rule,
                                                           cube_rule,
                                                           xstring_rule,
                                                           ystring_rule,
                                                           zstring_rule])

# %% for every h, create model, load params, calculate observables
observables = geneqs.utils.eval_obs.ObservableCollector(key_names=("hx", "hy", "hz"))
for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    print(f"evaluation for field={h}")
    checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    vqs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    sampler_state = vqs.sampler_state

    with open(f"{save_dir}/vqs_{eval_model}_L{shape}_h{h}.mpack", 'rb') as file:
        vqs = flax.serialization.from_bytes(vqs, file.read())
        rank_sigmas = vqs.sampler_state.σ.reshape(n_ranks, -1, vqs.sampler_state.σ.shape[-1])
        rank_sampler_state = vqs.sampler_state.replace(σ=rank_sigmas[rank])
        vqs.sampler_state = rank_sampler_state
        vqs.n_chains_per_rank = vqs.sampler_state.σ.shape[0]
    vqs.chunk_size = chunk_size
    vqs.n_samples = n_samples
    print(vqs.n_samples, vqs.n_samples_per_rank, vqs.sampler.n_chains, vqs.sampler.n_chains_per_rank, vqs.sampler_state.σ.shape)
    
    # calculate energy and variance of energy
    energy_nk = geneqs.utils.eval_obs.expect_batched(vqs, checkerboard, n_samples=n_samples, chunk_size=chunk_size)
    observables.add_nk_obs("energy", h, energy_nk)
    # calculate magnetization
    magnetization_nk = geneqs.utils.eval_obs.expect_batched(vqs, magnetization, n_samples=n_samples, chunk_size=chunk_size)
    observables.add_nk_obs("mag", h, magnetization_nk)
    # calculate cube parts of the hamiltonian
    xcubes_nk = geneqs.utils.eval_obs.expect_batched(vqs, xcubes, n_samples=n_samples, chunk_size=chunk_size)
    observables.add_nk_obs("xcubes", h, xcubes_nk)
    zcubes_nk = geneqs.utils.eval_obs.expect_batched(vqs, zcubes, n_samples=n_samples, chunk_size=chunk_size)
    observables.add_nk_obs("zcubes", h, zcubes_nk)
    print(f"donw with {h}")
    
    if rank == 0:
        save_array = observables.obs_to_array(separate_keys=False)[-1].reshape(1, -1)
        with open(f"{save_dir}/L{shape}_{eval_model}_eval_obs.txt", "ab") as f:
            if os.path.getsize(f"{save_dir}/L{shape}_{eval_model}_eval_obs.txt") == 0:
                np.savetxt(f, save_array, header=" ".join(observables.key_names + observables.obs_names),
                           comments="")
            else:
                np.savetxt(f, save_array)
    
    # calculate absolute magnetization
    # abs_magnetization_nk = geneqs.utils.eval_obs.expect_batched(vqs, abs_magnetization, n_samples=n_samples, chunk_size=chunk_size)
    # observables.add_nk_obs("abs_mag", h, abs_magnetization_nk)
    # gather local estimators as each rank calculates them based on their own samples_per_rank
    # if np.any((h == hist_fields).all(axis=1)):
    #
    #     energy_locests = []
    #     mag_locests = []
    #     vqs.n_samples = 2*chunk_size
    #
    #     for _ in tqdm(range(1)):
    #         vqs.sample()
    #         energy_locests.append(np.array(comm.gather(vqs.local_estimators(checkerboard), root=0)).flatten().real)
    #         mag_locests.append(np.array(comm.gather(vqs.local_estimators(magnetization), root=0)).flatten().real)
    #     energy_locests = np.concatenate(energy_locests)
    #
    #
    #     observables.add_hist("epsite", h, np.histogram(np.asarray(energy_locests) / hilbert.size, n_bins))
    #     observables.add_hist("mag", h, np.histogram(np.asarray(mag_locests), n_bins))
    #
    # # %% save histograms
    # if rank == 0:
    #     print(energy_locests.shape)
    #     for hist_name, _ in observables.histograms.items():
    #         np.save(f"{save_dir}/hist_{hist_name}_L{shape}_{eval_model}_{label}.npy",
    #                 observables.hist_to_array(hist_name))
# if rank == 0:
#     save_array = observables.obs_to_array(separate_keys=False)
#     np.savetxt(f"{save_dir}/L{shape}_{eval_model}_eval_obs.txt", save_array,
#                header=" ".join(observables.key_names + observables.obs_names), comments="")
