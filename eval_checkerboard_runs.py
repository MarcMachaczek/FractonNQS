import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
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

matplotlib.rcParams.update({'font.size': 12})

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/checkerboard"

# %%
shape = jnp.array([4, 4, 4])
hilbert = nk.hilbert.Spin(s=1 / 2, N=jnp.prod(shape))
eval_model = "CheckerCRBM"
# get fields
direction_index = 0  # 0 for x, 1 for y, 2 for z;
obs = pd.read_csv(f"{save_dir}/L{shape}_{eval_model}_observables.txt", sep=" ", header=0)
field_strengths = obs.iloc[:, :3].values

field_strengths = np.array([[0., 0., 0.90],
                            [0., 0., 0.80],
                            [0., 0., 0.70],
                            [0., 0., 0.60],
                            [0., 0., 0.50],
                            [0., 0., 0.45],
                            [0., 0., 0.44],
                            [0., 0., 0.43],
                            [0., 0., 0.42],
                            [0., 0., 0.41],
                            [0., 0., 0.40],
                            [0., 0., 0.39],
                            [0., 0., 0.38],
                            [0., 0., 0.37],
                            [0., 0., 0.36],
                            [0., 0., 0.35],
                            [0., 0., 0.34],
                            [0., 0., 0.33],
                            [0., 0., 0.30],
                            [0., 0., 0.20],
                            [0., 0., 0.10],
                            [0., 0., 0.00]])

n_chains = 1024  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
chunk_size = 1024 * 4
n_samples = chunk_size * 4 * 16
n_discard_per_chain = 0

# %% create observables
# define some observables
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
cRBM = geneqs.models.CheckerLoopCRBM(symmetries=perms,
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

# %% for every h, create model, load params calculate observables
observables = geneqs.utils.eval_obs.ObservableCollector(key_names=("hx", "hy", "hz"))
for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    print(f"evaluation for field={h}")
    checkerboard = geneqs.operators.checkerboard.Checkerboard(hilbert, shape, h)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    vqs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    with open(f"{save_dir}/vqs_{eval_model}_L{shape}_h{h}.mpack", 'rb') as file:
        vqs = flax.serialization.from_bytes(vqs, file.read())
    vqs.chunk_size = chunk_size
    vqs.n_samples = n_samples

    # calculate energy and variance of energy
    energy_nk = vqs.expect(checkerboard)
    observables.add_nk_obs("energy", h, energy_nk)
    # calculate magnetization
    magnetization_nk = vqs.expect(magnetization)
    observables.add_nk_obs("mag", h, magnetization_nk)
    # calculate absolute magnetization
    abs_magnetization_nk = vqs.expect(abs_magnetization)
    observables.add_nk_obs("abs_mag", h, abs_magnetization_nk)
    # calculate cube parts of the hamiltonian
    xcubes_nk = vqs.expect(xcubes)
    zcubes_nk = vqs.expect(zcubes)
    observables.add_nk_obs("xcubes", h, xcubes_nk)
    observables.add_nk_obs("zcubes", h, zcubes_nk)

if rank == 0:
    save_array = observables.obs_to_array(separate_keys=False)
    np.savetxt(f"{save_dir}/L{shape}_{eval_model}_eval_obs.txt", save_array,
                   header=" ".join(observables.key_names + observables.obs_names), comments="")
