import matplotlib
from matplotlib import pyplot as plt
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

import netket as nk
from netket.utils import HashableArray
import flax

import geneqs
from global_variables import RESULTS_PATH

matplotlib.rcParams.update({'font.size': 12})

cmap = matplotlib.colormaps["Set1"]
f_dict = {0: "x", 1: "y", 2: "z"}
save_dir = f"{RESULTS_PATH}/toric_h"

# %%
shape = jnp.array([8, 8])
square_graph = nk.graph.Square(length=shape[0], pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)
eval_model = "ToricCRBM"
# get fields
direction_index = 0  # 0 for x, 1 for y, 2 for z;
obs = pd.read_csv(f"{save_dir}/L{shape}_{eval_model}_observables.txt", sep=" ", header=0)
field_strengths = obs.iloc[:, :3].values

field_strengths = np.array([[0., 0., 0.8],
                            [0., 0., 0.7],
                            [0., 0., 0.65],
                            [0., 0., 0.6],
                            [0., 0., 0.55],
                            [0., 0., 0.53],
                            [0., 0., 0.51],
                            [0., 0., 0.49],
                            [0., 0., 0.47],
                            [0., 0., 0.45],
                            [0., 0., 0.43],
                            [0., 0., 0.41],
                            [0., 0., 0.39],
                            [0., 0., 0.37],
                            [0., 0., 0.35],
                            [0., 0., 0.33],
                            [0., 0., 0.3],
                            [0., 0., 0.2],
                            [0., 0., 0.1],
                            [0., 0., 0.]])

n_chains = 512 * 1  # total number of MCMC chains, when runnning on GPU choose ~O(1000)
chunk_size = 1024 * 8
n_samples = chunk_size * 50
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
wilsonob = geneqs.operators.observables.get_netket_wilsonob(hilbert, shape)

positions = jnp.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
A_B = 1 / hilbert.size * sum([geneqs.operators.toric_2d.get_netket_star(hilbert, p, shape) for p in positions]) - \
      1 / hilbert.size * sum([geneqs.operators.toric_2d.get_netket_plaq(hilbert, p, shape) for p in positions])

# %% create model and update rule
# get (specific) symmetries of the model, in our case translations
perms = geneqs.utils.indexing.get_translations_cubical2d(shape, shift=1)
# noinspection PyArgumentList
link_perms = HashableArray(geneqs.utils.indexing.get_linkperms_cubical2d(shape, shift=1))

bl_bonds, lt_bonds, tr_bonds, rb_bonds = geneqs.utils.indexing.get_bonds_cubical2d(shape)
bl_perms, lt_perms, tr_perms, rb_perms = geneqs.utils.indexing.get_bondperms_cubical2d(shape)
# noinspection PyArgumentList
correlators = (HashableArray(geneqs.utils.indexing.get_plaquettes_cubical2d(shape)),  # plaquette correlators
               HashableArray(bl_bonds), HashableArray(lt_bonds), HashableArray(tr_bonds), HashableArray(rb_bonds))

# noinspection PyArgumentList
correlator_symmetries = (HashableArray(jnp.asarray(perms)),  # plaquettes permute like sites
                         HashableArray(bl_perms), HashableArray(lt_perms),
                         HashableArray(tr_perms), HashableArray(rb_perms))
# noinspection PyArgumentList
loops = (HashableArray(geneqs.utils.indexing.get_strings_cubical2d(0, shape)),  # x-string correlators
         HashableArray(geneqs.utils.indexing.get_strings_cubical2d(1, shape)))  # y-string correlators
# noinspection PyArgumentList
loop_symmetries = (HashableArray(geneqs.utils.indexing.get_xstring_perms(shape)),
                   HashableArray(geneqs.utils.indexing.get_ystring_perms(shape)))

alpha = 1
default_kernel_init = jax.nn.initializers.normal(0.01)
cRBM = geneqs.models.ToricLoopCRBM(symmetries=link_perms,
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
vertex_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_stars_cubical2d(shape))
xstring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(0, shape))
ystring_rule = geneqs.sampling.update_rules.MultiRule(geneqs.utils.indexing.get_strings_cubical2d(1, shape))
weighted_rule = geneqs.sampling.update_rules.WeightedRule((0.5, 0.25, 0.125, 0.125),
                                                          [single_rule, vertex_rule, xstring_rule, ystring_rule])

# %% for every h, create model, load params calculate observables
observables = geneqs.utils.eval_obs.ObservableCollector(key_names=("hx", "hy", "hz"))
for h in tqdm(field_strengths, "external_field"):
    h = tuple(h)
    print(f"evaluation for field={h}")
    toric = geneqs.operators.toric_2d.ToricCode2d(hilbert, shape, h)
    sampler = nk.sampler.MetropolisSampler(hilbert, rule=weighted_rule, n_chains=n_chains, dtype=jnp.int8)
    vqs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    with open(f"{save_dir}/vqs_{eval_model}_L{shape}_h{h}.mpack", 'rb') as file:
        vqs = flax.serialization.from_bytes(vqs, file.read())
    vqs.chunk_size = chunk_size
    vqs.n_samples = n_samples

    # calculate energy and specific heat / variance of energy
    energy_nk = vqs.expect(toric)
    observables.add_nk_obs("energy", h, energy_nk)
    # calculate magnetization
    magnetization_nk = vqs.expect(magnetization)
    observables.add_nk_obs("mag", h, magnetization_nk)
    # calculate absolute magnetization
    abs_magnetization_nk = vqs.expect(abs_magnetization)
    observables.add_nk_obs("abs_mag", h, abs_magnetization_nk)
    # calcualte wilson loop operator
    wilsonob_nk = vqs.expect(wilsonob)
    observables.add_nk_obs("wilson", h, wilsonob_nk)

save_array = observables.obs_to_array(separate_keys=False)
np.savetxt(f"{save_dir}/L{shape}_{eval_model}_eval_obs.txt", save_array,
               header=" ".join(observables.key_names + observables.obs_names), comments="")
