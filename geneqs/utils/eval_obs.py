import numpy as np
import jax
import jax.numpy as jnp
import netket as nk

from typing import Union
from collections import OrderedDict

from dataclasses import dataclass, field


# %%
@dataclass
class ObservableEvaluator:
    observables: Union[dict, OrderedDict] = field(default_factory=OrderedDict)

    def __post_init__(self):
        pass

    def add_nk_obs(self, key, name: str, nk_obs: nk.stats):
        # add key if not present
        if key not in self.observables.keys():
            self.observables[key] = {}

        self.observables[key][name] = nk_obs.Mean.item()
        self.observables[key][f"{name}_var"] = nk_obs.Variance.item()

    def add_array(self, key, name: str, array):
        # add key if not present
        if key not in self.observables.keys():
            self.observables[key] = {}

        self.observables[key][name] = array

    def get_observable(self, name):
        var_value = True  # if corresponding error is saved

        name_obs = []
        for keys, observables in self.observables.items():
            name_obs.append(observables[name])

            # check if var value for observable is present
            if f"{name}_var" is not observables.keys():
                var_value = False

        if var_value:
            name_obs_var = []
            for keys, observables in self.observables.items():
                name_obs_var.append(observables[f"{name}_var"])

            return list[self.observables.keys()], name_obs, name_obs_var
        else:
            print(f"no var value was found for {name} observable")
            return list[self.observables.keys()], name_obs, None

    def to_txt(self, save_path: str):
        rows = list(self.observables.keys())
        pass

    @classmethod
    def from_txt(cls, load_path: str):
        pass


# %%
L = 3
shape = jnp.array([L, L])
square_graph = nk.graph.Square(length=L, pbc=True)
hilbert = nk.hilbert.Spin(s=1 / 2, N=square_graph.n_edges)

obs = nk.operator.spin.sigmaz(hilbert, 0) * nk.operator.spin.sigmaz(hilbert, 1)
sampler = nk.sampler.MetropolisLocal(hilbert, dtype=jnp.int8)
model = nk.models.RBM(alpha=1)
vqs = nk.vqs.MCState(sampler, model)

value = vqs.expect(obs)

test = {"a": 1, "b": 2, "c": {(0, 0): "in", (0, 1): "out"}}
#

# %%

def susc_from_mag(magnetizations: Union[jax.Array, np.ndarray], fields: Union[jax.Array, np.ndarray]):
    """
    Calculate the magnetic susceptibility from observed magnetization values via finite differences.
    This function also returns the external field values for which the susceptibilites were evaluated.
    Args:
        magnetizations: Magnetization values for different external fields.
        fields: Array of shape (n_fields, 3) containing for each magnetization the external field in x, y, z direction.

    Returns:
        susceptibility array (n_fields), susceptibility fields (n_fields, 3)

    """
    
    assert len(magnetizations) == len(fields), "length of magnetizations and fields must be the same"

    sus = (magnetizations[1:] - magnetizations[:-1]) / np.linalg.norm(fields[1:] - fields[:-1], axis=1)
    sus_fields = (fields[1:] + fields[:-1]) / 2

    return sus, sus_fields
