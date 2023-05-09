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
    arrays: Union[dict, OrderedDict] = field(default_factory=OrderedDict)

    def __post_init__(self):
        self.observables = OrderedDict(self.observables)
        self.arrays = OrderedDict(self.arrays)

    def add_nk_obs(self, name: str, key, nk_obs: nk.stats):
        # add name if not present
        if name not in self.observables.keys():
            self.observables[name] = OrderedDict()
            self.observables[f"{name}_var"] = OrderedDict()

        self.observables[name][key] = nk_obs.Mean.item()
        self.observables[f"{name}_var"][key] = nk_obs.Variance.item()

    def add_array(self, name: str, key, array):
        self.arrays[name] = array

    def obs_to_array(self, names: Union[list[str, ...], str] = "all"):
        if names == "all":
            names = list(self.observables.keys())
        elif type(names) == str:
            names = [names]

        assert self.check_keys(names), f"keys for provided names {names} don't match"

        obs_array = np.asarray(list(self.observables[names[0]].keys()))
        
        for name in names:
            obs_values = []
            for key, value in self.observables[name].items():
                obs_values.append(value)
            obs_values = np.asarray(obs_values).reshape(-1, 1)
            obs_array = np.concatenate((obs_array, obs_values), axis=1)
        
        return obs_array
    
    def get_array(self, name):
        return self.arrays[name]

    def check_keys(self, names: list[str, ...]):
        compatible = True
        keys = np.asarray(self.observables[names[0]].keys())
        for name in names[1:]:
            name_keys = np.asarray(self.observables[name].keys())
            if not (name_keys == keys).all():
                compatible = False
        return compatible

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

# %%
logger = ObservableEvaluator()
logger.add_nk_obs("obs", (1, 2, 3), value)
logger.add_nk_obs("obs", (2, 2, 3), value)
logger.add_nk_obs("obs", (3, 2, 3), value)
logger.obs_to_array("obs")

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
