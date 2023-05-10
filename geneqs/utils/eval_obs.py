import numpy as np
import jax
import jax.numpy as jnp
import netket as nk

from typing import Union
from collections import OrderedDict

from dataclasses import dataclass, field


# %%
@dataclass
class ObservableCollector:
    key_names: Union[list[str, ...], tuple[str, ...]]
    observables: Union[dict, OrderedDict] = field(default_factory=OrderedDict)
    histograms: Union[dict, OrderedDict] = field(default_factory=OrderedDict)
    arrays: Union[dict, OrderedDict] = field(default_factory=OrderedDict)

    def __post_init__(self):
        self.key_names = list(self.key_names)
        self.observables = OrderedDict(self.observables)
        self.histograms = OrderedDict(self.histograms)
        self.arrays = OrderedDict(self.arrays)

    @property
    def obs_names(self):
        return list(self.observables.keys())

    @property
    def hist_names(self):
        return list(self.histograms.keys())

    @property
    def array_names(self):
        return list(self.arrays.keys())

    def add_nk_obs(self, name: str, key, nk_obs: nk.stats):
        assert len(key) == len(self.key_names), f"key {key} is not compatible with key_names {self.key_names}"
        # add name if not present
        if name not in self.observables.keys():
            self.observables[name] = OrderedDict()
            self.observables[f"{name}_var"] = OrderedDict()

        self.observables[name][key] = nk_obs.Mean.item().real
        self.observables[f"{name}_var"][key] = nk_obs.Variance.item().real

    def add_hist(self, name: str, key, histogram: tuple[np.ndarray, np.ndarray]):
        assert len(key) == len(self.key_names), f"key {key} is not compatible with key_names {self.key_names}"

        if name not in self.histograms.keys():
            self.histograms[name] = OrderedDict()

        hist, bin_edges = histogram[0], histogram[1]
        self.histograms[name][key] = {"hist": hist, "bin_edges": bin_edges}

    def add_array(self, name: str, array):
        self.arrays[name] = array

    def obs_to_array(self, names: Union[list[str, ...], str] = "all", separate_keys: bool = True):
        if names == "all":
            names = list(self.observables.keys())
        elif type(names) == str:
            names = [names]

        assert self.check_keys(names), f"keys for provided names {names} don't match"

        keys_array = np.asarray(list(self.observables[names[0]].keys()))
        obs_array = np.asarray(list(self.observables[names[0]].values())).reshape(-1, 1)

        for name in names[1:]:
            obs_values = []
            for key, value in self.observables[name].items():
                obs_values.append(value)
            obs_values = np.asarray(obs_values).reshape(-1, 1)
            obs_array = np.concatenate((obs_array, obs_values), axis=1)

        if separate_keys:
            return keys_array, obs_array
        return np.concatenate((keys_array, obs_array), axis=1)

    def hist_to_array(self, name: str):
        """
        Returns the histograms identified by name over all keys.
        Args:
            name: name of the histograms

        Returns:
            key, histogram ndarray of shape (n_keys, 2) containing arrays

        """
        histograms = []
        for key, val in self.histograms[name].items():
            histograms.append(np.asarray([np.asarray(key), val["hist"], val["bin_edges"]], dtype=object))

        return histograms

    def get_array(self, name: str):
        return self.arrays[name]

    def check_keys(self, names: list[str, ...]):
        compatible = True
        keys = np.asarray(self.observables[names[0]].keys())
        for name in names[1:]:
            name_keys = np.asarray(self.observables[name].keys())
            if not (name_keys == keys).all():
                compatible = False
        return compatible

    def __repr__(self):
        return f"saved observables: {list(self.observables.keys())} \n" \
               f"saved histograms: {list(self.histograms.keys())} \n" \
               f"saved arrays: {list(self.arrays.keys())}"

    @classmethod
    def from_txt(cls, load_path: str):
        pass


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
