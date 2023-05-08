import numpy as np
import jax

from typing import Union


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
