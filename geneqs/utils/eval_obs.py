# %%
import numpy as np
import jax.numpy as jnp
import jax

from matplotlib import pyplot as plt

from typing import Union

def susc_from_mag(magnetizations: Union[jax.Array, np.ndarray], fields: Union[jax.Array, np.ndarray]):
    
    assert len(magnetizations) == len(fields), "length of magnetizations and fields must be the same"

    sus = (magnetizations[1:] - magnetizations[:-1]) / np.linalg.norm(fields[1:] - fields[:-1], axis=1)
    sus_fields = (fields[1:] + fields[:-1]) / 2

    return sus, sus_fields

# %%
direction = np.array([1, 0, 1]).reshape(-1, 1)
field_strengths = (np.linspace(0, 1, 22) * direction).T

magnetization = field_strengths[:,0]**3


# %%
sus, sus_fields = susc_from_mag(magnetization, field_strengths)


plt.plot(field_strengths[:,0], magnetization)
plt.plot(sus_fields[:,0], sus)

plt.show()

# %%