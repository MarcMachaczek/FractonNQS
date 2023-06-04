import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import flax


# %%
def params_to_txt(vqs: nk.vqs.VariationalState, filepath: str):
    param_dict = vqs.parameters
    with open(f"{filepath}", "w") as file:
        for par_name, values in param_dict.items():
            file.write(f"{par_name} of shape {values.shape} \n")
            np.savetxt(file, np.array(values), fmt="%.4e")
            file.write("\n")
