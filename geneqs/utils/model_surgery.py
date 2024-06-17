import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import flax


# %%
def params_to_txt(vqs: nk.vqs.VariationalState, filepath: str):
    param_dict = vqs.parameters
    flat_param_dict, _ = jax.tree_util.tree_flatten_with_path(param_dict)
    with open(f"{filepath}", "w") as file:
        for par_name, values in flat_param_dict:
            name = jax.tree_util.keystr(par_name)
            val = np.atleast_1d(np.squeeze(values))
            file.write(f"{name} of shape {values.shape} \n")
            np.savetxt(file, val, fmt="%.4e")
            file.write("\n")
