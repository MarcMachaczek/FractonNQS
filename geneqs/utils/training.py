import numpy as np
import jax.numpy as jnp
import jax


def custom_callback(step: int, logdata: dict, driver):
    sampler_state = driver._variational_state.sampler_state
    logdata["acceptance_rate"] = sampler_state.acceptance.item()
    return True
