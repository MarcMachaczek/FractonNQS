import jax
import jax.numpy as jnp

import flax
from flax.linen.dtypes import promote_dtype

import functools


# %%
# adapted from flax library
def dotproduct_attention(query: jax.Array,
                         key: jax.Array,
                         value: jax.Array,
                         temperature: float = 1,
                         precision: Any = None) -> jax.Array:
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of
      `[batch..., q_length, num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of
      `[batch..., kv_length, num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of
      `[batch..., kv_length, num_heads, v_depth_per_head]`.
    temperature: sharpens (T<1) or broadens (T>1) the probabilities 
        calculated from the softmax function.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
    query, key, value = promote_dtype(query, key, value)
    dtype = query.dtype
    
    # some sanity checks
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], "q, k, v num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert temperature > 0, "temperature muste be positive."
    
    # first calculate attention weights
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key, precision=precision)
    
    # normalize the attention weights
    attn_weights = jax.nn.softmax(attn_weights / temperature).astype(dtype)
    
    # return accumulated weighted values
    return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
    