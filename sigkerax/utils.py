import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=3)
def interpolate_fn(X: jnp.ndarray, old_grid: jnp.ndarray, new_grid: jnp.ndarray, kind: str = "linear") -> jnp.ndarray:
  if kind != "linear":
    raise ValueError("Only linear interpolation is supported for now.")

  def _interpolate_one_channel(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.interp(new_grid, old_grid, x)

  def _interpolate(x: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(_interpolate_one_channel, in_axes=1, out_axes=1)(x)

  return jax.vmap(_interpolate)(X)


@jax.jit
def add_time_fn(X: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
  return jnp.concatenate([jnp.tile(jnp.expand_dims(grid, axis=[0, 2]), (X.shape[0], 1, 1)), X], axis=2)
