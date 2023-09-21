import jax
import jax.numpy as jnp
import random
from functools import partial


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def interpolate_fn(X: jnp.ndarray, t_min: float, t_max: float, step: float, kind: str = "linear") -> jnp.ndarray:
  if kind != "linear":
    raise ValueError("Only linear interpolation is supported for now.")
  old_grid = jnp.linspace(t_min, t_max, X.shape[1], dtype=X.dtype)
  new_grid = jnp.linspace(t_min, t_max, int((t_max - t_min) / step), dtype=X.dtype)
  return jax.vmap(jax.vmap(lambda x: jnp.interp(new_grid, old_grid, x), in_axes=1, out_axes=1))(X)


@partial(jax.jit, static_argnums=(1, 2))
def add_time_fn(X: jnp.ndarray, t_min: float, t_max: float) -> jnp.ndarray:
  grid = jnp.linspace(t_min, t_max, X.shape[1], dtype=X.dtype)
  return jnp.concatenate([jnp.tile(jnp.expand_dims(grid, axis=[0, 2]), (X.shape[0], 1, 1)), X], axis=2)


def getkey():
  return jax.random.PRNGKey(random.randint(0, 2 ** 31 - 1))


def I0(x_squared, num_terms, dtype):
  """Modified Bessel function of order 0"""
  factorials = jnp.cumprod(jnp.arange(1, num_terms, dtype=dtype))
  return 1.0 + jnp.sum(
    jnp.array([(1.0 / (factorials[k - 1] ** 2)) * ((x_squared / 4.0) ** k) for k in jnp.arange(1, num_terms)]), axis=0)
