import jax.numpy as jnp


def linear_kernel(x: jnp.ndarray, y: jnp.ndarray, scale: float = 1.0) -> float:
  return jnp.dot(scale * x, scale * y)


def rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, scale: float = 1.0) -> float:
  return jnp.exp(-jnp.linalg.norm(x - y, ord=2) ** 2 / (2.0 * scale ** 2))
