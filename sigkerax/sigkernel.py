import jax
import jax.numpy as jnp
from functools import partial
from solver import PDESolver
from utils import interpolate_fn, add_time_fn


class SigKernel():
  def __init__(self,
               pde_solver: PDESolver,
               s0: float = 0.,
               t0: float = 0.,
               S: float = 1.,
               T: float = 1.,
               ds: float = 1e-1,
               dt: float = 1e-1,
               add_time: bool = False,
               interpolation: str = "linear"):

    self.s0 = s0
    self.t0 = t0
    self.S = S
    self.T = T
    self.ds = ds
    self.dt = dt
    self.pde_solver = pde_solver
    self.add_time = add_time
    self.interpolation = interpolation

  @partial(jax.jit, static_argnums=0)
  def kernel_matrix(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:

    if self.interpolation != "linear":
      raise ValueError("Only linear interpolation is implemented for now.")

    # original grids
    original_grid_x = jnp.linspace(self.s0, self.S, X.shape[1], dtype=X.dtype)
    original_grid_y = jnp.linspace(self.t0, self.T, Y.shape[1], dtype=Y.dtype)

    # new evaluation grids
    new_grid_x = jnp.linspace(self.s0, self.S, int((self.S - self.s0) / self.ds), dtype=X.dtype)
    new_grid_y = jnp.linspace(self.t0, self.T, int((self.T - self.t0) / self.dt), dtype=Y.dtype)

    # interpolate on new grid
    X = interpolate_fn(X, original_grid_x, new_grid_x, self.interpolation)
    Y = interpolate_fn(Y, original_grid_y, new_grid_y, self.interpolation)

    # add time channel (optionally)
    if self.add_time:
      X = add_time_fn(X, new_grid_x)
      Y = add_time_fn(Y, new_grid_y)

    return self.pde_solver.solve(X, Y)
