import jax
import jax.numpy as jnp
from functools import partial
from .solver import PDESolver
from .utils import interpolate_fn, add_time_fn


class SigKernel:
  def __init__(self, pde_solver: PDESolver, s0: float = 0., t0: float = 0., S: float = 1., T: float = 1.,
               ds: float = 1e-1, dt: float = 1e-1, add_time: bool = False, interpolation: str = "linear"):
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

    # interpolate on new grid
    X = interpolate_fn(X, t_min=self.s0, t_max=self.S, step=self.ds, kind=self.interpolation)
    Y = interpolate_fn(Y, t_min=self.t0, t_max=self.T, step=self.dt, kind=self.interpolation)

    # add time channel (optionally)
    if self.add_time:
      X = add_time_fn(X, t_min=self.s0, t_max=self.S)
      Y = add_time_fn(Y, t_min=self.t0, t_max=self.T)

    return self.pde_solver.solve(X, Y)
