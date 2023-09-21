import jax
import jax.numpy as jnp
from functools import partial
from .solver import FiniteDifferenceSolver
from .utils import interpolate_fn, add_time_fn


class SigKernel:
  def __init__(self, static_kernel_kind: str = 'linear', scale: float = 1e0,
               s0: float = 0., t0: float = 0., S: float = 1., T: float = 1., ds: float = 1e-1, dt: float = 1e-1,
               pde_solver_type: str = 'finite-difference', add_time: bool = False, interpolation: str = "linear"):
    if static_kernel_kind not in ["linear", "rbf"]:
      raise ValueError("Only linear and rbf static_kernels are implemented for now.")
    if pde_solver_type not in ["finite-difference"]:
      raise ValueError("Only finite-difference scheme is implemented for now.")
    self.pde_solver = FiniteDifferenceSolver(static_kernel_kind=static_kernel_kind, scale=scale)
    self.s0 = s0
    self.t0 = t0
    self.S = S
    self.T = T
    self.ds = ds
    self.dt = dt
    self.add_time = add_time
    self.interpolation = interpolation

  @partial(jax.jit, static_argnums=(0,))
  def kernel_matrix(self, X: jnp.ndarray, Y: jnp.ndarray, directions: jnp.ndarray = jnp.array([])) -> jnp.ndarray:

    # interpolate on new grid
    X = interpolate_fn(X, t_min=self.s0, t_max=self.S, step=self.ds, kind=self.interpolation)
    Y = interpolate_fn(Y, t_min=self.t0, t_max=self.T, step=self.dt, kind=self.interpolation)
    if directions.shape != (0,):
      interpolate_fn_vmap = jax.vmap(lambda Z: interpolate_fn(Z, t_min=self.s0, t_max=self.S, step=self.ds, kind=self.interpolation), in_axes=0, out_axes=0)
      directions = interpolate_fn_vmap(directions)

    # add time channel (optionally)
    if self.add_time:
      X = add_time_fn(X, t_min=self.s0, t_max=self.S)
      Y = add_time_fn(Y, t_min=self.t0, t_max=self.T)
      if directions.shape != (0,):
        add_time_fn_vmap = jax.vmap(lambda Z: add_time_fn(Z, t_min=self.s0, t_max=self.S), in_axes=0, out_axes=0)
        directions = add_time_fn_vmap(directions)
    return self.pde_solver.solve(X, Y, directions)