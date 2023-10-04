import jax
import jax.numpy as jnp
from functools import partial
from .solver import FiniteDifferenceSolver
from .utils import interpolate_fn, add_time_fn


class SigKernel:
  def __init__(self, static_kernel_kind: str = 'linear', scales: jnp.ndarray = jnp.array([1e0]),
               s0: float = 0., t0: float = 0., S: float = 1., T: float = 1., refinement_factor: float = 1.,
               add_time: bool = False, interpolation: str = "linear"):
    if static_kernel_kind not in ["linear", "rbf"]:
      raise ValueError("Only linear and rbf static_kernels are implemented for now.")
    self.s0 = s0
    self.t0 = t0
    self.S = S
    self.T = T
    self.refinement_factor = refinement_factor
    self.add_time = add_time
    self.interpolation = interpolation
    self.scales = scales
    self.pde_solver = FiniteDifferenceSolver(static_kernel_kind=static_kernel_kind, scale=1e0)

  @partial(jax.jit, static_argnums=(0,))
  def kernel_matrix(self, X: jnp.ndarray, Y: jnp.ndarray, directions: jnp.ndarray = jnp.array([])) -> jnp.ndarray:

    # interpolate on new grid
    X = interpolate_fn(X, t_min=self.s0, t_max=self.S, refinement_factor=self.refinement_factor, kind=self.interpolation)
    Y = interpolate_fn(Y, t_min=self.t0, t_max=self.T, refinement_factor=self.refinement_factor, kind=self.interpolation)
    if directions.shape != (0,):
      interpolate_fn_vmap = jax.vmap(lambda Z: interpolate_fn(Z, t_min=self.s0, t_max=self.S,
                                                              refinement_factor=self.refinement_factor,
                                                              kind=self.interpolation), in_axes=0, out_axes=0)
      directions = interpolate_fn_vmap(directions)

    # add time channel (optionally)
    if self.add_time:
      X = add_time_fn(X, t_min=self.s0, t_max=self.S)
      Y = add_time_fn(Y, t_min=self.t0, t_max=self.T)
      if directions.shape != (0,):
        add_time_fn_vmap = jax.vmap(lambda Z: add_time_fn(Z, t_min=self.s0, t_max=self.S), in_axes=0, out_axes=0)
        directions = add_time_fn_vmap(directions)

    def body_fun(i, accum):
      s = self.scales[i]
      result = self.pde_solver.solve(s * X, s * Y, s * directions)
      return accum + result
    initial_value = self.pde_solver.solve(self.scales[0] * X, Y, self.scales[0] * directions)
    return jax.lax.fori_loop(1, len(self.scales), body_fun, initial_value) / len(self.scales)
    # return jnp.mean(jax.vmap(lambda s: self.pde_solver.solve(s * X, Y, s * directions), in_axes=0, out_axes=0)(self.scales), axis=0)