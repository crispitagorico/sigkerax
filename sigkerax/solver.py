import jax
import jax.numpy as jnp
from functools import partial
from .static_kernels import linear_kernel, rbf_kernel


class FiniteDifferenceSolver:
  def __init__(self, static_kernel_kind: str = 'linear', scale: float = 1e0, multi_gpu: str = False):
    self.static_kernel_kind = static_kernel_kind
    if static_kernel_kind == "linear":
      self.static_kernel = lambda x, y: linear_kernel(x, y, scale)
    elif static_kernel_kind == "rbf":
      self.static_kernel = lambda x, y: rbf_kernel(x, y, scale)
    else:
      raise ValueError("only linear and rbf static_kernels are implemented for now")
    self.multi_gpu = multi_gpu

  @staticmethod
  def _diag_axis_mask(p: int, length_X: int, length_Y: int) -> jnp.ndarray:
    diag_length = length_X  # TODO: select max between length_X and length_Y
    diag_axis = jnp.arange(diag_length)
    start_row = jnp.maximum(0, p - (length_Y - 1))
    end_row = jnp.minimum(length_X, p + 1)
    return jnp.where((diag_axis >= start_row) & (diag_axis < end_row), diag_axis, -1)

  @staticmethod
  def _initial_condition(state_space_dim: int, dtype: type):
    ic = jnp.zeros(shape=state_space_dim, dtype=dtype)
    return ic.at[0].set(1.0)

  @partial(jax.jit, static_argnums=(0,))
  def _static_kernel_diag_update(self, p: int, diag_axis_masked: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """
    This function computes k_static(X[i,k], Y[j,k]), for k in 'diag_axis', where k_static is the chosen static kernel.
    The updated diagonal is determined by a conditional check on each element of the `diag_axis` array; if an element
    in this array is not equal to -1, the static kernel value is returned; otherwise, the value is set to 0.0.

    Inputs:
    - p (int): An index offset to adjust the position within Y when computing the static kernel value.
    - diag_axis_masked (jnp.ndarray): Indexing array for static kernel diagonal update. Values -1 indicate positions set to 0.
                                      Shape: (length_X,).
    - X (jnp.ndarray): First input paths tensor. Shape: (batch_X, length_X, channel).
    - Y (jnp.ndarray): Second input paths tensor. Shape: (batch_Y, length_Y, channel).

    Outputs:
    - jnp.ndarray: The updated diagonal of the static kernel evaluations. Shape: (batch_X, batch_Y, length_X).
    """
    return jax.vmap(
               jax.vmap(
                    jax.vmap(lambda i, j, k: jnp.where(k != -1, self.static_kernel(X[i, k], Y[j, p - k]), 0.0),
                    in_axes=(None, None, 0)),
               in_axes=(None, 0, None)),
            in_axes=(0, None, None)
            )(jnp.arange(X.shape[0]), jnp.arange(Y.shape[0]), diag_axis_masked)

  @partial(jax.jit, static_argnums=(0,))
  def _data_diag_update(self,
                        p: int,
                        diag_axis_masked: jnp.ndarray,
                        X: jnp.ndarray,
                        Y: jnp.ndarray,
                        directions: jnp.ndarray = jnp.array([]),
                        ) -> jnp.ndarray:

    # Problem sizes
    batch_X, length_X = X.shape[0], X.shape[1]
    batch_Y, length_Y = Y.shape[0], Y.shape[1]
    diag_length = length_X  # TODO: select max between length_X and length_Y
    order_derivative = directions.shape[0] if directions.shape != (0,) else 0
    assert order_derivative <= 2, "directions must be an array of shape (l,...) where l is less or equal to 2."
    state_space_dim = 2 ** order_derivative

    # Initialise data diagonals of inner products of increments.
    diag_data = jnp.zeros(shape=(batch_X, batch_Y, diag_length, state_space_dim, state_space_dim), dtype=X.dtype)
    A_XY = self._static_kernel_diag_update(p, diag_axis_masked, X, Y)
    diag_data = diag_data.at[..., 0, 0].set(A_XY)

    # If derivatives are required
    if order_derivative >= 1:
      assert self.static_kernel_kind == "linear", "Only derivatives for the linear kernel are implemented for now."
      A_XY_eta = self._static_kernel_diag_update(p, diag_axis_masked, directions[0], Y)
      diag_data = diag_data.at[..., 1, 0].set(A_XY_eta)
      diag_data = diag_data.at[..., 1, 1].set(A_XY)
      if order_derivative == 2:
        A_XY_etab = self._static_kernel_diag_update(p, diag_axis_masked, directions[1], Y)
        diag_data = diag_data.at[..., 2, 0].set(A_XY_etab)
        diag_data = diag_data.at[..., 2, 2].set(A_XY)
        diag_data = diag_data.at[..., 3, 1].set(A_XY_etab)
        diag_data = diag_data.at[..., 3, 2].set(A_XY_eta)
        diag_data = diag_data.at[..., 3, 3].set(A_XY)

    return diag_data

  @partial(jax.jit, static_argnums=(0, 1))
  def _solution_diag_update(self,
                            p: int,
                            diag_axis_masked: jnp.ndarray,
                            diag_solution_minus2: jnp.ndarray,
                            diag_solution_minus1: jnp.ndarray,
                            diag_data_minus2: jnp.ndarray,
                            diag_data_minus1: jnp.ndarray,
                            diag_data: jnp.ndarray,
                            ) -> jnp.ndarray:

    # Problem sizes
    state_space_dim = diag_solution_minus2.shape[-1]
    dtype = diag_solution_minus2.dtype
    ic = self._initial_condition(state_space_dim, dtype)
    zeros = jnp.zeros_like(ic)
    id_mat = jnp.eye(state_space_dim, dtype=dtype)

    def _solution_single_update(i: int, j: int, k: int) -> jnp.ndarray:
      k00 = diag_solution_minus2[i, j, k - 1]
      k01 = diag_solution_minus1[i, j, k - 1]
      k10 = diag_solution_minus1[i, j, k]
      inc = diag_data_minus2[i, j, k - 1] - diag_data_minus1[i, j, k - 1] - diag_data_minus1[i, j, k] + diag_data[i, j, k]
      inc2 = jnp.matmul(inc, inc)
      # update = jnp.matmul(id_mat + 0.5 * inc, k01 + k10) - jnp.matmul(id_mat, k00)
      update = jnp.matmul(id_mat + 0.5 * inc + (1. / 12.) * inc2, k01 + k10) - jnp.matmul(id_mat - (1. / 12.) * inc2, k00)
      return jnp.where(k != -1, jnp.where((k == 0) | (p - k == 0), ic, update), zeros)

    return jax.vmap(
                jax.vmap(
                      jax.vmap(_solution_single_update,
                      in_axes=(None, None, 0)),
                in_axes=(None, 0, None)),
            in_axes=(0, None, None)
            )(jnp.arange(diag_data.shape[0]), jnp.arange(diag_data.shape[1]), diag_axis_masked)

  @partial(jax.jit, static_argnums=(0,))
  def _solve(self, X: jnp.ndarray, Y: jnp.ndarray, directions: jnp.ndarray = jnp.array([])) -> jnp.ndarray:

    # Problem sizes
    batch_X, length_X = X.shape[0], X.shape[1]
    batch_Y, length_Y = Y.shape[0], Y.shape[1]
    diag_length = length_X                        # TODO: select max between length_X and length_Y
    diag_iterations = length_X + length_Y - 1

    order_derivative = directions.shape[0] if directions.shape != (0,) else 0
    assert order_derivative <= 2, "directions must be a list of length less or equal to 2."
    state_space_dim = 2 ** order_derivative

    # Set initial conditions
    diag_solution_minus2 = jnp.zeros(shape=(batch_X, batch_Y, diag_length, state_space_dim), dtype=X.dtype)
    diag_solution_minus1 = jnp.zeros_like(diag_solution_minus2)
    diag_solution_minus2 = diag_solution_minus2.at[..., 0, 0].set(1.0)
    diag_solution_minus1 = diag_solution_minus1.at[..., :2, 0].set(1.0)

    diag_data_minus2 = self._data_diag_update(0, self._diag_axis_mask(0, length_X, length_Y), X, Y, directions)
    diag_data_minus1 = self._data_diag_update(1, self._diag_axis_mask(1, length_X, length_Y), X, Y, directions)

    def _loop(p, carry):
      diag_solution_minus2, diag_solution_minus1, diag_data_minus2, diag_data_minus1 = carry
      diag_axis_masked = self._diag_axis_mask(p, length_X, length_Y)
      diag_data = self._data_diag_update(p, diag_axis_masked, X, Y, directions)
      diag_solution = self._solution_diag_update(p, diag_axis_masked,
                                                 diag_solution_minus2, diag_solution_minus1,
                                                 diag_data_minus2, diag_data_minus1, diag_data)
      return diag_solution_minus1, diag_solution, diag_data_minus1, diag_data

    _, diag_solution, _, _ = jax.lax.fori_loop(2, diag_iterations, _loop, (diag_solution_minus2,
                                                                           diag_solution_minus1,
                                                                           diag_data_minus2,
                                                                           diag_data_minus1))

    return diag_solution[:, :, -1, :]

  @partial(jax.jit, static_argnums=(0,))
  def solve(self, X: jnp.ndarray, Y: jnp.ndarray, directions: jnp.ndarray = jnp.array([])) -> jnp.ndarray:

    try:
      num_gpus = jax.lib.xla_bridge.get_backend('gpu').device_count()
    except RuntimeError:
      num_gpus = 0

    if (num_gpus <= 1) | (not self.multi_gpu):
      return self._solve(X, Y, directions)

    # Find the largest integer "num_parallel" <= "total" that is divisible by num_gpus
    total = X.shape[0]
    num_parallel = (total // num_gpus) * num_gpus
    remaining = total - num_parallel

    # Parallise solver across GPUs
    X_parallel = X[:num_parallel]
    directions_parallel = directions[:, :num_parallel, ...] if directions.shape != (0,) else jnp.array([])
    X_sub_tensors = jnp.stack(jnp.array_split(X_parallel, num_gpus))
    directions_sub_tensors = jnp.stack(jnp.array_split(directions_parallel, num_gpus))
    Z_sub_tensors = jax.pmap(self._solve, in_axes=(0, None, 0))(X_sub_tensors, Y, directions_sub_tensors)
    Z = jnp.concatenate(Z_sub_tensors, axis=0)

    # if all the data has been used just return it
    if remaining == 0:
      return Z

    # otherwise perform the final computation on a single GPU and concatenate with the rest
    X_remainder = X[num_parallel:]
    directions_parallel = directions[:, num_parallel:, ...] if directions.shape != (0,) else jnp.array([])
    Z_remainder = self._solve(X_remainder, Y, directions_parallel)
    return jnp.concatenate([Z, Z_remainder], axis=0)
