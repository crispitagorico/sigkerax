import jax
from typing import Callable
import jax.numpy as jnp
from functools import partial


class PDESolver:
  def __init__(self, static_kernel: Callable[[jnp.ndarray, jnp.ndarray], float]):
    self.static_kernel = static_kernel

  def solve(self):
    pass


class FiniteDifferenceSolver(PDESolver):
  def __init__(self, static_kernel: Callable[[jnp.ndarray, jnp.ndarray], float], multi_gpu: str = False):
    super().__init__(static_kernel)
    self.multi_gpu = multi_gpu

  @partial(jax.jit, static_argnums=(0, 2))
  def _static_kernel_diag_update(self,
                                 diag_axis:  jnp.ndarray,
                                 diag_index: int,
                                 X:          jnp.ndarray,
                                 Y:          jnp.ndarray
                                 ) -> jnp.ndarray:

    def _static_kernel_single_update(i: int, j: int, k: int) -> float:
      return jnp.where(k != -1, self.static_kernel(X[i, k], Y[j, diag_index - k]), 0.0)

    return jax.vmap(
                    jax.vmap(
                             jax.vmap(_static_kernel_single_update,
                             in_axes=(None, None, 0)),
                    in_axes=(None, 0, None)),
            in_axes=(0, None, None)
            )(jnp.arange(X.shape[0]), jnp.arange(Y.shape[0]), diag_axis)

  @partial(jax.jit, static_argnums=(0, 2))
  def _solution_diag_update(self,
                            diag_axis: jnp.ndarray,
                            diag_index: int,
                            diag_solution_minus2: jnp.ndarray,
                            diag_solution_minus1: jnp.ndarray,
                            diag_data_minus2: jnp.ndarray,
                            diag_data_minus1: jnp.ndarray,
                            diag_data: jnp.ndarray,
                            ) -> jnp.ndarray:

    def _solution_single_update(i: int, j: int, k: int) -> float:
      k_00 = diag_solution_minus2[i, j, k - 1]
      k_01 = diag_solution_minus1[i, j, k - 1]
      k_10 = diag_solution_minus1[i, j, k]

      inc = diag_data_minus2[i, j, k - 1] - diag_data_minus1[i, j, k - 1] - diag_data_minus1[i, j, k] + diag_data[
        i, j, k]

      update = (k_01 + k_10) * (1. + 0.5 * inc + (1. / 12.) * inc ** 2) - k_00 * (1. - (1. / 12.) * inc ** 2)

      return jnp.where(k != -1, jnp.where((k == 0) | (diag_index - k == 0), 1.0, update), 0.0)

    return jax.vmap(
                    jax.vmap(
                             jax.vmap(_solution_single_update,
                             in_axes=(None, None, 0)),
                    in_axes=(None, 0, None)),
           in_axes=(0, None, None)
           )(jnp.arange(diag_data.shape[0]), jnp.arange(diag_data.shape[1]), diag_axis)

  @partial(jax.jit, static_argnums=0)
  def _solve(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:

    # problem sizes
    batch_X, length_X = X.shape[0], X.shape[1]
    batch_Y, length_Y = Y.shape[0], Y.shape[1]

    # sizes of diagonals
    diag_length = length_X                    # TODO: select max between length_X and length_Y
    diag_axis = jnp.arange(diag_length)
    diag_iterations = length_X + length_Y - 1

    # initialise first two solution diagonals
    diag_solution_minus2 = jnp.zeros(shape=(batch_X, batch_Y, diag_length), dtype=X.dtype)
    diag_solution_minus2 = diag_solution_minus2.at[:, :, 0].set(1.0)
    diag_solution_minus1 = jnp.zeros(shape=(batch_X, batch_Y, diag_length), dtype=X.dtype)
    diag_solution_minus1 = diag_solution_minus1.at[:, :, :2].set(1.0)

    # initialise first two data diagonals
    diag_data_minus2 = self._static_kernel_diag_update(jnp.where((diag_axis >= 0) & (diag_axis < 1), diag_axis, -1), 0, X, Y)
    diag_data_minus1 = self._static_kernel_diag_update(jnp.where((diag_axis >= 0) & (diag_axis < 2), diag_axis, -1), 1, X, Y)

    def _loop(p, carry):
      diag_solution_minus2, diag_solution_minus1, diag_data_minus2, diag_data_minus1 = carry

      # starting and ending row for current diagonal p
      start_row = jnp.maximum(0, p - (length_Y - 1))
      end_row = jnp.minimum(length_X, p + 1)
      diag_axis_masked = jnp.where((diag_axis >= start_row) & (diag_axis < end_row), diag_axis, -1)

      # current data diagonal
      diag_data = self._static_kernel_diag_update(diag_axis_masked, p, X, Y)

      # populate current diagonal
      diag_solution = self._solution_diag_update(diag_axis_masked, p, diag_solution_minus2, diag_solution_minus1,
                                                 diag_data_minus2, diag_data_minus1, diag_data)

      return diag_solution_minus1, diag_solution, diag_data_minus1, diag_data

    _, diag_solution, _, _ = jax.lax.fori_loop(2, diag_iterations, _loop,
                                               (diag_solution_minus2, diag_solution_minus1, diag_data_minus2, diag_data_minus1))

    return diag_solution[..., -1]

  @partial(jax.jit, static_argnums=0)
  def solve(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:

    try:
      num_gpus = jax.lib.xla_bridge.get_backend('gpu').device_count()
    except RuntimeError:
      num_gpus = 0

    if (num_gpus <= 1) | (not self.multi_gpu):
      return self._solve(X, Y)

    # find the largest integer "parallelisable" <= "total" that is divisible by num_gpus
    total = X.shape[0]
    parallelisable = (total // num_gpus) * num_gpus
    remaining = total - parallelisable

    # parallise solver across GPUs
    X_parallelisable = X[:parallelisable]
    split_size = parallelisable // num_gpus
    X_sub_tensors = jnp.stack(jnp.array_split(X_parallelisable, num_gpus))
    Z_sub_tensors = jax.pmap(self._solve, in_axes=(0, None))(X_sub_tensors, Y)
    Z = jnp.concatenate(Z_sub_tensors, axis=0)

    # if all the data has been used just return it
    if remaining == 0:
      return Z

    # otherwise perform the final computation on a single GPU and concatenate with the rest
    X_remainder = X[parallelisable:]
    Z_remainder = self._solve(X_remainder, Y)
    return jnp.concatenate([Z, Z_remainder], axis=0)
