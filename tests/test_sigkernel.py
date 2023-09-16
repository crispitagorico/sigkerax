import random
import jax
import jax.numpy as jnp
from sigkerax.static_kernels import linear_kernel
from sigkerax.solver import FiniteDifferenceSolver
from sigkerax.sigkernel import SigKernel

jax.config.update("jax_enable_x64", True)


def getkey():
  return jax.random.PRNGKey(random.randint(0, 2 ** 31 - 1))


def I0(x_squared, num_terms, dtype):
  """Modified Bessel function of order 0"""
  factorials = jnp.cumprod(jnp.arange(1, num_terms, dtype=dtype))
  return 1.0 + jnp.sum(
    jnp.array([(1.0 / (factorials[k - 1] ** 2)) * ((x_squared / 4.0) ** k) for k in jnp.arange(1, num_terms)]), axis=0)


class TestSigKernel:
  tol = 1e-5
  dtype = jnp.float64

  static_kernel = lambda x, y: linear_kernel(x, y)
  pde_solver = FiniteDifferenceSolver(static_kernel=static_kernel, multi_gpu=False)
  signature_kernel = SigKernel(pde_solver=pde_solver,
                               s0=0.0, t0=0.0, S=1.0, T=1.0,
                               ds=1e-3, dt=1e-3,
                               add_time=False,
                               interpolation="linear")

  batch_X, batch_Y, length_X, length_Y, dim = 5, 5, 10, 10, 5

  def test_constant_path(self):
    """Inner product with signature of a constant paths is always equal to 1"""
    X = jax.random.normal(getkey(), shape=(self.batch_X, self.dim), dtype=self.dtype)
    X = jnp.tile(X[:, jnp.newaxis, :], (1, self.length_X, 1))
    Y = jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype)
    k_mat = self.signature_kernel.kernel_matrix(X, Y)
    assert jnp.all(jnp.isclose(k_mat, 1.0, atol=self.tol))

  def test_padded_path(self):
    """Padding with lastly observed value doesn't change the inner product."""
    X = 1e-1*jax.random.normal(getkey(), shape=(self.batch_X, self.length_X, self.dim), dtype=self.dtype)
    X_pad = jnp.concatenate([X, jnp.tile(X[:, -1:, :], (1, 5, 1))], axis=1)
    Y = 1e-1*jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype)
    k_mat = self.signature_kernel.kernel_matrix(X, Y)
    k_mat_pad = self.signature_kernel.kernel_matrix(X_pad, Y)
    assert jnp.allclose(k_mat, k_mat_pad, atol=self.tol)

  def test_linear_paths(self):
    """signature kernel of linear paths is equal to the modified Bessel function of order 0"""
    X = 5e-1*jax.random.normal(getkey(), shape=(self.batch_X, 2, self.dim), dtype=self.dtype).cumsum(axis=1)
    Y = 5e-1*jax.random.normal(getkey(), shape=(self.batch_Y, 2, self.dim), dtype=self.dtype).cumsum(axis=1)
    k_mat = self.signature_kernel.kernel_matrix(X, Y)
    X_inc = X[:, 1, :] - X[:, 0, :]
    Y_inc = Y[:, 1, :] - Y[:, 0, :]
    ip_inc_mat = jnp.einsum('ik,jk->ij', X_inc, Y_inc)
    k_mat_bessel = I0(4.0 * ip_inc_mat, num_terms=50, dtype=self.dtype)
    assert jnp.allclose(k_mat, k_mat_bessel, atol=self.tol)
