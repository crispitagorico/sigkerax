import jax
import jax.numpy as jnp
from sigkerax.sigkernel import SigKernel
from sigkerax.utils import getkey, I0

jax.config.update("jax_enable_x64", True)


class TestSigKernel:
  tol = 1e-5
  dtype = jnp.float64
  signature_kernel = SigKernel(ds=1e-3, dt=1e-3)
  batch_X, batch_Y, length_X, length_Y, dim = 10, 5, 15, 20, 3

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
    assert jnp.allclose(k_mat[...,0], k_mat_bessel, atol=self.tol)