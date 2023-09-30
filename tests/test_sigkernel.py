import jax
import jax.numpy as jnp
from sigkerax.sigkernel import SigKernel
from sigkerax.utils import getkey, I0, iisig_gram

jax.config.update("jax_enable_x64", True)


class TestSigKernel:
  dtype = jnp.float64
  signature_kernel = SigKernel(refinement_factor=5)
  batch_X, batch_Y, length_X, length_Y, dim = 10, 5, 15, 20, 3

  def test_constant_path(self):
    """Inner product with signature of a constant paths is always equal to 1"""
    X = jax.random.normal(getkey(), shape=(self.batch_X, self.dim), dtype=self.dtype)
    X = jnp.tile(X[:, jnp.newaxis, :], (1, self.length_X, 1))
    Y = jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype)
    k_mat = self.signature_kernel.kernel_matrix(X, Y)
    assert jnp.all(jnp.isclose(k_mat, 1.0, atol=1e-12))

  def test_padded_path(self):
    """Padding with lastly observed value doesn't change the inner product."""
    length_to_pad = 10
    X = 1e-1*jax.random.normal(getkey(), shape=(self.batch_X, self.length_X, self.dim), dtype=self.dtype)
    X_pad = jnp.concatenate([X, jnp.tile(X[:, -1:, :], (1, length_to_pad, 1))], axis=1)
    Y = 1e-1*jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype)
    k_mat = self.signature_kernel.kernel_matrix(X, Y)
    k_mat_pad = self.signature_kernel.kernel_matrix(X_pad, Y)
    assert jnp.allclose(k_mat, k_mat_pad, atol=1e-12)

  def test_linear_paths(self):
    """signature kernel of linear paths is equal to the modified Bessel function of order 0"""
    X = 1e-1*jax.random.normal(getkey(), shape=(self.batch_X, 2, self.dim), dtype=self.dtype).cumsum(axis=1)
    Y = 1e-1*jax.random.normal(getkey(), shape=(self.batch_Y, 2, self.dim), dtype=self.dtype).cumsum(axis=1)
    k_mat = self.signature_kernel.kernel_matrix(X, Y)
    X_inc = X[:, 1, :] - X[:, 0, :]
    Y_inc = Y[:, 1, :] - Y[:, 0, :]
    ip_inc_mat = jnp.einsum('ik,jk->ij', X_inc, Y_inc)
    k_mat_bessel = I0(4.0 * ip_inc_mat, num_terms=60, dtype=self.dtype)
    assert jnp.mean((k_mat_bessel - k_mat[..., 0])**2) < 1e-12

  def test_with_iisignature(self):
    X = 1e-1 * jax.random.normal(getkey(), shape=(self.batch_X, self.length_X, self.dim), dtype=self.dtype).cumsum(axis=1)
    Y = 1e-1 * jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype).cumsum(axis=1)
    k_mat = self.signature_kernel.kernel_matrix(X, Y)
    iisig_matrix = iisig_gram(X, Y, width=12)
    print(jnp.mean((iisig_matrix - k_mat[..., 0])**2))
    assert jnp.mean((iisig_matrix - k_mat[..., 0])**2) < 1e-9