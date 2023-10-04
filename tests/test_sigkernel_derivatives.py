import jax
import jax.numpy as jnp
from sigkerax.sigkernel import SigKernel
from sigkerax.utils import getkey

jax.config.update("jax_enable_x64", True)


class TestSigKernelDerivatives:
  dtype = jnp.float64
  signature_kernel = SigKernel(refinement_factor=5)
  batch_X, batch_Y, length_X, length_Y, dim = 10, 5, 15, 20, 3

  def test_derivatives_along_zero_direction(self):
    """Derivatives along the zero path is always equal to 0"""
    X = jax.random.normal(getkey(), shape=(self.batch_X, self.length_X, self.dim), dtype=self.dtype).cumsum(axis=1)
    Y = jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype).cumsum(axis=1)
    directions = jnp.zeros(shape=(2, self.batch_X, self.length_X, self.dim), dtype=self.dtype)
    k_mat = self.signature_kernel.kernel_matrix(X, Y, directions)
    assert jnp.all(jnp.isclose(k_mat[..., 1:], 0.0, atol=1e-8))

  def test_derivatives_along_constant_direction(self):
    """Derivatives along the zero path is always equal to 0"""
    X = 1e-1 * jax.random.normal(getkey(), shape=(self.batch_X, self.length_X, self.dim), dtype=self.dtype).cumsum(axis=1)
    Y = 1e-1 * jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype).cumsum(axis=1)
    directions1 = 1e-1 * jax.random.normal(getkey(), shape=(2, self.batch_X, self.dim), dtype=self.dtype)
    directions1 = jnp.tile(directions1[:, :, jnp.newaxis, :], (1, 1, self.length_X, 1))
    directions2 = 1e-1 * jax.random.normal(getkey(), shape=(2, self.batch_X, self.dim), dtype=self.dtype)
    directions2 = jnp.tile(directions2[:, :, jnp.newaxis, :], (1, 1, self.length_X, 1))
    k_mat1 = self.signature_kernel.kernel_matrix(X, Y, directions1)
    k_mat2 = self.signature_kernel.kernel_matrix(X, Y, directions2)
    assert jnp.all(jnp.isclose(k_mat1[..., 1:], k_mat2[..., 1:], atol=1e-8))

  def test_derivative_padded_direction(self):
    """Padding with lastly observed value doesn't change the derivative."""
    length_to_pad = 10
    X = jax.random.normal(getkey(), shape=(self.batch_X, self.length_X, self.dim), dtype=self.dtype).cumsum(axis=1)
    X_pad = jnp.concatenate([X, jnp.tile(X[:, -1:, :], (1, length_to_pad, 1))], axis=1)
    directions = jax.random.normal(getkey(), shape=(2, self.batch_X, self.length_X, self.dim), dtype=self.dtype).cumsum(axis=2)
    directions_pad = jnp.concatenate([directions, jnp.tile(directions[:, :, -1:, :], (1, 1, length_to_pad, 1))], axis=2)
    Y = jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype).cumsum(axis=1)
    k_mat = self.signature_kernel.kernel_matrix(X, Y, directions)
    k_mat_pad = self.signature_kernel.kernel_matrix(X_pad, Y, directions_pad)
    assert jnp.allclose(k_mat, k_mat_pad, atol=1e-8)

  def test_derivatives_perturbative(self, eps=1e-6):
    X = 5e-1 * jax.random.normal(getkey(), shape=(self.batch_X, self.length_X, self.dim), dtype=self.dtype).cumsum(axis=1)
    Y = 5e-1 * jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype).cumsum(axis=1)
    directions = 5e-1 * jax.random.normal(getkey(), shape=(2, self.batch_X, self.length_X, self.dim), dtype=self.dtype).cumsum(axis=2)
    k_mats = self.signature_kernel.kernel_matrix(X, Y, directions)
    k_diff, k2_diff = k_mats[..., 1], k_mats[..., 2]
    k_eps = (1. / eps) * (self.signature_kernel.kernel_matrix(X + eps * directions[0], Y)
                          - self.signature_kernel.kernel_matrix(X, Y))[..., 0]
    k2_eps = (1. / eps**2) * (self.signature_kernel.kernel_matrix(X + eps * (directions[0] + directions[1]), Y)
                              - self.signature_kernel.kernel_matrix(X + eps * directions[0], Y)
                              - self.signature_kernel.kernel_matrix(X + eps * directions[1], Y)
                              + self.signature_kernel.kernel_matrix(X, Y))[..., 0]
    # print(jnp.mean(k_diff), jnp.mean(k_eps), jnp.mean(k2_diff), jnp.mean(k2_eps))
    assert jnp.allclose(k_diff, k_eps, atol=1e-4) #& jnp.allclose(k2_diff, k2_eps, atol=1e-3)