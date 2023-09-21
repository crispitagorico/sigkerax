import random
import jax
import jax.numpy as jnp
from sigkerax.sigkernel import SigKernel
from sigkerax.utils import getkey, I0

jax.config.update("jax_enable_x64", True)


class TestSigKernelDerivatives:
  tol = 1e-5
  dtype = jnp.float64
  signature_kernel = SigKernel(ds=1e-2, dt=1e-2)
  batch_X, batch_Y, length_X, length_Y, dim = 10, 5, 15, 20, 3

  def test_derivatives_along_zero_direction(self):
    """Derivatives along the zero path is always equal to 0"""
    X = 1e-1 * jax.random.normal(getkey(), shape=(self.batch_X, self.length_X, self.dim), dtype=self.dtype)
    Y = 1e-1 * jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype)
    directions = jnp.zeros(shape=(2, self.batch_X, self.length_X, self.dim), dtype=self.dtype)
    k_mat = self.signature_kernel.kernel_matrix(X, Y, directions)
    assert jnp.all(jnp.isclose(k_mat[..., 1:], 0.0, atol=self.tol))

  def test_derivatives_along_constant_direction(self):
    """Derivatives along the zero path is always equal to 0"""
    X = 1e-1 * jax.random.normal(getkey(), shape=(self.batch_X, self.length_X, self.dim), dtype=self.dtype)
    Y = 1e-1 * jax.random.normal(getkey(), shape=(self.batch_Y, self.length_Y, self.dim), dtype=self.dtype)
    directions1 = jax.random.normal(getkey(), shape=(2, self.batch_X, self.dim), dtype=self.dtype)
    directions1 = jnp.tile(directions1[:, :, jnp.newaxis, :], (1, 1, self.length_X, 1))
    directions2 = jax.random.normal(getkey(), shape=(2, self.batch_X, self.dim), dtype=self.dtype)
    directions2 = jnp.tile(directions2[:, :, jnp.newaxis, :], (1, 1, self.length_X, 1))
    k_mat1 = self.signature_kernel.kernel_matrix(X, Y, directions1)
    k_mat2 = self.signature_kernel.kernel_matrix(X, Y, directions2)
    assert jnp.all(jnp.isclose(k_mat1[..., 1:], k_mat2[..., 1:], atol=self.tol))