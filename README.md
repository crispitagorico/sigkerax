<h1 align='center'>Sigkerax</h1>

Sigkerax is a [JAX](https://github.com/google/jax) library for [signature kernels](https://arxiv.org/pdf/2006.14794.pdf). Features include:
- Custom static kernels (at time of writing only linear and RBF kernels are implemented);
- All the benefits of working with JAX: autodiff, autoparallelism, GPU/TPU support etc.

## Installation

```bash
pip install sigkerax
```

Requires Python 3.8+, JAX 0.4.11+.

## Documentation

Coming soon...

## Quick example

Lineax can solve a least squares problem with an explicit matrix operator:

```python
import jax
from sigkerax.sigkernel import SigKernel
key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), num=3)
batch_X, batch_Y, length_X, length_Y, channels = 20, 50, 100, 200, 10
X = 1e-1 * jax.random.normal(key1, shape=(batch_X, length_X, channels)).cumsum(axis=1)
Y = 1e-1 * jax.random.normal(key2, shape=(batch_Y, length_Y, channels)).cumsum(axis=1)
scales = jax.random.exponential(key3, shape=(10,))
signature_kernel = SigKernel(refinement_factor=2, static_kernel_kind="linear", scales=scales, add_time=False)
signature_kernel_matrix = signature_kernel.kernel_matrix(X, Y)
```

## Other signature libraries in JAX

[Signax](https://github.com/Anh-Tong/signax): signatures.
