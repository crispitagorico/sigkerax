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
batch_dim1, batch_dim2, length1, length2, channels = 20, 50, 100, 200, 10
first_batch_paths = jax.random.normal(key1, shape=(batch_dim1, length1, channels)).cumsum(axis=1)
second_batch_paths = jax.random.normal(key2, shape=(batch_dim2, length2, channels)).cumsum(axis=1)
scales = jax.random.exponential(key3, shape=(10,))
signature_kernel = SigKernel(ds=1e-3, dt=1e-3, static_kernel_kind="linear", scales=scales, add_time=False)
signature_kernel_matrix = signature_kernel.kernel_matrix(first_batch_paths, second_batch_paths)
```

## Other signature libraries in JAX

[Signax](https://github.com/Anh-Tong/signax): signatures.
