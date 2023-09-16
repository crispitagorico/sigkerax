<h1 align='center'>Sigkerax</h1>

Sigkerax is a [JAX](https://github.com/google/jax) library for [signature kernels](https://arxiv.org/pdf/2006.14794.pdf). 

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![][versions-image]][versions-url]

...

<!-- Badges: -->

[pypi-image]: https://img.shields.io/pypi/v/sigkerax
[pypi-url]: https://pypi.org/project/sigkerax/
[build-image]: https://github.com/crispitagorico/sigkerax/actions/workflows/build.yaml/badge.svg
[build-url]: https://github.com/crispitagorico/sigkerax/actions/workflows/build.yaml
[coverage-image]: https://codecov.io/gh/mathspp/extendedjson/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/gh/mathspp/sigkerax/
[stars-image]: https://img.shields.io/github/stars/mathspp/sigkerax/
[stars-url]: https://github.com/mathspp/sigkerax
[versions-image]: https://img.shields.io/pypi/pyversions/sigkerax/
[versions-url]: https://pypi.org/project/sigkerax/


Features include:
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
import jax.numpy as jnp
from sigkerax.static_kernels import linear_kernel
from sigkerax.solver import FiniteDifferenceSolver
from sigkerax.sigkernel import SigKernel

static_kernel = lambda x, y: linear_kernel(x, y, scale=1.0)
pde_solver = FiniteDifferenceSolver(static_kernel=static_kernel)
signature_kernel = SigKernel(pde_solver=pde_solver, ds=1e-3, dt=1e-3, add_time=False)

batch_dim1, batch_dim2, length1, length2, channels = 20, 50, 100, 200, 10
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
first_batch_paths = jax.random.normal(key1, shape=(batch_dim1, length1, channels)).cumsum(axis=1)
second_batch_paths = jax.random.normal(key2, shape=(batch_dim2, length2, channels)).cumsum(axis=1)
sigker_matrix = signature_kernel.kernel_matrix(first_batch_paths, second_batch_paths)
```

## Other signature libraries in JAX

[Signax](https://github.com/Anh-Tong/signax): signatures.
