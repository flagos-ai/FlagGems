# Get Start With FlagGems

## Introduction

FlagGems is a high-performance general operator library implemented in [OpenAI Triton](https://github.com/openai/triton). It aims to provide a suite of kernel functions to accelerate LLM training and inference.

By registering with the ATen backend of PyTorch, FlagGems facilitates a seamless transition, allowing users to switch to the Triton function library without the need to modify their model code.

## Quick Installation

FlagGems can be installed either as a pure python package or a package with C-extensions for better runtime performance. By default, it does not build the C extensions, See [installation](./installation.md) for how to use C++ runtime.

### Install Build Dependencies

```sh
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

### Installation

```shell
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
pip install --no-build-isolation .
# or editble install
pip install --no-build-isolation -e .
```

Or build a wheel

```shell
pip install -U build
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
python -m build --no-isolation --wheel .
```

## How To Use Gems

### Import

```python
# Enable flag_gems permanently
import flag_gems
flag_gems.enable()

# Or Enable flag_gems temporarily
with flag_gems.use_gems():
    pass
```

For example:

```python
import torch
import flag_gems

M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
B = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
with flag_gems.use_gems():
    C = torch.mm(A, B)
```


## How To Use Experimental Gems
The `experimental` module provides a space for new operators that are not yet ready for production release. Operators in this module are accessible via `flag_gems.experimental.*` and follow the same development patterns as core operators.
```
import flag_gems

# Global enablement
flag_gems.enable()
result = flag_gems.experimental.layer_norm(*args)

# Or scoped usage
with flag_gems.use_gems():
    result = flag_gems.experimental.layer_norm(*args)
```
