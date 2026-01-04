# Get Start With FlagGems

## Introduction

FlagGems is a high-performance general operator library implemented in Triton language.
It aims to provide a suite of kernel functions to accelerate LLM training and inference.

By registering with the ATen backend of PyTorch, FlagGems facilitates a seamless transition,
allowing users to switch to the Triton function library without the need to modify their model code.
FlagGems is supported by the [FlagTree compiler](https://github.com/flagos-ai/flagtree/)
for different AI chipsets, and OpenAI Triton compiler (for NVIDIA and AMD).

## Quick Installation

FlagGems can be installed either as a pure python package or a package with C-extensions
for better runtime performance.
By default, it does not build the C extensions, See [installation](./installation.md) for how to use C++ runtime.

### Install Build Dependencies

```shell
pip install -U scikit-build-core>=0.11 pybind11>3.0.1 ninja>=1.13.0 cmake>=4.2.1
```

### Installation

Clone the repo to your local environment:

```shell
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
```

Install the FlagTree compiler using a command similar to the following one.

```shell
uv pip install .[flagtree_nvidia]
```

> [!Tip]
> - If you want to use the native Triton instead of FlagTree, you can skip this step.
> - The currently supported variants are `flagtree_ascend`, `flagtree_enflame`, `flagtree_hcu`,
>   `flagtree_iluvatar`, `flagtree_nvidia`, `flagtree_xpu`.

Then use the following command to trigger an installation:

```shell
pip install --no-build-isolation .
```

You can also make an _editable install_ using the following command:

```shell
pip install --no-build-isolation -e .
```

In addition to this, you can build a wheel for install.

```shell
pip install -U build
git clone https://github.com/flagos-ai/FlagGems.git
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

The `experimental_ops` module provides a space for new operators that are not yet ready for production release.
Operators in this module are accessible via `flag_gems.experimental_ops.*`.
These operators follow the same development patterns as the core operators.

```python
import flag_gems

# Global enablement
flag_gems.enable()
result = flag_gems.experimental_ops.rmsnorm(*args)

# Or scoped usage
with flag_gems.use_gems():
    result = flag_gems.experimental_ops.rmsnorm(*args)
```
