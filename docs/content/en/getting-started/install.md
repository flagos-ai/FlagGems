---
title: Installation
weight: 20
---
# Installing FlagGems

## 1. Prerequisites

- You must ensure that the kernel driver and user-space SDK/toolkits for
  your hardware have been installed and configured properly.
  This applies to both NVIDIA platforms and other AI accelerator hardware.

- If you are trying out [the integration with vLLM](/FlagGems/usage/frameworks/#vllm),
  you will need to install [vLLM](https://github.com/vllm-project/vllm)
  or its vendor-customized version if any.

> [!NOTE]
> You do **not** need to manually install Python, PyTorch, or Triton.
> The `setup.sh` script handles all of these automatically based on the
> backend you choose.

## 2. Install from PyPI

*FlagGems* can be installed from [PyPI](https://pypi.org/project/flag-gems/)
using your favorite Python package manager (e.g. `pip`).

```shell
pip install flag_gems
```

> [!INFO]
> **Info**
>
> This Python installation only installs the PyTorch operators implemented
> in Python from *FlagGems*.
> To install the C++-wrapped operators, you will have to
> [build and install from source](#install-from-source).

## 3. Build and install from source {#install-from-source}

### 3.1. Clone the source

```shell
git clone https://github.com/flagos-ai/FlagGems
cd FlagGems/
```

### 3.2. Run setup.sh

The `setup.sh` script is the recommended way to install FlagGems from source.
It reads all configuration from `src/flag_gems/backends.yaml` and automatically:

- Installs [uv](https://github.com/astral-sh/uv) (if not present)
- Installs the correct Python version for your backend
- Creates a virtual environment (`.venv/`)
- Installs build tools, PyTorch, and vendor-specific dependencies
- Installs FlagGems with the appropriate extras
- Installs a compiler ([FlagTree](https://github.com/flagos-ai/flagtree/) or Triton)
- Installs test dependencies
- Writes backend environment variables into `.venv/bin/activate`

```shell
./setup.sh <backend>
```

For example:

```shell
# NVIDIA CUDA 12.8
./setup.sh nvidia-cuda128

# Huawei Ascend CANN 9.0.0
./setup.sh ascend-cann900

# MetaX MACA
./setup.sh metax
```

To see available backends:

```shell
./setup.sh invalid  # prints the list of available backends
```

After setup completes, activate the environment and start working:

```shell
source .venv/bin/activate
pytest tests/test_abs.py -vs
```

> [!TIP]
> **Tips**
>
> - The environment variables for your backend are automatically included
>   in `.venv/bin/activate`. No separate environment setup step is needed.
> - By default, FlagTree is installed as the compiler when available.
>   To use vanilla Triton instead, set `COMPILER=triton` before running setup.sh:
>   ```shell
>   COMPILER=triton ./setup.sh nvidia-cuda128
>   ```

### 3.3. Editable install (for development)

If you are working on the *FlagGems* project (e.g. developing new operators),
you can perform an editable install so that changes to the Python source take
effect immediately without reinstalling:

```shell
source .venv/bin/activate
uv pip install --no-build-isolation -e .
```

> [!NOTE]
> `setup.sh` already installs FlagGems in non-editable mode. Run the command
> above **after** `setup.sh` completes if you want to switch to editable mode.
> The `--no-build-isolation` flag reuses the build tools already in the venv.

### 3.4. C++ extensions (optional)

To build with C++ wrapped operators, set `ENABLE_CPP=1`:

```shell
ENABLE_CPP=1 ./setup.sh nvidia-cuda128
```

This sets the appropriate `CMAKE_ARGS` for your backend automatically.
C++ extensions are still experimental — please assess before using in production.

For manual control over CMake options, see the [CMake options reference](#cmake-options).

## 4. References

### 4.1 Available backends

The full list of supported backends is defined in `src/flag_gems/backends.yaml`.
Each backend specifies:

- Python version
- PyTorch and vendor-specific dependencies
- Triton / FlagTree compiler packages
- Runtime environment variables

### 4.2 Environment variables {#env-vars}

The `COMPILER` environment variable controls which compiler to use:

| Value | Behavior |
|-------|----------|
| _(unset)_ | Auto: FlagTree if available, otherwise Triton |
| `flagtree` | Use FlagTree |
| `triton` | Use vendor Triton |

The `ENABLE_CPP` environment variable enables C++ extensions:

| Value | Behavior |
|-------|----------|
| _(unset or 0)_ | Python-only installation (default) |
| `1` | Build C++ wrapped operators |

### 4.3 CMake options {#cmake-options}

When building with C++ extensions (`ENABLE_CPP=1`), the following CMake
options are set automatically by `setup.sh`. For manual builds, you can
pass them via the `CMAKE_ARGS` environment variable.

| Option | Description | Default |
|--------|-------------|---------|
| `FLAGGEMS_BUILD_C_EXTENSIONS` | Build C++ extensions | `OFF` |
| `FLAGGEMS_BACKEND` | Target backend (`CUDA`, `IX`, `MUSA`, `NPU`, `GCU`) | `CUDA` |
| `FLAGGEMS_BUILD_CTESTS` | Build C++ unit tests | same as `FLAGGEMS_BUILD_C_EXTENSIONS` |
| `FLAGGEMS_INSTALL` | Install CMake package | `ON` |
| `FLAGGEMS_USE_EXTERNAL_TRITON_JIT` | Use external Triton JIT library | `OFF` |
| `FLAGGEMS_USE_EXTERNAL_PYBIND11` | Use external pybind11 | `ON` |
| `FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP` | Build pointwise dynamic C++ module | `OFF` |

### 4.4 `scikit-build-core` options {#scikit-build-core-options}

The `scikit-build-core` tool is a build-backend that bridges CMake
and the Python build system, making it easier to create Python modules with CMake.
Some commonly used environment variables for configuring `scikit-build-core` include:

1. `SKBUILD_CMAKE_BUILD_TYPE`, used to configure the build type of the project.
   Valid values are `Release`, `Debug`, `RelWithDebInfo` and `MinSizeRel`;

1. `SKBUILD_BUILD_DIR`, which configures the build directory of the project.
   The default value is `build/{cache_tag}`, which is defined in `pyproject.toml`.

Note that for the environment variable `SKBUILD_CMAKE_ARGS`, multiple options
are separated by semicolons (`;`), whereas for `CMAKE_ARGS`, they are separated
by spaces.

### 4.5 The `libtriton_jit` library {#libtriton-jit}

The C++ extension of FlagGems depends on [TritonJIT](https://github.com/flagos-ai/libtriton_jit/),
a library that implements a Triton JIT runtime in C++ and enables calling
Triton JIT functions from C++ code.

If you are building with an external TritonJIT, build and install it first,
then pass `-DTritonJIT_ROOT=<install path>` to CMake:

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=/usr/local/lib/libtriton_jit" \
ENABLE_CPP=1 ./setup.sh nvidia-cuda128
```
