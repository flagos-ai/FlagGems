---
title: Installation
weight: 20
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Installing FlagGems

## 1. Prerequisites

- You must ensure that the kernel driver and user-space SDK/toolkits for
  your hardware have been installed and configured properly.
  This applies to both NVIDIA platforms and other AI accelerator hardware.

- If you are trying out [the integration with vLLM](../usage/frameworks.md#vllm),
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

> [!NOTE]
> This installs the pure-Python operators from *FlagGems*.
>
> To use the C++ wrapped operators (which reduce dispatch overhead for
> performance-critical paths), you can install a prebuilt native extension
> wheel via an extra:
>
> ```shell
> pip install "flag-gems[cpp-cuda]"
> ```
>
> This pulls in the matching `flag-gems-cpp-cuda` package from the flagOS
> PyPI index. Replace `cpp-cuda` with the extension for your vendor:
> `cpp-musa`, `cpp-npu`, `cpp-gcu`, or `cpp-ix`.
>
> If a prebuilt wheel is not available for your platform, see
> [Build and install from source](#install-from-source).

### 2.1. Set up backend dependencies with `flaggems-setup` {#flaggems-setup}

`pip install flag_gems` installs the pure-Python operator library, but it does
**not** pull in PyTorch or the vendor-specific runtime packages for your
accelerator. The `flaggems-setup` console script — installed alongside FlagGems
— completes the environment by installing the PyTorch stack and vendor packages
for a chosen backend from the flagOS PyPI index.

This is the recommended follow-up when you installed FlagGems from PyPI (as
opposed to installing from source with `setup.sh`, which already performs these
steps for you).

List the available backends:

```shell
flaggems-setup --list
```

Install the dependencies for your backend (the backend keys are the same ones
`setup.sh` accepts):

```shell
# NVIDIA CUDA 12.8
flaggems-setup nvidia-cuda128

# Huawei Ascend CANN 9.0.0
flaggems-setup ascend-cann900

# MetaX MACA 3.8.1
flaggems-setup metax-maca3810
```

Preview the exact commands without running them:

```shell
flaggems-setup nvidia-cuda128 --dry-run
```

By default the script uses `uv pip` when `uv` is on your `PATH`, and falls back
to `pip` otherwise. Override the installer explicitly with `--pip`:

```shell
flaggems-setup nvidia-cuda128 --pip "pip"
```

`flaggems-setup` also installs a Triton-family compiler for you. By default it
selects **FlagTree** when the backend provides one, and falls back to **Triton**
otherwise — the same policy as `setup.sh`. Override the choice with the
`--compiler` flag or the `COMPILER` environment variable:

```shell
# Force vanilla Triton instead of the auto-selected FlagTree
flaggems-setup nvidia-cuda128 --compiler triton
COMPILER=triton flaggems-setup nvidia-cuda128
```

See [Environment variables](#env-vars) and the [Build and Configuration Options Reference](build-options.md#env-vars) for the full `COMPILER` behavior.

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

# MetaX MACA 3.8.1
./setup.sh metax-maca3810
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

FlagGems supports C++ wrapped operators for reduced dispatch overhead on
performance-critical operations. The C++ extension is a **separate per-vendor
package** (e.g., `flag-gems-cpp-cuda`) that installs compiled `.so` files into
the `flag_gems/` namespace alongside the pure-Python operator implementations.

There are two ways to get the C++ extensions:

#### Option A: Build from source with `setup.sh`

```shell
ENABLE_CPP=1 ./setup.sh nvidia-cuda128
```

`setup.sh` automatically:
- Injects the correct vendor name into `cpp/pyproject.toml` via
  `tools/set_cpp_vendor.sh`
- Sets the appropriate `CMAKE_ARGS` (`-DFLAGGEMS_BACKEND=...`)
- Builds and installs the C++ extension from the `cpp/` subdirectory

#### Option B: Manual build from the `cpp/` subdirectory

The C++ extension uses `scikit-build-core` as its build-backend and requires
CMake, a C++ toolchain, and your vendor's SDK. Build from the `cpp/`
subdirectory:

```shell
# Set the vendor name (cuda, musa, npu, gcu, or ix)
tools/set_cpp_vendor.sh cuda

# Build and install
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=CUDA" \
  uv pip install --no-build-isolation ./cpp
```

For manual control over CMake options, see the [CMake options reference](build-options.md#cmake-options).

#### Runtime: enable with `USE_C_EXTENSION`

After installation, set the environment variable to activate C++ paths:

```shell
export USE_C_EXTENSION=1
```

Without this, only `torch.ops.flag_gems.*` and the `c_operators` pybind module
are active; the ATen replacement and `flag_gems.enable()` C++ branches require
it. See the [C++ usage guide](../usage/cpp.md) for details.

## 4. Core Environment Variables Quick Reference {#env-vars}

The following environment variables are essential during installation, build, and runtime:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `COMPILER` | Auto | Controls compiler selection: `flagtree` (forces FlagTree compiler) or `triton` (uses vendor/upstream Triton). Defaults to FlagTree when supported by the backend. |
| `ENABLE_CPP` | `0` | Controls whether to compile C++ native extensions during `./setup.sh`. Set to `1` to build the `cpp/` package. |
| `USE_C_EXTENSION` | `0` | Controls whether to activate C++ operator dispatch at runtime. Set to `1` to dispatch ATen calls via the C++ layer and reduce Python dispatch overhead. |

## 5. Build and Configuration Reference {#build-options-reference}

For detailed specifications and advanced build parameters, please refer to the dedicated reference document:

- **[Full Accelerator Backends Matrix](build-options.md#1-supported-accelerator-backends)**: View all supported hardware platform identifiers (`nvidia-cuda128`, `ascend-cann900`, `hygon`, `metax-maca3810`, `mthreads-musa520`, etc.) and dependencies.
- **[CMake Options Reference](build-options.md#cmake-options)** {#cmake-options}: Definitions for `FLAGGEMS_BUILD_C_EXTENSIONS`, `FLAGGEMS_BACKEND`, `FLAGGEMS_BUILD_CTESTS`, and `CMAKE_ARGS` usage.
- **[`scikit-build-core` Configuration](build-options.md#scikit-build-core-options)** {#scikit-build-core-options}: Build backend settings for packaging C++ extensions (`SKBUILD_CMAKE_BUILD_TYPE`, `SKBUILD_BUILD_DIR`).
- **[External `libtriton_jit` Integration](build-options.md#libtriton-jit)** {#libtriton-jit}: Instructions for linking custom-built TritonJIT libraries.
