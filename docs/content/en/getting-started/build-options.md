---
title: Build and Configuration Options Reference
weight: 25
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

# Build and Configuration Options Reference

This document provides a comprehensive technical reference for accelerator backends, environment variables, CMake build options, and C++ native extension integration in *FlagGems*.

## 1. Supported Accelerator Backends

*FlagGems* maintains [`src/flag_gems/backends.yaml`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/backends.yaml) as the single source of truth for all hardware backend configurations and dependency stacks.

### 1.1 Querying Available Backends

In an environment where `flag_gems` is installed, you can query all supported backend names using the `flaggems-setup` CLI tool:

```shell
flaggems-setup --list
```

### 1.2 Backends and Build Matrix

The table below summarizes the mainstream hardware accelerator platforms, recommended Python versions, and corresponding C++ native extension backends (`cmake_backend`):

| Backend Key (`<backend>`) | Hardware Platform / Vendor | Python | C++ Extension (`FLAGGEMS_BACKEND`) | Recommended Compiler |
| :--- | :--- | :--- | :--- | :--- |
| `nvidia-cuda128` | NVIDIA GPU (CUDA 12.8) | 3.12 | `CUDA` | FlagTree / Triton |
| `nvidia-cuda133` | NVIDIA GPU (CUDA 13.3) | 3.12 | `CUDA` | FlagTree / Triton |
| `ascend-cann900` | Huawei Ascend NPU (CANN 9.0.0) | 3.11 | `NPU` | FlagTree / Triton |
| `ascend-cann850` | Huawei Ascend NPU (CANN 8.5.0) | 3.11 | `NPU` | FlagTree / Triton |
| `hygon` | Hygon DCU (DTK) | 3.10 | Python-only / Under adaptation | Triton |
| `metax-maca3810` | MetaX GPU (MACA 3.8.1) | 3.12 | Python-only | Triton |
| `metax-maca3720` | MetaX GPU (MACA 3.7.2) | 3.12 | Python-only | Triton |
| `mthreads-musa520` | Moore Threads GPU (MUSA 5.2.0) | 3.10 | `MUSA` | FlagTree / Triton |
| `mthreads-musa436` | Moore Threads GPU (MUSA 4.3.6) | 3.10 | `MUSA` | FlagTree / Triton |
| `enflame-tops1910` | Enflame GCU (TopsRider 1.9.1) | 3.12 | `GCU` | Triton |
| `cambricon-neuware472`| Cambricon MLU (Neuware 4.7.2)| 3.12 | `MLU` | Triton |
| `cambricon-neuware443`| Cambricon MLU (Neuware 4.4.3)| 3.10 | `MLU` | Triton |
| `iluvatar` | Iluvatar CoreX GPU (BI-V) | 3.12 | `IX` | Triton |
| `kunlunxin` | Kunlunxin XPU | 3.10 | Python-only | Triton |
| `biren-br2xx` | Biren Technology GPU (BR2xx) | 3.10 | Python-only | Triton |

Each entry in `backends.yaml` specifies:
- Recommended Python runtime version;
- PyTorch dependencies and vendor-specific accelerator packages;
- Triton or FlagTree compiler distribution compatible with the hardware;
- Environment variables and setup scripts automatically sourced into `.venv/bin/activate`.

---

## 2. Environment Variables Reference {#env-vars}

Use the following environment variables to control installation, build options, and runtime dispatch:

### 2.1 Compiler and Build Control

| Variable | Values | Description |
| :--- | :--- | :--- |
| `COMPILER` | `flagtree` | Forces the FlagTree compiler. |
| | `triton` | Forces upstream or vendor-specific Triton compiler. |
| | *(unset)* | **Default auto-selection**: Prefers FlagTree if provided by the backend configuration, otherwise falls back to vendor Triton. |
| `ENABLE_CPP` | `1` | Automatically compiles and installs the C++ native extension under `cpp/` when running `./setup.sh`. |
| | `0` or *(unset)* | **Default**: Python-only operator installation. |

### 2.2 Runtime Control

| Variable | Values | Description |
| :--- | :--- | :--- |
| `USE_C_EXTENSION` | `1` | Activates C++ accelerated dispatch path. When enabled, `flag_gems.enable()` intercepts ATen calls via the C++ layer, reducing Python dispatch overhead. |
| | `0` or *(unset)* | **Default**: Uses pure-Python ATen dispatch. |

### 2.3 Device Visibility and Isolation

When running in multi-GPU or shared multi-tenant clusters, isolate visible devices using the platform-specific environment variables:

| Accelerator Platform | Variable | Example |
| :--- | :--- | :--- |
| NVIDIA GPU | `CUDA_VISIBLE_DEVICES` | `export CUDA_VISIBLE_DEVICES=0` |
| Huawei Ascend NPU | `ASCEND_RT_VISIBLE_DEVICES` | `export ASCEND_RT_VISIBLE_DEVICES=0` |
| Moore Threads MUSA | `MUSA_VISIBLE_DEVICES` | `export MUSA_VISIBLE_DEVICES=0` |
| Enflame GCU | `TOPS_VISIBLE_DEVICES` | `export TOPS_VISIBLE_DEVICES=0` |
| Cambricon MLU | `MLU_VISIBLE_DEVICES` | `export MLU_VISIBLE_DEVICES=0` |

---

## 3. CMake Build Options Reference {#cmake-options}

When building C++ extensions (`ENABLE_CPP=1`), `setup.sh` sets appropriate CMake options automatically.

For **manual builds** or custom packaging, pass options via the `CMAKE_ARGS` environment variable:

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=CUDA -DFLAGGEMS_BUILD_CTESTS=ON" \
  uv pip install --no-build-isolation ./cpp
```

### 3.1 CMake Parameters

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `FLAGGEMS_BUILD_C_EXTENSIONS` | BOOL | `OFF` | Whether to build C++ native extensions. |
| `FLAGGEMS_BACKEND` | STRING | `CUDA` | Target hardware backend (`CUDA`, `NPU`, `MUSA`, `GCU`, `MLU`, `IX`, etc.). |
| `FLAGGEMS_BUILD_CTESTS` | BOOL | same as `FLAGGEMS_BUILD_C_EXTENSIONS` | Whether to compile C++ native unit tests (`ctest`). |
| `FLAGGEMS_INSTALL` | BOOL | `ON` | Install CMake export package for downstream consumers. |
| `FLAGGEMS_USE_EXTERNAL_TRITON_JIT` | BOOL | `OFF` | Link against an externally built `libtriton_jit` library. |
| `FLAGGEMS_USE_EXTERNAL_PYBIND11` | BOOL | `ON` | Use existing system/environment pybind11 installation. |
| `FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP` | BOOL | `OFF` | Build C++ pointwise dynamic codegen module. |

---

## 4. `scikit-build-core` Options {#scikit-build-core-options}

> [!NOTE]
> The primary Python package `flag-gems` uses standard `setuptools`. `scikit-build-core` is used **only** when building C++ extensions from the `cpp/` directory (e.g. `flag-gems-cpp-cuda`).

`scikit-build-core` bridges CMake with Python packaging tools and accepts configuration through the following environment variables:

### 4.1 Configuration Variables

1. **`SKBUILD_CMAKE_BUILD_TYPE`**: CMake build configuration mode.
   - Values: `Release` (recommended), `Debug`, `RelWithDebInfo`, `MinSizeRel`.
2. **`SKBUILD_BUILD_DIR`**: Output directory for build artifacts.
   - Default: `build/{cache_tag}` (configured in `pyproject.toml`).
3. **`SKBUILD_CMAKE_ARGS`**: Arguments passed directly to CMake.

> [!WARNING]
> **Delimiter difference**:
> - Multiple options in `CMAKE_ARGS` are separated by **spaces** (`CMAKE_ARGS="-DFOO=1 -DBAR=2"`).
> - Multiple options in `SKBUILD_CMAKE_ARGS` must be separated by **semicolons** (`SKBUILD_CMAKE_ARGS="-DFOO=1;-DBAR=2"`).

---

## 5. External `libtriton_jit` Library Integration {#libtriton-jit}

*FlagGems* C++ extensions rely on [libtriton_jit](https://github.com/flagos-ai/libtriton_jit/), an open-source C++ runtime that enables C++ programs to execute Triton JIT GPU/NPU kernels.

By default, the build system handles dependencies automatically. If you want to use a locally compiled or custom version of TritonJIT:

1. Build and install `libtriton_jit` to a custom path (e.g. `/usr/local/lib/libtriton_jit`);
2. Enable external TritonJIT and pass `-DTritonJIT_ROOT=<install path>` to CMake:

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=/usr/local/lib/libtriton_jit" \
ENABLE_CPP=1 ./setup.sh nvidia-cuda128
```
