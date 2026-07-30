---
title: 安装 FlagGems
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


<!--
# Installing FlagGems
-->
# 安装 FlagGems

<!--
## 1. Prerequisites

- You must ensure that the kernel driver and user-space SDK/toolkits for
  your hardware have been installed and configured properly.
  This applies to both NVIDIA platforms and other AI accelerator hardware.
-->
## 1. 环境准备

- 你必须确保为自己的硬件正确地安装了内核态的驱动程序和用户空间的 SDK 或工具链，
  并且均已配置正确，工作正常。无论是 NVIDIA 平台还是其他 AI 加速器硬件，
  这一点都适用。

<!--
- If you are trying out [the integration with vLLM](/FlagGems/usage/frameworks/#vllm),
  you will need to install [vLLM](https://github.com/vllm-project/vllm)
  or its vendor-customized version if any.
-->
- 如果你想尝试将 *FlagGems* [与 vLLM 集成](/FlagGems/zh-cn/usage/frameworks/#vllm)，
  则需要安装 [vLLM](https://github.com/vllm-project/vllm)，或者厂商定制版本
  （如果有的话）。

<!--
> [!NOTE]
> You do **not** need to manually install Python, PyTorch, or Triton.
> The `setup.sh` script handles all of these automatically based on the
> backend you choose.
-->
> [!NOTE]
> 你**不需要**手动安装 Python、PyTorch 或 Triton。
> `setup.sh` 脚本会根据你选择的后端自动完成所有这些安装。

<!--
## 2. Install from PyPI

*FlagGems* can be installed from [PyPI](https://pypi.org/project/flag-gems/)
using your favorite Python package manager (e.g. `pip`).
-->
## 2. 从 PyPI 安装

你可以使用自己常用的 Python 包管理器（例如 `pip`）从
[PyPI](https://pypi.org/project/flag-gems/) 安装 *FlagGems*：

```shell
pip install flag_gems
```

<!--
> [!INFO]
> **Info**
>
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
>-->
> [!INFO]
> **提示**
>
> 上述命令仅安装 *FlagGems* 中纯 Python 实现的算子。
>
> 如需使用 C++ 封装的算子（可减少性能关键路径上的 dispatch 开销），
> 可以通过 extras 安装预构建的原生扩展 wheel：
>
> ```shell
> pip install "flag-gems[cpp-cuda]"
> ```
>
> 该命令会从 flagOS PyPI 仓库拉取匹配的 `flag-gems-cpp-cuda` 包。
> 请将 `cpp-cuda` 替换为你所使用的硬件对应的扩展：
> `cpp-musa`、`cpp-npu`、`cpp-gcu` 或 `cpp-ix`。
>
> 如果你的平台上没有可用的预构建 wheel，请参阅
> [从源码构建安装](#install-from-source)。

<!--
## 3. Build and install from source {#install-from-source}

### 3.1. Clone the source
-->
## 3. 从源码构建、安装 {#install-from-source}

### 3.1 克隆源代码

```shell
git clone https://github.com/flagos-ai/FlagGems
cd FlagGems/
```

<!--
### 3.2. Run setup.sh

The `setup.sh` script is the recommended way to install FlagGems from source.
It reads all configuration from `src/flag_gems/backends.yaml` and automatically:

- Installs uv (if not present)
- Installs the correct Python version for your backend
- Creates a virtual environment (`.venv/`)
- Installs build tools, PyTorch, and vendor-specific dependencies
- Installs FlagGems with the appropriate extras
- Installs a compiler (FlagTree or Triton)
- Installs test dependencies
- Writes backend environment variables into `.venv/bin/activate`
-->
### 3.2 运行 setup.sh

`setup.sh` 脚本是从源码安装 FlagGems 的推荐方式。
它从 `src/flag_gems/backends.yaml` 读取所有配置信息，并自动完成以下操作：

- 安装 [uv](https://github.com/astral-sh/uv)（如果尚未安装）
- 安装你的后端所需的正确 Python 版本
- 创建虚拟环境（`.venv/`）
- 安装构建工具、PyTorch 以及厂商特定的依赖包
- 安装 FlagGems 及其对应的 extras
- 安装编译器（[FlagTree](https://github.com/flagos-ai/flagtree/) 或 Triton）
- 安装测试依赖
- 将后端环境变量写入 `.venv/bin/activate`

```shell
./setup.sh <backend>
```

<!--
For example:
-->
例如：

```shell
# NVIDIA CUDA 12.8
./setup.sh nvidia-cuda128

# Huawei Ascend CANN 9.0.0
./setup.sh ascend-cann900

# MetaX MACA
./setup.sh metax
```

<!--
To see available backends:
-->
查看可用的后端列表：

```shell
./setup.sh invalid  # 打印可用后端列表
```

<!--
After setup completes, activate the environment and start working:
-->
安装完成后，激活环境即可开始使用：

```shell
source .venv/bin/activate
pytest tests/test_abs.py -vs
```

<!--
> [!TIP]
> **Tips**
>
> - The environment variables for your backend are automatically included
>   in `.venv/bin/activate`. No separate environment setup step is needed.
> - By default, FlagTree is installed as the compiler when available.
>   To use vanilla Triton instead, set `COMPILER=triton` before running setup.sh:
-->
> [!TIP]
> **提示**
>
> - 后端所需的环境变量已自动包含在 `.venv/bin/activate` 中，
>   无需额外的环境配置步骤。
> - 默认情况下，如果 FlagTree 可用则安装 FlagTree 作为编译器。
>   如需使用原生 Triton，可在运行 setup.sh 前设置 `COMPILER=triton`：
>   ```shell
>   COMPILER=triton ./setup.sh nvidia-cuda128
>   ```

<!--
### 3.3. Editable install (for development)

If you are working on the *FlagGems* project (e.g. developing new operators),
you can perform an editable install so that changes to the Python source take
effect immediately without reinstalling:
-->
### 3.3 可编辑安装（用于开发）

如果你在参与 *FlagGems* 的开发（例如开发新的算子），
可以执行可编辑模式的安装，使得对 Python 源码的修改无需重新安装即可生效：

```shell
source .venv/bin/activate
uv pip install --no-build-isolation -e .
```

<!--
> [!NOTE]
> `setup.sh` already installs FlagGems in non-editable mode. Run the command
> above **after** `setup.sh` completes if you want to switch to editable mode.
> The `--no-build-isolation` flag reuses the build tools already in the venv.
-->
> [!NOTE]
> `setup.sh` 默认以非可编辑模式安装 FlagGems。如需切换为可编辑模式，
> 请在 `setup.sh` 完成**之后**执行上述命令。
> `--no-build-isolation` 参数会复用 venv 中已安装的构建工具。

<!--
### 3.4. C++ extensions (optional)

FlagGems supports C++ wrapped operators for reduced dispatch overhead on
performance-critical operations. The C++ extension is a **separate per-vendor
package** (e.g., `flag-gems-cpp-cuda`) that installs compiled `.so` files into
the `flag_gems/` namespace alongside the pure-Python operator implementations.

There are two ways to get the C++ extensions:

#### Option A: Build from source with `setup.sh`
-->
### 3.4 C++ 扩展（可选）

*FlagGems* 通过 C++ 封装算子在性能关键路径上减少 dispatch 开销。
C++ 扩展是一个**独立的、按硬件供应商区分的包**（如 `flag-gems-cpp-cuda`），
它将编译好的 `.so` 文件安装到 `flag_gems/` 命名空间下，与纯 Python 算子并列。

C++ 扩展的获取方式有两种：

#### 方式 A：通过 `setup.sh` 从源码构建

```shell
ENABLE_CPP=1 ./setup.sh nvidia-cuda128
```

`setup.sh` 会自动完成以下操作：
- 通过 `tools/set_cpp_vendor.sh` 向 `cpp/pyproject.toml` 注入正确的 vendor 标识
- 设置对应的 `CMAKE_ARGS`（`-DFLAGGEMS_BACKEND=...`）
- 从 `cpp/` 子目录构建并安装 C++ 扩展

#### 方式 B：在 `cpp/` 子目录中手动构建

C++ 扩展使用 `scikit-build-core` 作为构建后端，需要 CMake、C++ 工具链以及
对应硬件厂商的 SDK。从 `cpp/` 子目录进行构建：

```shell
# 设置 vendor 标识（cuda、musa、npu、gcu 或 ix）
tools/set_cpp_vendor.sh cuda

# 构建并安装
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=CUDA" \
  uv pip install --no-build-isolation ./cpp
```

<!--
For manual control over CMake options, see the CMake options reference.
-->
如需手动控制 CMake 选项，请参阅 [CMake 选项参考](#cmake-options)。

#### 运行时：通过 `USE_C_EXTENSION` 启用

安装完成后，设置以下环境变量来激活 C++ 路径：

```shell
export USE_C_EXTENSION=1
```

如果不设置该变量，只有 `torch.ops.flag_gems.*` 及 `c_operators` pybind
模块生效；ATen 替换和 `flag_gems.enable()` 的 C++ 分支仍需此变量。
详见 [C++ 使用指南](/FlagGems/zh-cn/usage/cpp/)。

<!--
## 4. References

### 4.1 Available backends

The full list of supported backends is defined in `src/flag_gems/backends.yaml`.
Each backend specifies:

- Python version
- PyTorch and vendor-specific dependencies
- Triton / FlagTree compiler packages
- Runtime environment variables
-->
## 4. 参考资料

### 4.1 可用后端

所有支持的后端定义在 `src/flag_gems/backends.yaml` 中。
每个后端指定了：

- Python 版本
- PyTorch 及厂商特定的依赖包
- Triton / FlagTree 编译器包
- 运行时环境变量

<!--
### 4.2 Environment variables {#env-vars}

The `COMPILER` environment variable controls which compiler to use:
-->
### 4.2 环境变量 {#env-vars}

环境变量 `COMPILER` 控制使用哪个编译器：

<!--
| Value | Behavior |
|-------|----------|
| _(unset)_ | Auto: FlagTree if available, otherwise Triton |
| `flagtree` | Use FlagTree |
| `triton` | Use vendor Triton |
-->
| 取值 | 行为 |
|------|------|
| _(未设置)_ | 自动选择：优先 FlagTree，否则使用 Triton |
| `flagtree` | 使用 FlagTree |
| `triton` | 使用厂商 Triton |

<!--
The `ENABLE_CPP` environment variable enables C++ extensions:
-->
环境变量 `ENABLE_CPP` 用来启用 C++ 扩展：

<!--
| Value | Behavior |
|-------|----------|
| _(unset or 0)_ | Python-only installation (default) |
| `1` | Build C++ wrapped operators |
-->
| 取值 | 行为 |
|------|------|
| _(未设置或 0)_ | 仅安装 Python 包（默认） |
| `1` | 构建 C++ 封装的算子 |

<!--
### 4.3 CMake options {#cmake-options}

When building with C++ extensions (`ENABLE_CPP=1`), the following CMake
options are set automatically by `setup.sh`. For manual builds, you can
pass them via the `CMAKE_ARGS` environment variable.
-->
### 4.3 CMake 选项 {#cmake-options}

使用 C++ 扩展构建（`ENABLE_CPP=1`）时，以下 CMake 选项由 `setup.sh` 自动设置。
如需手动构建，可通过 `CMAKE_ARGS` 环境变量传递。

<!--
| Option | Description | Default |
-->
| 选项 | 描述 | 默认值 |
|--------|------|--------|
| `FLAGGEMS_BUILD_C_EXTENSIONS` | 构建 C++ 扩展 | `OFF` |
| `FLAGGEMS_BACKEND` | 目标后端（`CUDA`、`IX`、`MUSA`、`NPU`、`GCU`） | `CUDA` |
| `FLAGGEMS_BUILD_CTESTS` | 构建 C++ 单元测试 | 取值同 `FLAGGEMS_BUILD_C_EXTENSIONS` |
| `FLAGGEMS_INSTALL` | 安装 CMake 包 | `ON` |
| `FLAGGEMS_USE_EXTERNAL_TRITON_JIT` | 使用外部 Triton JIT 库 | `OFF` |
| `FLAGGEMS_USE_EXTERNAL_PYBIND11` | 使用外部 pybind11 | `ON` |
| `FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP` | 构建 pointwise 动态 C++ 模块 | `OFF` |

<!--
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
-->
### 4.4 `scikit-build-core` 选项 {#scikit-build-core-options}

> [!NOTE]
> 主包 `flag-gems` 使用 `setuptools` 作为构建后端。
> `scikit-build-core` **仅用于**从 `cpp/` 子目录构建的 C++ 扩展。

`scikit-build-core` 是一个构建后端，用来桥接 CMake 和 Python 构建系统，
简化使用 CMake 构建 Python 模块的过程。
常用的配置环境变量包括：

1. `SKBUILD_CMAKE_BUILD_TYPE`：配置项目的构建类型。
   合法取值包括 `Release`、`Debug`、`RelWithDebInfo` 和 `MinSizeRel`。

1. `SKBUILD_BUILD_DIR`：配置项目的构建目录。
   默认值为 `pyproject.toml` 中定义的 `build/{cache_tag}`。

需要注意的是，环境变量 `SKBUILD_CMAKE_ARGS` 中多个选项使用分号（`;`）分隔，
而 `CMAKE_ARGS` 中使用空格分隔。

<!--
### 4.5 The `libtriton_jit` library {#libtriton-jit}

The C++ extension of FlagGems depends on TritonJIT,
a library that implements a Triton JIT runtime in C++ and enables calling
Triton JIT functions from C++ code.

If you are building with an external TritonJIT, build and install it first,
then pass `-DTritonJIT_ROOT=<install path>` to CMake:
-->
### 4.5 关于 `libtriton_jit` 库 {#libtriton-jit}

*FlagGems* 的 C++ 扩展依赖于
[TritonJIT](https://github.com/flagos-ai/libtriton_jit/)，
一个用 C++ 实现 Triton JIT 运行时的库，可以在 C++ 代码中调用 Triton JIT 函数。

如需使用外部的 TritonJIT 库，请先单独构建安装它，
然后通过 `-DTritonJIT_ROOT=<安装路径>` 选项告知 CMake：

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=/usr/local/lib/libtriton_jit" \
ENABLE_CPP=1 ./setup.sh nvidia-cuda128
```
