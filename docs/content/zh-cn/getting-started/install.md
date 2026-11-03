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

## 1. 前置环境要求

- **底层驱动与硬件工具链**：请确保已在宿主机环境中正确安装了对应加速芯片的内核驱动程序与用户态 SDK / Toolkit（包括 NVIDIA CUDA，以及昇腾 CANN、沐曦 MACA、摩尔线程 MUSA、天数智芯等异构硬件工具链）。
- **上层推理框架依赖**：如需体验 [FlagGems 与 vLLM 的融合加速](../usage/frameworks.md#vllm)，请预先安装原生 vLLM 或硬件厂商定制的 vLLM 发行版。

> [!NOTE]
> 您**无需**提前手动安装 Python、PyTorch 或 Triton。FlagGems 提供的 `setup.sh` 脚本会根据您指定的硬件后端自动配置并补全所有兼容的依赖项。

## 2. 从 PyPI 安装

您可以直接使用 Python 包管理器（如 `pip` 或 `uv pip`）从 [PyPI](https://pypi.org/project/flag-gems/) 安装 *FlagGems*：

```shell
pip install flag_gems
```

<!--
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
>-->
> [!NOTE]
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
> 请将 `cpp-cuda` 替换为您所使用的硬件对应的扩展：
> `cpp-musa`、`cpp-npu`、`cpp-gcu` 或 `cpp-ix`。
>
> 如果您的平台上没有可用的预构建 wheel，请参阅
> [从源码构建安装](#install-from-source)。

<!--
### 2.1. Set up backend dependencies with `flaggems-setup` {#flaggems-setup}

`pip install flag_gems` installs the pure-Python operator library, but it does
**not** pull in PyTorch or the vendor-specific runtime packages for your
accelerator. The `flaggems-setup` console script — installed alongside FlagGems
— completes the environment by installing the PyTorch stack and vendor packages
for a chosen backend from the flagOS PyPI index.

This is the recommended follow-up when you installed FlagGems from PyPI (as
opposed to installing from source with `setup.sh`, which already performs these
steps for you).
-->
### 2.1. 使用 `flaggems-setup` 安装后端依赖 {#flaggems-setup}

`pip install flag_gems` 只安装纯 Python 实现的算子库，**不会**为您的加速器
安装 PyTorch 或厂商专属的运行时依赖包。随 FlagGems 一同安装的 `flaggems-setup`
命令行脚本会从 flagOS PyPI 仓库为您选定的后端安装 PyTorch 及厂商依赖包，
从而补全运行环境。

如果您是从 PyPI 安装 FlagGems（而非用 `setup.sh` 从源码安装，后者已经替您
完成了这些步骤），推荐在安装完成后执行这一步。

<!--
List the available backends:
-->
列出可用的后端：

```shell
flaggems-setup --list
```

<!--
Install the dependencies for your backend (the backend keys are the same ones
`setup.sh` accepts):
-->
为您的后端安装依赖（后端名称与 `setup.sh` 接受的完全一致）：

```shell
# NVIDIA CUDA 12.8
flaggems-setup nvidia-cuda128

# Huawei Ascend CANN 9.0.0
flaggems-setup ascend-cann900

# MetaX MACA 3.8.1
flaggems-setup metax-maca3810
```

<!--
Preview the exact commands without running them:
-->
在不实际执行的情况下预览将要运行的命令：

```shell
flaggems-setup nvidia-cuda128 --dry-run
```

<!--
By default the script uses `uv pip` when `uv` is on your `PATH`, and falls back
to `pip` otherwise. Override the installer explicitly with `--pip`:
-->
默认情况下，当 `uv` 位于 `PATH` 中时脚本会使用 `uv pip`，否则回退到 `pip`。
您也可以用 `--pip` 显式指定安装器：

```shell
flaggems-setup nvidia-cuda128 --pip "pip"
```

<!--
`flaggems-setup` also installs a Triton-family compiler for you. By default it
selects **FlagTree** when the backend provides one, and falls back to **Triton**
otherwise — the same policy as `setup.sh`. Override the choice with the
`--compiler` flag or the `COMPILER` environment variable:

```shell
# Force vanilla Triton instead of the auto-selected FlagTree
flaggems-setup nvidia-cuda128 --compiler triton
COMPILER=triton flaggems-setup nvidia-cuda128
```

See [Environment variables](#env-vars) for the full `COMPILER` behavior.
-->
`flaggems-setup` 还会为您安装 Triton 系编译器。默认情况下，当后端提供
FlagTree 时优先选择 **FlagTree**，否则回退到 **Triton**——策略与 `setup.sh`
一致。您可以用 `--compiler` 标志或 `COMPILER` 环境变量覆盖该选择：

```shell
# 强制使用原生 Triton，而非自动选择的 FlagTree
flaggems-setup nvidia-cuda128 --compiler triton
COMPILER=triton flaggems-setup nvidia-cuda128
```

`COMPILER` 的完整行为参见[环境变量速查](#env-vars)与[构建与配置选项参考](build-options.md#env-vars)。

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
- 安装您的后端所需的正确 Python 版本
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

# MetaX MACA 3.8.1
./setup.sh metax-maca3810
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

如果您在参与 *FlagGems* 的开发（例如开发新的算子），
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
如需手动控制 CMake 选项，请参阅 [CMake 选项参考](build-options.md#cmake-options)。

#### 运行时：通过 `USE_C_EXTENSION` 启用

安装完成后，设置以下环境变量来激活 C++ 路径：

```shell
export USE_C_EXTENSION=1
```

如果不设置该变量，只有 `torch.ops.flag_gems.*` 及 `c_operators` pybind
模块生效；ATen 替换和 `flag_gems.enable()` 的 C++ 分支仍需此变量。
详见 [C++ 使用指南](../usage/cpp.md)。

## 4. 常用环境变量速查 {#env-vars}

在安装、编译构建及运行 *FlagGems* 过程中，以下三个环境变量最为核心：

| 环境变量 | 默认值 | 作用说明 |
| :--- | :--- | :--- |
| `COMPILER` | 自动选择 | 控制 Triton 系编译器的选择：`flagtree`（强制使用 FlagTree）或 `triton`（使用厂商/原生 Triton）。若未显式设置，将根据后端配置优先自动选择 FlagTree。 |
| `ENABLE_CPP` | `0` | 控制是否在运行 `./setup.sh` 脚本时编译构建 C++ 原生扩展。设置为 `1` 时自动编译构建 `cpp/` 扩展包。 |
| `USE_C_EXTENSION` | `0` | 控制运行时是否激活 C++ 算子调度链路。设置为 `1` 时通过 C++ 层分发 ATen 拦截，有效降低 Python 层的调度开销。 |

## 5. 构建与高级配置参数参考 {#build-options-reference}

为保持安装指南的专注度，更详尽的底层编译配置与多硬件支持字典已独立整理至专门的参考文档中：

- **[完整加速器后端清单与映射表](build-options.md#1-支持的加速器后端清单)**：查看所有受支持的芯片平台标识符（如 `nvidia-cuda128`、`ascend-cann900`、`hygon`、`metax-maca3810`、`mthreads-musa520` 等）及依赖说明。
- **[CMake 编译选项参考](build-options.md#cmake-options)** {#cmake-options}：查看 `FLAGGEMS_BUILD_C_EXTENSIONS`、`FLAGGEMS_BACKEND`、`FLAGGEMS_BUILD_CTESTS` 等 CMake 构建选项定义与 `CMAKE_ARGS` 传参范例。
- **[`scikit-build-core` 构建参数](build-options.md#scikit-build-core-options)** {#scikit-build-core-options}：查看用于 C++ 扩展打包的构建后端参数（`SKBUILD_CMAKE_BUILD_TYPE`、`SKBUILD_BUILD_DIR` 等）。
- **[外部 `libtriton_jit` 库集成](build-options.md#libtriton-jit)** {#libtriton-jit}：查看如何自定义编译并链接独立的 TritonJIT 依赖库。
