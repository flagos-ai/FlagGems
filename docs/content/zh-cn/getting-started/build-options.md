---
title: 构建与配置选项参考
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

# 构建与配置选项参考

本文档集中汇总了 *FlagGems* 在异构硬件后端适配、环境配置、编译构建参数以及高阶 C++ 扩展集成方面的完整参考字典。

## 1. 支持的加速器后端清单

*FlagGems* 采用“单一事实来源（Single Source of Truth）”原则，所有支持的硬件后端及其依赖规范均定义在 [`src/flag_gems/backends.yaml`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/backends.yaml) 中。

### 1.1 查询可用后端

在已安装 `flag_gems` 的环境中，您可以通过自带的 `flaggems-setup` 命令行工具实时查询当前支持的全部后端名称：

```shell
flaggems-setup --list
```

### 1.2 主流后端与构建规格

下表汇总了 *FlagGems* 支持的主流硬件芯片平台、推荐 Python 版本以及对应的 C++ 扩展映射（`cmake_backend`）：

| 后端标识符 (`<backend>`) | 加速硬件 / 厂商 | 推荐 Python 版本 | C++ 扩展支持 (`FLAGGEMS_BACKEND`) | 推荐编译器 |
| :--- | :--- | :--- | :--- | :--- |
| `nvidia-cuda128` | NVIDIA GPU (CUDA 12.8) | 3.12 | `CUDA` | FlagTree / Triton |
| `nvidia-cuda133` | NVIDIA GPU (CUDA 13.3) | 3.12 | `CUDA` | FlagTree / Triton |
| `ascend-cann900` | 华为昇腾 NPU (CANN 9.0.0) | 3.11 | `NPU` | FlagTree / Triton |
| `ascend-cann850` | 华为昇腾 NPU (CANN 8.5.0) | 3.11 | `NPU` | FlagTree / Triton |
| `hygon` | 海光 DCU (DTK) | 3.10 | 纯 Python / 扩展适配中 | Triton |
| `metax-maca3810` | 沐曦 GPU (MACA 3.8.1) | 3.12 | 纯 Python | Triton |
| `metax-maca3720` | 沐曦 GPU (MACA 3.7.2) | 3.12 | 纯 Python | Triton |
| `mthreads-musa520` | 摩尔线程 GPU (MUSA 5.2.0) | 3.10 | `MUSA` | FlagTree / Triton |
| `mthreads-musa436` | 摩尔线程 GPU (MUSA 4.3.6) | 3.10 | `MUSA` | FlagTree / Triton |
| `enflame-tops1910` | 燧原 GCU (TopsRider 1.9.1) | 3.12 | `GCU` | Triton |
| `cambricon-neuware472`| 寒武纪 MLU (Neuware 4.7.2)| 3.12 | `MLU` | Triton |
| `cambricon-neuware443`| 寒武纪 MLU (Neuware 4.4.3)| 3.10 | `MLU` | Triton |
| `iluvatar` | 天数智芯 GPU (BI-V) | 3.12 | `IX` | Triton |
| `kunlunxin` | 昆仑芯 XPU | 3.10 | 纯 Python | Triton |
| `biren-br2xx` | 壁仞科技 GPU (BR2xx) | 3.10 | 纯 Python | Triton |

每个后端在 `backends.yaml` 中精确规定了：
- 推荐的 Python 解释器版本；
- PyTorch 核心依赖及各厂商专属的 PyTorch 加速扩展包；
- 适配该硬件的专用 Triton 或 FlagTree 编译器分发包；
- 需要自动写入虚拟环境激活脚本（`activate`）的环境变量及 vendor 初始化脚本。

---

## 2. 环境变量速查 {#env-vars}

在安装、编译构建或运行 *FlagGems* 时，可以通过以下环境变量精确控制行为：

### 2.1 编译器与构建控制

| 环境变量 | 可选值 | 说明 |
| :--- | :--- | :--- |
| `COMPILER` | `flagtree` | 强制使用 FlagTree 编译器。 |
| | `triton` | 强制使用硬件厂商或社区原生的 Triton 编译器。 |
| | *(未设置)* | **默认自动选择**：当该后端配置提供了 FlagTree 时优先选用 FlagTree，否则自动回退到 vendor Triton。 |
| `ENABLE_CPP` | `1` | 在执行 `./setup.sh` 时，自动编译并安装 `cpp/` 目录下的 C++ 原生扩展。 |
| | `0` 或 *(未设置)* | **默认值**：仅安装纯 Python 实现的算子库。 |

### 2.2 运行时控制

| 环境变量 | 可选值 | 说明 |
| :--- | :--- | :--- |
| `USE_C_EXTENSION` | `1` | 激活 C++ 加速调度链路。开启后，在 `flag_gems.enable()` 时会优先通过 C++ 封装层进行 ATen 拦截派发，降低 Python 层的调度与参数解析开销。 |
| | `0` 或 *(未设置)* | **默认值**：使用 Python 层的 ATen 调度分发。 |

### 2.3 多后端设备可见性隔离

在多卡并行或多租户共享计算节点上，请通过对应硬件的环境变量隔离目标计算设备：

| 硬件平台 | 隔离环境变量 | 示例 |
| :--- | :--- | :--- |
| NVIDIA GPU | `CUDA_VISIBLE_DEVICES` | `export CUDA_VISIBLE_DEVICES=0` |
| 华为昇腾 NPU | `ASCEND_RT_VISIBLE_DEVICES` | `export ASCEND_RT_VISIBLE_DEVICES=0` |
| 摩尔线程 MUSA | `MUSA_VISIBLE_DEVICES` | `export MUSA_VISIBLE_DEVICES=0` |
| 燧原 GCU | `TOPS_VISIBLE_DEVICES` | `export TOPS_VISIBLE_DEVICES=0` |
| 寒武纪 MLU | `MLU_VISIBLE_DEVICES` | `export MLU_VISIBLE_DEVICES=0` |

---

## 3. CMake 编译选项参考 {#cmake-options}

当构建 C++ 原生扩展（`ENABLE_CPP=1`）时，`setup.sh` 脚本会根据指定的后端自动推导并设置 CMake 选项。

如果您需要脱离脚本进行**手动构建**，或在打包/定制开发时自定义编译参数，可以通过 `CMAKE_ARGS` 环境变量进行传递：

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=CUDA -DFLAGGEMS_BUILD_CTESTS=ON" \
  uv pip install --no-build-isolation ./cpp
```

### 3.1 核心 CMake 参数表

| 选项名 | 类型 | 默认值 | 功能描述 |
| :--- | :--- | :--- | :--- |
| `FLAGGEMS_BUILD_C_EXTENSIONS` | BOOL | `OFF` | 是否启用并构建 C++ 原生扩展。 |
| `FLAGGEMS_BACKEND` | STRING | `CUDA` | 指定目标硬件后端，可选值包括：`CUDA`、`NPU`、`MUSA`、`GCU`、`MLU`、`IX` 等。 |
| `FLAGGEMS_BUILD_CTESTS` | BOOL | 同 `FLAGGEMS_BUILD_C_EXTENSIONS` | 是否编译底层的 C++ 原生单元测试用例（`ctest`）。 |
| `FLAGGEMS_INSTALL` | BOOL | `ON` | 是否安装 CMake targets 供下游依赖查找。 |
| `FLAGGEMS_USE_EXTERNAL_TRITON_JIT` | BOOL | `OFF` | 是否链接外部预先构建安装好的 `libtriton_jit` 库。 |
| `FLAGGEMS_USE_EXTERNAL_PYBIND11` | BOOL | `ON` | 是否优先使用系统或 Python 环境中已有的 pybind11。 |
| `FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP` | BOOL | `OFF` | 是否编译 pointwise 动态代码生成的 C++ 模块。 |

---

## 4. `scikit-build-core` 构建配置 {#scikit-build-core-options}

> [!NOTE]
> *FlagGems* 的主 Python 包（`flag-gems`）使用标准的 `setuptools` 进行打包；`scikit-build-core` **仅用于**从 `cpp/` 子目录构建 C++ 原生扩展模块（如 `flag-gems-cpp-cuda`）。

`scikit-build-core` 作为连接 CMake 与 Python 构建生态的现代化构建后端，支持通过以下环境变量进行精细化调整：

### 4.1 常用配置变量

1. **`SKBUILD_CMAKE_BUILD_TYPE`**：配置 CMake 工程的编译模式。
   - 可选值：`Release`（默认推荐）、`Debug`、`RelWithDebInfo`、`MinSizeRel`。
2. **`SKBUILD_BUILD_DIR`**：指定 CMake 编译中间产物的存放路径。
   - 默认值：`build/{cache_tag}`（已在 `pyproject.toml` 中配置）。
3. **`SKBUILD_CMAKE_ARGS`**：直接传递给底层 CMake 的选项列表。

> [!WARNING]
> **参数分隔符差异**：
> - 环境变量 `CMAKE_ARGS` 中，多个选项之间使用**空格**分隔（例如 `CMAKE_ARGS="-DFOO=1 -DBAR=2"`）。
> - 环境变量 `SKBUILD_CMAKE_ARGS` 中，多个选项之间必须使用**分号（`;`）**分隔（例如 `SKBUILD_CMAKE_ARGS="-DFOO=1;-DBAR=2"`）。

---

## 5. 外部 `libtriton_jit` 库集成 {#libtriton-jit}

*FlagGems* 的 C++ 封装层依赖于 [libtriton_jit](https://github.com/flagos-ai/libtriton_jit/) 库。该库在 C++ 层面实现了轻量高效的 Triton JIT 运行时，使得原生 C++ 程序能够直接调度与执行 Triton 编写的 GPU/NPU 算子内核。

默认情况下，构建系统会自动解析依赖。如果您希望基于本地修改或特定版本的外部 TritonJIT 进行联合构建，请按如下步骤操作：

1. 单独克隆并构建安装 `libtriton_jit` 至指定目录（例如 `/usr/local/lib/libtriton_jit`）；
2. 在构建 *FlagGems* C++ 扩展时，通过 `-DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON` 开启外部引用，并通过 `-DTritonJIT_ROOT=<安装路径>` 告知 CMake：

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=/usr/local/lib/libtriton_jit" \
ENABLE_CPP=1 ./setup.sh nvidia-cuda128
```
