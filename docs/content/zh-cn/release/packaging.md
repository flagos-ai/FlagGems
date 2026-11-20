---
title: 打包与分发
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

# 源码与二进制包打包构建

创建 *FlagGems* 的源码包（sdist）或二进制包（wheel）的过程与[从源码构建安装](../getting-started/install.md#install-from-source)类似。打包流程通过构建前端（如 `build` 或 `pip`）调用对应的构建后端执行（主包 `flag-gems` 使用 `setuptools`；`cpp/` 目录下的 C++ 扩展包使用 `scikit-build-core`）。

## 1. 使用 `build` 工具打包 Python 主包（推荐）

通过官方标准的 `build` 模块构建 wheel 包：

```shell
pip install -U build
python -m build --no-isolation --no-deps .
```

默认情况下，该命令会依次生成源码分发包（sdist）与二进制 wheel 包。您可以按需指定参数过滤输出：

- `--sdist`：仅生成源码包（`dist/*.tar.gz`）；
- `--wheel`：仅生成二进制 Wheel 包（`dist/*.whl`）。

生成的产物将保存在项目根目录的 `dist/` 文件夹中。

## 2. 使用 `pip` 构建 Wheel

也可直接利用 `pip wheel` 进行编译打包：

```shell
pip wheel --no-build-isolation --no-deps -w dist .
```

生成完毕后，可直接在环境中安装：

```shell
pip install dist/flag_gems-*.whl
```

## 3. 构建 C++ 扩展 Wheel 包

C++ 封装算子作为**面向特定硬件厂商的原生扩展 Wheel 包**独立提供（如 `flag-gems-cpp-cuda`、`flag-gems-cpp-musa` 等），其编译产物 `.so` 将被安装到 `flag_gems/` 命名空间中。

构建前，需先在 `cpp/pyproject.toml` 中声明目标硬件厂商标识：

```shell
# 可选值：cuda、musa、npu、gcu、ix 等
tools/set_cpp_vendor.sh cuda
```

随后进入 `cpp/` 子目录，结合目标硬件平台的编译器与 SDK 完成构建：

```shell
cd cpp/
CMAKE_ARGS="-DFLAGGEMS_BACKEND=CUDA" python -m build --no-isolation --wheel .
```

构建生成的特定平台 Wheel（如 `flag_gems_cpp_cuda-x.y.z-cp312-cp312-linux_x86_64.whl`）将位于 `cpp/dist/` 目录下。相关 CMake 构建配置选项请参阅[构建与配置选项参考](../getting-started/build-options.md#cmake-options)。
