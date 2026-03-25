---
title: 安装 FlagGems
weight: 20
---

<!--
# Installing FlagGems
-->
# 安装 FlagGems

<!--
## 1. Prerequisites

- You have to ensure that the kernel driver and user-space SDK/toolkits for
  your hardware have been installed and configured properly.
  This applies to both the NVIDIA platforms and other AI accelerator hardwares.
-->
## 1. 环境准备

- 你必须确保为自己的硬件正确地安装了内核态的驱动程序和用户空间的 SDK 或工具链，
  并且均已配置正确，工作正常。无论是 NVIDIA 平台还是其他 AI 加速器硬件，
  这一点都适用。

<!--
- You have to ensure that a proper Python version has been installed on your node.
  The currently recommended version is Python 3.10. This may change in the future
  and there could be other version constraints if you are working
  on a [non-NVIDIA platform](/FlagGems/usage/non-nvidia/).
-->
- 你必须确保已经在节点上安装了合适的 Python 版本。目前建议的版本是 Python 3.10。
  在将来这一建议版本可能会不同，并且，如果你在使用
  [非 NVIDIA 硬件平台](/FlagGems/zh-cn/usage/non-nvidia/)，
  厂商可能对可以使用的 Python 版本有额外的约束。

<!--
- You have to install [PyTorch](https://github.com/pytorch/pytorch),
  [Triton](https://github.com/triton-lang/triton) before installing *FlagGems*.

  You may need to install the *custom* PyTorch, Triton or vLLM libraries that
  are tailered for your hardware if you are running *FlagGems* and your workload
  on a [non-NVIDIA platform](/FlagGems/usage/non-nvidia/).
-->
- 你可能需要在安装 *FlagGems* 之前先安装 [PyTorch](https://github.com/pytorch/pytorch)、
  [Triton](https://github.com/triton-lang/triton) 等软件环境。

  如果你在使用[非 NVIDIA 硬件平台](/FlagGems/zh-cn/usage/non-nvidia/)来运行
  *FlagGems* 以及你的工作负载，则你可能需要安装针对这类平台**定制、裁剪过的**
  PyTorch、Triton 软件包。

<!--
- If you are trying out [the integration with vLLM](/FlagGems/usage/frameworks/#vllm),
  you will need to install [vLLM](https://github.com/vllm-project/vllm)
  or its vendor-customized version if any.
-->
- 如果你想尝试将将 *FlagGems* [与 VLLM 集成](/FlagGems/zh-cn/usage/frameworks/#vllm)，
  则需要安装 [vLLM](https://github.com/vllm-project/vllm)，或者厂商定制版本
  （如果有的话）。

<!--
## 2. Install from PyPI

*FlagGems* can be installed from [PyPI](https://pypi.org/project/flag-gems/)
using your favorite Python package manager (e.g. `pip`).
-->
## 2. 从 PyPI 安装

你可以使用自己常用的 Python 包管理器（例如 `pip`）从 [PyPI](https://pypi.org/project/flag-gems/)
安装 *FlagGem* 的软件包：

```shell
pip install flag_gems
```

<!--
> [!INFO]
> **Info**
>
> This Python installation only installs the PyTorch operators implemented
> in Python from *FlagGems*.
> To install the C++-wrapped operators, you will have to
> [build and install from source](#install-from-source).
-->
> [!INFO]
> **提示**
>
> 这种纯 Python 包的安装方式仅安装 *FlagGems* 中用 Python 实现的算子。
> 如果需要安装 C++ 封装的算子，你必须采用
> [从源码构建安装](#install-from-source)方式。

<!--
## 3. Build and install from source {#install-from-source}

*FlagGems* can be built and installed from source just like any other
open source software.
-->
## 3. 从源码构建、安装 {#install-from-source}

与很多其他开源软件类似，*FlagGems* 支持从源码构建、安装。

<!--
### 3.1. Clone the source
-->
### 3.1 克隆源代码

```shell
git clone https://github.com/flagos-ai/FlagGems
cd FlagGems/
```

<!--
### 3.2. Install FlagTree

If you want to use the vanilla Triton compiler instead of *FlagTree*, you can skip this step.

[FlagTree](https://github.com/flagos-ai/flagtree/) is an open source,
unified compiler for multiple AI platforms. Please make sure you have
read the environment requirements from the FlagTree project before
installing it.
-->
### 3.2 安装 FlagTree

如果你希望使用原生的 Triton 编译器而不是 *FlagTree*，可以跳过这一步。

[FlagTree](/https://github.com/flagos-ai/FlagTree) 是一个针对多种
AI 平台的、开源的统一编译器。在安装 FlagTree 之前，请先阅读 FlagTree
项目的运行环境需求。

```shell
pip install -r flag_tree_requirements/requirements_nvidia.txt
```

<!--
> [!TIP]
> **Tips**
>
> - For [non-NVIDIA platforms](/FlagGems/usage/non-nvidia/), you
>   **have to** use different `requirements_<backend>.txt` under
>   the `flag_tree_requirements/` directory.
> - There are on-going efforts to simplify this step. Stay tuned.
-->
> [!TIP]
> **提示**
>
> - 对于[非 NVIDIA 平台](/FlagGems/zh-cn/usage/non-nvidia/)，
>   你必须**使用** `flag_tree_requirements/` 目录下的其他
>   `requirements_<backend>.txt` 文件。
> - 我们正在努力简化这一安装步骤。请持续关注。

<!--
### 3.3. Prepare the build dependencies

FlagGems follows the [PEP 518](https://peps.python.org/pep-0518/) standard
and provides a `pyproject.toml` file to govern the installation process.
-->
### 3.3 准备构建依赖项

*FlagGems* 遵从 [PEP 518](https://peps.python.org/pep-0518/) 标准，
提供 `pyproject.toml` 文件来管理整个安装过程。

<!--
The Python package `flag_gems` uses [`scikit-build-core`](https://scikit-build-core.readthedocs.io/en/latest/)
as its [build backend](https://peps.python.org/pep-0517/#build-backend-interface).

You can run the following command to install/upgrade the build dependencies.
-->
`flag_gems` Python 软件包使用 [`scikit-build-core`](https://scikit-build-core.readthedocs.io/en/latest/)
作为其[构建后端（build backend）](https://peps.python.org/pep-0517/#build-backend-interface)。

你可以运行下面的命令来安装（或升级）构建依赖环境。

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

<!--
You can refer to [scikit-build-core](#scikit-build-core-options) reference
for more details about the commonly used options.
-->
关于 `scikit-build-core` 的一些常用选项细节，可以参阅
[scikit-build-core](#scikit-build-core-options) 参考资料小节。

<!--
### 3.4. Install the package

FlagGems can be installed either a pure Python package or a package with C++ extensions.
The C++ extensions are  still an experimental feature, so please make sure
you have conducted some assessments before using them in production environments.
-->
### 3.4 安装软件包

*FlagGems* 既可以作为纯 Python 软件包来安装，也支持带 C++ 扩展特性的安装。
C++ 扩展特性仍然是一种实验性特性，所以如果你计划在生产环境中使用，
请确保在开始使用之前进行必要的评估测试。

<!--
#### 3.4.1 Install with C++ extension

If you are NOT enabling the C++ wrapped operators, you can skip to the next step.
-->
### 3.4.1 带 C++ 扩展特性的安装

如果你不打算启用 C++ 封装的算子，可以跳过这一步。

<!--
To build and install the C++ extensions in *FlagGems*, the CMake option
`-DFLAGGEMS_BUILD_C_EXTENSION=ON` must be specified during installation.
This can be done by passing arguments to CMake via the `SKBUILD_CMAKE_ARGS` or
the `CMAKE_ARGS` environment variable.
The following command installs the `flag_gems` package in an editable mode,
while enabling the C++ extensions using the `CMAKE_ARGS` environment variable:
-->
如要构建、安装 *FlagGems* 中的 C++ 扩展特性，则必须在安装过程中为 CMake
指定选项 `-DFLAGGEMS_BUILD_C_EXTENSION=ON`。你可以通过设置 `SKBUILD_CMAKE_ARGS`
或者 `CMAKE_ARGS` 环境变量这两种方式来为 CMake 提供参数。
下面的命令以可编辑模式（Editable Mode）安装 `flag_gems` 包，同时使用
`CMAKE_ARGS` 环境变量来启用 C++ 扩展特性：

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON" \
pip install --no-build-isolation -v -e .
```

<!--
Note that the above command installs the
[libtriton_jit library](https://github.com/flagos-ai/libtriton_jit)
by cloning its GIT repository and installing it from source.
-->
注意，上面的命令会安装 [libtriton_jit 库](https://github.com/flagos-ai/libtriton_jit)，
并且安装方式是克隆其 GIT 仓库并从源码来安装。

<!--
For more detailed discussions about the command line options, you can
check the following sections:

- [CMake options](#cmake-options).
- [pip options](#pip-options).
- [build isolation](#build-isolation)
- [installing libtriton_jit](#libtriton-jit)
-->
关于命令行选项的更详细的信息，你可以参考以下小节：

- [CMake 选项](#cmake-options).
- [pip 选项](#pip-options).
- [构建隔离（build isolation）](#build-isolation)
- [安装 libtriton_jit](#libtriton-jit)

<!--
#### 3.4.2 Install the Python package only

You can install *flag_gems* as a pure Python package.
If you are using *FlagGems* as is with no intent to customize it,
you can install the package to your Python environment:
-->
#### 3.4.2 仅安装 Python 软件包

你可以将 `flag_gems` 作为纯 Python 软件包来安装。
如果你希望直接使用 `flag_gems`，无意对其实现作任何修改或定制，
可以将其安装到你的 Python 运行环境中：

```shell
pip install .
```

<!--
This is similar to what `pip install flag_gems` does.
The only difference is that you are instaling the package from its source
rather than a prebuilt Python wheel distribution.
-->
这种安装方式类似于 `pip install flag_gems` 命令所完成的动作。
唯一的区别是在这里你所使用的不是预先构建好的 Python wheel 发行包，
而是从项目的源代码来安装。

<!--
If you are working on the *FlagGems* project, e.g. developing new operators
or performing some similar development/testing works, you can perform an
_editable_ install by specifying `-e` to the command line as shown below:
-->
如果你在参与 *FlagGems* 的开发或测评、优化工作，例如开发新的算子，
或者执行一些类似的开发、测试工作，你可以通过在命令行中指定 `-e`
参数完成**可编辑的**安装部署，如下例所示：

```shell
pip install -e .
```

<!--
Check the [pip options reference](#pip-options) for more information
about some common `pip` options.
-->
关于一些 `pip` 的常用选项，你可以在 [pip 选项参考资料](#pip-options)
一节中阅读一些详细说明。

<!--
## 4. References

### 4.1 Frequently used `pip` options  {#pip-options}
-->
## 4. 参考资料

### 4.1 常用的 `pip` 选项 {#pip-options}

<!--
Some commonly used `pip` options are:

1. `-v`: show the log of the configuration and building process;
-->
一些经常使用的 `pip` 命令行选项包括：

1. `-v`：显示配置与构建过程的详细日志信息。

<!--
1. `-e`: create an editable installation. Note that in an editable installation,
   the C++ section (headers, libraries, cmake package files) is installed
   to the `site-packages` directory, while the Python code remains in
   the current repository with a loader installed in the `site-packages`
   directory to find it.

   For more details about this installation modes, please refer to the
   `scikit-build-core`'s [documentation](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#editable-installs).
-->
2. `-e`：执行**可编辑模式的**安装部署。需要注意的是，在可编辑的部署环境中，
   C++ 部分的代码（头文件、库、cmake 包文件等等）都会被安装到 Python
   （可以是虚拟环境）的 `site-packages` 目录下，而 `flag_gems` 包中的
   Python 代码会继续存放在当前目录下。包安装程序会在 `site-packages`
   目录下安装一个加载器来找到 `flag_gems` 的 Python 代码。

<!--
1. `--no-build-isolation`：Do not to create a separate virtual environment
   (aka. virtualenv or venv for short)  to build the project.
   This is commonly used with an editable installation.
   Note that when building without isolation, you have to install
   the build dependencies manually. Check [build isolation](#build-isolation)
   for more details.
-->
3. `--no-build-isolation`：指示 `pip` 在构建项目时**不要**创建独立的虚拟环境
   （即 virtualenv 或者缩写为 venv）。这一选项通常用于可编辑模式的安装部署。
   需要注意的是，如果在构建时不使用隔离环境，你必须手动完成构造依赖项的安装。
   更多的细节可参阅[构建隔离](#build-isolation)小节。

<!--
1. `--no-deps`: Do not install package dependencies.
   This can be useful when you do not want the dependencies to be updated.
-->
4. `--no-deps`：指示 `pip` 不要安装依赖包。
   如果你不希望 `pip` 更新已经安装部署的依赖包时，这一选项是有用的。

<!--
### 4.2 Build isolation  {#build-isolation}

Following the community recommendations for build frontends in
[PEP 517](https://peps.python.org/pep-0517/#recommendations-for-build-frontends-non-normative),
`pip` or other modern build frontends uses an isolated environment to build packages.
This involves creating a virtual environment and installing the build requirements in it
before building the package.
-->
### 4.2 关于构建隔离 {#build-isolation}

根据社区 [PEP 517](https://peps.python.org/pep-0517/#recommendations-for-build-frontends-non-normative) 所给的倡议，
诸如 `pip` 或者其他一些较新的构建前端（build frontend）一般会使用隔离环境来构建软件包。
构建过程会创建一个**虚拟环境（Virtual Environment）**，并在该环境中安装构建的需求，
之后才会启动软件包的构建过程。

<!--
If you do not want build isolation (often in the case with editable installation),
you can pass `--no-build-isolation` flag to `pip install`.
In this case, the installer will attempt to reuse any existing, compatible
packages when it identifies a dependency to install.
This means you will need
to install `build-requirements` in your current environment beforehand.
Check the `[build-system.requires]` section in the `pyproject.toml` file and
install the required packages.
-->
如果你不希望使用构建隔离（通常是指可编辑模式的安装），可以为 `pip install`
命令指定 `--no-build-isolation` 参数。设置这一参数之后，安装程序会在识别出依赖项后，
尽可能复用系统上已经安装了的、版本兼容的软件包。
这也意味着作为用户的你要在自己的环境中预先安装 `build-requirements` 的内容。
如果不启用构建隔离，你需要查阅 `pyproject.toml` 文件中的 `[build-system.requires]`
小节，手动安装其中列举的软件包。

<!--
### 4.3 About CMake options  {#cmake-options}

As mentioned before, you can enable the C++ extensions when building/installing
`flag_gems` by passing arguments to CMake via the `SKBUILD_CMAKE_ARGS` or
the `CMAKE_ARGS` environment variable.
Note that, for the environment variable `SKBUILD_CMAKE_ARGS`, multiple options
are separated by semicolons (`;`), whereas for `CMAKE_ARGS`, they are separated by spaces.
This relates to the difference between `scikit-build-core` and its predecessor,
`scikit-build`.

The CMake options for configuring `flag_gems` are listed below:
-->
### 4.3 关于 CMake 选项 {#cmake-options}

如前所述，在构建、安装 `flag_gems` 时，你可以通过环境变量 `SKBUILD_CMAKE_ARGS`
或 `CMAKE_ARGS` 向 CMake 传递参数选项。
需要注意的是，对于环境变量 `SKBUILD_CMAKE_ARGS` 而言，如果需要指定多个参数选项，
这些参数要使用分号（`:`）隔开；对于 `CMAKE_ARGS`，多个选项要使用空格来分隔。
造成这一差别的主要原因是 `scikit-build-core` 与其前身 `scikit-build`
之间存在不兼容的变更。

用来配置 `flag_gems` 安装的 CMake 选项如下：

<table>
<thead>
<tr>
  <th><!--Option-->选项</th><th><!--Description-->描述</th><th><!--Default Value-->默认值</th>
<tr>
</thead>
<tbody>
<tr>
  <td><code>FLAGGEMS_USE_EXTERNAL_TRITON_JIT</code></td>
  <td>
    <!--Whether to use external <a href="#libtriton-jit">Triton JIT library</a>.-->
    是否使用外部的 <a href="#libtriton-jit">Triton JIT 库</a>。
  </td>
  <td><code>OFF</code></td>
</tr>
<tr>                                      |
  <td><code>FLAGGEMS_USE_EXTERNAL_PYBIND11</code></td>
  <td>
    <!--Whether to use external `pybind11` library.-->
    是否使用外部的 `pybind11` 库。
  </td>
  <td><code>ON</code></td>
</tr>
<tr>                                      |
  <td><code>FLAGGEMS_BUILD_C_EXTENSIONS</code></td>
  <td>
    <!--Whether to build C++ extension. This is recommended when installed in development mode.-->
    是否构建 C++ 扩展。以开发模式安装时建议启用此选项。
  </td>
  <td><code>ON</code></td>
</tr>
<tr>
  <td><code>FLAGGEMS_BUILD_CTESTS</code></td>
  <td>
    <!--Whether to build C++ unit tests.-->
    是否构建 C++ 扩展的单元测试。
  </td>
  <td><!--same as -->取值为 <code>FLAGGEMS_BUILD_C_EXTENSIONS</code></td>
</tr>
<tr>
  <td><code>FLAGGEMS_INSTALL</code></td>
  <td>
    <!--Whether to install FlagGems's cmake package.
      Recommended for development mode installation.
    -->
    是否安装 FlagGems 的 CMake 包。建议以开发模式安装时启用此选项。
  </td>
  <td>ON</td>
</tr>
</tbody>
</table>

<!--
### 4.4 `scikit-build-core` options {#scikit-build-core-options}

The `scikit-build-core` tool is a build-backend that bridges the CMake
and the Python build system, making it easier to create Python modules with CMake.
Some commonly used environemnt variables for configuring `scikit-build-core` inlcude:
-->
### 4.4 `scikit-build-core` 选项  {#scikit-build-core-options}

工具 `scikit-build-core` 是一个**构建后端（Build Backend）**，用来桥接 CMake
和 Python 这两个构建系统，进而简化使用 CMake 来构建 Python 模块的过程。
用来配置 `scikit-build-core` 的一些常用环境变量包括：

<!--
1. `SKBUILD_CMAKE_BUILD_TYPE`, used to configure the build type of the project.
   Valid values are `Release`, `Debug`, `RelWithDebInfo` and `MinSizeRel`;

1. `SKBUILD_BUILD_DIR`, which configures the build directory of the project.
   The default value is `build/{cache_tag}`, which is defined in `pyproject.toml`.
-->
1. `SKBUILD_CMAKE_BUILD_TYPE`：用来配置项目的构建类型。
   合法的取值包括 `Release`、`Debug`、`RelWithDebInfo` 和 `MinSizeRel` 等。
1. `SKBUILD_BUILD_DIR`：用来设置项目的构建目录。
   默认取值为 `pyproject.toml` 文件中定义的 `build/{cache_tag}`。

<!--
### 4.5 The `libtriton_jit` library  {#libtriton-jit}

The C++ extension of FlagGems depends on [TritonJIT](https://github.com/flagos-ai/libtriton_jit/),
which is a library that implements a Triton JIT runtime in C++
and enables calling Triton JIT functions from C++ code.
If you are building/installing `flag_gems` with an external TritonJIT,
you should build and install it as a precondition and then
pass the option `-DTritonJIT_ROOT=<install path>` to CMake.
-->
### 4.5 关于 `libtriton_jit` 库   {#libtriton-jit}

*FlagGems* 的 C++ 扩展特性依赖于 [TritonJIT](https://github.com/flagos-ai/libtriton_jit/)，
一个用 C++ 来实现 Triton JIT 运行时的库。利用 TritonJIT，我们可以在 C++
代码中调用 Triton 的 JIT 函数。
如果在构建、安装 `flag_gems` 时使用外部的 TritonJIT 库，
就意味着你需要单独构建并安装它，从而满足 `flag_gems` CMake 构建的前提条件；
之后要使用 `-DTritonJIT_ROOT=<安装路径>` 选项将安装位置告知 CMake。

<!--
For example, the following command triggers an editable installation
with external *Triton JIT* installed  at `/usr/local/lib/libtriton_jit`:
-->
例如，下面的命令会启动一个可编辑的、没有构建隔离的安装动作，
并且使用在 `/usr/local/lib/libtriton_jit` 下已经安装好的外部 *TritonJIT* 实例。

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=/usr/local/lib/libtriton_jit" \
pip install --no-build-isolation -v -e .
```
