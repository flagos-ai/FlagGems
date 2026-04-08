---
title: Installation
weight: 20
---
# Installing FlagGems

## 1. Prerequisites

- You have to ensure that the kernel driver and user-space SDK/toolkits for
  your hardware have been installed and configured properly.
  This applies to both the NVIDIA platforms and other AI accelerator hardwares.

- You have to ensure that a proper Python version has been installed on your node.
  The currently recommended version is Python 3.10. This may change in the future
  and there could be other version constraints if you are working
  on a [non-NVIDIA platform](/FlagGems/usage/non-nvidia/).

- You have to install [PyTorch](https://github.com/pytorch/pytorch),
  [Triton](https://github.com/triton-lang/triton) before installing *FlagGems*.

  You may need to install the *custom* PyTorch, Triton or vLLM libraries that
  are tailered for your hardware if you are running *FlagGems* and your workload
  on a [non-NVIDIA platform](/FlagGems/usage/non-nvidia/).

- If you are trying out [the integration with vLLM](/FlagGems/usage/frameworks/#vllm),
  you will need to install [vLLM](https://github.com/vllm-project/vllm)
  or its vendor-customized version if any.

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

*FlagGems* can be built and installed from source just like any other
open source software.

### 3.1. Clone the source

```shell
git clone https://github.com/flagos-ai/FlagGems
cd FlagGems/
```

### 3.2. Install FlagTree

If you want to use the vanilla Triton compiler instead of *FlagTree*, you can skip this step.

[FlagTree](https://github.com/flagos-ai/flagtree/) is an open source,
unified compiler for multiple AI platforms. Please make sure you have
read the environment requirements from the FlagTree project before
installing it.

```shell
pip install -r flag_tree_requirements/requirements_nvidia.txt
```

> [!TIP]
> **Tips**
>
> - For [non-NVIDIA platforms](/FlagGems/usage/non-nvidia/), you
>   **have to** use different `requirements_<backend>.txt` under
>   the `flag_tree_requirements/` directory.
> - There are on-going efforts to simplify this step. Stay tuned.

### 3.3. Prepare the build dependencies

FlagGems follows the [PEP 518](https://peps.python.org/pep-0518/) standard
and provides a `pyproject.toml` file to govern the installation process.

The Python package `flag_gems` uses [`scikit-build-core`](https://scikit-build-core.readthedocs.io/en/latest/)
as its [build backend](https://peps.python.org/pep-0517/#build-backend-interface).

You can run the following command to install/upgrade the build dependencies.

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

You can refer to [scikit-build-core](#scikit-build-core-options) reference
for more details about the commonly used options.

### 3.4. Install the package

FlagGems can be installed either a pure Python package or a package with C++ extensions.
The C++ extensions are  still an experimental feature, so please make sure
you have conducted some assessments before using them in production environments.

#### 3.4.1 Install with C++ extension

If you are NOT enabling the C++ wrapped operators, you can skip to the next step.

To build and install the C++ extensions in *FlagGems*, the CMake option
`-DFLAGGEMS_BUILD_C_EXTENSION=ON` must be specified during installation.
This can be done by passing arguments to CMake via the `SKBUILD_CMAKE_ARGS` or
the `CMAKE_ARGS` environment variable.
The following command installs the `flag_gems` package in an editable mode,
while enabling the C++ extensions using the `CMAKE_ARGS` environment variable:

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON" \
pip install --no-build-isolation -v -e .
```

The above command builds for the default **CUDA** backend. To build for
a different backend or to enable the pointwise dynamic C++ module,
pass the corresponding CMake options. Below are examples for each
supported platform:

**NVIDIA CUDA (with pointwise dynamic C++ support)**

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP=ON" \
pip install --no-build-isolation -v -e .
```

**Iluvatar CoreX (IX)**

```shell
export LIBRARY_PATH=<corex-install-dir>/lib64:$LIBRARY_PATH

CMAKE_ARGS="-DFLAGGEMS_BACKEND=IX -DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BUILD_CTESTS=ON -DCMAKE_BUILD_TYPE=Release" \
pip install --no-build-isolation -v -e .
```

**Moore Threads (MUSA)**

```shell
export MUSA_HOME=<musa-install-dir>

LD_PRELOAD=$CONDA_PREFIX/lib/libittnotify.so \
pip install --no-build-isolation -v -e . \
  --config-settings=cmake.args="-DFLAGGEMS_BACKEND=MUSA;-DFLAGGEMS_BUILD_C_EXTENSIONS=ON;-DFLAGGEMS_BUILD_CTESTS=ON"
```

**Huawei Ascend (NPU)**

```shell
CMAKE_ARGS="-DFLAGGEMS_BACKEND=NPU -DFLAGGEMS_BUILD_C_EXTENSIONS=ON" \
pip install --no-build-isolation -e .
```

Note that the above commands install the
[libtriton_jit library](https://github.com/flagos-ai/libtriton_jit)
by cloning its GIT repository and installing it from source.

For more detailed discussions about the command line options, you can
check the following sections:

- [CMake options](#cmake-options).
- [pip options](#pip-options).
- [build isolation](#build-isolation)
- [installing libtriton_jit](#libtriton-jit)

#### 3.4.2 Install the Python package only

You can install *flag_gems* as a pure Python package.
If you are using *FlagGems* as is with no intent to customize it,
you can install the package to your Python environment:

```shell
pip install .
```

This is similar to what `pip install flag_gems` does.
The only difference is that you are instaling the package from its source
rather than a prebuilt Python wheel distribution.

If you are working on the *FlagGems* project, e.g. developing new operators
or performing some similar development/testing works, you can perform an
_editable_ install by specifying `-e` to the command line as shown below:

```shell
pip install -e .
```

Check the [pip options reference](#pip-options) for more information
about some common `pip` options.

## 4. References

### 4.1 Frequently used `pip` options  {#pip-options}

Some commonly used `pip` options are:

1. `-v`: show the log of the configuration and building process;

1. `-e`: create an editable installation. Note that in an editable installation,
   the C++ section (headers, libraries, cmake package files) is installed
   to the `site-packages` directory, while the Python code remains in
   the current repository with a loader installed in the `site-packages`
   directory to find it.

   For more details about this installation modes, please refer to the
   `scikit-build-core`'s [documentation](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#editable-installs).

1. `--no-build-isolation`：Do not to create a separate virtual environment
   (aka. virtualenv or venv for short)  to build the project.
   This is commonly used with an editable installation.
   Note that when building without isolation, you have to install
   the build dependencies manually. Check [build isolation](#build-isolation)
   for more details.

1. `--no-deps`: Do not install package dependencies.
   This can be useful when you do not want the dependencies to be updated.

### 4.2 Build isolation  {#build-isolation}

Following the community recommendations for build frontends in
[PEP 517](https://peps.python.org/pep-0517/#recommendations-for-build-frontends-non-normative),
`pip` or other modern build frontends uses an isolated environment to build packages.
This involves creating a virtual environment and installing the build requirements in it
before building the package.

If you do not want build isolation (often in the case with editable installation),
you can pass `--no-build-isolation` flag to `pip install`.
In this case, the installer will attempt to reuse any existing, compatible
packages when it identifies a dependency to install.
This means you will need
to install `build-requirements` in your current environment beforehand.
Check the `[build-system.requires]` section in the `pyproject.toml` file and
install the required packages.

### 4.3 About CMake options  {#cmake-options}

As mentioned before, you can enable the C++ extensions when building/installing
`flag_gems` by passing arguments to CMake via the `SKBUILD_CMAKE_ARGS` or
the `CMAKE_ARGS` environment variable.
Note that, for the environment variable `SKBUILD_CMAKE_ARGS`, multiple options
are separated by semicolons (`;`), whereas for `CMAKE_ARGS`, they are separated by spaces.
This relates to the difference between `scikit-build-core` and its predecessor,
`scikit-build`.

The CMake options for configuring `flag_gems` are listed below:

<table>
<thead>
<tr>
  <th>Option</th><th>Description</th><th>Default Value</th>
<tr>
</thead>
<tbody>
<tr>
  <td><code>FLAGGEMS_USE_EXTERNAL_TRITON_JIT</code></td>
  <td>Whether to use external <a href="#libtriton-jit">Triton JIT library</a>.</td>
  <td><code>OFF</code></td>
</tr>
<tr>                                      |
  <td><code>FLAGGEMS_USE_EXTERNAL_PYBIND11</code></td>
  <td>Whether to use external `pybind11` library.</td>
  <td><code>ON</code></td>
</tr>
<tr>                                      |
  <td><code>FLAGGEMS_BUILD_C_EXTENSIONS</code></td>
  <td>Whether to build C++ extension. This is recommended when installed in development mode.</td>
  <td><code>ON</code></td>
</tr>
<tr>
  <td><code>FLAGGEMS_BUILD_CTESTS</code></td>
  <td>Whether to build C++ unit tests.</td>
  <td>same as <code>FLAGGEMS_BUILD_C_EXTENSIONS</code></td>
</tr>
<tr>
  <td><code>FLAGGEMS_INSTALL</code></td>
  <td>Whether to install FlagGems's cmake package.
      Recommended for development mode installation.</td>
  <td>ON</td>
</tr>
<tr>
  <td><code>FLAGGEMS_BACKEND</code></td>
  <td>Target backend for building. Valid values are <code>CUDA</code>,
      <code>IX</code>, <code>MUSA</code>, and <code>NPU</code>.</td>
  <td><code>CUDA</code></td>
</tr>
<tr>
  <td><code>FLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP</code></td>
  <td>Whether to build the pointwise dynamic C++ support module.</td>
  <td><code>OFF</code></td>
</tr>
</tbody>
</table>

### 4.4 `scikit-build-core` options {#scikit-build-core-options}

The `scikit-build-core` tool is a build-backend that bridges the CMake
and the Python build system, making it easier to create Python modules with CMake.
Some commonly used environemnt variables for configuring `scikit-build-core` inlcude:

1. `SKBUILD_CMAKE_BUILD_TYPE`, used to configure the build type of the project.
   Valid values are `Release`, `Debug`, `RelWithDebInfo` and `MinSizeRel`;

1. `SKBUILD_BUILD_DIR`, which configures the build directory of the project.
   The default value is `build/{cache_tag}`, which is defined in `pyproject.toml`.

### 4.5 The `libtriton_jit` library  {#libtriton-jit}

The C++ extension of FlagGems depends on [TritonJIT](https://github.com/flagos-ai/libtriton_jit/),
which is a library that implements a Triton JIT runtime in C++
and enables calling Triton JIT functions from C++ code.
If you are building/inistalling `flag_gems` with an external TritonJIT,
you should build and install it as a precondition and then
pass the option `-DTritonJIT_ROOT=<install path>` to CMake.

For example, the following command triggers an editable installation
with external *Triton JIT* installed  at `/usr/local/lib/libtriton_jit`:

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=/usr/local/lib/libtriton_jit" \
pip install --no-build-isolation -v -e .
```
