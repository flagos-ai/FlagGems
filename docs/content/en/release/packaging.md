---
title: Packaging
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


# Packaging

Creating a source or binary distribution is similar to
[building and installing from source](../getting-started/install.md#install-from-source).
It involves invoking a build frontend (such as `pip` or `build`) and passing the command
to the build backend (`setuptools` for the main `flag-gems` package;
`scikit-build-core` for the `cpp/` C++ extension package).

## 1. Using the `build` build frontend

To build a wheel with the `build` package (recommended):

```shell
pip install -U build
python -m build --no-isolation --no-deps .
```

This will first create a source distribution (sdist) and then build a binary distribution (wheel)
from the source distribution.

If you want to customize the build artifacts, you can:

- pass `--sdist` to build only a source distribution (`dist/*.tar.gz`), or
- pass `--wheel` to build only a binary wheel (`dist/*.whl`), or
- pass both `--sdist` and `--wheel` to build both.

The build artifacts are placed in the `dist/` directory.

## 2. Using the `pip` build frontend

Alternatively, you can build a wheel with `pip`:

```shell
pip wheel --no-build-isolation --no-deps -w dist .
```

The environment variables used to configure `setuptools` work in the same way
as described in the [installation guide](../getting-started/install.md).

After the binary distribution (wheel) is built, you can use `pip` to install it.

```shell
pip install dist/flag_gems-*.whl
```

## 3. Building C++ extension wheels

The C++ wrapped operators are packaged as **per-vendor native extension wheels**
built from the `cpp/` subdirectory. Each vendor produces a separate package
(`flag-gems-cpp-cuda`, `flag-gems-cpp-musa`, etc.) that installs its `.so`
files into the `flag_gems/` namespace.

Before building, inject the vendor name into `cpp/pyproject.toml`:

```shell
tools/set_cpp_vendor.sh cuda    # or musa, npu, gcu, ix
```

Then build from the `cpp/` subdirectory. The build requires the vendor's SDK
and toolchain (CMake, a C++ compiler, and PyTorch for that backend):

```shell
cd cpp/
CMAKE_ARGS="-DFLAGGEMS_BACKEND=CUDA" python -m build --no-isolation --wheel .
```

This produces a platform-specific wheel (e.g.
`flag_gems_cpp_cuda-x.y.z-cp312-cp312-linux_x86_64.whl`) in `cpp/dist/`.

The environment variables used to configure `scikit-build-core` (see the
[installation guide](../getting-started/install.md#scikit-build-core-options))
apply when building from `cpp/`.
