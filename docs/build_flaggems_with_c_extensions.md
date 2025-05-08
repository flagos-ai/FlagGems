# Build FlagGems with C extensions

## Build System
FlagGems can be installed either as a pure python package or a package with C-extensions. By default, it does not build the C extensions, which is still an experimental feature.

The python package `flag_gems` leverages the `scikit-build-core` backend to streamline the build and packaging process for C extensions, which are configured using the CMake build system.

## How to build and install FlagGems with C extensions

To enable C extension building in FlagGems, a CMake options has to be passed to CMake during configuration stage. This can be done by passing arguments to CMake via the `SKBUILD_CMAKE_ARGS` or `CMAKE_ARGS` environment variable.

For example,

```sh
SKBUILD_CMAKE_ARGS="-DBUILD_C_EXTENSIONS=ON;-DBUILD_CTESTS=ON" pip install --no-build-isolation -e .
```

or

```sh
CMAKE_ARGS="-DBUILD_C_EXTENSIONS=ON -DBUILD_CTESTS=ON" pip install --no-build-isolation -e .
```

Note that for environment variable `SKBUILD_CMAKE_ARGS`, multiple options are separated by semi-colons(`;`), while for `CMAKE_ARGS`, multiple options are separated by spaces.

To build without isolation, you have to install build dependency manually

`pip install -U scikit-build-core ninja cmake pybind11`

To make a debug build, add environment variable `SKBUILD_CMAKE_BUILD_TYPE=Debug`.

Here is the compilation options table

| Option                  | Description                                 | Default |
|-------------------------|---------------------------------------------|---------|
| BUILD_C_EXTENSIONS      | Whether to build C extension                | OFF     |
| USE_EXTERNAL_TRITON_JIT | Whether to use external Triton JIT library  | OFF     |
| USE_EXTERNAL_PYBIND11   | Whether to use external pybind11 library    | ON      |
| BUILD_CTESTS            | Whether build CPP unit tests                | OFF     |

To show verbose log, pass `-v` to `pip install`.

## Editbale Installation

To make an editable installation, pass `-e` to `pip install`. In an editable installation, the C extension is built in a subdirectory in `build` and then installed to `<site_packages_dir>/flag_gems`, while the python package stays in-situ.

If you want more details about this installation mode, please refer to [scikit-build-core's documentation](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#editable-installs).

## How to build a wheel

To build a wheel with `build`.

```sh
pip install -U build
python -m build --no-isolation .
```

It would first make a source distribution(sdist) and build a binary distribution(bdist) from the source distribution. The results are in `dist/`.

Alternatively, you can build a wheel with `pip`.

```sh
pip wheel --no-build-isolation --no-deps -w dist .
```
