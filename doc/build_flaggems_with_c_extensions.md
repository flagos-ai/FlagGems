# Build FlagGems with C extensions

## Build System
FlagGems can be installed either as a pure python package or a package with C-extensions. By default, it does not build the C extensions, which is still an experimental feature.

The flag_gems python package leverages the `scikit-build-core` backend to streamline the build and packaging process for C extensions, which are configured using the CMake build system.

## How to build FlagGems with C extensions

To enable C extension building in FlagGems, a CMake options has to be passed to CMake during configuration stage. This can be done by passing arguments to CMake via the `SKBUILD_CMAKE_ARGS` or `CMAKE_ARGS` environment variable.

`SKBUILD_CMAKE_ARGS="-DBUILD_C_EXTENSIONS=ON;-DBUILD_CTESTS=ON" pip install --no-build-isolation -e .`

To make a debug build, add environment `SKBUILD_CMAKE_BUILD_TYPE=Debug`.

Here is the compilation options table

| Option                  | Description                                 | Default |
|-------------------------|---------------------------------------------|---------|
| BUILD_C_EXTENSIONS      | Whether to build C extension                | OFF     |
| USE_EXTERNAL_TRITON_JIT | Whether to use external Triton JIT library  | OFF     |
| USE_EXTERNAL_PYBIND11   | Whether to use external pybind11 library    | ON      |
| BUILD_CTESTS            | Whether build CPP unit tests                | OFF     |
