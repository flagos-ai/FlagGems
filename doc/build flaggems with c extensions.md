# Build FlagGems with C extensions

## Build System
FlagGems can be installed as a pure python package or a package with C-extensions. By default, it does not build the C extensions, which is still an experimental feature.

The C extensions of flaggems are configured with CMake build system generator. And the python package `flag_gems` is configured with build backend scikit-build-core, which is designed to make the building and packaging easier for python packages with C extension, which is already configured with CMake.

## How to build FlagGems with C extensions

To tell FlagGems to build C extension, an CMake options has to be passed to CMake at configuration stage. To do this, pass arguments to cmake via environment `SKBUILD_CMAKE_ARGS` or `CMAKE_ARGS`.

`SKBUILD_CMAKE_ARGS="-DFLAGGEMS_BUILD_EXTENSIONS=ON pip install --build-isolation -e ."

To make a debug build, add environment `SKBUILD_CMAKE_BUILD_TYPE=Debug`.
