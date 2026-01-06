# Build FlagGems with C extensions

## Build System
FlagGems can be installed either as a pure python package or a package with C-extensions. By default, it does not build the C extensions, which is still an experimental feature.

The python package `flag_gems` leverages the `scikit-build-core` backend to streamline the building and packaging process for C extensions, which are configured using the CMake build system.

## Install dependencies

To build without isolation, you have to install build dependency manually.

```sh
pip install -U scikit-build-core cmake ninja pybind11
```

## How to build and install FlagGems with C extensions

To enable C extension building in FlagGems, a CMake option `-DFLAGGEMS_BUILD_C_EXTENSION=ON` has to be passed to CMake during configuration stage. This can be done by passing arguments to CMake via the `SKBUILD_CMAKE_ARGS` or `CMAKE_ARGS` environment variable.

Note that for environment variable `SKBUILD_CMAKE_ARGS`, multiple options are separated by semi-colons(`;`), while for `CMAKE_ARGS`, multiple options are separated by spaces. This is related to the the relation of `scikit-build-core` and its predecessor `scikit-build`.

Options to configure FlagGems are listed below:

| Option                           | Description                                 | Default                                 |
|----------------------------------|---------------------------------------------|-----------------------------------------|
| FLAGGEMS_USE_EXTERNAL_TRITON_JIT | Whether to use external Triton JIT library  | OFF                                     |
| FLAGGEMS_USE_EXTERNAL_PYBIND11   | Whether to use external pybind11 library    | ON                                      |
| FLAGGEMS_BUILD_C_EXTENSIONS      | Whether to build C extension                | ON when it is the op level project      |
| FLAGGEMS_BUILD_CTESTS            | Whether build CPP unit tests                | the value of FLAGGEMS_BUILD_C_EXTENSIONS|
| FLAGGEMS_INSTALL                 | Whether to install FlagGems's cmake package | ON when it is the op level project      |

Note that when build with external TritonJIT, you should build and install [libtriton_jit](https://github.com/iclementine/libtorch_example/) beforehand, and pass option `-DTritonJIT_ROOT=<install path>` to cmake.

Other commonly environemnt variable that are used to configure scikit-build-core are:

1. `SKBUILD_CMAKE_BUILD_TYPE`, to configure the build type of the project, valid values are `Release`, `Debug`, `RelWithDebInfo` and `MinSizeRel`;
2. `SKBUILD_BUILD_DIR`, to configure the build dir of the project, the default value is `build/<cache_tag>`, which is defined in `pyproject.toml`.

Commonly used options for pip is

1. `-v`, to show the log of the configuring and building process;
2. `-e`, to make an editable installation. Note that, in editable installation, the C-part is installed to the site-packages dir, while the python part stay insitu and a loader is installed to site-packages dir to find it. If you want more details about this installation mode, please refer to [scikit-build-core's documentation](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#editable-installs).
3. `--no-build-isolation`, not to create a separate virtualenv to build the project. This is commonly used with editable installation. Note that when building without isolation, you have to install build dependency manually.
4. `--no-deps`, do not install package dependencies. This can be useful when you do not want the dependcies to be updated.

### example recipies

Here are recipies to

Editable installation with external TritonJIT

```sh
pip install -U scikit-build-core ninja cmake pybind11

CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DDFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=<install path of triton-jit>" \
pip install --no-build-isolation -v -e .
```

Editable installation with TritonJIT as a sub-project via FetchContent

```sh
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON" \
pip install --no-build-isolation -v -e .
```


## How to build a wheel

To build a wheel with `build`.

```sh
pip install -U build
python -m build --no-isolation --no-deps .
```

It would first make a source distribution(sdist) and build a binary distribution(bdist) from the source distribution. The results are in `dist/`.

Alternatively, you can build a wheel with `pip`.

```sh
pip wheel --no-build-isolation --no-deps -w dist .
```

Environment variables to configure scikit-build-core work in the same way, as described above.
