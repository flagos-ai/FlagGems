# Installation

## Install from PyPI

Since FlagGems has not yet been released on PyPI, it can only be installed from source for now.

## Build and install from source

### Clone the source

```shell
git clone https://github.com/FlagOpen/FlagGems
cd FlagGems/
```

### Build system

FlagGems follows [PEP 518](https://peps.python.org/pep-0518/) and contains a `pyproject.toml` file to specify how to build the package.

The Python package `flag_gems` uses `scikit-build-core` as the [build backend](https://peps.python.org/pep-0517/#build-backend-interface). As a brief introduction, [`scikit-build-core`]([scikit-build-core 0.11.5.dev2 documentation](https://scikit-build-core.readthedocs.io/en/latest/)) is a build-backend that provides a bridge between CMake and the Python build system, making it easier to create Python modules with CMake. We use it to avoid wrapping CMake in `setup.py` ourselves.

### Build-isolation

Following the [recommendations for build frontends](https://peps.python.org/pep-0517/#recommendations-for-build-frontends-non-normative) in PEP 517, `pip` or other modern build frontends uses an isolated environment to build packages. This involves creating a virtual environment and installing the build requirements in it before building the package.

If you do not want build isolation (often in the case with editable installation), you can pass `--no-build-isolation` flag to `pip install`, but you will need install build-requirements in your current environment beforehand. Check the `[build-system.requires]` section in pyproject.toml and install the required packages.

Example command:

```shell
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
```

FlagGems can be installed as either a pure python package or a package with C extensions. By default, the C extensions are not built, as this is still an experimental feature.

### Install as a pure Python package

To install FlagGems as a pure Python package, use the commands below.

```shell
# install to site-packages
pip install .

# or editable install
pip install -e .
```

### Install with C extension

To enable C extension building in FlagGems, the CMake option `-DFLAGGEMS_BUILD_C_EXTENSION=ON` must be passed to CMake during the configuration stage. This can be done by passing arguments to CMake via the `SKBUILD_CMAKE_ARGS` or `CMAKE_ARGS` environment variable.

Note that, for the environment variable `SKBUILD_CMAKE_ARGS`, multiple options are separated by semicolons (`;`), whereas for `CMAKE_ARGS`, they are separated by spaces. This relates to the difference between `scikit-build-core` and its predecessor, `scikit-build`.

The options for configuring FlagGems are listed below:

| Option                           | Description                                 | Default                                  |
| -------------------------------- | ------------------------------------------- | ---------------------------------------- |
| FLAGGEMS_USE_EXTERNAL_TRITON_JIT | Whether to use external Triton JIT library  | OFF                                      |
| FLAGGEMS_USE_EXTERNAL_PYBIND11   | Whether to use external pybind11 library    | ON                                       |
| FLAGGEMS_BUILD_C_EXTENSIONS      | Whether to build C extension                | ON when it is the op level project       |
| FLAGGEMS_BUILD_CTESTS            | Whether build CPP unit tests                | the value of FLAGGEMS_BUILD_C_EXTENSIONS |
| FLAGGEMS_INSTALL                 | Whether to install FlagGems's cmake package | ON when it is the op level project       |

The C extension of FlagGems depends on [TritonJIT](https://github.com/iclementine/libtorch_example/), which is a library that implements a Triton JIT runtime in C++ and enables calling Triton jit functions from C++. Note that if you are building FlagGems with an external TritonJIT, you should build and install it beforehand and pass the option `-DTritonJIT_ROOT=<install path>` to CMake.

Other commonly used environemnt variables that configures scikit-build-core are:

1. `SKBUILD_CMAKE_BUILD_TYPE`, which is used to configure the build type of the project. Valid values are `Release`, `Debug`, `RelWithDebInfo` and `MinSizeRel`;
2. `SKBUILD_BUILD_DIR`, which configures the build directory of the project. The default value is `build/<cache_tag>`, which is defined in `pyproject.toml`.

Commonly used pip options are:

1. `-v` to show the log of the configuration and building process;
2. `-e` to create an editable installation. Note that in an editable installation, the C part(headers, libraries, cmake package files) is installed to the site-packages directory, while the Python part stays in situ and a loader is installed in the site-packages directory to find it. For more details about this installation mode, please refer to the [scikit-build-core's documentation](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#editable-installs).
3. `--no-build-isolation`：Do not to create a separate virtualenv to build the project. This is commonly used with an editable installation. Note that when building without isolation, you have to install the build dependencies manually.
4. `--no-deps`： Do not install package dependencies. This can be useful when you do not want the dependencies to be updated.

Example recipes are provided for easy copy-and-paste.

Editable installation with external TritonJIT

```shell
pip install -U scikit-build-core ninja cmake pybind11

CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DDFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=<install path of triton-jit>" \
pip install --no-build-isolation -v -e .
```

Editable installation with TritonJIT as a sub-project via FetchContent

```shell
CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON" \
pip install --no-build-isolation -v -e .
```

## Packaging

Creating a source or binary distribution is similar to building and installing from source. It involves invoking a build-frontend (such as pip and build) and pass the command to the build-backend (scikit-build-core here).

### Use the Build frontend: build

To build a wheel with `build` package (recommended).

```sh
pip install -U build
python -m build --no-isolation --no-deps .
```

This will first create a source distribution (sdist) and then build a binary distribution (wheel) from the source distribution.

If you want to disable the default behavior (source_dir -> sdist -> wheel). You can

- pass `--sdist` to build a source distribution from the source(source_dir -> sdist);

- Or pass `--wheel` to build a binary distribution from the source(source_dir -> wheel).

- Or pass both `--sdist` and `--wheel` to build both the source and binary distributions from the source(source_dir -> sdist, and source_dir->wheel).

The result is in the `.dist/` directory.

### Use Build frontend: pip

Alternatively, you can build a wheel with `pip`.

```sh
pip wheel --no-build-isolation --no-deps -w dist .
```

The environment variables used to configure scikit-build-core work in the same way as described above.

After the binary distribution (wheel) is built, use pip to install it.
