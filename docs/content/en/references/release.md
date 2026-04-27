---
title: Packaging and Release
weight: 50
---

# Packaging

Creating a source or binary distribution is similar to
[building and installing from source](/FlagGems/getting-started/installation/#install-from-source).
It involves invoking a build-frontend (such as `pip` or `build`) and pass the command
to the build-backend (`scikit-build-core` here).

## 1. Using the `build` build frontend

To build a wheel with the `build` package (recommended).

```shell
pip install -U build
python -m build --no-isolation --no-deps .
```

This will first create a source distribution (sdist) and then build a binary distribution (wheel)
from the source distribution.

If you want to disable the default behavior (source-dir -> sdist -> wheel), You can

- pass `--sdist` to build a source distribution from the source(source-dir -> sdist), or

- pass `--wheel` to build a binary distribution from the source(source-dir -> wheel), or

- pass both `--sdist` and `--wheel` to build both the source and binary distributions
  from the source (source-dir -> sdist, and source-dir -> wheel).

The result is placed in the `.dist/` directory.

## 2. Using the `pip` build frontend

Alternatively, you can build a wheel with `pip`:

```shell
pip wheel --no-build-isolation --no-deps -w dist .
```

The environment variables used to configure `scikit-build-core` work in the same way
as described in the [installation guide](/FlagGems/getting-started/installation/).

After the binary distribution (wheel) is built, you can use `pip` to install it.

```shell
cd FlagGems
python -m build --no-isolation --wheel .
```
