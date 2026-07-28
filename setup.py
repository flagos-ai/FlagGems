"""Build-time package configuration for flag_gems.

This shim exists only to ship the operator test and benchmark suites inside the
wheel under collision-safe, importable names — without moving them on disk:

  tests/     -> flaggems_tests
  benchmark/ -> flaggems_benchmark

All project metadata (name, version, dependencies, …) lives in ``pyproject.toml``
under ``[project]``; this file intentionally sets *only* ``packages`` and
``package_dir`` (fields that PEP 621 does not own), so there is no metadata
conflict. It is consumed by the setuptools build backend (PEP 517); the legacy
``python setup.py install`` path is never used by pip/uv.

Note: on ``master`` the declared backend is scikit-build-core, which ignores this
file. The released wheel is built by ``.github/workflows/release.yaml`` after it
patches the backend to ``setuptools.build_meta`` — that build honors this shim.
"""

from setuptools import find_packages, setup

setup(
    package_dir={
        "": "src",
        "flaggems_tests": "tests",
        "flaggems_benchmark": "benchmark",
    },
    packages=(
        find_packages("src")
        + ["flaggems_tests", *(f"flaggems_tests.{p}" for p in find_packages("tests"))]
        + [
            "flaggems_benchmark",
            *(f"flaggems_benchmark.{p}" for p in find_packages("benchmark")),
        ]
    ),
)
