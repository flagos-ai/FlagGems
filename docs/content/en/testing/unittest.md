---
title: Testing Python Operators
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


# Testing Python Operators

*FlagGems* uses `pytest` for operator accuracy testing and performance benchmarking.
It  leverages Triton's `triton.testing.do_bench` for kernel-level performance evaluation.

## 1. Accuracy tests for operators

To run unit tests on a specific backend like CUDA:

```shell
pytest tests/test_${name}.py
```

The following command runs the tests on CPU:

```shell
pytest tests/test_foo.py --ref cpu
```

## 2. Accuracy in the context of models

```shell
pytest examples/${name}_test.py
```

## 3. Test operator performance

To test CUDA performance

```shell
pytest benchmark/test_foo.py -s
```

To benchmark the end-to-end performance for operators:

```shell
pytest benchmark/test_foo.py -s --ref cpu
```

## 4. Command-line options supported by the test suite

Besides the built-in `pytest` options (such as `-k` for keyword filtering,
`-m` for marker-based filtering, and `-x` for stopping at the first failure),
`tests/conftest.py` registers a few *FlagGems*-specific options that give you
finer control over how accuracy tests run.

| Option            | Values                    | Description                                                                                  |
| ------------------ | -------------------------- | ---------------------------------------------------------------------------------------------- |
| `--ref`            | `<device>` (default), `cpu` | Device used to compute the reference result the *FlagGems* output is compared against.        |
| `--quick`          | flag                        | Shrink the shape/dtype combinations used by `tests/accuracy_utils.py`, for faster smoke runs.  |
| `--record`         | `none` (default), `log`, `json` | Record test execution details to a log file or a JSON report.                            |
| `--output`         | path                        | Destination file when `--record json` is used. Defaults to `accuracy_result.json`.            |
| `--collect-marks`  | path                        | Instead of running the tests, dump every collected test's markers to the given YAML file.      |

For example, to quickly sanity-check an operator on CPU and keep a JSON report:

```shell
pytest tests/test_softmax.py --ref cpu --quick --record json --output softmax_result.json
```

### Marker-based filtering

Every accuracy test is tagged with a marker named after the operator it
covers, for example `@pytest.mark.fliplr` in `tests/test_fliplr.py`. This lets
you run tests for one or more operators regardless of which file they live
in:

```shell
pytest tests/ -m "fliplr or rms_norm"
```

### Reporting and coverage integration

*FlagGems* CI uses the [`pytest-md-report`](https://pypi.org/project/pytest-md-report/)
plugin to turn a test run into a Markdown summary, and wraps the whole
invocation with [`coverage`](https://coverage.readthedocs.io/) to collect
unit-test coverage:

```shell
coverage run -m pytest -s tests/test_foo.py \
    --md-report --md-report-verbose=1 --md-report-output=summary.md
coverage html
```

See [Unit-test coverage](../coverage) for more details on the coverage report.
