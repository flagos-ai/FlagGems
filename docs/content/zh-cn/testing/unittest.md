---
title: 测试 Python 算子
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


<!--
# Testing Python Operators

*FlagGems* uses `pytest` for operator accuracy testing and performance benchmarking.
It  leverages Triton's `triton.testing.do_bench` for kernel-level performance evaluation.
-->
# 测试 Python 算子

*FlagGems* 使用 `pytest` 来驱动算子精度测试和性能基准测试。
项目使用 Triton 的 `triton.testing.do_bench` 来执行内核层级的性能评估。

<!--
## 1. Accuracy tests for operators

To run unit tests on a specific backend like CUDA:
-->
## 1. 算子精度测试

要在特定的后端硬件（如 CUDA）上运行测试：

```shell
pytest tests/test_${name}.py
```

<!--
The following command runs the tests on CPU:
-->
下面的命令执行在 CPU 上的精度测试：

```shell
pytest tests/test_${case}.py --ref cpu
```

<!--
## 2. Accuracy in the context of models
-->
## 2. 在具体模型下执行精度测试

```shell
pytest examples/${name}_test.py
```

<!--
## 3. Test operator performance

To test operator performance on CUDA:
-->
## 3. 测试算子的性能

在 CUDA 平台上测试算子的性能：

```shell
pytest benchmark/test_foo.py -s
```

<!--
To benchmark the end-to-end performance for operators:
-->
下面的命令对算子执行端到端的性能基准测试：

```shell
pytest benchmark/test_foo.py -s --ref cpu
```

<!--
## 4. Command-line options supported by the test suite

Besides the built-in `pytest` options (such as `-k` for keyword filtering,
`-m` for marker-based filtering, and `-x` for stopping at the first failure),
`tests/conftest.py` registers a few *FlagGems*-specific options that give you
finer control over how accuracy tests run.
-->
## 4. 测试套件支持的命令行选项

除了 `pytest` 自带的选项（例如用于关键字过滤的 `-k`、用于按标记过滤的 `-m`，
以及在首次失败时终止的 `-x`）之外，`tests/conftest.py` 还注册了若干
*FlagGems* 特有的选项，可以更精细地控制精度测试的运行方式。

<!--
| Option            | Values                    | Description                                                                                  |
| ------------------ | -------------------------- | ---------------------------------------------------------------------------------------------- |
| `--ref`            | `<device>` (default), `cpu` | Device used to compute the reference result the *FlagGems* output is compared against.        |
| `--quick`          | flag                        | Shrink the shape/dtype combinations used by `tests/accuracy_utils.py`, for faster smoke runs.  |
| `--record`         | `none` (default), `log`, `json` | Record test execution details to a log file or a JSON report.                            |
| `--output`         | path                        | Destination file when `--record json` is used. Defaults to `accuracy_result.json`.            |
| `--collect-marks`  | path                        | Instead of running the tests, dump every collected test's markers to the given YAML file.      |
-->

| 选项              | 取值                        | 描述                                                                       |
| ----------------- | --------------------------- | -------------------------------------------------------------------------- |
| `--ref`           | `<device>`（默认）、`cpu`   | 用于计算参照结果的设备，*FlagGems* 的输出将与其进行比对。                  |
| `--quick`         | 开关                        | 精简 `tests/accuracy_utils.py` 中使用的形状/数据类型组合，以加快冒烟测试。 |
| `--record`        | `none`（默认）、`log`、`json` | 将测试执行的详细信息记录到日志文件或 JSON 报告中。                        |
| `--output`        | 路径                        | 使用 `--record json` 时的输出文件路径，默认为 `accuracy_result.json`。     |
| `--collect-marks` | 路径                        | 不实际运行测试，而是将所有被收集到的测试用例的标记信息导出到指定的 YAML 文件。 |

<!--
For example, to quickly sanity-check an operator on CPU and keep a JSON report:
-->
例如，下面的命令会在 CPU 上快速检查某个算子，并保留一份 JSON 报告：

```shell
pytest tests/test_softmax.py --ref cpu --quick --record json --output softmax_result.json
```

<!--
### Marker-based filtering

Every accuracy test is tagged with a marker named after the operator it
covers, for example `@pytest.mark.fliplr` in `tests/test_fliplr.py`. This lets
you run tests for one or more operators regardless of which file they live
in:
-->
### 基于标记的过滤

每一个精度测试都会带有一个以其所覆盖算子命名的标记，例如
`tests/test_fliplr.py` 中的 `@pytest.mark.fliplr`。借助这一机制，
你可以按算子名称运行测试，而不必关心测试用例分布在哪些文件中：

```shell
pytest tests/ -m "fliplr or rms_norm"
```

<!--
### Reporting and coverage integration

*FlagGems* CI uses the [`pytest-md-report`](https://pypi.org/project/pytest-md-report/)
plugin to turn a test run into a Markdown summary, and wraps the whole
invocation with [`coverage`](https://coverage.readthedocs.io/) to collect
unit-test coverage:
-->
### 报告与覆盖率集成

*FlagGems* 的 CI 流程使用 [`pytest-md-report`](https://pypi.org/project/pytest-md-report/)
插件，将一次测试运行转换为 Markdown 格式的摘要报告；同时使用
[`coverage`](https://coverage.readthedocs.io/) 包裹整个测试过程以收集单元测试覆盖率：

```shell
coverage run -m pytest -s tests/test_foo.py \
    --md-report --md-report-verbose=1 --md-report-output=summary.md
coverage html
```

<!--
See [Unit-test coverage](../coverage) for more details on the coverage report.
-->
关于覆盖率报告的更多细节，请参阅[单元测试覆盖率](../coverage)。
