---
title: 概要
weight: 10
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
# Overview

In pull requests, contributor should describe what changed and why.
Please also provide test cases if applicable.
Pull requests require approvals from **one member** before merging.
Additionally, they must pass continuous integration checks.

Currently, continuous integration checks include three jobs.
-->

# 概述

在拉取请求（Pull Request）中，贡献者应该就所提议的变更给出描述，包括变更的原因。
在需要的情况下，请一并提交单元测试用例。
在拉取请求被最终合入之前，需要**一个项目成员**的批准。
此外，这类拉取请求也必须通过持续集成（Continuous Integration，CI）测试。

<!--
## 1. Dev Container (Recommended)
If you use VS Code and your code runs inside a container, the recommended way to set up
your development environment is through the provided Dev Container configurations.
See the dedicated [Dev Container](/FlagGems/contribution/devcontainer/) page for setup
instructions, including local and SSH remote workflows.
-->

## 1. 开发容器（推荐）

如果您使用 VS Code 进行开发，且程序运行在容器中，推荐参考项目提供的 Dev Container 配置来搭建开发环境。
详细的环境搭建步骤（包括本地和 SSH 远端两种使用场景）请参阅独立页面
[开发容器](/FlagGems/zh-cn/contribution/devcontainer/)。

## 2. 算子命名规范

FlagGems 主要提供 PyTorch Aten 算子库的替代实现，后者的注册表为 [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)。FlagGems API 约定为：

* 与所替代算子的名称尽可能一致；
* 采用 snake_case 命名风格；
* 算子名不能以下划线开头；
* 算子名不包含点号。

<!--
## 2. Operator inventory

Starting from v4.2, the FlagGems project introduced an operator inventory which can be found
as the `conf/operators.yaml` file. Each operator has a unique ID denoted as the `id` field.
Other fields for an operator include:
-->

因此，对于一个 Aten 算子，通过以下步骤得到对应的 FlagGems 算子：

* 移除前缀下划线；

* 名称中的点号替换为下划线；

* 转为 snake_case 命名风格。

上述转换能够确保 FlagGems 算子和原始算子是一一对应的。 

> **注意**：如果仓库中两个算子仅存在前缀下划线的差异，则保留算子的前缀下划线。算子测试时的 marker 额外添加 underscore 前缀绕过 pytest 的限制。

## 3. 算子元信息管理

从 v4.2 版本开始，FlagGems 项目开始引入算子目录，即 `conf/operators.yaml` 文件。
其中每个算子都有一个用 `id` 字段来表述的唯一标识符（ID）。Yaml 文件字段说明如下：

| 字段          | 类型     | 用途                                       | 约束                                           |
| ----------- | ------ | ---------------------------------------- | -------------------------------------------- |
| id          | 字符串    | 唯一标识 FlagGems 仓库的算子                      | 与 FlagGems 算子 API 一致，如：`add`、`add_`；不能以下划线开头 |
| description | 字符串    | 关于算子用途的一段简要描述                            | 参考外部原始算子的描述                                  |
| for         | 字符串数组  | 标记算子用来替代的 PyTorch 操作或函数（如果有的话）           | 无填 `None`                                    |
| labels      | 字符串数组  | 在不同维度对算子进行分组。如标识是否为 KernelGen            |                                              |
| kind        | 字符串    | 算子的主要分类类别，如 Math、NeuralNetwork、LinearAlg |                                              |
| stages      | 键-值对数组 | 记述算子的演化历史                                | 主键取值为 alpha、beta、stable 或者 removed。          |
| name        | 字符串    | 各种变体对应的抽象算子                              | 与 OpInfo 的 name 字段语义一致，snake_case 风格         |
| source      | 字符串    | 算子的来源（算子库或者推理/训练框架）                      | 只能取在候选列表中的值                                  |

算子成熟度的指标，具体定义如下：

- 一个新的、手工编写的算子通常以 `beta` 阶段作为起点。

- 一个新的、AI 生成的算子通常以 `alpha` 阶段作为起点。

- 当某个算子被持续测试一段时间，在一整个发版周期内都没有发现重大问题，
  就可能在接下来的发布版本中被提升为新的阶段。例如，假定有一个算子在 *5.0* 版本内以 `alpha` 阶段引入，并且在至少一个发版周期内都没有发现重大缺陷，那么它就可能在下一个发布版本（*5.1*）
  中被提升为 `beta` 阶段算子。

- 已有的算子在开始经常出错时，也可能会被从 `stable` 降格为 `beta` 或 `alpha`。

yaml 文件示例如下：

```yaml
  - id: log_softmax_out
    description: An internal IR for applying a softmax followed by a logarithm.
    for:
      - _log_softmax.out
    labels:
      - aten
      - Reduction
    kind:
      - NeuralNetwork
    stages:
      - alpha: '2.0'
      - stable: '3.0'
    name:
      - log_softmax
    source:
      - aten
```

## 4. 算子交付件

开发一个新算子时，需要提交以下内容：

<!--
- For each aten operator registered in `src/flag_gems/__init__.py`, there must be a distinct
  entry in the `conf/operators.yaml` file.
- For each fused operator registered in `src/flag_gems/fused/__init__.py` file, there must
  be a distinct entry in the `conf/operators.yaml` file.
-->

- `conf/operators.yaml`中填写算子信息；

- 在`src/flag_gems/ops`、`src/flag_gems/fused`或者`src/flag_gems/experimental_ops`下添加 triton 算子实现；

- 在算子文件夹所在的`__init__.py`中导出 API；

- 在 `src/flag_gems/__init__.py` 中的`_FULL_CONFIG`注册 aten 算子。
  如有 backend 特化实现放到`src/flag_gems/runtime/backend`路径下。

- 在`tests`文件夹下添加单元测试；

- 在`benchmark`文件夹下添性能测试。

<!--
## 3. Code Format Check

Using `pre-commit` git hooks with FlagGems, you can format source Python code
and perform basic code pre-checks when calling the `git commit` command.
-->

## 3. 代码格式检查

在 FlagGems 项目中使用 `pre-commit` GIT 回调机制，你可以较容易地完成对 Python
源代码的格式化，并且在执行 `git commit` 命令时自动执行一些基本的代码预检工作。

```shell
pip install pre-commit
pre-commit install
pre-commit
```

<!--
## 4. Operator unit tests

The unit tests check the correctness of operators.
When adding new operators, you need to add unit test cases in the corresponding file
under the `tests` directory.
-->

## 4. 算子单元测试 {#operator-unit-tests}

单元测试的目的是检查算子实现的正确性。在添加新的算子实现时，你需要在 `tests` 目录下对应的文件中为其添加单元测试。

> **注意**：测试代码显式调用 FlagGems API，如`flag_gems.log_softmax_out`。禁止使用 `flag_gems.use_gems`隐式调用。

<!--
### Model test

Model tests check the correctness of models.
Adding a new model follows a process similar to adding a new operator.
-->

添加新的测试文件时，针对算子的单元测试，需要在测试函数之前使用 `@pytest.mark.{OP_ID}` 修饰符进行修饰，这样方便我们使用 `pytest -m` 命令来启动针对特定算子的单元测试。

> **注意**：单元测试 mark 名需要与算子的 api 名相同。如果 api 名带有下划线前缀，则额外添加 underscore 前缀绕过 pytest 的限制。

当添加新的 C++ 封装的算子时，你需要为算子添加对应的 *ctest*。参见[添加 C++ 封装的算子](https://github.com/flagos-ai/FlagGems/blob/gh-pages/FlagGems/zh-cn/contribution/cpp-wrapper)。

### 模型测试  {#model-test}

模型测试的作用是检查模型的正确性。添加新模型的过程与添加一个新算子的过程类似。

<!--
### Test Coverage

Python test coverage checks the unit test coverage on an operator.
The `coverage` tool is used when invoking a unit test and the tool
will collect lines covered by unit tests and compute a coverage rate.

Test coverage are summarized during an unit test and the daily full unit test job.
The unit test coverage data are reported on the FlagGems website.
-->

### 测试覆盖率 {#test-coverage}

Python 测试覆盖率检测某个算子的单元测试覆盖率。在执行单元测试时，可以使用 `coverage` 工具来收集单元测试所覆盖的代码行，工具会自行计算覆盖率数值。

测试覆盖率会在单元测试和每日的全量单元测试任务中进行汇总。汇总后的单元测试率数据会通过 FlagGems 的项目网站公布。

<!--
## 5. Operator Performance Benchmarking

An *operator benchmark* is used to evaluate the performance of operators.
If you are adding a new operator or optimizing an existing operator,
you need to add performance test cases in the corresponding file
under the `benchmark` directory.
-->

## 5. 算子的性能基准测试 {#operator-performance-benchmarking}

**算子基准测试（Operator Benchmark）** 用来评估算子实现的性能状况。在添加新的算子实现或者优化现有算子时，你需要在 `benchmark/` 目录下对应的文件中添加性能测试用例。

<!--
When new test cases are added to the `benchmark/` subdirectory, or existing
test cases are modified, the CI pipeline can automatically detect these changes
and trigger a benchmark operation.
-->

当有新的测试用例被添加到 `benchmark/` 子目录，或者该子目录下现有的测试用例被更改时，CI 流水线会自动检测到这类变更并触发对应的性能测试操作。

<!--
For detailed instructions on writing performance test case, please refer to
[Python performance tests](/FlagGems/performance/python/).
-->

关于如何编写性能测试用例的详细信息，可参阅 [Python 性能测试](/FlagGems/zh-cn/performance/benchmark/)一节。

<!--
## 6. About test case marking

The `pytest` tool we used for driving accuracy tests (unit tests) and performance
tests (benchmarks) provides a mechanism to annotate a test case with *custom marks*.
The FlagGems project makes uses of this facility for testing/benchmarking operators
selectively. In the example below, test case is annotated with `@pytest.mark.abs`
to indicate that this test case is for the `abs` operator.
-->

## 6. 关于测例的标记（marks）

精度测试（单元测试）和性能测试（基准测试）均使用 pytest。pytest 的定制标记（Custom Marks） 机制，允许我们为测试用例添加注解。
FlagGems 项目利用这一设施来选择性地执行针对某个（某些）算子的测试或性能分析。
在下面的例子中，测试用例的注解 `@pytest.mark.abs` 标明此测试用例是用来测试 `abs` 算子的。

```python
@pytest.mark.abs
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs(shape, dtype):
   inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
   # ...
```

<!--
Note that the custom mark (`abs` here) is treated as the identifier (ID) of the operator.
Each unit test and performance benchmark has to be marked with an operator ID.
-->

注意，我们将定制标记（这里的 `abs`）视为算子的标识符（ID）。
每一个单元测试用例或者性能测试用例都必须使用算子的 ID 进行标记。
