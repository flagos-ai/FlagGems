---
title: 概要
weight: 10
---

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

目前，CI 测试检查包含三项主要工作。

<!--
## 1. Code Format Check

Using `pre-commit` git hooks with FlagGems, you can format source Python code
and perform basic code pre-checks when calling the `git commit` command.
-->
## 1. 代码格式检查

在 FlagGems 项目中使用 `pre-commit` GIT 回调机制，你可以较容易地完成对 Python
源代码的格式化，并且在执行 `git commit` 命令时自动执行一些基本的代码预检工作。

```shell
pip install pre-commit
pre-commit install
pre-commit
```

<!--
## 2 Operator unit tests

The unit tests check the correctness of operators.
When adding new operators, you need to add unit test cases in the corresponding file
under the `tests` directory.
-->
## 2. 算子单元测试 {#operator-unit-tests}

单元测试的目的是检查算子实现的正确性。
在添加新的算子实现时，你需要在 `tests` 目录下对应的文件中为其添加单元测试。
添加新的测试文件时，

<!--
For operator testing, decorate `@pytest.mark.{OP_NAME}` before the test function
so that we can run the unit test function of the specified OP through `pytest -m`.
A unit test function can be decorated with multiple custom marks.
-->
针对算子的单元测试，需要在测试函数之前使用 `@pytest.mark.{OP_NAME}` 修饰符进行修饰，
这样方便我们使用 `pytest -m` 命令来启动针对特定算子的单元测试。
每个单元测试函数可以使用多个定制的标记（mark）进行修饰。

<!--
If you are adding a C++ wrapped operator, you should add a corresponding *ctest* as well.
See [Add a C++ wrapper](/FlagGems/contribution/cpp-wrapper/) for more details.
-->
当添加新的 C++ 封装的算子时，你需要为算子添加对应的 *ctest*。
参见[添加 C++ 封装的算子](/FlagGems/zh-cn/contribution/cpp-wrapper/)。

<!--
### Model test

Model tests check the correctness of models.
Adding a new model follows a process similar to adding a new operator.
-->
### 模型测试  {#model-test}

模型测试的作用是检查模型的正确性。
添加新模型的过程与添加一个新算子的过程类似。

<!--
### Test Coverage

Python test coverage checks the unit test coverage on an operator.
The `coverage` tool is used when invoking a unit test and the tool
will collect lines covered by unit tests and compute a coverage rate.

Test coverage are summarized during an unit test and the daily full unit test job.
The unit test coverage data are reported on the FlagGems website.
-->
### 测试覆盖率 {#test-coverage}

Python 测试覆盖率检测某个算子的单元测试覆盖率。
我们在执行单元测试时使用 `coverage` 工具来收集单元测试所覆盖的代码行，
工具会自行计算覆盖率数值。

测试覆盖率会在单元测试和每日的全量单元测试任务中进行汇总。
汇总后的单元测试率数据会通过 FlagGems 的项目网站公布。

<!--
## 3. Operator Performance Benchmarking

An *operator benchmark* is used to evaluate the performance of operators.
Currently, the CI pipeline does not check the performance of operators.
This situation is currently being addressed by the project team.
-->
## 3. 算子的性能基准测试 {#operator-performance-benchmarking}

**算子基准测试（Operator Benchmark）** 用来评估算子实现的性能状况。
目前，CI 流水线不会检查算子实现的性能。项目正在努力改变这一状况。

<!--
If you are adding a new operator or optimizing an existing operator,
you need to add performance test cases in the corresponding file
under the `benchmark` directory. For detailed instructions on writing
performance test case, please refer to
[Python performance tests](/FlagGems/performance/python/).
-->
在添加新的算子实现或者优化现有算子时，你需要在 `benchmark/` 目录下
对应的文件中添加性能测试用例。
关于如何编写性能测试用例的详细信息，可参阅
[Python 性能测试](/FlagGems/zh-cn/performance/benchmark/)一节。
