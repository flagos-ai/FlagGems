---
title: 概要
weight: 10
---

<!--
# Code Contribution

In pull requests, contributor should describe what changed and why.
Please also provide test cases if applicable.
Pull requests require approvals from **one member** before merging.
Additionally, they must pass continuous integration checks.

Currently, continuous integration checks include four pipelines.
-->
# 概述

在拉取请求（Pull Request）中，贡献者应该就所提议的变更给出描述，包括变更的原因。
在需要的情况下，请一并提交单元测试用例。
在拉取请求被最终合入之前，需要**一个项目成员**的批准。
此外，这类拉取请求也必须通过持续集成（Continuous Integration，CI）测试。

目前，CI 测试检查包含四项主要工作。

<!--
## 1. Code Format Check

Using pre-commit git hooks with FlagGems, you can format source Python code
and perform basic code pre-checks when calling the git commit command.
-->
## 1. 代码格式检查

在 FlagGems 项目中使用 pre-commit GIT 回调机制，你可以较容易地完成对 Python 源代码的格式化，
并且在执行 `git commit` 命令时自动执行一些基本的代码预检工作。

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
See [Add a C++ wrapper](./cpp-wrapper/) for more details.
-->
当添加新的 C++ 封装的算子时，你需要为算子添加对应的 *ctest*。
参见[添加 C++ 封装的算子](./cpp-wrapper/)。

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
If you are adding a new operator, you need to add corresponding test cases in the corresponding file
under the `benchmark` directory. It is recommended to follow the steps below to add test cases
for a new operator:
-->
在添加新的算子实现时，你需要在 `benchmark/` 目录下对应的文件中添加测试用例。
建议用户参照如下步骤来为新的算子添加测试用例。

<!--
1. **Select the appropriate test file**
1. **Check existing benchmark classes**
1. **Add test cases**
-->
{{% steps %}}

1. **选择合适的测试用例文件**
   <!--
   Based on the type of operator, choose the corresponding file in the `benchmark` directory:

   - For reduction operators, add the test case to `test_reduction_perf.py`.
   - For tensor constructor operators, add the test case to `test_tensor_constructor_perf.py`.
   - If the operator doesn't fit into an existing category, you can add it to `test_special_perf.py`
     or create a new file for the new operator category.
   -->

   基于要测试的算子类型，在 `benchmark/` 目录下选择对应的文件：

   - 对于 *reduction（规约）* 算子，可以将测试用例添加到 `test_reduction_perf.py`。
   - 对于 Tensor（张量）构造算子，可以将测试用例添加到 `test_tensor_constructor_perf.py` 文件中。
   - 如果算子无法归类到以上类别，可以将测试用例添加到 `test_special_perf.py` 中，
     或者为新的算子类型添加一个新文件。

2. **查阅现有的基准测试类**

   <!--
      Once you've identified the correct file, review the existing classes that inherit
      from the `Benchmark` structure to see if any fit the test scenario for your operator,
      specifically considering:

      - Whether the **metric collection** is suitable.
      - Whether the **input generation function** (`input_generator` or `input_fn`) is appropriate.
   -->

   一旦你确定了合适的测试文件，可以先查阅现有的、从 `Benchmark` 结构派生而来的测试类，
   了解现有的测试类是否能够满足你的算子的测试需要。主要考察点包括：

   - 是否其中实现的指标搜集（metric collection）动作符合需要；
   - 是否其中的输入生成函数（`input_generator` 或 `input_fn`）的实现满足需要。


3. **添加测试用例**

   <!--
   Depending on the test scenario, follow one of the approaches below to add the test case:

   - **Using existing metric and input generator**

      If the existing metric collection and input generation function meet the requirements of your operator,
      you can add a line of `pytest.mark.parametrize` directly, following the code organization in the file.
      For example, see the operators in `test_binary_pointwise_perf.py`.
   -->
   取决于具体的测试场景，按以下方法之一来添加测试用例：

   - **使用现有的指标和输入生成逻辑**

     如果现有的指标采集和输入生成函数满足你的算子的需求，你可以直接添加一行
     `pytest.mark.parametrize`，保持文件中的代码组织不变。
     你可以在 `test_binary_pointwise_perf.py` 文件中查阅此类实现的例子。

   <!--
   - **Custom input generator**

     If the metric collection is suitable but the input generation function does not meet the operator's requirements,
     you can implement a custom `input_generator`.
     Refer to the `topk_input_fn` function in `test_special_perf.py` as an example of a custom input function
     for the `topk` operator.
   -->
   - **定制输入生成机制**

     如果指标采集逻辑合适，但输入生成函数不满足算子的需求，你可以实现一个定制的 `input_generator`。
     你可以参照 `test_special_perf.py` 文件中的 `topk_input_fn` 实现，
     了解如何为 `topk` 算子添加一个自定义的输入参数生成函数。

   <!--
   - **Custom metric and input generator**

     If neither the existing metric collection nor the input generation function meets the operator's needs,
     you can create a new class. The new class should define operator-specific metric collection logic
     and a custom input generator. You can refer to various `Benchmark` subclasses across the `benchmark` directory
     for examples.
   -->
   - **自定义指标采集和输入生成函数**

     如果现有的指标采集机制和输入生成函数都无法满足算子的需求，
     你可以创建一个新的测试类。新的测试类要为算子定义特定的指标采集逻辑，
     以及一个自定义的输入参数生成函数。
     你可以参照 `benchmark/` 目录下不同的 `Benchmark` 子类，了解这类定制的机制。

{{% /steps %}}
