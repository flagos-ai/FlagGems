---
title: 性能基准测试
weight: 20
---
<!--
# Performance Benchmarking in FlagGems

It is recommended to follow the steps below to add test cases for a new operator.
These steps apply to Python-based operators as well as C++-wrapped operators.
-->
# FlagGems 中的性能基准测试

我们建议开发者基于下面的过程来为新的算子添加测试用例。
这些步骤既适用于 Python 实现的算子，也适用于 C++ 封装的算子。

<!--
1. **Select the appropriate test file**
1. **Check existing benchmark classes**
1. **Add test cases**
-->
{{% steps %}}

1. **选择合适的测试用例文件**
   <!--
   Based on the type of operator, choose the corresponding file in the `benchmark`
   directory:

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
