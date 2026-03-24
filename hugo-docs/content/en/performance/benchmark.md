---
title: Performance Benchmark
weight: 20
---

# Performance Benchmarking in FlagGems

It is recommended to follow the steps below to add test cases for a new operator.
These steps apply to Python-based operators as well as C++-wrapped operators.

{{% steps %}}

1. **Select the appropriate test file**

   <!--TODO(Qiming): remove the following constraints. -->
   Based on the type of operator, choose the corresponding file in the `benchmark`
   directory:

   - For reduction operators, add the test case to `test_reduction_perf.py`.

   - For tensor constructor operators, add the test case to `test_tensor_constructor_perf.py`.

   - If the operator doesn't fit into an existing category, you can add it to `test_special_perf.py`
     or create a new file for the new operator category.

1. **Check existing benchmark classes**

   Once you've identified the correct file, review the existing classes that inherit
   from the `Benchmark` structure to see if any fit the test scenario for your operator,
   specifically considering:

   - Whether the **metric collection** is suitable.

   - Whether the **input generation function** (`input_generator` or `input_fn`) is appropriate.

1. **Add test cases**

   Depending on the test scenario, follow one of the approaches below to add the test case:

   - **Using existing metric and input generator**

     If the existing metric collection and input generation function meet the requirements of your operator,
     you can add a line of `pytest.mark.parametrize` directly, following the code organization in the file.
     For example, see the operators in `test_binary_pointwise_perf.py`.

   - **Custom input generator**

     If the metric collection is suitable but the input generation function does not meet the operator's requirements,
     you can implement a custom `input_generator`.
     Refer to the `topk_input_fn` function in `test_special_perf.py` as an example of a custom input function
     for the `topk` operator.

   - **Custom metric and input generator**

     If neither the existing metric collection nor the input generation function meets the operator's needs,
     you can create a new class. The new class should define operator-specific metric collection logic
     and a custom input generator. You can refer to various `Benchmark` subclasses across the `benchmark` directory
     for examples.
{{% /steps %}}
