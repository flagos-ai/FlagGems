---
title: Overview
weight: 10
---
# Overview

In pull requests, contributor should describe what changed and why.
Please also provide test cases if applicable.
Pull requests require approvals from **one member** before merging.
Additionally, they must pass continuous integration checks.

Currently, continuous integration checks include three jobs.

## 1. Code Format Check

Using pre-commit git hooks with FlagGems, you can format source Python code
and perform basic code pre-checks when calling the git commit command.

```shell
pip install pre-commit
pre-commit install
pre-commit
```

## 2 Operator unit tests

The unit tests check the correctness of operators.
When adding new operators, you need to add unit test cases in the corresponding file
under the `tests` directory.

For operator testing, decorate `@pytest.mark.{OP_NAME}` before the test function
so that we can run the unit test function of the specified OP through `pytest -m`.
A unit test function can be decorated with multiple custom marks.

If you are adding a C++ wrapped operator, you should add a corresponding *ctest* as well.
See [Add a C++ wrapper](./cpp-wrapper/) for more details.

### Model test

Model tests check the correctness of models.
Adding a new model follows a process similar to adding a new operator.

### Test Coverage

Python test coverage checks the unit test coverage on an operator.
The `coverage` tool is used when invoking a unit test and the tool
will collect lines covered by unit tests and compute a coverage rate.

Test coverage are summarized during an unit test and the daily full unit test job.
The unit test coverage data are reported on the FlagGems website.

## 3. Operator Performance Benchmarking

An *operator benchmark* is used to evaluate the performance of operators.
Currently, the CI pipeline does not check the performance of operators.
This situation is currently being addressed by the project team.

If you are adding a new operator, you need to add corresponding test cases in the corresponding file
under the `benchmark` directory. It is recommended to follow the steps below to add test cases
for a new operator:

{{% steps %}}

1. **Select the appropriate test file**

   Based on the type of operator, choose the corresponding file in the `benchmark` directory:

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
