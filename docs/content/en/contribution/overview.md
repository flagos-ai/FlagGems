---
title: Overview
weight: 10
---
# Overview

In pull requests, contributor should describe what changed and why.
Please also provide test cases if applicable.
Pull requests require approvals from **one member** before merging.
Additionally, they must pass continuous integration checks.

## 1. Operator inventory

Starting from v4.2, the FlagGems project introduced an operator inventory which can be found
as the `conf/operators.yaml` file. Each operator has a unique ID denoted as the `id` field.
Other fields for an operator include:

- `description`: A brief introduction to what the operator is used for.
- `for`: The target pytorch operation/function to replace, if any.
- `labels`: A list of labels associated with the operator, for the purpose of grouping operators
  along different dimension.
- `kind`: The major kind of the operator.
- `stages`: A list of key-value pairs capturing the history of the operator in question.
  Each stage has a key of `alpha`, `beta`, `stable`, `removed`, with a version value indicating
  the FlagGems release since which the stage is effective. The operator *stage* is an indicator
  of its maturity, defined as follows:

  - A new, hand-written operator usually starts with a `beta` stage.
  - A new, AI generated operator (labelled with `KernelGen`) usually starts with an `alpha` stage.
  - When an operator has been continuously tested without significant issues for a release cycle,
    it may get promoted to the next stage in the followin release. For example, consider an operator
    introduced in version *5.0* as `alpha`, if it works without serious flaws for at least one
    release cycle, it may get promoted to `beta` in the next release, i.e. *5.1*.
  - An existing operator may get demoted from `stable` to `beta` or `alpha` if its starts to
    fail frequently.

All new operators have to be registered into the `conf/operators.yaml` file for maturity
tracking. When deciding the identifier for an operator, please follow the following guidelines:

- For each aten operator registered in `src/flag_gems/__init__.py`, there must be a distinct
  entry in the `conf/operators.yaml` file.
- For each fused operator registered in `src/flag_gems/fused/__init__.py` file, there must
  be a distinct entry in the `conf/operators.yaml` file.
- For a variant of an existing operator, such as an *in-place* operator that has a trailing `_`,
  or a variant that assigns the output to a given `out` parameter, it usually needs a separate entry
  in the `conf/operators.yaml` file. This is because they are supposed to be dispatched in PyTorch
  as a separate operator.
- For other variants such as operators processing tensor or scalar inputs, we use the same
  criteria to determine if it needs a separate entry in the `conf/operators.yaml` file.

## 2. Code Format Check

Using `pre-commit` git hooks with FlagGems, you can format source Python code
and perform basic code pre-checks when calling the `git commit` command.

```shell
pip install pre-commit
pre-commit install
pre-commit
```

## 3. Operator unit tests

The unit tests check the correctness of operators.
When adding new operators, you need to add unit test cases in the corresponding file
under the `tests` directory.

For operator testing, decorate `@pytest.mark.{OP_ID}` before the test function
so that we can selectively run the unit test function for the specified operator
through `pytest -m`.

If you are adding a C++ wrapped operator, you should add a corresponding *ctest* as well.
See [Add a C++ wrapper](/FlagGems/contribution/cpp-wrapper/) for more details.

### Model test

Model tests check the correctness of models.
Adding a new model follows a process similar to adding a new operator.

### Test coverage

Python test coverage checks the unit test coverage on an operator.
The `coverage` tool can be used when invoking a unit test and the tool
will collect lines covered by unit tests and compute a coverage rate.

Test coverage are summarized during an unit test and the daily full unit test job.
The unit test coverage data are reported on the FlagGems website.

## 4. Operator performance benchmarking

An *operator benchmark* is used to evaluate the performance of operators.
If you are adding a new operator or optimizing an existing operator,
you need to add performance test cases in the corresponding file
under the `benchmark` directory.

When new test cases are added to the `benchmark/` subdirectory, or existing
test cases are modified, the CI pipeline can automatically detect these changes
and trigger a benchmark operation.

For detailed instructions on writing performance test case, please refer to
[Python performance tests](/FlagGems/performance/python/).

## 5. About test case marking

The `pytest` tool we used for driving accuracy tests (unit tests) and performance
tests (benchmarks) provides a mechanism to annotate a test case with *custom marks*.
The FlagGems project makes uses of this facility for testing/benchmarking operators
selectively. In the example below, test case is annotated with `@pytest.mark.abs`
to indicate that this test case is for the `abs` operator.

```python
@pytest.mark.abs
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs(shape, dtype):
   inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
   # ...
```

Note that the custom mark (`abs` here) is treated as the identifier (ID) of the operator.
Each unit test and performance benchmark has to be marked with an operator ID.
