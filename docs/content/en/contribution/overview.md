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

Using `pre-commit` git hooks with FlagGems, you can format source Python code
and perform basic code pre-checks when calling the `git commit` command.

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
See [Add a C++ wrapper](/FlagGems/contribution/cpp-wrapper/) for more details.

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

If you are adding a new operator or optimizing an existing operator,
you need to add performance test cases in the corresponding file
under the `benchmark` directory. For detailed instructions on writing
performance test case, please refer to
[Python performance tests](/FlagGems/performance/python/).
