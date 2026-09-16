---
title: Overview
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

# Overview

In pull requests, contributors should describe what changed and why. Please also provide test cases if applicable. Pull requests require approvals from **one member** before merging. Additionally, they must pass continuous integration checks.

## 1. Dev Container (Recommended) {#dev-container}

If you use VS Code and your code runs inside a container, the recommended way to set up your development environment is through the provided Dev Container configurations. See the dedicated [Dev Container](/FlagGems/contribution/devcontainer/) page for setup instructions, including local and SSH remote workflows.

## 2. Operator naming conventions {#operator-naming-conventions}

FlagGems primarily provides alternative implementations of the PyTorch ATen operator library, with its operator registry being [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml). The FlagGems API naming conventions are:

* Keep names as consistent as possible with the operators being replaced;
* Use snake_case naming style;
* Operator names cannot start with an underscore;
* Operator names do not contain dots.

Therefore, for an ATen operator, the corresponding FlagGems operator name can be derived through the following steps:

* Remove leading underscores from the name;
* Replace dots in the name with underscores;
* Convert to snake_case naming style.

These conversion rules ensure a one-to-one correspondence between FlagGems operators and original ATen operators.

> **Note**: If two operators in the repository differ only by a leading underscore, preserve the leading underscore. The corresponding operator's test marker should have an additional `underscore` prefix to bypass pytest's limitations.

## 3. Operator metadata management {#operator-metadata-management}

Starting from v4.2, the FlagGems project introduced an operator inventory which can be found as the `conf/operators.yaml` file. Each operator has a unique ID denoted as the `id` field. The fields are described as follows:

| Field       | Type            | Purpose                                                                 | Constraints                                                           |
| ----------- | --------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------- |
| id          | string          | Uniquely identifies an operator in the FlagGems repository              | Matches FlagGems operator API name, e.g., `add`, `add_`; cannot start with underscore |
| description | string          | Brief description of the operator's purpose                             | Refer to the description of the corresponding external operator       |
| for         | array of string | Marks the PyTorch operation/function this operator replaces; use `None` if not applicable |                                                                       |
| labels      | array of string | Groups operators along different dimensions, e.g., whether it's KernelGen |                                                                       |
| kind        | string          | Major category of the operator, such as `Math`, `NeuralNetwork`, `LinearAlg` |                                                                       |
| stages      | array of key-value pairs | Records the evolution history of the operator                  | Keys can be `alpha`, `beta`, `stable`, or `removed`                   |
| name        | string          | Abstract operator name corresponding to variants                        | Consistent with OpInfo's `name` field semantics, using snake_case style |
| source      | string          | Source of the operator (operator library or inference/training framework) | Must be a value from the predefined candidate list                    |

Operator maturity stages are defined as follows:

- A new, hand-written operator typically starts at the `beta` stage.
- A new, AI-generated operator typically starts at the `alpha` stage.
- When an operator has been continuously tested without significant issues for an entire release cycle, it may be promoted to the next stage in the following release. For example, if an operator is introduced at the `alpha` stage in version *5.0* and has no major defects for at least one release cycle, it may be promoted to `beta` in version *5.1*.
- An existing operator at the `stable` stage may be demoted to `beta` or `alpha` if it starts to fail frequently.

Example yaml file entry:

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

## 4. Operator deliverables {#operator-deliverables}

Before developing a new operator, check the `for` field in `conf/operators.yaml` to avoid duplicating an existing operator. New operators must include the following:

- Fill in operator metadata in `conf/operators.yaml`;
- Add the Triton operator implementation in `src/flag_gems/ops`, `src/flag_gems/fused`, or `src/flag_gems/experimental_ops`;
- Export the API in the corresponding `__init__.py` file;
- Register the ATen operator in `_FULL_CONFIG` in `src/flag_gems/__init__.py`; if there are backend-specific implementations, place them under `src/flag_gems/runtime/backend`;
- Add unit tests in the `tests` directory;
- Add performance tests in the `benchmark` directory.

> **Note**: FlagGems is a Triton operator library. Operators with no direct or indirect device-side calls are not accepted.

## 5. Operator host function conventions {#operator-host-function-conventions}

On the host side, operators may call data initialization functions such as `torch.empty_like()`, `torch.zeros()`, and `torch.randn()`, as well as other FlagGems operators. Calling PyTorch computation operators directly is prohibited.

## 6. Code format check {#code-format-check}

Using `pre-commit` git hooks with FlagGems, you can format source Python code and perform basic code pre-checks when calling the `git commit` command.

```shell
pip install pre-commit
pre-commit install
pre-commit
```

## 7. Operator unit tests {#operator-unit-tests}

Unit tests check the correctness of operators. When adding new operators, you need to add unit test cases in the corresponding file under the `tests` directory.

> **Note**: Test code must explicitly call FlagGems APIs, such as `flag_gems.log_softmax_out`. Implicit calls using `flag_gems.use_gems` are prohibited. Calling operator functions directly by path is also prohibited, as it bypasses backend dispatch (e.g. `flag_gems.ops.abs_`).

When adding new test files, decorate test functions with `@pytest.mark.{OP_ID}` so that we can selectively run unit tests for specific operators using the `pytest -m` command.

> **Note**: The unit test mark name must match the operator's API name. If the API name has a leading underscore, add an additional `underscore` prefix to bypass pytest's limitations.

If you are adding a C++ wrapped operator, you should add a corresponding *ctest* as well. See [Add a C++ wrapper](https://github.com/flagos-ai/FlagGems/blob/gh-pages/FlagGems/contribution/cpp-wrapper) for more details.

### Model test {#model-test}

Model tests check the correctness of models. Adding a new model follows a process similar to adding a new operator.

### Test coverage {#test-coverage}

Python test coverage checks the unit test coverage on an operator. The `coverage` tool can be used when invoking a unit test to collect lines covered by unit tests and compute a coverage rate.

Test coverage is summarized during unit tests and the daily full unit test job. The unit test coverage data are reported on the FlagGems website.

## 8. Operator performance benchmarking {#operator-performance-benchmarking}

An *operator benchmark* is used to evaluate the performance of operators. If you are adding a new operator or optimizing an existing operator, you need to add performance test cases in the corresponding file under the `benchmark` directory.

When new test cases are added to the `benchmark/` subdirectory, or existing test cases are modified, the CI pipeline can automatically detect these changes and trigger a benchmark operation.

For detailed instructions on writing performance test cases, please refer to [Python performance tests](/FlagGems/performance/python/).

## 9. About test case marking {#test-case-marking}

The `pytest` tool we use for driving accuracy tests (unit tests) and performance tests (benchmarks) provides a mechanism to annotate test cases with *custom marks*. The FlagGems project uses this facility for testing/benchmarking operators selectively. In the example below, the test case is annotated with `@pytest.mark.abs` to indicate that this test case is for the `abs` operator.

```python
@pytest.mark.abs
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_abs(shape, dtype):
   inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
   # ...
```

Note that the custom mark (`abs` here) is treated as the identifier (ID) of the operator. Each unit test and performance benchmark must be marked with an operator ID.
