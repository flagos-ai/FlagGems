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

If you use VS Code and your code runs inside a container, the recommended way to set up your development environment is through the provided Dev Container configurations. See the dedicated [Dev Container](devcontainer.md) page for setup instructions, including local and SSH remote workflows.

## 2. Device isolation and visibility {#device-visibility}

On a node shared by several people, you usually want a test or benchmark run to use only a subset of the available devices. FlagGems does not implement its own device allocation; it relies on each backend's own visibility environment variable to do the isolation.

A visibility variable must be set **before** the process starts, and the name differs from one backend to another:

| Backend | Vendor name | Variable                    |
| ------- | ----------- | --------------------------- |
| NVIDIA  | `nvidia`    | `CUDA_VISIBLE_DEVICES`      |
| Ascend  | `ascend`    | `ASCEND_RT_VISIBLE_DEVICES` |
| Enflame | `enflame`   | `TOPS_VISIBLE_DEVICES`      |

For example, to run the test suite on GCUs 4 and 6 only:

```shell
TOPS_VISIBLE_DEVICES=4,6 pytest tests/
```

Note that the runtime renumbers the devices that survive the filter, so in the command above the two remaining GCUs appear inside the process as `0` and `1`, not as `4` and `6`.

For the common pitfalls around each backend's variable name, the full mapping, and how to verify that a setting took effect, see [Selecting Visible Devices](device-visibility.md).

## 3. Operator naming conventions {#operator-naming-conventions}

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

> [!NOTE]
> If two operators in the repository differ only by a leading underscore, preserve the leading underscore. The corresponding operator's test marker should have an additional `underscore` prefix to bypass pytest's limitations.

## 4. Operator metadata management {#operator-metadata-management}

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

## 5. Operator deliverables {#operator-deliverables}

Before developing a new operator, check the `for` field in `conf/operators.yaml` to avoid duplicating an existing operator. New operators must include the following:

> [!IMPORTANT]
> **Notice on vLLM Operator Contributions**:
> The vLLM-specific fused operators formerly residing in `src/flag_gems/fused/` have migrated to the dedicated [**FlagGems-vllm**](https://github.com/flagos-ai/FlagGems-vllm) repository.
> If your work involves optimizing or adding vLLM-specialized kernels (e.g., PagedAttention, FlashMLA, FusedMoE in vLLM), **please submit your PRs and code directly to [FlagGems-vllm](https://github.com/flagos-ai/FlagGems-vllm)**. The FlagGems core repository is dedicated to universal PyTorch ATen operators and standard neural network building blocks.

- Fill in operator metadata in `conf/operators.yaml`;
- Add the Triton operator implementation in `src/flag_gems/ops`, `src/flag_gems/fused`, or `src/flag_gems/experimental_ops`;
- Export the API in the corresponding `__init__.py` file;
- Register the ATen operator in `_FULL_CONFIG` in `src/flag_gems/__init__.py`; if there are backend-specific implementations, place them under `src/flag_gems/runtime/backend`;
- Add unit tests in the `tests` directory;
- Add performance tests in the `benchmark` directory.

> [!IMPORTANT]
> **Generic Baseline Implementations and Compatibility Requirement**:
> All operator implementations located under `src/flag_gems/ops`, `src/flag_gems/fused`, and `src/flag_gems/experimental_ops` are considered **generic baseline operator implementations ("公版算子实现")**.
> To guarantee maximum portability and neutrality across all AI hardware platforms, **generic baseline implementations must be able to run with the official upstream Triton compiler on the NVIDIA Ampere architecture (e.g. A100)**.
> Generic kernels must not introduce vendor-proprietary compiler directives, non-standard language extensions, or chip-specific inline assembly. Hardware-specific optimizations targeting particular architectures (e.g. Ascend, Hygon, MetaX, Moore Threads) must be implemented under `src/flag_gems/runtime/backend/_<vendor>/` as backend overrides instead of altering the generic codebase.

> [!NOTE]
> FlagGems is a Triton operator library. Operators with no direct or indirect device-side calls are not accepted.

## 6. Operator host function conventions {#operator-host-function-conventions}

On the host side, operators may call data initialization functions such as `torch.empty_like()`, `torch.zeros()`, and `torch.randn()`, as well as other FlagGems operators. Calling PyTorch computation operators directly is prohibited.

The first statement in every operator function should emit a DEBUG log. The message must include the vendor prefix (`GEMS` for the generic implementation or `GEMS_<VENDOR>` for a vendor implementation), followed by a space and the operator name in uppercase snake case, for example `GEMS_NVIDIA SPECIAL_MULTIGAMMALN`. The vendor prefix may optionally append a device name, e.g. `GEMS_ENFLAME_GCU300 CONV1D`.

## 7. Code format check {#code-format-check}

Using `pre-commit` git hooks with FlagGems, you can format source Python code and perform basic code pre-checks when calling the `git commit` command.

```shell
pip install pre-commit
pre-commit install
pre-commit
```

## 8. Operator unit tests {#operator-unit-tests}

Unit tests check the correctness of operators. When adding new operators, you need to add unit test cases in the corresponding file under the `tests` directory.

> [!NOTE]
> Test code must explicitly call FlagGems APIs, such as `flag_gems.log_softmax_out`. Implicit calls using `flag_gems.use_gems` are prohibited. Calling operator functions directly by path is also prohibited, as it bypasses backend dispatch (e.g. `flag_gems.ops.abs_`).

When adding new test files, decorate test functions with `@pytest.mark.{OP_ID}` so that we can selectively run unit tests for specific operators using the `pytest -m` command.

> [!NOTE]
> The unit test mark name must match the operator's API name. If the API name has a leading underscore, add an additional `underscore` prefix to bypass pytest's limitations.

If you are adding a C++ wrapped operator, you should add a corresponding *ctest* as well. See [Add a C++ wrapper](cpp-wrapper.md) for more details.

### Testing operator logs {#testing-operator-logs}

In operator unit tests, beyond validating numerical accuracy (`gems_assert_close` / `gems_assert_equal`), you must also verify that runtime execution is dispatched to FlagGems' Triton implementation rather than silently falling back to PyTorch native ATen operators.

According to FlagGems operator conventions, the first statement in every operator function emits a formatted DEBUG log. Unit tests capture and assert this log using pytest's `caplog` fixture to verify the dispatch path and operator liveness.

#### Core testing utilities

The test helper module `tests.accuracy_utils` (conventionally imported as `utils`) provides dedicated helpers for log verification:

1. **`caplog` (pytest fixture)**: Captures log messages during test execution. Test functions must explicitly declare `caplog` as a parameter (e.g. `def test_add(shape, dtype, caplog):`). If performing multiple assertions within the same test, call `caplog.clear()` between them.
2. **`utils.gems_log_logger(fn)`**: Returns the logger name used by the resolved operator implementation (namely `fn.__module__`). Operator source files configure their logger via `logger = logging.getLogger(__name__)`. This helper scopes DEBUG capture strictly to the target operator's module (such as generic `flag_gems.ops.add` or vendor override `_kunlunxin.ops.add`), preventing noise from root loggers or unrelated external libraries.
3. **`utils.gems_log_prefix(fn)`**: Automatically derives the expected log prefix from the resolved function's module path. If a vendor override is executed (its module path contains `_{flag_gems.vendor_name}`), it returns `GEMS_<VENDOR>` (e.g., `GEMS_NVIDIA`, `GEMS_ASCEND`, `GEMS_HYGON`); for generic implementations, it returns `GEMS`. This ensures unit tests run unchanged across diverse hardware backends.

#### Standard test patterns

The standard idiom wraps operator invocation in a `caplog.at_level` context and asserts the expected log message with the dynamic prefix and uppercase operator name:

```python
# 1. Standard operator test (e.g., flag_gems.true_divide)
ref_out = torch.ops.aten.true_divide.Tensor(ref_inp1, ref_inp2)
with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.true_divide)):
    res_out = flag_gems.true_divide(inp1, inp2)

# Verify dispatch log: dynamic prefix + uppercase snake_case operator name
assert f"{utils.gems_log_prefix(flag_gems.true_divide)} TRUE_DIVIDE" in caplog.text
# Verify accuracy
utils.gems_assert_close(res_out, ref_out, dtype)
```

For in-place, out-variant, or private forward operators, ensure the asserted name matches the operator function's log statement:

```python
# 2. In-place operator test (e.g., flag_gems.true_divide_)
with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.true_divide_)):
    res_out = flag_gems.true_divide_(inp1, inp2)
assert f"{utils.gems_log_prefix(flag_gems.true_divide_)} TRUE_DIVIDE_" in caplog.text

# 3. Out-variant operator test (e.g., flag_gems.special_i1e_out)
with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.special_i1e_out)):
    res_out = flag_gems.special_i1e_out(inp, out)
assert f"{utils.gems_log_prefix(flag_gems.special_i1e_out)} SPECIAL_I1E_OUT" in caplog.text
```

#### Best practices and caveats

- **Never hardcode prefixes**: Avoid writing literal strings like `"GEMS ADD"` or `"GEMS_NVIDIA ADD"`. Always use `f"{utils.gems_log_prefix(...)} <OP_NAME>"` so tests run seamlessly on all backends (NVIDIA, Ascend, Kunlunxin, Hygon, Cambricon, etc.).
- **Scope loggers strictly**: Do not enable DEBUG logging globally across the root logger (i.e. avoid `caplog.at_level("DEBUG")` without specifying `logger`), as unrelated debug logs from PyTorch or compilers can introduce flakiness or false positives.
- **Match operator names exactly**: The operator name in the assertion must be in uppercase snake_case and match the string in the host function's `logger.debug(...)` (including trailing or leading underscores, such as `ADD_` or `_FLASH_ATTENTION_FORWARD`).
- **Static CI rule validation**: Before submitting a PR, run `python tools/ci_checks/check_api_logs.py --operators '["<op_id>"]'` to verify that your operator implementation follows the API logging conventions.

### Model test {#model-test}

Model tests check the correctness of models. Adding a new model follows a process similar to adding a new operator.

### Test coverage {#test-coverage}

Python test coverage checks the unit test coverage on an operator. The `coverage` tool can be used when invoking a unit test to collect lines covered by unit tests and compute a coverage rate.

Test coverage is summarized during unit tests and the daily full unit test job. The unit test coverage data are reported on the FlagGems website.

## 9. Operator performance benchmarking {#operator-performance-benchmarking}

An *operator benchmark* is used to evaluate the performance of operators. If you are adding a new operator or optimizing an existing operator, you need to add performance test cases in the corresponding file under the `benchmark` directory.

When new test cases are added to the `benchmark/` subdirectory, or existing test cases are modified, the CI pipeline can automatically detect these changes and trigger a benchmark operation.

For detailed instructions on writing performance test cases, please refer to [Python performance tests](../performance/benchmark.md).

## 10. About test case marking {#test-case-marking}

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
