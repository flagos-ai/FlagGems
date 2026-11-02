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

在拉取请求（Pull Request）中，贡献者应就所提议的变更进行描述，包括变更的原因。如有需要，请一并提交单元测试用例。拉取请求被合入之前，需要**一个项目成员**的审批。此外，所有拉取请求都必须通过持续集成（Continuous Integration，CI）检查。

<!--
## 1. Dev Container (Recommended)
If you use VS Code and your code runs inside a container, the recommended way to set up
your development environment is through the provided Dev Container configurations.
See the dedicated [Dev Container](devcontainer.md) page for setup
instructions, including local and SSH remote workflows.
-->

## 1. 开发容器（推荐） {#dev-container}

如果您使用 VS Code 进行开发，且程序运行在容器中，推荐参考项目提供的 Dev Container 配置来搭建开发环境。详细的环境搭建步骤（包括本地和 SSH 远端两种使用场景）请参阅独立页面[开发容器](devcontainer.md)。

<!--
## 2. Device isolation and visibility

On a node shared by several people, you usually want a test or benchmark run to use only
a subset of the available devices. FlagGems does not implement its own device allocation;
it relies on each backend's own visibility environment variable to do the isolation.

A visibility variable must be set **before** the process starts, and the name differs
from one backend to another:

| Backend | Vendor name | Variable                    |
| ------- | ----------- | --------------------------- |
| NVIDIA  | `nvidia`    | `CUDA_VISIBLE_DEVICES`      |
| Ascend  | `ascend`    | `ASCEND_RT_VISIBLE_DEVICES` |
| Enflame | `enflame`   | `TOPS_VISIBLE_DEVICES`      |

For example, to run the test suite on GCUs 4 and 6 only:

```shell
TOPS_VISIBLE_DEVICES=4,6 pytest tests/
```

Note that the runtime renumbers the devices that survive the filter, so in the command
above the two remaining GCUs appear inside the process as `0` and `1`, not as `4` and `6`.

For the common pitfalls around each backend's variable name, the full mapping, and how to
verify that a setting took effect, see
[Selecting Visible Devices](device-visibility.md).
-->

## 2. 设备隔离与可见性 {#device-visibility}

在多人共享的节点上运行单元测试或性能测试时，通常只希望当前进程使用其中的一部分设备。FlagGems 并不自行实现一套设备分配机制，而是依赖各后端自身的可见性环境变量来完成隔离。

可见性变量必须在进程启动**之前**设置，并且各后端的变量名互不相同：

| 后端    | 厂商名      | 环境变量                        |
| ----- | -------- | --------------------------- |
| NVIDIA | `nvidia` | `CUDA_VISIBLE_DEVICES`      |
| Ascend | `ascend` | `ASCEND_RT_VISIBLE_DEVICES` |
| Enflame | `enflame` | `TOPS_VISIBLE_DEVICES`      |

例如，在 Enflame 后端上只使用 4 号和 6 号 GCU 运行测试：

```shell
TOPS_VISIBLE_DEVICES=4,6 pytest tests/
```

需要注意，运行时会为过滤后保留下来的设备重新编号，因此上面这条命令中剩余的两张 GCU 在进程内看到的编号是 `0` 和 `1`，而不是 `4` 和 `6`。

关于各后端变量名的常见陷阱、完整对应关系以及验证设置是否生效的方法，请参阅[设置设备可见性](device-visibility.md)。

## 3. 算子命名规范 {#operator-naming-conventions}

FlagGems 主要提供 PyTorch ATen 算子库的替代实现，其算子注册表为 [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)。FlagGems API 的命名约定如下：

* 与所替代算子的名称尽可能保持一致；
* 采用 snake_case 命名风格；
* 算子名不能以下划线开头；
* 算子名不包含点号。

<!--
## 2. Operator inventory

Starting from v4.2, the FlagGems project introduced an operator inventory which can be found
as the `conf/operators.yaml` file. Each operator has a unique ID denoted as the `id` field.
Other fields for an operator include:
-->

因此，对于一个 ATen 算子，可通过以下步骤得到对应的 FlagGems 算子名称：

* 移除名称中的前缀下划线；
* 将名称中的点号替换为下划线；
* 转为 snake_case 命名风格。

上述转换规则确保了 FlagGems 算子与原始 ATen 算子之间的一一对应关系。

> [!NOTE]
> 若仓库中两个算子仅存在前缀下划线的差异，则保留其前缀下划线。对应算子在测试时的 marker 需额外添加 `underscore` 前缀，以规避 pytest 的限制。

## 4. 算子元信息管理 {#operator-metadata-management}

从 v4.2 版本开始，FlagGems 项目引入了算子目录，即 `conf/operators.yaml` 文件。其中每个算子都有一个由 `id` 字段标识的唯一标识符。各字段说明如下：

| 字段          | 类型     | 用途                                           | 约束                                             |
| ----------- | ------ | -------------------------------------------- | ---------------------------------------------- |
| id          | 字符串    | 唯一标识 FlagGems 仓库中的算子                         | 与 FlagGems 算子 API 名称一致，如 `add`、`add_`；不能以下划线开头 |
| description | 字符串    | 关于算子用途的简要描述                                  | 参考对应外部算子的描述                                    |
| for         | 字符串数组  | 标记该算子所替代的 PyTorch 操作或函数；若无则填 `None`          |                                                |
| labels      | 字符串数组  | 在不同维度对算子进行分组，如标识是否为 KernelGen                |                                                |
| kind        | 字符串    | 算子的主要类别，如 `Math`、`NeuralNetwork`、`LinearAlg` |                                                |
| stages      | 键-值对数组 | 记录算子的演化历史                                    | 主键取值为 `alpha`、`beta`、`stable` 或 `removed`      |
| name        | 字符串    | 各变体对应的抽象算子名称                                 | 与 OpInfo 的 `name` 字段语义一致，采用 snake_case 风格      |
| source      | 字符串    | 算子的来源（算子库或推理/训练框架）                           | 只能取预定义候选列表中的值                                  |

算子成熟度的阶段定义如下：

- 新的手工编写算子通常以 `beta` 阶段起步。
- 新的 AI 生成算子通常以 `alpha` 阶段起步。
- 当某算子经过持续测试、在整个发版周期内未发现重大问题后，可在下一个版本中晋升至更高阶段。例如，某算子在 *5.0* 版本以 `alpha` 阶段引入，若在至少一个发版周期内均无重大缺陷，则可能在 *5.1* 版本晋升为 `beta`。
- 若某个 `stable` 阶段的算子开始频繁出错，也可能被降级为 `beta` 或 `alpha`。

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

## 5. 算子交付件 {#operator-deliverables}

开发新算子前，需要浏览`conf/operators.yaml` 中的`for`字段，避免开发重复算子。新算子需要提交以下内容：

> [!IMPORTANT]
> **关于 vLLM 算子贡献的特别说明**：
> 原 `src/flag_gems/fused` 下的 vLLM 专有算子及相关热补丁已正式迁移至独立仓库 [**FlagGems-vllm**](https://github.com/flagos-ai/FlagGems-vllm)。
> 若您的优化工作或新增代码涉及 vLLM 算子（如 PagedAttention、FlashMLA、FusedMoE 等在 vLLM 中的特化实现），**请直接向 [FlagGems-vllm](https://github.com/flagos-ai/FlagGems-vllm) 仓库提交 Issue 与 PR**。FlagGems 主仓库聚焦于通用的 PyTorch ATen 算子及基础核心组件。

- 在 `conf/operators.yaml` 中填写算子元信息；
- 在 `src/flag_gems/ops`、`src/flag_gems/fused` 或 `src/flag_gems/experimental_ops` 目录下添加 Triton 算子实现；
- 在对应的 `__init__.py` 文件中导出 API；
- 在 `src/flag_gems/__init__.py` 的 `_FULL_CONFIG` 中注册 ATen 算子；如有 backend 特化实现，放到 `src/flag_gems/runtime/backend` 路径下；
- 在 `tests` 目录下添加单元测试；
- 在 `benchmark` 目录下添加性能测试。

> [!IMPORTANT]
> **公版算子定位与兼容性基准要求**：
> `src/flag_gems/ops`、`src/flag_gems/fused` 以及 `src/flag_gems/experimental_ops` 目录下的代码均为 **公版算子实现**（Generic Baseline Implementations）。
> 为确保 FlagGems 能够平滑移植至各异构 AI 芯片并具备最大兼容性，**公版算子实现必须能够基于官方上游 Triton 编译器（Upstream Official Triton）在 NVIDIA Ampere 架构（如 A100）上直接稳定运行**。
> 公版算子中严禁使用仅特定厂商或私有编译器支持的非标语言扩展、私有编译指令或特定芯片内联汇编。若您的优化仅适用于特定硬件架构（如昇腾、海光、沐曦、摩尔线程等），请将其作为厂商特化算子放置于 `src/flag_gems/runtime/backend/_<vendor>/` 目录下进行派发重载，以保持公版实现的纯粹性与高兼容性。

> [!NOTE]
> FlagGems 仓库是一个 Triton 算子库，不接受无直接或间接设备端调用的算子。

## 6. 算子 Host 函数规范 {#operator-host-function-conventions}

算子在 Host 端可以调用 `torch.empty_like()`、`torch.zeros()`、`torch.randn()` 等数据初始化函数，以及 FlagGems 的其他算子。禁止调用 PyTorch 运算相关的算子。

每个算子函数的第一条语句都应输出 DEBUG 日志。日志内容必须包含 vendor 前缀（通用实现使用 `GEMS`，vendor 实现使用 `GEMS_<VENDOR>`），前缀后用空格分隔，并使用大写 snake_case 形式的算子名，例如 `GEMS_NVIDIA SPECIAL_MULTIGAMMALN`。vendor 前缀还可以选择性地附加设备名称，例如 `GEMS_ENFLAME_GCU300 CONV1D`。

## 7. 代码格式检查 {#code-format-check}

在 FlagGems 项目中使用 `pre-commit` GIT 钩子，可以对 Python 源代码进行格式化，并在执行 `git commit` 命令时自动完成基本的代码预检。

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

## 8. 算子单元测试 {#operator-unit-tests}

单元测试用于检验算子实现的正确性。在添加新算子时，需要在 `tests` 目录下的对应文件中添加单元测试用例。

> [!NOTE]
> 测试代码需显式调用 FlagGems API，例如 `flag_gems.log_softmax_out`，禁止使用 `flag_gems.use_gems` 进行隐式调用。禁止根据目录直接调用算子函数，此时会导致后端派发失效，如 `flag_gems.ops.abs_`。

添加新的测试文件时，需在测试函数前使用 `@pytest.mark.{OP_ID}` 进行修饰，以便通过 `pytest -m` 命令有针对性地执行特定算子的单元测试。

> [!NOTE]
> 单元测试的 mark 名需与算子的 API 名保持一致。若 API 名带有下划线前缀，则额外添加 `underscore` 前缀以规避 pytest 的限制。

当添加新的 C++ 封装算子时，需同时添加对应的 *ctest*。详见[添加 C++ 封装的算子](cpp-wrapper.md)。

### 测试算子日志 {#testing-operator-logs}

在算子单元测试中，除了验证计算结果的数值精度（`gems_assert_close` / `gems_assert_equal`）外，还需验证运行时是否确实派发（dispatch）到了 FlagGems 的 Triton 实现，而不是静默回退（silent fallback）到 PyTorch 原生 ATen 算子。

根据 FlagGems 算子规范，每个算子函数的第一条语句都输出带有规范前缀的 DEBUG 日志。单元测试通过 pytest 的 `caplog` fixture 捕获并断言该日志，以验证派发路径与算子存活性（liveness）。

#### 核心辅助工具

测试辅助模块 `tests.accuracy_utils`（通常简写为 `utils`）提供了针对日志验证的专用工具：

1. **`caplog` (pytest fixture)**：用于捕获测试用例执行期间的日志信息。测试用例入参需显式声明 `caplog`（例如 `def test_add(shape, dtype, caplog):`）。如需在同一测试函数中进行多轮验证，可调用 `caplog.clear()` 清空此前捕获的日志。
2. **`utils.gems_log_logger(fn)`**：返回被测算子实现函数所使用的 logger 名称（即 `fn.__module__`）。算子源文件中统一通过 `logger = logging.getLogger(__name__)` 初始化 logger。使用该辅助函数可将 DEBUG 捕获范围严格限定在当前算子所在的模块（例如通用实现 `flag_gems.ops.add`，或厂商特化模块 `_kunlunxin.ops.add`），避免无差别捕获 root logger 或全局其他模块的调试日志造成干扰。
3. **`utils.gems_log_prefix(fn)`**：根据最终解析出的算子实现函数自动推导期望的日志前缀。若运行时命中了厂商特化实现（模块路径中包含 `_{flag_gems.vendor_name}`），则返回 `GEMS_<VENDOR>`（如 `GEMS_NVIDIA`、`GEMS_ASCEND`、`GEMS_HYGON` 等）；若使用的是通用实现，则返回 `GEMS`。这使得测试用例无需修改即可在不同硬件后端上保持通用性。

#### 标准测试代码范例

在编写单元测试时，标准做法是将算子调用放置在 `caplog.at_level` 上下文管理器中，并在调用后断言包含动态前缀与大写算子名称的日志：

```python
# 1. 基础算子测试（如 flag_gems.true_divide）
ref_out = torch.ops.aten.true_divide.Tensor(ref_inp1, ref_inp2)
with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.true_divide)):
    res_out = flag_gems.true_divide(inp1, inp2)

# 验证派发日志：动态前缀 + 大写 snake_case 算子名
assert f"{utils.gems_log_prefix(flag_gems.true_divide)} TRUE_DIVIDE" in caplog.text
# 验证计算精度
utils.gems_assert_close(res_out, ref_out, dtype)
```

针对原地（In-place）算子、指定输出（Out-variant）算子或私有算子，日志名称须与算子函数中的输出严格对应：

```python
# 2. 原地（In-place）算子测试（如 flag_gems.true_divide_）
with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.true_divide_)):
    res_out = flag_gems.true_divide_(inp1, inp2)
assert f"{utils.gems_log_prefix(flag_gems.true_divide_)} TRUE_DIVIDE_" in caplog.text

# 3. 指定输出（Out-variant）算子测试（如 flag_gems.special_i1e_out）
with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.special_i1e_out)):
    res_out = flag_gems.special_i1e_out(inp, out)
assert f"{utils.gems_log_prefix(flag_gems.special_i1e_out)} SPECIAL_I1E_OUT" in caplog.text
```

#### 规范与避坑要点

- **禁止硬编码前缀**：严禁在断言中直接书写 `"GEMS ADD"` 或 `"GEMS_NVIDIA ADD"` 等固定字符串，必须使用 `f"{utils.gems_log_prefix(...)} <OP_NAME>"`，以保证测试用例在全硬件后端（NVIDIA、Ascend、Kunlunxin、Hygon、Cambricon 等）的兼容性。
- **严格限定 Logger 作用域**：切勿直接在不指定 `logger` 的情况下对 root logger 启用 DEBUG（如 `caplog.at_level("DEBUG")`），否则可能捕获到 PyTorch 内部或编译器的海量日志，甚至引起误判。
- **算子名称格式严格匹配**：日志断言中的算子名称须为大写 snake_case 格式，并严格与 Host 函数中 `logger.debug(...)` 输出保持一致（包括下划线后缀，如 `ADD_` 或 `_FLASH_ATTENTION_FORWARD`）。
- **静态 CI 规则检查**：提交 PR 前，可运行 `python tools/ci_checks/check_api_logs.py --operators '["<op_id>"]'` 验证算子源文件中的日志定义是否符合规范。

<!--
### Model test

Model tests check the correctness of models.
Adding a new model follows a process similar to adding a new operator.
-->

### 模型测试 {#model-test}

模型测试用于检验模型的正确性，添加新模型的流程与添加新算子类似。

<!--
### Test Coverage

Python test coverage checks the unit test coverage on an operator.
The `coverage` tool is used when invoking a unit test and the tool
will collect lines covered by unit tests and compute a coverage rate.

Test coverage are summarized during an unit test and the daily full unit test job.
The unit test coverage data are reported on the FlagGems website.
-->

### 测试覆盖率 {#test-coverage}

Python 测试覆盖率用于检测算子单元测试的代码覆盖情况。执行单元测试时，可使用 `coverage` 工具收集被覆盖的代码行并计算覆盖率。

测试覆盖率数据会在单元测试和每日全量单元测试任务中进行汇总，并通过 FlagGems 项目网站公布。

<!--
## 5. Operator Performance Benchmarking

An *operator benchmark* is used to evaluate the performance of operators.
If you are adding a new operator or optimizing an existing operator,
you need to add performance test cases in the corresponding file
under the `benchmark` directory.
-->

## 9. 算子性能基准测试 {#operator-performance-benchmarking}

**算子基准测试（Operator Benchmark）** 用于评估算子实现的性能。在添加新算子或优化现有算子时，需要在 `benchmark/` 目录下的对应文件中添加性能测试用例。

<!--
When new test cases are added to the `benchmark/` subdirectory, or existing
test cases are modified, the CI pipeline can automatically detect these changes
and trigger a benchmark operation.
-->

当有新的测试用例添加到 `benchmark/` 目录，或已有测试用例被修改时，CI 流水线会自动检测到相关变更并触发对应的性能测试。

<!--
For detailed instructions on writing performance test case, please refer to
[Python performance tests](../performance/benchmark.md).
-->

关于如何编写性能测试用例的详细说明，请参阅 [Python 性能测试](../performance/benchmark.md) 一节。

<!--
## 6. About test case marking

The `pytest` tool we used for driving accuracy tests (unit tests) and performance
tests (benchmarks) provides a mechanism to annotate a test case with *custom marks*.
The FlagGems project makes uses of this facility for testing/benchmarking operators
selectively. In the example below, test case is annotated with `@pytest.mark.abs`
to indicate that this test case is for the `abs` operator.
-->

## 10. 关于测例的标记（marks） {#test-case-marking}

精度测试（单元测试）和性能测试（基准测试）均使用 pytest。pytest 提供的定制标记（Custom Marks）机制允许我们为测试用例添加注解。FlagGems 项目利用这一机制来选择性地执行针对特定算子的测试或性能分析。在下面的示例中，`@pytest.mark.abs` 注解表明此测试用例用于测试 `abs` 算子。

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

注意，定制标记（此处为 `abs`）即算子的标识符（ID）。每一个单元测试用例或性能测试用例都必须使用算子 ID 进行标记。
