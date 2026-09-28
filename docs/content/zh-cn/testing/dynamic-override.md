---
title: 使用动态算子重载进行测试
weight: 30
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
# Testing with Dynamic Operator Override
-->
# 使用动态算子重载进行测试

如果尚未生成候选，只需要验证 benchmark 的原始 baseline 能否运行，使用独立的 [reference-only 模式](reference-only.md)；它与 override 互斥，不执行 correctness reference，也不替代本章的候选正确性验证。

<!--
Most accuracy tests in `tests/` call *FlagGems* operators directly, for
example `flag_gems.softmax(x)` or `flag_gems._list_to_tensor(...)`. When you
are iterating on a new kernel implementation, or comparing several candidate
implementations side by side, it is often inconvenient to edit the operator
source file under `src/flag_gems/ops/` for every variant you want to try.
-->
`tests/` 目录下的大多数精度测试都会直接调用 *FlagGems* 的算子 API，
例如 `flag_gems.softmax(x)` 或 `flag_gems._list_to_tensor(...)`。
当你正在迭代开发一个新的内核实现，或者想要并排比较若干候选实现时，
每试一个新版本就去修改 `src/flag_gems/ops/` 下的算子源码文件，往往并不方便。

<!--
The `flag_gems.dynamic_registry` and `flag_gems.cli_override` modules solve
this by letting a test process swap out the implementation bound to a
`flag_gems.<op_name>` attribute at runtime, without touching the operator's
source code. This makes it possible to:
-->
`flag_gems.dynamic_registry` 和 `flag_gems.cli_override` 两个模块为此提供了解决方案：
它们允许测试进程在运行时替换绑定到 `flag_gems.<算子名>` 属性上的实现，
而不需要改动算子的源码。这使得下面几件事情成为可能：

<!--
- Point a test run at an implementation living in an arbitrary `.py` file.
- Drive the override from the command line or from a YAML/JSON config file,
  so no test code needs to change between runs.
- Run several such test processes concurrently, each overriding the same
  operator with a different implementation, without interfering with each
  other.
-->
- 让一次测试运行指向任意 `.py` 文件中的实现。
- 通过命令行或 YAML/JSON 配置文件来驱动重载，使得测试代码在不同运行之间无需改动。
- 并发运行多个这样的测试进程，各自用不同的实现重载同一个算子，彼此互不干扰。

<!--
## 1. Overriding an operator from Python
-->
## 1. 在 Python 中重载算子

<!--
`DynamicOpOverride` tracks the original implementation of every operator it
touches, so it can restore it later. The recommended usage is as a context
manager, which restores all overrides automatically on exit — including when
the test body raises:
-->
`DynamicOpOverride` 会追踪它所接触过的每一个算子的原始实现，以便之后恢复。
推荐将其作为上下文管理器使用，这样在退出时会自动恢复所有重载——即使测试主体抛出了异常也是如此：

```python
import torch
import flag_gems
from flag_gems.dynamic_registry import DynamicOpOverride

def my_softmax(input, dim=-1, dtype=None):
    return torch.softmax(input, dim=dim, dtype=dtype)

with DynamicOpOverride() as registry:
    registry.override("softmax", my_softmax)
    x = torch.randn(10, 10, device=flag_gems.device)
    result = flag_gems.softmax(x)  # 使用 my_softmax
# 到这里，原始的 flag_gems.softmax 已经被恢复
```

<!--
Without the context manager, call `restore()` or `restore_all()` explicitly:
-->
如果不使用上下文管理器，则需要显式调用 `restore()` 或 `restore_all()`：

```python
registry = DynamicOpOverride()
registry.override("softmax", my_softmax)
...
registry.restore("softmax")     # 恢复单个算子
registry.restore_all()          # 或者一次性恢复所有算子
```

尚未提供公开 FlagGems 实现的算子也可以注入。例如，调用
`registry.override("sparse_csr_tensor", candidate)` 后，即使此前不存在该属性，
`flag_gems.sparse_csr_tensor(...)` 也会调用候选实现。恢复时会删除临时新增的属性；
原本存在的属性则恢复原值，包括 `None`。`--override` 和 `--override-config`
同样支持这种用法。

新增公开 callable 不会自动添加 ATen dispatcher 注册。这类算子的测试需要直接调用
`flag_gems.<op_name>(...)`。通过文件加载的候选若一次也没有被调用，清理时仍会报错，
包括注入时拼错算子名的情况。

<!--
## 2. Loading an implementation from a standalone file
-->
## 2. 从独立文件中加载实现

<!--
Rather than defining the replacement inline, `override_from_file` loads a
function from any `.py` file on disk — the file does not need to live inside
the *FlagGems* package or be importable as a module:
-->
除了直接在代码中定义替代实现之外，`override_from_file` 还可以从磁盘上任意的
`.py` 文件中加载函数——该文件不需要位于 *FlagGems* 包内，也不需要能作为模块被导入：

```python
registry.override_from_file(
    op_name="softmax",
    filepath="./candidates/softmax_v2.py",
    func_name="my_softmax",  # 省略时默认与 op_name 相同
)
```

<!--
Multiple operators can be swapped in one call with
`override_batch_from_files`:
-->
使用 `override_batch_from_files` 可以在一次调用中替换多个算子：

```python
registry.override_batch_from_files({
    "softmax": ("./candidates/softmax_v2.py", "my_softmax"),
    "rms_norm": "./candidates/rms_norm_v2.py",  # 函数名与算子名相同
})
```

<!--
## 3. Driving overrides from the command line
-->
## 3. 通过命令行驱动重载

<!--
`flag_gems.cli_override` exposes `add_override_arguments()` and
`apply_overrides_from_args()`, which add a consistent `--override` /
`--override-config` interface to any `argparse`-based script or `pytest`
`conftest.py`.
-->
`flag_gems.cli_override` 提供了 `add_override_arguments()` 和
`apply_overrides_from_args()` 两个函数，可以为任意基于 `argparse` 的脚本，
或者 `pytest` 的 `conftest.py`，添加统一的 `--override` / `--override-config` 接口。

<!--
### `--override op_name:filepath[:func_name]`
-->
### `--override op_name:filepath[:func_name]`

<!--
Pass one or more `--override` flags, each pointing at the implementation file
for one operator:
-->
可以传入一个或多个 `--override` 参数，每一个指向一个算子的实现文件：

```shell
python my_test.py \
    --override softmax:./candidates/softmax_v2.py:my_softmax \
    --override rms_norm:./candidates/rms_norm_v2.py
```

<!--
The `op_name=filepath[:func_name]` form is also accepted, if you prefer `=`
over `:` as the top-level separator.
-->
如果你更喜欢用 `=` 而不是 `:` 作为顶层分隔符，`op_name=filepath[:func_name]`
这种写法也同样被支持。

<!--
### `--override-config path/to/overrides.yaml`
-->
### `--override-config path/to/overrides.yaml`

<!--
For a larger set of overrides, collect them in a YAML (or JSON) file instead:
-->
如果需要重载的算子较多，可以将它们统一整理到一个 YAML（或 JSON）文件中：

```yaml
overrides:
  softmax:
    file: ./candidates/softmax_v2.py
    function: my_softmax
  rms_norm: ./candidates/rms_norm_v2.py       # 函数名与算子名相同
  layer_norm: ./candidates/layer_norm_v2.py:my_layer_norm
```

```shell
python my_test.py --override-config ./overrides.yaml
```

<!--
`--override` and `--override-config` can be combined; entries from
`--override` are applied after the config file, so they take precedence for
any operator listed in both places.
-->
`--override` 和 `--override-config` 可以组合使用；`--override`
中的条目会在配置文件之后被应用，因此对于两处都列出的算子，
命令行参数的设置具有更高的优先级。

<!--
## 4. Integrating with `pytest`
-->
## 4. 与 `pytest` 集成

<!--
The `--override` and `--override-config` options are already integrated into
both `tests/conftest.py` and `benchmark/conftest.py`, so any test file under
`tests/` or `benchmark/` can be pointed at a custom implementation without
code changes.
-->
`--override` 和 `--override-config` 选项已经集成到 `tests/conftest.py` 和
`benchmark/conftest.py` 中，因此 `tests/` 或 `benchmark/` 目录下的任何测试文件
都可以在不修改代码的前提下指向自定义实现。

<!--
Running the existing accuracy test for `softmax` against a candidate
implementation requires no change to `test_softmax.py` itself:
-->
要针对候选实现运行现有的 `softmax` 精度测试，`test_softmax.py` 本身完全不需要任何改动：

```shell
pytest tests/test_softmax.py \
    --override softmax:./candidates/softmax_v2.py:my_softmax
```

<!--
Similarly, benchmark tests can use the same options:
-->
类似地，性能基准测试也可以使用相同的选项：

```shell
pytest benchmark/test_reduction_perf.py \
    --override sum:./candidates/sum_v2.py:my_sum \
    --level core -s
```

<!--
The override is applied once per test session in `pytest_configure`, and
automatically restored in `pytest_unconfigure` after all tests complete.
-->
重载会在 `pytest_configure` 中对每个测试会话应用一次，
并在所有测试完成后在 `pytest_unconfigure` 中自动恢复。

<!--
## 5. Overrides are picked up when operators are (re-)registered
-->
## 5. 算子被（重新）注册时会自动感知到重载

<!--
`GeneralOpRegistrar`, the class that binds each `flag_gems.<op_name>`
implementation to its ATen dispatch key, re-resolves every config entry
against the live `flag_gems` module before registering it. This means that
if you apply an override *before* `flag_gems.enable()` / `only_enable()`
runs (or before an operator is re-registered for any other reason), the
dispatch table picks up your override instead of the reference captured in
the module's internal config tuple at import time:
-->
`GeneralOpRegistrar` 是负责将每个 `flag_gems.<算子名>` 实现绑定到对应
ATen 调度键的类，它在注册每一项配置之前，都会重新在当前的 `flag_gems`
模块上解析一次对应的实现。这意味着，如果你在 `flag_gems.enable()` /
`only_enable()` 运行之前（或者算子因为其他原因被重新注册之前）就应用了重载，
调度表会使用你的重载实现，而不是模块内部配置元组在导入时捕获的那个旧引用：

```python
with DynamicOpOverride() as registry:
    registry.override("softmax", my_softmax)
    flag_gems.enable()  # 注册 `_softmax`（及其重载）时会使用 my_softmax
```

<!--
To keep this resolution unambiguous, the registrar looks each dispatch key
up in `flag_gems._FULL_CONFIG` — the authoritative source of registrable
ops — to find the function the key was *originally* bound to, then resolves
the live attribute by that function's name. This avoids collisions between
overloads that share a dispatch-key prefix but bind to different functions
(e.g. `_softmax` vs. `_softmax.out`).
-->
为了保证这个解析过程没有歧义，registrar 会在 `flag_gems._FULL_CONFIG`
（也就是可注册算子的权威数据源）中查找每个调度键，找到该键最初绑定的函数，
再根据这个函数自身的名字去解析当前生效的属性。这样可以避免共享同一个
调度键前缀、但实际绑定到不同函数的重载互相冲突
（例如 `_softmax` 与 `_softmax.out`）。

<!--
As a consequence, if a dispatch key passed to the registrar cannot be found
in `flag_gems._FULL_CONFIG` at all, registration raises a `ValueError`
rather than guessing at a name — registering (or overriding) an operator
that was never a real registration is not allowed.
-->
因此，如果传给 registrar 的某个调度键在 `flag_gems._FULL_CONFIG`
中根本查不到，注册过程会直接抛出 `ValueError`，而不会去猜测一个名字——
功能上不允许注册（或重载）一个本来就不存在的算子。

<!--
## 6. Unused overrides fail the test run
-->
## 6. 未被实际调用的重载会导致测试失败

<!--
`override_from_file` (and, transitively, `override_batch_from_files` and
`--override`/`--override-config`) wraps the loaded candidate so that
`DynamicOpOverride` can track whether it was actually called. If
`restore()` or `restore_all()` runs and finds that a tracked override was
never invoked, it raises an `AssertionError` — this turns "I pointed
`--override` at the wrong operator name (or a candidate with a typo that
never gets exercised)" into a hard test failure instead of a silently
useless run:
-->
`override_from_file`（以及间接地，`override_batch_from_files` 和
`--override`/`--override-config`）会对加载到的候选实现进行包装，
使得 `DynamicOpOverride` 能够追踪它是否被实际调用过。如果
`restore()` 或 `restore_all()` 执行时发现某个被追踪的重载从未被调用过，
就会抛出 `AssertionError`——这样可以把"`--override` 指向了错误的算子名
（或者候选实现里有个从未被执行到的笔误）"这类问题，
从一次悄无声息的无效运行，变成一次明确的测试失败：

```shell
pytest tests/test_softmax.py \
    --override softmax:./candidates/softmax_v2.py:my_softmax
# 如果 my_softmax 从未被真正调用过，整个会话会失败
```

<!--
Overrides applied directly via the bare `registry.override(...)` call (as
opposed to `override_from_file`) are not tracked this way, since there is
no file-loading step to guard against typos or unreachable candidates.
-->
直接通过裸的 `registry.override(...)` 调用应用的重载（而不是通过
`override_from_file`）不会被这样追踪，因为这种方式并没有文件加载这一步，
也就不存在需要防范的笔误或无法触达的候选实现。

<!--
## 7. Concurrent testing of multiple implementations
-->
## 7. 并发测试多个实现

<!--
Because each `DynamicOpOverride` only ever mutates attributes of the
already-imported `flag_gems` module *within its own process*, independent
`pytest` invocations — run in separate OS processes — can each load a
different candidate implementation for the same operator without conflict:
-->
由于每个 `DynamicOpOverride` 实例都只在**自己所在的进程内**修改已导入的
`flag_gems` 模块的属性，因此在各自独立的操作系统进程中启动的多次
`pytest` 调用，可以分别为同一个算子加载不同的候选实现，而不会相互冲突：

```shell
pytest tests/test_softmax.py --override softmax:./variant_a.py &
pytest tests/test_softmax.py --override softmax:./variant_b.py &
pytest tests/test_softmax.py &   # 基线版本，不做任何重载
wait
```

<!--
This is useful for A/B-testing kernel variants, or for running the full test
suite against several candidate implementations in a CI matrix, without
maintaining separate copies of the operator source tree.
-->
这一特性对于内核变体的 A/B 测试非常有用，也可以在 CI 的矩阵作业中，
针对多个候选实现分别运行完整的测试套件，而无需为每个变体维护一份独立的算子源码目录。

<!--
> [!WARNING]
> **Warning**
>
> A `DynamicOpOverride` instance overrides attributes on the shared
> `flag_gems` module object. Within a *single* process, overrides from
> different `DynamicOpOverride` instances (or different threads) are not
> isolated from each other — the most recent `override()` call wins, and
> `restore_all()` on one registry only restores the operators it itself
> overrode. Prefer one registry per process/test session, scoped with the
> `with` statement, to keep behavior predictable.
-->
> [!WARNING]
> **警告**
>
> `DynamicOpOverride` 实例是在共享的 `flag_gems` 模块对象上修改属性的。
> 在**同一个**进程内，来自不同 `DynamicOpOverride` 实例（或不同线程）的重载
> 彼此并不隔离——最近一次的 `override()` 调用会生效，并且某个
> registry 上的 `restore_all()` 只会恢复它自己重载过的算子。
> 建议每个进程/测试会话只使用一个 registry，并通过 `with`
> 语句限定其作用范围，以保持行为的可预测性。

## Candidate-only profiling 与外部采集

`--profile-only --case-id <id>` 精确重放 `--list-cases` 中的一个 case。FlagGems 构造输入、执行 `--profile-warmup` 次预热并同步，然后在可选的采集 context 中执行 `--profile-iterations` 次候选调用及最后同步；不运行 correctness reference 或 benchmark latency 计时。

外部执行器通过 `pytest.main(pytest_args, plugins=[plugin])` 显式传入插件对象。插件实现 `pytest_flaggems_profile_scope(backend, case_id)` 并返回 context manager；`backend` 是 FlagGems vendor 名称。context 的进入/退出负责开始/结束采集，并应通过 `finally` 保证候选异常时也清理。benchmark conftest 注册 first-result hookspec，默认没有插件（或插件返回 None）时只做普通 candidate-only 重放，不执行外部采集。

这条接口不需要环境变量指定模块、动态导入执行器或依赖 `__main__`。需要采集的执行器必须自行核对插件实际进入且完成，不能只凭 pytest 退出 0 判定成功；编译器/profiler 准备、completion marker 和产物仍由执行器负责。FlagGems 不依赖 KGS。每个 benchmark session 使用独立进程，不在一个进程中并发调用多个 `pytest.main()`。

## Candidate-only Preflight

普通 correctness/benchmark 同样使用 `--override`。正确性 JSON 的每个 pytest case 带有 `candidate_calls`，记录该 case 的测试调用阶段内实际调用的候选次数；benchmark 只有真正调用过注入候选才写入 `candidate_source: override`，baseline 仍单独调用原始 `torch_op`。不要用 pytest 整体通过代替候选覆盖检查。正确性 case 全部 skip 时仅恢复 override，不因零调用额外报错；这并不代表正确性通过，KGS 会保留 `ALL_SKIP` 语义并继续适用的 benchmark。

`benchmark/` 支持 `--preflight-only`：沿用原有 case 枚举和输入构造，对每个选中 case 调用一次候选并同步设备，不运行正确性 reference、不进行 warmup/benchmark 计时，也不计算加速比。这里的一次指 Python 算子入口调用一次；候选内部的编译或 autotune 仍可能启动多个 kernel。通过仅表示候选可执行，不表示数值正确或性能达标。

在 FlagGems checkout 根目录运行，例如检查 `addmm_` 的全部 core case：

```bash
python -m pytest -q benchmark/test_addmm_.py --level core \
  --preflight-only --override addmm_:/path/candidate.py:run \
  --record json --output /tmp/preflight.json
```

不指定 `--case-id` 时检查全部枚举 case；也可重复传入 `--case-id <id>` 选择子集，ID 从同一 checkout 的 `--list-cases` 获得。未提供 override 时检查仓库当前 Gems 实现。尚未支持 case 枚举的 benchmark 会明确报错，不回退执行完整 benchmark。

JSON 报告使用 `schema_version: flaggems.preflight/v1`，`records` 中逐 case 记录 `operator`、`nodeid`、`case_id`、是否使用 `override`、入口调用次数 `count` 和 `status`（`passed`/`failed`）；失败可附带 `error`。它不含 latency/speedup。报告每次覆盖，不合并旧运行结果。消费者仍须同时检查 pytest 退出码、期望 case 的完整覆盖和候选确实被注入；空报告不能算通过。

`--preflight-only` 不可与 `--profile-only`、`--list-cases`、`--query` 或非零 `--parallel` 合用。未知/未执行的显式 case、无 case、全部 skip 或候选异常不会作为成功返回。正确性测试仍使用 `tests/`，性能评测仍使用普通 `benchmark/`，profiling 使用独立的 `--profile-only`。
