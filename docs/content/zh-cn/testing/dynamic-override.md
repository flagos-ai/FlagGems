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
## 5. Concurrent testing of multiple implementations
-->
## 5. 并发测试多个实现

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
