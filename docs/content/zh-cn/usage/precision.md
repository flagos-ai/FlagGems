---
title: 精度检查（实验性）
weight: 46
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
# Precision Checking (Experimental)
-->
# 精度检查（实验性功能）

<!--
FlagGems provides an experimental precision-checking mechanism that
automatically compares the output of FlagGems operators against native
PyTorch (CPU) results, and logs any discrepancies to a file.
This is useful for verifying numerical correctness during development.
-->
*FlagGems* 提供了一个实验性的精度检查机制，能够自动将 FlagGems 算子的输出
与原生 PyTorch（CPU）的计算结果进行对比，并将精度不一致的情况记录到日志文件中。
这对于开发过程中验证数值正确性非常有用。

<!--
## How to Enable
-->
## 如何启用

<!--
Enabling precision checking requires two steps:
1. Call `enable_precision_check()` from `flag_gems.logging_utils` to
   configure the precision logger.
2. Pass `PrecisionCheckRegister` as the `registrar` parameter to
   `enable()` or `only_enable()`, so that operators are wrapped with
   precision-checking logic.
-->
启用精度检查需要两步：

1. 从 `flag_gems.logging_utils` 中调用 `enable_precision_check()` 配置精度日志。
2. 将 `PrecisionCheckRegister` 作为 `registrar` 参数传递给
   `enable()` 或 `only_enable()`，使算子在注册时被包装上精度检查逻辑。

```python
import flag_gems
from flag_gems.logging_utils import enable_precision_check
from flag_gems.runtime.precision_register import PrecisionCheckRegister

# 第一步：配置精度检查（初始化精度日志）
enable_precision_check()

# 第二步：使用 PrecisionCheckRegister 注册所有算子
flag_gems.enable(registrar=PrecisionCheckRegister)

# 运行你的模型或算子
output = model(input)
```

<!--
You can also use it with `only_enable()` to check specific operators:
-->
也可以配合 `only_enable()` 仅对特定算子进行精度检查：

```python
from flag_gems.logging_utils import enable_precision_check
from flag_gems.runtime.precision_register import PrecisionCheckRegister

enable_precision_check(rtol=1e-3, atol=1e-4)
flag_gems.only_enable(
    include=["mm", "add", "softmax"],
    registrar=PrecisionCheckRegister,
)
```

<!--
## Configuration
-->
## 配置参数

<!--
You can customize the precision checking behavior by passing parameters
to `enable_precision_check()`.
-->
你可以通过向 `enable_precision_check()` 传递参数来自定义精度检查的行为。

<!--
| Parameter      | Type        | Default                        | Description                                         |
| -------------- | ----------- | ------------------------------ | --------------------------------------------------- |
| `rtol`         | `float`     | `1e-4`                         | Relative tolerance                                  |
| `atol`         | `float`     | `1e-5`                         | Absolute tolerance                                  |
| `max_checks`   | `int`       | `10`                           | Max checks per operator before skipping             |
| `log_once`     | `bool`      | `True`                         | Only log the first failure per operator             |
| `path`         | `str`       | `~/.flaggems/precision.log`    | Log file path                                       |
-->
| 参数名称       | 数据类型    | 默认值                         | 描述                                                   |
| -------------- | ----------- | ------------------------------ | ------------------------------------------------------ |
| `rtol`         | `float`     | `1e-4`                         | 相对误差容忍度                                         |
| `atol`         | `float`     | `1e-5`                         | 绝对误差容忍度                                         |
| `max_checks`   | `int`       | `10`                           | 每个算子最多检查的调用次数（超过后不再检查以减少开销） |
| `log_once`     | `bool`      | `True`                         | 每个算子仅记录一次失败                                 |
| `path`         | `str`       | `~/.flaggems/precision.log`    | 日志文件路径                                           |

```python
from flag_gems.logging_utils import enable_precision_check

enable_precision_check(
    rtol=1e-3,
    atol=1e-5,
    max_checks=20,
    path="./my_precision.log",
)
```

<!--
## Disabling
-->
## 关闭精度检查

<!--
To disable precision checking at runtime:
-->
如需在运行时关闭精度检查：

```python
from flag_gems.logging_utils import disable_precision_check

disable_precision_check()
```

<!--
## Log Output
-->
## 日志输出

<!--
Precision check results are written to `~/.flaggems/precision.log` by default.
Only operators that fail the tolerance check will be logged.
-->
精度检查的结果默认写入 `~/.flaggems/precision.log` 文件。
只有未通过容忍度检查的算子才会被记录。

<!--
Sample log content:
-->
日志输出示例：

```shell
$ cat ~/.flaggems/precision.log
2025-05-19 10:00:01 [WARNING] Op: add.Tensor | FAIL | in: [(2, 3):torch.float16] | out: (2, 3):torch.float16 | max_abs: 1.200000e-03 | max_rel: 2.500000e-02 | rtol=0.01, atol=0.01
```

<!--
## Behavior Details
-->
## 行为细节

<!--
The precision checker has several built-in safeguards to minimize
performance impact:
-->
精度检查器内置了多项保护措施以尽量减少对性能的影响：

<!--
- Only the first N calls per operator are checked (controlled by `max_checks`)
- Tensors larger than 1M elements are skipped to avoid copy overhead
- Once an operator logs a failure, it will not be checked again
- Pure layout/memory ops (clone, view, copy_, etc.) are automatically skipped
- Random sampling ops (uniform_, normal_, etc.) are automatically skipped
- `.out` variant operators are skipped
- For float16/bfloat16 inputs, tolerance is automatically relaxed to at least 1e-2
-->
- 每个算子仅检查前 N 次调用（由 `max_checks` 控制）
- 超过 100 万元素的张量会被跳过，以避免大张量拷贝的开销
- 一旦某个算子记录了一次失败，后续不再对其进行检查
- 纯布局/内存操作（如 `clone`、`view`、`copy_`）会被自动跳过
- 随机采样算子（如 `uniform_`、`normal_`）会被自动跳过
- `.out` 变体的算子也会被跳过
- 对于 `float16` / `bfloat16` 类型的输入，容忍度会自动放宽到至少 `1e-2`

<!--
## How It Works
-->
## 工作原理

<!--
When `PrecisionCheckRegister` is used as the registrar, each operator
is wrapped with a precision-checking decorator. The wrapper:
1. Executes the FlagGems (GPU) implementation normally.
2. Copies inputs to CPU and runs the native `aten` operator as reference.
3. Compares the two results using the configured tolerance.
4. Logs a warning if the results diverge beyond tolerance.
-->
当使用 `PrecisionCheckRegister` 作为注册器时，每个算子会被包装上
一个精度检查装饰器。该装饰器的工作流程为：

1. 正常执行 FlagGems（GPU）实现，得到结果。
2. 将输入拷贝到 CPU，调用原生 `aten` 算子计算参考结果。
3. 使用配置的容忍度对比两个结果。
4. 如果结果超出容忍度范围，记录一条警告日志。

> [!WARNING]
> **警告**
>
> 精度检查会将 GPU 张量拷贝到 CPU 并执行原生 PyTorch 计算作为参考，
> 这会带来显著的性能开销。此功能仅建议在开发调试阶段使用，
> 不应在生产环境中启用。
