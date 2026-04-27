---
title: 调试支持
weight: 45
---

<!--
# Enable Debug Logging
-->
# 启用调试日志

<!--
To facilitate operator development (especially debugging or profiling),
*FlagGems* supports some optional parameters for the `enable()` and
the `only_enable()` API interface, as shown below.
-->
为了方便算子开发（尤其是调试和性能分析优化），*FlagGems* 为 `enable()`
和 `only_enable()` API 接口提供一些可选的参数，方便问题诊断。
具体如下：

<!--
| Parameter      | Type        | Description                                         |
| -------------- | ----------- | --------------------------------------------------- |
| `record`       | `bool`      | Log operator calls for debugging or profiling       |
| `path`         | `str`       | Log file path (only used when `record=True`)        |
-->
| 参数名称       | 数据类型    | 描述                                            |
| -------------- | ----------- | ----------------------------------------------- |
| `record`       | `bool`      | 记录算子调用以便调试或性能分析                  |
| `path`         | `str`       | 给出日志文件路径（仅适用于 `record=True` 情形） |

<!--
If you want to log the operator usage during runtime, you can
set `record=True` along with `path` set to the path of the log file.
-->
如果你希望在运行时记录算子的调用情况，你可以设置 `record=True`，
同事将 `path` 设置为日志文件的路径字符串。

```python
flag_gems.enable(
    record=True,
    path="./gems_debug.log"
)
```

<!--
When your program has completed execution, you can inspect the log file
(e.g., `./gems_debug.log`) to check the list of operators that have been invoked
(aka. accelerated) through `flag_gems`.
-->
在你的程序结束执行之后，你可以检视日志文件（如 `./gems_debug.log`）
以了解哪些算子被通过 `flag_gems` 调用过（即被加速过）。

<!--
Sample log content:
-->
下面是日志输出的样子：

```shell
$ cat ./gems_debug.log
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.fused.reshape_and_cache: GEMS RESHAPE_AND_CACHE
```

<!--
> [!WARNING]
> **Warning**
>
> The logging behavior involves some I/O operations, so it may have some negative impact
> on your workload. The impact could be non-trivial if the operator is performing
> simple tasks or if the operator is invoked very frequently.
-->
> [!WARNING]
> **警告**
>
> 记录日志的行为牵涉到磁盘 I/O 操作，可能会对工作负载的性能带来负面影响。
> 如果算子所执行的任务是比较简单的计算或者被调用的频率很高，
> 日志记录带来的性能影响可能不容忽视。
