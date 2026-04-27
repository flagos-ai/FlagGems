---
title: Debugging Support
weight: 45
---

# Enable Debug Logging

To facilitate operator development (especially debugging or profiling),
*FlagGems* supports some optional parameters for the `enable()` and
the `only_enable()` API interface, as shown below.

| Parameter      | Type        | Description                                         |
| -------------- | ----------- | --------------------------------------------------- |
| `record`       | `bool`      | Log operator calls for debugging or profiling       |
| `path`         | `str`       | Log file path (only used when `record=True`)        |

If you want to log the operator usage during runtime, you can
set `record=True` along with `path` set to the path of the log file.

```python
flag_gems.enable(
    record=True,
    path="./gems_debug.log"
)
```

When your program has completed execution, you can inspect the log file
(e.g., `./gems_debug.log`) to check the list of operators that have been invoked
(aka. accelerated) through `flag_gems`.

Sample log content:

```shell
$ cat ./gems_debug.log
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.fused.reshape_and_cache: GEMS RESHAPE_AND_CACHE
```

> [!WARNING]
> **Warning**
>
> The logging behavior involves some I/O operations, so it may have some negative impact
> on your workload. The impact could be non-trivial if the operator is performing
> simple tasks or if the operator is invoked very frequently.
