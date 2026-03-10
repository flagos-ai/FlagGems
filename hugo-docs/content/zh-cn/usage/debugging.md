---
title: 调试支持
weight: 35
---

# Enable Debug Logging

If you want to log the operator usage during runtime, you can
set `record=True` along with `path` set to  the path of the log file.

```python
flag_gems.enable(
    record=True,
    path="./gems_debug.log"
)
```

After running your script, inspect the log file (e.g., `./gems_debug.log`) to check
the list of operators that have been invoked through `flag_gems`.

Sample log content:

```shell
$ cat ./gems_debug.log
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.fused.reshape_and_cache: GEMS RESHAPE_AND_CACHE
```
