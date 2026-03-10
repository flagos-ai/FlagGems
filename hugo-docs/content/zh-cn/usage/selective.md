---
title: 选择性启用算子
weight: 30
---

# Selective Usage

The `flag_gems.enable(...)` and `flag_gems.only_enable(...)` functions support several optional parameters
which give you finer-grained control over how acceleration is applied.
This allows for more flexible integration and easier debugging or profiling when working with complex workflows.

## Parameters

| Parameter      | Type      | Description                                         |
| -------------- | --------- | --------------------------------------------------- |
| `unused`       | List[str] | Disable specific operators (for `enable`)           |
| `include`      | List[str] | Enable only specific operators (for `only_enable`)  |
| `record`       | bool      | Log operator calls for debugging or profiling       |
| `path`         | str       | Log file path (only used when `record=True`)        |

## Example : Selectively Disable Specific Operators

You can use the `unused` parameter in `enable()` to exclude certain operators from being accelerated by `FlagGems`.
This is especially useful when a particular operator does not behave as expected in your workload,
or if you're seeing suboptimal performance and want to use the original implementation.

```python
flag_gems.enable(unused=["sum", "add"])
```

With this configuration, `sum` and `add` will continue to use the native PyTorch implementations,
while all other supported operators will use the *FlagGems* version.

## Example : Selectively Enable Specific Operators

You can use `only_enable()` with the `include` parameter to accelerate only a subset of operators:

```python
flag_gems.only_enable(include=["rms_norm", "softmax"])
```

This registers only the specified operators, skipping all the others.
