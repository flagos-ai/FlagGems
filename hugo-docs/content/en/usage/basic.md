---
title: Basic usage
weight: 20
---

# Basic Usage

To use the *FlagGems* operator library, import `flag_gems` and enable acceleration
before running your program. You can enable it globally, selectively, or temporarily.

## Option 1: Global Enablement

To apply *FlagGems* optimizations across your entire script or your interactive session:

```python
import flag_gems

# Enable all FlagGems operators globally
flag_gems.enable()
```

Once enabled, all supported operators in your code will be replaced automatically
by the optimized *FlagGems* implementations — no further changes needed.

## Option 2: Selective Enablement

To enable only specific operators and skip the rest:

```python
import flag_gems

# Enable only selected ops
flag_gems.only_enable(include=["rms_norm", "softmax"])
```

This is useful when you want to accelerate only a subset of operators.

## Option 3: Scoped Enablement

For finer controls, you can enable *FlagGems* only within a specific code block
using a context manager:

```python
import flag_gems

# Enable flag_gems temporarily
with flag_gems.use_gems():
    # Code inside this block will use FlagGems-accelerated operators
    ...
```

This scoped usage is useful when you want to:

- perform performance benchmarks, or
- compare correctness between implementations, or
- apply acceleration selectively in complex workflows.

You can also use selective enablement in a context manager:

```python
# Enable only specific ops in the scope
with flag_gems.use_gems(include=["sum", "add"]):
    # Only sum and add will be accelerated
    ...

# Or exclude specific ops
with flag_gems.use_gems(exclude=["mul", "div"]):
    # All operators except mul and div will be accelerated
    ...
```

Note: The `include` parameter has higher priority than `exclude`.
If both are provided, `exclude` is ignored.
