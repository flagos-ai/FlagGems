---
title: Using Experimental Operators
weight: 40
---
# Using Experimental Operators

The `experimental_ops` module provides a space for new operators
that are not yet ready for production release.
Operators in this package are accessible via `flag_gems.experimental_ops.*`.
These operators follow the same development patterns as the core, stable operators.

```python
from flag_gems import experimental_ops as ops

result = ops.rmsnorm(*args)
```

You can also use experimental operators in a `use_gems()` context,
however, you have to explicitly specify the full path for accessing the operator.

```python
with flag_gems.use_gems():
    result = flag_gems.experimental_ops.rmsnorm(*args)
```

<!--TODO(Qiming): Add link to experimental operators-->
