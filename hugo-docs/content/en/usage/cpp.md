---
title: Using C++ Wrapped Operators
weight: 90
---

## Using C++-Based Operator Wrappers for Further Performance Gains

Another advanced optimization path with *FlagGems* is the use of **C++ wrappers** for selected operators.
While Triton kernels offer reasonably good compute performance, Triton itself is a Python-embedded DSL.
This means that both the operator definitions and the runtime dispatches are in Python,
which can introduce **non-trivial overhead** in latency-sensitive or high-throughput scenarios.

To address this, *FlagGems* provides a C++ runtime solution that encapsulates the operator’s wrapper logic,
registration mechanism, and runtime management entirely in C++,
while still reusing the underlying Triton kernels for the actual computation.
This approach preseves Triton's kernel-level efficiency while significantly reducing Python-related overhead,
enabling tighter integration with low-level CUDA workflows and improving overall inference performance.

## Installation & Setup

To use the C++ operator wrappers:

1. Follow the [installation guide](./installation.md) to compile
   and install the C++ version of `flag_gems`.

1. Verify that the installation is successful using the following snippet:

   ```python
   try:
       from flag_gems import c_operators
       has_c_extension = True
   except Exception as e:
       c_operators = None  # avoid import error if c_operators is not available
       has_c_extension = False
   ```

   If `has_c_extension` is `True`, then the C++ runtime path is available.

1. When installed successfully, C++ wrappers will be preferred **in patch mode**
   and when explicitly building models using *FlagGems*-defined modules.

   For example, `gems_rms_forward` will by default use the C++ wrapper version of `rms_norm`.
   You can refer to the actual usage in the
   [`normalization.py`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/modules/normalization.py#L46)
   to better understand how C++ operator wrappers are integrated and invoked.

## Explicitly Using C++ Operators

If you want to *invoke the C++-wrapped operators directly*, bypassing any patch logics
or fall back paths, you can use the `torch.ops.flag_gems` namespace as shown below:

```python
output = torch.ops.flag_gems.fused_add_rms_norm(...)
```

This gives you *precise control* over operator dispatch, which can be beneficial
in performance-sensitive contexts.

## Currently Supported C++-Wrapped Operators

| Operator Name        | Description                              |
| -------------------- | ---------------------------------------- |
| `add`                | Element-wise addition                    |
| `bmm`                | Batch Matrix Multiplication              |
| `cat`                | Concatenate                              |
| `fused_add_rms_norm` | Fused addition + RMSNorm                 |
| `mm`                 | Matrix multiplication                    |
| `nonzero`            | Returns the indices of non-zero elements |
| `rms_norm`           | Root Mean Square normalization           |
| `rotary_embedding`   | Rotary position embedding                |
| `sum`                | Reduction across dimensions              |

We are actively expanding this list as part of our ongoing performance roadmap.
