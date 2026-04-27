---
title: Using C++ Wrapped Operators
weight: 90
---

# Using C++-Based Operators for Optimal Performance

Another advanced optimization path with *FlagGems* is the use of *C++ wrapped operators*
for selected operations. While Triton kernels offer reasonably good compute performance,
Triton itself is a DSL implemented in Python. This means that both the operator definitions and
the runtime dispatchers are written in Python, which can introduce **non-trivial overhead**
in latency-sensitive or high-throughput scenarios.

To address this, *FlagGems* provides a C++ runtime solution that encapsulates
the operator's wrapper logic, registration mechanism, and runtime management in C++,
while still reusing the underlying Triton kernels for the actual computation.
This approach preseves the kernel-level efficiency from Triton
while significantly reducing Python-related overhead, enabling tighter integration
with low-level CUDA workflows and improving overall inference performance.

## 1. Installation

To use the C++ operator wrappers:

{{% steps %}}

1. Follow the [installation guide](/FlagGems/installation/) to compile
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

   If `has_c_extension` is `True`, then the C++ runtime execution path is available.

1. When installed successfully, C++ wrappers will be preferred **in patch mode**.
   When explicitly [building models](/FlagGems/usage/modules/) using modules
   provided by *FlagGems*, they have a higher precedence over their Python
   equivalents as well.

   For example, the operator `gems_rms_forward` will by default use the C++ wrapper
   version of `rms_norm`. You can refer to the actual usage in the
   [`normalization.py`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/modules/normalization.py#L33)
   to better understand how C++ wrapped operators can be integrated and invoked.

{{% /steps %}}

## 2. Invoke  C++ operators explicitly

If you want to **directly invoke** the C++-wrapped operators, thus bypassing
any patching logics or fall back paths, you can use the `torch.ops.flag_gems`
namespace as shown below:

```python
output = torch.ops.flag_gems.fused_add_rms_norm(...)
```

This gives you *precise control* over operator dispatching, which can be beneficial
in some performance-critical contexts.

## Reference: Currently supported C++-wrapped operators

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

We are actively expanding this list as part of our ongoing performance optimization work.
