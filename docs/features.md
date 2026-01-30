## Features

### Rich Collection of Operators

FlagGems features a large collection of PyTorch compatible operators.
Operators will be implemented according to [operator list](./operators.md).

### Hand-optimized Performance for Selected Operators

The following chart shows the speedup of FlagGems compared with PyTorch ATen library in eager mode.
The speedup is calculated by averaging the speedup on each shape, representing the overall performance of the operator.

![Operator Speedup](./assets/speedup-20251225.png)

### Eager-mode ready, independent of `torch.compile`

> TBD

### Automatic Code Generation

FlagGems provides an automatic code generation mechanism that enables developers
to easily generate both pointwise and fused operators.
The auto-generation system supports a variety of requirements, including standard
element-wise computations, non-tensor parameters, and specifying output types.
Please refer to [pointwise_dynamic](./pointwise_dynamic.md) document for more details.

### Function-level Kernel Dispatching

FlagGems introduces `LibEntry`, which independently manages the kernel cache and bypasses the runtime of `Autotuner`,
`Heuristics`, and `JitFunction`. To use this feature, simply decorate the Triton kernel with LibEntry.

`LibEntry` also supports direct wrapping of `Autotuner`, `Heuristics`, and `JitFunction`,
preserving full tuning functionality.
However, it avoids nested runtime type invocations, eliminating redundant parameter processing.
This means no need for binding or type wrapping, resulting in a simplified cache key format
and reduced unnecessary key computation.

### Generic Interface for Diverse Platforms

<div id="platforms-supported"></div>

FlagGems supports a wide range of hardware platforms and has been extensively tested
across different hardware configurations.

The currently supported platforms are:

| Vendor     | State           | float16 | float32 | bfloat16 |
| ---------- | --------------- | ------- | ------- | -------- |
| AIPU       | ✅ （Partial ） | ✅      | ✅      | ✅       |
| ARM(CPU)   | 🚧              |         |         |          |
| Ascend     | ✅ （Partial ） | ✅      | ✅      | ✅       |
| Cambricon  | ✅              | ✅      | ✅      | ✅       |
| Hygon      | ✅              | ✅      | ✅      | ✅       |
| Iluvatar   | ✅              | ✅      | ✅      | ✅       |
| Kunlunxin  | ✅              | ✅      | ✅      | ✅       |
| MetaX      | ✅              | ✅      | ✅      | ✅       |
| Mthreads   | ✅              | ✅      | ✅      | ✅       |
| NVIDIA     | ✅              | ✅      | ✅      | ✅       |
| TsingMicro | 🚧              |         |        |          |
| Sunrise | ✅ （Partial ）    | ✅       | ✅      | ✅       |


### Backend Supports

FlagGems supports 10+ backends.

### C++ Triton Function Dispatcher

The C++ Triton function dispatcher is an ongoing work.

### C++ Runtime

FlagGems can be installed either as a pure Python package or as a package with C++ extensions.
The C++ runtime is designed to address the overhead of the Python runtime
and improve end-to-end performance.
