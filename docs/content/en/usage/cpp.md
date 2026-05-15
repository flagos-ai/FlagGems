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
This approach preserves the kernel-level efficiency from Triton
while significantly reducing Python-related overhead, enabling tighter integration
with low-level CUDA workflows and improving overall inference performance.

## 1. Architecture

The C++ wrapped operators in *FlagGems* are built on top of
[`libtriton_jit`](https://github.com/flagos-ai/libtriton_jit), a multi-backend
C++ runtime for Triton JIT functions. `libtriton_jit` reimplements the Triton
JIT runtime in C++ (argument specialization, kernel caching, and launch) while
delegating the actual compilation to the upstream Triton compiler.

In this stack:

- The Triton kernels (`*.py`) remain the source of truth for device-side computation.
- `libtriton_jit` handles JIT specialization, kernel caching, and backend-specific
  launches (currently supporting **NVIDIA (CUDA)**, **Moore Threads (MUSA)**,
  **Huawei Ascend (NPU)** and **Iluvatar CoreX (IX)**).
- FlagGems's C++ wrappers (under `lib/`, e.g. `rms_norm.cpp`, `mm.cpp`) implement
  tensor metadata handling, shape/type promotion, and argument preparation in C++,
  then invoke the Triton kernels through `libtriton_jit::TritonJITFunction`.
- On top of the wrappers, *FlagGems* ships two Python-facing extension modules
  (`src/flag_gems/csrc/cstub.cpp` and `src/flag_gems/csrc/aten_patch.cpp`)
  and one installable C++ library target (`FlagGems::operators`), which together
  expose the wrappers through four different invocation paths (see
  [§3. Ways to invoke C++ operators](#3-ways-to-invoke-c-operators)).

Regardless of which invocation path is used, the **wrapper logic itself
is always executed in C++** (tensor metadata handling, argument type and
specialization analysis, kernel cache lookup, and launch-argument preparation) instead of in Python —
that part of the Python overhead is eliminated unconditionally, while the
compute path continues to use the same Triton kernels.

Whether the **PyTorch dispatcher overhead** is also avoided depends on
the path you pick:

- Paths that go through the dispatcher (`torch.ops.flag_gems.*` and the
  ATen direct replacement) still pay the usual dispatcher cost, but since
  the op implementation sitting behind the dispatcher is C++ rather than
  a Python wrapper, the boxed-call overhead is still noticeably smaller
  than for a pure-Python custom op.
- Paths that bypass the dispatcher (the `c_operators` pybind module and
  the native C++ API) remove the dispatcher cost entirely; the native
  C++ API additionally removes any Python-interpreter involvement on the
  call path.

See [§3. Ways to invoke C++ operators](#3-ways-to-invoke-c-operators) for
the trade-offs of each path.

## 2. Install and enable

To make the C++ wrapper **fully effective** you need both of the following:

{{% steps %}}

1. **At build/install time: enable the C++ extension and build in Release mode**

   Install from source with at least `-DFLAGGEMS_BUILD_C_EXTENSIONS=ON` and
   `-DCMAKE_BUILD_TYPE=Release` (the latter ensures both FlagGems itself
   and the `libtriton_jit` subproject built alongside it are compiled with
   platform-targeted optimizations; without it the wrapper will be noticeably
   slower):

   ```shell
   CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DCMAKE_BUILD_TYPE=Release" \
   pip install -v -e .
   ```

   > [!NOTE]
   > If the command above fails, try adding `--no-build-isolation` so
   > that pip reuses the PyTorch already installed in your environment
   > and the build dependencies from `requirements_<backend>.txt`.

   Other useful options:

   - `-DFLAGGEMS_BACKEND=<CUDA|IX|MUSA|NPU>`: select the target backend (default `CUDA`);
   - `-DFLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP=ON`: build the pointwise-dynamic
     operators (`add`, `div`, `fill`);
   - `-DFLAGGEMS_BUILD_CTESTS=ON`: build the `ctests/` GTest suite
     (the only way to verify the native C++ API in §3.4);
   - `-DFLAGGEMS_USE_EXTERNAL_TRITON_JIT=ON -DTritonJIT_ROOT=<path>`:
     build against an externally installed `libtriton_jit`.

   See the [install guide](../../getting-started/install/#install-from-source)
   for the complete per-backend examples and `libtriton_jit` details.

1. **At runtime: `export USE_C_EXTENSION=1`**

   Building the C++ extension alone is **not enough**. `src/flag_gems/config.py`
   gates several higher-level behaviors behind this env var — if you don't
   set it, the following paths silently fall back to Python:

   | Path / behavior                                      | Available after build | Also needs `USE_C_EXTENSION=1` |
   | ---------------------------------------------------- | :-------------------: | :----------------------------: |
   | §3.1 `torch.ops.flag_gems.*`                         | ✅                    | —                              |
   | §3.3 `c_operators` pybind                            | ✅                    | —                              |
   | §3.2 ATen direct replacement (`aten_patch`)          | ❌                    | ✅                             |
   | C++ branch in `flag_gems.enable()`                   | ❌                    | ✅                             |
   | C++ branch in `GemsRMSNorm` and other `nn.Module`s   | ❌                    | ✅                             |

   So for normal use:

   ```shell
   export USE_C_EXTENSION=1
   ```

1. **Quick sanity check**

   The following snippet verifies, in one go, all three paths that are
   observable from Python:

   ```python
   import torch
   import flag_gems
   from flag_gems import c_operators, aten_patch
   from flag_gems.config import has_c_extension, use_c_extension

   assert has_c_extension, "C++ extension was not built"
   assert use_c_extension, "please `export USE_C_EXTENSION=1`"
   assert hasattr(torch.ops.flag_gems, "mm"), "§3.1 torch.ops.flag_gems.* not registered"
   assert aten_patch.get_registered_ops(), "§3.2 no ATen op has been replaced"
   _ = c_operators.mm                                                 # §3.3
   ```

   The §3.4 native C++ API is not observable from Python. To verify it, build
   with `-DFLAGGEMS_BUILD_CTESTS=ON` and run `ctest`:

   ```shell
   BUILD_DIR=$(ls -d build/*/ | head -n 1)
   ctest --test-dir "${BUILD_DIR}" --output-on-failure
   ```

   > [!NOTE]
   > When running a single test binary manually (e.g.
   > `"${BUILD_DIR}/ctests/test_triton_mm"`), you must
   > `export FLAGGEMS_SOURCE_DIR=$(pwd)/src/flag_gems` so the C++ runtime
   > can locate the Triton kernel `.py` files; `ctest` sets this
   > automatically.

1. **Typical usage scenarios**

   With the two steps above in place, the following two usage styles will
   **automatically prefer the C++ wrapped operators** — you don't need to
   change any call sites:

   - **Patch mode (`flag_gems.enable()`)**: monkey-patches `torch.*` entry
     points. When `use_c_extension` is `True`, the patched functions
     dispatch to `torch.ops.flag_gems.*` (§3.1); otherwise they fall back
     to the pure-Python implementation.
   - **Building models with the `nn.Module` classes FlagGems ships**,
     e.g. [`flag_gems.modules.GemsRMSNorm`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/modules/normalization.py).
     These modules already contain the "if C++ is available → call
     `torch.ops.flag_gems.*`, otherwise call the Python implementation"
     branch internally. See
     [`gems_rms_forward`](https://github.com/flagos-ai/FlagGems/blob/v4.2.0/src/flag_gems/modules/normalization.py#L33)
     for a concrete example.

{{% /steps %}}

## 3. Ways to invoke C++ operators

Once the C++ extensions are built, the same underlying C++ wrapper can be
invoked through **four** different paths. Each path targets a different use
case and has a different level of dispatcher overhead.

### 3.1 Via `torch.ops.flag_gems.*` (custom-op namespace)

All C++ wrappers are registered as PyTorch *custom ops* under the
`flag_gems` namespace via `TORCH_LIBRARY(flag_gems, m)` in
`src/flag_gems/csrc/cstub.cpp`. You can call them explicitly from Python,
bypassing any patching logic or Python-side fall back paths:

```python
output = torch.ops.flag_gems.fused_add_rms_norm(...)
out    = torch.ops.flag_gems.mm(a, b)
```

### 3.2 Via ATen direct replacement (transparent `torch.*` patching at the dispatcher)

For a subset of operators, *FlagGems* additionally registers the C++
implementations directly under the **`aten::` namespace** using
`TORCH_LIBRARY_IMPL(aten, <dispatch_key>, m)` in
`src/flag_gems/csrc/aten_patch.cpp`. The dispatch key is chosen by backend:

- `CUDA` for NVIDIA CUDA and Iluvatar CoreX (IX);
- `PrivateUse1` for Huawei Ascend (NPU) and Moore Threads (MUSA).

Because the registration goes straight into the PyTorch dispatcher, calling
standard PyTorch APIs such as `torch.nonzero(x)` or `x.copy_(y)` on a
supported device will transparently dispatch to the FlagGems C++
implementation — no Python-level monkey patching required. This is the
lowest-friction way to accelerate an existing model.

> [!NOTE]
> Because `TORCH_LIBRARY_IMPL` runs at module import time, the set of
> ops replaced this way is fixed at build time. Per-op opt-out is not
> currently supported through this path.

### 3.3 Via the `c_operators` pybind module (direct, dispatcher-free)

The same C++ wrappers are also exported through a `PYBIND11_MODULE(c_operators, …)`
in `src/flag_gems/csrc/cstub.cpp`:

```python
from flag_gems import c_operators

out = c_operators.mm(a, b)
c_operators.fused_add_rms_norm(input, residual, weight, eps)
```

This path completely **bypasses the PyTorch dispatcher**, making it the
lowest-overhead way to call a FlagGems C++ operator from Python. It is
most useful in latency-critical microbenchmarks or tight inner loops where
even the boxed dispatcher call is measurable.

### 3.4 Via the native C++ API (`flag_gems::` functions and GTest)

Every wrapper is also a regular C++ function in the `flag_gems::`
namespace, declared in `include/flag_gems/operators.h` and shipped in the
installed CMake target `FlagGems::operators`. Downstream C++ code can link
against this target and call the operators directly:

```cpp
#include "flag_gems/operators.h"

at::Tensor c = flag_gems::mm_tensor(a, b);
at::Tensor y = flag_gems::rms_norm(x, weight, eps);
```

This is exactly what the in-tree GTest suite under `ctests/` uses (e.g.
`ctests/test_triton_mm.cpp`), and it is the right path when embedding
FlagGems into a non-Python C++ application or when writing C++ unit tests.

### Summary

| Path                           | Entry point                       | Dispatcher |
| ------------------------------ | --------------------------------- | ---------- |
| `torch.ops.flag_gems.*`        | `TORCH_LIBRARY(flag_gems, …)`     | Yes        |
| ATen replacement               | `TORCH_LIBRARY_IMPL(aten, …)`     | Yes        |
| `flag_gems.c_operators` pybind | `PYBIND11_MODULE(c_operators, …)` | No         |
| Native C++ API                 | `flag_gems::*` in `operators.h`   | No         |

## Reference: Currently supported C++-wrapped operators

The following operators currently have C++ wrappers shipped with *FlagGems*.

- `add` (pointwise dynamic C++)
- `div` (pointwise dynamic C++)
- `fill` (pointwise dynamic C++)
- `addmm`
- `mm`
- `bmm`
- `cat`
- `contiguous`
- `copy`
- `embedding`
- `exponential_`
- `zeros`
- `argmax`
- `max`
- `sum`
- `softmax`
- `sort`
- `topk`
- `nonzero`
- `rms_norm`
- `fused_add_rms_norm`
- `rotary_embedding`
- `flash_attn_varlen_func`
- `reshape_and_cache_flash`
- `rwkv_mm_sparsity`
- `rwkv_ka_fusion`

> [!NOTE]
> Operators marked as *pointwise dynamic C++* are built only when the
> `-DFLAGGEMS_BUILD_POINTWISE_DYNAMIC_CPP=ON` CMake option is enabled.
> See the [install guide](../../getting-started/install/#install-from-source)
> for details.

We are actively expanding this list as part of our ongoing performance optimization work.
